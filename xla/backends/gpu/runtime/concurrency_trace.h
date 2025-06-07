#ifndef XLA_BACKENDS_GPU_RUNTIME_TRACE_H_
#define XLA_BACKENDS_GPU_RUNTIME_TRACE_H_

#include "nccl_all_reduce_thunk.h"
#include "thunk.h"
#include <string>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/btree_map.h"

namespace xla::gpu {

struct ConcurrencyTraceOptions {
  struct SyntheticBugOptions {
    bool wait_for_streams_thunk = false;
  };

  SyntheticBugOptions synthetic_bug_options{};

  // If true, traces will be collected to
  bool instrument_streams = true;
};

class ConcurrencyTracer {
 public:
  explicit ConcurrencyTracer();
  ~ConcurrencyTracer();
  using EventId = const void*;
  using StreamId = const void*;

  // region Thunk Traces
  void OnThunkLaunch(const Thunk& thunk, const Thunk::ExecuteParams& params);
  void OnStreamEventRecord(const se::Stream& stream, const se::Event& event);
  void OnStreamEventWait(const se::Stream& stream, const se::Event& event);

  void PrintTraces(std::ostream& os) const;

  // Returns an approximation of the memory used to store the collected traces
  // in bytes.  This method is thread-safe.
  size_t GetApproximateMemoryUsage() const;

  struct TraceStats {
    size_t buffer_reads = 0;
    size_t async_buffer_reads = 0;
    size_t buffer_writes = 0;
    size_t async_buffer_writes = 0;
    size_t event_records = 0;
    size_t wait_for_events = 0;
    size_t unique_streams = 0;
  };

  TraceStats GetTraceStats() const;

  enum class AccessKind { kRead, kWrite };

  struct VectorClock {
    absl::flat_hash_map<StreamId, uint64_t> clk;

    void Join(const VectorClock& other) {
      for (const auto& [s, t] : other.clk)
        clk[s] = std::max(clk[s], t);
    }

    bool HappensBefore(const VectorClock& other) const {
      for (const auto& [s, t] : clk) {
        auto it = other.clk.find(s);
        if (const uint64_t ot = it == other.clk.end() ? 0 : it->second; t > ot) return false;
      }
      return true;
    }

    bool Concurrent(const VectorClock& other) const {
      return !HappensBefore(other) && !other.HappensBefore(*this);
    }
  };

  struct SourceInfo final {
    const Thunk* thunk = nullptr;
    const std::string instruction;

    SourceInfo(const Thunk* thunk, const std::string& instruction)
        : thunk(thunk), instruction(instruction) {}
    explicit SourceInfo(const Thunk* thunk)
        : thunk(thunk), instruction(thunk->source_instruction()) {}

    SourceInfo() : instruction("") {}
  };

  struct Buffer final {
    const int device_ordinal;
    const BufferAllocation::Slice slice;

    bool operator==(const Buffer& another) const;
    bool operator!=(const Buffer& another) const { return !(*this == another); }
    bool Overlaps(const Buffer& another) const;  // true if byte-ranges overlap
  };

  struct BufferHandle final {
    int device_ordinal;
    BufferAllocation::Index allocation_index;

    bool operator==(const BufferHandle& other) const {
      return device_ordinal == other.device_ordinal &&
             allocation_index == other.allocation_index;
    }

    template <typename H>
    friend H AbslHashValue(H h, const BufferHandle& b) {
      return H::combine(std::move(h), b.device_ordinal, b.allocation_index);
    }
  };

  struct MemAccessInfo final {
    const StreamId stream_id;
    const Buffer buffer;
    const AccessKind kind;
    const size_t trace_idx;  // position inside trace_
    const SourceInfo source;
    const EventId completion_event_id =
        nullptr;  // if not null, it's an async event.
    VectorClock vc;

    MemAccessInfo(const StreamId stream_id, const Buffer& buffer,
                  const AccessKind kind, const size_t trace_idx,
                  const SourceInfo& source, const VectorClock& vc,
                  const EventId completion_event_id = nullptr)
        : stream_id(stream_id),
          buffer(buffer),
          kind(kind),
          trace_idx(trace_idx),
          source(source),
          completion_event_id(completion_event_id),
          vc(vc) {}

    bool IsWrite() const { return kind == AccessKind::kWrite; }
    bool IsAsync() const { return completion_event_id != nullptr; }
    EventId GetCompletionEventId() const { return completion_event_id; }
  };

  struct DataRace final {
    MemAccessInfo first;
    MemAccessInfo second;

    DataRace(const MemAccessInfo& first, const MemAccessInfo& second)
        : first(first), second(second) {
      ABSL_ASSERT(first.buffer.Overlaps(second.buffer) && "Buffer mismatch");
    }

    const Buffer& buffer() const { return first.buffer; }
  };

  // Returns the list of detected races.  Thread-safe, may be called many times.
  std::vector<DataRace> DetectDataRaces() const;

  // Pretty-print the races returned by DetectDataRaces().
  void PrintDataRaces(std::ostream& os) const;

  void PrintDot(std::ostream& out) const {
    const EdgeList graph = BuildHappensBeforeGraph();
    PrintDot(graph, out);
  }

 private:
  struct Trace {
    VectorClock vc;
    const SourceInfo source{};
    Trace(const VectorClock& vc, const SourceInfo& source)
        : vc(vc), source(source) {}
    virtual ~Trace() = default;
  };
  struct BufferRead final : Trace {
    const StreamId stream_id;
    const Buffer buffer;

    explicit BufferRead(const VectorClock& vc, const StreamId stream_id,
                        const Buffer& buffer, const SourceInfo& source)
        : Trace(vc, source), stream_id(stream_id), buffer(buffer) {}
  };
  struct AsyncBufferRead final : Trace {
    const StreamId source_stream_id;
    const StreamId async_stream_id;
    const EventId completion_event_id;
    const Buffer buffer;

    explicit AsyncBufferRead(const VectorClock& vc, const StreamId source_stream_id,
                             const StreamId async_stream_id,
                             const EventId completion_event_id,
                             const Buffer& buffer, const SourceInfo& source)
        : Trace(vc, source),
          source_stream_id(source_stream_id),
          async_stream_id(async_stream_id),
          completion_event_id(completion_event_id),
          buffer(buffer) {}
  };
  struct BufferWrite final : Trace {
    const StreamId stream_id;
    const Buffer buffer;

    explicit BufferWrite(const VectorClock& vc, const StreamId stream_id,
                         const Buffer& buffer, const SourceInfo& source)
        : Trace(vc, source), stream_id(stream_id), buffer(buffer) {}
    BufferWrite(BufferWrite& other) = default;
    BufferWrite(BufferWrite&& other) = default;
  };
  struct AsyncBufferWrite final : Trace {
    const StreamId source_stream_id;
    const StreamId async_stream_id;
    const EventId completion_event_id;
    Buffer buffer;

    explicit AsyncBufferWrite(const VectorClock& vc, const StreamId source_stream_id,
                              const StreamId async_stream_id,
                              const EventId completion_event_id,
                              const Buffer& buffer, const SourceInfo& source)
        : Trace(vc, source),
          source_stream_id(source_stream_id),
          async_stream_id(async_stream_id),
          completion_event_id(completion_event_id),
          buffer(buffer) {}
  };
  struct WaitForEvent final : Trace {
    const StreamId stream_id;
    const EventId event_id;

    WaitForEvent(const VectorClock& vc, const StreamId stream_id, const EventId event_id)
        : Trace(vc, {}), stream_id(stream_id), event_id(event_id) {}
  };
  struct EventRecord final : Trace {
    const StreamId stream_id;
    const EventId event_id;

    EventRecord(const VectorClock& vc, const StreamId stream_id, const EventId event_id)
        : Trace(vc, {}), stream_id(stream_id), event_id(event_id) {}
  };

  mutable std::mutex mutex_{};
  std::vector<std::unique_ptr<Trace>> trace_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<StreamId, VectorClock> stream_clock_;
  absl::flat_hash_map<EventId, VectorClock> event_clock_;

  template <typename T, typename... Args>
  void AddTrace(Args&&... args) {
    static_assert(std::is_base_of_v<Trace, T>, "T must derive from Trace");

    std::lock_guard lock(mutex_);
    trace_.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
  }

  template <typename T>
  void AddTrace(T&& trace) {
    static_assert(std::is_base_of_v<Trace, T>, "T must derive from Trace");

    std::lock_guard lock(mutex_);
    trace_.push_back(std::make_unique<T>(trace));
  }

  // Vector clock helpers.
  VectorClock SnapshotStream(StreamId stream) { return stream_clock_[stream]; }
  void AdvanceStream(StreamId stream) { stream_clock_[stream].clk[stream]++; }
  void JoinStream(StreamId stream, const VectorClock& other) {
    stream_clock_[stream].Join(other);
  }

  void AddBufferRead(StreamId stream_id, const Buffer& buffer, SourceInfo source);
  void AddBufferWrite(StreamId stream_id, const Buffer& buffer, SourceInfo source);
  void AddAsyncBufferRead(StreamId source_stream_id, StreamId async_stream_id,
                          EventId event_id, const Buffer& buffer,
                          SourceInfo source);
  void AddAsyncBufferWrite(StreamId source_stream_id, StreamId async_stream_id,
                           EventId event_id, const Buffer& buffer,
                           SourceInfo source);
  void AddEventRecord(StreamId stream_id, EventId event_id);
  void AddWaitForEvent(StreamId stream_id, EventId event_id);

  using EdgeList = absl::flat_hash_map<size_t, absl::flat_hash_set<size_t>>;
  EdgeList BuildHappensBeforeGraph() const;
  static void PrintDot(const EdgeList& graph, std::ostream& out);

  void RecordAsyncBufferAccesses(
      absl::Span<const NcclCollectiveThunk::Buffer> buffers,
      const stream_executor::Event* event, const Thunk::ExecuteParams& params,
      const stream_executor::Stream* stream, int device_ordinal,
      SourceInfo source, AsyncStreamKind async_stream_kind);

  void RecordSyncBufferAccesses(
      absl::Span<const NcclCollectiveThunk::Buffer> buffers,
      const stream_executor::Stream* stream, int device_ordinal,
      SourceInfo source);
};

}  // namespace xla::gpu

#endif
