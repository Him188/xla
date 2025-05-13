#ifndef XLA_BACKENDS_GPU_RUNTIME_TRACE_H_
#define XLA_BACKENDS_GPU_RUNTIME_TRACE_H_

#include "thunk.h"

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

  void PrintTraces(std::ostream& os);

  enum class AccessKind { kRead, kWrite };

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

  struct MemAccessInfo final {
    const StreamId stream_id;
    const Buffer buffer;
    const AccessKind kind;
    const size_t trace_idx;  // position inside trace_
    const SourceInfo source;
    const EventId completion_event_id =
        nullptr;  // if not null, it's an async event.

    MemAccessInfo(const StreamId stream_id, const Buffer& buffer,
                  const AccessKind kind, const size_t trace_idx,
                  const SourceInfo& source,
                  const EventId completion_event_id = nullptr)
        : stream_id(stream_id),
          buffer(buffer),
          kind(kind),
          trace_idx(trace_idx),
          source(source),
          completion_event_id(completion_event_id) {}

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
    const SourceInfo source{};
    explicit Trace(const SourceInfo& source) : source(source) {}
    virtual ~Trace() = default;
  };
  struct BufferRead final : Trace {
    const StreamId stream_id;
    const Buffer buffer;

    explicit BufferRead(const StreamId stream_id, const Buffer& buffer,
                        const SourceInfo& source)
        : Trace(source), stream_id(stream_id), buffer(buffer) {}
  };
  struct AsyncBufferRead final : Trace {
    const StreamId source_stream_id;
    const StreamId async_stream_id;
    const EventId completion_event_id;
    const Buffer buffer;

    explicit AsyncBufferRead(const StreamId source_stream_id,
                             const StreamId async_stream_id,
                             const EventId completion_event_id,
                             const Buffer& buffer, const SourceInfo& source)
        : Trace(source),
          source_stream_id(source_stream_id),
          async_stream_id(async_stream_id),
          completion_event_id(completion_event_id),
          buffer(buffer) {}
  };
  struct BufferWrite final : Trace {
    const StreamId stream_id;
    const Buffer buffer;

    explicit BufferWrite(const StreamId stream_id, const Buffer& buffer,
                         const SourceInfo& source)
        : Trace(source), stream_id(stream_id), buffer(buffer) {}
    BufferWrite(BufferWrite& other) = default;
    BufferWrite(BufferWrite&& other) = default;
  };
  struct AsyncBufferWrite final : Trace {
    const StreamId source_stream_id;
    const StreamId async_stream_id;
    const EventId completion_event_id;
    Buffer buffer;

    explicit AsyncBufferWrite(const StreamId source_stream_id,
                              const StreamId async_stream_id,
                              const EventId completion_event_id,
                              const Buffer& buffer, const SourceInfo& source)
        : Trace(source),
          source_stream_id(source_stream_id),
          async_stream_id(async_stream_id),
          completion_event_id(completion_event_id),
          buffer(buffer) {}
  };
  struct WaitForEvent final : Trace {
    const StreamId stream_id;
    const EventId event_id;

    WaitForEvent(const StreamId stream_id, const EventId event_id)
        : Trace({}), stream_id(stream_id), event_id(event_id) {}
  };
  struct EventRecord final : Trace {
    const StreamId stream_id;
    const EventId event_id;

    EventRecord(const StreamId stream_id, const EventId event_id)
        : Trace({}), stream_id(stream_id), event_id(event_id) {}
  };

  std::mutex mutex_{};
  std::vector<std::unique_ptr<Trace>> trace_ ABSL_GUARDED_BY(mutex_);

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

  using EdgeList = absl::flat_hash_map<size_t, absl::flat_hash_set<size_t>>;
  EdgeList BuildHappensBeforeGraph() const;
  static void PrintDot(const EdgeList& graph, std::ostream& out);
};

}  // namespace xla::gpu

#endif
