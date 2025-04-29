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

  // region Thunk Traces
  void OnThunkLaunch(const Thunk& thunk, const Thunk::ExecuteParams& params);
  void OnStreamEventRecord(const se::Stream& stream, const se::Event& event);
  void OnStreamEventWait(const se::Stream& stream, const se::Event& event);

  void PrintTraces(std::ostream& os);

 private:
  class Trace {
   public:
    virtual ~Trace() = default;
  };
  struct BufferRead final : Trace {
    void* stream_id;
    BufferAllocation::Slice buffer;

    explicit BufferRead(void* stream_id, const BufferAllocation::Slice& buffer)
        : stream_id(stream_id), buffer(buffer) {}
  };
  struct BufferWrite final : Trace {
    void* stream_id;
    BufferAllocation::Slice buffer;

    explicit BufferWrite(void* stream_id, const BufferAllocation::Slice& buffer)
        : stream_id(stream_id), buffer(buffer) {}
  };
  struct WaitForEvent final : Trace {
    void* stream_id;
    void* event_id;

    WaitForEvent(void* stream_id, void* event_id)
        : stream_id(stream_id), event_id(event_id) {}
  };
  struct EventRecord final : Trace {
    void* stream_id;
    void* event_id;

    EventRecord(void* stream_id, void* event_id)
        : stream_id(stream_id), event_id(event_id) {}
  };

  std::mutex mutex_{};
  std::vector<std::unique_ptr<Trace>> trace_ ABSL_GUARDED_BY(mutex_);

  template <typename T, typename... Args>
  void AddTrace(Args&&... args) {
    static_assert(std::is_base_of_v<Trace, T>, "T must derive from Trace");

    std::lock_guard lock(mutex_);
    trace_.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
  }
};

}  // namespace xla::gpu

#endif
