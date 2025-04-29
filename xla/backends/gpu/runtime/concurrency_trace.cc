#include "concurrency_trace.h"

#include "dynamic_slice_thunk.h"
#include "gemm_thunk.h"
#include "xla/stream_executor/cuda/cuda_event.h"

namespace xla::gpu {

static const stream_executor::gpu::CudaEvent& AssertCuda(
    const se::Event* event) {
  return *dynamic_cast<const stream_executor::gpu::CudaEvent*>(event);
}
static const stream_executor::gpu::CudaEvent& AssertCuda(
    const se::Event& event) {
  return AssertCuda(&event);
}

// static const stream_executor::gpu::CudaStream& AssertCuda(
//     const se::Stream* stream) {
//   return *dynamic_cast<const stream_executor::gpu::CudaStream*>(stream);
// }
// static const stream_executor::gpu::CudaStream& AssertCuda(
//     const se::Stream& stream) {
//   return AssertCuda(&stream);
// }

ConcurrencyTracer::ConcurrencyTracer() {}
ConcurrencyTracer::~ConcurrencyTracer() {}
void ConcurrencyTracer::OnThunkLaunch(const Thunk& thunk,
                                      const Thunk::ExecuteParams& params) {
#define THUNK_CASE(type)                             \
  const auto* t = dynamic_cast<const type*>(&thunk); \
  t != nullptr

  auto* stream =
      Thunk::GetStreamForExecution(thunk.execution_stream_id(), params).value();

  std::cout << "[Stream] Launching thunk on stream " << "S_"
            << Thunk::GetStreamForExecution(thunk.execution_stream_id(), params)
                   .value()
                   ->GetName()
            << ": " << Thunk::KindToString(thunk.kind()) << std::endl;

  if (THUNK_CASE(GemmThunk)) {
    AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                         t->lhs_buffer());
    AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                         t->rhs_buffer());
    AddTrace<BufferWrite>(stream->platform_specific_handle().stream,
                          t->output_buffer());
  } else if (THUNK_CASE(DynamicSliceThunk)) {
    // TODO
  }
}
void ConcurrencyTracer::OnStreamEventRecord(const se::Stream& stream,
                                            const se::Event& event) {
  std::cout << "[Stream] " << "S_" << stream.GetName() << " recorded ";

  std::cout << "E_"
            << dynamic_cast<const stream_executor::gpu::CudaEvent*>(&event)
                   ->GetHandle();  // ptr
  // if (event == &stream.completed_event_) {
  //   std::cout << " (completed_event)";
  // }

  std::cout << std::endl;

  AddTrace<EventRecord>(stream.platform_specific_handle().stream,
                        static_cast<void*>(AssertCuda(&event).GetHandle()));
}
void ConcurrencyTracer::OnStreamEventWait(const se::Stream& stream,
                                          const se::Event& event) {
  AddTrace<WaitForEvent>(stream.platform_specific_handle().stream,
                         static_cast<void*>(AssertCuda(&event).GetHandle()));

  // std::cout << "[Stream] " << "E_" << AssertCuda(event).GetHandle() << "->"
  //           << "S_" << AssertCuda(stream).GetName() << std::endl;
}
void ConcurrencyTracer::PrintTraces(std::ostream& os) {
  // Protect the trace_ vector while we read from it.
  std::lock_guard lock(mutex_);

  // Preserve whatever formatting the caller had.
  const auto old_flags = os.flags();

  os << "───────────────────────  Concurrency trace  (" << trace_.size()
     << " entries)  ───────────────────────\n";

  for (const std::unique_ptr<Trace>& p : trace_) {
    // `dynamic_cast` works because Trace now has a virtual destructor.
    if (const auto* t = dynamic_cast<const BufferRead*>(p.get()); t) {
      os << "[MemoryRead ] stream=0x" << std::hex << t->stream_id << " @ "
         << std::hex << t->buffer << std::dec << '\n';
      continue;
    }
    if (const auto* t = dynamic_cast<const BufferWrite*>(p.get()); t) {
      os << "[MemoryWrite]  stream=0x" << std::hex << t->stream_id << " @ "
         << std::hex << t->buffer << std::dec << '\n';
      continue;
    }
    if (const auto* t = dynamic_cast<const EventRecord*>(p.get()); t) {
      os << "[EventRecord]  stream=0x" << std::hex << t->stream_id
         << "  event=0x" << t->event_id << std::dec << '\n';
      continue;
    }
    if (const auto* t = dynamic_cast<const WaitForEvent*>(p.get()); t) {
      os << "[WaitForEvt ]  stream=0x" << std::hex << t->stream_id
         << "  waits on event=0x" << t->event_id << std::dec << '\n';
      continue;
    }

    // Fallback – should never happen.
    os << "[Unknown     ]  (unrecognised trace type)\n";
  }

  os << "─────────────────────────────  end trace  "
        "───────────────────────────\n";

  // Restore caller’s formatting.
  os.flags(old_flags);
}
}  // namespace xla::gpu
