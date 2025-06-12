#include "thunk_sanitizer.h"

#include <iostream>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "copy_thunk.h"
#include "gemm_thunk.h"
#include "kernel_thunk.h"
#include "memset_thunk.h"
#include "fft_thunk.h"
#include "gpublas_lt_matmul_thunk.h"
#include "convolution_thunk.h"
#include "triangular_solve_thunk.h"
#include "cholesky_thunk.h"
#include "norm_thunk.h"
#include "nccl_all_reduce_thunk.h"
#include "nccl_collective_permute_thunk.h"
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

constexpr bool ENABLE_LOGS = false;

ThunkSanitizer::ThunkSanitizer() = default;
ThunkSanitizer::~ThunkSanitizer() = default;

void ThunkSanitizer::AddBufferRead(StreamId stream_id, const Buffer& buffer,
                                      SourceInfo source) {
  std::lock_guard lock(mutex_);
  VectorClock vc = stream_clock_[stream_id];
  AdvanceStream(stream_id);
  trace_.push_back(std::make_unique<BufferRead>(vc, stream_id, buffer, source));
}

void ThunkSanitizer::AddBufferWrite(StreamId stream_id, const Buffer& buffer,
                                       SourceInfo source) {
  std::lock_guard lock(mutex_);
  VectorClock vc = stream_clock_[stream_id];
  AdvanceStream(stream_id);
  trace_.push_back(std::make_unique<BufferWrite>(vc, stream_id, buffer, source));
}

void ThunkSanitizer::AddAsyncBufferRead(StreamId source_stream_id,
                                           StreamId async_stream_id,
                                           EventId event_id, const Buffer& buffer,
                                           SourceInfo source) {
  std::lock_guard lock(mutex_);
  JoinStream(async_stream_id, stream_clock_[source_stream_id]);
  VectorClock vc = stream_clock_[async_stream_id];
  AdvanceStream(async_stream_id);
  trace_.push_back(std::make_unique<AsyncBufferRead>(vc, source_stream_id,
                                                    async_stream_id, event_id,
                                                    buffer, source));
}

void ThunkSanitizer::AddAsyncBufferWrite(StreamId source_stream_id,
                                            StreamId async_stream_id,
                                            EventId event_id,
                                            const Buffer& buffer,
                                            SourceInfo source) {
  std::lock_guard lock(mutex_);
  JoinStream(async_stream_id, stream_clock_[source_stream_id]);
  VectorClock vc = stream_clock_[async_stream_id];
  AdvanceStream(async_stream_id);
  trace_.push_back(std::make_unique<AsyncBufferWrite>(vc, source_stream_id,
                                                     async_stream_id, event_id,
                                                     buffer, source));
}

void ThunkSanitizer::AddEventRecord(StreamId stream_id, EventId event_id) {
  std::lock_guard lock(mutex_);
  VectorClock vc = stream_clock_[stream_id];
  event_clock_[event_id] = vc;
  AdvanceStream(stream_id);
  trace_.push_back(std::make_unique<EventRecord>(vc, stream_id, event_id));
}

void ThunkSanitizer::AddWaitForEvent(StreamId stream_id, EventId event_id) {
  std::lock_guard lock(mutex_);
  JoinStream(stream_id, event_clock_[event_id]);
  VectorClock vc = stream_clock_[stream_id];
  AdvanceStream(stream_id);
  trace_.push_back(std::make_unique<WaitForEvent>(vc, stream_id, event_id));
}
void ThunkSanitizer::RecordAsyncBufferAccesses(
    const absl::Span<const NcclCollectiveThunk::Buffer> buffers,
    const stream_executor::Event* const event,
    const Thunk::ExecuteParams& params, const stream_executor::Stream* stream,
    const int device_ordinal, SourceInfo source,
    const AsyncStreamKind async_stream_kind) {
  const auto completion_event_id =
      static_cast<void*>(AssertCuda(event).GetHandle());

  const auto& async_stream = *params.collective_params->async_streams.at(
      static_cast<size_t>(async_stream_kind));

  // Async
  for (const auto& buf : buffers) {
    if (buf.source_buffer != buf.destination_buffer) {
      AddAsyncBufferRead(stream->platform_specific_handle().stream,
                         async_stream.platform_specific_handle().stream,
                         completion_event_id,
                         Buffer{device_ordinal, buf.source_buffer}, source);
    }
    AddAsyncBufferWrite(stream->platform_specific_handle().stream,
                        async_stream.platform_specific_handle().stream,
                        completion_event_id,
                        Buffer{device_ordinal, buf.destination_buffer}, source);
  }
}
void ThunkSanitizer::RecordSyncBufferAccesses(
    const absl::Span<const NcclCollectiveThunk::Buffer> buffers,
    const stream_executor::Stream* stream, const int device_ordinal,
    SourceInfo source) {
  for (const auto& buf : buffers) {
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, buf.source_buffer}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, buf.destination_buffer}, source);
  }
}
void ThunkSanitizer::OnThunkLaunch(const Thunk& thunk,
                                      const Thunk::ExecuteParams& params) {
#define THUNK_CASE(type)                             \
  const auto* t = dynamic_cast<const type*>(&thunk); \
  t != nullptr

  auto* stream =
      Thunk::GetStreamForExecution(thunk.execution_stream_id(), params).value();
  const int device_ordinal = params.buffer_allocations->device_ordinal();

  if (ENABLE_LOGS) {
    std::cout << "[device=" << device_ordinal
              << "][Stream] Launching thunk on stream S_" << stream->GetName()
              << ": " << Thunk::KindToString(thunk.kind()) << std::endl;
  }

  SourceInfo source{&thunk};

  /* ---------------------------- ordinary thunks --------------------------- */
  if (THUNK_CASE(GemmThunk)) {
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->lhs_buffer()}, source);
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->rhs_buffer()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->output_buffer()}, source);

  } else if (THUNK_CASE(gpu::KernelThunk)) {
    const auto& arguments = t->arguments();

    // reads first
    for (int i = 0; i < arguments.size(); ++i)
      if (!t->written()[i])
        AddBufferRead(stream->platform_specific_handle().stream,
                      Buffer{device_ordinal, arguments[i]}, source);

    // writes
    for (int i = 0; i < arguments.size(); ++i)
      if (t->written()[i])
        AddBufferWrite(stream->platform_specific_handle().stream,
                       Buffer{device_ordinal, arguments[i]}, source);

  } else if (THUNK_CASE(gpu::CopyThunk)) {
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->source()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->destination()}, source);

  } else if (THUNK_CASE(gpu::MemzeroThunk)) {
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->destination()}, source);

  } else if (THUNK_CASE(gpu::Memset32BitValueThunk)) {
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->destination()}, source);

  } else if (THUNK_CASE(gpu::FftThunk)) {
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->input_buffer()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->output_buffer()}, source);

  } else if (THUNK_CASE(gpu::ConvolutionThunk)) {
    for (auto buf : t->operand_buffers())
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, buf}, source);
    for (auto buf : t->result_buffers())
      AddBufferWrite(stream->platform_specific_handle().stream,
                     Buffer{device_ordinal, buf}, source);
    if (t->scratch_buffer().size() != 0)
      AddBufferWrite(stream->platform_specific_handle().stream,
                     Buffer{device_ordinal, t->scratch_buffer()}, source);

  } else if (THUNK_CASE(gpu::ConvolutionReorderThunk)) {
    for (auto buf : t->operand_buffers())
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, buf}, source);
    for (auto buf : t->result_buffers())
      AddBufferWrite(stream->platform_specific_handle().stream,
                     Buffer{device_ordinal, buf}, source);

  } else if (THUNK_CASE(gpu::CublasLtMatmulThunk)) {
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->a_buffer()}, source);
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->b_buffer()}, source);
    if (t->c_buffer().allocation() != nullptr)
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, t->c_buffer()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->d_buffer()}, source);
    if (t->bias_buffer().allocation() != nullptr)
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, t->bias_buffer()}, source);
    if (t->aux_buffer().allocation() != nullptr)
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, t->aux_buffer()}, source);
    if (t->a_scale_buffer().allocation() != nullptr)
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, t->a_scale_buffer()}, source);
    if (t->b_scale_buffer().allocation() != nullptr)
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, t->b_scale_buffer()}, source);
    if (t->c_scale_buffer().allocation() != nullptr)
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, t->c_scale_buffer()}, source);
    if (t->d_scale_buffer().allocation() != nullptr)
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, t->d_scale_buffer()}, source);
    if (t->d_amax_buffer().allocation() != nullptr)
      AddBufferWrite(stream->platform_specific_handle().stream,
                     Buffer{device_ordinal, t->d_amax_buffer()}, source);
    if (t->workspace())
      AddBufferWrite(stream->platform_specific_handle().stream,
                     Buffer{device_ordinal, *t->workspace()}, source);

  } else if (THUNK_CASE(gpu::TriangularSolveThunk)) {
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->a_buffer()}, source);
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->b_buffer()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->b_buffer()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->temp_buffer()}, source);

  } else if (THUNK_CASE(gpu::CholeskyThunk)) {
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->a_buffer()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->a_buffer()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->workspace_buffer()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->info_buffer()}, source);

  } else if (THUNK_CASE(gpu::NormThunk)) {
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->x_buffer()}, source);
    AddBufferRead(stream->platform_specific_handle().stream,
                  Buffer{device_ordinal, t->scale_buffer()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->y_or_dx_buffer()}, source);
    if (t->bias_buffer())
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, *t->bias_buffer()}, source);
    if (t->expectation_buffer()) {
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, *t->expectation_buffer()}, source);
      AddBufferWrite(stream->platform_specific_handle().stream,
                     Buffer{device_ordinal, *t->expectation_buffer()}, source);
    }
    if (t->norm_factor_buffer()) {
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, *t->norm_factor_buffer()}, source);
      AddBufferWrite(stream->platform_specific_handle().stream,
                     Buffer{device_ordinal, *t->norm_factor_buffer()}, source);
    }
    if (t->dy_buffer())
      AddBufferRead(stream->platform_specific_handle().stream,
                    Buffer{device_ordinal, *t->dy_buffer()}, source);
    if (t->dscale_buffer())
      AddBufferWrite(stream->platform_specific_handle().stream,
                     Buffer{device_ordinal, *t->dscale_buffer()}, source);
    if (t->dbias_buffer())
      AddBufferWrite(stream->platform_specific_handle().stream,
                     Buffer{device_ordinal, *t->dbias_buffer()}, source);
    AddBufferWrite(stream->platform_specific_handle().stream,
                   Buffer{device_ordinal, t->scratch_buffer()}, source);

    /* -------------------------- NCCL collective ⇩ ---------------------------
     */
  } else if (THUNK_CASE(gpu::NcclAllReduceReduceScatterThunkBase)) {
    /*  The start thunk issues the NCCL call on an *async* stream:
        – all input buffers are READ;
        – all destination buffers are WRITTEN but are **not ready**
          until the accompanying CollectiveDoneThunk waits on the
          asynchronous event recorded by the start thunk.
     */

    if (const auto async_events = t->async_events()) {
      if (const auto event =
              async_events->GetEvent(params.stream->parent()).value()) {
        RecordAsyncBufferAccesses(t->buffers(), event, params, stream,
                                  device_ordinal, source,
                                  t->GetAsyncStreamKind());
      }
    } else {
      // Sync
      RecordSyncBufferAccesses(t->buffers(), stream, device_ordinal, source);
    }
  } else if (THUNK_CASE(gpu::NcclCollectivePermuteStartThunk)) {
    if (const auto async_events = t->async_events()) {
      if (const auto event =
              async_events->GetEvent(params.stream->parent()).value()) {
        RecordAsyncBufferAccesses(t->buffers(), event, params, stream,
                                  device_ordinal, source,
                                  t->GetAsyncStreamKind());
      }
    } else {
      RecordSyncBufferAccesses(t->buffers(), stream, device_ordinal, source);
    }
  } else if (THUNK_CASE(gpu::NcclCollectiveDoneThunk)) {
    /*  NcclCollectiveDoneThunk executes a `stream->WaitFor(event)` that
        corresponds to the event recorded by the start thunk.  The tracing
        callback OnStreamEventWait invoked from our stream shim will record
        the WaitForEvent edge, so there is nothing to do here in terms of
        buffer accesses — the thunk touches no user data itself. */
  }
#undef THUNK_CASE
}

/* ----------------------------- Stream events ------------------------------ */

void ThunkSanitizer::OnStreamEventRecord(const se::Stream& stream,
                                            const se::Event& event) {
  if (ENABLE_LOGS) {
    std::cout << "[Stream] S_" << stream.GetName() << " recorded "
              << "E_" << AssertCuda(event).GetHandle() << std::endl;
  }

  AddEventRecord(stream.platform_specific_handle().stream,
                 static_cast<void*>(AssertCuda(&event).GetHandle()));
}

void ThunkSanitizer::OnStreamEventWait(const se::Stream& stream,
                                          const se::Event& event) {
  if (ENABLE_LOGS) {
    std::cout << "[Stream] E_" << AssertCuda(event).GetHandle() << " -> "
              << "S_" << stream.GetName() << std::endl;
  }

  AddWaitForEvent(stream.platform_specific_handle().stream,
                  static_cast<void*>(AssertCuda(&event).GetHandle()));
}

void ThunkSanitizer::PrintTraces(std::ostream& os) const {
  // Protect the trace_ vector while we read from it.
  std::lock_guard lock(mutex_);

  // Preserve whatever formatting the caller had.
  const auto old_flags = os.flags();

  os << "───────────────────────  Concurrency trace  (" << trace_.size()
     << " entries)  ───────────────────────\n";

  for (const std::unique_ptr<Trace>& p : trace_) {
    if (const auto* t = dynamic_cast<const BufferRead*>(p.get()); t) {
      os << "[MemoryRead ][device " << t->buffer.device_ordinal << "] stream=0x"
         << std::hex << t->stream_id << " @ " << std::hex << t->buffer.slice
         << std::dec << '\n';
      continue;
    }
    if (const auto* t = dynamic_cast<const AsyncBufferRead*>(p.get()); t) {
      os << "[AsyncMemoryRead][device " << t->buffer.device_ordinal << "] "
         << "event=0x" << std::hex << t->completion_event_id << ", "
         << "source_stream=0x" << std::hex << t->source_stream_id << ", "
         << "async_stream=0x" << std::hex << t->async_stream_id << " @ "
         << std::hex << t->buffer.slice << std::dec << '\n';
      continue;
    }
    if (const auto* t = dynamic_cast<const BufferWrite*>(p.get()); t) {
      os << "[MemoryWrite][device " << t->buffer.device_ordinal << "] stream=0x"
         << std::hex << t->stream_id << " @ " << std::hex << t->buffer.slice
         << std::dec << '\n';
      continue;
    }
    if (const auto* t = dynamic_cast<const AsyncBufferWrite*>(p.get()); t) {
      os << "[AsyncMemoryWrite][device " << t->buffer.device_ordinal << "] "
         << "event=0x" << std::hex << t->completion_event_id << ", "
         << "source_stream=0x" << std::hex << t->source_stream_id << ", "
         << "async_stream=0x" << std::hex << t->async_stream_id << " @ "
         << std::hex << t->buffer.slice << std::dec << '\n';
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

size_t ThunkSanitizer::GetApproximateMemoryUsage() const {
  std::lock_guard lock(mutex_);
  size_t bytes =
      sizeof(*this) + trace_.capacity() * sizeof(std::unique_ptr<Trace>);
  for (const auto& p : trace_) {
    if (dynamic_cast<const BufferRead*>(p.get())) {
      bytes += sizeof(BufferRead);
    } else if (dynamic_cast<const BufferWrite*>(p.get())) {
      bytes += sizeof(BufferWrite);
    } else if (dynamic_cast<const AsyncBufferRead*>(p.get())) {
      bytes += sizeof(AsyncBufferRead);
    } else if (dynamic_cast<const AsyncBufferWrite*>(p.get())) {
      bytes += sizeof(AsyncBufferWrite);
    } else if (dynamic_cast<const EventRecord*>(p.get())) {
      bytes += sizeof(EventRecord);
    } else if (dynamic_cast<const WaitForEvent*>(p.get())) {
      bytes += sizeof(WaitForEvent);
    } else {
      bytes += sizeof(Trace);
    }
  }
  return bytes;
}

ThunkSanitizer::TraceStats ThunkSanitizer::GetTraceStats() const {
  std::lock_guard lock(mutex_);
  TraceStats stats;
  absl::flat_hash_set<StreamId> streams;
  for (const auto& p : trace_) {
    if (auto* t = dynamic_cast<const BufferRead*>(p.get()); t) {
      stats.buffer_reads++;
      streams.insert(t->stream_id);
    } else if (auto* t = dynamic_cast<const AsyncBufferRead*>(p.get()); t) {
      stats.async_buffer_reads++;
      streams.insert(t->source_stream_id);
      streams.insert(t->async_stream_id);
    } else if (auto* t = dynamic_cast<const BufferWrite*>(p.get()); t) {
      stats.buffer_writes++;
      streams.insert(t->stream_id);
    } else if (auto* t = dynamic_cast<const AsyncBufferWrite*>(p.get()); t) {
      stats.async_buffer_writes++;
      streams.insert(t->source_stream_id);
      streams.insert(t->async_stream_id);
    } else if (auto* t = dynamic_cast<const EventRecord*>(p.get()); t) {
      stats.event_records++;
      streams.insert(t->stream_id);
    } else if (auto* t = dynamic_cast<const WaitForEvent*>(p.get()); t) {
      stats.wait_for_events++;
      streams.insert(t->stream_id);
    }
  }
  stats.unique_streams = streams.size();
  return stats;
}
bool ThunkSanitizer::Buffer::operator==(const Buffer& another) const {
  if (device_ordinal != another.device_ordinal) return false;
  if (slice != another.slice) return false;
  return true;
}
bool ThunkSanitizer::Buffer::Overlaps(const Buffer& another) const {
  if (device_ordinal != another.device_ordinal) return false;
  if (slice.allocation() != another.slice.allocation()) return false;
  const uint64_t a_begin = slice.offset();
  const uint64_t a_end = a_begin + slice.size();
  const uint64_t b_begin = another.slice.offset();
  const uint64_t b_end = b_begin + another.slice.size();
  return a_begin < b_end && b_begin < a_end;
}
std::vector<ThunkSanitizer::DataRace> ThunkSanitizer::DetectDataRaces()
    const {
  std::lock_guard lock(mutex_);

  std::vector<MemAccessInfo> acc;
  acc.reserve(trace_.size());
  absl::flat_hash_map<BufferHandle, std::vector<size_t>> by_buffer;

  for (size_t i = 0; i < trace_.size(); ++i) {
    if (auto* r = dynamic_cast<const BufferRead*>(trace_[i].get()); r) {
      acc.push_back({r->stream_id, r->buffer, AccessKind::kRead, i, r->source,
                     r->vc});
      by_buffer[{r->buffer.device_ordinal,
                 r->buffer.slice.allocation()->index()}]
          .push_back(acc.size() - 1);
    } else if (auto* w = dynamic_cast<const BufferWrite*>(trace_[i].get()); w) {
      acc.push_back({w->stream_id, w->buffer, AccessKind::kWrite, i, w->source,
                     w->vc});
      by_buffer[{w->buffer.device_ordinal,
                 w->buffer.slice.allocation()->index()}]
          .push_back(acc.size() - 1);
    } else if (auto* ar = dynamic_cast<const AsyncBufferRead*>(trace_[i].get());
               ar) {
      acc.push_back({ar->async_stream_id, ar->buffer, AccessKind::kRead, i,
                     ar->source, ar->vc, ar->completion_event_id});
      by_buffer[{ar->buffer.device_ordinal,
                 ar->buffer.slice.allocation()->index()}]
          .push_back(acc.size() - 1);
    } else if (auto* aw =
                   dynamic_cast<const AsyncBufferWrite*>(trace_[i].get());
               aw) {
      acc.push_back({aw->async_stream_id, aw->buffer, AccessKind::kWrite, i,
                     aw->source, aw->vc, aw->completion_event_id});
      by_buffer[{aw->buffer.device_ordinal,
                 aw->buffer.slice.allocation()->index()}]
          .push_back(acc.size() - 1);
    }
  }

  auto happens_before = [](const VectorClock& a, const VectorClock& b) {
    return a.HappensBefore(b);
  };

  std::vector<DataRace> races;
  for (const auto& [handle, indices] : by_buffer) {
    std::vector<size_t> active;
    for (size_t idx : indices) {
      const auto& cur = acc[idx];

      std::vector<size_t> next_active;
      for (size_t prev_idx : active) {
        const auto& prev = acc[prev_idx];

        if (happens_before(prev.vc, cur.vc)) {
          continue;  // ordered by HB
        }
        next_active.push_back(prev_idx);

        if (prev.stream_id == cur.stream_id) continue;
        if (!prev.buffer.Overlaps(cur.buffer)) continue;
        if (!(prev.IsWrite() || cur.IsWrite())) continue;
        if (prev.vc.Concurrent(cur.vc)) {
          races.push_back({prev, cur});
        }
      }
      active.swap(next_active);
      active.push_back(idx);
    }
  }
  return races;
}

std::ostream& operator<<(std::ostream& os,
                         const ThunkSanitizer::SourceInfo& source) {
  if (source.thunk) {
    os << source.thunk->profile_annotation();
  } else {
    os << "<unknown thunk>";
  }
  // os << "(";
  // if (!source.instruction.empty()) {
  //   os << "Hlo: " << source.instruction;
  // } else {
  //   os << "<unknown source>";
  // }
  // os << ")";
  return os;
}

void ThunkSanitizer::PrintDataRaces(std::ostream& os) const {
  const auto races = DetectDataRaces();
  if (races.empty()) {
    os << "✅  No data-races detected in this execution.\n";
    return;
  }
  os << "❌  Detected " << races.size() << " data-race"
     << (races.size() == 1 ? "" : "s") << ":\n";
  for (const auto& r : races) {
    os << "  • Buffer @" << r.buffer().device_ordinal << "/"
       << r.buffer().slice.allocation()->index()
       << " accessed "
       // << "at trace[" << r.first.trace_idx << "] ("
       // << (r.first.IsWrite() ? "W" : "R") << ") and trace["
       // << r.second.trace_idx << "] (" << (r.second.IsWrite() ? "W" : "R")
       // << ") "
       << "without ordering.\n";
    os << "    " << (r.first.IsWrite() ? "Write" : "Read")
       << " access: " << r.first.source << "\n";
    os << "    " << (r.second.IsWrite() ? "Write" : "Read")
       << " access: " << r.second.source << "\n";
  }
  os << std::dec;
}

ThunkSanitizer::EdgeList ThunkSanitizer::BuildHappensBeforeGraph() const {
  EdgeList hb;
  auto add_edge = [&](const size_t a, const size_t b) { hb[a].insert(b); };

  // Pass 0: remember the last op we have seen on every real stream
  //  for program-order edges; also record / wait positions.
  absl::flat_hash_map<StreamId, size_t> last_seen_on_stream;
  absl::flat_hash_map<const void*, size_t> record_pos;
  absl::flat_hash_map<const void*, std::vector<size_t>> wait_pos;

  for (size_t i = 0; i < trace_.size(); ++i) {
    const auto* current = trace_[i].get();
    if (auto* rec = dynamic_cast<const EventRecord*>(current); rec) {
      record_pos[rec->event_id] = i;
    } else if (auto* w = dynamic_cast<const WaitForEvent*>(current); w) {
      wait_pos[w->event_id].push_back(i);
    }

    // Adds a stream -> i edge.
    auto add_po_edge = [&](const StreamId stream,
                           const StreamId async_stream_id = nullptr) {
      if (const auto it = last_seen_on_stream.find(stream);
          it != last_seen_on_stream.end()) {
        add_edge(it->second, i);
      }

      // If the op is async, it starts on another stream.
      if (async_stream_id) {
        last_seen_on_stream[async_stream_id] = i;
      } else {
        last_seen_on_stream[stream] = i;
      }
    };

    if (auto* r = dynamic_cast<const BufferRead*>(current); r) {
      add_po_edge(r->stream_id, nullptr);
    } else if (auto* w = dynamic_cast<const BufferWrite*>(current); w) {
      add_po_edge(w->stream_id, nullptr);
    } else if (auto* ar = dynamic_cast<const AsyncBufferRead*>(current); ar) {
      add_po_edge(ar->source_stream_id, ar->async_stream_id);
    } else if (auto* aw = dynamic_cast<const AsyncBufferWrite*>(current); aw) {
      add_po_edge(aw->source_stream_id, aw->async_stream_id);
    } else if (auto* rec = dynamic_cast<const EventRecord*>(current); rec) {
      add_po_edge(rec->stream_id, nullptr);
    } else if (auto* wait_for_event =
                   dynamic_cast<const WaitForEvent*>(current);
               wait_for_event) {
      add_po_edge(wait_for_event->stream_id, nullptr);
    }
  }

  // Pass 1: event edges  (Record -> Wait)
  for (const auto& [ev, rec_i] : record_pos) {
    if (auto it = wait_pos.find(ev); it != wait_pos.end())
      for (const size_t w_i : it->second) add_edge(rec_i, w_i);
  }

  // Pass 2: async memory-access edges. Record -> AsyncAccess -> Wait for every access on that event
  for (size_t i = 0; i < trace_.size(); ++i) {
    const void* ev = nullptr;
    StreamId async_stream_id = nullptr;
    const auto* current = trace_[i].get();
    if (auto* ar = dynamic_cast<const AsyncBufferRead*>(current); ar) {
      ev = ar->completion_event_id;
      async_stream_id = ar->async_stream_id;
    } else if (auto* aw = dynamic_cast<const AsyncBufferWrite*>(current); aw) {
      ev = aw->completion_event_id;
      async_stream_id = aw->async_stream_id;
    }

    if (!ev) continue;  // not an async access

    // Add i -> completion_event.
    if (auto rec_it = record_pos.find(const_cast<void*>(ev));
        rec_it != record_pos.end()) {
      const auto event_index = rec_it->second;
      const auto event =
          dynamic_cast<const EventRecord*>(trace_[event_index].get());
      ABSL_ASSERT(event != nullptr && "EventRecord expected");
      ABSL_ASSERT(event->stream_id == async_stream_id &&
                  "Stream id of the completion event must match the async "
                  "stream id of the CollectiveStart");
      add_edge(i, rec_it->second);
    }

    // Add completion_event -> waits
    if (auto w_it = wait_pos.find(const_cast<void*>(ev));
        w_it != wait_pos.end()) {
      for (const size_t w_i : w_it->second) {
        add_edge(i, w_i);
      }
    }
  }

  return hb;
}
void ThunkSanitizer::PrintDot(const EdgeList& graph, std::ostream& out) {
  // Header and optional global attributes
  out << "digraph {\n"
      << "  rankdir=LR;\n";  // left-to-right layout (nice for timelines)

  // Collect every node that appears either as a source or a sink
  absl::flat_hash_set<size_t> nodes;
  for (const auto& [src, dsts] : graph) {
    nodes.insert(src);
    nodes.insert(dsts.begin(), dsts.end());
  }

  // Emit node declarations so even isolated vertices appear
  for (size_t v : nodes) {
    out << "  " << v << ";\n";
  }

  // Emit edges
  for (const auto& [src, dsts] : graph) {
    for (size_t dst : dsts) {
      out << "  " << src << " -> " << dst << ";\n";
    }
  }

  out << "}\n";
}

}  // namespace xla::gpu
