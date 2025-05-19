#include "concurrency_trace.h"

#include <iostream>
#include <utility>

#include "copy_thunk.h"
#include "gemm_thunk.h"
#include "kernel_thunk.h"
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

ConcurrencyTracer::ConcurrencyTracer() = default;
ConcurrencyTracer::~ConcurrencyTracer() = default;
void ConcurrencyTracer::RecordAsyncBufferAccesses(
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
      AddTrace<AsyncBufferRead>(
          stream->platform_specific_handle().stream,
          async_stream.platform_specific_handle().stream, completion_event_id,
          Buffer{device_ordinal, buf.source_buffer}, source);
    } else {
      // Buffer aliased, we only record a write.
    }

    AddTrace<AsyncBufferWrite>(
        stream->platform_specific_handle().stream,
        async_stream.platform_specific_handle().stream, completion_event_id,
        Buffer{device_ordinal, buf.destination_buffer}, source);
  }
}
void ConcurrencyTracer::RecordSyncBufferAccesses(
    const absl::Span<const NcclCollectiveThunk::Buffer> buffers,
    const stream_executor::Stream* stream, const int device_ordinal,
    SourceInfo source) {
  for (const auto& buf : buffers) {
    AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                         Buffer{device_ordinal, buf.source_buffer}, source);
    AddTrace<BufferWrite>(stream->platform_specific_handle().stream,
                          Buffer{device_ordinal, buf.destination_buffer},
                          source);
  }
}
void ConcurrencyTracer::OnThunkLaunch(const Thunk& thunk,
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
    AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                         Buffer{device_ordinal, t->lhs_buffer()}, source);
    AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                         Buffer{device_ordinal, t->rhs_buffer()}, source);
    AddTrace<BufferWrite>(stream->platform_specific_handle().stream,
                          Buffer{device_ordinal, t->output_buffer()}, source);

  } else if (THUNK_CASE(gpu::KernelThunk)) {
    const auto& arguments = t->arguments();

    // reads first
    for (int i = 0; i < arguments.size(); ++i)
      if (!t->written()[i])
        AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                             Buffer{device_ordinal, arguments[i]}, source);

    // writes
    for (int i = 0; i < arguments.size(); ++i)
      if (t->written()[i])
        AddTrace<BufferWrite>(stream->platform_specific_handle().stream,
                              Buffer{device_ordinal, arguments[i]}, source);

  } else if (THUNK_CASE(gpu::CopyThunk)) {
    AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                         Buffer{device_ordinal, t->source()}, source);
    AddTrace<BufferWrite>(stream->platform_specific_handle().stream,
                          Buffer{device_ordinal, t->destination()}, source);

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

void ConcurrencyTracer::OnStreamEventRecord(const se::Stream& stream,
                                            const se::Event& event) {
  if (ENABLE_LOGS) {
    std::cout << "[Stream] S_" << stream.GetName() << " recorded "
              << "E_" << AssertCuda(event).GetHandle() << std::endl;
  }

  AddTrace<EventRecord>(stream.platform_specific_handle().stream,
                        static_cast<void*>(AssertCuda(&event).GetHandle()));
}

void ConcurrencyTracer::OnStreamEventWait(const se::Stream& stream,
                                          const se::Event& event) {
  if (ENABLE_LOGS) {
    std::cout << "[Stream] E_" << AssertCuda(event).GetHandle() << " -> "
              << "S_" << stream.GetName() << std::endl;
  }

  AddTrace<WaitForEvent>(stream.platform_specific_handle().stream,
                         static_cast<void*>(AssertCuda(&event).GetHandle()));
}

void ConcurrencyTracer::PrintTraces(std::ostream& os) {
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
bool ConcurrencyTracer::Buffer::operator==(const Buffer& another) const {
  if (device_ordinal != another.device_ordinal) return false;
  if (slice != another.slice) return false;
  return true;
}
bool ConcurrencyTracer::Buffer::Overlaps(const Buffer& another) const {
  if (device_ordinal != another.device_ordinal) return false;
  if (slice.allocation() != another.slice.allocation()) return false;
  const uint64_t a_begin = slice.offset();
  const uint64_t a_end = a_begin + slice.size();
  const uint64_t b_begin = another.slice.offset();
  const uint64_t b_end = b_begin + another.slice.size();
  return a_begin < b_end && b_begin < a_end;
}
std::vector<ConcurrencyTracer::DataRace> ConcurrencyTracer::DetectDataRaces()
    const {
  /* ── Collect every memory access (sync + async) ───────────────────── */
  std::vector<MemAccessInfo> acc;
  acc.reserve(trace_.size());

  for (size_t i = 0; i < trace_.size(); ++i) {
    if (auto* r = dynamic_cast<const BufferRead*>(trace_[i].get()); r) {
      acc.push_back({r->stream_id, r->buffer, AccessKind::kRead, i, r->source});
    } else if (auto* w = dynamic_cast<const BufferWrite*>(trace_[i].get()); w) {
      acc.push_back(
          {w->stream_id, w->buffer, AccessKind::kWrite, i, w->source});
    } else if (auto* ar = dynamic_cast<const AsyncBufferRead*>(trace_[i].get());
               ar) {
      acc.push_back({ar->async_stream_id, ar->buffer, AccessKind::kRead, i,
                     ar->source, ar->completion_event_id});
    } else if (auto* aw =
                   dynamic_cast<const AsyncBufferWrite*>(trace_[i].get());
               aw) {
      acc.push_back({aw->async_stream_id, aw->buffer, AccessKind::kWrite, i,
                     aw->source, aw->completion_event_id});
    }
  }

  /* ── Build HB-graph once ───────────────────────────────────────────── */
  EdgeList hb = BuildHappensBeforeGraph();

  auto happens_before = [&](const size_t a, const size_t b) {
    absl::flat_hash_set<size_t> seen;
    std::vector stack = {a};
    while (!stack.empty()) {
      size_t cur = stack.back();
      stack.pop_back();
      if (cur == b) return true;
      if (!seen.insert(cur).second) continue;
      auto it = hb.find(cur);
      if (it == hb.end()) continue;
      stack.insert(stack.end(), it->second.begin(), it->second.end());
    }
    return false;
  };

  /* ── Pair-wise race detection ─────────────────────────────────────── */
  std::vector<DataRace> races;
  for (size_t i = 0; i < acc.size(); ++i) {
    for (size_t j = i + 1; j < acc.size(); ++j) {
      const auto& A = acc[i];
      const auto& B = acc[j];

      if (A.stream_id == B.stream_id) continue;     // same (virtual) stream
      if (!A.buffer.Overlaps(B.buffer)) continue;   // no overlap
      if (!(A.IsWrite() || B.IsWrite())) continue;  // read–read is safe
      if (!happens_before(A.trace_idx, B.trace_idx) &&
          !happens_before(B.trace_idx, A.trace_idx))
        races.push_back({A, B});
    }
  }
  return races;
}

std::ostream& operator<<(std::ostream& os,
                         const ConcurrencyTracer::SourceInfo& source) {
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

void ConcurrencyTracer::PrintDataRaces(std::ostream& os) const {
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

ConcurrencyTracer::EdgeList ConcurrencyTracer::BuildHappensBeforeGraph() const {
  EdgeList hb;
  auto add_edge = [&](const size_t a, const size_t b) { hb[a].insert(b); };

  /* ────────────────────────────────────────────────────────────────────
     Pass 0: remember the last op we have seen on every real stream
             (for program-order edges); also record / wait positions.
     ──────────────────────────────────────────────────────────────────── */
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

  /* ────────────────────────────────────────────────────────────────────
     Pass 1: event edges  (Record  →  Wait)
     ──────────────────────────────────────────────────────────────────── */
  for (const auto& [ev, rec_i] : record_pos) {
    if (auto it = wait_pos.find(ev); it != wait_pos.end())
      for (const size_t w_i : it->second) add_edge(rec_i, w_i);
  }

  /* ────────────────────────────────────────────────────────────────────
     Pass 2: async memory-access edges
             Record → AsyncAccess → Wait   for every access on that event
     ──────────────────────────────────────────────────────────────────── */
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
void ConcurrencyTracer::PrintDot(const EdgeList& graph, std::ostream& out) {
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
