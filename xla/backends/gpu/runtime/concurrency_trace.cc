#include "concurrency_trace.h"

#include <iostream>
#include <utility>

#include "copy_thunk.h"
#include "gemm_thunk.h"
#include "kernel_thunk.h"
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

ConcurrencyTracer::ConcurrencyTracer() = default;
ConcurrencyTracer::~ConcurrencyTracer() = default;
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

  SourceInfo source{&thunk};
  const int device_ordinal = params.buffer_allocations->device_ordinal();
  if (THUNK_CASE(GemmThunk)) {
    AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                         Buffer{device_ordinal, t->lhs_buffer()}, source);
    AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                         Buffer{device_ordinal, t->rhs_buffer()}, source);
    AddTrace<BufferWrite>(stream->platform_specific_handle().stream,
                          Buffer{device_ordinal, t->output_buffer()}, source);
  } else if (THUNK_CASE(gpu::KernelThunk)) {
    const auto& arguments = t->arguments();

    // Add reads first
    for (int i = 0; i < arguments.size(); ++i) {
      const auto& argument = arguments[i];
      if (!t->written()[i]) {
        AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                             Buffer{device_ordinal, argument}, source);
      }
    }

    // Then add writes
    for (int i = 0; i < arguments.size(); ++i) {
      const auto& argument = arguments[i];
      if (t->written()[i]) {
        AddTrace<BufferWrite>(stream->platform_specific_handle().stream,
                              Buffer{device_ordinal, argument}, source);
      }
    }
  } else if (THUNK_CASE(gpu::CopyThunk)) {
    AddTrace<BufferRead>(stream->platform_specific_handle().stream,
                         Buffer{device_ordinal, t->source()}, source);
    AddTrace<BufferWrite>(stream->platform_specific_handle().stream,
                          Buffer{device_ordinal, t->destination()}, source);
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
  std::cout << "[Stream] " << "E_" << AssertCuda(event).GetHandle() << "->"
            << "S_" << stream.GetName() << std::endl;
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
    if (const auto* t = dynamic_cast<const BufferRead*>(p.get()); t) {
      os << "[MemoryRead ][device " << t->buffer.device_ordinal << "] stream=0x"
         << std::hex << t->stream_id << " @ " << std::hex << t->buffer.slice
         << std::dec << '\n';
      continue;
    }
    if (const auto* t = dynamic_cast<const BufferWrite*>(p.get()); t) {
      os << "[MemoryWrite][device " << t->buffer.device_ordinal << "] stream=0x"
         << std::hex << t->stream_id << " @ " << std::hex << t->buffer.slice
         << std::dec << '\n';
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
  // Collect all memory accesses.
  std::vector<MemAccessInfo> accesses;
  for (size_t i = 0; i < trace_.size(); ++i) {
    if (const auto* r = dynamic_cast<const BufferRead*>(trace_[i].get()); r) {
      accesses.push_back(
          {r->stream_id, r->buffer, AccessKind::kRead, i, r->source});
    } else if (const auto* w =
                   dynamic_cast<const BufferWrite*>(trace_[i].get());
               w) {
      accesses.push_back(
          {w->stream_id, w->buffer, AccessKind::kWrite, i, w->source});
    }
  }

  // Build happens-before graph once.
  EdgeList hb = BuildHappensBeforeGraph();

  // Helper: reachability with DFS (graph is tiny).
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

  // Pairwise race detection
  std::vector<DataRace> races;
  for (size_t i = 0; i < accesses.size(); ++i) {
    for (size_t j = i + 1; j < accesses.size(); ++j) {
      const auto& a = accesses[i];
      const auto& b = accesses[j];

      // Different streams + overlapping buffer range + at least one write?
      if (a.stream_id == b.stream_id) continue;
      if (!a.buffer.Overlaps(b.buffer)) continue;
      if (!(a.kind == AccessKind::kWrite || b.kind == AccessKind::kWrite))
        continue;

      // If neither happens-before the other, we have a race.
      if (!happens_before(a.trace_idx, b.trace_idx) &&
          !happens_before(b.trace_idx, a.trace_idx)) {
        races.push_back({a, b});
      }
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
}

ConcurrencyTracer::EdgeList ConcurrencyTracer::BuildHappensBeforeGraph() const {
  EdgeList hb;
  auto add_edge = [&](size_t a, size_t b) { hb[a].insert(b); };

  // program order: consecutive records on the same stream
  absl::flat_hash_map<void*, size_t> last_seen_on_stream;
  for (size_t i = 0; i < trace_.size(); ++i) {
    if (auto* t = dynamic_cast<const BufferRead*>(trace_[i].get()); t) {
      if (auto it = last_seen_on_stream.find(t->stream_id);
          it != last_seen_on_stream.end())
        add_edge(it->second, i);
      last_seen_on_stream[t->stream_id] = i;
    } else if (auto* t = dynamic_cast<const BufferWrite*>(trace_[i].get()); t) {
      if (auto it = last_seen_on_stream.find(t->stream_id);
          it != last_seen_on_stream.end())
        add_edge(it->second, i);
      last_seen_on_stream[t->stream_id] = i;
    } else if (auto* t = dynamic_cast<const EventRecord*>(trace_[i].get()); t) {
      if (auto it = last_seen_on_stream.find(t->stream_id);
          it != last_seen_on_stream.end())
        add_edge(it->second, i);
      last_seen_on_stream[t->stream_id] = i;
    } else if (auto* t = dynamic_cast<const WaitForEvent*>(trace_[i].get());
               t) {
      if (auto it = last_seen_on_stream.find(t->stream_id);
          it != last_seen_on_stream.end())
        add_edge(it->second, i);
      last_seen_on_stream[t->stream_id] = i;
    }
  }

  // event edges: Record -> Wait on the *same* event id
  absl::flat_hash_map<void*, size_t> record_pos;
  for (size_t i = 0; i < trace_.size(); ++i) {
    if (auto* rec = dynamic_cast<const EventRecord*>(trace_[i].get()); rec) {
      record_pos[rec->event_id] = i;
    } else if (auto* wait = dynamic_cast<const WaitForEvent*>(trace_[i].get());
               wait) {
      auto rec_it = record_pos.find(wait->event_id);
      if (rec_it != record_pos.end()) add_edge(rec_it->second, i);
    }
  }
  return hb;
}

}  // namespace xla::gpu
