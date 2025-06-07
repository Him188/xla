#include "xla/tests/concurrency_trace/trace_stats.h"

#include <ostream>

namespace xla {

void PrintPerfStatsJson(const RunPerfStats &perf_stats, std::ostream &os,
                        int indent) {
  auto indent_str = std::string(indent, ' ');
  auto indent_inner = std::string(indent + 2, ' ');

  os << indent_str << "\"perf_stats\": {\n";
  os << indent_inner << "\"compilation_time_ms\": " << perf_stats.compilation_time_ms << ",\n";
  os << indent_inner << "\"execution_time_ms\": " << perf_stats.execution_time_ms << ",\n";
  os << indent_inner << "\"compilation_memory_delta_bytes\": " << perf_stats.compilation_memory_delta_bytes << ",\n";
  os << indent_inner << "\"execution_memory_delta_bytes\": " << perf_stats.execution_memory_delta_bytes << ",\n";
  os << indent_inner << "\"tracer_memory_usage_bytes\": " << perf_stats.tracer_memory_usage_bytes << ",\n";
  os << indent_inner << "\"race_detection_time_ms\": " << perf_stats.race_detection_time_ms << ",\n";
  os << indent_inner << "\"races\": " << perf_stats.races << "\n";
  os << indent_str << "},";
}

void PrintTraceAndExecutableStatsJson(const gpu::ConcurrencyTracer::TraceStats &trace_stats, const gpu::ExecutableStats &exec_stats, std::ostream &os,
                                      int indent) {
  auto indent_str = std::string(indent, ' ');
  auto indent_inner = std::string(indent + 2, ' ');

  os << indent_str << "\"trace_stats\": {\n";
  os << indent_inner << "\"buffer_reads\": " << trace_stats.buffer_reads << ",\n";
  os << indent_inner << "\"async_buffer_reads\": " << trace_stats.async_buffer_reads << ",\n";
  os << indent_inner << "\"buffer_writes\": " << trace_stats.buffer_writes << ",\n";
  os << indent_inner << "\"async_buffer_writes\": " << trace_stats.async_buffer_writes << ",\n";
  os << indent_inner << "\"event_records\": " << trace_stats.event_records << ",\n";
  os << indent_inner << "\"wait_for_events\": " << trace_stats.wait_for_events << ",\n";
  os << indent_inner << "\"unique_streams\": " << trace_stats.unique_streams << "\n";
  os << indent_str << "},\n";

  os << indent_str << "\"executable_stats\": {\n";
  os << indent_inner << "\"hlo_instruction_count\": " << exec_stats.hlo_instruction_count << ",\n";
  os << indent_inner << "\"static_buffer_footprint_bytes\": " << exec_stats.static_buffer_footprint_bytes << ",\n";
  os << indent_inner << "\"thunk_counts\": {";
  bool first = true;
  for (const auto &kv : exec_stats.thunk_counts) {
    if (!first)
      os << ", ";
    os << "\\\"" << kv.first << "\\\": " << kv.second;
    first = false;
  }
  os << "}\n";
  os << indent_str << "}\n";
}

} // namespace xla
