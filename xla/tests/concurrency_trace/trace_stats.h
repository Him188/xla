#ifndef XLA_TESTS_CONCURRENCY_TRACE_TRACE_STATS_H_
#define XLA_TESTS_CONCURRENCY_TRACE_TRACE_STATS_H_

#include <iosfwd>

#include "xla/backends/gpu/runtime/concurrency_trace.h"
#include "xla/backends/gpu/runtime/executable_stats.h"

namespace xla {

struct RunPerfStats {
  double compilation_time_ms = 0.0;
  double execution_time_ms = 0.0;
  size_t compilation_memory_delta_bytes = 0;
  size_t execution_memory_delta_bytes = 0;
  size_t tracer_memory_usage_bytes = 0;
  double race_detection_time_ms = 0.0;
  size_t races = 0;
};

// Prints JSON describing the trace statistics and executable statistics
// collected from a run. `indent` controls the number of spaces used for
// indentation before the "trace_stats" key.
void PrintTraceAndExecutableStatsJson(const gpu::ConcurrencyTracer::TraceStats &trace_stats, const gpu::ExecutableStats &exec_stats, std::ostream &os,
                                      int indent = 2);

// Prints JSON describing run performance statistics.
void PrintPerfStatsJson(const RunPerfStats &perf_stats, std::ostream &os,
                        int indent = 2);

} // namespace xla

#endif // XLA_TESTS_CONCURRENCY_TRACE_TRACE_STATS_H_
