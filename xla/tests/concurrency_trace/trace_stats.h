#ifndef XLA_TESTS_CONCURRENCY_TRACE_TRACE_STATS_H_
#define XLA_TESTS_CONCURRENCY_TRACE_TRACE_STATS_H_

#include <iosfwd>

#include "xla/backends/gpu/runtime/concurrency_trace.h"
#include "xla/backends/gpu/runtime/executable_stats.h"

namespace xla {

// Prints JSON describing the trace statistics and executable statistics
// collected from a run. `indent` controls the number of spaces used for
// indentation before the "trace_stats" key.
void PrintTraceAndExecutableStatsJson(const gpu::ConcurrencyTracer::TraceStats &trace_stats, const gpu::ExecutableStats &exec_stats, std::ostream &os,
                                      int indent = 2);

} // namespace xla

#endif // XLA_TESTS_CONCURRENCY_TRACE_TRACE_STATS_H_
