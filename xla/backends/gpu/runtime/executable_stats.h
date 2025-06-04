#ifndef XLA_BACKENDS_GPU_RUNTIME_EXECUTABLE_STATS_H_
#define XLA_BACKENDS_GPU_RUNTIME_EXECUTABLE_STATS_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

namespace xla {
class PjRtLoadedExecutable;
}

namespace xla::gpu {

struct ExecutableStats {
  int64_t hlo_instruction_count = 0;
  absl::flat_hash_map<std::string, int64_t> thunk_counts;
  int64_t static_buffer_footprint_bytes = 0;
};

absl::StatusOr<ExecutableStats> GetExecutableStats(
    PjRtLoadedExecutable* executable);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_EXECUTABLE_STATS_H_
