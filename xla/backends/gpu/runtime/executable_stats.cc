#include "xla/backends/gpu/runtime/executable_stats.h"

#include <string>

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/client/local_client.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/backends/gpu/runtime/thunk.h"

namespace xla::gpu {

absl::StatusOr<ExecutableStats> GetExecutableStats(
    PjRtLoadedExecutable* executable) {
  auto* se_loaded =
      dynamic_cast<PjRtStreamExecutorLoadedExecutable*>(executable);
  if (!se_loaded) {
    return absl::InvalidArgumentError("Executable is not StreamExecutor type");
  }
  if (se_loaded->executables().empty()) {
    return absl::InternalError("No underlying executables");
  }
  auto* local_exec = se_loaded->executables()[0].get();
  auto* exec = local_exec->executable();
  auto* gpu_exec = dynamic_cast<gpu::GpuExecutable*>(exec);
  if (!gpu_exec) {
    return absl::InvalidArgumentError(
        "Underlying executable is not GpuExecutable");
  }
  ExecutableStats stats;
  if (gpu_exec->has_module()) {
    stats.hlo_instruction_count = gpu_exec->module().instruction_count();
  }
  gpu_exec->GetThunk().ForAllThunks([&](const Thunk* t) {
    std::string kind = std::string(Thunk::KindToString(t->kind()));
    stats.thunk_counts[kind]++;
  });
  if (auto* ba = gpu_exec->buffer_assignment()) {
    stats.static_buffer_footprint_bytes = ba->GetStats().total_allocation_bytes;
  }
  return stats;
}

}  // namespace xla::gpu
