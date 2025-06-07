#ifndef XLA_TESTS_CONCURRENCY_TRACE_BASE_CONCURRENCY_TESTS_H_
#define XLA_TESTS_CONCURRENCY_TRACE_BASE_CONCURRENCY_TESTS_H_

#include "xla/backends/gpu/runtime/executable_stats.h"
#include "xla/backends/gpu/runtime/thunk_sanitizer.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tests/concurrency_trace/concurrency_test_base.h"
#include "xla/tests/concurrency_trace/trace_stats.h"
#include "xla/tests/tg/test_util.h"

#include "absl/time/time.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace xla {

// Provides helper utilities for concurrency tests that compile and run HLO and
// optionally measure tracer performance.
class BaseConcurrencyTests : public PjRtGpuStreamExecutorConcurrencyTestBase {
protected:
  bool measure_performance_ = true;
  bool enable_async_dot_ = false;
  bool enable_wait_for_streams_bug_ = false;

  void SetUp() override {
    PjRtGpuStreamExecutorConcurrencyTestBase::SetUp();
    if (const char *env = getenv("XLA_MEASURE_TRACER_PERF"); env && std::string(env) == "1")
      measure_performance_ = true;
  }

  DebugOptions GetDebugOptionsForTest() const override;

  absl::StatusOr<std::unique_ptr<HloModule>> ParseHloText(absl::string_view hlo_string);

  struct DeviceMesh {
    int num_replicas;
    int num_partitions;
  };

  using ExeWithModule = std::pair<std::unique_ptr<PjRtLoadedExecutable>, std::unique_ptr<HloModule>>;

  void EnablePerformanceMeasurements() { measure_performance_ = true; }
  void EnableAsyncDot() { enable_async_dot_ = true; }
  void EnableSyntheticWaitForStreamsBug() { enable_wait_for_streams_bug_ = true; }

  static size_t GetCurrentRSSBytes();

  void RunTest(std::string_view hlo_string, bool expect_race, int warmup_iters = 3, int measure_iters = 3);

  absl::StatusOr<ExeWithModule> CompileWithModule(std::string_view hlo_string, const DeviceMesh &mesh, DebugOptions *debug_options = nullptr);

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(std::string_view hlo_string, const DeviceMesh &mesh, DebugOptions *debug_options = nullptr);

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(HloModule *module, const DeviceMesh &mesh, DebugOptions *debug_options = nullptr);

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileStableHlo(std::string_view stablehlo_string, const DeviceMesh &mesh,
                                                                         DebugOptions *debug_options = nullptr);

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  Execute(PjRtLoadedExecutable &executable, absl::Span<const absl::Span<const LiteralSlice>> args, const ExecuteOptions &exec_opts = {}) const;

  static std::vector<absl::Span<const LiteralSlice>> MakeInnerSpan(const std::vector<std::vector<LiteralSlice>> &fake_arg_slices);
};

} // namespace xla

#endif // XLA_TESTS_CONCURRENCY_TRACE_BASE_CONCURRENCY_TESTS_H_
