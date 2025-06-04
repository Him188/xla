#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"
#include "xla/backends/gpu/runtime/concurrency_trace.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/tests/concurrency_trace/concurrency_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/test_utils.h"
#include "xla/tests/tg/test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <utility>

namespace xla {

class LatencyHidingSchedulerConcurrencyTests : public PjRtGpuStreamExecutorConcurrencyTestBase {
protected:
  bool measure_performance_ = true;

  void SetUp() override {
    PjRtGpuStreamExecutorConcurrencyTestBase::SetUp();
    if (const char *env = getenv("XLA_MEASURE_TRACER_PERF"); env && std::string(env) == "1")
      measure_performance_ = true;
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions dbg = PjRtGpuStreamExecutorConcurrencyTestBase::GetDebugOptionsForTest();
    dbg.set_xla_gpu_enable_latency_hiding_scheduler(true);
    // dbg.set_xla_gpu_dump_llvmir(true);
    // dbg.set_xla_dump_hlo_as_html(true);
    dbg.set_xla_gpu_enable_pipelined_collectives(true);
    dbg.set_xla_gpu_enable_pipelined_all_reduce(true);
    // dbg.set_xla_gpu_all_reduce_combine_threshold_bytes(999999999999);
    dbg.set_xla_gpu_copy_insertion_use_region_analysis(true);
    dbg.clear_xla_gpu_enable_command_buffer();
    dbg.add_xla_gpu_enable_command_buffer(DebugOptions_CommandBufferCmdType_INVALID);
    return dbg;
  }

  absl::StatusOr<std::unique_ptr<HloModule>> ParseHloText(absl::string_view hlo_string) {
    return ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest());
  }

  struct DeviceMesh {
    int num_replicas;
    int num_partitions;
  };

  using ExeWithModule = std::pair<std::unique_ptr<PjRtLoadedExecutable>, std::unique_ptr<HloModule>>;

  void EnablePerformanceMeasurements() { measure_performance_ = true; }

  static size_t GetCurrentRSSBytes() {
    long rss = 0;
    FILE *fp = fopen("/proc/self/statm", "r");
    if (fp != nullptr) {
      if (fscanf(fp, "%*s%ld", &rss) != 1)
        rss = 0;
      fclose(fp);
    }
    return rss * sysconf(_SC_PAGESIZE);
  }

  void RunTest(const std::string_view hlo_string, bool expect_race, int warmup_iters = 3, int measure_iters = 3) {
    setenv("NCCL_DEBUG", "INFO", 1);

    ASSERT_GE(client().addressable_devices().size(), 2) << "Need at least two visible CUDA devices.";

    // Helper struct storing all resources required for a single iteration.
    struct IterationResources {
      std::unique_ptr<PjRtLoadedExecutable> exe;
      std::unique_ptr<HloModule> module;
      std::vector<std::vector<Literal>> fake_args;
      std::vector<std::vector<LiteralSlice>> fake_arg_slices;
      std::vector<absl::Span<const LiteralSlice>> exec_args;
    };

    // Prepare a fresh executable and fake arguments for each iteration while
    // keeping the parsed module alive for the lifetime of the executable.
    auto prepare_fn = [&]() -> absl::StatusOr<IterationResources> {
      IterationResources res;
      TF_ASSIGN_OR_RETURN(res.module, ParseHloText(hlo_string));
      TF_ASSIGN_OR_RETURN(res.fake_args, MakeFakeArgumentsForDevices(res.module.get(), 2));
      res.fake_arg_slices = MakeFakeArgumentSlices(res.fake_args);
      res.exec_args = MakeInnerSpan(res.fake_arg_slices);
      TF_ASSIGN_OR_RETURN(res.exe, Compile(res.module.get(), {2, 1}));
      return res;
    };

    if (!measure_performance_) {
      gpu::ConcurrencyTracer tracer;
      ExecuteOptions exec_opts;
      exec_opts.gpu_concurrency_tracer = &tracer;
      exec_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;

      TF_ASSERT_OK_AND_ASSIGN(auto res, prepare_fn());
      auto &exe = res.exe;
      {
        TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe, res.exec_args, exec_opts));
      }

      xla_test_util::print_gpu_thunk_info(exe.get());
      auto races = tracer.DetectDataRaces();
      std::cout << "races=" << races.size() << std::endl;
      if (!races.empty()) {
        tracer.PrintDataRaces(std::cout);
      }
      tracer.PrintTraces(std::cout);
      ASSERT_EQ(races.empty(), !expect_race);
      exe->Delete();
    } else {
      // Retrieve warmup iterations from env if provided.
      if (const char *env = getenv("XLA_TRACER_WARMUP")) {
        warmup_iters = std::stoi(env);
      }

      std::vector<double> base_times;
      std::vector<size_t> base_memory;
      std::vector<double> traced_times;
      std::vector<size_t> traced_memory;
      std::vector<size_t> tracer_memory;

      ExecuteOptions base_opts;
      base_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;

      // Warm up baseline runs.
      for (int i = 0; i < warmup_iters; ++i) {
        TF_ASSERT_OK_AND_ASSIGN(auto res, prepare_fn());
        auto &exe = res.exe;
        {
          TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe, res.exec_args, base_opts));
        }
        exe->Delete();
      }

      for (int i = 0; i < measure_iters; ++i) {
        TF_ASSERT_OK_AND_ASSIGN(auto res, prepare_fn());
        auto &exe = res.exe;
        size_t rss_before = GetCurrentRSSBytes();
        absl::Time t0 = absl::Now();
        {
          TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe, res.exec_args, base_opts));
        }
        absl::Time t1 = absl::Now();
        size_t rss_after = GetCurrentRSSBytes();
        base_times.push_back(absl::ToDoubleMilliseconds(t1 - t0));
        base_memory.push_back(rss_after - rss_before);
        exe->Delete();
      }

      // Warm up traced runs.
      for (int i = 0; i < warmup_iters; ++i) {
        TF_ASSERT_OK_AND_ASSIGN(auto res, prepare_fn());
        auto &exe = res.exe;
        gpu::ConcurrencyTracer tracer;
        ExecuteOptions exec_opts;
        exec_opts.gpu_concurrency_tracer = &tracer;
        exec_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;
        {
          TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe, res.exec_args, exec_opts));
        }
        auto races = tracer.DetectDataRaces();
        ASSERT_EQ(races.empty(), !expect_race);
        exe->Delete();
      }

      for (int i = 0; i < measure_iters; ++i) {
        TF_ASSERT_OK_AND_ASSIGN(auto res, prepare_fn());
        auto &exe = res.exe;
        gpu::ConcurrencyTracer tracer;
        ExecuteOptions exec_opts;
        exec_opts.gpu_concurrency_tracer = &tracer;
        exec_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;
        size_t rss_before = GetCurrentRSSBytes();
        absl::Time t0 = absl::Now();
        {
          TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe, res.exec_args, exec_opts));
        }
        absl::Time t1 = absl::Now();
        size_t rss_after = GetCurrentRSSBytes();
        traced_times.push_back(absl::ToDoubleMilliseconds(t1 - t0));
        traced_memory.push_back(rss_after - rss_before);
        tracer_memory.push_back(tracer.GetApproximateMemoryUsage());
        auto races = tracer.DetectDataRaces();
        ASSERT_EQ(races.empty(), !expect_race);
        exe->Delete();
      }

      auto mean_std_err = [](const std::vector<double> &vals) {
        double mean = 0.0;
        for (double v : vals)
          mean += v;
        mean /= vals.size();
        double var = 0.0;
        for (double v : vals)
          var += (v - mean) * (v - mean);
        double stddev = std::sqrt(var / (vals.size() > 1 ? vals.size() - 1 : 1));
        double stderr = stddev / std::sqrt(vals.size());
        return std::pair<double, double>(mean, stderr);
      };

      auto mean_std_err_size = [&mean_std_err](const std::vector<size_t> &vals) {
        std::vector<double> tmp(vals.begin(), vals.end());
        return mean_std_err(tmp);
      };

      auto [base_mean, base_err] = mean_std_err(base_times);
      auto [base_mem_mean, base_mem_err] = mean_std_err_size(base_memory);
      auto [traced_mean, traced_err] = mean_std_err(traced_times);
      auto [traced_mem_mean, traced_mem_err] = mean_std_err_size(traced_memory);
      auto [tracer_mem_mean, tracer_mem_err] = mean_std_err_size(tracer_memory);

      std::ostringstream os;
      os << "{\n";
      os << "  \"warmup_iterations\": " << warmup_iters << ",\n";
      os << "  \"measure_iterations\": " << measure_iters << ",\n";
      os << "  \"baseline\": {\n";
      os << "    \"avg_execution_time_ms\": " << base_mean << ",\n";
      os << "    \"stderr_execution_time_ms\": " << base_err << ",\n";
      os << "    \"avg_memory_delta_bytes\": " << base_mem_mean << ",\n";
      os << "    \"stderr_memory_delta_bytes\": " << base_mem_err << "\n";
      os << "  },\n";
      os << "  \"traced\": {\n";
      os << "    \"avg_execution_time_ms\": " << traced_mean << ",\n";
      os << "    \"stderr_execution_time_ms\": " << traced_err << ",\n";
      os << "    \"avg_memory_delta_bytes\": " << traced_mem_mean << ",\n";
      os << "    \"stderr_memory_delta_bytes\": " << traced_mem_err << ",\n";
      os << "    \"avg_tracer_memory_usage_bytes\": " << tracer_mem_mean << ",\n";
      os << "    \"stderr_tracer_memory_usage_bytes\": " << tracer_mem_err << "\n";
      os << "  }\n";
      os << "}";
      std::cout << os.str() << std::endl;
    }
  }

  absl::StatusOr<ExeWithModule> CompileWithModule(const std::string_view hlo_string, const DeviceMesh &mesh, DebugOptions *debug_options = nullptr) {
    TF_ASSIGN_OR_RETURN(auto module, ParseHloText(hlo_string));
    TF_ASSIGN_OR_RETURN(auto exe, Compile(module.get(), mesh, debug_options));
    return ExeWithModule(std::move(exe), std::move(module));
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(const std::string_view hlo_string, const DeviceMesh &mesh,
                                                                DebugOptions *debug_options = nullptr) {
    TF_ASSIGN_OR_RETURN(auto pair, CompileWithModule(hlo_string, mesh, debug_options));
    return std::move(pair.first);
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(HloModule *module, const DeviceMesh &mesh, DebugOptions *debug_options = nullptr) {
    module->mutable_config().set_replica_count(mesh.num_replicas);

    CompileOptions copts;
    auto &eb = copts.executable_build_options;
    eb.set_num_replicas(mesh.num_replicas);
    eb.set_num_partitions(mesh.num_partitions);

    TF_ASSIGN_OR_RETURN(const auto device_assignment, client().GetDefaultDeviceAssignment(mesh.num_replicas, mesh.num_partitions));
    eb.set_device_assignment(device_assignment);
    if (debug_options == nullptr) {
      *eb.mutable_debug_options() = GetDebugOptionsForTest();
    } else {
      *eb.mutable_debug_options() = *debug_options;
    }
    return client().Compile({module->ToProto()}, copts);
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileStableHlo(const std::string_view stablehlo_string, const DeviceMesh &mesh,
                                                                         DebugOptions *debug_options = nullptr) {
    mlir::DialectRegistry registry;
    mlir::func::registerAllExtensions(registry);
    mlir::stablehlo::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(stablehlo_string, &context);
    if (!module) {
      return diagnostic_handler.ConsumeStatus();
    }

    CompileOptions copts;
    auto &eb = copts.executable_build_options;
    eb.set_num_replicas(mesh.num_replicas);
    eb.set_num_partitions(mesh.num_partitions);

    TF_ASSIGN_OR_RETURN(const auto device_assignment, client().GetDefaultDeviceAssignment(mesh.num_replicas, mesh.num_partitions));
    eb.set_device_assignment(device_assignment);
    if (debug_options == nullptr) {
      *eb.mutable_debug_options() = GetDebugOptionsForTest();
    } else {
      *eb.mutable_debug_options() = *debug_options;
    }
    return client().Compile(module.get(), copts);
  }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  Execute(PjRtLoadedExecutable &executable, const absl::Span<const absl::Span<const LiteralSlice>> args, const ExecuteOptions &exec_opts = {}) const {
    // Create device buffers for literals
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> buffers;
    buffers.reserve(args.size());

    for (size_t device_index = 0; device_index < args.size(); ++device_index) {
      TF_ASSIGN_OR_RETURN(auto *mem_space, client().addressable_devices()[device_index]->default_memory_space());
      auto &device_buffers = buffers.emplace_back();
      device_buffers.reserve(args[device_index].size());

      for (const LiteralSlice &arg : args.at(device_index)) {
        TF_ASSIGN_OR_RETURN(auto buffer, client().BufferFromHostLiteral(arg, mem_space));
        TF_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
        device_buffers.emplace_back(std::move(buffer));
      }
    }

    // Get pointer views
    std::vector<std::vector<PjRtBuffer *>> buffer_ptrs;
    buffer_ptrs.reserve(buffers.size());

    for (auto &device_buffers : buffers) {
      auto &ptrs = buffer_ptrs.emplace_back();
      ptrs.reserve(device_buffers.size());
      for (auto &buf : device_buffers)
        ptrs.push_back(buf.get());
    }

    TF_ASSIGN_OR_RETURN(auto res, executable.Execute(buffer_ptrs, exec_opts));

    // Wait for the transfer
    for (auto &device_buffers : res) {
      for (const auto &buf : device_buffers) {
        TF_RETURN_IF_ERROR(buf->GetReadyFuture().Await());
      }
    }
    return res;
  }

  static std::vector<absl::Span<const LiteralSlice>> MakeInnerSpan(const std::vector<std::vector<LiteralSlice>> &fake_arg_slices) {
    std::vector<absl::Span<const LiteralSlice>> exec_args;
    exec_args.reserve(fake_arg_slices.size());

    for (const auto &slices : fake_arg_slices) {
      exec_args.emplace_back(slices);
    }
    return exec_args;
  }
};

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllScatterBug) {
  setenv("NCCL_DEBUG", "WARN", 1);

  ASSERT_GE(client().addressable_devices().size(), 2) << "Need at least two visible CUDA devices.";

  const auto hlo_string = R"(
HloModule module_with_counter

while_cond {
  param = (bf16[8]{0}, bf16[8]{0}, s32[]) parameter(0)
  iters = s32[] get-tuple-element(param), index=2
  zero = s32[] constant(0)
  ROOT keep_going = pred[] compare(iters, zero), direction=GT
}

while_body {
  param = (bf16[8]{0}, bf16[8]{0}, s32[]) parameter(0)
  A_in = bf16[8]{0} get-tuple-element(param), index=0
  bitcast = bf16[8]{0} bitcast(A_in)
  cp1 = bf16[8]{0} collective-permute(A_in), source_target_pairs={{0,1},{1,0}}
  add0 = bf16[8]{0} add(cp1, bitcast)
  neg = bf16[8]{0} negate(add0)
  cp2 = bf16[8]{0} collective-permute(cp1), source_target_pairs={{1,0},{0,1}}
  iters_in = s32[] get-tuple-element(param), index=2
  one = s32[] constant(1)
  iters_next = s32[] subtract(iters_in, one)
  ROOT tuple = (bf16[8]{0}, bf16[8]{0}, s32[]) tuple(cp2, neg, iters_next)
}

ENTRY entry {
  p0 = bf16[8]{0} parameter(0)
  p1 = bf16[8]{0} parameter(1)
  p2 = s32[] parameter(2)
  init_state = (bf16[8]{0}, bf16[8]{0}, s32[]) tuple(p0, p1, p2)
  loop_state = (bf16[8]{0}, bf16[8]{0}, s32[]) while(init_state), condition=while_cond, body=while_body
  A_final = bf16[8]{0} get-tuple-element(loop_state), index=0
  B_final = bf16[8]{0} get-tuple-element(loop_state), index=1
  ROOT out = bf16[8]{0} add(A_final, B_final)
}
)";
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_latency_hiding_scheduler_synthetic_remove_control_deps(true);
  TF_ASSERT_OK_AND_ASSIGN(auto exe_with_module, CompileWithModule(hlo_string, {2, 1}, &debug_options));
  auto &exe = exe_with_module.first;

  // ---------------- host data -------------------------------------------- //
  Literal mat = LiteralUtil::CreateFull({8}, static_cast<bfloat16>(1.0f));
  Literal pred = LiteralUtil::CreateR0<int>(1);

  gpu::ConcurrencyTracer tracer;
  ExecuteOptions exec_opts;
  exec_opts.gpu_concurrency_tracer = &tracer;
  exec_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;

  // Execute
  TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe,
                                             {
                                                 {mat, mat, pred},
                                                 {mat, mat, pred},
                                             },
                                             exec_opts));
  // Print compiled thunks
  xla_test_util::print_gpu_thunk_info(exe.get());
  auto races = tracer.DetectDataRaces();
  std::cout << "races=" << races.size() << std::endl;
  if (!races.empty()) {
    tracer.PrintDataRaces(std::cout);
  }
  tracer.PrintTraces(std::cout);
  ASSERT_FALSE(races.empty());
}

// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, RunStableHloModule) {
//   auto stablehlo = R"(
// module @stablehlo_program {
//   func.func @main(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
//     %0 = stablehlo.add %a, %b : tensor<8xf32>
//     func.return %0 : tensor<8xf32>
//   }
// }
// )";
//
//   ASSERT_GE(client().addressable_devices().size(), 1);
//   DebugOptions debug_options = GetDebugOptionsForTest();
//   TF_ASSERT_OK_AND_ASSIGN(auto exe, CompileStableHlo(stablehlo, {1, 1}, &debug_options));
//
//   Literal lhs = LiteralUtil::CreateR1<float>({1, 2, 3, 4, 5, 6, 7, 8});
//   Literal rhs = LiteralUtil::CreateR1<float>({1, 1, 1, 1, 1, 1, 1, 1});
//   TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe, {{lhs, rhs}}));
//
//   TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> result, outs[0][0]->ToLiteralSync());
//   Literal expected = LiteralUtil::CreateR1<float>({2, 3, 4, 5, 6, 7, 8, 9});
//   EXPECT_TRUE(LiteralTestUtil::Equal(expected, *result));
// }

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllGatherAsyncSimple) {
  RunTest(R"(
        HloModule module, is_scheduled=false


ENTRY %module {
  %constant.19 = u32[] constant(0)
  %replica_id = u32[]{:T(128)} replica-id()
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(
    f32[8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done = f32[16,256,256] all-gather-done(
    (f32[8,256,256], f32[16,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT a2 = f32[16,256,256]{2,1,0} add(%ag-done, c0)
}
        )",
          false);
}

// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllGatherAsyncBalance) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY %module {
//   %constant.19 = u32[] constant(0)
//   %replica_id = u32[]{:T(128)} replica-id()
//   %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
//   %color_operand.1 = f32[1,8,256,256]{3,2,1,0:T(8,128)} broadcast(
//     f32[]{:T(128)} %convert), dimensions={}
//   %ag-start = (f32[1,8,256,256], f32[2,8,256,256]) all-gather-start(
//     f32[1,8,256,256] %color_operand.1), replica_groups={{0,1}, {1,0}}, dimensions={0},
//     metadata={op_type="AllGather" op_name="ag0"}
//   %ag-done = f32[2,8,256,256] all-gather-done(
//     (f32[1,8,256,256], f32[2,8,256,256]) %ag-start),
//     metadata={op_type="AllGather" op_name="ag0"}
//   %ag-done-bc = f32[16,256,256] bitcast(f32[2,8,256,256] %ag-done),
//     metadata={op_type="Bitcast" op_name="ag0"}
//   %ag-start.2 = (f32[1,8,256,256], f32[2,8,256,256]) all-gather-start(
//     f32[1,8,256,256] %color_operand.1), replica_groups={{0,1}, {1,0}}, dimensions={0},
//     metadata={op_type="AllGather" op_name="ag1"}
//   %ag-done.2 = f32[2,8,256,256] all-gather-done(
//     (f32[1,8,256,256], f32[2,8,256,256]) %ag-start.2),
//     metadata={op_type="AllGather" op_name="ag1"}
//   %ag-done-bc.2 = f32[16,256,256] bitcast(f32[2,8,256,256] %ag-done.2),
//     metadata={op_type="Bitcast" op_name="ag1"}
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[16,64,256]{2,1,0} parameter(1)
//   p2 = f32[16,256,256]{2,1,0} parameter(2)
//   p3 = f32[16,256,256]{2,1,0} parameter(3)
//   c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//     metadata={op_type="AllGather" op_name="c0"}
//   c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//     metadata={op_type="AllGather" op_name="c1"}
//   a2 = f32[16,256,256]{2,1,0} add(c1, c0)
//   ROOT t = (f32[16,256,256], f32[16,256,256], f32[16,256,256]) tuple(a2, %ag-done-bc.2, %ag-done-bc)
// }
//         )",
//           true);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllGatherAsyncReshaped) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
//
// ENTRY %module {
//   %constant.19 = u32[] constant(0)
//   %replica_id = u32[]{:T(128)} replica-id()
//   %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
//   %color_operand.1 = f32[1,8,256,256]{3,2,1,0:T(8,128)} broadcast(
//     f32[]{:T(128)} %convert), dimensions={}
//   %ag-start = (f32[1,8,256,256], f32[2,8,256,256]) all-gather-start(
//     f32[1,8,256,256] %color_operand.1), replica_groups={{0,1}, {1,0}}, dimensions={0},
//     metadata={op_type="AllGather" op_name="ag0"}
//   %ag-done = f32[2,8,256,256] all-gather-done(
//     (f32[1,8,256,256], f32[2,8,256,256]) %ag-start),
//     metadata={op_type="AllGather" op_name="ag0"}
//   %ag-done-bc = f32[16,256,256] bitcast(f32[2,8,256,256] %ag-done),
//     metadata={op_type="Bitcast" op_name="ag0"}
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[16,64,256]{2,1,0} parameter(1)
//   p2 = f32[16,256,256]{2,1,0} parameter(2)
//   p3 = f32[16,256,256]{2,1,0} parameter(3)
//   c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   ROOT a2 = f32[16,256,256]{2,1,0} add(%ag-done-bc, c0)
// }
//         )",
//           false);
// }

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllGatherAsyncOverlapped) {
  RunTest(R"(
        HloModule module, is_scheduled=false


ENTRY %module {
  %constant.19 = u32[] constant(1)
  %replica_id = u32[]{:T(128)} replica-id()
  %add.1 = u32[]{:T(128)} add(replica_id, constant.19)
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %convert.1 = f32[]{:T(128)} convert(u32[]{:T(128)} %add.1)
  %color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert), dimensions={}
  %color_operand.2 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert.1), dimensions={}
  %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-start.2 = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.2), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag1"}
  %ag-done = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done.2 = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start.2),
    metadata={op_type="AllGather" op_name="ag1"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT a2 = f32[16,256,256]{2,1,0} add(%ag-done, %ag-done.2)
}
        )",
          false);
}

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllGatherAsyncOverlapped2) {
  RunTest(R"(
        HloModule module, is_scheduled=false


ENTRY %module {
  %constant.19 = u32[] constant(1)
  %replica_id = u32[]{:T(128)} replica-id()
  %add.1 = u32[]{:T(128)} add(replica_id, constant.19)
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %convert.1 = f32[]{:T(128)} convert(u32[]{:T(128)} %add.1)
  %color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert), dimensions={}
  %color_operand.2 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert.1), dimensions={}
  %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-start.2 = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.2), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag1"}
  %ag-done = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done.2 = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start.2),
    metadata={op_type="AllGather" op_name="ag1"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  c0 = f32[16,256,256]{2,1,0} convolution(ag-done, ag-done.2),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT a2 = f32[16,256,256]{2,1,0} add(%c0, %c1)
}
        )",
          false);
}

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllGatherAsyncOverlapped3) {
  RunTest(R"(
        HloModule module, is_scheduled=false


ENTRY %module {
  %constant.19 = u32[] constant(1)
  %replica_id = u32[]{:T(128)} replica-id()
  %add.1 = u32[]{:T(128)} add(replica_id, constant.19)
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %convert.1 = f32[]{:T(128)} convert(u32[]{:T(128)} %add.1)
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  %color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert), dimensions={}
  %color_operand.2 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert.1), dimensions={}
  %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-start.2 = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.2), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag1"}
  %ag-done = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done.2 = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start.2),
    metadata={op_type="AllGather" op_name="ag1"}
  c0 = f32[16,256,256]{2,1,0} convolution(ag-done, ag-done.2),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT a2 = f32[16,256,256]{2,1,0} add(%ag-done, %ag-done.2)
}
        )",
          false);
}

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllReduceAsyncBalance) {
  return;
  RunTest(R"(
        HloModule module, is_scheduled=false

%add {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %a = f32[] add(p0, p1)
}

ENTRY %module {
  %constant.19 = u32[] constant(0)
  %replica_id = u32[]{:T(128)} replica-id()
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %color_operand.1 = f32[2,8,256,256]{3,2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %color_operand.2 = f32[2,8,256,256]{3,2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %ar-start = f32[2,8,256,256] all-reduce-start(
    f32[2,8,256,256] %color_operand.1), replica_groups={{0,1}, {1,0}}, to_apply=%add,
    metadata={op_type="AllReduce" op_name="ar0"}
  %ar-start.2 = f32[2,8,256,256] all-reduce-start(
    f32[2,8,256,256] %color_operand.2), replica_groups={{0,1}, {1,0}}, to_apply=%add,
    metadata={op_type="AllReduce" op_name="ar1"}
  %ar-done = f32[2,8,256,256] all-reduce-done(
    f32[2,8,256,256] %ar-start),
    metadata={op_type="AllReduce" op_name="ar0"}
  %ar-done-bc = f32[16,256,256] bitcast(f32[2,8,256,256] %ar-done),
    metadata={op_type="Bitcast" op_name="ar0"}
  %ar-done.2 = f32[2,8,256,256] all-reduce-done(
    f32[2,8,256,256] %ar-start.2),
    metadata={op_type="AllReduce" op_name="ar1"}
  %ar-done-bc.2 = f32[16,256,256] bitcast(f32[2,8,256,256] %ar-done.2),
    metadata={op_type="Bitcast" op_name="ar1"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllReduce" op_name="c0"}
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllReduce" op_name="c1"}
  a2 = f32[16,256,256]{2,1,0} add(c1, c0)
  ROOT t = (f32[16,256,256], f32[16,256,256], f32[16,256,256]) tuple(a2, %ar-done-bc.2, %ar-done-bc)
}
        )",
          false);
}
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, WhileLoopAliasingBug) {
//   return;
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// while_cond {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   gte0 = bf16[8]{0} get-tuple-element(param), index=0
//   gte1 = pred[] get-tuple-element(param), index=2
//   bitcast = bf16[8]{0} bitcast(gte0)
//   collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1},{1,0}}
//   add0 = bf16[8]{0} add(collective-permute.1, bitcast)
//   negate = bf16[8]{0} negate(add0)
//   collective-permute.2 = bf16[8]{0} collective-permute(collective-permute.1), source_target_pairs={{1,0},{0,1}}
//   ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte1)
// }
//
// ENTRY entry {
//   p0 = bf16[8]{0} parameter(0)
//   p1 = bf16[8]{0} parameter(1)
//   p2 = pred[] parameter(2)
//   tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
//   while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
//   gte0 = bf16[8]{0} get-tuple-element(while), index=0
//   gte1 = bf16[8]{0} get-tuple-element(while), index=1
//   ROOT add = bf16[8]{0} add(gte0, gte1)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, WhileLoopAliasingBug2) {
//   return;
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// while_cond {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   gte0 = bf16[8]{0} get-tuple-element(param), index=0
//   gte1 = bf16[8]{0} get-tuple-element(param), index=1
//   gte2 = pred[] get-tuple-element(param), index=2
//   negate1 = bf16[8]{0} negate(gte1)
//   collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1}, {1,0}}
//   negate0 = bf16[8]{0} negate(collective-permute.1)
//   collective-permute.2 = bf16[8]{0} collective-permute(negate1), source_target_pairs={{1,0}}
//   ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate0, gte2)
// }
//
// ENTRY entry {
//   p0 = bf16[8]{0} parameter(0)
//   p1 = bf16[8]{0} parameter(1)
//   p2 = pred[] parameter(2)
//   tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
//   while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
//   gte0 = bf16[8]{0} get-tuple-element(while), index=0
//   gte1 = bf16[8]{0} get-tuple-element(while), index=1
//   ROOT add = bf16[8]{0} add(gte0, gte1)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, SingleCollectivePermuteTest) {
//   return; // NCCL operation ncclGroupEnd() failed: unhandled cuda error (run with NCCL_DEBUG=INFO for details). Last NCCL warning(error) log entry (may be
//           // unrelated) 'Failed to CUDA calloc async 24 bytes'.
//   RunTest(R"(
//       HloModule single_collective_permute_test, is_scheduled=false
//   ENTRY after_optimizations_test {
//   %parameter.1 = bf16[8]{0} parameter(0), sharding={replicated}
//   ROOT %collective-permute.1 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1}, {1,0}}, channel_id=1
// }
//       )",
//           false);
// }

// invalid layout
// 2025-06-03 18:56:14.631219: F ./xla/shape.h:203] Check failed: has_layout() element_type: F32 dimensions: 2 dimensions: 4 dimensions: 128
// is_dynamic_dimension: false is_dynamic_dimension: false is_dynamic_dimension: false
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, InplaceUpdateCPTest) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// %fused_computation.1 (param_0.1: f32[4,4,128], param_1.2: u32[]) -> f32[4,4,128] {
//   %param_0.1 = f32[4,4,128]{2,1,0:T(4,128)} parameter(0)
//   %constant.15 = f32[]{:T(128)} constant(1)
//   %broadcast.2 = f32[2,4,128]{2,1,0:T(4,128)} broadcast(f32[]{:T(128)} %constant.15), dimensions={}
//   %param_1.2 = u32[] parameter(1)
//   %constant.14 = u32[] constant(0)
//   ROOT %dynamic-update-slice.1 = f32[4,4,128]{2,1,0:T(4,128)} dynamic-update-slice(f32[4,4,128]{2,1,0:T(4,128)} %param_0.1, f32[2,4,128]{2,1,0:T(4,128)}
//   %broadcast.2, u32[] %param_1.2, u32[] %constant.14, u32[] %constant.14)
// }
//
// ENTRY %module_spmd () -> f32[4,4,128] {
//   %constant.8 = u32[] constant(0)
//   %constant.5 = u32[] constant(2)
//   %tuple.1 = (u32[], u32[], u32[]) tuple(u32[] %constant.8, u32[] %constant.8, u32[] %constant.8)
//   %tuple = (u32[], u32[], u32[]) tuple(u32[] %constant.5, u32[] %constant.8, u32[] %constant.8)
//   %custom-call = f32[4,4,128]{2,1,0:T(4,128)} custom-call(), custom_call_target="AllocateBuffer"
//   %fusion.1 = f32[4,4,128]{2,1,0:T(4,128)} fusion(f32[4,4,128]{2,1,0:T(4,128)} %custom-call, u32[] %constant.5), kind=kLoop, calls=%fused_computation.1
//   %collective-permute = f32[4,4,128]{2,1,0:T(4,128)} collective-permute(f32[4,4,128]{2,1,0:T(4,128)} %fusion.1, f32[4,4,128]{2,1,0:T(4,128)} %fusion.1,
//   (u32[], u32[], u32[]) %tuple, (u32[], u32[], u32[]) %tuple.1), channel_id=958, source_target_pairs={{0,1},{1,0}}, slice_sizes={{2,4,128}},
//   backend_config="{}" ROOT %copy.3 = f32[4,4,128]{2,1,0:T(4,128)} copy(f32[4,4,128]{2,1,0:T(4,128)} %collective-permute)
// }
//         )",
//           false);
// }

// INVALID_ARGUMENT: layout minor_to_major field contains 2 elements, but shape is rank 3: {1, 0}; shape: element_type: F32 dimensions: 33712 dimensions: 8
// dimensions: 128 layout { minor_to_major: 2 minor_to_major: 1 minor_to_major: 0 tiles { dimensions: 8 dimensions: 128 } tail_padding_alignment_in_elements: 1
// } is_dynamic_dimension: false is_dynamic_dimension: false is_dynamic_dimension: false
//  XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, InplaceUpdateCPTest2) {
//    RunTest(R"(
//          HloModule module, is_scheduled=false
//
//  %sum (x.336: f32[], y.336: f32[]) -> f32[] {
//    %x.336 = f32[]{:T(128)} parameter(0)
//    %y.336 = f32[]{:T(128)} parameter(1)
//    ROOT %add.5252 = f32[]{:T(128)} add(f32[]{:T(128)} %x.336, f32[]{:T(128)} %y.336)
//  }
//
//  ENTRY %module () -> f32[33708,1024] {
//    %constant.19 = u32[] constant(0)
//    %replica_id = u32[]{:T(128)} replica-id()
//    %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
//    %color_operand.1 = f32[2128,8,128]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert), dimensions={}
//    %all-gather.1 = f32[4256,8,128]{2,1,0:T(8,128)} all-gather(f32[2128,8,128]{2,1,0:T(8,128)} %color_operand.1), replica_groups={{0,1}}, dimensions={0}
//    %custom-call = f32[33712,8,128]{2,1,0:T(8,128)} custom-call(), custom_call_target="AllocateBuffer"
//    %dynamic-update-slice = f32[33712,8,128]{2,1,0:T(8,128)} dynamic-update-slice(f32[33712,8,128]{2,1,0:T(8,128)} %custom-call,
//    f32[4256,8,128]{2,1,0:T(8,128)} %all-gather.1, u32[] %constant.19, u32[] %constant.19, u32[] %constant.19) %tuple.7 = (u32[], u32[], u32[]) tuple(u32[]
//    %constant.19, u32[] %constant.19, u32[] %constant.19) %constant.20 = u32[] constant(4256) %tuple.8 = (u32[], u32[], u32[]) tuple(u32[] %constant.20, u32[]
//    %constant.19, u32[] %constant.19) %collective-permute.3 = f32[33712,8,128]{2,1,0:T(8,128)} collective-permute(f32[33712,8,128]{2,1,0:T(8,128)}
//    %dynamic-update-slice, f32[33712,8,128]{2,1,0:T(8,128)} %dynamic-update-slice, (u32[], u32[], u32[]) %tuple.7, (u32[], u32[], u32[]) %tuple.8),
//    source_target_pairs={{0,1},{1,0}}, slice_sizes={{4256,8,128}} %tuple.9 = (u32[], u32[], u32[]) tuple(u32[] %constant.20, u32[] %constant.19, u32[]
//    %constant.19) %constant.21 = u32[] constant(8512) %tuple.10 = (u32[], u32[], u32[]) tuple(u32[] %constant.21, u32[] %constant.19, u32[] %constant.19)
//    %collective-permute.4 = f32[33712,8,128]{2,1,0:T(8,128)} collective-permute(f32[33712,8,128]{2,1,0:T(8,128)} %collective-permute.3,
//    f32[33712,8,128]{2,1,0:T(8,128)} %collective-permute.3, (u32[], u32[], u32[]) %tuple.9, (u32[], u32[], u32[]) %tuple.10),
//    source_target_pairs={{0,1},{1,0}}, slice_sizes={{4256,8,128}} %tuple.11 = (u32[], u32[], u32[]) tuple(u32[] %constant.21, u32[] %constant.19, u32[]
//    %constant.19) %constant.22 = u32[] constant(12768) %tuple.12 = (u32[], u32[], u32[]) tuple(u32[] %constant.22, u32[] %constant.19, u32[] %constant.19)
//    %collective-permute.5 = f32[33712,8,128]{2,1,0:T(8,128)} collective-permute(f32[33712,8,128]{2,1,0:T(8,128)} %collective-permute.4,
//    f32[33712,8,128]{2,1,0:T(8,128)} %collective-permute.4, (u32[], u32[], u32[]) %tuple.11, (u32[], u32[], u32[]) %tuple.12),
//    source_target_pairs={{0,1},{1,0}}, slice_sizes={{4256,8,128}} ROOT %bitcast.16 = f32[33708,1024]{1,0:T(8,128)} bitcast(f32[33712,8,128]{2,1,0:T(8,128)}
//    %collective-permute.5)
//  }
//          )",
//            false);
//  }

// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, TwoCollectivePermuteTypesOverlap) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, f32[16,128,256]{2,1,0}) parameter(0)
//   gte0 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=0
//   gte1 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=1
//   cp0 = f32[16,64,256]{2,1,0} collective-permute(gte0),
//     source_target_pairs={{0,1},{1,0}},
//     metadata={op_type="CollectivePermute" op_name="cp0"}
//   cp1 = f32[16,64,256]{2,1,0} collective-permute(cp0),
//     source_target_pairs={{0,1},{1,0}},
//     metadata={op_type="CollectivePermute" op_name="cp1"}
//   c0 = f32[16,256,256]{2,1,0} convolution(gte0, gte1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   cp2 = f32[16,64,256]{2,1,0} collective-permute(gte1),
//     source_target_pairs={{0,1},{1,0}},
//     metadata={op_type="CollectivePermute" op_name="cp2"}
//   c1 = f32[16,256,256]{2,1,0} convolution(cp0, gte1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   cp3 = f32[16,64,256]{2,1,0} collective-permute(cp2),
//     source_target_pairs={{0,1},{1,0}},
//     metadata={op_type="CollectivePermute" op_name="cp3"}
//   gte2 = f32[16,128,256]{2,1,0} get-tuple-element(param), index=2
//   const0 = u32[] constant(0)
//   const1 = u32[] constant(8)
//   tuple0 = (u32[], u32[], u32[]) tuple(u32[] const0, u32[] const0, u32[] const0)
//   tuple1 = (u32[], u32[], u32[]) tuple(u32[] const1, u32[] const0, u32[] const0)
//   cp4 = f32[16,128,256]{2,1,0} collective-permute(gte2, gte2, tuple0, tuple1),
//     source_target_pairs={{0,1},{1,0}},
//     slice_sizes={{8,128,256}},
//     metadata={op_type="CollectivePermute" op_name="cp4"}
//   cp5 = f32[16,128,256]{2,1,0} collective-permute(cp4, cp4, tuple0, tuple1),
//     source_target_pairs={{0,1},{1,0}},
//     slice_sizes={{8,128,256}},
//     metadata={op_type="CollectivePermute" op_name="cp5"}
//   ROOT tuple = (f32[16,256,256]{2,1,0}, f32[16,64,256]{2,1,0}, f32[16,128,256]{2,1,0}) tuple(c1, cp3, cp5)
// }
//         )",
//           false);
// }

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, SerialCollectivePermutesTest) {
  RunTest(R"(
      HloModule serial_collective_permute_test, is_scheduled=false
  ENTRY after_optimizations_test {
  %parameter.1 = bf16[8]{0} parameter(0)
  %collective-permute.2 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1}, {1,0}}
  %add.3 = bf16[8]{0} add(%parameter.1, %parameter.1)
  %add.4 = bf16[8]{0} add(%add.3, parameter.1)
  %add.5 = bf16[8]{0} add(%collective-permute.2, %add.4)
  %collective-permute.6 = bf16[8]{0} collective-permute(bf16[8]{0} add.5), source_target_pairs={{1,0}}
}
      )",
          false);
}

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, BackToBackCollectivePerGmutesTest) {
  RunTest(R"(
      HloModule back_to_back_collective_permute_test, is_scheduled=false
  ENTRY after_optimizations_test {
  %parameter.1 = bf16[8]{0} parameter(0)
  %collective-permute.2 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1}, {1,0}}
  %collective-permute.6 = bf16[8]{0} collective-permute(bf16[8]{0} collective-permute.2), source_target_pairs={{1,0}}
}
      )",
          false);
}

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, ParallelCollectivePermutesTest) {
  RunTest(R"(
      HloModule single_collective_permute_test, is_scheduled=false
  ENTRY after_optimizations_test {
  %parameter.1 = bf16[8]{0} parameter(0)
  %collective-permute.2 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1}, {1,0}}
  %constant.3 = bf16[] constant(1)
  %broadcast.4 = bf16[8]{0} broadcast(bf16[] %constant.3), dimensions={}
  %add.5 = bf16[8]{0} add(bf16[8]{0} %collective-permute.2, bf16[8]{0} %broadcast.4)
  %collective-permute.6 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{1,0}}
  %add.6 = bf16[8]{0} add(bf16[8]{0} %collective-permute.6, bf16[8]{0} %add.5)
}
      )",
          false);
}

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, MaxConcurrentCollectivePermutesTest) {
  RunTest(R"(
      HloModule single_collective_permute_test, is_scheduled=false
  ENTRY after_optimizations_test {
  %parameter.1 = bf16[8]{0} parameter(0)
  %parameter.2 = bf16[8]{0} parameter(1)
  %parameter.3 = bf16[8]{0} parameter(2)
  %collective-permute.4 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1}, {1,0}}
  %collective-permute.5 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{1,0}}
  %collective-permute.6 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.2), source_target_pairs={{0,1}, {1,0}}
  %collective-permute.7 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.2), source_target_pairs={{1,0}}
  %collective-permute.8 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.3), source_target_pairs={{0,1}, {1,0}}
  %collective-permute.9 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.3), source_target_pairs={{1,0}}
  %add.10 = bf16[8]{0} add(bf16[8]{0} %collective-permute.8, bf16[8]{0} %collective-permute.9)
  %add.11 = bf16[8]{0} add(bf16[8]{0} %collective-permute.7, bf16[8]{0} %add.10)
  %add.12 = bf16[8]{0} add(bf16[8]{0} %collective-permute.6, bf16[8]{0} %add.11)
  %add.13 = bf16[8]{0} add(bf16[8]{0} %collective-permute.5, bf16[8]{0} %add.12)
  ROOT %add.14 = bf16[8]{0} add(bf16[8]{0} %collective-permute.4, bf16[8]{0} %add.13)
}
      )",
          false);
}

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, BalanceChainedCollectivePermutesNoOverlap) {
  RunTest(R"(
        HloModule module, is_scheduled=false

ENTRY entry {
  param = bf16[8]{0} parameter(0)
  collective-permute.1 = bf16[8]{0} collective-permute(param), source_target_pairs={{0,1}, {1,0}}
  copy.2 = bf16[8]{0} copy(collective-permute.1)
  ROOT collective-permute.2 = bf16[8]{0} collective-permute(copy.2), source_target_pairs={{1,0}}
}
        )",
          false);
}

// Device configuration error?
// https://chatgpt.com/c/68404f9e-60e4-8007-805a-75bf29901924
// INTERNAL: NCCL operation ncclRecv( recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype, source_rank->value(), comm_,
// se::gpu::AsGpuStreamValue(stream)) failed: invalid argument (run with NCCL_DEBUG=WARN for details). Last NCCL warning(error) log entry (may be unrelated)
// 'Recv : invalid root 1 (root should be in the 0..1 range)'.: while running replica 0 and partition 0 of a replicated computation (other replicas may have
// failed as well).
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests,
//            ExistingSingleCollectivePermuteAsyncSmallTest) {
//   RunTest(R"(
//       HloModule single_collective_permute_test, is_scheduled=false
//   ENTRY after_optimizations_test {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[16,64,256]{2,1,0} parameter(1)
//   p2 = f32[16,256,256]{2,1,0} parameter(2)
//   c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   %collective-permute-start.1 = (f32[16,256,256]{2,1,0},
//     f32[16,256,256]{2,1,0}) collective-permute-start(
//     f32[16,256,256]{2,1,0} p2), source_target_pairs={{0,1}, {1,0}},
//     channel_id=1, metadata={op_type="CollectivePermute" op_name="cp0"}
//   %collective-permute-done.1 = f32[16,256,256]{2,1,0} collective-permute-done(
//     (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}) collective-permute-start.1),
//     metadata={op_type="CollectivePermute" op_name="cp0"}
//   ROOT a = f32[16,256,256]{2,1,0} add(c0, collective-permute-done.1)
// }
//       )",
//           false);
// }

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, BalanceChainExtended) {
  RunTest(R"(
        HloModule module, is_scheduled=false

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  cp0 = f32[16,256,256]{2,1,0} collective-permute(p2),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp0"}
  cp1 = f32[16,256,256]{2,1,0} collective-permute(p3),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp1"}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  t0 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}) tuple(cp0, cp1)
  gte0 = f32[16,256,256]{2,1,0} get-tuple-element(t0), index=0
  gte1 = f32[16,256,256]{2,1,0} get-tuple-element(t0), index=1
  cp2 = f32[16,256,256]{2,1,0} collective-permute(gte0),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp2"}
  a2 = f32[16,256,256]{2,1,0} add(cp2, c0)
  cp3 = f32[16,256,256]{2,1,0} collective-permute(gte1),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp3"}
  ROOT tuple = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}) tuple(a2, cp3)
}
        )",
          false);
}

// OOM
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, BalanceChainedCollectivePermutesLoopedEinsum) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// %fused_computation.1793 (param_0.4944: s32[16], param_1.5648: u32[], param_2.3959: u32[], param_3.3338: u32[], param_4.2302: u32[]) -> (s32[1], s32[1],
// s32[1], s32[1]) {
//   %param_0.4944 = s32[16]{0:T(128)} parameter(0)
//   %param_1.5648 = u32[]{:T(128)} parameter(1)
//   %dynamic-slice.1806 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4944, u32[]{:T(128)} %param_1.5648), dynamic_slice_sizes={1}
//   %param_2.3959 = u32[]{:T(128)} parameter(2)
//   %dynamic-slice.1807 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4944, u32[]{:T(128)} %param_2.3959), dynamic_slice_sizes={1}
//   %param_3.3338 = u32[]{:T(128)} parameter(3)
//   %dynamic-slice.1808 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4944, u32[]{:T(128)} %param_3.3338), dynamic_slice_sizes={1}
//   %param_4.2302 = u32[]{:T(128)} parameter(4)
//   %dynamic-slice.1809 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4944, u32[]{:T(128)} %param_4.2302), dynamic_slice_sizes={1}
//   ROOT %tuple.1384 = (s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %dynamic-slice.1806, s32[1]{0:T(128)}
//   %dynamic-slice.1807, s32[1]{0:T(128)} %dynamic-slice.1808, s32[1]{0:T(128)} %dynamic-slice.1809)
// }
//
// %fused_computation.109 (param_0.225: bf16[8,1024,1,20,256,1,1]) -> bf16[8,1024,1,20,256,1,1,1] {
//   %param_0.225 = bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} parameter(0)
//   ROOT %bitcast.713 = bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} bitcast(bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %param_0.225)
// }
//
// %fused_computation.110.clone (param_0.251: s32[], param_1.277: bf16[1,20,256,1,16,4,288,1], param_2.190: s32[]) -> bf16[1,20,256,2,1,4,288,1] {
//   %param_1.277 = bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} parameter(1)
//   %constant.6014 = bf16[]{:T(256)} constant(-inf)
//   %pad.370 = bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} pad(bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_1.277,
//   bf16[]{:T(256)} %constant.6014), padding=0_0x0_0x0_0x0_1x0_0x0_0x0_0x0_0 %constant.6004 = s32[]{:T(128)} constant(0) %param_0.251 = s32[]{:T(128)}
//   parameter(0) %dynamic-slice.1503 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)}
//   dynamic-slice(bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %pad.370, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004,
//   s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, /*index=5*/s32[]{:T(128)} %param_0.251, s32[]{:T(128)} %constant.6004, s32[]{:T(128)}
//   %constant.6004, s32[]{:T(128)} %constant.6004), dynamic_slice_sizes={1,20,256,2,1,4,288,1} %pad.369 =
//   bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} pad(bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_1.277, bf16[]{:T(256)}
//   %constant.6014), padding=0_0x0_0x0_0x1_0x0_0x0_0x0_0x0_0 %param_2.190 = s32[]{:T(128)} parameter(2) %dynamic-slice.1502 =
//   bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} dynamic-slice(bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %pad.369, s32[]{:T(128)}
//   %constant.6004, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, /*index=5*/s32[]{:T(128)} %param_2.190,
//   s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004), dynamic_slice_sizes={1,20,256,2,1,4,288,1} ROOT %maximum.513
//   = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} maximum(bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %dynamic-slice.1503,
//   bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %dynamic-slice.1502)
// }
//
// %fused_computation.108 (param_0.235: bf16[8,1024,1,20,256,1,1], param_1.276: s32[], param_2.187: bf16[1,20,256,1,16,4,288,1], param_3.145: s32[]) ->
// bf16[2,1,4,288,8,1024,1,1] {
//   %param_1.276 = s32[]{:T(128)} parameter(1)
//   %param_2.187 = bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} parameter(2)
//   %param_3.145 = s32[]{:T(128)} parameter(3)
//   %fusion.132 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} fusion(s32[]{:T(128)} %param_1.276,
//   bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_2.187, s32[]{:T(128)} %param_3.145), kind=kLoop, calls=%fused_computation.110.clone
//   %param_0.235 = bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} parameter(0)
//   %fusion.129 = bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} fusion(bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %param_0.235),
//   kind=kLoop, calls=%fused_computation.109 ROOT %convolution.170 = bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)}
//   convolution(bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %fusion.132, bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} %fusion.129),
//   window={size=1x1x8x1x20x1 pad=0_0x0_0x7_7x0_0x0_0x0_0 rhs_reversal=0x0x1x0x0x0}, dim_labels=34f501b2_2o34i015->501b2f34
// }
//
// %fused_computation.117 (param_0.248: bf16[1,4,288,8,1024,1,1], param_1.273: bf16[2,1,4,288,8,1024,1,1]) -> bf16[1,4,288,8,1024,1,1] {
//   %param_0.248 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} parameter(0)
//   %param_1.273 = bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} parameter(1)
//   %slice.1252 = bf16[1,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} slice(bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %param_1.273),
//   slice={[0:1], [0:1], [0:4], [0:288], [0:8], [0:1024], [0:1], [0:1]} %bitcast.719 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}
//   bitcast(bf16[1,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %slice.1252) ROOT %add.3083 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}
//   add(bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %param_0.248, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %bitcast.719)
// }
//
// %fused_computation.107 (param_0.223: bf16[8,1024,1,20,256,1,1]) -> bf16[8,1024,1,20,256,1,1,1] {
//   %param_0.223 = bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} parameter(0)
//   ROOT %bitcast.711 = bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} bitcast(bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %param_0.223)
// }
//
// %fused_computation.111.clone (param_0.250: s32[], param_1.275: bf16[1,20,256,1,16,4,288,1], param_2.189: s32[]) -> bf16[1,20,256,2,1,4,288,1] {
//   %param_1.275 = bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} parameter(1)
//   %constant.6009 = bf16[]{:T(256)} constant(-inf)
//   %pad.374 = bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} pad(bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_1.275,
//   bf16[]{:T(256)} %constant.6009), padding=0_0x0_0x0_0x0_1x0_0x0_0x0_0x0_0 %constant.5999 = s32[]{:T(128)} constant(0) %param_0.250 = s32[]{:T(128)}
//   parameter(0) %dynamic-slice.1507 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)}
//   dynamic-slice(bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %pad.374, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999,
//   s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, /*index=5*/s32[]{:T(128)} %param_0.250, s32[]{:T(128)} %constant.5999, s32[]{:T(128)}
//   %constant.5999, s32[]{:T(128)} %constant.5999), dynamic_slice_sizes={1,20,256,2,1,4,288,1} %pad.373 =
//   bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} pad(bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_1.275, bf16[]{:T(256)}
//   %constant.6009), padding=0_0x0_0x0_0x1_0x0_0x0_0x0_0x0_0 %param_2.189 = s32[]{:T(128)} parameter(2) %dynamic-slice.1506 =
//   bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} dynamic-slice(bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %pad.373, s32[]{:T(128)}
//   %constant.5999, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, /*index=5*/s32[]{:T(128)} %param_2.189,
//   s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999), dynamic_slice_sizes={1,20,256,2,1,4,288,1} ROOT %maximum.514
//   = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} maximum(bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %dynamic-slice.1507,
//   bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %dynamic-slice.1506)
// }
//
// %fused_computation.106 (param_0.239: bf16[8,1024,1,20,256,1,1], param_1.274: s32[], param_2.185: bf16[1,20,256,1,16,4,288,1], param_3.144: s32[]) ->
// bf16[2,1,4,288,8,1024,1,1] {
//   %param_1.274 = s32[]{:T(128)} parameter(1)
//   %param_2.185 = bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} parameter(2)
//   %param_3.144 = s32[]{:T(128)} parameter(3)
//   %fusion.133 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} fusion(s32[]{:T(128)} %param_1.274,
//   bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_2.185, s32[]{:T(128)} %param_3.144), kind=kLoop, calls=%fused_computation.111.clone
//   %param_0.239 = bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} parameter(0)
//   %fusion.127 = bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} fusion(bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %param_0.239),
//   kind=kLoop, calls=%fused_computation.107 ROOT %convolution.169 = bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)}
//   convolution(bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %fusion.133, bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} %fusion.127),
//   window={size=1x1x8x1x20x1 pad=0_0x0_0x7_7x0_0x0_0x0_0 rhs_reversal=0x0x1x0x0x0}, dim_labels=34f501b2_2o34i015->501b2f34
// }
//
// %fused_computation.115 (param_0.244: bf16[1,4,288,8,1024,1,1], param_1.270: bf16[2,1,4,288,8,1024,1,1]) -> bf16[1,4,288,8,1024,1,1] {
//   %param_0.244 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} parameter(0)
//   %param_1.270 = bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} parameter(1)
//   %slice.1249 = bf16[1,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} slice(bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %param_1.270),
//   slice={[0:1], [0:1], [0:4], [0:288], [0:8], [0:1024], [0:1], [0:1]} %bitcast.716 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}
//   bitcast(bf16[1,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %slice.1249) ROOT %add.3082 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}
//   add(bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %param_0.244, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %bitcast.716)
// }
//
// %fused_computation.113 (param_0.241: bf16[1,4,288,8,1024,1,1], param_1.267: bf16[2,4,288,8,1024,1,1]) -> bf16[1,4,288,8,1024,1,1] {
//   %param_0.241 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} parameter(0)
//   %param_1.267 = bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} parameter(1)
//   %slice.1246 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} slice(bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %param_1.267),
//   slice={[1:2], [0:4], [0:288], [0:8], [0:1024], [0:1], [0:1]} ROOT %add.3081 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)}
//   add(bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %param_0.241, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %slice.1246)
// }
//
// %fused_computation.112 (param_0.240: bf16[1,4,288,8,1024,1,1], param_1.265: bf16[2,4,288,8,1024,1,1]) -> bf16[1,4,288,8,1024,1,1] {
//   %param_0.240 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} parameter(0)
//   %param_1.265 = bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} parameter(1)
//   %slice.1245 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} slice(bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %param_1.265),
//   slice={[1:2], [0:4], [0:288], [0:8], [0:1024], [0:1], [0:1]} ROOT %add.3080 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)}
//   add(bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %param_0.240, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %slice.1245)
// }
//         )",
//           false);
// }

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, MoveCentainConv) {
  RunTest(R"(
        HloModule module, is_scheduled=false

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  cp0 = f32[16,256,256]{2,1,0} collective-permute(p2),
    source_target_pairs={{0,1},{1,0}}
  cp1 = f32[16,256,256]{2,1,0} collective-permute(p3),
    source_target_pairs={{0,1},{1,0}}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  a0 = f32[16,256,256]{2,1,0} add(cp0, c1)
  cp2 = f32[16,256,256]{2,1,0} collective-permute(a0),
    source_target_pairs={{0,1},{1,0}}
  a2 = f32[16,256,256]{2,1,0} add(cp2, c0)
  a1 = f32[16,256,256]{2,1,0} add(cp1, c1)
  cp3 = f32[16,256,256]{2,1,0} collective-permute(a1),
    source_target_pairs={{0,1},{1,0}}
  ROOT tuple = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}) tuple(a2, cp3)
}
        )",
          false);
}

// OOM
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, BalanceChainedCollectivePermutesLoopedEinsum2) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// %fused_computation.1851 (param_0.5170: s32[32], param_1.5848: u32[], param_2.4103: u32[], param_3.3513: u32[], param_4.2356: u32[]) -> (s32[1], s32[1],
// s32[1], s32[1]) {
//   %param_0.5170 = s32[32]{0:T(128)} parameter(0)
//   %param_1.5848 = u32[]{:T(128)} parameter(1)
//   %dynamic-slice.1636 = s32[1]{0:T(128)} dynamic-slice(s32[32]{0:T(128)} %param_0.5170, u32[]{:T(128)} %param_1.5848), dynamic_slice_sizes={1}
//   %param_2.4103 = u32[]{:T(128)} parameter(2)
//   %dynamic-slice.1637 = s32[1]{0:T(128)} dynamic-slice(s32[32]{0:T(128)} %param_0.5170, u32[]{:T(128)} %param_2.4103), dynamic_slice_sizes={1}
//   %param_3.3513 = u32[]{:T(128)} parameter(3)
//   %dynamic-slice.1638 = s32[1]{0:T(128)} dynamic-slice(s32[32]{0:T(128)} %param_0.5170, u32[]{:T(128)} %param_3.3513), dynamic_slice_sizes={1}
//   %param_4.2356 = u32[]{:T(128)} parameter(4)
//   %dynamic-slice.1639 = s32[1]{0:T(128)} dynamic-slice(s32[32]{0:T(128)} %param_0.5170, u32[]{:T(128)} %param_4.2356), dynamic_slice_sizes={1}
//   ROOT %tuple.1297 = (s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %dynamic-slice.1636, s32[1]{0:T(128)}
//   %dynamic-slice.1637, s32[1]{0:T(128)} %dynamic-slice.1638, s32[1]{0:T(128)} %dynamic-slice.1639)
// }
//
// %fused_computation.117 (param_0.249: bf16[16,1024,1,10,256,1]) -> bf16[16,1024,1,10,256,1,1] {
//   %param_0.249 = bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} parameter(0)
//   ROOT %bitcast.672 = bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} bitcast(bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %param_0.249)
// }
//
// %fused_computation.124.clone (param_0.277: s32[], param_1.330: bf16[1,10,256,1,32,576,1], param_2.233: s32[]) -> bf16[1,10,256,2,1,576,1] {
//   %param_1.330 = bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} parameter(1)
//   %constant.5658 = bf16[]{:T(256)} constant(-inf)
//   %pad.357 = bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} pad(bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_1.330, bf16[]{:T(256)}
//   %constant.5658), padding=0_0x0_0x0_0x0_1x0_0x0_0x0_0 %constant.5648 = s32[]{:T(128)} constant(0) %param_0.277 = s32[]{:T(128)} parameter(0)
//   %dynamic-slice.1327 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} dynamic-slice(bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %pad.357,
//   s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, /*index=5*/s32[]{:T(128)}
//   %param_0.277, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648), dynamic_slice_sizes={1,10,256,2,1,576,1} %pad.363 =
//   bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} pad(bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_1.330, bf16[]{:T(256)}
//   %constant.5658), padding=0_0x0_0x0_0x1_0x0_0x0_0x0_0 %param_2.233 = s32[]{:T(128)} parameter(2) %dynamic-slice.1333 =
//   bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} dynamic-slice(bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %pad.363, s32[]{:T(128)}
//   %constant.5648, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, /*index=5*/s32[]{:T(128)} %param_2.233,
//   s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648), dynamic_slice_sizes={1,10,256,2,1,576,1} ROOT %maximum.510 =
//   bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} maximum(bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %dynamic-slice.1327,
//   bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %dynamic-slice.1333)
// }
//
// %fused_computation.116 (param_0.264: bf16[16,1024,1,10,256,1], param_1.329: s32[], param_2.230: bf16[1,10,256,1,32,576,1], param_3.197: s32[]) ->
// bf16[2,1,576,16,1024,1,1] {
//   %param_1.329 = s32[]{:T(128)} parameter(1)
//   %param_2.230 = bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} parameter(2)
//   %param_3.197 = s32[]{:T(128)} parameter(3)
//   %fusion.155 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} fusion(s32[]{:T(128)} %param_1.329,
//   bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_2.230, s32[]{:T(128)} %param_3.197), kind=kLoop, calls=%fused_computation.124.clone
//   %param_0.264 = bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} parameter(0)
//   %fusion.147 = bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} fusion(bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %param_0.264), kind=kLoop,
//   calls=%fused_computation.117 ROOT %convolution.168 = bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)}
//   convolution(bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %fusion.155, bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} %fusion.147),
//   window={size=1x16x1x10x1 pad=0_0x15_15x0_0x0_0x0_0 rhs_reversal=0x1x0x0x0}, dim_labels=23f40b1_1o23i04->40b1f23
// }
//
// %fused_computation.123 (param_0.258: bf16[1,576,16,1024,1,1], param_1.306: bf16[2,1,576,16,1024,1,1]) -> bf16[1,576,16,1024,1,1] {
//   %param_0.258 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} parameter(0)
//   %param_1.306 = bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} parameter(1)
//   %slice.1132 = bf16[1,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} slice(bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %param_1.306),
//   slice={[0:1], [0:1], [0:576], [0:16], [0:1024], [0:1], [0:1]} %bitcast.678 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}
//   bitcast(bf16[1,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %slice.1132) ROOT %add.3125 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}
//   add(bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %param_0.258, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %bitcast.678)
// }
//
// %fused_computation.115 (param_0.247: bf16[16,1024,1,10,256,1]) -> bf16[16,1024,1,10,256,1,1] {
//   %param_0.247 = bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} parameter(0)
//   ROOT %bitcast.670 = bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} bitcast(bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %param_0.247)
// }
//
// %fused_computation.125.clone (param_0.276: s32[], param_1.328: bf16[1,10,256,1,32,576,1], param_2.232: s32[]) -> bf16[1,10,256,2,1,576,1] {
//   %param_1.328 = bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} parameter(1)
//   %constant.5653 = bf16[]{:T(256)} constant(-inf)
//   %pad.360 = bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} pad(bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_1.328, bf16[]{:T(256)}
//   %constant.5653), padding=0_0x0_0x0_0x0_1x0_0x0_0x0_0 %constant.5643 = s32[]{:T(128)} constant(0) %param_0.276 = s32[]{:T(128)} parameter(0)
//   %dynamic-slice.1330 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} dynamic-slice(bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %pad.360,
//   s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, /*index=5*/s32[]{:T(128)}
//   %param_0.276, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643), dynamic_slice_sizes={1,10,256,2,1,576,1} %pad.366 =
//   bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} pad(bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_1.328, bf16[]{:T(256)}
//   %constant.5653), padding=0_0x0_0x0_0x1_0x0_0x0_0x0_0 %param_2.232 = s32[]{:T(128)} parameter(2) %dynamic-slice.1336 =
//   bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} dynamic-slice(bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %pad.366, s32[]{:T(128)}
//   %constant.5643, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, /*index=5*/s32[]{:T(128)} %param_2.232,
//   s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643), dynamic_slice_sizes={1,10,256,2,1,576,1} ROOT %maximum.512 =
//   bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} maximum(bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %dynamic-slice.1330,
//   bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %dynamic-slice.1336)
// }
//
// %fused_computation.114 (param_0.269: bf16[16,1024,1,10,256,1], param_1.327: s32[], param_2.228: bf16[1,10,256,1,32,576,1], param_3.196: s32[]) ->
// bf16[2,1,576,16,1024,1,1] {
//   %param_1.327 = s32[]{:T(128)} parameter(1)
//   %param_2.228 = bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} parameter(2)
//   %param_3.196 = s32[]{:T(128)} parameter(3)
//   %fusion.157 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} fusion(s32[]{:T(128)} %param_1.327,
//   bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_2.228, s32[]{:T(128)} %param_3.196), kind=kLoop, calls=%fused_computation.125.clone
//   %param_0.269 = bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} parameter(0)
//   %fusion.145 = bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} fusion(bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %param_0.269), kind=kLoop,
//   calls=%fused_computation.115 ROOT %convolution.167 = bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)}
//   convolution(bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %fusion.157, bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} %fusion.145),
//   window={size=1x16x1x10x1 pad=0_0x15_15x0_0x0_0x0_0 rhs_reversal=0x1x0x0x0}, dim_labels=23f40b1_1o23i04->40b1f23
// }
//
// %fused_computation.121 (param_0.254: bf16[1,576,16,1024,1,1], param_1.303: bf16[2,1,576,16,1024,1,1]) -> bf16[1,576,16,1024,1,1] {
//   %param_0.254 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} parameter(0)
//   %param_1.303 = bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} parameter(1)
//   %slice.1129 = bf16[1,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} slice(bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %param_1.303),
//   slice={[0:1], [0:1], [0:576], [0:16], [0:1024], [0:1], [0:1]} %bitcast.675 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}
//   bitcast(bf16[1,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %slice.1129) ROOT %add.3124 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}
//   add(bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %param_0.254, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %bitcast.675)
// }
//
// %fused_computation.119 (param_0.251: bf16[1,576,16,1024,1,1], param_1.300: bf16[2,576,16,1024,1,1]) -> bf16[1,576,16,1024,1,1] {
//   %param_0.251 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} parameter(0)
//   %param_1.300 = bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} parameter(1)
//   %slice.1126 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} slice(bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %param_1.300), slice={[1:2],
//   [0:576], [0:16], [0:1024], [0:1], [0:1]} ROOT %add.3123 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)}
//   add(bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %param_0.251, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %slice.1126)
// }
//
// %fused_computation.118 (param_0.250: bf16[1,576,16,1024,1,1], param_1.298: bf16[2,576,16,1024,1,1]) -> bf16[1,576,16,1024,1,1] {
//   %param_0.250 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} parameter(0)
//   %param_1.298 = bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} parameter(1)
//   %slice.1125 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} slice(bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %param_1.298), slice={[1:2],
//   [0:576], [0:16], [0:1024], [0:1], [0:1]} ROOT %add.3122 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)}
//   add(bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %param_0.250, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %slice.1125)
// }
//         )",
//           false);
// }

// INVALID_ARGUMENT: layout minor_to_major field contains 6 elements, but shape is rank 7: {5, 4, 3, 2, 1, 0}; shape: element_type: BF16 dimensions: 8
// dimensions: 2048 dimensions: 1 dimensions: 36 dimensions: 256 dimensions: 1 dimensions: 1 layout { minor_to_major: 4 minor_to_major: 1 minor_to_major: 6
// minor_to_major: 5 minor_to_major: 3 minor_to_major: 2 minor_to_major: 0 tiles { dimensions: 8 dimensions: 128 } tiles { dimensions: 2 dimensions: 1 }
// tail_padding_alignment_in_elements: 1 } is_dynamic_dimension: false is_dynamic_dimension: false is_dynamic_dimension: false is_dynamic_dimension: false
// is_dynamic_dimension: false is_dynamic_dimension: false is_dynamic_dimension: false
//  XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, BalanceChainedCollectivePermutesLoopedEinsum3) {
//    RunTest(R"(
//          HloModule module, is_scheduled=false
//
//  %fused_computation.1799 (param_0.4926: s32[16], param_1.5709: u32[], param_2.3976: u32[], param_3.3386: u32[], param_4.2299: u32[]) -> (s32[1], s32[1],
//  s32[1], s32[1]) {
//    %param_0.4926 = s32[16]{0:T(128)} parameter(0)
//    %param_1.5709 = u32[]{:T(128)} parameter(1)
//    %dynamic-slice.1611 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4926, u32[]{:T(128)} %param_1.5709), dynamic_slice_sizes={1}
//    %param_2.3976 = u32[]{:T(128)} parameter(2)
//    %dynamic-slice.1612 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4926, u32[]{:T(128)} %param_2.3976), dynamic_slice_sizes={1}
//    %param_3.3386 = u32[]{:T(128)} parameter(3)
//    %dynamic-slice.1613 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4926, u32[]{:T(128)} %param_3.3386), dynamic_slice_sizes={1}
//    %param_4.2299 = u32[]{:T(128)} parameter(4)
//    %dynamic-slice.1614 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4926, u32[]{:T(128)} %param_4.2299), dynamic_slice_sizes={1}
//    ROOT %tuple.1346 = (s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %dynamic-slice.1611, s32[1]{0:T(128)}
//    %dynamic-slice.1612, s32[1]{0:T(128)} %dynamic-slice.1613, s32[1]{0:T(128)} %dynamic-slice.1614)
//  }
//
//  %fused_computation.243 (param_0.505: bf16[8,2048,2,576,1,1], param_1.586: bf16[8,2048,2,576,1,1]) -> bf16[8,2048,4,576,1,1] {
//    %param_1.586 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} parameter(1)
//    %constant.5838 = bf16[]{:T(256)} constant(-inf)
//    %pad.368 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} pad(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %param_1.586, bf16[]{:T(256)}
//    %constant.5838), padding=0_0x0_0x0_2x0_0x0_0x0_0 %param_0.505 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} parameter(0) %pad.367 =
//    bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} pad(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %param_0.505, bf16[]{:T(256)} %constant.5838),
//    padding=0_0x0_0x2_0x0_0x0_0x0_0 ROOT %maximum.528 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}
//    maximum(bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %pad.368, bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %pad.367)
//  }
//
//  %fused_computation.244 (param_0.507: bf16[8,2048,2,576,1,1], param_1.585: bf16[8,2048,2,576,1,1]) -> bf16[8,2048,4,576,1,1] {
//    %param_1.585 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} parameter(1)
//    %constant.5832 = bf16[]{:T(256)} constant(-inf)
//    %pad.370 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} pad(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %param_1.585, bf16[]{:T(256)}
//    %constant.5832), padding=0_0x0_0x0_2x0_0x0_0x0_0 %param_0.507 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} parameter(0) %pad.369 =
//    bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} pad(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %param_0.507, bf16[]{:T(256)} %constant.5832),
//    padding=0_0x0_0x2_0x0_0x0_0x0_0 ROOT %maximum.529 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}
//    maximum(bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %pad.370, bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %pad.369)
//  }
//
//  %fused_computation.247 (param_0.511: bf16[8,2048,2,2,576,1,1]) -> bf16[8,2048,2,2,576,1,1] {
//    %param_0.511 = bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} parameter(0)
//    ROOT %copy.2292 = bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} copy(bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} %param_0.511)
//  }
//
//  %fused_computation.248.clone (param_0.526: s32[], param_1.589: bf16[1,32,576,1,36,256,1], param_2.400: s32[]) -> bf16[2,2,576,1,36,256,1] {
//    %param_1.589 = bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} parameter(1)
//    %constant.5843 = bf16[]{:T(256)} constant(-inf)
//    %pad.378 = bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} pad(bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_1.589, bf16[]{:T(256)}
//    %constant.5843), padding=0_1x0_0x0_0x0_0x0_0x0_0x0_0 %constant.5853 = s32[]{:T(128)} constant(0) %param_0.526 = s32[]{:T(128)} parameter(0)
//    %dynamic-slice.1382 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} dynamic-slice(bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %pad.378,
//    s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %param_0.526, s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %constant.5853, /*index=5*/s32[]{:T(128)}
//    %constant.5853, s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %constant.5853), dynamic_slice_sizes={2,2,576,1,36,256,1} %pad.377 =
//    bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} pad(bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_1.589, bf16[]{:T(256)}
//    %constant.5843), padding=1_0x0_0x0_0x0_0x0_0x0_0x0_0 %param_2.400 = s32[]{:T(128)} parameter(2) %dynamic-slice.1381 =
//    bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} dynamic-slice(bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %pad.377, s32[]{:T(128)}
//    %constant.5853, s32[]{:T(128)} %param_2.400, s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %constant.5853, /*index=5*/s32[]{:T(128)} %constant.5853,
//    s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %constant.5853), dynamic_slice_sizes={2,2,576,1,36,256,1} ROOT %maximum.532 =
//    bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} maximum(bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %dynamic-slice.1382,
//    bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %dynamic-slice.1381)
//  }
//
//  %fused_computation.246 (param_0.521: bf16[8,2048,2,2,576,1,1], param_1.588: s32[], param_2.399: bf16[1,32,576,1,36,256,1], param_3.247: s32[]) ->
//  bf16[8,2048,1,36,256,1,1] {
//    %param_0.521 = bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} parameter(0)
//    %fusion.268 = bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} fusion(bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} %param_0.521),
//    kind=kLoop, calls=%fused_computation.247 %param_1.588 = s32[]{:T(128)} parameter(1) %param_2.399 = bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)}
//    parameter(2) %param_3.247 = s32[]{:T(128)} parameter(3) %fusion.271 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} fusion(s32[]{:T(128)}
//    %param_1.588, bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_2.399, s32[]{:T(128)} %param_3.247), kind=kLoop,
//    calls=%fused_computation.248.clone ROOT %convolution.172 = bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)}
//    convolution(bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} %fusion.268, bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %fusion.271),
//    window={size=1x1x36x2x2 pad=0_0x0_0x35_35x0_0x0_0 rhs_reversal=0x1x1x0x0}, dim_labels=0b43f12_43i12o0->0b12f34
//  }
//
//  %fused_computation.245 (param_0.508: bf16[8,2048,2,2,576,1,1]) -> bf16[8,2048,2,2,576,1,1] {
//    %param_0.508 = bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} parameter(0)
//    ROOT %copy.2290 = bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} copy(bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} %param_0.508)
//  }
//
//  %fused_computation.249.clone (param_0.525: s32[], param_1.587: bf16[1,32,576,1,36,256,1], param_2.398: s32[]) -> bf16[2,2,576,1,36,256,1] {
//    %param_1.587 = bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} parameter(1)
//    %constant.5837 = bf16[]{:T(256)} constant(-inf)
//    %pad.382 = bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} pad(bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_1.587, bf16[]{:T(256)}
//    %constant.5837), padding=0_1x0_0x0_0x0_0x0_0x0_0x0_0 %constant.5848 = s32[]{:T(128)} constant(0) %param_0.525 = s32[]{:T(128)} parameter(0)
//    %dynamic-slice.1386 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} dynamic-slice(bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %pad.382,
//    s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %param_0.525, s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %constant.5848, /*index=5*/s32[]{:T(128)}
//    %constant.5848, s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %constant.5848), dynamic_slice_sizes={2,2,576,1,36,256,1} %pad.381 =
//    bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} pad(bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_1.587, bf16[]{:T(256)}
//    %constant.5837), padding=1_0x0_0x0_0x0_0x0_0x0_0x0_0 %param_2.398 = s32[]{:T(128)} parameter(2) %dynamic-slice.1385 =
//    bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} dynamic-slice(bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %pad.381, s32[]{:T(128)}
//    %constant.5848, s32[]{:T(128)} %param_2.398, s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %constant.5848, /*index=5*/s32[]{:T(128)} %constant.5848,
//    s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %constant.5848), dynamic_slice_sizes={2,2,576,1,36,256,1} ROOT %maximum.533 =
//    bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} maximum(bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %dynamic-slice.1386,
//    bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %dynamic-slice.1385)
//  }
//
//  %fused_computation.241 (param_0.503: bf16[8,2048,1,36,256,1], param_1.561: bf16[8,2048,1,36,256,1,1], param_2.397: bf16[8,2048,2,2,576,1,1], param_3.246:
//  s32[], param_4.127: bf16[1,32,576,1,36,256,1], param_5.55: s32[]) -> bf16[8,2048,1,36,256,1] {
//    %param_0.503 = bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} parameter(0)
//    %param_1.561 = bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)} parameter(1)
//    %bitcast.599 = bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} bitcast(bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)} %param_1.561)
//    %add.3146 = bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} add(bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %param_0.503,
//    bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %bitcast.599) %param_2.397 = bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} parameter(2)
//    %fusion.266 = bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} fusion(bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} %param_2.397),
//    kind=kLoop, calls=%fused_computation.245 %param_3.246 = s32[]{:T(128)} parameter(3) %param_4.127 = bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)}
//    parameter(4) %param_5.55 = s32[]{:T(128)} parameter(5) %fusion.272 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} fusion(s32[]{:T(128)}
//    %param_3.246, bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_4.127, s32[]{:T(128)} %param_5.55), kind=kLoop,
//    calls=%fused_computation.249.clone %convolution.171 = bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)}
//    convolution(bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} %fusion.266, bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %fusion.272),
//    window={size=1x1x36x2x2 pad=0_0x0_0x35_35x0_0x0_0 rhs_reversal=0x1x1x0x0}, dim_labels=0b43f12_43i12o0->0b12f34 %bitcast.596 =
//    bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} bitcast(bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)} %convolution.171) ROOT %add.3143 =
//    bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} add(bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %add.3146,
//    bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %bitcast.596)
//  }
//          )",
//            false);
//  }

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, MoveCentainConv2) {
  RunTest(R"(
        HloModule module, is_scheduled=false

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,64]{2,1,0} parameter(3)
  cp0 = f32[16,64,256]{2,1,0} collective-permute(p0),
    source_target_pairs={{0,1},{1,0}}
  cp1 = f32[16,64,256]{2,1,0} collective-permute(p1),
    source_target_pairs={{0,1},{1,0}}
  cp2 = f32[16,64,256]{2,1,0} collective-permute(cp0),
    source_target_pairs={{0,1},{1,0}}
  cp3 = f32[16,64,256]{2,1,0} collective-permute(cp1),
    source_target_pairs={{0,1},{1,0}}
  a0 = f32[16,64,256]{2,1,0} add(cp0, cp1)
  c0 = f32[16,64,256]{2,1,0} convolution(p2, p3),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(a0, c0),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, f32[16,256,256]{2,1,0}) tuple(cp2, cp3, c1)
}
        )",
          false);
}

// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, WhileOverlapLimit) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// while_cond {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   gte0 = bf16[8]{0} get-tuple-element(param), index=0
//   gte1 = pred[] get-tuple-element(param), index=2
//   bitcast = bf16[8]{0} bitcast(gte0)
//   collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1}, {1,0}}
//   add0 = bf16[8]{0} add(collective-permute.1, bitcast)
//   negate = bf16[8]{0} negate(add0)
//   collective-permute.2 = bf16[8]{0} collective-permute(collective-permute.1), source_target_pairs={{1,0}}
//   ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte1)
// }
//
// ENTRY entry {
//   p0 = bf16[8]{0} parameter(0)
//   p1 = bf16[8]{0} parameter(1)
//   p2 = pred[] parameter(2)
//   tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
//   while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
//   collective-permute.3 = bf16[8]{0} collective-permute(p1), source_target_pairs={{0,1}, {1,0}}
//   gte0 = bf16[8]{0} get-tuple-element(while), index=0
//   gte1 = bf16[8]{0} get-tuple-element(while), index=1
//   add = bf16[8]{0} add(gte0, gte1)
//   ROOT add2 = bf16[8]{0} add(add, collective-permute.3)
// }
//         )",
//           false);
// }

// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, WhileNestedOverlapLimit) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// while_cond {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   gte0 = bf16[8]{0} get-tuple-element(param), index=0
//   gte1 = pred[] get-tuple-element(param), index=2
//   bitcast = bf16[8]{0} bitcast(gte0)
//   collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1}, {1,0}}
//   add0 = bf16[8]{0} add(collective-permute.1, bitcast)
//   negate = bf16[8]{0} negate(add0)
//   ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.1, negate, gte1)
// }
//
// while_cond2 {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body2 {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   while.1 = (bf16[8]{0}, bf16[8]{0}, pred[]) while(param), condition=while_cond, body=while_body
//   gte0 = bf16[8]{0} get-tuple-element(while.1), index=0
//   gte1 = pred[] get-tuple-element(while.1), index=2
//   bitcast = bf16[8]{0} bitcast(gte0)
//   negate = bf16[8]{0} negate(bitcast)
//   collective-permute.2 = bf16[8]{0} collective-permute(negate), source_target_pairs={{1,0}}
//   ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte1)
// }
//
// ENTRY entry {
//   p0 = bf16[8]{0} parameter(0)
//   p1 = bf16[8]{0} parameter(1)
//   p2 = pred[] parameter(2)
//   tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
//   while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond2, body=while_body2
//   collective-permute.3 = bf16[8]{0} collective-permute(p1), source_target_pairs={{0,1}, {1,0}}
//   gte0 = bf16[8]{0} get-tuple-element(while), index=0
//   gte1 = bf16[8]{0} get-tuple-element(while), index=1
//   add = bf16[8]{0} add(gte0, gte1)
//   ROOT add2 = bf16[8]{0} add(add, collective-permute.3)
// }
//         )",
//           false);
// }

// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, WhileOverlapUnderLimit) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// while_cond {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body {
//   param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
//   gte0 = bf16[8]{0} get-tuple-element(param), index=0
//   gte1 = pred[] get-tuple-element(param), index=2
//   bitcast = bf16[8]{0} bitcast(gte0)
//   collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1}, {1,0}}
//   add0 = bf16[8]{0} add(collective-permute.1, bitcast)
//   negate = bf16[8]{0} negate(add0)
//   collective-permute.2 = bf16[8]{0} collective-permute(collective-permute.1), source_target_pairs={{1,0}, {1,0}}
//   ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte1)
// }
//
// ENTRY entry {
//   p0 = bf16[8]{0} parameter(0)
//   p1 = bf16[8]{0} parameter(1)
//   p2 = pred[] parameter(2)
//   tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
//   while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
//   collective-permute.3 = bf16[8]{0} collective-permute(p1), source_target_pairs={{0,1}, {1,0}}
//   gte0 = bf16[8]{0} get-tuple-element(while), index=0
//   gte1 = bf16[8]{0} get-tuple-element(while), index=1
//   add = bf16[8]{0} add(gte0, gte1)
//   ROOT add2 = bf16[8]{0} add(add, collective-permute.3)
// }
//         )",
//           false);
// }

// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, WhileOverlapLimitAllGather) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// while_cond {
//   param = (bf16[4]{0}, bf16[8]{0}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body {
//   param = (bf16[4]{0}, bf16[8]{0}, pred[]) parameter(0)
//   gte0 = bf16[4]{0} get-tuple-element(param), index=0
//   gte1 = bf16[8]{0} get-tuple-element(param), index=1
//   gte2 = pred[] get-tuple-element(param), index=2
//   bitcast = bf16[8]{0} bitcast(gte0)
//   all-gather.1 = bf16[8]{0} all-gather(gte0), replica_groups={{0,1}, {1,0}}, dimensions={0}, channel_id=1
//   add0 = bf16[8]{0} add(all-gather.1, bitcast)
//   negate = bf16[8]{0} negate(add0)
//   collective-permute.2 = bf16[4]{0} collective-permute(gte0), source_target_pairs={{0,1}, {1,0}}
//   ROOT tuple = (bf16[4]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte2)
// }
//
// ENTRY entry {
//   p0 = bf16[4]{0} parameter(0)
//   p1 = bf16[8]{0} parameter(1)
//   p2 = pred[] parameter(2)
//   tuple = (bf16[4]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
//   while = (bf16[4]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
//   all-gather.2 = bf16[8]{0} all-gather(p0), replica_groups={{0,1}, {1,0}}, dimensions={0}, channel_id=2
//   gte0 = bf16[4]{0} get-tuple-element(while), index=0
//   gte1 = bf16[8]{0} get-tuple-element(while), index=1
//   ROOT tuple.2 = (bf16[4]{0}, bf16[8]{0}, bf16[8]{0}) tuple(gte0, gte1, all-gather.2)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, WhileOverlapUnderLimitAllGather) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// while_cond {
//   param = (bf16[4]{0}, bf16[8]{0}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body {
//   param = (bf16[4]{0}, bf16[8]{0}, pred[]) parameter(0)
//   gte0 = bf16[4]{0} get-tuple-element(param), index=0
//   gte1 = bf16[8]{0} get-tuple-element(param), index=1
//   gte2 = pred[] get-tuple-element(param), index=2
//   bitcast = bf16[8]{0} bitcast(gte0)
//   all-gather.1 = bf16[8]{0} all-gather(gte0), replica_groups={{0,1}, {1,0}}, dimensions={0}, channel_id=1
//   add0 = bf16[8]{0} add(all-gather.1, bitcast)
//   negate = bf16[8]{0} negate(add0)
//   collective-permute.2 = bf16[4]{0} collective-permute(gte0), source_target_pairs={{1,0}, {0,1}}
//   ROOT tuple = (bf16[4]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte2)
// }
//
// ENTRY entry {
//   p0 = bf16[4]{0} parameter(0)
//   p1 = bf16[8]{0} parameter(1)
//   p2 = pred[] parameter(2)
//   tuple = (bf16[4]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
//   while = (bf16[4]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
//   all-gather.2 = bf16[8]{0} all-gather(p0), replica_groups={{0,1}, {1,0}}, dimensions={0}, channel_id=2
//   gte0 = bf16[4]{0} get-tuple-element(while), index=0
//   gte1 = bf16[8]{0} get-tuple-element(while), index=1
//   ROOT tuple.2 = (bf16[4]{0}, bf16[8]{0}, bf16[8]{0}) tuple(gte0, gte1, all-gather.2)
// }
//         )",
//           false);
// }

// hangs
// 2025-06-03 16:35:20.227679: W xla/hlo/transforms/simplifiers/hlo_rematerialization.cc:3021] Can't reduce memory use below -85.40GiB (-91702322813 bytes) by
// rematerialization; only reduced to 112.00GiB (120263278632 bytes), down from 112.00GiB (120263278632 bytes) originally
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllToAllAsyncBalance) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// async_computation {
//   p = f32[2,8,256,256] parameter(0)
//   ROOT ata = f32[2,8,256,256] all-to-all(p), dimensions={0}, replica_groups={{0,1}}
// }
//
// async_computation.2 {
//   p.2 = f32[2,8,256,256] parameter(0)
//   ROOT ata.1 = f32[2,8,256,256] all-to-all(p.2), dimensions={0}, replica_groups={{0,1}}
// }
//
//
// ENTRY %module {
//   %constant.19 = u32[] constant(0)
//   %replica_id = u32[]{:T(128)} replica-id()
//   %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
//   %color_operand.1 = f32[2,8,256,256]{3,2,1,0} broadcast(
//     f32[]{:T(128)} %convert), dimensions={}
//   %color_operand.2 = f32[2,8,256,256]{3,2,1,0} broadcast(
//     f32[]{:T(128)} %convert), dimensions={}
//   %ata-start = ((f32[2,8,256,256]), f32[2,8,256,256], u32[], u32[]) async-start(
//     f32[2,8,256,256] %color_operand.1), calls=async_computation,
//     metadata={op_type="AllToAll" op_name="ata0"}
//   %ata-start.2 = ((f32[2,8,256,256]), f32[2,8,256,256], u32[], u32[]) async-start(
//     f32[2,8,256,256] %color_operand.2), calls=async_computation.2,
//     metadata={op_type="AllToAll" op_name="ata1"}
//   %ata-done = f32[2,8,256,256] async-done(%ata-start), calls=async_computation,
//     metadata={op_type="AllToAll" op_name="ata0"}
//   %ata-done-bc = f32[16,256,256] bitcast(f32[2,8,256,256] %ata-done),
//     metadata={op_type="Bitcast" op_name="ata0"}
//   %ata-done.2 = f32[2,8,256,256] async-done(%ata-start.2), calls=async_computation.2,
//     metadata={op_type="AllToAll" op_name="ata1"}
//   %ata-done-bc.2 = f32[16,256,256] bitcast(f32[2,8,256,256] %ata-done.2),
//     metadata={op_type="Bitcast" op_name="ata1"}
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[16,64,256]{2,1,0} parameter(1)
//   p2 = f32[16,256,256]{2,1,0} parameter(2)
//   p3 = f32[16,256,256]{2,1,0} parameter(3)
//   c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//     metadata={op_type="AllToAll" op_name="c0"}
//   c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//     metadata={op_type="AllToAll" op_name="c1"}
//   a2 = f32[16,256,256]{2,1,0} add(c1, c0)
//   ROOT t = (f32[16,256,256], f32[16,256,256], f32[16,256,256]) tuple(a2, %ata-done-bc.2, %ata-done-bc)
// }
//         )",
//           false);
// }

// hangs
// maybe we shouldn't change collective-permute-start params
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, ReleaseOneThatStallsLessFirst) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[16,64,256]{2,1,0} parameter(1)
//   p2 = f32[1024,2048,2048]{2,1,0} parameter(2)
//   p3 = f32[2048,2048,2048]{2,1,0} parameter(3)
//   cp1s = (f32[1024,2048,2048]{2,1,0}, f32[1024,2048,2048]{2,1,0}) collective-permute-start(p2), source_target_pairs={{1,0}, {0,1}}
//   cp2s = (f32[2048,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}) collective-permute-start(p3), source_target_pairs={{1,0}, {0,1}}
//   c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//     metadata={op_type="AllToAll" op_name="c0"}
//   cp1d = f32[1024,2048,2048]{2,1,0} collective-permute-done(cp1s)
//   cp2d = f32[2048,2048,2048]{2,1,0} collective-permute-done(cp2s)
//   ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1024,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}) tuple(c0, cp1d, cp2d)
// }
//         )",
//           false);
// }

// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, ReleaseStartWhenLatencyDue) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   p2 = f32[512,2048,2048]{2,1,0} parameter(2)
//   cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}) collective-permute-start(p2), source_target_pairs={{1,0}}
//   cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
//   cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) collective-permute-start(p1), source_target_pairs={{1,0}}
//   cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s)
//   cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) collective-permute-start(cp2d), source_target_pairs={{1,0}}
//   cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s)
//   slice = f32[16,64,256]{2,1,0} slice(f32[512,2048,2048]{2,1,0} cp1d), slice={[0:16], [0:64], [0:256]}
//   c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   c1 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
// }
//         )",
//           false);
// }

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, DepthPressureReduction) {
  RunTest(R"(
      HloModule serial_collective_permute_test, is_scheduled=false
  ENTRY after_optimizations_test {
  %parameter.1 = bf16[8]{0} parameter(0)
  %parameter.2 = bf16[8]{0} parameter(1)
  %parameter.3 = bf16[8]{0} parameter(2)
  %parameter.4 = bf16[8]{0} parameter(3)
  %collective-permute.2 = bf16[8]{0} collective-permute(parameter.1), source_target_pairs={{0,1}, {1,0}}
  %a = bf16[8]{0} add(collective-permute.2, parameter.2)
  %b = bf16[8]{0} add(a, parameter.3)
  %c = bf16[8]{0} add(b, parameter.4)
  %d = bf16[8]{0} add(c, parameter.4)
  %c1 = bf16[8]{0} copy(d)
  %e = bf16[8]{0} add(d, parameter.3)
  %c0 = bf16[8]{0} copy(e)
  %f = bf16[8]{0} add(e, parameter.2)
  %h = bf16[8]{0} add(c0, b)
  %g = bf16[8]{0} add(c1, c)
  %i = bf16[8]{0} add(f, a)
  ROOT %t = (bf16[8]{0}, bf16[8]{0}, bf16[8]{0}, bf16[8]{0}) tuple(f, g, h, i)
}
      )",
          false);
}

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, RerunWithSmallerMemoryLimit) {
  RunTest(R"(
      HloModule rerun_scheduler_test, is_scheduled=false
  ENTRY main {
   p0 = bf16[8]{0} parameter(0)
   c = bf16[] constant(0)
   b = bf16[43]{0} broadcast(c), dimensions={}
   s = bf16[1]{0} slice(b), slice={[0:1]}
   cp = bf16[8]{0} collective-permute(p0), source_target_pairs={{0,1}, {1,0}}
  ROOT tuple = (bf16[8]{0}, bf16[1]{0}) tuple(cp, s)
}
      )",
          false);
}

// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, MultipleAsyncDoneOperationsDoNotCreateLoop) {
//   RunTest(R"(
//         HloModule multiple_async_done_scheduler_test, is_scheduled=false
//
// called_computation {
//   ROOT %param = s32[<=4096]{0:T(8)M(1024)} parameter(0)
// }
//
// ENTRY main {
//   %while_body_forward_pass_input_tuple = (s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}) parameter(0),
//   backend_config={"compute_type":"COMPUTE_TYPE_SCALAR"}
//
//   %get-tuple-element.0 = s32[<=4096]{0:T(8)M(1024)} get-tuple-element(
//       (s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}) %while_body_forward_pass_input_tuple),
//       index=0, backend_config={"flag_configs":[],"compute_type":"COMPUTE_TYPE_SCALAR"}
//
//   %get-tuple-element.1 = s32[<=4096]{0:T(8)M(1024)} get-tuple-element(
//       (s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}) %while_body_forward_pass_input_tuple),
//       index=1, backend_config={"flag_configs":[],"compute_type":"COMPUTE_TYPE_SCALAR"}
//
//   %call-start.1 = ((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)})
//     call-start(s32[<=4096]{0:T(8)M(1024)} %get-tuple-element.1),
//       async_execution_thread="sparsecore", to_apply=%called_computation
//
//   %call-done.1 = s32[<=4096]{0:T(8)M(1024)}
//     call-done(((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)}) %call-start.1)
//
//   %call-start.2 = ((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)})
//     call-start(s32[<=4096]{0:T(8)M(1024)} %call-done.1),
//       async_execution_thread="sparsecore", to_apply=%called_computation
//
//   %call-done.2 = s32[<=4096]{0:T(8)M(1024)}
//     call-done(((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)}) %call-start.2)
//
//   %call-start.3 = ((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)})
//     call-start(s32[<=4096]{0:T(8)M(1024)} %get-tuple-element.0),
//       async_execution_thread="sparsecore", to_apply=%called_computation
//
//   %call-done.3 = s32[<=4096]{0:T(8)M(1024)}
//     call-done(((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)}) %call-start.3)
//
//   ROOT %tuple.6 = (s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)})
//     tuple(s32[<=4096]{0:T(8)M(1024)} %call-done.2, s32[<=4096]{0:T(8)M(1024)} %call-done.3),
//       backend_config={"flag_configs":[],"compute_type":"COMPUTE_TYPE_SCALAR"}
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, CopyScheduling) {
//   RunTest(R"(
//         HloModule EinsumTest, is_scheduled=false
// ENTRY AddR2 {
//   y_host = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(1)
//   z = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(2)
//   x = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(0)
//   convolution = bf16[12800,12800]{1,0:T(8,128)(2,1)} convolution(x, z), dim_labels=bf_io->bf
//   copy-start = (bf16[12800,12800]{1,0:T(8,128)(2,1)}, bf16[12800,12800]{1,0:T(8,128)(2,1)}, u32[]{:S(2)}) copy-start(y_host)
//   copy-done = bf16[12800,12800]{1,0:T(8,128)(2,1)} copy-done(copy-start)
//   ROOT convolution.1 = bf16[12800,12800]{1,0:T(8,128)(2,1)} convolution(convolution, copy-done), dim_labels=bf_io->bf
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, MaxCopyScheduling) {
//   RunTest(R"(
//         HloModule EinsumTest, is_scheduled=false
// ENTRY AddR2 {
//   y_host = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(1)
//   q_host = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(3)
//   z = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(2)
//   x = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(0)
//   convolution = bf16[12800,12800]{1,0:T(8,128)(2,1)} convolution(x, z), dim_labels=bf_io->bf
//   copy-start = (bf16[12800,12800]{1,0:T(8,128)(2,1)}, bf16[12800,12800]{1,0:T(8,128)(2,1)}, u32[]{:S(2)}) copy-start(y_host)
//   copy-done = bf16[12800,12800]{1,0:T(8,128)(2,1)} copy-done(copy-start)
//   copy-start2 = (bf16[12800,12800]{1,0:T(8,128)(2,1)}, bf16[12800,12800]{1,0:T(8,128)(2,1)}, u32[]{:S(2)}) copy-start(q_host)
//   copy-done2 = bf16[12800,12800]{1,0:T(8,128)(2,1)} copy-done(copy-start2)
//   ROOT t = (bf16[12800,12800]{1,0:T(8,128)(2,1)}, bf16[12800,12800]{1,0:T(8,128)(2,1)})  tuple(copy-done2, copy-done)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, ScheduleLoopPeeledSendDoneBeforeWhile) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// while_cond {
//   param = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body {
//   param = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, pred[]) parameter(0)
//   gte0 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} get-tuple-element(param), index=0
//   gte1 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} get-tuple-element(param), index=1
//   add.0 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} add(gte0, gte1)
//   gte2 = pred[] get-tuple-element(param), index=2
//   ROOT tuple = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, pred[]) tuple(add.0, gte1, gte2)
// }
//
// ENTRY %entry {
//   p0 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} parameter(0)
//   p1 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} parameter(1)
//   after-all = token[] after-all()
//   send = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, u32[], token[]) send(p0, after-all), channel_id=1246
//   recv = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, u32[], token[]) recv(after-all), channel_id=1247
//   recv-done = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, token[]) recv-done(recv), channel_id=1247
//   get-tuple-element = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} get-tuple-element(recv-done), index=0
//   send-done = token[] send-done(send), channel_id=1246, control-predecessors={recv-done}
//   p2 = pred[] parameter(2)
//   tuple = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, pred[]) tuple(get-tuple-element, p1, p2)
//   while = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, pred[]) while(tuple), condition=while_cond,
//   body=while_body ROOT gte0 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} get-tuple-element(while), index=0
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllGatherWithSelectiveOverlap) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY %module {
//   %constant.19 = u32[] constant(0)
//   %replica_id = u32[]{:T(128)} replica-id()
//   %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
//   %color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(
//     f32[]{:T(128)} %convert), dimensions={}
//   %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(
//     f32[8,256,256] %color_operand.1), replica_groups={{0,1}, {1,0}}, dimensions={0},
//     metadata={op_type="AllGather" op_name="ag0"}
//   %ag-done = f32[16,256,256] all-gather-done(
//     (f32[8,256,256], f32[16,256,256]) %ag-start),
//     metadata={op_type="AllGather" op_name="ag0"}
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[16,64,256]{2,1,0} parameter(1)
//   p2 = f32[16,256,256]{2,1,0} parameter(2)
//   p3 = f32[16,256,256]{2,1,0} parameter(3)
//   c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   c2 = f32[16,256,256]{2,1,0} convolution(p0, p1),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   ROOT a2 = f32[16,256,256]{2,1,0} add(%ag-done, c0)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AnnotationFirstDataIndependentConv) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   p2 = f32[512,2048,2048]{2,1,0} parameter(2)
//   cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0}}
//   cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
//   cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s),
//   frontend_attributes={_scheduling_group_id="0"} cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d),
//   source_target_pairs={{1,0}} cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s) slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16],
//   [0:64], [0:256]} c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
//   c1 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AnnotationSecondDataIndependentConv) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   p2 = f32[512,2048,2048]{2,1,0} parameter(2)
//   cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0}}
//   cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
//   cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s),
//   frontend_attributes={_scheduling_group_id="0"} cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d),
//   source_target_pairs={{1,0}} cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s) slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16],
//   [0:64], [0:256]} c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   c1 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
//   ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AnnotationBothDataIndependentConvs) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   p2 = f32[512,2048,2048]{2,1,0} parameter(2)
//   cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0}}
//   cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
//   cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s),
//   frontend_attributes={_scheduling_group_id="0"} cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d),
//   source_target_pairs={{1,0}} cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s) slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16],
//   [0:64], [0:256]} c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
//   c1 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
//   ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AnnotationFirstDataDependentConv) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   p2 = f32[512,2048,2048]{2,1,0} parameter(2)
//   cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0}}
//   cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
//   cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s),
//   frontend_attributes={_scheduling_group_id="0"} cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d),
//   source_target_pairs={{1,0}} cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s) slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16],
//   [0:64], [0:256]} c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
//   c1 = f32[1,256,256]{2,1,0} convolution(c0, c0),
//     window={size=16 stride=15}, dim_labels=0fb_0io->0fb
//   ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AnnotationSecondDataDependentConv) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   p2 = f32[512,2048,2048]{2,1,0} parameter(2)
//   cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0}}
//   cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
//   cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s),
//   frontend_attributes={_scheduling_group_id="0"} cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d),
//   source_target_pairs={{1,0}} cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s) slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16],
//   [0:64], [0:256]} c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   c1 = f32[1,256,256]{2,1,0} convolution(c0, c0),
//     window={size=16 stride=15}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
//   ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AnnotationBothDataDependentConvs) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   p2 = f32[512,2048,2048]{2,1,0} parameter(2)
//   cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0}}
//   cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
//   cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s),
//   frontend_attributes={_scheduling_group_id="0"} cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d),
//   source_target_pairs={{1,0}} cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s) slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16],
//   [0:64], [0:256]} c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
//   c1 = f32[1,256,256]{2,1,0} convolution(c0, c0),
//     window={size=16 stride=15}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
//   ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AnnotationWithTwoAsyncOps) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   p2 = f32[512,2048,2048]{2,1,0} parameter(2)
//   cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s),
//   frontend_attributes={_scheduling_group_id="0"} cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1),
//   source_target_pairs={{1,0}}, frontend_attributes={_scheduling_group_id="0"} cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s),
//   frontend_attributes={_scheduling_group_id="0"} cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d),
//   source_target_pairs={{1,0}} cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s) c0 = f32[16,256,256]{2,1,0} convolution(p0, p0),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
//   ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[512,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, cp1d, cp3d)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, SchedulingAnnotationMakesAnotherGroupReady) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// fused_computation {
//   param0 = f32[16,64,256]{2,1,0} parameter(0)
//   param1 = f32[16,64,256]{2,1,0} parameter(1)
//   ROOT c0 = f32[16,256,256]{2,1,0} convolution(param0, param1), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//   frontend_attributes={_scheduling_group_id="0"}
// }
//
// fused_computation.1 {
//   param0.1 = f32[16,256,256]{2,1,0} parameter(0)
//   param1.1 = f32[16,256,256]{2,1,0} parameter(1)
//   ROOT c1 = f32[1,256,256]{2,1,0} convolution(param0.1, param1.1), window={size=16 stride=15}, dim_labels=0fb_0io->0fb,
//   frontend_attributes={_scheduling_group_id="1"}
// }
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   cp0s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cp0d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp0s),
//   frontend_attributes={_scheduling_group_id="0"} cp1s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp0d),
//   source_target_pairs={{1,0}}, frontend_attributes={_scheduling_group_id="1"} cp1d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp1s),
//   frontend_attributes={_scheduling_group_id="1"} f0 = f32[16,256,256]{2,1,0} fusion(p0, p0), kind=kOutput, calls=fused_computation,
//   frontend_attributes={_scheduling_group_id="0"} f1 = f32[1,256,256]{2,1,0} fusion(f0, f0), kind=kOutput, calls=fused_computation.1,
//   frontend_attributes={_scheduling_group_id="1"} ROOT tuple = (f32[128,2048,2048]{2,1,0}, f32[1,256,256]{2,1,0}) tuple(cp1d, f1)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AnnotatedRoot) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// fused_computation {
//   param0 = f32[16,64,256]{2,1,0} parameter(0)
//   param1 = f32[16,64,256]{2,1,0} parameter(1)
//   ROOT c0 = f32[16,256,256]{2,1,0} convolution(param0, param1), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//   frontend_attributes={_scheduling_group_id="0"}
// }
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   cp0s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cp0d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp0s),
//   frontend_attributes={_scheduling_group_id="0"} ROOT f0 = f32[16,256,256]{2,1,0} fusion(p0, p0), kind=kOutput, calls=fused_computation,
//   frontend_attributes={_scheduling_group_id="0"}
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AnnotatedNoOp) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// fused_computation {
//   param0 = f32[128,2048]{1,0} parameter(0)
//   param1 = f32[8,2048]{1,0} parameter(1)
//   constant0 = s32[] constant(0)
//   dynamic-update-slice = f32[128,2048]{1,0} dynamic-update-slice(param0, param1, constant0, constant0)
//   ROOT tuple = (f32[128,2048]{1,0}, f32[128,2048]{1,0}) tuple(dynamic-update-slice, param0)
// }
//
// ENTRY entry {
//   p0 = f32[128,2048]{1,0} parameter(0)
//   p1 = f32[8,2048]{1,0} parameter(1)
//   p2 = f32[128,2048]{1,0} parameter(2)
//   cps = (f32[128,2048]{1,0}, f32[128,2048]{1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cpd = f32[128,2048]{1,0} collective-permute-done(cps), frontend_attributes={_scheduling_group_id="0"} fusion
//   = (f32[128,2048]{1,0}, f32[128,2048]{1,0}) fusion(p0, p1), kind=kLoop, calls=fused_computation, frontend_attributes={_scheduling_group_id="0"} gte =
//   f32[128,2048]{1,0} get-tuple-element(fusion), index=0, frontend_attributes={_scheduling_group_id="0"} ROOT add = f32[128,2048]{1,0} add(gte, cpd)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, OutOfOrderStartAndDone) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// while_condition {
//   tuple = ((f32[16,16], u32[], token[]), f32[16,16], u32[]) parameter(0)
//   i = get-tuple-element(tuple), index=2
//   n = u32[] constant(2)
//   ROOT predicate = pred[] compare(i, n), direction=LT
// }
//
// while_body {
//   tuple = ((f32[16,16], u32[], token[]), f32[16,16], u32[]) parameter(0)
//   gte = get-tuple-element(tuple), index=0
//   param = get-tuple-element(tuple), index=1
//   i = get-tuple-element(tuple), index=2
//   dot = f32[16,16] dot(param, param), lhs_contracting_dims={0}, rhs_contracting_dims={1}
//   recv_done = (f32[16], token[]) recv-done(gte), frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}, {1,0}}}
//   after_all = token[] after-all()
//   recv = (f32[16,16], u32[], token[]) recv(after_all), frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}, {1,0}}},
//   control-predecessors={recv_done} c1 = u32[] constant(1) add = add(i, c1) ROOT tuple_ = ((f32[16,16], u32[], token[]), f32[16,16], u32[]) tuple(recv, dot,
//   add)
// }
//
// ENTRY main {
//   param0 = f32[16,16] parameter(0)
//   after_all0 = token[] after-all()
//   recv0 = (f32[16,16], u32[], token[]) recv(after_all0), frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}, {1,0}}}
//   c0 = u32[] constant(0)
//   tuple = ((f32[16,16], u32[], token[]), f32[16,16], u32[]) tuple(recv0, param0, c0)
//   while = ((f32[16,16], u32[], token[]), f32[16,16], u32[]) while(tuple), body=while_body, condition=while_condition
//   gte0 = (f32[16,16], u32[], token[]) get-tuple-element(while), index=0
//   ROOT recv_done0 = (f32[16], token[]) recv-done(gte0), frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}, {1,0}}}
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, SchedulingAnnotationCrossesOverlapLimit) {
//   RunTest(R"(
//         HloModule module, is_scheduled=false
//
// ENTRY entry {
//   p0 = f32[16,64,256]{2,1,0} parameter(0)
//   p1 = f32[128,2048,2048]{2,1,0} parameter(1)
//   cp1s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0}},
//   frontend_attributes={_scheduling_group_id="0"} cp1d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp1s),
//   frontend_attributes={_scheduling_group_id="0"} cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1),
//   source_target_pairs={{1,0}} cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s) slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16],
//   [0:64], [0:256]} c1 = f32[16,256,256]{2,1,0} convolution(p0, p0),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
//   c2 = f32[16,256,256]{2,1,0} convolution(p0, slice),
//     window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
//   ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c1, c2, cp2d)
// }
//         )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, CrossComputationAnnotation) {
//   RunTest(R"(
//       HloModule module, is_scheduled=false
//
// while_cond {
//   param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body {
//   param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
//   gte0 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=0
//   gte1 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=1
//   gte2 = pred[] get-tuple-element(param), index=2
//   cps1 = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, u32[], u32[]) collective-permute-start(gte1), source_target_pairs={{0,1},{1,2},{2,3},{3,0}},
//   frontend_attributes={_scheduling_group_id="1"} cpd1 = f32[16,64,256]{2,1,0} collective-permute-done(cps1), frontend_attributes={_scheduling_group_id="1"}
//   c1 = f32[16,256,256]{2,1,0} convolution(gte0, gte0), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//   frontend_attributes={_scheduling_group_id="1"} slice = f32[16,64,256]{2,1,0} slice(c1), slice={[0:16], [0:64], [0:256]} add = f32[16,64,256]{2,1,0}
//   add(gte0, slice) ROOT tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(add, cpd1, gte2)
// }
//
// ENTRY entry {
//   p0 = f32[256,1024]{1,0} parameter(0)
//   p1 = f32[16,64,256]{2,1,0} parameter(1)
//   p2 = f32[16,64,256]{2,1,0} parameter(2)
//   p3 = pred[] parameter(3)
//   c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//   frontend_attributes={_scheduling_group_id="1"} ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1}, {1,0}},
//   dimensions={0}, frontend_attributes={_scheduling_group_id="1"} tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(p1, p2, p3) while =
//   (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) while(tuple), condition=while_cond, body=while_body agd0 = f32[1024,1024]{1,0}
//   all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"} gte = f32[16,64,256]{2,1,0} get-tuple-element(while), index=0 ROOT tuple1 =
//   (f32[16,64,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(gte, c0, agd0)
// }
//       )",
//           false);
// }
//
// XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, InvalidAnnotationOverlap) {
//   RunTest(R"(
//       HloModule module, is_scheduled=false
//
// while_cond {
//   param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
//   ROOT gte = pred[] get-tuple-element(param), index=2
// }
//
// while_body {
//   param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
//   gte0 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=0
//   gte1 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=1
//   gte2 = pred[] get-tuple-element(param), index=2
//   cps1 = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, u32[], u32[]) collective-permute-start(gte1), source_target_pairs={{0,1},{1,2},{2,3},{3,0}},
//   frontend_attributes={_scheduling_group_id="1"} cpd1 = f32[16,64,256]{2,1,0} collective-permute-done(cps1), frontend_attributes={_scheduling_group_id="1"}
//   c1 = f32[16,256,256]{2,1,0} convolution(gte0, gte0), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//   frontend_attributes={_scheduling_group_id="1"} slice = f32[16,64,256]{2,1,0} slice(c1), slice={[0:16], [0:64], [0:256]} add = f32[16,64,256]{2,1,0}
//   add(gte0, slice) ROOT tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(add, cpd1, gte2)
// }
//
// ENTRY entry {
//   p0 = f32[256,1024]{1,0} parameter(0)
//   p1 = f32[16,64,256]{2,1,0} parameter(1)
//   p2 = f32[16,64,256]{2,1,0} parameter(2)
//   p3 = pred[] parameter(3)
//   c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
//   frontend_attributes={_scheduling_group_id="1"} ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1}, {1,0}},
//   dimensions={0}, frontend_attributes={_scheduling_group_id="1"} ags1 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0),
//   replica_groups={{0,1}, {1,0}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"} tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0},
//   pred[]) tuple(p1, p2, p3) while = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) while(tuple), condition=while_cond, body=while_body agd1 =
//   f32[1024,1024]{1,0} all-gather-done(ags1), frontend_attributes={_scheduling_group_id="1"} agd0 = f32[1024,1024]{1,0} all-gather-done(ags0),
//   frontend_attributes={_scheduling_group_id="1"} gte = f32[16,64,256]{2,1,0} get-tuple-element(while), index=0 ROOT tuple1 = (f32[16,64,256]{2,1,0},
//   f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) tuple(gte, c0, agd0, agd1)
// }
//       )",
//           false);
// }

} // namespace xla
