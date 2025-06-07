#include "xla/tests/concurrency_trace/base_concurrency_tests.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tests/concurrency_trace/perf_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

namespace xla {

DebugOptions BaseConcurrencyTests::GetDebugOptionsForTest() const {
  DebugOptions dbg = PjRtGpuStreamExecutorConcurrencyTestBase::GetDebugOptionsForTest();
  dbg.set_xla_gpu_enable_latency_hiding_scheduler(true);
  dbg.set_xla_gpu_enable_pipelined_collectives(true);
  dbg.set_xla_gpu_enable_pipelined_all_reduce(true);
  dbg.set_xla_gpu_copy_insertion_use_region_analysis(true);
  dbg.clear_xla_gpu_enable_command_buffer();
  dbg.add_xla_gpu_enable_command_buffer(DebugOptions_CommandBufferCmdType_INVALID);
  if (enable_async_dot_) {
    dbg.set_xla_gpu_async_dot(true);
  }
  return dbg;
}

absl::StatusOr<std::unique_ptr<HloModule>> BaseConcurrencyTests::ParseHloText(absl::string_view hlo_string) {
  return ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest());
}

size_t BaseConcurrencyTests::GetCurrentRSSBytes() {
  return xla::GetCurrentRSSBytes();
}

void BaseConcurrencyTests::RunTest(std::string_view hlo_string, bool expect_race, int warmup_iters, int measure_iters) {
  setenv("NCCL_DEBUG", "INFO", 1);
  ASSERT_GE(client().addressable_devices().size(), 2) << "Need at least two visible CUDA devices.";

  struct IterationResources {
    std::unique_ptr<PjRtLoadedExecutable> exe;
    std::unique_ptr<HloModule> module;
    std::vector<std::vector<Literal>> fake_args;
    std::vector<std::vector<LiteralSlice>> fake_arg_slices;
    std::vector<absl::Span<const LiteralSlice>> exec_args;
  };

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
    gpu::ThunkSanitizer tracer;
    ExecuteOptions exec_opts;
    exec_opts.gpu_thunk_sanitizer = &tracer;
    exec_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;
    exec_opts.gpu_synthetic_bug_options.wait_for_streams_thunk = enable_wait_for_streams_bug_;

    TF_ASSERT_OK_AND_ASSIGN(auto res, prepare_fn());
    auto &exe = res.exe;
    {
      TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe, res.exec_args, exec_opts));
      (void)outs;
    }

    xla_test_util::print_gpu_thunk_info(exe.get());
    auto races = tracer.DetectDataRaces();
    if (!races.empty())
      tracer.PrintDataRaces(std::cout);
    tracer.PrintTraces(std::cout);
    ASSERT_EQ(races.empty(), !expect_race);
    exe->Delete();
  } else {
    if (const char *env = getenv("XLA_TRACER_WARMUP")) {
      warmup_iters = std::stoi(env);
    }

    std::vector<double> base_times;
    std::vector<size_t> base_memory;
    std::vector<double> traced_times;
    std::vector<size_t> traced_memory;
    std::vector<size_t> tracer_memory;
    gpu::ExecutableStats exec_stats;
    gpu::ThunkSanitizer::TraceStats trace_stats;
    bool stats_collected = false;

    ExecuteOptions base_opts;
    base_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;
    base_opts.gpu_synthetic_bug_options.wait_for_streams_thunk = enable_wait_for_streams_bug_;

    for (int i = 0; i < warmup_iters; ++i) {
      TF_ASSERT_OK_AND_ASSIGN(auto res, prepare_fn());
      auto &exe = res.exe;
      {
        TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe, res.exec_args, base_opts));
        (void)outs;
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
        (void)outs;
      }
      absl::Time t1 = absl::Now();
      size_t rss_after = GetCurrentRSSBytes();
      base_times.push_back(absl::ToDoubleMilliseconds(t1 - t0));
      base_memory.push_back(rss_after - rss_before);
      exe->Delete();
    }

    for (int i = 0; i < warmup_iters; ++i) {
      TF_ASSERT_OK_AND_ASSIGN(auto res, prepare_fn());
      auto &exe = res.exe;
      gpu::ThunkSanitizer tracer;
      ExecuteOptions exec_opts;
      exec_opts.gpu_thunk_sanitizer = &tracer;
      exec_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;
      exec_opts.gpu_synthetic_bug_options.wait_for_streams_thunk = enable_wait_for_streams_bug_;
      {
        TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe, res.exec_args, exec_opts));
        (void)outs;
      }
      auto races = tracer.DetectDataRaces();
      ASSERT_EQ(races.empty(), !expect_race);
      exe->Delete();
    }

    for (int i = 0; i < measure_iters; ++i) {
      TF_ASSERT_OK_AND_ASSIGN(auto res, prepare_fn());
      auto &exe = res.exe;
      gpu::ThunkSanitizer tracer;
      ExecuteOptions exec_opts;
      exec_opts.gpu_thunk_sanitizer = &tracer;
      exec_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;
      exec_opts.gpu_synthetic_bug_options.wait_for_streams_thunk = enable_wait_for_streams_bug_;
      size_t rss_before = GetCurrentRSSBytes();
      absl::Time t0 = absl::Now();
      {
        TF_ASSERT_OK_AND_ASSIGN(auto outs, Execute(*exe, res.exec_args, exec_opts));
        (void)outs;
      }
      absl::Time t1 = absl::Now();
      size_t rss_after = GetCurrentRSSBytes();
      traced_times.push_back(absl::ToDoubleMilliseconds(t1 - t0));
      traced_memory.push_back(rss_after - rss_before);
      tracer_memory.push_back(tracer.GetApproximateMemoryUsage());
      auto races = tracer.DetectDataRaces();
      ASSERT_EQ(races.empty(), !expect_race);
      if (!stats_collected) {
        auto exec_stats_or = gpu::GetExecutableStats(exe.get());
        if (exec_stats_or.ok())
          exec_stats = *exec_stats_or;
        trace_stats = tracer.GetTraceStats();
        stats_collected = true;
      }
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
    os << "  },\n";
    PrintTraceAndExecutableStatsJson(trace_stats, exec_stats, os, 2);
    os << "}";
    std::cout << os.str() << std::endl;
  }
}

absl::StatusOr<BaseConcurrencyTests::ExeWithModule> BaseConcurrencyTests::CompileWithModule(std::string_view hlo_string, const DeviceMesh &mesh,
                                                                                            DebugOptions *debug_options) {
  TF_ASSIGN_OR_RETURN(auto module, ParseHloText(hlo_string));
  TF_ASSIGN_OR_RETURN(auto exe, Compile(module.get(), mesh, debug_options));
  return ExeWithModule(std::move(exe), std::move(module));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> BaseConcurrencyTests::Compile(std::string_view hlo_string, const DeviceMesh &mesh,
                                                                                    DebugOptions *debug_options) {
  TF_ASSIGN_OR_RETURN(auto pair, CompileWithModule(hlo_string, mesh, debug_options));
  return std::move(pair.first);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> BaseConcurrencyTests::Compile(HloModule *module, const DeviceMesh &mesh, DebugOptions *debug_options) {
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

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> BaseConcurrencyTests::CompileStableHlo(std::string_view stablehlo_string, const DeviceMesh &mesh,
                                                                                             DebugOptions *debug_options) {
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
BaseConcurrencyTests::Execute(PjRtLoadedExecutable &executable, absl::Span<const absl::Span<const LiteralSlice>> args, const ExecuteOptions &exec_opts) const {
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

  std::vector<std::vector<PjRtBuffer *>> buffer_ptrs;
  buffer_ptrs.reserve(buffers.size());
  for (auto &device_buffers : buffers) {
    auto &ptrs = buffer_ptrs.emplace_back();
    ptrs.reserve(device_buffers.size());
    for (auto &buf : device_buffers)
      ptrs.push_back(buf.get());
  }

  TF_ASSIGN_OR_RETURN(auto res, executable.Execute(buffer_ptrs, exec_opts));

  for (auto &device_buffers : res) {
    for (const auto &buf : device_buffers) {
      TF_RETURN_IF_ERROR(buf->GetReadyFuture().Await());
    }
  }
  return res;
}

std::vector<absl::Span<const LiteralSlice>> BaseConcurrencyTests::MakeInnerSpan(const std::vector<std::vector<LiteralSlice>> &fake_arg_slices) {
  std::vector<absl::Span<const LiteralSlice>> exec_args;
  exec_args.reserve(fake_arg_slices.size());
  for (const auto &slices : fake_arg_slices) {
    exec_args.emplace_back(slices);
  }
  return exec_args;
}

} // namespace xla
