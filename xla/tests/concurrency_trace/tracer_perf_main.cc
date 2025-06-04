#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "stablehlo/dialect/Register.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/statusor.h"
#include "xla/backends/gpu/runtime/concurrency_trace.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sys/resource.h>
#include <unistd.h>

ABSL_FLAG(std::string, input, "", "Path to HLO or StableHLO file");
ABSL_FLAG(bool, stablehlo, false, "Input file is StableHLO (MLIR) format");
ABSL_FLAG(bool, trace, true, "Enable concurrency tracer");
ABSL_FLAG(int, replicas, 2, "Number of replicas");

namespace xla {

static size_t GetCurrentRSSBytes() {
  long rss = 0;
  FILE* fp = fopen("/proc/self/statm", "r");
  if (fp != nullptr) {
    if (fscanf(fp, "%*s%ld", &rss) != 1) rss = 0;
    fclose(fp);
  }
  return rss * sysconf(_SC_PAGESIZE);
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>> ParseHlo(
    absl::string_view hlo_text) {
  HloModuleConfig config;
  config.set_debug_options(DefaultDebugOptionsIgnoringFlags());
  config.set_replica_count(absl::GetFlag(FLAGS_replicas));
  config.set_num_partitions(1);
  auto module = std::make_unique<VerifiedHloModule>(
      "module", config,
      /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision_in_hlo_verifier=*/true,
      [](const Shape& s) { return ShapeUtil::ByteSizeOfElements(s); });
  TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_text));
  return std::move(module);
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
    PjRtClient& client, PjRtLoadedExecutable& executable,
    absl::Span<const absl::Span<const LiteralSlice>> args,
    const ExecuteOptions& exec_opts = {}) {
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> buffers;
  buffers.reserve(args.size());
  for (size_t device_index = 0; device_index < args.size(); ++device_index) {
    TF_ASSIGN_OR_RETURN(auto* mem_space,
                        client.addressable_devices()[device_index]
                            ->default_memory_space());
    auto& device_buffers = buffers.emplace_back();
    device_buffers.reserve(args[device_index].size());

    for (const LiteralSlice& arg : args.at(device_index)) {
      TF_ASSIGN_OR_RETURN(auto buffer,
                          client.BufferFromHostLiteral(arg, mem_space));
      TF_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
      device_buffers.emplace_back(std::move(buffer));
    }
  }

  std::vector<std::vector<PjRtBuffer*>> buffer_ptrs;
  buffer_ptrs.reserve(buffers.size());
  for (auto& device_buffers : buffers) {
    auto& ptrs = buffer_ptrs.emplace_back();
    ptrs.reserve(device_buffers.size());
    for (auto& buf : device_buffers) ptrs.push_back(buf.get());
  }

  TF_ASSIGN_OR_RETURN(auto res, executable.Execute(buffer_ptrs, exec_opts));

  for (auto& device_buffers : res) {
    for (const auto& buf : device_buffers) {
      TF_RETURN_IF_ERROR(buf->GetReadyFuture().Await());
    }
  }
  return res;
}

absl::Status Run() {
  std::string path = absl::GetFlag(FLAGS_input);
  if (path.empty()) {
    return absl::InvalidArgumentError("--input is required");
  }
  bool use_stablehlo = absl::GetFlag(FLAGS_stablehlo);
  bool enable_trace = absl::GetFlag(FLAGS_trace);
  int num_replicas = absl::GetFlag(FLAGS_replicas);

  std::string text;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), path, &text));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      GetStreamExecutorGpuClient(GpuClientOptions()));

  CompileOptions copts;
  auto& eb = copts.executable_build_options;
  eb.set_num_replicas(num_replicas);
  eb.set_num_partitions(1);
  TF_ASSIGN_OR_RETURN(const auto device_assignment, client->GetDefaultDeviceAssignment(num_replicas, 1));
  eb.set_device_assignment(device_assignment);
  DebugOptions debug = DefaultDebugOptionsIgnoringFlags();
  debug.set_xla_gpu_enable_latency_hiding_scheduler(true);
  *eb.mutable_debug_options() = debug;

  std::unique_ptr<PjRtLoadedExecutable> executable;
  absl::Time t0 = absl::Now();
  size_t rss_before_compile = GetCurrentRSSBytes();

  if (use_stablehlo) {
    mlir::DialectRegistry registry;
    mlir::func::registerAllExtensions(registry);
    mlir::mhlo::registerAllMhloDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
    mlir::MLIRContext context(registry);
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module) {
      return diagnostic_handler.ConsumeStatus();
    }
    TF_ASSIGN_OR_RETURN(executable, client->Compile(*module, copts));
  } else {
    TF_ASSIGN_OR_RETURN(auto module, ParseHlo(text));
    TF_ASSIGN_OR_RETURN(executable,
                        client->Compile({module->ToProto()}, copts));
  }

  absl::Time t1 = absl::Now();
  size_t rss_after_compile = GetCurrentRSSBytes();

  std::minstd_rand0 rng(42);
  std::vector<std::vector<Literal>> args_per_device;
  if (!use_stablehlo) {
    // Use HLO module shapes for argument generation
    TF_ASSIGN_OR_RETURN(auto module, ParseHlo(text));
    TF_ASSIGN_OR_RETURN(auto args, MakeFakeArgumentsForDevices(module.get(), num_replicas));
    args_per_device = std::move(args);
  } else {
    // For StableHLO we cannot easily infer shapes; require XLA HLO parsing
    // via executable->GetHloModules().
    TF_ASSIGN_OR_RETURN(auto hlo_modules, executable->GetHloModules());
    const HloModule* mod = hlo_modules.at(0).get();
    TF_ASSIGN_OR_RETURN(auto args, MakeFakeArgumentsForDevices(mod, num_replicas));
    args_per_device = std::move(args);
  }

  auto arg_slices = MakeFakeArgumentSlices(args_per_device);
  std::vector<absl::Span<const LiteralSlice>> exec_args;
  exec_args.reserve(arg_slices.size());
  for (const auto& v : arg_slices) exec_args.emplace_back(v);

  ExecuteOptions exec_opts;
  gpu::ConcurrencyTracer tracer;
  if (enable_trace) exec_opts.gpu_concurrency_tracer = &tracer;

  size_t rss_before_exec = GetCurrentRSSBytes();
  absl::Time t2 = absl::Now();
  TF_ASSIGN_OR_RETURN(auto outs,
                      Execute(*client, *executable, exec_args, exec_opts));
  absl::Time t3 = absl::Now();
  size_t rss_after_exec = GetCurrentRSSBytes();

  absl::Duration compile_dur = t1 - t0;
  absl::Duration exec_dur = t3 - t2;

  std::cout << "Compilation time (ms): "
            << absl::ToDoubleMilliseconds(compile_dur) << std::endl;
  std::cout << "Execution time (ms): "
            << absl::ToDoubleMilliseconds(exec_dur) << std::endl;
  std::cout << "Compilation memory delta (bytes): "
            << (rss_after_compile - rss_before_compile) << std::endl;
  std::cout << "Execution memory delta (bytes): "
            << (rss_after_exec - rss_before_exec) << std::endl;

  if (enable_trace) {
    std::cout << "Tracer memory usage (bytes): "
              << tracer.GetApproximateMemoryUsage() << std::endl;
    absl::Time td0 = absl::Now();
    auto races = tracer.DetectDataRaces();
    absl::Time td1 = absl::Now();
    std::cout << "Race detection time (ms): "
              << absl::ToDoubleMilliseconds(td1 - td0) << std::endl;
    std::cout << "races=" << races.size() << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace xla

int main(int argc, char** argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);
  absl::ParseCommandLine(argc, argv);
  auto status = xla::Run();
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return 1;
  }
  return 0;
}

