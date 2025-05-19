#include "xla/backends/gpu/runtime/concurrency_trace.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/tests/concurrency_trace/concurrency_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/tg/test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

namespace xla {

class LatencyHidingSchedulerConcurrencyTests : public PjRtGpuStreamExecutorConcurrencyTestBase {
protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions dbg = PjRtGpuStreamExecutorConcurrencyTestBase::GetDebugOptionsForTest();
    dbg.set_xla_gpu_enable_latency_hiding_scheduler(true);
    // dbg.set_xla_gpu_dump_llvmir(true);
    // dbg.set_xla_dump_hlo_as_html(true);
    dbg.set_xla_gpu_enable_pipelined_collectives(true);
    dbg.set_xla_gpu_enable_pipelined_all_reduce(true);
    dbg.set_xla_gpu_all_reduce_combine_threshold_bytes(999999999999);
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

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(const std::string_view hlo_string, const DeviceMesh &mesh) {
    TF_ASSIGN_OR_RETURN(const auto module, ParseHloText(hlo_string));

    module->mutable_config().set_replica_count(mesh.num_replicas);

    CompileOptions copts;
    auto &eb = copts.executable_build_options;
    eb.set_num_replicas(mesh.num_replicas);
    eb.set_num_partitions(mesh.num_partitions);

    TF_ASSIGN_OR_RETURN(const auto device_assignment, client().GetDefaultDeviceAssignment(mesh.num_replicas, mesh.num_partitions));
    eb.set_device_assignment(device_assignment);
    *eb.mutable_debug_options() = GetDebugOptionsForTest();
    return client().Compile({module->ToProto()}, copts).value();
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

      for (const LiteralSlice &arg : args[device_index]) {
        TF_ASSIGN_OR_RETURN(auto buffer, client().BufferFromHostLiteral(arg, mem_space));
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
    return res;
  }
};

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllScatterBug) {
  setenv("NCCL_DEBUG", "WARN", 1);

  ASSERT_GE(client().addressable_devices().size(), 2) << "Need at least two visible CUDA devices.";

  const auto hlo_string = R"(
HloModule module

while_cond {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param), index=2
}

while_body {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  gte0 = bf16[8]{0} get-tuple-element(param), index=0
  gte1 = pred[] get-tuple-element(param), index=2
  bitcast = bf16[8]{0} bitcast(gte0)
  collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1},{1,0}}
  add0 = bf16[8]{0} add(collective-permute.1, bitcast)
  negate = bf16[8]{0} negate(add0)
  collective-permute.2 = bf16[8]{0} collective-permute(collective-permute.1), source_target_pairs={{1,0},{0,1}}
  ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte1)
}

ENTRY entry {
  p0 = bf16[8]{0} parameter(0)
  p1 = bf16[8]{0} parameter(1)
  p2 = pred[] parameter(2)
  tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
  while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
  gte0 = bf16[8]{0} get-tuple-element(while), index=0
  gte1 = bf16[8]{0} get-tuple-element(while), index=1
  ROOT add = bf16[8]{0} add(gte0, gte1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto exe, Compile(hlo_string, {2, 1}));

  // ---------------- host data -------------------------------------------- //
  Literal mat = LiteralUtil::CreateFull({8}, static_cast<bfloat16>(1.0f));
  Literal pred = LiteralUtil::CreateR0<bool>(false);

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

  // Check
  ASSERT_EQ(outs.size(), 2);
  for (int p = 0; p < 2; ++p) {
    ASSERT_EQ(outs[p].size(), 1);
    TF_ASSERT_OK_AND_ASSIGN(auto lit, outs[p][0]->ToLiteralSync());
    float v = lit->Get<tsl::bfloat16>({0});
    EXPECT_NEAR(v, 2.0f, 1e-6);
  }

} // namespace xla
} // namespace xla
