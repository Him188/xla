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
#include "xla/tsl/lib/core/status_test_util.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

namespace xla {

template <typename NativeT>
std::pair<std::unique_ptr<PjRtBuffer>, std::shared_ptr<Literal>> CreateDeviceBufferR1(PjRtClient &client, const Shape &shape, NativeT value,
                                                                                      PjRtDevice &device) {
  std::vector<NativeT> host(shape.dimensions(0), value);
  Literal literal = LiteralUtil::CreateR1<NativeT>(host);
  const auto literal_ptr = std::make_shared<Literal>(std::move(literal));

  auto buffer = client.BufferFromHostLiteral(*literal_ptr.get(), device.default_memory_space().value()).value();
  return {std::move(buffer), literal_ptr};
}

template <typename NativeT>
std::pair<std::unique_ptr<PjRtBuffer>, std::shared_ptr<Literal>> CreateDeviceBufferR0(PjRtClient &client, NativeT value, PjRtDevice &device) {
  Literal literal = LiteralUtil::CreateR0<NativeT>(value);
  const auto literal_ptr = std::make_shared<Literal>(std::move(literal));

  auto buffer = client.BufferFromHostLiteral(*literal_ptr.get(), device.default_memory_space().value()).value();
  return {std::move(buffer), literal_ptr};
}

// Builds and returns the HLO module.
std::unique_ptr<HloModule> BuildModule() {

  using xla::BF16;
  using xla::HloComputation;
  using xla::HloInstruction;
  using xla::HloModule;
  using xla::HloModuleConfig;
  using xla::Shape;
  using xla::ShapeUtil;

  // ---------------------------------------------------------------------------
  // Module & common shapes
  // ---------------------------------------------------------------------------
  HloModuleConfig config;
  auto module = std::make_unique<HloModule>("module", config);

  const Shape array_shape = ShapeUtil::MakeShape(BF16, {8});
  const Shape pred_shape = ShapeUtil::MakeShape(xla::PRED, {});
  const Shape tuple_shape = ShapeUtil::MakeTupleShape({array_shape, array_shape, pred_shape});

  // ---------------------------------------------------------------------------
  // while_cond computation
  // ---------------------------------------------------------------------------
  {
    HloComputation::Builder cond_builder("while_cond");

    auto param = cond_builder.AddInstruction(HloInstruction::CreateParameter(/*parameter_number=*/0, tuple_shape, "param"));

    auto gte = cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(pred_shape, param, /*index=*/2));

    module->AddEmbeddedComputation(cond_builder.Build(gte));
  }
  HloComputation *cond_comp = module->GetComputationWithName("while_cond");

  // ---------------------------------------------------------------------------
  // while_body computation
  // ---------------------------------------------------------------------------
  {
    HloComputation::Builder body_builder("while_body");

    auto param = body_builder.AddInstruction(HloInstruction::CreateParameter(/*parameter_number=*/0, tuple_shape, "param"));

    auto gte0 = body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(array_shape, param, /*index=*/0));
    auto gte1 = body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(pred_shape, param, /*index=*/2));

    auto bitcast = body_builder.AddInstruction(HloInstruction::CreateBitcast(array_shape, gte0));

    const std::vector<std::pair<int64_t, int64_t>> permute1_pairs = {{0, 1}, {1, 0}};
    auto cp1 = body_builder.AddInstruction(HloInstruction::CreateCollectivePermute(array_shape, gte0, permute1_pairs, std::nullopt));

    auto add0 = body_builder.AddInstruction(HloInstruction::CreateBinary(array_shape, xla::HloOpcode::kAdd, cp1, bitcast));

    auto negate = body_builder.AddInstruction(HloInstruction::CreateUnary(array_shape, xla::HloOpcode::kNegate, add0));

    const std::vector<std::pair<int64_t, int64_t>> permute2_pairs = {{1, 0}, {0, 1}};
    auto cp2 = body_builder.AddInstruction(HloInstruction::CreateCollectivePermute(array_shape, cp1, permute2_pairs, std::nullopt));

    auto tuple_inst = body_builder.AddInstruction(HloInstruction::CreateTuple({cp2, negate, gte1}));

    module->AddEmbeddedComputation(body_builder.Build(tuple_inst));
  }
  HloComputation *body_comp = module->GetComputationWithName("while_body");

  // ---------------------------------------------------------------------------
  // entry computation
  // ---------------------------------------------------------------------------
  HloComputation::Builder entry_builder("entry");

  auto p0 = entry_builder.AddInstruction(HloInstruction::CreateParameter(0, array_shape, "p0"));
  auto p1 = entry_builder.AddInstruction(HloInstruction::CreateParameter(1, array_shape, "p1"));
  auto p2 = entry_builder.AddInstruction(HloInstruction::CreateParameter(2, pred_shape, "p2"));

  auto tuple_entry = entry_builder.AddInstruction(HloInstruction::CreateTuple({p0, p1, p2}));

  auto while_inst = entry_builder.AddInstruction(HloInstruction::CreateWhile(tuple_shape, cond_comp, body_comp, tuple_entry));

  auto gte0_e = entry_builder.AddInstruction(HloInstruction::CreateGetTupleElement(array_shape, while_inst, /*index=*/0));
  auto gte1_e = entry_builder.AddInstruction(HloInstruction::CreateGetTupleElement(array_shape, while_inst, /*index=*/1));

  auto add_root = entry_builder.AddInstruction(HloInstruction::CreateBinary(array_shape, xla::HloOpcode::kAdd, gte0_e, gte1_e));

  module->AddEntryComputation(entry_builder.Build(add_root));

  return module;
}

class LatencyHidingSchedulerConcurrencyTests : public ConcurrencyTestBase {
protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions dbg = ConcurrencyTestBase::GetDebugOptionsForTest();
    dbg.set_xla_gpu_enable_latency_hiding_scheduler(true);
    dbg.set_xla_gpu_dump_llvmir(true);
    dbg.set_xla_dump_hlo_as_html(true);
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
};

XLA_TEST_F(LatencyHidingSchedulerConcurrencyTests, AllScatterBug) {
  setenv("NCCL_DEBUG", "WARN", 1);

  // ----------------- build HLO -------------------------------------------- //
  XlaBuilder builder("add_allreduce_2gpu");

  const Shape mat_shape = ShapeUtil::MakeShape(BF16, {8});
  const Shape pred_shape = ShapeUtil::MakeShape(PRED, {});

  // ---------------- PJRT client / compilation ---------------------------- //
  GpuClientOptions opts;
  auto client_uptr = xla::GetStreamExecutorGpuClient(opts).value();
  auto &client = *client_uptr;

  ASSERT_GE(client.addressable_devices().size(), 2) << "Need at least two visible CUDA devices.";

  CompileOptions copts;
  auto &eb = copts.executable_build_options;
  eb.set_num_replicas(2);
  eb.set_num_partitions(1);
  eb.set_device_assignment(client.GetDefaultDeviceAssignment(2, 1).value());
  *eb.mutable_debug_options() = GetDebugOptionsForTest();

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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloText(hlo_string));

  module->mutable_config().set_replica_count(2);
  auto exe = client.Compile({module->ToProto()}, copts).value();

  // ---------------- host data -------------------------------------------- //
  PjRtDevice *dev0 = client.addressable_devices()[0];
  PjRtDevice *dev1 = client.addressable_devices()[1];

  auto [bufferA0, literalA0] = CreateDeviceBufferR1(client, mat_shape, static_cast<bfloat16>(1.0f), *dev0);
  auto [bufferB0, literalB0] = CreateDeviceBufferR1(client, mat_shape, static_cast<bfloat16>(1.0f), *dev0);
  auto [bufferC0, literalC0] = CreateDeviceBufferR0(client, false, *dev0);

  auto [bufferA1, literalA1] = CreateDeviceBufferR1(client, mat_shape, static_cast<bfloat16>(1.0f), *dev1);
  auto [bufferB1, literalB1] = CreateDeviceBufferR1(client, mat_shape, static_cast<bfloat16>(1.0f), *dev1);
  auto [bufferC1, literalC1] = CreateDeviceBufferR0(client, false, *dev1);

  std::vector<std::vector<PjRtBuffer *>> args = {
      {bufferA0.get(), bufferB0.get(), bufferC0.get()},
      {bufferA1.get(), bufferB1.get(), bufferC1.get()},
  };

  // ---------------- execute & verify ------------------------------------ //
  gpu::ConcurrencyTracer tracer;
  ExecuteOptions exec_opts;
  exec_opts.gpu_concurrency_tracer = &tracer;
  exec_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;

  auto outs = exe->Execute(args, exec_opts).value();

  ASSERT_EQ(outs.size(), 2);
  for (int p = 0; p < 2; ++p) {
    ASSERT_EQ(outs[p].size(), 1);
    auto lit = outs[p][0]->ToLiteralSync().value();
    float v = lit->DecomposeTuple()[0].Get<float>({0});
    EXPECT_NEAR(v, 4000.0f, 1e-6);
  }

  auto races = tracer.DetectDataRaces();
  std::cout << "races=" << races.size() << std::endl;
  if (!races.empty()) {
    tracer.PrintDataRaces(std::cout);
  }
}
} // namespace xla
