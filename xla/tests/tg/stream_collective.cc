#include "test_util.h"
#include "xla/backends/gpu/runtime/concurrency_trace.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/tsl/lib/core/status_test_util.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

namespace xla {
namespace {
// Helper that copies a host vector<float> into a device buffer.
absl::StatusOr<std::unique_ptr<PjRtBuffer>> VectorToDevice(PjRtClient &client, PjRtDevice *device, const std::vector<float> &host, const Shape &shape) {
  CHECK_EQ(shape.rank(), 2);
  const int64_t rows = shape.dimensions(0);
  const int64_t cols = shape.dimensions(1);
  CHECK_EQ(static_cast<int64_t>(host.size()), rows * cols);

  Literal lit(shape);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      lit.Set<float>({i, j}, host[i * cols + j]);
    }
  }
  return client.BufferFromHostLiteral(lit, device->default_memory_space().value());
}

TEST(PJRTReplicasTest, DotAllReduceTwoReplicas) {
  constexpr int64_t kN = 1024;
  constexpr int64_t kM = 1024;
  const Shape mat_shape = ShapeUtil::MakeShape(F32, {kN, kM});

  // -------------------------------------------------------------------------
  // Build HLO with an explicit AllReduce (replica_count = 2 in CompileOptions).
  // -------------------------------------------------------------------------
  XlaBuilder builder("dot_allreduce_replicated");

  OpSharding row_shard = HloSharding::Tile({{2, 1}}).ToProto();
  builder.SetSharding(row_shard);

  XlaOp A = Parameter(&builder, /*parameter_number=*/0, mat_shape, "A");
  XlaOp B = Parameter(&builder, 1, mat_shape, "B");
  XlaOp dot = Dot(A, B);

  // Scalar add computation for the reduction.
  XlaComputation add = CreateScalarAddComputation(F32, &builder);

  // One replica‑group containing both replicas.

  XlaOp ar = AllReduce(dot, add);
  XlaOp root = Tuple(&builder, {ar});
  TF_ASSERT_OK_AND_ASSIGN(xla::XlaComputation computation, builder.Build(root));

  // -------------------------------------------------------------------------
  // PJRT client and compilation.
  // -------------------------------------------------------------------------
  GpuClientOptions gpu_opts;
  // gpu_opts.mock_gpu_topology = "1x1x2";
  // gpu_opts.enable_mock_nccl = true;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client_uptr, GetStreamExecutorGpuClient(gpu_opts));
  PjRtClient &client = *client_uptr;

  const auto compile_for_device = [&](const int ordinal) -> std::unique_ptr<PjRtLoadedExecutable> {
    CompileOptions copts;
    auto &eb = copts.executable_build_options;

    eb.set_num_replicas(1);             // single replica
    eb.set_num_partitions(2);           // two partitions
    eb.set_use_spmd_partitioning(true); // turn on GSPMD

    // build an assignment matrix  (replicas x partitions == 1 x 2)
    const auto assignment = client.GetDefaultDeviceAssignment(/*replicas=*/1, /*partitions=*/2).value();
    eb.set_device_assignment(assignment);

    ExecutableBuildOptions &ebopts = copts.executable_build_options;

    DebugOptions *dbg = ebopts.mutable_debug_options();
    dbg->set_xla_gpu_async_dot(true);
    dbg->set_xla_gpu_multi_streamed_windowed_einsum(true);
    dbg->set_xla_gpu_enable_latency_hiding_scheduler(true);
    dbg->set_xla_dump_hlo_as_html(true);
    dbg->clear_xla_gpu_enable_command_buffer();
    dbg->add_xla_gpu_enable_command_buffer(DebugOptions::INVALID);

    std::unique_ptr<PjRtLoadedExecutable> exe = client.Compile(computation, copts).value();
    return std::move(exe);
  };

  // -------------------------------------------------------------------------
  // Prepare per‑replica inputs.
  // -------------------------------------------------------------------------
  std::vector<float> hostA0(kN * kM, 1.0f);
  std::vector<float> hostB0(kN * kM, 1.01f);
  std::vector<float> hostA1(kN * kM, 2.0f);
  std::vector<float> hostB1(kN * kM, 0.99f);

  PjRtDevice *device0 = client.devices()[0];
  PjRtDevice *device1 = client.devices()[1];

  TF_ASSERT_OK_AND_ASSIGN(auto bufA0, VectorToDevice(client, device0, hostA0, mat_shape));
  TF_ASSERT_OK_AND_ASSIGN(auto bufB0, VectorToDevice(client, device0, hostB0, mat_shape));
  TF_ASSERT_OK_AND_ASSIGN(auto bufA1, VectorToDevice(client, device1, hostA1, mat_shape));
  TF_ASSERT_OK_AND_ASSIGN(auto bufB1, VectorToDevice(client, device1, hostB1, mat_shape));

  std::vector<std::vector<PjRtBuffer *>> args = {
      {bufA0.get(), bufB0.get()}, // replica 0
      {bufA1.get(), bufB1.get()}  // replica 1
  };

  // -------------------------------------------------------------------------
  // Execute.
  // -------------------------------------------------------------------------

  gpu::ConcurrencyTracer tracer;

  ExecuteOptions execute_options;
  execute_options.gpu_concurrency_tracer = &tracer;
  TF_ASSERT_OK_AND_ASSIGN(auto outputs, compile_for_device(0)->Execute(args, /*options=*/execute_options));

  // Each replica returns a tuple with one element (the reduced matrix).
  ASSERT_EQ(outputs.size(), 2);
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(outputs[i].size(), 1);
    TF_ASSERT_OK_AND_ASSIGN(auto literal, outputs[i][0]->ToLiteralSync());

    // Spot‑check a single element; both replicas should agree because of the AllReduce.
    float val = literal->DecomposeTuple()[0].Get<float>({0, 0});
    std::cout << "replica " << i << " value[0,0] = " << val << std::endl;
    ASSERT_NEAR(val, 1.0f * kM * 2, 1e-3); // Very coarse check.
  }
}

constexpr int64_t N = 1024, M = 1024;
const Shape s32_shape = ShapeUtil::MakeShape(xla::S32, {});
const Shape mat_shape = ShapeUtil::MakeShape(F32, {N, M});
const Shape sliced_shape = ShapeUtil::MakeShape(F32, {100, 100});

XlaComputation BuildWhileAllReduceComputation() {
  XlaBuilder top("add_allreduce_loop");

  // Feed-in matrices for the *first* iteration.
  XlaOp A0 = Parameter(&top, 0, mat_shape, "A");
  XlaOp B0 = Parameter(&top, 1, mat_shape, "B");
  XlaOp iter0 = xla::ConstantR0<int32_t>(&top, 0);

  // Tuple<iter, A, B> becomes the loop-carried state.
  XlaOp init_state = Tuple(&top, {iter0, A0, Slice(A0, {100, 100}, {200, 200}, {1,1})});

  // ---------------- loop condition --------------------------------------- //
  XlaBuilder cond_b("cond");
  {
    constexpr int32_t kLoopTripCount = 8;
    XlaOp p = Parameter(&cond_b, 0, ShapeUtil::MakeTupleShape({s32_shape, mat_shape, sliced_shape}), "state");
    XlaOp iter = GetTupleElement(p, 0);
    Lt(iter, xla::ConstantR0<int32_t>(&cond_b, kLoopTripCount));
  }
  XlaComputation cond = cond_b.Build().value();

  // ---------------- loop body -------------------------------------------- //
  XlaBuilder body_b("body");
  XlaOp next_state;
  {
    XlaOp p = Parameter(&body_b, 0, ShapeUtil::MakeTupleShape({s32_shape, mat_shape, sliced_shape}), "state");
    XlaOp iter = GetTupleElement(p, 0);
    XlaOp A = GetTupleElement(p, 1);
    XlaOp acc = GetTupleElement(p, 2);

    XlaOp a_slice = Slice(A, {100, 100}, {200, 200}, {1,1});
    XlaOp a_slice2 = Slice(A, {150, 150}, {250, 250}, {1,1});

    // --- your original computation ------------------------------------- //
    XlaOp add_acc = AllReduce(a_slice, CreateScalarAddComputation(F32, &body_b));
    add_acc = Dot(add_acc, a_slice2);
    XlaOp mul_acc = AllReduce(a_slice2, CreateScalarAddComputation(F32, &body_b));

    // TODO: these are for latency hiding tests
    // XlaOp add_acc = AllReduce(A / B, CreateScalarAddComputation(F32, &body_b));
    // XlaOp minus_acc = AllReduce(A + B, CreateScalarAddComputation(F32, &body_b));
    // // add_acc = Dot(add_acc, B);
    // minus_acc = Dot(minus_acc, A);
    // XlaOp mul_acc = AllReduce(add_acc * minus_acc, CreateScalarMultiplyComputation(F32, &body_b));
    // // add_acc = Dot(mul_acc, add_acc);

    // Feed results forward to preserve true dependencies between
    // successive iterations – required for pipelining legality.
    XlaOp next_iter = iter + xla::ConstantR0<int32_t>(&body_b, 1);
    next_state = Tuple(&body_b, {next_iter, A, acc + mul_acc});
    // Return the updated tuple
  }
  XlaComputation body = body_b.Build(next_state).value();

  // ---------------- close the loop & module root ------------------------- //
  XlaOp final_state = While(cond, body, init_state);
  XlaOp final_A = GetTupleElement(final_state, 1);
  XlaOp final_B = GetTupleElement(final_state, 2);
  XlaOp root = Tuple(&top, {final_A * final_B});

  auto hlo = top.Build(root).value();
  return hlo;
}

TEST(GpuSpmd, AddReduceTwoWay) {
  setenv("NCCL_DEBUG", "WARN", 1);
  // 1.  prepare dump directory
  std::string dump_dir = ::testing::TempDir() + "/xla_dump";
  std::filesystem::create_directory(dump_dir);
  xla_test_util::SetXlaDumpFlags(dump_dir);
  // xla_test_util::EnableLogs();

  // shard inputs row-wise : {devices=[2,1] 0,1}
  // auto shard_proto = HloSharding::IotaTile({2, 1}).ToProto();
  // builder.SetSharding(shard_proto);
  // builder.ClearSharding();

  auto round = [&](const int round_number) {
    using namespace xla_test_util;

    // ----------------- build HLO ------------------------------------------ //
    XlaBuilder builder("add_allreduce_2gpu");

    // shard inputs row-wise : {devices=[2,1] 0,1}
    // auto shard_proto = HloSharding::IotaTile({2, 1}).ToProto();
    // builder.SetSharding(shard_proto);
    XlaOp A = Parameter(&builder, 0, mat_shape, "A");
    XlaOp B = Parameter(&builder, 1, mat_shape, "B");
    // builder.ClearSharding();

    XlaOp plusAcc = AllReduce(A + B, CreateScalarAddComputation(F32, &builder));
    XlaOp mulAcc = AllReduce(plusAcc * A, CreateScalarAddComputation(F32, &builder));
    XlaOp minusAcc = AllReduce(A - B, CreateScalarAddComputation(F32, &builder));
    XlaOp res = AllReduce(plusAcc * minusAcc, CreateScalarAddComputation(F32, &builder));
    XlaOp root = Tuple(&builder, {res * res});

    // TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build(root));
    auto computation = BuildWhileAllReduceComputation();

    // ---------------- PJRT client / compilation -------------------------- //
    GpuClientOptions opts;
    TF_ASSERT_OK_AND_ASSIGN(auto client_uptr, xla::GetStreamExecutorGpuClient(opts));
    auto &client = *client_uptr;

    ASSERT_GE(client.addressable_devices().size(), 2) << "Need at least two visible CUDA devices.";

    CompileOptions copts;
    auto &eb = copts.executable_build_options;
    eb.set_num_replicas(2);
    eb.set_num_partitions(1);

    TF_ASSERT_OK_AND_ASSIGN(auto da, client.GetDefaultDeviceAssignment(2, 1));
    eb.set_device_assignment(da);

    auto *dbg = eb.mutable_debug_options();
    dbg->set_xla_gpu_enable_latency_hiding_scheduler(true);
    dbg->set_xla_gpu_dump_llvmir(true);
    dbg->set_xla_dump_hlo_as_html(true);
    dbg->set_xla_gpu_enable_pipelined_collectives(true);
    dbg->set_xla_gpu_enable_pipelined_all_reduce(true);
    // dbg->set_xla_gpu_experimental_parallel_collective_overlap_limit(4);
    // dbg->set_xla_gpu_all_reduce_combine_threshold_bytes(999999999999);
    // dbg->set_xla_gpu_enable_highest_priority_async_stream(true);
    // dbg->set_xla_gpu_async_dot(true);

    dbg->clear_xla_gpu_enable_command_buffer();
    dbg->add_xla_gpu_enable_command_buffer(DebugOptions_CommandBufferCmdType_INVALID);

    // dbg->set_xla_gpu_all_reduce_combine_threshold_bytes(0);
    // dbg->set_xla_gpu_enable_nccl_user_buffers(true);

    TF_ASSERT_OK_AND_ASSIGN(auto exe, client.Compile(computation, copts));

    // ---------------- host data ------------------------------------------ //
    PjRtDevice *dev0 = client.addressable_devices()[0];
    PjRtDevice *dev1 = client.addressable_devices()[1];
    ASSERT_TRUE(dev0 != nullptr);
    ASSERT_TRUE(dev1 != nullptr);

    auto [bufferA0, literalA0] = CreateDeviceBufferR2(client, mat_shape, 1.0f, *dev0);
    auto [bufferB0, literalB0] = CreateDeviceBufferR2(client, mat_shape, 2.0f, *dev0);

    auto [bufferA1, literalA1] = CreateDeviceBufferR2(client, mat_shape, 3.0f, *dev1);
    auto [bufferB1, literalB1] = CreateDeviceBufferR2(client, mat_shape, 4.0f, *dev1);

    std::vector<std::vector<PjRtBuffer *>> args = {
        {bufferA0.get(), bufferB0.get()}, // replica 0
        {bufferA1.get(), bufferB1.get()}  // replica 1
    };

    // ---------------- execute & verify ----------------------------------- //
    gpu::ConcurrencyTracer tracer;
    ExecuteOptions exec_opts;
    exec_opts.gpu_concurrency_tracer = &tracer;
    exec_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;
    auto outs = exe->Execute(args, exec_opts).value();

    ASSERT_EQ(outs.size(), 2);
    for (int p = 0; p < 2; ++p) {
      ASSERT_EQ(outs[p].size(), 1);
      TF_ASSERT_OK_AND_ASSIGN(auto lit, outs[p][0]->ToLiteralSync());
      float v = lit->DecomposeTuple()[0].Get<float>({0, 0});
      EXPECT_NEAR(v, 400.0f, 1e-4); // (1+2)+(3+4) = 10, *2 = 20
    }

    // PrintIrDumps(dump_dir, {IRDumpKind::kHTML});

    if (round_number == 0) {
      print_gpu_thunk_info(exe.get());
      tracer.PrintTraces(std::cout);
    }
    auto races = tracer.DetectDataRaces();

    std::cout << "Round " << round_number << ": " << "races=" << races.size() << std::endl;
    if (!races.empty()) {
      tracer.PrintDataRaces(std::cout);
    }
  };

  for (int i = 0; i < 1; ++i) {
    round(i);
  }
}


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
  const Shape pred_shape  = ShapeUtil::MakeShape(xla::PRED, {});
  const Shape tuple_shape = ShapeUtil::MakeTupleShape({array_shape, array_shape, pred_shape});

  // ---------------------------------------------------------------------------
  // while_cond computation
  // ---------------------------------------------------------------------------
  {
    HloComputation::Builder cond_builder("while_cond");

    auto param = cond_builder.AddInstruction(
        HloInstruction::CreateParameter(/*parameter_number=*/0, tuple_shape, "param"));

    auto gte = cond_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(pred_shape, param, /*index=*/2));

    module->AddEmbeddedComputation(cond_builder.Build(gte));
  }
  HloComputation* cond_comp = module->GetComputationWithName("while_cond");

  // ---------------------------------------------------------------------------
  // while_body computation
  // ---------------------------------------------------------------------------
  {
    HloComputation::Builder body_builder("while_body");

    auto param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(/*parameter_number=*/0, tuple_shape, "param"));

    auto gte0 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(array_shape, param, /*index=*/0));
    auto gte1 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(pred_shape,  param, /*index=*/2));

    auto bitcast = body_builder.AddInstruction(
        HloInstruction::CreateBitcast(array_shape, gte0));

    const std::vector<std::pair<int64_t, int64_t>> permute1_pairs = {{0,1},{1,0}};
    auto cp1 = body_builder.AddInstruction(
        HloInstruction::CreateCollectivePermute(array_shape, gte0, permute1_pairs, std::nullopt));

    auto add0 = body_builder.AddInstruction(
        HloInstruction::CreateBinary(array_shape, xla::HloOpcode::kAdd, cp1, bitcast));

    auto negate = body_builder.AddInstruction(
        HloInstruction::CreateUnary(array_shape, xla::HloOpcode::kNegate, add0));

    const std::vector<std::pair<int64_t, int64_t>> permute2_pairs = {{1,0},{0,1}};
    auto cp2 = body_builder.AddInstruction(
        HloInstruction::CreateCollectivePermute(array_shape, cp1, permute2_pairs, std::nullopt));

    auto tuple_inst = body_builder.AddInstruction(
        HloInstruction::CreateTuple({cp2, negate, gte1}));

    module->AddEmbeddedComputation(body_builder.Build(tuple_inst));
  }
  HloComputation* body_comp = module->GetComputationWithName("while_body");

  // ---------------------------------------------------------------------------
  // entry computation
  // ---------------------------------------------------------------------------
  HloComputation::Builder entry_builder("entry");

  auto p0 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, array_shape, "p0"));
  auto p1 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(1, array_shape, "p1"));
  auto p2 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(2, pred_shape,  "p2"));

  auto tuple_entry = entry_builder.AddInstruction(
      HloInstruction::CreateTuple({p0, p1, p2}));

  auto while_inst = entry_builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, cond_comp, body_comp, tuple_entry));

  auto gte0_e = entry_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(array_shape, while_inst, /*index=*/0));
  auto gte1_e = entry_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(array_shape, while_inst, /*index=*/1));

  auto add_root = entry_builder.AddInstruction(
      HloInstruction::CreateBinary(array_shape, xla::HloOpcode::kAdd, gte0_e, gte1_e));

  module->AddEntryComputation(entry_builder.Build(add_root));

  return module;
}


TEST(GpuSpmd, AllScatterBug) {
  setenv("NCCL_DEBUG", "WARN", 1);
  // 1.  prepare dump directory
  std::string dump_dir = ::testing::TempDir() + "/xla_dump";
  std::filesystem::create_directory(dump_dir);
  xla_test_util::SetXlaDumpFlags(dump_dir);
  // xla_test_util::EnableLogs();

  // shard inputs row-wise : {devices=[2,1] 0,1}
  // auto shard_proto = HloSharding::IotaTile({2, 1}).ToProto();
  // builder.SetSharding(shard_proto);
  // builder.ClearSharding();

  auto round = [&](const int round_number) {
    using namespace xla_test_util;

    // ----------------- build HLO ------------------------------------------ //
    XlaBuilder builder("add_allreduce_2gpu");

    const Shape mat_shape = ShapeUtil::MakeShape(BF16, {8});
    const Shape pred_shape = ShapeUtil::MakeShape(PRED, {});
    // shard inputs row-wise : {devices=[2,1] 0,1}
    // auto shard_proto = HloSharding::IotaTile({2, 1}).ToProto();
    // builder.SetSharding(shard_proto);
    XlaOp A = Parameter(&builder, 0, mat_shape, "A");
    XlaOp B = Parameter(&builder, 1, mat_shape, "B");
    // builder.ClearSharding();

    auto &body_b = builder;

    /* replica-sum of A+B → every element becomes 10 on the two-GPU test ---- */
    XlaOp plus_acc = AllReduce(Slice(A, {0, 0}, {512, 1024}, {1, 1}), CreateScalarAddComputation(F32, &body_b));
    XlaOp plus_acc2 = AllReduce(Slice(A, {256, 0}, {256 + 512, 1024}, {1, 1}), CreateScalarMaxComputation(F32, &body_b));

    A = A * xla::ConstantR0<float>(&body_b, 2) + xla::ConstantR0<float>(&body_b, 5);
    XlaOp plus_acc3 = AllReduce(Slice(A, {128, 0}, {128 + 512, 1024}, {1, 1}), CreateScalarMultiplyComputation(F32, &body_b));

    /* multiply by 2 so that every element is 20 --------------------------- */
    XlaOp twenty = plus_acc * plus_acc2;
    XlaOp concat = ConcatInDim(&body_b, {twenty, plus_acc3}, 0);

    /* carry identical 20×20 matrices forward in both slots ---------------- */
    auto root = Tuple(&body_b, {concat, concat});

    // TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build(root));
    // auto computation = BuildWhileAllReduceComputation();

    // ---------------- PJRT client / compilation -------------------------- //
    GpuClientOptions opts;
    auto client_uptr= xla::GetStreamExecutorGpuClient(opts).value();
    auto &client = *client_uptr;

    ASSERT_GE(client.addressable_devices().size(), 2) << "Need at least two visible CUDA devices.";

    CompileOptions copts;
    auto &eb = copts.executable_build_options;
    eb.set_num_replicas(2);
    eb.set_num_partitions(1);

    auto da= client.GetDefaultDeviceAssignment(2, 1).value();
    eb.set_device_assignment(da);

    auto *dbg = eb.mutable_debug_options();
    dbg->set_xla_gpu_enable_latency_hiding_scheduler(true);
    dbg->set_xla_gpu_dump_llvmir(true);
    dbg->set_xla_dump_hlo_as_html(true);
    dbg->set_xla_gpu_enable_pipelined_collectives(true);
    dbg->set_xla_gpu_enable_pipelined_all_reduce(true);
    // dbg->set_xla_gpu_experimental_parallel_collective_overlap_limit(0);
    dbg->set_xla_gpu_all_reduce_combine_threshold_bytes(999999999999);
    // dbg->set_xla_gpu_enable_highest_priority_async_stream(true);
    // dbg->set_xla_gpu_async_dot(true);

    dbg->set_xla_gpu_copy_insertion_use_region_analysis(true);

    dbg->clear_xla_gpu_enable_command_buffer();
    dbg->add_xla_gpu_enable_command_buffer(DebugOptions_CommandBufferCmdType_INVALID);

    // dbg->set_xla_gpu_all_reduce_combine_threshold_bytes(0);
    // dbg->set_xla_gpu_enable_nccl_user_buffers(true);

    const auto module = BuildModule();
    auto exe= client.Compile({module->ToProto()}, copts).value();

    // ---------------- host data ------------------------------------------ //
    PjRtDevice *dev0 = client.addressable_devices()[0];
    PjRtDevice *dev1 = client.addressable_devices()[1];
    ASSERT_TRUE(dev0 != nullptr);
    ASSERT_TRUE(dev1 != nullptr);

    auto [bufferA0, literalA0] = CreateDeviceBufferR1(client, mat_shape, static_cast<bfloat16>(1.0f), *dev0);
    auto [bufferB0, literalB0] = CreateDeviceBufferR1(client, mat_shape,  static_cast<bfloat16>(1.0f), *dev0);
    auto [bufferC0, literalC0] = CreateDeviceBufferR0(client, pred_shape, false, *dev0);

    auto [bufferA1, literalA1] = CreateDeviceBufferR1(client, mat_shape,  static_cast<bfloat16>(1.0f), *dev1);
    auto [bufferB1, literalB1] = CreateDeviceBufferR1(client, mat_shape,  static_cast<bfloat16>(1.0f), *dev1);
    auto [bufferC1, literalC1] = CreateDeviceBufferR0(client, pred_shape, false, *dev1);

    std::vector<std::vector<PjRtBuffer *>> args = {
        {bufferA0.get(), bufferB0.get(), bufferC0.get()}, // replica 0
        {bufferA1.get(), bufferB1.get(), bufferC1.get()}  // replica 1
    };

    // ---------------- execute & verify ----------------------------------- //
    gpu::ConcurrencyTracer tracer;
    ExecuteOptions exec_opts;
    exec_opts.gpu_concurrency_tracer = &tracer;
    exec_opts.gpu_synthetic_bug_options.nccl_collective_done_thunk = false;

    if (round_number == 0) {
      print_gpu_thunk_info(exe.get());
      tracer.PrintTraces(std::cout);
    }

    auto races = tracer.DetectDataRaces();

    std::cout << "Round " << round_number << ": " << "races=" << races.size() << std::endl;
    if (!races.empty()) {
      tracer.PrintDataRaces(std::cout);
    }

    ASSERT_TRUE(false);
    auto outs = exe->Execute(args, exec_opts).value();

    ASSERT_EQ(outs.size(), 2);
    for (int p = 0; p < 2; ++p) {
      ASSERT_EQ(outs[p].size(), 1);
      auto lit = outs[p][0]->ToLiteralSync().value();
      float v = lit->DecomposeTuple()[0].Get<float>({0});
      EXPECT_NEAR(v, 4000.0f, 1e-6); // (1+2)+(3+4) = 10, *2 = 20
    }

    // PrintIrDumps(dump_dir, {IRDumpKind::kHTML});
  };

  for (int i = 0; i < 1; ++i) {
    round(i);
  }
}
} // namespace

} // namespace xla