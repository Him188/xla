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

XlaComputation BuildWhileAllReduceComputation() {
  constexpr int64_t N = 1024, M = 1024;
  const Shape s32_shape = ShapeUtil::MakeShape(xla::S32, {});
  const Shape mat_shape = ShapeUtil::MakeShape(F32, {N, M});

  XlaBuilder top("add_allreduce_loop");

  // Feed-in matrices for the *first* iteration.
  XlaOp A0 = Parameter(&top, 0, mat_shape, "A");
  XlaOp B0 = Parameter(&top, 1, mat_shape, "B");
  XlaOp iter0 = xla::ConstantR0<int32_t>(&top, 0);

  // Tuple<iter, A, B> becomes the loop-carried state.
  XlaOp init_state = Tuple(&top, {iter0, A0, B0});

  // ---------------- loop condition --------------------------------------- //
  XlaBuilder cond_b("cond");
  {
    constexpr int32_t kLoopTripCount = 8;
    XlaOp p = Parameter(&cond_b, 0, ShapeUtil::MakeTupleShape({s32_shape, mat_shape, mat_shape}), "state");
    XlaOp iter = GetTupleElement(p, 0);
    Lt(iter, xla::ConstantR0<int32_t>(&cond_b, kLoopTripCount));
  }
  XlaComputation cond = cond_b.Build().value();

  // ---------------- loop body -------------------------------------------- //
  XlaBuilder body_b("body");
  XlaOp next_state;
  {
    XlaOp p = Parameter(&body_b, 0, ShapeUtil::MakeTupleShape({s32_shape, mat_shape, mat_shape}), "state");
    XlaOp iter = GetTupleElement(p, 0);
    XlaOp A = GetTupleElement(p, 1);
    XlaOp B = GetTupleElement(p, 2);

    // --- your original computation ------------------------------------- //
    XlaOp add_acc = AllReduce(A + B, CreateScalarAddComputation(F32, &body_b));
    XlaOp mul_acc = AllReduce(add_acc * A, CreateScalarAddComputation(F32, &body_b));

    // Feed results forward to preserve true dependencies between
    // successive iterations – required for pipelining legality.
    XlaOp next_iter = iter + xla::ConstantR0<int32_t>(&body_b, 1);
    next_state = Tuple(&body_b, {next_iter, add_acc, mul_acc});
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

  using namespace xla_test_util;

  constexpr int64_t N = 1024, M = 1024;
  const Shape mat_shape = ShapeUtil::MakeShape(F32, {N, M});
  const Shape slice_shape = ShapeUtil::MakeShape(F32, {N / 2, M});

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
  XlaOp root = Tuple(&builder, {plusAcc * mulAcc});

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
  dbg->set_xla_gpu_enable_highest_priority_async_stream(true);

  dbg->clear_xla_gpu_enable_command_buffer();
  dbg->add_xla_gpu_enable_command_buffer(DebugOptions_CommandBufferCmdType_INVALID);

  // dbg->set_xla_gpu_all_reduce_combine_threshold_bytes(0);
  dbg->set_xla_gpu_enable_nccl_user_buffers(true);

  TF_ASSERT_OK_AND_ASSIGN(auto exe, client.Compile(computation, copts));

  // ---------------- host data ------------------------------------------ //
  PjRtDevice *dev0 = client.addressable_devices()[0];
  PjRtDevice *dev1 = client.addressable_devices()[1];
  ASSERT_TRUE(dev0 != nullptr);
  ASSERT_TRUE(dev1 != nullptr);

  auto [bufferA0, literalA0] = CreateDeviceBuffer(client, mat_shape, 1.0f, *dev0);
  auto [bufferB0, literalB0] = CreateDeviceBuffer(client, mat_shape, 2.0f, *dev0);

  auto [bufferA1, literalA1] = CreateDeviceBuffer(client, mat_shape, 3.0f, *dev1);
  auto [bufferB1, literalB1] = CreateDeviceBuffer(client, mat_shape, 4.0f, *dev1);

  std::vector<std::vector<PjRtBuffer *>> args = {
      {bufferA0.get(), bufferB0.get()}, // replica 0
      {bufferA1.get(), bufferB1.get()}  // replica 1
  };

  // ---------------- execute & verify ----------------------------------- //
  gpu::ConcurrencyTracer tracer;
  ExecuteOptions exec_opts;
  exec_opts.gpu_concurrency_tracer = &tracer;
  auto outs = exe->Execute(args, exec_opts).value();

  print_gpu_thunk_info(exe.get());

  ASSERT_EQ(outs.size(), 2);
  for (int p = 0; p < 2; ++p) {
    ASSERT_EQ(outs[p].size(), 1);
    TF_ASSERT_OK_AND_ASSIGN(auto lit, outs[p][0]->ToLiteralSync());
    float v = lit->DecomposeTuple()[0].Get<float>({0, 0});
    EXPECT_NEAR(v, 400.0f, 1e-4); // (1+2)+(3+4) = 10, *2 = 20
  }

  PrintIrDumps(dump_dir, {IRDumpKind::kHTML});

  tracer.PrintTraces(std::cout);
  tracer.PrintDataRaces(std::cout);
}
} // namespace

} // namespace xla