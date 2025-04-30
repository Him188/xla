#include "test_util.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/tsl/lib/core/status_test_util.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

namespace xla {
namespace {
// Helper that copies a host vector<float> into a device buffer.
absl::StatusOr<std::unique_ptr<PjRtBuffer>> VectorToDevice(PjRtClient &client, PjRtDevice *device, const std::vector<float> &host, const xla::Shape &shape) {
  CHECK_EQ(shape.rank(), 2);
  const int64_t rows = shape.dimensions(0);
  const int64_t cols = shape.dimensions(1);
  CHECK_EQ(static_cast<int64_t>(host.size()), rows * cols);

  xla::Literal lit(shape);
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
  const xla::Shape mat_shape = ShapeUtil::MakeShape(xla::F32, {kN, kM});

  // -------------------------------------------------------------------------
  // Build HLO with an explicit AllReduce (replica_count = 2 in CompileOptions).
  // -------------------------------------------------------------------------
  xla::XlaBuilder builder("dot_allreduce_replicated");

  xla::OpSharding row_shard = xla::HloSharding::Tile({{2, 1}}).ToProto();
  builder.SetSharding(row_shard);

  xla::XlaOp A = xla::Parameter(&builder, /*parameter_number=*/0, mat_shape, "A");
  xla::XlaOp B = xla::Parameter(&builder, 1, mat_shape, "B");
  xla::XlaOp dot = xla::Dot(A, B);

  // Scalar add computation for the reduction.
  xla::XlaComputation add = xla::CreateScalarAddComputation(xla::F32, &builder);

  // One replica‑group containing both replicas.

  xla::XlaOp ar = xla::AllReduce(dot, add);
  xla::XlaOp root = xla::Tuple(&builder, {ar});
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
    auto& eb = copts.executable_build_options;

    eb.set_num_replicas(1);          // single replica
    eb.set_num_partitions(2);        // two partitions
    eb.set_use_spmd_partitioning(true);   // turn on GSPMD

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
    dbg->add_xla_gpu_enable_command_buffer(xla::DebugOptions::INVALID);

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

  TF_ASSERT_OK_AND_ASSIGN(auto outputs, compile_for_device(0)->Execute(args, /*options=*/{}));

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

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> VecToDevice(
    xla::PjRtClient& client, xla::PjRtDevice* dev,
    absl::Span<const float> host, const xla::Shape& global_shape,
    int64_t row_start) {
  const int64_t rows_per_part = global_shape.dimensions(0) / 2;
  const int64_t cols          = global_shape.dimensions(1);

  xla::Shape slice_shape = ShapeUtil::MakeShape(F32, {rows_per_part, cols});
  xla::Literal lit(slice_shape);
  for (int64_t i = 0; i < rows_per_part; ++i)
    for (int64_t j = 0; j < cols; ++j)
      lit.Set<float>({i, j}, host[(row_start + i) * cols + j]);

  return client.BufferFromHostLiteral(lit,
                                      dev->default_memory_space().value());
}


TEST(GpuSpmd, AddReduceTwoWay) {
  constexpr int64_t N = 1024, M = 1024;
  const xla::Shape  mat_shape = ShapeUtil::MakeShape(F32, {N, M});

  // ----------------- build HLO ------------------------------------------ //
  xla::XlaBuilder builder("add_allreduce_2gpu");

  // shard inputs row-wise : {devices=[2,1] 0,1}
  auto shard_proto =
      HloSharding::IotaTile({2, 1}).ToProto();
  builder.SetSharding(shard_proto);
  xla::XlaOp A = xla::Parameter(&builder, 0, mat_shape, "A");
  xla::XlaOp B = xla::Parameter(&builder, 1, mat_shape, "B");
  builder.ClearSharding();

  xla::XlaOp add = xla::Add(A, B);
  xla::XlaComputation add_fn = xla::CreateScalarAddComputation(F32, &builder);
  xla::XlaOp ar = xla::AllReduce(add, add_fn);  // across *partitions*
  xla::XlaOp root = xla::Tuple(&builder, {ar});

  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build(root));

  // ---------------- PJRT client / compilation -------------------------- //
  xla::GpuClientOptions opts;
  TF_ASSERT_OK_AND_ASSIGN(auto client_uptr, xla::GetStreamExecutorGpuClient(opts));
  auto& client = *client_uptr;

  ASSERT_GE(client.addressable_devices().size(), 2)
      << "Need at least two visible CUDA devices.";

  xla::CompileOptions copts;
  auto& eb = copts.executable_build_options;
  eb.set_num_replicas(1);
  eb.set_num_partitions(2);
  eb.set_use_spmd_partitioning(true);

  TF_ASSERT_OK_AND_ASSIGN(auto da,
      client.GetDefaultDeviceAssignment(/*repl=*/1, /*part=*/2));
  eb.set_device_assignment(da);

  auto* dbg = eb.mutable_debug_options();
  dbg->set_xla_gpu_multi_streamed_windowed_einsum(true);
  dbg->set_xla_gpu_enable_latency_hiding_scheduler(true);
  dbg->set_xla_gpu_dump_llvmir(true);

  TF_ASSERT_OK_AND_ASSIGN(auto exe, client.Compile(computation, copts));

  // ---------------- host data ------------------------------------------ //
  std::vector<float> host(N * M);
  for (int64_t i = 0; i < N * M; ++i)
    host[i] = (i < (N / 2) * M) ? 1.0f : 3.0f;  // rows 0..511 = 1, rows 512.. = 3

  std::vector<float> hostB(N * M);
  for (int64_t i = 0; i < N * M; ++i)
    hostB[i] = (i < (N / 2) * M) ? 2.0f : 4.0f;

  xla::PjRtDevice* dev0 = client.addressable_devices()[0];
  xla::PjRtDevice* dev1 = client.addressable_devices()[1];

  TF_ASSERT_OK_AND_ASSIGN(auto bufA0, VecToDevice(client, dev0, host,  mat_shape, 0));
  TF_ASSERT_OK_AND_ASSIGN(auto bufB0, VecToDevice(client, dev0, hostB, mat_shape, 0));
  TF_ASSERT_OK_AND_ASSIGN(auto bufA1, VecToDevice(client, dev1, host,  mat_shape, N/2));
  TF_ASSERT_OK_AND_ASSIGN(auto bufB1, VecToDevice(client, dev1, hostB, mat_shape, N/2));

  std::vector<std::vector<xla::PjRtBuffer*>> args = {
      {bufA0.get(), bufB0.get()},  // partition 0
      {bufA1.get(), bufB1.get()}   // partition 1
  };

  // ---------------- execute & verify ----------------------------------- //
  xla::ExecuteOptions exec_opts;
  TF_ASSERT_OK_AND_ASSIGN(auto outs, exe->Execute(args, exec_opts));

  ASSERT_EQ(outs.size(), 2);
  for (int p = 0; p < 2; ++p) {
    ASSERT_EQ(outs[p].size(), 1);
    TF_ASSERT_OK_AND_ASSIGN(auto lit, outs[p][0]->ToLiteralSync());
    float v = lit->DecomposeTuple()[0].Get<float>({0, 0});
    EXPECT_NEAR(v, 10.0f, 1e-4);   // (1+2)+(3+4) = 10
  }
}
} // namespace

} // namespace xla