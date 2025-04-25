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
  xla::XlaOp A = xla::Parameter(&builder, /*parameter_number=*/0, mat_shape, "A");
  xla::XlaOp B = xla::Parameter(&builder, 1, mat_shape, "B");
  xla::XlaOp dot = xla::Dot(A, B);

  // Scalar add computation for the reduction.
  xla::XlaComputation add = xla::CreateScalarAddComputation(xla::F32, &builder);

  // One replica‑group containing both replicas.
  xla::ReplicaGroup rg;
  rg.add_replica_ids(0);
  rg.add_replica_ids(1);

  xla::XlaOp ar = xla::AllReduce(dot, add, /*replica_groups=*/{rg});
  xla::XlaOp root = xla::Tuple(&builder, {ar});
  TF_ASSERT_OK_AND_ASSIGN(xla::XlaComputation computation, builder.Build(root));

  // -------------------------------------------------------------------------
  // PJRT client and compilation.
  // -------------------------------------------------------------------------
  GpuClientOptions gpu_opts;
  gpu_opts.mock_gpu_topology = "1x1x2";
  gpu_opts.enable_mock_nccl = true;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client_uptr, GetStreamExecutorGpuClient(gpu_opts));
  PjRtClient &client = *client_uptr;

  CompileOptions copts;
  ExecutableBuildOptions &ebopts = copts.executable_build_options;
  ebopts.set_num_replicas(2);
  ebopts.set_device_ordinal(0); // First GPU as target device (both replicas run here if single‑GPU)

  DebugOptions *dbg = ebopts.mutable_debug_options();
  dbg->set_xla_gpu_async_dot(true);
  dbg->set_xla_gpu_multi_streamed_windowed_einsum(true);
  dbg->set_xla_gpu_enable_latency_hiding_scheduler(true);
  dbg->set_xla_dump_hlo_as_html(true);
  dbg->clear_xla_gpu_enable_command_buffer();
  dbg->add_xla_gpu_enable_command_buffer(xla::DebugOptions::INVALID);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> exe, client.Compile(computation, copts));

  // -------------------------------------------------------------------------
  // Prepare per‑replica inputs.
  // -------------------------------------------------------------------------
  std::vector<float> hostA0(kN * kM, 1.0f);
  std::vector<float> hostB0(kN * kM, 1.01f);
  std::vector<float> hostA1(kN * kM, 2.0f);
  std::vector<float> hostB1(kN * kM, 0.99f);

  PjRtDevice *device0 = client.devices()[0];
  PjRtDevice *device1 = client.devices().size() > 1 ? client.devices()[1] : client.devices()[0]; // Fallback: same GPU.

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
  TF_ASSERT_OK_AND_ASSIGN(auto outputs, exe->Execute(args, /*options=*/{}));

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
} // namespace

} // namespace xla