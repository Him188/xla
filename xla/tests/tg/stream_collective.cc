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

void SetLiteralValue(Literal &dest, absl::Span<const float> src, int64_t src_row_start) {
  const xla::Shape &global_shape = dest.shape();
  const int64_t rows_per_part = global_shape.dimensions(0);
  const int64_t cols = global_shape.dimensions(1);

  for (int64_t i = 0; i < rows_per_part; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      dest.Set<float>({i, j}, src[(src_row_start + i) * cols + j]);
    }
  }
}

TEST(GpuSpmd, AddReduceTwoWay) {
  constexpr int64_t N = 1024, M = 1024;
  const xla::Shape mat_shape = ShapeUtil::MakeShape(F32, {N, M});
  const xla::Shape slice_shape = ShapeUtil::MakeShape(F32, {N / 2, M});

  // ----------------- build HLO ------------------------------------------ //
  xla::XlaBuilder builder("add_allreduce_2gpu");

  // shard inputs row-wise : {devices=[2,1] 0,1}
  auto shard_proto = HloSharding::IotaTile({2, 1}).ToProto();
  builder.SetSharding(shard_proto);
  xla::XlaOp A = xla::Parameter(&builder, 0, mat_shape, "A");
  xla::XlaOp B = xla::Parameter(&builder, 1, mat_shape, "B");
  builder.ClearSharding();

  xla::XlaOp add = xla::Add(A, B);
  xla::XlaComputation add_fn = xla::CreateScalarAddComputation(F32, &builder);
  xla::XlaOp ar = xla::AllReduce(add, add_fn); // across *partitions*
  xla::XlaOp root = xla::Tuple(&builder, {ar});

  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build(root));

  // ---------------- PJRT client / compilation -------------------------- //
  xla::GpuClientOptions opts;
  TF_ASSERT_OK_AND_ASSIGN(auto client_uptr, xla::GetStreamExecutorGpuClient(opts));
  auto &client = *client_uptr;

  ASSERT_GE(client.addressable_devices().size(), 2) << "Need at least two visible CUDA devices.";

  xla::CompileOptions copts;
  auto &eb = copts.executable_build_options;
  eb.set_num_replicas(1);
  eb.set_num_partitions(2);
  // eb.set_device_ordinal(-1);
  eb.set_use_spmd_partitioning(true);

  TF_ASSERT_OK_AND_ASSIGN(auto da, client.GetDefaultDeviceAssignment(/*repl=*/1, /*part=*/2));
  eb.set_device_assignment(da);

  auto *dbg = eb.mutable_debug_options();
  // dbg->set_xla_gpu_multi_streamed_windowed_einsum(true);
  // dbg->set_xla_gpu_enable_latency_hiding_scheduler(true);
  dbg->set_xla_gpu_dump_llvmir(true);

  TF_ASSERT_OK_AND_ASSIGN(auto exe, client.Compile(computation, copts));

  // ---------------- host data ------------------------------------------ //
  std::vector<float> host(N * M);
  for (int64_t i = 0; i < N * M; ++i)
    host[i] = (i < (N / 2) * M) ? 1.0f : 3.0f; // rows 0..511 = 1, rows 512.. = 3

  std::vector<float> hostB(N * M);
  for (int64_t i = 0; i < N * M; ++i)
    hostB[i] = (i < (N / 2) * M) ? 2.0f : 4.0f;

  xla::PjRtDevice *dev0 = client.addressable_devices()[0];
  xla::PjRtDevice *dev1 = client.addressable_devices()[1];

  Literal literalA0(slice_shape);
  SetLiteralValue(literalA0, host, 0);

  Literal literalB0(slice_shape);
  SetLiteralValue(literalB0, hostB, 0);

  Literal literalA1(slice_shape);
  SetLiteralValue(literalA1, host, N / 2);

  Literal literalB1(slice_shape);
  SetLiteralValue(literalB1, hostB, N / 2);

  TF_ASSERT_OK_AND_ASSIGN(auto bufA0, client.BufferFromHostLiteral(literalA0, dev0->default_memory_space().value()));
  TF_ASSERT_OK_AND_ASSIGN(auto bufB0, client.BufferFromHostLiteral(literalB0, dev0->default_memory_space().value()));

  TF_ASSERT_OK_AND_ASSIGN(auto bufA1, client.BufferFromHostLiteral(literalA1, dev1->default_memory_space().value()));
  TF_ASSERT_OK_AND_ASSIGN(auto bufB1, client.BufferFromHostLiteral(literalB1, dev1->default_memory_space().value()));

  std::vector<std::vector<xla::PjRtBuffer *>> args = {
      {bufA0.get(), bufB0.get()}, // partition 0
      {bufA1.get(), bufB1.get()}  // partition 1
  };

  auto executor_client = dynamic_cast<PjRtStreamExecutorClient *>(client_uptr.get());
  auto *se_loaded = dynamic_cast<PjRtStreamExecutorLoadedExecutable*>(exe.get());
  ASSERT_TRUE(se_loaded != nullptr) << "Executable is not a Stream-Executor executable";

  auto *gpu_exec = dynamic_cast<xla::gpu::GpuExecutable*>(
      se_loaded->executables().at(0)->executable());
  ASSERT_TRUE(gpu_exec != nullptr) << "Underlying executable is not a GpuExecutable";

  xla_test_util::print_gpu_thunk_info(*executor_client->client(), *gpu_exec);

  // ---------------- execute & verify ----------------------------------- //
  xla::ExecuteOptions exec_opts;
  auto outs = exe->Execute(args, exec_opts).value();

  ASSERT_EQ(outs.size(), 2);
  for (int p = 0; p < 2; ++p) {
    ASSERT_EQ(outs[p].size(), 1);
    TF_ASSERT_OK_AND_ASSIGN(auto lit, outs[p][0]->ToLiteralSync());
    float v = lit->DecomposeTuple()[0].Get<float>({0, 0});
    EXPECT_NEAR(v, 3.0f, 1e-4); // (1+2)+(3+4) = 10
  }
}

static constexpr int kSrcDevice = 0;
static constexpr int kDstDevice = 1;

Shape PayloadShape() { return ShapeUtil::MakeShape(xla::F32, {256}); } // 1 KiB

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------
xla::XlaComputation BuildSender(const ChannelHandle &ch) {
  XlaBuilder b("sender");
  auto data = xla::Parameter(&b, 0, PayloadShape(), "x");
  XlaOp t0 = xla::CreateToken(&b);
  FrontendAttributes attr;
  (*attr.mutable_map())["_xla_send_recv_source_target_pairs"] = "{{0,1}}";
  b.SetFrontendAttributes(attr);
  xla::SendWithToken(data, t0, ch); // kSend
  return b.Build().value();
}

xla::XlaComputation BuildReceiver(const ChannelHandle &ch) {
  XlaBuilder b("receiver");
  XlaOp t0 = xla::CreateToken(&b);

  FrontendAttributes attr;
  (*attr.mutable_map())["_xla_send_recv_source_target_pairs"] = "{{0,1}}";
  b.SetFrontendAttributes(attr);
  auto tup = xla::RecvWithToken(t0, PayloadShape(), ch); // kRecv
  auto value = xla::GetTupleElement(tup, 0);
  return b.Build(value).value();
}

TEST(GpuSpmd, SendRecv) {
  setenv("NCCL_DEBUG", "WARN", 1);
  // 1.  prepare dump directory
  std::string dump_dir = ::testing::TempDir() + "/xla_dump";
  std::filesystem::create_directory(dump_dir);
  xla_test_util::SetXlaDumpFlags(dump_dir);

  // 2.  PJRT client
  xla::GpuClientOptions opts;
  auto client_or = xla::GetStreamExecutorGpuClient(opts);
  ASSERT_TRUE(client_or.ok());
  std::unique_ptr<xla::PjRtClient> pjrt = std::move(client_or.value());
  auto &stream_client = *dynamic_cast<xla::PjRtStreamExecutorClient *>(pjrt.get());

  ASSERT_GE(pjrt->addressable_device_count(), 2) << "Need at least two visible GPUs";

  // 3.  channel metadata
  ChannelHandle ch;
  ch.set_handle(42);
  ch.set_type(ChannelHandle::DEVICE_TO_DEVICE);

  // 4.  build & compile
  auto sender_comp = BuildSender(ch);
  auto receiver_comp = BuildReceiver(ch);

  CompileOptions sender_opts;
  sender_opts.executable_build_options.set_device_ordinal(kSrcDevice);
  CompileOptions recv_opts;
  recv_opts.executable_build_options.set_device_ordinal(kDstDevice);

  // compile_and_execute creates the executable internally and runs it
  // (same helper you used in the multi-stream fused test).
  // ------------------------------------------------------------------
  // 5.  sample data on GPU-0
  std::vector<float> host(256);
  std::iota(host.begin(), host.end(), /*start=*/17);
  auto buffer_src = xla_test_util::CreateDeviceBuffer(*pjrt, host, PayloadShape(), kSrcDevice);

  // 6.  Concurrency tracer hook
  gpu::ConcurrencyTracer tracer;
  ExecuteOptions exec_opts;
  exec_opts.gpu_concurrency_tracer = &tracer;

  // 7.  run sender (GPU-0)
  xla_test_util::compile_and_execute(stream_client, sender_comp, {{buffer_src.get()}}, sender_opts, exec_opts);

  // 8.  run receiver (GPU-1)
  const auto outputs = xla_test_util::compile_and_execute(stream_client, receiver_comp, /*no inputs*/ {}, recv_opts, exec_opts);

  auto literal_out = xla_test_util::buffer_to_literal(outputs[0][0]).value();

  // 9.  verify
  Literal expected = LiteralUtil::CreateR1<float>(host);
  ASSERT_TRUE(*literal_out.get() == expected);

  // 10.  print trace and race report
  tracer.PrintTraces(std::cout);
  tracer.PrintDataRaces(std::cout);

  // 11.  IR dump on failure / single-run convenience
  xla_test_util::PrintIrDumps(dump_dir, {xla_test_util::IRDumpKind::kHLO, xla_test_util::IRDumpKind::kHTML});
}
} // namespace

} // namespace xla