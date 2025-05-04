#include "test_util.h"
#include "xla/backends/gpu/runtime/concurrency_trace.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/tsl/lib/core/status_test_util.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

namespace xla {
namespace {

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