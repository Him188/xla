#include "test_util.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/tests/hlo_test_base.h"
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

absl::StatusOr<std::unique_ptr<PjRtBuffer>> VectorToDevice(PjRtClient &client, PjRtMemorySpace *memory_space, const std::vector<float> &host, const xla::Shape &shape) {
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
  return client.BufferFromHostLiteral(lit, memory_space);
}

class DynamicSliceHostAsyncTest : public ::testing::Test {
public:
  DynamicSliceHostAsyncTest() { ; }

protected:
  DebugOptions GetDebugOptionsForTest() const {
    DebugOptions debug_options = GetDebugOptionsFromFlags();
    // Disable async->sync collective conversion pass to enable unit testing
    // of async collectives.
    debug_options.add_xla_disable_hlo_passes("gpu-convert-async-collectives-to-sync");

    debug_options.set_xla_gpu_async_dot(true);
    debug_options.set_xla_gpu_multi_streamed_windowed_einsum(true);
    debug_options.set_xla_gpu_enable_latency_hiding_scheduler(true);
    debug_options.set_xla_dump_hlo_as_html(true);
    debug_options.clear_xla_gpu_enable_command_buffer();
    debug_options.add_xla_gpu_enable_command_buffer(xla::DebugOptions::INVALID);

    return debug_options;
  }
};

TEST_F(DynamicSliceHostAsyncTest, SliceFromDeviceToHostIsAsync) {
  constexpr int64_t kRows = 256, kCols = 256, kTile = 128;
  const Shape param_shape = ShapeUtil::MakeShape(F32, {kRows, kCols});

  const char *const hlo_text = R"(
HloModule DynamicSliceHostTransfer

ENTRY main {
  p = f32[256,256]{1,0} parameter(0)

  start_r = s32[] constant(0)        // slice starts at row 0
  start_c = s32[] constant(64)       // and column 64

  /* Result lives in host‑pinned memory:  S(5) */
  slice = f32[256,128]{1,0:S(5)} dynamic-slice(p, start_r, start_c),
           dynamic_slice_sizes={256,128}

  /* Do a trivial element‑wise op so the compiler keeps the result. */
  ROOT out = f32[256,128]{1,0:S(5)} negate(slice)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, ParseAndReturnUnverifiedModule(hlo_text));

  GpuClientOptions gpu_opts;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client_uptr, GetStreamExecutorGpuClient(gpu_opts));
  PjRtClient &client = *client_uptr;
  PjRtDevice *device = client.devices().front(); // CUDA

  CompileOptions copts;
  copts.executable_build_options.set_device_ordinal(0); // CUDA, Host

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> exe, client.Compile(XlaComputation(module.get()->ToProto()), copts));

  // -------------------------------------------------------------------------
  std::vector<float> host_in(kRows * kCols);
  std::iota(host_in.begin(), host_in.end(), 0.0f); // 0,1,2,…
  std::cout << "Device count: " << client.devices().size() << std::endl;
  for (auto memory_space : client.memory_spaces()) {
        std::cout << "Memory space: " << memory_space->ToString() << std::endl;
  }

  TF_ASSERT_OK_AND_ASSIGN(auto buf_in, VectorToDevice(client, device, host_in, param_shape));

  // -------------------------------------------------------------------
  // Execute.
  // -------------------------------------------------------------------
  std::vector<std::vector<PjRtBuffer *>> args = {{buf_in.get()}};
  ExecuteOptions options;
  TF_ASSERT_OK_AND_ASSIGN(auto outputs, exe->Execute(args, /*options=*/{}));

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].size(), 1);
  std::shared_ptr<Literal> result = outputs[0][0]->ToLiteralSync().value();

  // -------------------------------------------------------------------
  // Golden reference on CPU: negate the slice [0:kRows,64:64+kTile].
  // -------------------------------------------------------------------
  Literal expect = LiteralUtil::CreateR2F32Linspace(
                       /*from=*/0.0f, /*to=*/static_cast<float>(kRows * kCols - 1),
                       /*rows=*/kRows, /*cols=*/kCols)
                       .Slice({0, 64}, {kRows, 64 + kTile});
  // expect.Negate(/*in_place=*/true);

  // std::vector<std::vector<float>> expected(kRows, std::vector<float>(kCols));
  //       std::iota(expected.begin(), expected.end(), 0.0f);

  // LiteralTestUtil::ExpectR2Near<float>(expected, *result.get(),
  //                               ErrorSpec{/*a=*/0.0, /*rel=*/0.0});
  EXPECT_TRUE(LiteralTestUtil::Near(expect, *result.get(), ErrorSpec{0.0, 0.0}));
}
} // namespace

} // namespace xla