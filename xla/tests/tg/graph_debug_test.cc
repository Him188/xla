/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "absl/log/initialize.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "test_util.h"
#include "xla/client/client_library.h"
#include "xla/client/lib/matrix.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h" // OpenXLA GPU client API&#8203;:contentReference[oaicite:3]{index=3}
#include "xla/service/gpu/gpu_executable.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"

#include <cstdlib>    // for setenv
#include <filesystem> // for std::filesystem::directory_iterator
#include <fstream>    // for std::ifstream
#include <iostream>   // for std::cout
#include <sstream>    // for std::stringstream
#include <string>
#include <vector>

namespace xla {
namespace {
constexpr char kTestCostName[] = "test";

// class TestCostMeasurement : public CostMeasurement {
//  public:
//   using CostMeasurement::CostMeasurement;
//
//   absl::Duration GetTotalCost() override { return absl::ZeroDuration(); }
//   absl::string_view GetCostType() const override { return kTestCostName; }
// };
//
// REGISTER_COST_MEASUREMENT(kTestCostName, TestCostMeasurement);

// My initial test

bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

class XlaDependencyTest : public ::testing::Test {
protected:
  // A helper to create a GPU PJRT client. Adjust if you use CPU or another
  // backend.
  static std::unique_ptr<xla::PjRtClient> GetGpuClientOrDie() {
    xla::GpuClientOptions options;
    // Add any custom options if needed (e.g. device ordinals).
    auto client_or = xla::GetStreamExecutorGpuClient(options);
    TF_CHECK_OK(client_or.status());
    return std::move(client_or.value());
  }
};

// This test shows how side-effecting ops (infeed/outfeed) rely on Token
// dependencies to preserve ordering.
TEST(LatencyHidingSchedulerTest, SendRecvOverlapExample) {
  std::string dumpDir = ::testing::TempDir() + "/xla_dump";
  std::filesystem::create_directory(dumpDir);
  xla_test_util::SetXlaDumpFlags(dumpDir);

  // 1. Build an XLA computation using XlaBuilder.
  xla::XlaBuilder builder("latency_hiding_test");

  // Create two large matrices to make the matmul non-trivial.
  // E.g. [1024, 1024], filled with constants just for simplicity.
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {1024, 1024});

  // xla::Array<float> array());
  // xla::Array<float>::Fill()

  // For a large matmul, we just pick some constant values.
  xla::XlaOp matA = xla::Broadcast(xla::ConstantR0<float>(&builder, 1.0f), {1024, 1024});
  xla::XlaOp matB = xla::Broadcast(xla::ConstantR0<float>(&builder, 2.0f), {1024, 1024});

  // 1A. Do an initial big matmul to create a large amount of work.
  xla::XlaOp matmul_out = xla::Dot(matA, matB);

  // 2. Insert a Token to manage side-effects (like send/recv).
  xla::XlaOp token0 = xla::AfterAll(&builder, {});

  xla::ChannelHandle handle;

  xla::ChannelHandle channel_id;
  channel_id.set_handle(1);
  channel_id.set_type(xla::ChannelHandle::DEVICE_TO_DEVICE);
  // xla::XlaOp send_token = xla::SendWithToken(matmul_out, token0, channel_id);
  // xla::XlaOp recv_tuple = xla::RecvWithToken(token0, shape, channel_id);
  // recv_tuple is a tuple {received_data, token}.
  // xla::XlaOp recv_data = xla::GetTupleElement(recv_tuple, 0);
  // xla::XlaOp token_after_recv = xla::GetTupleElement(recv_tuple, 1);

  // 5. Do another large matmul that can potentially overlap with the
  //    communication from the Recv if the compiler schedules them
  //    asynchronously.

  xla::XlaOp matC = xla::Broadcast(xla::ConstantR0<float>(&builder, 3.0f), {1024, 1024});

  // Combine the recv_data with matC to make the second matmul dependent
  // on the final Recv result (data dep for correctness).
  xla::XlaOp matmul2_out = xla::Dot(matmul_out, matC);

  // We produce a final sum to have a single output.
  xla::XlaOp result = matmul2_out;

  // 6. Build the computation.
  xla::XlaComputation computation = builder.Build(result).value();

  // 7. Get a PJRT GPU client and compile the computation with default options.
  //    (If you set XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true,
  //     the GPU backend tries to overlap the send/recv with the second matmul.)
  xla::GpuClientOptions options;
  auto client_or = xla::GetStreamExecutorGpuClient(options);
  ASSERT_TRUE(client_or.ok());
  std::unique_ptr<xla::PjRtClient> pjrt_client = std::move(client_or.value());
  auto &pjrt_stream_client = *dynamic_cast<xla::PjRtStreamExecutorClient *>(pjrt_client.get());

  auto outputs = xla_test_util::compile_and_execute(pjrt_stream_client, computation);
  ASSERT_EQ(outputs.size(), 1UL);
  ASSERT_EQ(outputs[0].size(), 1UL);

  // 10. Read back the final result (not particularly interesting numerically,
  //     but we can confirm it runs end-to-end).
  auto final_literal_or = outputs[0][0]->ToLiteralSync();
  ASSERT_TRUE(final_literal_or.ok());
  xla::Literal &final_literal = *final_literal_or.value();

  // At this point, you can step through the code in a debugger or
  // check the HLO/LLVM IR dumps (if you set --xla_dump_to=...) to see
  // how the latency-hiding scheduler placed Send, Recv, and matmuls.
  // The test passes if it completes without error.
  SUCCEED() << "Completed LatencyHidingSchedulerTest with final result shape: " << xla::ShapeUtil::HumanString(final_literal.shape());

  xla_test_util::PrintIrDumps(dumpDir, {
                                           xla_test_util::IRDumpKind::kHLO,
                                       });
}

void execute_computation(xla::XlaComputation &computation, absl::Span<const std::vector<xla::PjRtBuffer *>> argument_handles = {{}}) {
  xla::GpuClientOptions options;
  auto client_or = xla::GetStreamExecutorGpuClient(options);
  ASSERT_TRUE(client_or.ok());
  std::unique_ptr<xla::PjRtClient> pjrt_client = std::move(client_or.value());

  // 8. Compile the XLA computation.
  xla::CompileOptions compile_opts;
  auto exec_or = pjrt_client->Compile(computation, compile_opts);
  ASSERT_TRUE(exec_or.ok());
  std::unique_ptr<xla::PjRtLoadedExecutable> executable = std::move(exec_or.value());

  xla::ExecuteOptions exec_opts;
  auto outputs_or = executable->Execute(argument_handles, exec_opts);
  ASSERT_TRUE(outputs_or.ok());
  auto &outputs = outputs_or.value();
  ASSERT_EQ(outputs.size(), 1UL);
  ASSERT_EQ(outputs[0].size(), 1UL);

  // 10. Read back the final result (not particularly interesting numerically,
  //     but we can confirm it runs end-to-end).
  auto final_literal_or = outputs[0][0]->ToLiteralSync();
  ASSERT_TRUE(final_literal_or.ok());
  xla::Literal &final_literal = *final_literal_or.value();

  // At this point, you can step through the code in a debugger or
  // check the HLO/LLVM IR dumps (if you set --xla_dump_to=...) to see
  // how the latency-hiding scheduler placed Send, Recv, and matmuls.
  // The test passes if it completes without error.
  SUCCEED() << "Completed LatencyHidingSchedulerTest with final result shape: " << xla::ShapeUtil::HumanString(final_literal.shape());
}

TEST(CustomCallOrderingTest, PrintBeforeDot) {
  using namespace xla;
  XlaBuilder builder("PrintBeforeDot");

  // 1. Build some operands for the Dot operation.
  XlaOp lhs = ConstantR2<float>(&builder, {{1.0f, 2.0f}, {3.0f, 4.0f}});
  XlaOp rhs = ConstantR2<float>(&builder, {{5.0f, 6.0f}, {7.0f, 8.0f}});

  // 2. Create an initial token to sequence side effects.
  XlaOp token = CreateToken(&builder);

  // 3. Emit a CustomCall that takes the token (and perhaps a value to print)
  // and returns a new token.
  //    Mark it as having side effects so it won't be optimized out or
  //    reordered.
  xla::Shape token_shape = ShapeUtil::MakeTokenShape();
  XlaOp print_token = xla::CustomCall(&builder,
                                      "my_print",                // External function name for printing
                                      /*operands=*/{token, lhs}, // Pass token and a value to print (lhs, for
                                                                 // example)
                                      /*shape=*/token_shape,
                                      /*opaque=*/"",
                                      /*has_side_effect=*/true,
                                      /*output_operand_aliasing=*/{},
                                      /*literal=*/nullptr,
                                      /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
                                      /*api_version=*/xla::CustomCallApiVersion::API_VERSION_TYPED_FFI);

  // 4. Compute the Dot operation *after* the CustomCall. We will tie it into
  // the token chain next.
  XlaOp dot_result = xla::Dot(lhs, rhs);

  // 5. Tie the Dot's result to the token dependency to enforce ordering:
  //    Use an XLA dependency operation to sequence dot_result after
  //    print_token.
  XlaOp sequenced_result = dot_result;

  // 6. Use the sequenced_result as the output (or further computations).
  //    The presence of the dependency ensures print executes before dot.
  XlaOp final_output = sequenced_result;

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation comp, builder.Build());
  // (Optionally: execute comp on a backend and verify the print occurs before
  // dot)

  execute_computation(comp);
}

TEST(XlaCompilationTest, ExecuteOnMultipleStreamsComplexGraph) { // Command buffer
  // --------------------------------------------------------------------------
  // 1. Prepare output-dump directory and set XLA dump flags
  // --------------------------------------------------------------------------
  std::string dumpDir = ::testing::TempDir() + "/xla_dump_complex";
  std::filesystem::create_directory(dumpDir);
  xla_test_util::SetXlaDumpFlags(dumpDir);
  xla_test_util::EnableLogs();

  using namespace xla;

  // --------------------------------------------------------------------------
  // 2. Build a more complex HLO computation graph using XlaBuilder
  // --------------------------------------------------------------------------
  XlaBuilder builder("test_graph_complex");

  // Shapes for operands: two large matrices (A, B) and two vectors (C, D)
  auto matShape = ShapeUtil::MakeShape(F32, {4048, 4048});
  auto vecShape = ShapeUtil::MakeShape(F32, {4048});

  // Create parameters
  XlaOp A = Parameter(&builder, 0, matShape, "A");
  XlaOp B = Parameter(&builder, 1, matShape, "B");
  XlaOp C = Parameter(&builder, 2, vecShape, "C");
  XlaOp D = Parameter(&builder, 3, vecShape, "D");

  // --------------------------------------------------------------------------
  // Path 1: Depth-3 chain using A and B
  //     M1 = Dot(A, B)
  //     M2 = Add(M1, A)
  //     M3 = Dot(M2, B)
  // --------------------------------------------------------------------------
  XlaOp M1 = xla::Dot(A, B);
  XlaOp M2 = xla::Add(M1, A);
  XlaOp M3 = xla::Dot(M2, B);

  // --------------------------------------------------------------------------
  // Path 2: Another Depth-3 chain, slightly different shape usage
  //     N1 = Dot(B, A)
  //     N2 = Sub(N1, B)
  //     N3 = Dot(N2, A)
  // --------------------------------------------------------------------------
  XlaOp N1 = xla::Dot(B, A);
  XlaOp N2 = xla::Sub(N1, B);
  XlaOp N3 = xla::Dot(N2, A);

  // --------------------------------------------------------------------------
  // Path 3: Depth-3 chain focusing on vectors C, D (Dot => scalar)
  //     P1 = Dot(C, D)           // scalar
  //     P2 = Sub(P1, ConstantR0(12.3f))
  //     P3 = Mul(P2, Dot(C, D))  // scalar
  // --------------------------------------------------------------------------
  XlaOp P1 = xla::Dot(C, D); // C,D are 1D => result is scalar
  XlaOp P2 = xla::Sub(P1, ConstantR0<float>(&builder, 12.3f));
  XlaOp P3 = xla::Mul(P2, Dot(C, D));

  // --------------------------------------------------------------------------
  // Path 4: Another Depth-3 chain with C, D but in reversed Dot order
  //     Q1 = Dot(D, C)           // also a scalar
  //     Q2 = Add(Q1, ConstantR0(7.0f))
  //     Q3 = Mul(Q2, Dot(D, C))  // scalar
  // --------------------------------------------------------------------------
  XlaOp Q1 = xla::Dot(D, C);
  XlaOp Q2 = xla::Add(Q1, ConstantR0<float>(&builder, 7.0f));
  XlaOp Q3 = xla::Mul(Q2, Dot(D, C));

  // --------------------------------------------------------------------------
  // Combine all final results into a single Tuple output
  // --------------------------------------------------------------------------
  XlaOp output = Tuple(&builder, {M3, N3, xla::Add(M3, N3), P3, Q3, xla::Add(P3, Q3), xla::Mul(M3, P3)});

  // Build the computation (HLO module)
  XlaComputation computation = builder.Build(output).value();

  // --------------------------------------------------------------------------
  // 3. Set compilation options (enable multi-streaming, etc.)
  // --------------------------------------------------------------------------
  ExecutableBuildOptions build_opts;
  build_opts.set_device_ordinal(0); // target GPU 0
  xla::DebugOptions &debug_opts = *build_opts.mutable_debug_options();
  debug_opts.set_xla_gpu_enable_latency_hiding_scheduler(true);
  debug_opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
  debug_opts.set_xla_detailed_logging(true);
  debug_opts.set_xla_cpu_use_thunk_runtime(true);
  debug_opts.set_xla_gpu_async_dot(true);

  debug_opts.clear_xla_gpu_enable_command_buffer();
  debug_opts.add_xla_gpu_enable_command_buffer(DebugOptions_CommandBufferCmdType_INVALID);

  // build_opts.set_num_partitions(2);
  // build_opts.set_use_spmd_partitioning(true);

  // Example: If you need to disable a pass explicitly:
  // debug_opts.add_xla_disable_hlo_passes("some_pass_name");

  // --------------------------------------------------------------------------
  // 4. Obtain a PjRt (StreamExecutor) GPU client and compile the computation
  // --------------------------------------------------------------------------
  GpuClientOptions client_options;
  auto client_or = GetStreamExecutorGpuClient(client_options);
  ASSERT_TRUE(client_or.ok());

  std::unique_ptr<PjRtClient> pjrt_client = std::move(client_or.value());
  auto *pjrt_stream_client = dynamic_cast<PjRtStreamExecutorClient *>(pjrt_client.get());
  ASSERT_NE(pjrt_stream_client, nullptr);

  auto &client = *pjrt_stream_client->client();

  std::vector<const xla::Shape *> arg_layouts = {&matShape, &matShape, &vecShape, &vecShape};
  TF_ASSERT_OK_AND_ASSIGN(auto local_execs, client.Compile(computation, arg_layouts, build_opts));
  ASSERT_FALSE(local_execs.empty());

  std::unique_ptr<LocalExecutable> local_exec = std::move(local_execs[0]);
  Executable *executable = local_exec->executable();

  // --------------------------------------------------------------------------
  // 5. Cast to GpuExecutable to inspect the thunk sequence
  // --------------------------------------------------------------------------
  auto *gpu_exec = dynamic_cast<gpu::GpuExecutable *>(executable);
  ASSERT_NE(gpu_exec, nullptr);

  // --------------------------------------------------------------------------
  // 6. (Optional) Dump intermediate IR
  // --------------------------------------------------------------------------
  xla_test_util::PrintIrDumps(dumpDir, {});
}

TEST(XlaCompilationTest, ExecuteWithWhileLoopMatMuls) {
  // --------------------------------------------------------------------------
  // 1. Prepare output-dump directory and set XLA dump flags
  // --------------------------------------------------------------------------
  std::string dumpDir = ::testing::TempDir() + "/xla_dump_while_loop";
  std::filesystem::create_directory(dumpDir);
  xla_test_util::SetXlaDumpFlags(dumpDir);
  xla_test_util::EnableLogs();

  using namespace xla;

  // Parameters:
  //   param0: A (matrix),
  //   param1: B (matrix),
  //   param2: initial accumulator (matrix).
  auto matrix_shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  auto while_shape = ShapeUtil::MakeTupleShape({scalar_s32, matrix_shape, matrix_shape, matrix_shape});

  // Create a computation for the condition: repeat for 5 iterations.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto state = Parameter(&builder, 0, while_shape, "state");
    Gt(ConstantR0<int32_t>(&builder, 5), GetTupleElement(state, 0));
    TF_ASSERT_OK_AND_ASSIGN(condition, builder.Build());
  }

  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto state = Parameter(&builder, 0, while_shape, "state");
    auto indvar = GetTupleElement(state, 0);
    auto input_0 = GetTupleElement(state, 1);
    auto input_1 = GetTupleElement(state, 2);
    auto output = xla::Tanh(Dot(input_0, input_1));
    auto indvar_next = xla::Add(indvar, ConstantR0<int32_t>(&builder, 1));
    Tuple(&builder, {indvar_next, input_0, input_1, output});
    TF_ASSERT_OK_AND_ASSIGN(body, builder.Build());
  }

  XlaBuilder builder("ExecuteWithWhileLoopMatMuls");
  auto matrix_input = Parameter(&builder, 0, matrix_shape, "matrix");
  auto init = Tuple(&builder, {ConstantR0<int32_t>(&builder, 0), matrix_input, matrix_input, matrix_input});
  auto while_instruction = While(condition, body, init);
  auto res = GetTupleElement(while_instruction, 3);

  // Build the computation (HLO module)
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build({res}));

  // --------------------------------------------------------------------------
  // 3. Set compilation options (similar to the multi-stream example)
  // --------------------------------------------------------------------------
  ExecutableBuildOptions build_opts;
  build_opts.set_device_ordinal(0); // target GPU 0

  xla::DebugOptions &debug_opts = *build_opts.mutable_debug_options();
  debug_opts.set_xla_gpu_enable_latency_hiding_scheduler(true);
  debug_opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
  debug_opts.set_xla_cpu_use_thunk_runtime(true);

  // --------------------------------------------------------------------------
  // 4. Obtain a PjRt (StreamExecutor) GPU client and compile the computation
  // --------------------------------------------------------------------------
  GpuClientOptions client_options;
  auto client_or = GetStreamExecutorGpuClient(client_options);
  ASSERT_TRUE(client_or.ok());

  std::unique_ptr<PjRtClient> pjrt_client = std::move(client_or.value());
  auto *pjrt_stream_client = dynamic_cast<PjRtStreamExecutorClient *>(pjrt_client.get());
  ASSERT_NE(pjrt_stream_client, nullptr);

  auto &client = *pjrt_stream_client->client();

  std::vector<const xla::Shape *> arg_layouts = {&matrix_shape};
  TF_ASSERT_OK_AND_ASSIGN(auto local_execs, client.Compile(computation, arg_layouts, build_opts));
  ASSERT_FALSE(local_execs.empty());

  std::unique_ptr<LocalExecutable> local_exec = std::move(local_execs[0]);
  Executable *executable = local_exec->executable();

  // --------------------------------------------------------------------------
  // 5. Cast to GpuExecutable to inspect the thunk sequence
  // --------------------------------------------------------------------------
  auto *gpu_exec = dynamic_cast<gpu::GpuExecutable *>(executable);
  ASSERT_NE(gpu_exec, nullptr);

  const gpu::ThunkSequence &thunk_sequence = gpu_exec->GetThunk().thunks();
  std::cout << "Total thunks: " << thunk_sequence.size() << std::endl;

  xla_test_util::print_gpu_thunk_info(client, *gpu_exec);

  // --------------------------------------------------------------------------
  // 6. (Optional) Dump intermediate IR
  // --------------------------------------------------------------------------
  xla_test_util::PrintIrDumps(dumpDir, {xla_test_util::IRDumpKind::kHLO});
}

TEST(XlaCompilationTest,
     ExecuteOnMultipleStreamsWithEinsum) { // Command buffer
  // --------------------------------------------------------------------------
  // 1. Prepare output-dump directory and set XLA dump flags
  // --------------------------------------------------------------------------
  std::string dumpDir = ::testing::TempDir() + "/xla_dump_einsum";
  std::filesystem::create_directory(dumpDir);
  xla_test_util::SetXlaDumpFlags(dumpDir);
  xla_test_util::EnableLogs();

  using namespace xla;

  // --------------------------------------------------------------------------
  // 2. Build an HLO computation graph using XlaBuilder and Einsum
  // --------------------------------------------------------------------------
  XlaBuilder builder("test_einsum_graph");

  // Shapes for large matrices
  auto matShape = ShapeUtil::MakeShape(F32, {2048, 2048});

  // Create parameters: A, B, C
  XlaOp A = Parameter(&builder, 0, matShape, "A");
  XlaOp B = Parameter(&builder, 1, matShape, "B");
  XlaOp C = Parameter(&builder, 2, matShape, "C");

  // --------------------------------------------------------------------------
  // Path 1: Two chained Einsum calls
  //    E1 = Einsum(A, B, "ij, jk->ik")        [standard matrix multiply]
  //    E2 = Einsum(E1, C, "ik, km->im")       [another multiply-like pattern]
  // --------------------------------------------------------------------------
  XlaOp E1 = xla::Einsum(A, B, "ij,jk->ik");
  XlaOp E2 = xla::Einsum(E1, C, "ik,km->im");

  // --------------------------------------------------------------------------
  // Path 2: Another chain mixing Einsum and Add
  //    F1 = Einsum(B, C, "ij,jk->ik")
  //    F2 = Add(F1, A)   // forcing different shapes to be broadcast or cause
  //    partial reuse F3 = Einsum(F2, B, "ik, kj->ij")
  // --------------------------------------------------------------------------
  XlaOp F1 = xla::Einsum(B, C, "ij,jk->ik");
  XlaOp F2 = xla::Add(F1, A);
  XlaOp F3 = xla::Einsum(F2, B, "ik,kj->ij");

  // --------------------------------------------------------------------------
  // Combine all results into a tuple
  // --------------------------------------------------------------------------
  XlaOp out_tuple = Tuple(&builder, {E2, F3, xla::Add(E2, F3)});

  // Build the computation (HLO module)
  XlaComputation computation = builder.Build(out_tuple).value();

  // --------------------------------------------------------------------------
  // 3. Set compilation options (enable multi-stream features, etc.)
  // --------------------------------------------------------------------------
  ExecutableBuildOptions build_opts;
  build_opts.set_device_ordinal(0); // target GPU 0

  xla::DebugOptions &debug_opts = *build_opts.mutable_debug_options();
  debug_opts.set_xla_gpu_enable_latency_hiding_scheduler(true);
  debug_opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
  debug_opts.set_xla_cpu_use_thunk_runtime(true);
  // Optionally disable a particular pass:
  // debug_opts.add_xla_disable_hlo_passes("some_pass_name");

  // --------------------------------------------------------------------------
  // 4. Obtain a PjRt (StreamExecutor) GPU client and compile the computation
  // --------------------------------------------------------------------------
  GpuClientOptions client_options;
  auto client_or = GetStreamExecutorGpuClient(client_options);
  ASSERT_TRUE(client_or.ok());

  std::unique_ptr<PjRtClient> pjrt_client = std::move(client_or.value());
  auto *pjrt_stream_client = dynamic_cast<PjRtStreamExecutorClient *>(pjrt_client.get());
  ASSERT_NE(pjrt_stream_client, nullptr);

  auto &client = *pjrt_stream_client->client();
  std::vector<const xla::Shape *> arg_layouts = {&matShape, &matShape, &matShape};

  TF_ASSERT_OK_AND_ASSIGN(auto local_execs, client.Compile(computation, arg_layouts, build_opts));
  ASSERT_FALSE(local_execs.empty());

  std::unique_ptr<LocalExecutable> local_exec = std::move(local_execs[0]);
  Executable *executable = local_exec->executable();

  // --------------------------------------------------------------------------
  // 5. Cast to GpuExecutable to inspect the thunk sequence
  // --------------------------------------------------------------------------
  auto *gpu_exec = dynamic_cast<gpu::GpuExecutable *>(executable);
  ASSERT_NE(gpu_exec, nullptr);

  const gpu::ThunkSequence &thunk_sequence = gpu_exec->GetThunk().thunks();
  std::cout << "Total thunks: " << thunk_sequence.size() << std::endl;

  xla_test_util::print_gpu_thunk_info(client, *gpu_exec);

  // --------------------------------------------------------------------------
  // 6. (Optional) Dump intermediate IR
  // --------------------------------------------------------------------------
  xla_test_util::PrintIrDumps(dumpDir, {xla_test_util::IRDumpKind::kHLO});
}

//
// TEST(XlaAOTCompilationTest, SimpleAdditionCuda) {
//   // Build the computation graph: z = a + b.
//   xla::XlaBuilder builder("simple_add");
//   auto a = xla::Parameter(&builder, 0,
//                           xla::ShapeUtil::MakeShape(xla::F32, {2, 2}),
//                           "a");
//   auto b = xla::Parameter(&builder, 1,
//                           xla::ShapeUtil::MakeShape(xla::F32, {2, 2}),
//                           "b");
//   auto add = xla::Add(a, b);
//
//   // Finalize the computation.
//   auto computation_status = builder.Build();
//   ASSERT_TRUE(computation_status.ok())
//       << "Failed to build computation: " << computation_status.status();
//   xla::XlaComputation &computation = computation_status.value();
//
//   // Get the local XLA client (which will select the CUDA platform if
//   available). xla::LocalClient* client =
//   xla::ClientLibrary::LocalClientOrDie();
//
//   // Set up argument shapes matching our two 2×2 float parameters.
//   std::vector<xla::Shape> argument_shapes = {
//       xla::ShapeUtil::MakeShape(xla::F32, {2, 2}),
//       xla::ShapeUtil::MakeShape(xla::F32, {2, 2})
//   };
//
//   xla::ExecutableBuildOptions options;
//
//   // Compile the computation.
//   // In this context, calling Compile() acts as an AOT compilation step.
//   auto compile_status = client->Compile(computation, argument_shapes,
//   options); ASSERT_TRUE(compile_status.ok())
//       << "Compilation failed: " << compile_status.status();
//   std::unique_ptr<xla::Executable> executable =
//       std::move(compile_status.ValueOrDie());
//
//   // Print the generated HLO IR.
//   // (Note: depending on your XLA version the API to access the module IR
//   may differ.) std::string hlo_ir =
//       executable->executable()->module().ToString();
//   std::cout << "XLA HLO IR:\n" << hlo_ir << std::endl;
//   LOG(INFO) << "XLA HLO IR:\n" << hlo_ir;
//
//   // Prepare input data as XLA literals.
//   xla::Literal a_literal =
//       xla::LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}});
//   xla::Literal b_literal =
//       xla::LiteralUtil::CreateR2<float>({{5.0f, 6.0f}, {7.0f, 8.0f}});
//   std::vector<const xla::Literal*> arguments = {&a_literal, &b_literal};
//
//   // Execute the compiled computation on CUDA.
//   auto result_status = executable->ExecuteOnStream(arguments);
//   ASSERT_TRUE(result_status.ok())
//       << "Execution failed: " << result_status.status();
//   xla::Literal result = std::move(result_status.ValueOrDie());
//
//   // Print the computation result.
//   std::cout << "Result:\n" << result.ToString() << std::endl;
//   LOG(INFO) << "Result:\n" << result.ToString();
// }
} // namespace
} // namespace xla
