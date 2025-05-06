#include "test_util.h"
#include "xla/backends/gpu/runtime/concurrency_trace.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/tsl/lib/core/status_test_util.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

namespace xla {
namespace {

TEST(XlaCompilationTest, ExecuteOnMultpleStreamsFused) {
  std::string dumpDir = ::testing::TempDir() + "/xla_dump";
  std::filesystem::create_directory(dumpDir);
  xla_test_util::SetXlaDumpFlags(dumpDir);

  constexpr int device_ordinal = 0;

  using namespace xla;
  auto test_fun = [](int round_number) {
    XlaBuilder builder("test_graph");

    constexpr int64_t N = 4048;
    constexpr int64_t M = 4048;

    Shape matShape = ShapeUtil::MakeShape(F32, {N, M});

    // Create parameters
    XlaOp A = Parameter(&builder, 0, matShape, "A");
    XlaOp B = Parameter(&builder, 1, matShape, "B");

    auto ASlice = A;
    // auto ASlice = Slice(A, {2024, 2024}, {4048, 4048}, {1, 1});
    auto BSlice = B;
    // auto BSlice = Slice(B, {2024, 2024}, {4048, 4048}, {1, 1});

    XlaOp a_dot_b = Dot(ASlice, BSlice);

    XlaOp heavy = a_dot_b;
    for (int i = 0; i < 1; ++i) {
      heavy = Dot(heavy, BSlice);
      heavy = heavy - a_dot_b;
    }

    XlaOp matmul = heavy;
    XlaOp output = Tuple(&builder, {matmul});
    XlaComputation computation = builder.Build(output).value();

    // Set debug options
    CompileOptions compile_options;
    ExecutableBuildOptions &build_opts = compile_options.executable_build_options;
    build_opts.set_device_ordinal(device_ordinal);

    DebugOptions &debug_opts = *build_opts.mutable_debug_options();
    build_opts.mutable_debug_options()->add_xla_disable_hlo_passes();
    build_opts.mutable_debug_options()->set_xla_gpu_enable_latency_hiding_scheduler(true);
    debug_opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
    debug_opts.set_xla_cpu_use_thunk_runtime(true);
    debug_opts.set_xla_gpu_async_dot(true); // !!
    debug_opts.set_xla_dump_hlo_as_html(true);

    debug_opts.clear_xla_gpu_enable_command_buffer();
    debug_opts.add_xla_gpu_enable_command_buffer(DebugOptions_CommandBufferCmdType_INVALID);

    // Get PjRt client
    GpuClientOptions options;
    auto client_or = GetStreamExecutorGpuClient(options);
    ASSERT_TRUE(client_or.ok());
    std::unique_ptr<PjRtClient> pjrt_client = std::move(client_or.value());
    auto &pjrt_stream_client = *dynamic_cast<PjRtStreamExecutorClient *>(pjrt_client.get());

    // Create buffers for the input parameters
    std::vector hostA(N * M, 1.0f);
    std::vector hostB(N * M, 1.01f);

    std::unique_ptr<PjRtBuffer> bufferA = xla_test_util::CreateDeviceBuffer(*pjrt_client, hostA, matShape, device_ordinal);
    std::unique_ptr<PjRtBuffer> bufferB = xla_test_util::CreateDeviceBuffer(*pjrt_client, hostB, matShape, device_ordinal);

    gpu::ConcurrencyTracer tracer;
    ExecuteOptions execute_options;
    execute_options.gpu_synthetic_bug_options.wait_for_streams_thunk = true;
    execute_options.gpu_concurrency_tracer = &tracer;
    const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> outputs =
        xla_test_util::compile_and_execute(pjrt_stream_client, computation, {{bufferA.get(), bufferB.get()}}, compile_options, execute_options);

    // Print output for this round
    auto literal = xla_test_util::buffer_to_literal(outputs[0][0]).value();
    std::cout << "Shape: " << ShapeUtil::HumanString(literal->shape()) << std::endl;
    auto tuple = literal->DecomposeTuple();
    std::cout << "tuple size: " << tuple.size() << std::endl;

    constexpr auto expected = 1 + 2 + 2 * 100;
    std::cout << "Round " << round_number << " Value: " << tuple[0].Get<float>({0, 0}) << std::endl;

    tracer.PrintTraces(std::cout);
    tracer.PrintDataRaces(std::cout);
    ASSERT_NEAR(tuple[0].Get<float>({0, 0}), 16704678, 1e-7); // 12529370 if sliced
  };

  constexpr int num_runs = 1;

  for (int i = 0; i < num_runs; ++i) {
    test_fun(i);
  }
  if constexpr (num_runs == 1) {
    xla_test_util::PrintIrDumps(dumpDir, {
                                             // xla_test_util::IRDumpKind::kHLO,
                                             xla_test_util::IRDumpKind::kHTML,
                                         });
  }
}

TEST(XlaCompilationTest, SlicedCopy) {
  std::string dumpDir = ::testing::TempDir() + "/xla_dump";
  std::filesystem::create_directory(dumpDir);
  xla_test_util::SetXlaDumpFlags(dumpDir);

  auto test_fun = [](int round_number) {
    XlaBuilder b("sliced_copy_race");
    Shape shape = ShapeUtil::MakeShape(F32, {1 << 20});
    auto p = Parameter(&b, 0, shape, "p");

    // Slice A: bytes 0 .. 2MiB
    auto slice_a = Slice(p, {0}, {1 << 19}, {1});
    // Slice B: bytes 1MiB .. 3MiB   ← deliberate 1MiB overlap
    auto slice_b = Slice(p, {1 << 18}, {3 << 18}, {1});

    // Dummy use to keep both slices live until after concat.
    auto concat = ConcatInDim(&b, {slice_a, slice_b}, /*dimension=*/0);

    // Overwrite tensor so that a race is observable.
    auto sentinel = Broadcast(ConstantR0<float>(&b, 3.14f), {1 << 20});
    auto out = Add(concat, sentinel);

    XlaComputation computation = b.Build(out).value();

    // Set debug options
    CompileOptions compile_options;
    ExecutableBuildOptions &build_opts = compile_options.executable_build_options;
    build_opts.set_device_ordinal(0); // target GPU 0

    DebugOptions &debug_opts = *build_opts.mutable_debug_options();
    build_opts.mutable_debug_options()->add_xla_disable_hlo_passes();
    build_opts.mutable_debug_options()->set_xla_gpu_enable_latency_hiding_scheduler(true);
    build_opts.mutable_debug_options()->set_xla_gpu_enable_dynamic_slice_fusion(false);
    build_opts.mutable_debug_options()->set_xla_gpu_enable_host_memory_offloading(true);
    debug_opts.set_xla_dump_hlo_as_html(true);

    debug_opts.clear_xla_gpu_enable_command_buffer();
    debug_opts.add_xla_gpu_enable_command_buffer(DebugOptions_CommandBufferCmdType_INVALID);

    // Get PjRt client
    GpuClientOptions options;
    auto client_or = GetStreamExecutorGpuClient(options);
    ASSERT_TRUE(client_or.ok());
    std::unique_ptr<PjRtClient> pjrt_client = std::move(client_or.value());
    auto &pjrt_stream_client = *dynamic_cast<PjRtStreamExecutorClient *>(pjrt_client.get());

    // Create buffers for the input parameters
    std::vector hostA(1 << 20, 1.0f);

    std::unique_ptr<PjRtBuffer> bufferA = xla_test_util::CreateDeviceBuffer(*pjrt_client, hostA, shape);

    const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> outputs =
        xla_test_util::compile_and_execute(pjrt_stream_client, computation, {{bufferA.get()}}, compile_options);

    // Print output for this round
    auto literal = xla_test_util::buffer_to_literal(outputs[0][0]).value();
    std::cout << "Shape: " << ShapeUtil::HumanString(literal->shape()) << std::endl;
    // auto tuple = literal->DecomposeTuple();
    // std::cout << "tuple size: " << tuple.size() << std::endl;

    constexpr auto expected = 1 + 2 + 2 * 100;
    double value = literal->GetAsDouble({0}).value();
    std::cout << "Round " << round_number << " Value: " << value << std::endl;
    ASSERT_TRUE(std::abs(value - 1.67047e+07) < 1e07);
  };

  constexpr int num_runs = 1;

  for (int i = 0; i < num_runs; ++i) {
    test_fun(i);
  }
  if constexpr (num_runs == 1) {
    xla_test_util::PrintIrDumps(dumpDir, {
                                             xla_test_util::IRDumpKind::kHLO,
                                             xla_test_util::IRDumpKind::kHTML,
                                         });
  }
}

TEST(XlaCompilationTest, ExecuteOnMultpleStreamsWhile) {
  std::string dumpDir = ::testing::TempDir() + "/xla_dump";
  std::filesystem::create_directory(dumpDir);
  xla_test_util::SetXlaDumpFlags(dumpDir);

  using namespace xla;
  auto test_fun = [] {
    // 1. Create XlaBuilder
    XlaBuilder builder("complex_graph_test");

    const int64_t N = 2048;
    const int64_t M = 2048;
    const int64_t V = 2048;

    // Define shapes
    Shape matShape = ShapeUtil::MakeShape(F32, {N, M});
    Shape vecShape = ShapeUtil::MakeShape(F32, {V});
    Shape scalar_s32 = ShapeUtil::MakeShape(S32, {});

    // Create parameters
    XlaOp A = Parameter(&builder, 0, matShape, "A");
    XlaOp B = Parameter(&builder, 1, matShape, "B");
    XlaOp C = Parameter(&builder, 2, vecShape, "C");
    XlaOp D = Parameter(&builder, 3, vecShape, "D");

    // 2. Define several large matmul ops that could run in parallel
    XlaOp dotAB_1 = Dot(A, B); // Large matmul
    XlaOp dotAB_2 = Dot(A, B); // Another large matmul
    XlaOp dotBA = Dot(B, A);   // Different shape pattern, still large

    // A few elementwise ops that might be fused, but we combine them
    // with control-flow to break large fusions.
    XlaOp sum1 = Add(dotAB_1, dotAB_2);
    XlaOp sum2 = Add(dotBA, sum1);
    XlaOp sum3 = Add(sum2, Broadcast(C, {N}));

    // 3. Introduce a While loop to further inhibit single-kernel fusion.
    //    We'll iterate a few times, applying more matmul-like operations.
    //    (This is a simple integer loop that repeatedly does a Dot.)

    // Loop-carried tuple structure: (iter_count, partial_mat)
    XlaOp zero = ConstantR0<int32_t>(&builder, 0);

    // Initialize the loop tuple with (0, sum3).

    {
      const auto loop_itr_shape = ShapeUtil::MakeTupleShape({scalar_s32, ShapeUtil::MakeShape(F32, {N, M}), ShapeUtil::MakeShape(F32, {N, M})});

      // Build condition function: while (iter < loop_limit)
      auto cond_comp = [&scalar_s32, &loop_itr_shape] {
        XlaBuilder cond_builder("while_cond");
        const auto loop_limit = ConstantR0<int32_t>(&cond_builder, 30);
        const XlaOp param = Parameter(&cond_builder, 0, loop_itr_shape, "loop_param");
        const XlaOp iter = GetTupleElement(param, 0);
        const XlaOp lessThan = Lt(iter, loop_limit);
        return cond_builder.Build(lessThan).value();
      }();

      // Build body function
      auto body_comp = [&scalar_s32, &loop_itr_shape] {
        XlaBuilder body_builder("while_body");
        const XlaOp param_body = Parameter(&body_builder, 0, loop_itr_shape, "loop_param_body");

        const XlaOp iter = GetTupleElement(param_body, 0);
        const XlaOp mat_body = GetTupleElement(param_body, 1);
        const XlaOp data = GetTupleElement(param_body, 1);
        const XlaOp dotAB_1 = GetTupleElement(param_body, 2);

        // Example: do another Dot inside the loop and combine
        const XlaOp step_dot = Dot(mat_body, data);
        XlaOp next_iter = Add(iter, ConstantR0<int32_t>(&body_builder, 1));
        XlaOp updated = Sub(step_dot, dotAB_1); // Some random combination

        // Return the updated tuple
        const XlaOp new_tuple = Tuple(&body_builder, {next_iter, updated, dotAB_1});
        return body_builder.Build(new_tuple).value();
      }();

      /*
      *Thunk 0: Kind=kCopy, launches on 0
      Thunk 1: Kind=kWaitForStreams, launches on 0, waits for stream  0
      Thunk 2: Kind=kGemm, launches on 2
      Thunk 3: Kind=kWaitForStreams, launches on 0, waits for stream  0
      Thunk 4: Kind=kGemm, launches on 1
      Thunk 5: Kind=kKernel, launches on 0
      Thunk 6: Kind=kWaitForStreams, launches on 0, waits for stream  1
      Thunk 7: Kind=kWaitForStreams, launches on 0, waits for stream  2
      Thunk 8: Kind=kKernel, launches on 0
      Thunk 9: Kind=kCopy, launches on 0
      Thunk 10: Kind=kWhile, launches on 0
      Thunk 11: Kind=kKernel, launches on 0
      */

      // Now create the While op
      XlaOp loop_init = Tuple(&builder, {zero, sum3, dotAB_1});
      XlaOp loop_result = While(cond_comp, body_comp, loop_init);
      // Finally, extract the final matrix from the loop output
      XlaOp final_mat = GetTupleElement(loop_result, 1);

      // 4. Perform some more ops outside the loop:
      //    For instance, combine final_mat with dotAB_1 again
      XlaOp after_loop = Add(final_mat, Dot(A, B)); // Another big matmul

      // Also do a vector add so there's at least one separate kernel
      XlaOp elem_add = Add(C, D);

      // 5. Combine everything in a final Tuple to produce a single HLO
      // output.
      XlaOp final_output = Tuple(&builder, {dotAB_1, dotAB_2, dotBA, sum1, sum2, sum3, final_mat, after_loop, elem_add});

      // Build the computation
      auto computation_or = builder.Build(final_output);
      TF_ASSERT_OK(computation_or.status());
      XlaComputation computation = std::move(computation_or.value());

      // 6. Set up compilation options (multi-stream & concurrency-friendly)
      CompileOptions compile_options;
      ExecutableBuildOptions &build_opts = compile_options.executable_build_options;
      build_opts.set_device_ordinal(0); // GPU 0
      DebugOptions &debug_opts = *build_opts.mutable_debug_options();
      debug_opts.set_xla_gpu_enable_latency_hiding_scheduler(true);
      debug_opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
      debug_opts.set_xla_cpu_use_thunk_runtime(true);
      debug_opts.set_xla_gpu_async_dot(true);
      debug_opts.clear_xla_gpu_enable_command_buffer();
      debug_opts.add_xla_gpu_enable_command_buffer(DebugOptions_CommandBufferCmdType_INVALID);
      debug_opts.set_xla_dump_hlo_as_html(true);
      debug_opts.set_xla_dump_hlo_as_dot(true);

      // Optional: disable or enable more passes to see concurrency
      // debug_opts.add_xla_disable_hlo_passes("some_pass_name_if_needed");

      // 7. Get PjRtClient
      GpuClientOptions options;
      auto client_or = GetStreamExecutorGpuClient(options);
      std::unique_ptr<PjRtClient> pjrt_client = std::move(client_or.value());

      // Prepare host data
      std::vector<float> hostA(N * M, 1.0f);
      std::vector<float> hostB(N * M, 1.01f);
      std::vector<float> hostC(V, 3.0f);
      std::vector<float> hostD(V, 4.0f);

      // Convert host data to PjRtBuffers
      std::unique_ptr<PjRtBuffer> bufferA = xla_test_util::CreateDeviceBuffer(*pjrt_client, hostA, matShape);
      std::unique_ptr<PjRtBuffer> bufferB = xla_test_util::CreateDeviceBuffer(*pjrt_client, hostB, matShape);
      std::unique_ptr<PjRtBuffer> bufferC = xla_test_util::CreateDeviceBuffer(*pjrt_client, hostC, vecShape);
      std::unique_ptr<PjRtBuffer> bufferD = xla_test_util::CreateDeviceBuffer(*pjrt_client, hostD, vecShape);

      // 8. Compile and execute
      auto outputs = xla_test_util::compile_and_execute(*dynamic_cast<PjRtStreamExecutorClient *>(pjrt_client.get()), computation,
                                                        {{bufferA.get(), bufferB.get(), bufferC.get(), bufferD.get()}}, compile_options);

      // 9. Inspect results
      ASSERT_FALSE(outputs.empty());
      ASSERT_FALSE(outputs[0].empty());

      auto literal_or = xla_test_util::buffer_to_literal(outputs[0][0]);
      TF_ASSERT_OK(literal_or.status());

      auto final_literal = std::move(literal_or.value());
      std::cout << "Final Tuple shape: " << ShapeUtil::HumanString(final_literal->shape()) << std::endl;

      auto decomposed = final_literal->DecomposeTuple();
      std::cout << "Number of tuple elements = " << decomposed.size() << std::endl;

      // Simple check on one element
      float val0 = decomposed[0].Get<float>({0, 0});
      std::cout << "First matmul[0,0] = " << val0 << std::endl;

      // std::vector<ExecutionInput> vec;
      // vec.push_back({})
      // auto execution_output = local_exec->Run(std::move(vec),
      // run_opts).value(); auto device_memory_base =
      // execution_output.Result().buffer({0});

      // std::cout << "outputs.size=" << outputs.size() << " " <<
      // "outputs[0].size=" << outputs[0].size() << std::endl;

      auto literal = xla_test_util::buffer_to_literal(outputs[0][0]).value();
      auto tuple = literal->DecomposeTuple();
      std::cout << "Result tuple size: " << tuple.size() << std::endl;

      constexpr auto expected = 1 + 2 + 2 * 100;
      std::cout << "Value: " << tuple[0].Get<float>({0, 0}) << std::endl;
      ASSERT_EQ(2068, tuple[0].Get<float>({0, 0}));
      // ASSERT_TRUE(std::abs(tuple[0].Get<float>({0, 0}) - 1.67047e+07) < 1e07);
      // ASSERT_EQ(1.67047e+07, tuple[0].Get<float>({0, 0}));
      // ASSERT_EQ(8096, tuple[1].Get<float>({0, 0}));
      // ASSERT_EQ(8096, tuple[2].Get<float>({0, 0}));
      // ASSERT_EQ(8096, tuple[3].Get<float>({0, 0}));
      // ASSERT_EQ(3 + 4, tuple[4].Get<float>({0}));
    };
  };

  for (int i = 0; i < 10; ++i) {
    test_fun();
  }
  xla_test_util::PrintIrDumps(dumpDir, {
                                           xla_test_util::IRDumpKind::kHTML,
                                       });
}

} // namespace
} // namespace xla