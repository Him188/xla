#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "absl/log/initialize.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/wait_for_streams_thunk.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"  // OpenXLA GPU client API&#8203;:contentReference[oaicite:3]{index=3}
#include "xla/service/gpu/gpu_executable.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla::gpu {
class GpuExecutable;
}
namespace xla {

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


#include <cstdlib>     // for setenv
#include <filesystem>  // for std::filesystem::directory_iterator
#include <fstream>     // for std::ifstream
#include <iostream>    // for std::cout
#include <sstream>     // for std::stringstream
#include <string>
#include <vector>

namespace xla_test_util {

struct XlaDumpIr {
  XlaDumpIr();
};

// Which kinds of IR we want to dump
enum class IRDumpKind {
  kHLO,
  kLLVM,
  kMLIR,
  kPTX,
};

/**
 * Sets XLA_FLAGS for IR dumping, writing to `dump_dir`.
 *
 * For example:
 *   SetXlaDumpFlags("/tmp/xla_dumps", {IRDumpKind::kHLO, IRDumpKind::kLLVM});
 *
 * This will produce text dumps of HLO IR and LLVM IR.
 * Adjust or remove flags as needed for your version of XLA.
 */
inline void SetXlaDumpFlags(const std::string& dump_dir) {
  // Mandatory directory setting
  std::string flags = "--xla_dump_to=" + dump_dir + " --xla_dump_hlo_as_text --xla_gpu_dump_llvmir";
  // Example: you might also want to enable other dump flags, e.g.:
  // flags += " --xla_dump_hlo_module_reproducer";
  // or set per pass dumps, etc.
  // Adjust to your version’s capabilities.

  std::cout << "Enabling XLA dumps in " << dump_dir << std::endl;

  // Set environment variable so that XLA picks them up at compile time
  setenv("XLA_FLAGS", flags.c_str(), /*overwrite=*/1);

  std::string tf_flags = "--tf_xla_cpu_global_jit --tf_dump_graph_prefix=" + dump_dir;
  setenv("TF_XLA_FLAGS", tf_flags.c_str(), /*overwrite=*/1);
}

inline void EnableLogs() {
  setenv("TF_CPP_MIN_VLOG_LEVEL", "0", 1);
  setenv("TF_CPP_MAX_VLOG_LEVEL", "10", 1);
  setenv("TF_CPP_VMODULE", "xla_service=2,xla_compilation_cache=1,gpu_compiler=3,command_buffer_thunk=3,async_wrapper.cc=3", 1);
}

/**
 * Prints any files in `dump_dir` that match recognized IR dump file suffixes:
 * .txt (HLO), .mlir, .ll (LLVM), .ptx
 */
inline void PrintIrDumps(const std::string& dump_dir, const std::vector<IRDumpKind>& kinds) {
  for (const auto& entry : std::filesystem::directory_iterator(dump_dir)) {
    std::string path = entry.path().string();
    // Simple suffix checking. If your C++ standard is >= 20, you could use
    // std::string::ends_with(...)
    auto ends_with = [&](const std::string& s, const std::string& suffix) {
      if (s.size() < suffix.size()) return false;
      return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    auto need_kind = [&](const IRDumpKind kind) { return std::find(kinds.begin(), kinds.end(), kind) != kinds.end(); };

    if ((ends_with(path, ".txt") && need_kind(IRDumpKind::kHLO)) || (ends_with(path, ".mlir") && need_kind(IRDumpKind::kMLIR)) ||
        (ends_with(path, ".ll") && need_kind(IRDumpKind::kLLVM)) || (ends_with(path, ".ptx") && need_kind(IRDumpKind::kPTX))) {
      std::ifstream file(path);
      if (!file.is_open()) {
        continue;
      }
      std::stringstream buffer;
      buffer << file.rdbuf();
      std::cout << "\n--- IR Dump: " << entry.path().filename().string()  // just the file name
                << " ---\n";
      std::cout << buffer.str() << std::endl;
    }
  }
}

// Creates a buffer on the device from host data.
inline std::unique_ptr<xla::PjRtBuffer> CreateDeviceBuffer(xla::PjRtClient& client, absl::Span<const float> host_data, const xla::Shape& shape) {
  // Check shape sizes match:
  size_t expected_size = 1;
  for (auto dim : shape.dimensions()) {
    expected_size *= dim;
  }
  CHECK_EQ(expected_size, host_data.size()) << "Host data size must match shape size.";

  // Create the device buffer (for static shapes).
  // Note: For dynamic shapes, we also pass the dynamic dimension sizes.
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> buffer_or = client.BufferFromHostBuffer(
      /*data=*/host_data.data(),
      /*type=*/xla::PrimitiveType::F32,
      /*dims=*/shape.dimensions(),
      /*byte_strides=*/absl::nullopt,  // row-major by default
      xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
      [] {},
      client.devices()[0]->default_memory_space().value(),
      nullptr);

  TF_CHECK_OK(buffer_or.status());
  return std::move(buffer_or.value());
}

inline void print_gpu_thunk_info(const xla::LocalClient& client, xla::gpu::GpuExecutable& gpu_exec) {
  using namespace xla;

  const gpu::ThunkSequence& thunk_sequence = gpu_exec.GetThunk().thunks();
  std::cout << "Total thunks: " << thunk_sequence.size() << std::endl;

  int idx = 0;
  for (const std::unique_ptr<gpu::Thunk>& thunk_ptr : thunk_sequence) {
    const gpu::Thunk* thunk = thunk_ptr.get();
    std::string_view kind = gpu::Thunk::KindToString(thunk->kind());
    uint64_t stream_id = thunk->execution_stream_id().value();  // fallback to 0 if unset

    std::cout << "Thunk " << idx++ << ": Kind=" << kind << ", launches on " << stream_id;

    if (auto* command_buffer_thunk = const_cast<gpu::CommandBufferThunk*>(dynamic_cast<const gpu::CommandBufferThunk*>(thunk))) {
      std::cout << command_buffer_thunk->ToString(0) << std::endl;
      auto executor_buffer = command_buffer_thunk->GetOrCreateCommandBuffer(client.backend().stream_executor(0).value()).value();
      // auto & buffers = command_buffer_thunk->state_.get()->command_buffers;
      // std::cout << "Buffer size: " << (buffers.size()) << std::endl;
      // for (auto &[se, executor_buffer] : buffers) {
      auto* command_buffer = executor_buffer.get()->command_buffer.get();
      auto* gpu_command_buffer = dynamic_cast<stream_executor::gpu::GpuCommandBuffer*>(command_buffer);
      auto gpu_graph_node_infos = gpu_command_buffer->nodes();
      std::cout << ", GPU graph node info size: " << gpu_graph_node_infos.size() << std::endl;
      std::cout << ", Barriers size: " << gpu_command_buffer->barriers().size() << std::endl;
      // }
    }

    // Check for any explicit WaitForStreamsThunk dependencies:
    if (auto* sync_thunk = dynamic_cast<const gpu::WaitForStreamsThunk*>(thunk)) {
      auto waits = sync_thunk->wait_for_stream_id();
      std::cout << ", waits for stream  " << waits << "";
    }
    std::cout << std::endl;
  }
}

inline std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> compile_and_execute(xla::PjRtStreamExecutorClient& pjrt_client, const xla::XlaComputation& computation,
                                                                                      absl::Span<const std::vector<xla::PjRtBuffer*>> argument_handles = {{}},
                                                                                      const xla::CompileOptions& compile_opts = {},
                                                                                      const xla::ExecuteOptions& exec_opts = {}) {
  // Compile the XLA computation.
  auto exec_or = pjrt_client.Compile(computation, compile_opts);
  const std::unique_ptr<xla::PjRtLoadedExecutable> executable = std::move(exec_or.value());

  // Print GPU thunk info
  auto gpu_executable = dynamic_cast<xla::gpu::GpuExecutable*>(dynamic_cast<xla::PjRtStreamExecutorLoadedExecutable*>(executable.get())->executables()[0]->executable());
  print_gpu_thunk_info(*pjrt_client.client(), *gpu_executable);

  // Execute the compiled executable.
  auto outputs = executable->Execute(argument_handles, exec_opts).value();
  std::cout << "outputs.size=" << outputs.size() << " " << "outputs[0].size=" << outputs[0].size() << std::endl;
  return outputs;
}

inline tsl::StatusOr<std::shared_ptr<xla::Literal>> buffer_to_literal(xla::PjRtBuffer& buffer) {
  TF_ASSIGN_OR_RETURN(const auto final_literal_or, buffer.ToLiteralSync());
  const xla::Literal& final_literal = *final_literal_or;
  std::cout << "Single output shape: " << xla::ShapeUtil::HumanString(final_literal.shape()) << std::endl;
  return final_literal_or;
}

inline tsl::StatusOr<std::shared_ptr<xla::Literal>> buffer_to_literal(const std::unique_ptr<xla::PjRtBuffer>& buffer) {
  if (buffer) {
    return buffer_to_literal(*buffer);
  }
  return xla::InvalidArgument("Buffer is null");
}

}  // namespace xla_test_util
}  // namespace tensorflow

#endif  // TEST_UTIL_H
