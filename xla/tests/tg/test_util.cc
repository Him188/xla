#include "test_util.h"

#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/nccl_all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/wait_for_streams_thunk.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"

#include <fstream>

namespace xla {
namespace xla_test_util {

const std::string kXlaLocalDumpDir = "/home/him188/CLionProjects/xla/dumps";

void xla_test_util::SetXlaDumpFlags(const std::string &dump_dir) {
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

void xla_test_util::EnableLogs() {
  setenv("TF_CPP_MIN_VLOG_LEVEL", "0", 1);
  // setenv("TF_CPP_MAX_VLOG_LEVEL", "10", 1);
  setenv("TF_CPP_VMODULE",
         "xla_service=2,xla_compilation_cache=1,gpu_compiler=3,command_buffer_thunk=3,async_wrapper.cc=3,xla/backends/gpu/collectives/"
         "nccl_communicator.cc=10,collective_pipeliner=10",
         1);
}

void xla_test_util::PrintIrDumps(const std::string &dump_dir, const std::vector<IRDumpKind> &kinds) {
  for (const auto &entry : std::filesystem::directory_iterator(dump_dir)) {
    std::string path = entry.path().string();
    // Simple suffix checking. If your C++ standard is >= 20, you could use
    // std::string::ends_with(...)
    auto ends_with = [&](const std::string &s, const std::string &suffix) {
      if (s.size() < suffix.size())
        return false;
      return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    auto need_kind = [&](const IRDumpKind kind) { return std::find(kinds.begin(), kinds.end(), kind) != kinds.end(); };

    auto filename = entry.path().filename().string();
    if ((ends_with(path, ".txt") && need_kind(IRDumpKind::kHLO)) || (ends_with(path, ".mlir") && need_kind(IRDumpKind::kMLIR)) ||
        (ends_with(path, ".ll") && need_kind(IRDumpKind::kLLVM)) || (ends_with(path, ".ptx") && need_kind(IRDumpKind::kPTX)) ||
        (ends_with(path, ".dot") && need_kind(IRDumpKind::kDOT))) {
      std::ifstream file(path);
      if (!file.is_open()) {
        continue;
      }
      std::stringstream buffer;
      buffer << file.rdbuf();
      std::cout << "\n--- IR Dump: " << filename // just the file name
                << " ---\n";
      std::cout << buffer.str() << std::endl;
    }

    if (ends_with(path, ".html") && need_kind(IRDumpKind::kHTML)) {
      // Copy file to kXlaLocalDumpDir
      std::string new_path = kXlaLocalDumpDir + "/" + filename;
      std::filesystem::copy(path, new_path, std::filesystem::copy_options::overwrite_existing);
      std::cout << "\n--- HTML Dump: " << filename // just the file name
                << " ---\n";
      std::cout << filename << std::endl;
    }
  }
}

std::unique_ptr<PjRtBuffer> xla_test_util::CreateDeviceBuffer(PjRtClient &client, absl::Span<const float> host_data, const Shape &shape, int device_ordinal) {
  // Check shape sizes match:
  size_t expected_size = 1;
  for (auto dim : shape.dimensions()) {
    expected_size *= dim;
  }
  CHECK_EQ(expected_size, host_data.size()) << "Host data size must match shape size.";

  // Create the device buffer (for static shapes).
  // Note: For dynamic shapes, we also pass the dynamic dimension sizes.
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer_or = client.BufferFromHostBuffer(
      /*data=*/host_data.data(),
      /*type=*/F32,
      /*dims=*/shape.dimensions(),
      /*byte_strides=*/absl::nullopt, // row-major by default
      PjRtClient::HostBufferSemantics::kImmutableZeroCopy, [] {}, client.devices()[device_ordinal]->default_memory_space().value(), nullptr);

  TF_CHECK_OK(buffer_or.status());
  return std::move(buffer_or.value());
}

void xla_test_util::print_gpu_thunk_sequence(se::StreamExecutor *stream_executor, const gpu::ThunkSequence &thunk_sequence, int &idx, int depth) {
  for (const std::unique_ptr<gpu::Thunk> &thunk_ptr : thunk_sequence) {
    const gpu::Thunk *thunk = thunk_ptr.get();
    std::string_view kind = gpu::Thunk::KindToString(thunk->kind());
    uint64_t stream_id = thunk->execution_stream_id().value(); // fallback to 0 if unset

    auto start_line = [depth]() -> std::ostream & {
      for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
      }
      return std::cout;
    };

    start_line() << "Thunk " << idx++ << ": Kind=" << kind << ", launches on " << stream_id;

    if (auto *command_buffer_thunk = const_cast<gpu::CommandBufferThunk *>(dynamic_cast<const gpu::CommandBufferThunk *>(thunk))) {
      std::cout << command_buffer_thunk->ToString(0);
      auto executor_buffer = command_buffer_thunk->GetOrCreateCommandBuffer(stream_executor).value();
      // auto & buffers = command_buffer_thunk->state_.get()->command_buffers;
      // std::cout << "Buffer size: " << (buffers.size()) << std::endl;
      // for (auto &[se, executor_buffer] : buffers) {
      auto *command_buffer = executor_buffer->command_buffer.get();
      auto *gpu_command_buffer = dynamic_cast<stream_executor::gpu::GpuCommandBuffer *>(command_buffer);
      auto gpu_graph_node_infos = gpu_command_buffer->nodes();
      std::cout << ", GPU graph node info size: " << gpu_graph_node_infos.size();
      std::cout << ", Barriers size: " << gpu_command_buffer->barriers().size();
      // }
      std::cout << std::endl;
    } else if (auto *sync_thunk = dynamic_cast<const gpu::WaitForStreamsThunk *>(thunk)) {
      // Check for any explicit WaitForStreamsThunk dependencies:
      auto waits = sync_thunk->wait_for_stream_id();
      std::cout << ", waits for stream  " << waits << "";
      std::cout << std::endl;
    } else if (auto *while_thunk = dynamic_cast<const gpu::WhileThunk *>(thunk)) {
      std::cout << std::endl;
      start_line() << "  Loop Condition: " << std::endl;
      print_gpu_thunk_sequence(stream_executor, while_thunk->condition_thunk_sequence()->thunks(), idx, depth + 2);
      start_line() << "  Loop Body: " << std::endl;
      print_gpu_thunk_sequence(stream_executor, while_thunk->body_thunk_sequence()->thunks(), idx, depth + 2);
    } else if (auto *gemm_thunk = dynamic_cast<const gpu::GemmThunk *>(thunk)) {
      std::cout << ", LHS: " << gemm_thunk->lhs_buffer();
      std::cout << ", RHS: " << gemm_thunk->rhs_buffer();
      std::cout << ", output: " << gemm_thunk->output_buffer();
      std::cout << std::endl;
    } else if (auto *nccl_async_start_thunk = dynamic_cast<const gpu::NcclAllReduceStartThunk *>(thunk)) {
      std::cout << ", source: " << nccl_async_start_thunk->buffers()[0].source_buffer;
      std::cout << ", dest: " << nccl_async_start_thunk->buffers()[0].destination_buffer;
      std::cout << std::endl;
    } else if (auto *kernel_thunk = dynamic_cast<const gpu::KernelThunk *>(thunk)) {
      for (int i = 0; i < kernel_thunk->arguments().size(); ++i) {
        const auto &argument = kernel_thunk->arguments()[i];
        const auto is_write = kernel_thunk->written()[i];
        std::cout << ", ";
        if (is_write) {
          std::cout << "in";
        } else {
          std::cout << "out";
        }
        std::cout << argument.allocation();
      }
      std::cout << std::endl;
    } else {
      std::cout << std::endl;
    }
  }
}

void print_gpu_thunk_info(const LocalClient &client, gpu::GpuExecutable &gpu_exec) {
  // std::cout << "\n=== Thunk Text (NVPTX) ===" << std::endl;
  // std::cout << gpu_exec.text() << std::endl;

  const gpu::ThunkSequence &thunk_sequence = gpu_exec.GetThunk().thunks();
  std::cout << "\n=== Thunk List (" << thunk_sequence.size() << ") ===" << std::endl;

  int idx = 0;
  const auto executor = client.backend().stream_executor(0).value();
  print_gpu_thunk_sequence(executor, thunk_sequence, idx);
  std::cout << "=== End of Thunk List ===\n" << std::endl;
}

// inline std::ostream& operator<<(std::ostream& os,
//                                 const BufferAllocation::Slice& slice) {
//   const BufferAllocation* alloc = slice.allocation();   // may be nullptr
//
//   os << "Slice(index=" << slice.index()
//      << ", offset="     << slice.offset()
//      << ", size="       << slice.size();
//
//   if (alloc) {  // extra context that lives on the parent allocation
//     os << ", allocation_size=" << alloc->size()
//        << ", is_input="        << std::boolalpha << alloc->is_input()
//        << ", is_output="       << alloc->is_output()
//        << ", is_constant="     << alloc->is_constant()
//        << ", maybe_live_out="  << alloc->maybe_live_out();
//   } else {
//     os << ", alloc is nullptr";
//   }
//
//   os << ')';
//   return os;
// }

std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> compile_and_execute(PjRtStreamExecutorClient &pjrt_client, const XlaComputation &computation,
                                                                          absl::Span<const std::vector<PjRtBuffer *>> argument_handles,
                                                                          const CompileOptions &compile_opts, const ExecuteOptions &exec_opts) {
  // Compile the XLA computation.
  auto exec_or = pjrt_client.Compile(computation, compile_opts);
  const std::unique_ptr<PjRtLoadedExecutable> executable = std::move(exec_or.value());

  // Print GPU thunk info
  auto gpu_executable =
      dynamic_cast<gpu::GpuExecutable *>(dynamic_cast<PjRtStreamExecutorLoadedExecutable *>(executable.get())->executables()[0]->executable());
  print_gpu_thunk_info(*pjrt_client.client(), *gpu_executable);

  // Execute the compiled executable.
  auto outputs = executable->Execute(argument_handles, exec_opts).value();
  std::cout << "outputs.size=" << outputs.size() << " " << "outputs[0].size=" << outputs[0].size() << std::endl;
  return outputs;
}

tsl::StatusOr<std::shared_ptr<Literal>> buffer_to_literal(PjRtBuffer &buffer) {
  TF_ASSIGN_OR_RETURN(const auto final_literal_or, buffer.ToLiteralSync());
  const Literal &final_literal = *final_literal_or;
  std::cout << "Single output shape: " << ShapeUtil::HumanString(final_literal.shape()) << std::endl;
  return final_literal_or;
}

tsl::StatusOr<std::shared_ptr<Literal>> buffer_to_literal(const std::unique_ptr<PjRtBuffer> &buffer) {
  if (buffer) {
    return buffer_to_literal(*buffer);
  }
  return InvalidArgument("Buffer is null");
}
void SetLiteralValue(Literal &dest, const absl::Span<const float> src, const int64_t src_row_start) {
  const xla::Shape &shape = dest.shape();
  const int64_t rows_per_part = shape.dimensions(0);
  const int64_t cols = shape.dimensions(1);

  for (int64_t i = 0; i < rows_per_part; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      dest.Set<float>({i, j}, src[(src_row_start + i) * cols + j]);
    }
  }
}
std::pair<std::unique_ptr<PjRtBuffer>, Literal> CreateDeviceBuffer(PjRtClient &client, const Shape shape, const float value, const PjRtDevice &device) {
  Literal literal(shape);
  std::vector<float> host;
  host.resize(shape.dimensions(0) * shape.dimensions(1), value);
  SetLiteralValue(literal, host, 0);

  std::pair<std::unique_ptr<PjRtBuffer>, Literal> pair = std::make_pair(nullptr, std::move(literal));

  auto buffer = client.BufferFromHostLiteral(pair.second, device.default_memory_space().value()).value();
  pair.first = std::move(buffer);
  return pair;
}

} // namespace xla_test_util
} // namespace xla