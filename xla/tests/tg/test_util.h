#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <filesystem>

#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h" // OpenXLA GPU client API&#8203;:contentReference[oaicite:3]{index=3}
#include "xla/service/gpu/gpu_executable.h"
#include "xla/tests/literal_test_util.h"

#include <string>
#include <vector>

namespace xla::xla_test_util {

struct XlaDumpIr {
  XlaDumpIr();
};

// Which kinds of IR we want to dump
enum class IRDumpKind {
  kHLO,
  kLLVM,
  kMLIR,
  kPTX,
  kHTML,
  kDOT,
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
void SetXlaDumpFlags(const std::string &dump_dir);

void EnableLogs();

/**
 * Prints any files in `dump_dir` that match recognized IR dump file suffixes:
 * .txt (HLO), .mlir, .ll (LLVM), .ptx
 */
void PrintIrDumps(const std::string &dump_dir, const std::vector<IRDumpKind> &kinds);

// Creates a buffer on the device from host data. Note: you must hold the host_data until the buffer is not needed.
std::unique_ptr<PjRtBuffer> CreateDeviceBuffer(PjRtClient &client, absl::Span<const float> host_data, const Shape &shape, int device_ordinal = 0);

void print_gpu_thunk_sequence(se::StreamExecutor *stream_executor, const gpu::ThunkSequence &thunk_sequence, int &idx, int depth = 0);

void print_gpu_thunk_info(const LocalClient &client, gpu::GpuExecutable &gpu_exec);
inline void print_gpu_thunk_info(const LocalClient &client, gpu::GpuExecutable *gpu_exec) {
  ASSERT_TRUE(gpu_exec != nullptr) << "Underlying executable is not a GpuExecutable";
  // ReSharper disable once CppDFANullDereference
  return print_gpu_thunk_info(client, *gpu_exec);
}
inline void print_gpu_thunk_info(const LocalClient &client, const absl::Span<const std::shared_ptr<LocalExecutable>> executables) {
  for (auto ptr : executables) {
    if (auto *executable = ptr.get()->executable(); executable != nullptr) {
      auto *exec = dynamic_cast<gpu::GpuExecutable *>(executable);
      if (exec == nullptr)
        continue;

      std::cout << "\n=== GpuExecutable: " << exec->module().name() << " ===" << std::endl;
      print_gpu_thunk_info(client, exec);
      std::cout << std::endl;
    }
  }
}
inline void print_gpu_thunk_info(const PjRtLoadedExecutable *executable) {
  const auto *se_loaded = dynamic_cast<const PjRtStreamExecutorLoadedExecutable *>(executable);
  ASSERT_TRUE(se_loaded != nullptr) << "Executable is not a Stream-Executor executable";
  const auto executor_client = dynamic_cast<PjRtStreamExecutorClient *>(executable->client());
  ASSERT_TRUE(executor_client != nullptr) << "Executable client is not a Stream-Executor client";
  const LocalClient *local_client = executor_client->client();
  ASSERT_TRUE(local_client != nullptr) << "Local client is null";
  print_gpu_thunk_info(*local_client, se_loaded->executables());
}
inline LocalClient *GetLocalClient(PjRtClient *pjrt) {
  const auto executor_client = dynamic_cast<PjRtStreamExecutorClient *>(pjrt);
  if (executor_client == nullptr) {
    throw std::runtime_error("Executable client is not a Stream-Executor client");
  }
  LocalClient *local_client = executor_client->client();
  if (local_client == nullptr) {
    throw std::runtime_error("Local client is null");
  }
  return local_client;
}

std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> compile_and_execute(PjRtStreamExecutorClient &pjrt_client, const XlaComputation &computation,
                                                                          absl::Span<const std::vector<PjRtBuffer *>> argument_handles = {{}},
                                                                          const CompileOptions &compile_opts = {}, const ExecuteOptions &exec_opts = {});

tsl::StatusOr<std::shared_ptr<Literal>> buffer_to_literal(PjRtBuffer &buffer);

tsl::StatusOr<std::shared_ptr<Literal>> buffer_to_literal(const std::unique_ptr<PjRtBuffer> &buffer);

auto SetLiteralValue(Literal &dest, absl::Span<const float> src, int64_t src_row_start) -> void;

std::pair<std::unique_ptr<PjRtBuffer>, Literal> CreateDeviceBuffer(PjRtClient &client, const Shape shape, const float value, const PjRtDevice &device);

} // namespace xla::xla_test_util

#endif // TEST_UTIL_H
