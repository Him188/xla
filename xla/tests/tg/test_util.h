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

// Creates a buffer on the device from host data.
std::unique_ptr<PjRtBuffer> CreateDeviceBuffer(PjRtClient &client, absl::Span<const float> host_data, const Shape &shape);

void print_gpu_thunk_sequence(se::StreamExecutor *stream_executor, const gpu::ThunkSequence &thunk_sequence, int &idx, int depth = 0);

void print_gpu_thunk_info(const LocalClient &client, gpu::GpuExecutable &gpu_exec);

std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> compile_and_execute(PjRtStreamExecutorClient &pjrt_client, const XlaComputation &computation,
                                                                          absl::Span<const std::vector<PjRtBuffer *>> argument_handles = {{}},
                                                                          const CompileOptions &compile_opts = {}, const ExecuteOptions &exec_opts = {});

tsl::StatusOr<std::shared_ptr<Literal>> buffer_to_literal(PjRtBuffer &buffer);

tsl::StatusOr<std::shared_ptr<Literal>> buffer_to_literal(const std::unique_ptr<PjRtBuffer> &buffer);

} // namespace xla::xla_test_util

#endif // TEST_UTIL_H
