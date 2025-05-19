#ifndef XLA_TESTS_CONCURRENCY_TRACE_CONCURRENCY_TEST_BASE_H_
#define XLA_TESTS_CONCURRENCY_TRACE_CONCURRENCY_TEST_BASE_H_

#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla {

// Base class for all concurrency tests. It provides a PJRT backend for the test
// runner and uses the interpreter backend as the reference implementation via
// HloRunnerAgnosticReferenceMixin.
class PjRtGpuStreamExecutorConcurrencyTestBase : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
protected:
  PjRtGpuStreamExecutorConcurrencyTestBase() : client_ptr(GetStreamExecutorGpuClient({}).value()) {}

  ~PjRtGpuStreamExecutorConcurrencyTestBase() override = default;

  std::unique_ptr<PjRtClient> const client_ptr;

  PjRtClient &client() const { return *client_ptr; }
};

} // namespace xla

#endif // XLA_TESTS_CONCURRENCY_TRACE_CONCURRENCY_TEST_BASE_H_
