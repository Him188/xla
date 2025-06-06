#include "xla/tests/concurrency_trace/base_concurrency_tests.h"
#include "xla/tests/test_macros.h"

namespace xla {

class CrossStreamSynchronizationTests : public BaseConcurrencyTests {};

XLA_TEST_F(CrossStreamSynchronizationTests, ExecuteOnMultipleStreamsFused) {
  RunTest(R"(
HloModule test_graph

ENTRY test_graph {
  p0 = f32[4048,4048]{1,0} parameter(0)
  p1 = f32[4048,4048]{1,0} parameter(1)
  dot0 = f32[4048,4048]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot1 = f32[4048,4048]{1,0} dot(dot0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  sub = f32[4048,4048]{1,0} subtract(dot1, dot0)
  ROOT out = (f32[4048,4048]{1,0}) tuple(sub)
}
  )",
          false);
}

}  // namespace xla
