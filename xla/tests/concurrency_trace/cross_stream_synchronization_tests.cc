#include "xla/tests/concurrency_trace/base_concurrency_tests.h"
#include "xla/tests/test_macros.h"

namespace xla {

class CrossStreamSynchronizationTests : public BaseConcurrencyTests {};

XLA_TEST_F(CrossStreamSynchronizationTests, SimpleDot) {
  RunTest(R"(
HloModule test_graph

ENTRY test_graph {
  p0 = f32[2048,2048]{1,0} parameter(0)
  p1 = f32[2048,2048]{1,0} parameter(1)
  dot0 = f32[2048,2048]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot1 = f32[2048,2048]{1,0} dot(dot0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  sub = f32[2048,2048]{1,0} subtract(dot1, dot0)
  ROOT out = (f32[2048,2048]{1,0}) tuple(sub)
}
  )",
          false);
}

XLA_TEST_F(CrossStreamSynchronizationTests, AsyncDot) {
  EnableAsyncDot();
  RunTest(R"(
HloModule test_graph

ENTRY test_graph {
  p0 = f32[2048,2048]{1,0} parameter(0)
  p1 = f32[2048,2048]{1,0} parameter(1)
  dot0 = f32[2048,2048]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot1 = f32[2048,2048]{1,0} dot(dot0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  sub = f32[2048,2048]{1,0} subtract(dot1, dot0)
  ROOT out = (f32[2048,2048]{1,0}) tuple(sub)
}
  )",
          false);
}

XLA_TEST_F(CrossStreamSynchronizationTests, AsyncDotBug) {
  EnableAsyncDot();
  EnableSyntheticWaitForStreamsBug();
  RunTest(R"(
HloModule test_graph

ENTRY test_graph {
  p0 = f32[2048,2048]{1,0} parameter(0)
  p1 = f32[2048,2048]{1,0} parameter(1)
  dot0 = f32[2048,2048]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot1 = f32[2048,2048]{1,0} dot(dot0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  sub = f32[2048,2048]{1,0} subtract(dot1, dot0)
  ROOT out = (f32[2048,2048]{1,0}) tuple(sub)
}
  )",
          true);
}

} // namespace xla
