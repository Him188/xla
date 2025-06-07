#ifndef XLA_TESTS_CONCURRENCY_TRACE_PERF_UTILS_H_
#define XLA_TESTS_CONCURRENCY_TRACE_PERF_UTILS_H_

#include <cstdio>
#include <unistd.h>

namespace xla {

inline size_t GetCurrentRSSBytes() {
  long rss = 0;
  FILE* fp = fopen("/proc/self/statm", "r");
  if (fp != nullptr) {
    if (fscanf(fp, "%*s%ld", &rss) != 1) rss = 0;
    fclose(fp);
  }
  return rss * sysconf(_SC_PAGESIZE);
}

}  // namespace xla

#endif  // XLA_TESTS_CONCURRENCY_TRACE_PERF_UTILS_H_
