# Various experiments related to the compiler that are not a part of the final XLA binary.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("//xla:package_groups.bzl", "xla_tests_package_groups")
load("//xla:xla.bzl", "tests_build_defs_bzl_deps", "xla_cc_binary", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "generate_backend_suites", "generate_backend_test_macros", "xla_test", "xla_test_library")
load("//xla/tsl:tsl.bzl", "if_google", "if_oss", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # keep visibility private, if you need to depend on this, move it out of experiments
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

xla_tests_package_groups()

generate_backend_suites()

cc_library(
    name = "concurrency_test_base",
    testonly = True,
    srcs = ["concurrency_test_base.cc"],
    hdrs = ["concurrency_test_base.h"],
    deps = [
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
    ],
)

xla_test(
    name = "cross_stream_synchronization_tests",
    srcs = [
        "cross_stream_synchronization_tests.cc",
    ],
    tags = [
        "requires-gpu-nvidia:2",
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":base_concurrency_tests",
        "//xla:test",
        "//xla/backends/gpu/runtime:executable_stats",
        "//xla/hlo/testlib:test",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/tests/tg:test_util",
        "//xla/tools/hlo_opt:gpu_opt",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "base_concurrency_tests",
    testonly = True,
    srcs = ["base_concurrency_tests.cc"],
    hdrs = ["base_concurrency_tests.h"],
    deps = [
        ":concurrency_test_base",
        ":trace_stats",
        "//xla:test",
        "//xla/backends/gpu/runtime:executable_stats",
        "//xla/hlo/testlib:test",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/tests:test_utils",
        "//xla/tests/tg:test_util",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "trace_stats",
    testonly = True,
    srcs = ["trace_stats.cc"],
    hdrs = [
        "perf_utils.h",
        "trace_stats.h",
    ],
    deps = [
        "//xla/backends/gpu/runtime:executable_stats",
        "//xla/backends/gpu/runtime:thunk_sanitizer",
    ],
)

xla_test(
    name = "latency_hiding_scheduler_concurrency_tests",
    srcs = [
        "latency_hiding_scheduler_concurrency_tests.cc",
    ],
    tags = [
        "requires-gpu-nvidia:2",
        "test_migrated_to_hlo_runner_pjrt",
    ],
    deps = [
        ":base_concurrency_tests",
        ":concurrency_test_base",
        ":trace_stats",
        "//xla:test",
        "//xla/backends/gpu/runtime:executable_stats",
        "//xla/hlo/testlib:test",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/tests/tg:test_util",
        "//xla/tools/hlo_opt:gpu_opt",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
    ],
)

xla_cc_binary(
    name = "tracer_perf_main",
    testonly = True,
    srcs = ["tracer_perf_main.cc"],
    deps = [
        ":trace_stats",
        "//xla:debug_options_flags",
        "//xla/backends/gpu/runtime:executable_stats",
        "//xla/backends/gpu/runtime:thunk_sanitizer",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/tests:test_utils",
        "//xla/tools/hlo_opt:gpu_opt",
        "@com_google_absl//absl/flags:parse",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
    ],
)
