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

# Generate test_suites for all backends, named "${backend}_tests".
generate_backend_suites()

cc_library(
    name = "test_util",
    testonly = True,
    srcs = [
        "test_util.cc",
        "test_util.h",
    ],
    deps = [
        "//xla:test",
        "//xla/backends/gpu/runtime:command_buffer_thunk",
        "//xla/backends/gpu/runtime:wait_for_streams_thunk",
        "//xla/backends/gpu/runtime:while_thunk",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/service/gpu:gpu_executable",
        "//xla/tests:literal_test_util",
        "//xla/stream_executor/gpu:gpu_command_buffer",

        #        "//xla:array2d",
        #        "//xla:array4d",
        #        "//xla:literal",
        #        "//xla:literal_util",
        #        "//xla:test",
        #        "//xla/backends/gpu/runtime:command_buffer_thunk",
        #        "//xla/client:client_library",
        #        "//xla/client:local_client",
        #        "//xla/client:xla_builder",
        #        "//xla/client/lib:matrix",
        #        "//xla/hlo/builder:xla_builder",
        #        "//xla/hlo/testlib:test",
        #        "//xla/pjrt:pjrt_client",
        #        "//xla/pjrt:pjrt_executable",
        #        "//xla/pjrt:pjrt_stream_executor_client",
        #        "//xla/pjrt/gpu:gpu_topology",
        #        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        #        "//xla/stream_executor:stream_executor_common",
        #        "//xla/tests:client_library_test_base",
        #        "//xla/tests:literal_test_util",
        #        "//xla/tests:test_macros_header",
        #        "//xla/tests:xla_internal_test_main",
        #        "//xla/tools:hlo-opt",
        #        "//xla/tools/hlo_opt:gpu_opt",
        #        "//xla/tsl/lib/core:status_test_util",
        #        "//xla/tsl/platform:status_matchers",
        #        "@com_google_absl//absl/log",
        #        "@com_google_absl//absl/log:initialize",
        #        "@com_google_absl//absl/status:statusor",
        #        "@com_google_absl//absl/strings:string_view",
        #        "@com_google_absl//absl/time",
    ],
)

xla_test(
    name = "graph_debug_test",
    srcs = [
        "graph_debug_test.cc",
    ],
    env = {
        "XLA_FLAGS": "",
        "TF_CPP_MIN_VLOG_LEVEL": "0",
    },
    deps = [
        "//xla:test",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:initialize",
        "//xla/tests:client_library_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_macros_header",
        "//xla/tests:xla_internal_test_main",
        "//xla:array2d",
        "//xla:array4d",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/testlib:test",
        "//xla/pjrt/gpu:gpu_topology",
        "//xla/pjrt:pjrt_executable",
        "//xla/client:client_library",
        "//xla/client:xla_builder",
        "//xla/backends/gpu/runtime:command_buffer_thunk",
        "//xla/tools/hlo_opt:gpu_opt",
        "//xla/tools:hlo-opt",
        "//xla/stream_executor:stream_executor_common",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/client/lib:matrix",
        #        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/pjrt:pjrt_stream_executor_client",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "test_util",
        #        "@local_xla//xla/tests:test_utils",
        #        "@local_xla//xla/hlo/utils:hlo_matchers",
    ],
)

xla_test(
    name = "thunk_stream_test",
    srcs = [
        "thunk_stream_test.cc",
    ],
    deps = [
        "test_util",
        "//xla:test",
        "//xla/hlo/testlib:test",
        "//xla/tools/hlo_opt:gpu_opt",
        #        "//xla/service/gpu:gpu_compiler",  # CUDA compiler
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "stream_collective",
    srcs = [
        "stream_collective.cc",
    ],
    deps = [
        "test_util",
        "//xla:test",
        "//xla/hlo/testlib:test",
        "//xla/tools/hlo_opt:gpu_opt",
        #        "//xla/service/gpu:gpu_compiler",  # CUDA compiler
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "dynamic_slice",
    srcs = [
        "dynamic_slice.cc",
    ],
    deps = [
        "test_util",
        "//xla/tests:hlo_test_base",
        "//xla:test",
        "//xla/hlo/testlib:test",
        "//xla/tools/hlo_opt:gpu_opt",
        #        "//xla/service/gpu:gpu_compiler",  # CUDA compiler
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
