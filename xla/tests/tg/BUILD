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
