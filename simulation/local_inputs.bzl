"""Make local workload files and installed compilers visible to Bazel."""

def _path(ctx, value):
    if value.startswith("/"):
        return ctx.path(value)
    return ctx.path(ctx.attr.workspace_marker).dirname.get_child(value)

def _local_inputs_impl(ctx):
    selected = ctx.os.environ.get("BUDDY_WORKLOAD_DIR", "")
    if not selected:
        fail("Set BUDDY_WORKLOAD_DIR before running bazel build; see simulation/README.md")
    workload = _path(ctx, selected)
    lowered_override = ctx.os.environ.get("BUDDY_LOWERED_DIR", "")
    lowered = _path(ctx, lowered_override) if lowered_override else workload.get_child("lowered")
    llvm = _path(ctx, ctx.os.environ.get("BUDDY_LLVM_DIR", "third_party/buddy-mlir/llvm/build"))
    riscv = _path(ctx, ctx.os.environ.get("RISCV_TOOLCHAIN_DIR", "third_party/chipyard/.conda-env/riscv-tools"))
    for required in [
        lowered.get_child("06_lowered.ll"),
        workload.get_child("simulation/workload_config.h"),
        llvm.get_child("bin/llc"),
        riscv.get_child("bin/riscv64-unknown-elf-gcc"),
    ]:
        ctx.watch(required)
        if not required.exists or required.is_dir:
            fail("Required input is missing: %s; see simulation/README.md" % required)
    ctx.symlink(lowered.get_child("06_lowered.ll"), "kernel.ll")
    ctx.symlink(workload.get_child("simulation/workload_config.h"), "workload_config.h")
    ctx.symlink(llvm, "llvm")
    ctx.symlink(riscv, "riscv")
    ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])

exports_files([
    "kernel.ll",
    "workload_config.h",
    "llvm/bin/llc",
    "riscv/bin/riscv64-unknown-elf-gcc",
])

filegroup(
    name = "llvm_support",
    srcs = glob(["llvm/lib/*.so", "llvm/lib/*.so.*"]),
)

filegroup(
    name = "gcc_support",
    srcs = glob([
        "riscv/bin/riscv64-unknown-elf-*",
        "riscv/lib/gcc/**",
        "riscv/libexec/gcc/**",
        "riscv/riscv64-unknown-elf/**",
    ]),
)
""")

local_inputs = repository_rule(
    implementation = _local_inputs_impl,
    attrs = {"workspace_marker": attr.label(default = Label("//:MODULE.bazel"))},
    environ = ["BUDDY_WORKLOAD_DIR", "BUDDY_LOWERED_DIR", "BUDDY_LLVM_DIR", "RISCV_TOOLCHAIN_DIR"],
    local = True,
    configure = True,
)
