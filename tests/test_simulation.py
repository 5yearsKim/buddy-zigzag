"""Exercise the C memref runtime, Bazel ELF build, and static harness ABI."""

import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from buddy_zigzag import DsePipeline, ScheduleCompiler, ScheduleCompileRequest
from buddy_zigzag.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
SOURCES = ROOT / "simulation"


class MemRef2D(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.POINTER(ctypes.c_float)),
        ("aligned", ctypes.POINTER(ctypes.c_float)),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]


class UnrankedMemRef(ctypes.Structure):
    _fields_ = [("rank", ctypes.c_int64), ("descriptor", ctypes.c_void_p)]


@pytest.fixture
def runtime(tmp_path):
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("Host gcc is unavailable")
    library = tmp_path / "runtime.so"
    subprocess.run(
        [gcc, "-shared", "-fPIC", "-std=c99", str(SOURCES / "memref_runtime.c"), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    loaded = ctypes.CDLL(str(library))
    loaded.memrefCopy.argtypes = [ctypes.c_int64, ctypes.POINTER(UnrankedMemRef), ctypes.POINTER(UnrankedMemRef)]
    loaded.memrefCopy.restype = None
    return loaded


def test_runtime_copies_offsets_and_independent_negative_strides(runtime):
    source = (ctypes.c_float * 20)(*range(20))
    destination = (ctypes.c_float * 20)(*([-1] * 20))
    src = MemRef2D(source, source, 3, (2, 3), (5, -1))
    dst = MemRef2D(destination, destination, 2, (2, 3), (7, 2))
    src_arg = UnrankedMemRef(2, ctypes.addressof(src))
    dst_arg = UnrankedMemRef(2, ctypes.addressof(dst))
    runtime.memrefCopy(4, ctypes.byref(src_arg), ctypes.byref(dst_arg))
    expected = [-1] * 20
    for row in range(2):
        for col in range(3):
            expected[2 + row * 7 + col * 2] = source[3 + row * 5 - col]
    assert list(destination) == expected


def test_runtime_handles_scalar_and_empty_shapes(runtime):
    source = (ctypes.c_float * 4)(1, 2, 3, 4)
    destination = (ctypes.c_float * 4)(-1, -1, -1, -1)
    src = MemRef2D(source, source, 1, (0, 2), (2, 1))
    dst = MemRef2D(destination, destination, 2, (0, 2), (2, 1))
    src_arg = UnrankedMemRef(2, ctypes.addressof(src))
    dst_arg = UnrankedMemRef(2, ctypes.addressof(dst))
    runtime.memrefCopy(4, ctypes.byref(src_arg), ctypes.byref(dst_arg))
    assert list(destination) == [-1] * 4
    src_arg.rank = dst_arg.rank = 0
    runtime.memrefCopy(4, ctypes.byref(src_arg), ctypes.byref(dst_arg))
    assert list(destination) == [-1, -1, 2, -1]


def test_bazel_builds_riscv_elf_and_harness_matches_native_cpu_kernel(tmp_path):
    bazel = shutil.which("bazel") or shutil.which("bazelisk")
    settings = Settings.load()
    llvm = settings.llvm_mlir_build_dir
    toolchain = Path(os.environ.get("RISCV_TOOLCHAIN_DIR", str(ROOT / "third_party/chipyard/.conda-env/riscv-tools")))
    if (
        bazel is None
        or llvm is None
        or not (llvm / "bin/llc").is_file()
        or not (toolchain / "bin/riscv64-unknown-elf-gcc").is_file()
    ):
        pytest.skip("Bazel, Buddy LLVM, and the RISC-V HTIF toolchain are required")
    prepared = DsePipeline(settings).prepare_case("matmul_small", output_dir=tmp_path / "workload")
    compiler = ScheduleCompiler(settings)
    compiler.compile(
        ScheduleCompileRequest(
            workload_dir=prepared.output_dir,
            result=FIXTURES / "zigzag/gemmini_matmul_small_complete.json",
            target="gemmini",
            snapshot=FIXTURES / "hardware/fp32_parameters.json",
        )
    )
    simulation_dir = prepared.output_dir / "simulation"
    simulation_dir.mkdir()
    supplied_header = """/* Manually maintained configuration. */
#ifndef BUDDY_ZIGZAG_WORKLOAD_CONFIG_H
#define BUDDY_ZIGZAG_WORKLOAD_CONFIG_H
#define WORKLOAD_M 3
#define WORKLOAD_N 7
#define WORKLOAD_K 5
#define WORKLOAD_ENTRY _mlir_ciface_forward
#define WORKLOAD_ARG0_IS_LHS 1
#define WORKLOAD_ATOL 1e-4
#define WORKLOAD_RTOL 1e-4
#endif
"""
    (simulation_dir / "workload_config.h").write_text(supplied_header)
    subprocess.run(
        [
            bazel,
            "--output_base=" + str(tmp_path / "bazel-output"),
            "build",
            "//simulation:program",
            "--repo_env=BUDDY_WORKLOAD_DIR=" + str(prepared.output_dir),
            "--repo_env=BUDDY_LLVM_DIR=" + str(llvm),
            "--repo_env=RISCV_TOOLCHAIN_DIR=" + str(toolchain.resolve()),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=300,
    )
    elf = (ROOT / "bazel-bin/simulation/program/program.riscv").read_bytes()
    assert elf[:6] == b"\x7fELF\x02\x01"
    assert int.from_bytes(elf[18:20], "little") == 243
    assert (simulation_dir / "workload_config.h").read_text() == supplied_header

    if not (llvm / "bin/clang").is_file():
        pytest.skip("Native clang is unavailable for the harness ABI check")
    cpu = compiler.compile(
        ScheduleCompileRequest(
            workload_dir=prepared.output_dir,
            result=FIXTURES / "zigzag/matmul_small_complete.json",
            serialize_spatial=True,
            output_dir=tmp_path / "cpu",
        )
    )
    host = tmp_path / "host-harness"
    subprocess.run(
        [
            str(llvm / "bin/clang"),
            "-O2",
            "-I",
            str(simulation_dir),
            str(SOURCES / "main.c"),
            str(SOURCES / "memref_runtime.c"),
            str(cpu.output_dir / "06_lowered.ll"),
            "-lm",
            "-o",
            str(host),
        ],
        check=True,
        capture_output=True,
    )
    output = subprocess.check_output([str(host)], text=True)
    verification = json.loads(
        next(line.removeprefix("BZ_SIM_RESULT ") for line in output.splitlines() if line.startswith("BZ_SIM_RESULT "))
    )
    assert verification == {"status": "passed", "cycles": 0, "mismatches": 0}
