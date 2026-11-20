"""Build the project-owned pass plugin against the selected MLIR installation."""

import hashlib
import shutil
import subprocess
from pathlib import Path

from ...schedule.errors import ScheduleError
from ...settings import Settings


def build_plugin(settings: Settings) -> Path:
    """Let CMake track source/header changes; keep all outputs in this project."""
    llvm = settings.llvm_mlir_build_dir
    if llvm is None:
        raise ScheduleError("LLVM_MLIR_BUILD_DIR must be configured to build the Gemmini pass plugin")
    cmake = shutil.which("cmake")
    if cmake is None:
        raise ScheduleError("CMake was not found. Please install it to build the Gemmini pass plugin.")
    source = Path(__file__).with_name("native")
    # Separate caches when switching compiler installations, including a different
    # Buddy host, rather than accidentally loading an ABI-incompatible plugin.
    identity = f"{llvm.resolve()}\n{settings.buddy_mlir_build_dir}"
    key = hashlib.sha256(identity.encode()).hexdigest()[:12]
    build = settings.project_dir / "build" / "gemmini-plugin" / key
    build.mkdir(parents=True, exist_ok=True)
    commands = []
    if not (build / "Makefile").is_file() and not (build / "build.ninja").is_file():
        commands.append(
            [
                cmake,
                "-S",
                str(source),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DMLIR_DIR={llvm / 'lib/cmake/mlir'}",
                f"-DLLVM_DIR={llvm / 'lib/cmake/llvm'}",
            ]
        )
    commands.append([cmake, "--build", str(build), "--parallel", "2"])
    log = build / "build.log"
    with log.open("w") as output:
        for command in commands:
            try:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=300)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ScheduleError(f"Gemmini plugin build could not finish. See {log}") from exc
            if result.returncode:
                raise ScheduleError(f"Gemmini plugin build failed. See {log}")
    plugin = build / "lib/libBuddyZigzagGemmini.so"
    if not plugin.is_file():
        raise ScheduleError(f"Gemmini plugin was not produced. See {log}")
    return plugin
