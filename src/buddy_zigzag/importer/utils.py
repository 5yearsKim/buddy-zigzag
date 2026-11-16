"""Compiler process and supported-subset utilities."""

import hashlib
import subprocess
from pathlib import Path

from ..exporter.schema import Workload
from .errors import ScheduleError


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run_tool(executable: Path, arguments: list[str], text: str) -> str:
    if not executable.is_file():
        raise ScheduleError(f"Compiler tool is unavailable: {executable}; check .env build directories")
    try:
        result = subprocess.run(
            [str(executable), *arguments],
            input=text,
            text=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScheduleError(f"Compiler tool failed: {executable.name}: {exc}") from exc
    if result.returncode:
        raise ScheduleError(f"{executable.name} failed:\n{result.stderr.strip()}")
    return result.stdout


def require_supported_workload(workload: Workload):
    if len(workload.layers) != 1:
        raise ScheduleError("The importer currently supports one matmul per function")
    layer = workload.layers[0]
    if layer.mlir_op != "linalg.matmul" or layer.loop_dims != ("M", "N", "K"):
        raise ScheduleError("The importer currently supports canonical, unbatched linalg.matmul")
    expected = {"I": ("M", "K"), "W": ("K", "N"), "O": ("M", "N")}
    if any(
        operand.indices != expected[name] or operand.tensor.dtype != "f32" for name, operand in layer.operands.items()
    ):
        raise ScheduleError("The importer currently requires canonical FP32 matrix accesses")
    if any(layer.operands[name].source_kind != "input" for name in ("I", "W")):
        raise ScheduleError("The importer currently requires two external matrix inputs")
    if {layer.operands[name].source_name for name in ("I", "W")} != {"arg0", "arg1"}:
        raise ScheduleError("The importer requires two distinct matrix inputs in the function's first two arguments")
    if workload.outputs != [layer.operands["O"].source_name]:
        raise ScheduleError("The importer requires one returned matmul result")
    return layer
