"""Lower explicit FP32 WS tiles to Gemmini operations and LLVM intrinsics."""

from ..exporter.schema import Schema
from ..settings import Settings
from .errors import ScheduleError
from .lowering import BUFFERIZE_PIPELINE
from .schema import GemminiSchedule
from .utils import run_tool


class GemminiCode(Schema):
    bufferized_mlir: str
    gemmini_mlir: str
    llvm_mlir: str
    llvm_ir: str
    pipelines: list[str]


class GemminiLowerer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def lower(self, scheduled: str, schedule: GemminiSchedule) -> GemminiCode:
        if self.settings.buddy_mlir_build_dir is None:
            raise ScheduleError("BUDDY_MLIR_BUILD_DIR must be configured")
        tool = self.settings.buddy_mlir_build_dir / "bin/buddy-opt"
        hw = schedule.snapshot
        convert = "builtin.module(convert-linalg-to-gemmini{acc_t=f32})"
        lower = (
            "builtin.module(convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
            "canonicalize,func.func(llvm-request-c-wrappers),lower-gemmini{"
            f"dim={hw.array.rows} bank_rows={hw.scratchpad.rows_per_bank} "
            f"acc_rows={hw.accumulator.banks * hw.accumulator.rows_per_bank} elem_t=f32 acc_t=f32"
            "},reconcile-unrealized-casts)"
        )
        bufferized = run_tool(tool, [f"--pass-pipeline={BUFFERIZE_PIPELINE}"], scheduled)
        gemmini = run_tool(tool, [f"--pass-pipeline={convert}"], bufferized)
        if "gemmini.tile_matmul" not in gemmini or "gemmini.schedule" not in gemmini or "linalg.matmul" in gemmini:
            raise ScheduleError(
                "Buddy did not honor the explicit Gemmini schedule; rebuild buddy-opt with this backend"
            )
        lowered = run_tool(tool, [f"--pass-pipeline={lower}"], gemmini)
        llvm_ir = run_tool(self.settings.buddy_mlir_build_dir / "bin/buddy-translate", ["--buddy-to-llvmir"], lowered)
        return GemminiCode(
            bufferized_mlir=bufferized,
            gemmini_mlir=gemmini,
            llvm_mlir=lowered,
            llvm_ir=llvm_ir,
            pipelines=[BUFFERIZE_PIPELINE, convert, lower],
        )
