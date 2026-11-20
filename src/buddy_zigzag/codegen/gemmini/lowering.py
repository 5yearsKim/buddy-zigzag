"""Lower explicit FP32 WS tiles to Gemmini operations and LLVM intrinsics."""

from ...common.schema import Schema
from ...schedule.errors import ScheduleError
from ...schedule.schema import GemminiSchedule
from ...settings import Settings
from ..cpu.lowering import BUFFERIZE_PIPELINE
from ..utils import run_tool
from .plugin import build_plugin


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
        plugin = build_plugin(self.settings)
        plugin_argument = f"--load-pass-plugin={plugin}"
        convert = "builtin.module(buddy-zigzag-convert-linalg-to-gemmini)"
        lower = (
            "builtin.module(convert-linalg-to-loops,buddy-zigzag-lower-gemmini-tiles{"
            f"dim={hw.array.rows} bank_rows={hw.scratchpad.rows_per_bank} "
            f"acc_rows={hw.accumulator.banks * hw.accumulator.rows_per_bank}"
            "},expand-strided-metadata,lower-affine,"
            "canonicalize,func.func(llvm-request-c-wrappers),lower-gemmini{"
            f"dim={hw.array.rows} bank_rows={hw.scratchpad.rows_per_bank} "
            f"acc_rows={hw.accumulator.banks * hw.accumulator.rows_per_bank} elem_t=f32 acc_t=f32"
            "},reconcile-unrealized-casts)"
        )
        bufferized = run_tool(tool, [f"--pass-pipeline={BUFFERIZE_PIPELINE}"], scheduled)
        try:
            gemmini = run_tool(tool, [plugin_argument, f"--pass-pipeline={convert}"], bufferized)
        except ScheduleError as exc:
            if "Failed to load passes" in str(exc) or "undefined symbol" in str(exc):
                raise ScheduleError(
                    "Could not load the Gemmini plugin. Check the matching LLVM build and Buddy's "
                    f"exported symbols; see docs/setup.md.\n{exc}"
                ) from exc
            raise
        if "gemmini.tile_matmul" not in gemmini or "gemmini.schedule" not in gemmini or "linalg.matmul" in gemmini:
            raise ScheduleError("The Gemmini pass plugin did not preserve the explicit schedule; check its build log")
        lowered = run_tool(tool, [plugin_argument, f"--pass-pipeline={lower}"], gemmini)
        llvm_ir = run_tool(self.settings.buddy_mlir_build_dir / "bin/buddy-translate", ["--buddy-to-llvmir"], lowered)
        return GemminiCode(
            bufferized_mlir=bufferized,
            gemmini_mlir=gemmini,
            llvm_mlir=lowered,
            llvm_ir=llvm_ir,
            pipelines=[BUFFERIZE_PIPELINE, convert, lower],
        )
