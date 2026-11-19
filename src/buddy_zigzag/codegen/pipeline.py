"""Compile a selected ZigZag mapping into scheduled Linalg and LLVM IR."""

import json

from ..common.artifacts import digest, versions, write_json
from ..schedule.cpu_projection import CpuScheduleProjector
from ..schedule.errors import ScheduleError
from ..schedule.gemmini_projection import GemminiScheduleProjector
from ..settings import Settings
from ..workload.linalg import LinalgExtractor
from .cpu.lowering import BUFFERIZE_PIPELINE, LLVM_PIPELINE, CpuLowerer
from .cpu.transforms import CpuScheduleApplier
from .gemmini.lowering import GemminiLowerer
from .gemmini.transforms import GemminiScheduleApplier
from .schema import ScheduleCompileRequest, ScheduleCompileResult
from .utils import require_supported_workload


class ScheduleCompiler:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings if settings is not None else Settings.load()
        self.projector = CpuScheduleProjector()
        self.applier = CpuScheduleApplier(self.settings)
        self.lowerer = CpuLowerer(self.settings)

    def compile(self, request: ScheduleCompileRequest) -> ScheduleCompileResult:
        out = (request.output_dir or request.workload_dir / "lowered").resolve()
        out.mkdir(parents=True, exist_ok=True)
        if out == request.workload_dir.resolve():
            raise ScheduleError("Choose a separate compilation output directory to preserve the original workload")
        report = {
            "status": "running",
            "execution_target": "gemmini_ws_tile_projection"
            if request.target == "gemmini"
            else "cpu_temporal_projection",
            "memory_placement": "gemmini_ws_kernel_managed" if request.target == "gemmini" else "not_applied",
        }
        write_json(out / "validation.json", report)
        try:
            self.settings.configure_runtime()
            text = (request.workload_dir / "01_imported.mlir").read_text()
            manifest = json.loads((request.workload_dir / "manifest.json").read_text())
            if manifest.get("mlir_sha256") != digest(text):
                raise ScheduleError(
                    "Workload MLIR no longer matches its manifest; rebuild the workload before compiling a schedule"
                )
            layer = require_supported_workload(LinalgExtractor().extract(text))
            if request.target == "gemmini":
                return self._run_gemmini(request, out, text, layer, report)
            schedule = self.projector.load(request.result, layer, text, serialize_spatial=request.serialize_spatial)
            write_json(out / "schedule.json", schedule.model_dump(mode="json"))
            script, scheduled = self.applier.apply(text, schedule)
            (out / "02_transform.mlir").write_text(script)
            (out / "02_scheduled.mlir").write_text(scheduled)
            code = self.lowerer.lower(scheduled, layer)
            (out / "03_bufferized.mlir").write_text(code.bufferized_mlir)
            (out / "05_lowered.mlir").write_text(code.llvm_mlir)
            (out / "06_lowered.ll").write_text(code.llvm_ir)
            (out / "lowering_pipeline.txt").write_text(BUFFERIZE_PIPELINE + "\n" + LLVM_PIPELINE + "\n")
            verification = (
                self.lowerer.verify(code, layer, trials=request.trials)
                if request.verify
                else {
                    "status": "skipped",
                    "reason": "Use --verify to execute generated CPU code against Torch",
                }
            )
            report.update(
                status="passed",
                transformation="passed",
                lowering="passed",
                native_verification=verification,
                spatial_policy=schedule.spatial_policy,
            )
            write_json(
                out / "manifest.json",
                {
                    "source_workload": str(request.workload_dir.resolve()),
                    "source_result": str(request.result.resolve()),
                    "mlir_sha256": schedule.mlir_sha256,
                    "result_sha256": schedule.result_sha256,
                    "scheduled_sha256": digest(scheduled),
                    "entry_point": code.entry_point,
                    "llvm_ir_sha256": digest(code.llvm_ir),
                    "versions": versions(self.settings),
                },
            )
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            write_json(out / "validation.json", report)
            raise
        write_json(out / "validation.json", report)
        return ScheduleCompileResult(output_dir=out, schedule=schedule, verification=verification)

    def _run_gemmini(self, request, out, text, layer, report):
        schedule = GemminiScheduleProjector().load(
            request.result, layer, text, request.snapshot, tiles=request.gemmini_tiles
        )
        report.update(execution_target=schedule.execution_target, memory_placement=schedule.memory_placement)
        write_json(out / "schedule.json", schedule.model_dump(mode="json"))
        scheduled = GemminiScheduleApplier().apply(text, schedule)
        (out / "02_scheduled.mlir").write_text(scheduled)
        code = GemminiLowerer(self.settings).lower(scheduled, schedule)
        for name, value in (
            ("03_bufferized.mlir", code.bufferized_mlir),
            ("04_gemmini.mlir", code.gemmini_mlir),
            ("05_lowered.mlir", code.llvm_mlir),
            ("06_lowered.ll", code.llvm_ir),
        ):
            (out / name).write_text(value)
        (out / "lowering_pipeline.txt").write_text("\n".join(code.pipelines) + "\n")
        verification = {"status": "skipped", "reason": "Use --verify to execute the tiled Linalg on CPU"}
        if request.verify:
            reference = self.lowerer.lower(scheduled, layer)
            verification = self.lowerer.verify(reference, layer, trials=request.trials)
            verification["scope"] = "tiled Linalg semantics; Gemmini instructions are not executed"
        report.update(
            status="passed",
            transformation="passed",
            lowering="passed",
            native_verification=verification,
            gemmini_execution={
                "status": "not_run",
                "reason": "Build with Bazel and run Chipyard separately; see simulation/README.md",
            },
        )
        write_json(
            out / "manifest.json",
            {
                "source_workload": str(request.workload_dir.resolve()),
                "source_result": str(request.result.resolve()),
                "mlir_sha256": schedule.mlir_sha256,
                "result_sha256": schedule.result_sha256,
                "snapshot_sha256": schedule.snapshot_sha256,
                "scheduled_sha256": digest(scheduled),
                "execution_target": schedule.execution_target,
                "llvm_ir_sha256": digest(code.llvm_ir),
                "versions": versions(self.settings),
            },
        )
        write_json(out / "validation.json", report)
        return ScheduleCompileResult(output_dir=out, schedule=schedule, verification=verification)
