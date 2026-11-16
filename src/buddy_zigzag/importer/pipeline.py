"""The reverse bridge: a ZigZag result becomes scheduled and lowered MLIR."""

import json

from ..exporter.extract_mlir import LinalgExtractor
from ..exporter.utils import versions, write_json
from ..settings import Settings
from .apply_buddy import BuddyScheduleApplier
from .apply_gemmini import GemminiScheduleApplier
from .errors import ScheduleError
from .gemmini_schedule import GemminiScheduleImporter
from .import_zigzag import ZigzagScheduleImporter
from .lower_gemmini import GemminiLowerer
from .lowering import BUFFERIZE_PIPELINE, LLVM_PIPELINE, BuddyLowerer
from .schema import ImportRequest, ImportResult
from .utils import digest, require_supported_workload


class ZigzagImporter:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings if settings is not None else Settings.load()
        self.schedule_importer = ZigzagScheduleImporter()
        self.applier = BuddyScheduleApplier(self.settings)
        self.lowerer = BuddyLowerer(self.settings)

    def run(self, request: ImportRequest) -> ImportResult:
        folder = "gemmini_importer" if request.target == "gemmini" else "importer"
        out = (request.output_dir or request.export_dir / folder).resolve()
        out.mkdir(parents=True, exist_ok=True)
        if out == request.export_dir.resolve():
            raise ScheduleError("Choose a separate importer output directory to preserve the original export")
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
            text = (request.export_dir / "imported.mlir").read_text()
            manifest = json.loads((request.export_dir / "manifest.json").read_text())
            if manifest.get("mlir_sha256") != digest(text):
                raise ScheduleError(
                    "Exported MLIR no longer matches its manifest; re-export before importing a schedule"
                )
            layer = require_supported_workload(LinalgExtractor().extract(text))
            if request.target == "gemmini":
                return self._run_gemmini(request, out, text, layer, report)
            schedule = self.schedule_importer.load(
                request.result, layer, text, serialize_spatial=request.serialize_spatial
            )
            write_json(out / "schedule.json", schedule.model_dump(mode="json"))
            script, scheduled = self.applier.apply(text, schedule)
            (out / "transform.mlir").write_text(script)
            (out / "scheduled.mlir").write_text(scheduled)
            code = self.lowerer.lower(scheduled, layer)
            (out / "bufferized.mlir").write_text(code.bufferized_mlir)
            (out / "lowered.mlir").write_text(code.llvm_mlir)
            (out / "lowered.ll").write_text(code.llvm_ir)
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
                    "source_export": str(request.export_dir.resolve()),
                    "source_result": str(request.result.resolve()),
                    "mlir_sha256": schedule.mlir_sha256,
                    "result_sha256": schedule.result_sha256,
                    "scheduled_sha256": digest(scheduled),
                    "entry_point": code.entry_point,
                    "versions": versions(self.settings),
                },
            )
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            write_json(out / "validation.json", report)
            raise
        write_json(out / "validation.json", report)
        return ImportResult(output_dir=out, schedule=schedule, verification=verification)

    def _run_gemmini(self, request, out, text, layer, report):
        schedule = GemminiScheduleImporter().load(
            request.result, layer, text, request.snapshot, tiles=request.gemmini_tiles
        )
        report.update(execution_target=schedule.execution_target, memory_placement=schedule.memory_placement)
        write_json(out / "schedule.json", schedule.model_dump(mode="json"))
        scheduled = GemminiScheduleApplier().apply(text, schedule)
        (out / "scheduled.mlir").write_text(scheduled)
        code = GemminiLowerer(self.settings).lower(scheduled, schedule)
        for name, value in (
            ("bufferized.mlir", code.bufferized_mlir),
            ("gemmini.mlir", code.gemmini_mlir),
            ("lowered.mlir", code.llvm_mlir),
            ("lowered.ll", code.llvm_ir),
        ):
            (out / name).write_text(value)
        (out / "lowering_pipeline.txt").write_text("\n".join(code.pipelines) + "\n")
        verification = {"status": "skipped", "reason": "Use --verify to execute the tiled Linalg on CPU"}
        if request.verify:
            reference = self.lowerer.lower(scheduled, layer)
            (out / "cpu_reference.mlir").write_text(reference.llvm_mlir)
            verification = self.lowerer.verify(reference, layer, trials=request.trials)
            verification["scope"] = "tiled Linalg semantics; Gemmini instructions are not executed"
        report.update(
            status="passed",
            transformation="passed",
            lowering="passed",
            native_verification=verification,
            gemmini_execution={
                "status": "not_run",
                "reason": "RISC-V linking and Chipyard simulation are a later step",
            },
        )
        write_json(
            out / "manifest.json",
            {
                "source_export": str(request.export_dir.resolve()),
                "source_result": str(request.result.resolve()),
                "mlir_sha256": schedule.mlir_sha256,
                "result_sha256": schedule.result_sha256,
                "snapshot_sha256": schedule.snapshot_sha256,
                "scheduled_sha256": digest(scheduled),
                "execution_target": schedule.execution_target,
                "versions": versions(self.settings),
            },
        )
        write_json(out / "validation.json", report)
        return ImportResult(output_dir=out, schedule=schedule, verification=verification)
