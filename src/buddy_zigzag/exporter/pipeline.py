"""Class-based public export API, shared by Python callers and the CLI."""

import hashlib
from importlib.resources import files
from pathlib import Path

from ..settings import Settings
from ..workloads.factory import CaseRepository, TorchWorkloadFactory
from .emit_workload import ZigzagWorkloadWriter
from .errors import ExportError
from .extract_mlir import LinalgExtractor
from .identity import LinalgIdentity
from .import_torch import TorchImporter
from .schema import ExportRequest, ExportResult
from .utils import versions, write_json
from .validate import ExportValidator


class ZigzagExporter:
    """Import, extract, serialize, and validate a workload in one place."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings if settings is not None else Settings.load()
        self.importer = TorchImporter()
        self.extractor = LinalgExtractor()
        self.identity = LinalgIdentity()
        self.writer = ZigzagWorkloadWriter()
        self.validator = ExportValidator()
        self.cases = CaseRepository()
        self.workloads = TorchWorkloadFactory()

    def export_case(
        self,
        name: str,
        *,
        output_dir: Path | None = None,
        cases: Path | None = None,
        mapping: Path | None = None,
        skip_zigzag: bool = False,
    ) -> ExportResult:
        return self.export(
            ExportRequest(
                case=name,
                output_dir=output_dir,
                cases=cases,
                mapping=mapping,
                skip_zigzag=skip_zigzag,
            )
        )

    def export_mlir(
        self,
        path: Path,
        *,
        output_dir: Path | None = None,
        mapping: Path | None = None,
        skip_zigzag: bool = False,
    ) -> ExportResult:
        return self.export(ExportRequest(mlir=path, output_dir=output_dir, mapping=mapping, skip_zigzag=skip_zigzag))

    def export(self, request: ExportRequest) -> ExportResult:
        name = request.case if request.case is not None else request.mlir.stem
        out = request.output_dir if request.output_dir is not None else self.settings.artifacts_dir / name
        out = out.resolve()
        out.mkdir(parents=True, exist_ok=True)
        report = {"status": "running"}
        write_json(out / "validation.json", report)
        try:
            self.settings.configure_runtime()
            spec = None
            inputs = reference = None
            if request.case is not None:
                spec = self.cases.load(request.case, request.cases)
                model, inputs, reference = self.workloads.build(spec)
                text = self.importer.import_model(model, inputs)
            else:
                text = request.mlir.read_text()
            text = self.identity.annotate(text)
            write_json(
                out / "manifest.json",
                {
                    "case": name,
                    "input": spec.model_dump(mode="json") if spec else {"mlir": str(request.mlir.resolve())},
                    "versions": versions(self.settings),
                    "mlir_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "mapping_mode": "baseline or supplied mapping; Buddy schedule export is not implemented",
                },
            )
            (out / "imported.mlir").write_text(text)
            workload = self.extractor.extract(text)
            write_json(out / "normalized.json", workload.to_dict())
            report["semantics"] = self.validator.validate_semantics(workload, spec.expected_macs if spec else None)
            if spec is not None and spec.expected_loops is not None:
                actual = [dict(zip(layer.loop_dims, layer.loop_sizes)) for layer in workload.layers]
                if actual != spec.expected_loops:
                    raise ExportError(f"Loop dimensions differ: expected {spec.expected_loops}, extracted {actual}")
            if reference is not None:
                report["reference"] = self.validator.validate_reference(workload, inputs, reference)
            else:
                report["reference"] = {"status": "skipped", "reason": "No Torch model supplied for MLIR input"}
            self.writer.write(workload, out / "workload.yaml")
            source = request.mapping or files("buddy_zigzag").joinpath("configs/mapping.yaml")
            mapping_text = source.read_text()
            (out / "mapping.yaml").write_text(mapping_text)
            if request.skip_zigzag:
                report["zigzag"] = {"status": "skipped", "reason": "--skip-zigzag requested"}
            else:
                report["zigzag"] = self.validator.validate_zigzag(out / "workload.yaml", out / "mapping.yaml", workload)
            report["status"] = "passed"
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            write_json(out / "validation.json", report)
            raise
        write_json(out / "validation.json", report)
        return ExportResult(output_dir=out, workload=workload, report=report)
