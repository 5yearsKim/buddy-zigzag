"""Build a validated workload and preserve its original Linalg artifact."""

import hashlib
from pathlib import Path

from ..common.artifacts import versions, write_json
from ..examples.factory import CaseRepository, TorchWorkloadFactory
from ..settings import Settings
from .errors import WorkloadError
from .identity import LinalgIdentity
from .linalg import LinalgExtractor
from .schema import WorkloadBuildRequest, WorkloadBuildResult
from .torch_frontend import TorchFrontend
from .validation import WorkloadValidator
from .zigzag_yaml import ZigzagWorkloadWriter


class WorkloadBuilder:
    """Translate Torch examples or saved Linalg into a validated workload."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings if settings is not None else Settings.load()
        self.frontend = TorchFrontend()
        self.extractor = LinalgExtractor()
        self.identity = LinalgIdentity()
        self.writer = ZigzagWorkloadWriter()
        self.validator = WorkloadValidator()
        self.cases = CaseRepository()
        self.examples = TorchWorkloadFactory()

    def build_case(
        self,
        name: str,
        *,
        output_dir: Path | None = None,
        cases: Path | None = None,
    ) -> WorkloadBuildResult:
        return self.build(
            WorkloadBuildRequest(
                case=name,
                output_dir=output_dir,
                cases=cases,
            )
        )

    def build_mlir(
        self,
        path: Path,
        *,
        output_dir: Path | None = None,
    ) -> WorkloadBuildResult:
        return self.build(WorkloadBuildRequest(mlir=path, output_dir=output_dir))

    def build(self, request: WorkloadBuildRequest) -> WorkloadBuildResult:
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
                model, inputs, reference = self.examples.build(spec)
                text = self.frontend.import_model(model, inputs)
            else:
                text = request.mlir.read_text()
            text = self.identity.annotate(text)
            (out / "01_imported.mlir").write_text(text)
            workload = self.extractor.extract(text)
            write_json(out / "normalized.json", workload.to_dict())
            report["semantics"] = self.validator.validate_semantics(workload, spec.expected_macs if spec else None)
            if spec is not None and spec.expected_loops is not None:
                actual = [dict(zip(layer.loop_dims, layer.loop_sizes)) for layer in workload.layers]
                if actual != spec.expected_loops:
                    raise WorkloadError(f"Loop dimensions differ: expected {spec.expected_loops}, extracted {actual}")
            if reference is not None:
                report["reference"] = self.validator.validate_reference(workload, inputs, reference)
            else:
                report["reference"] = {"status": "skipped", "reason": "No Torch model supplied for MLIR input"}
            self.writer.write(workload, out / "workload.yaml")
            write_json(
                out / "manifest.json",
                {
                    "case": name,
                    "input": spec.model_dump(mode="json") if spec else {"mlir": str(request.mlir.resolve())},
                    "versions": versions(self.settings),
                    "mlir_sha256": hashlib.sha256(text.encode()).hexdigest(),
                },
            )
            report["status"] = "passed"
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            write_json(out / "validation.json", report)
            raise
        write_json(out / "validation.json", report)
        return WorkloadBuildResult(output_dir=out, workload=workload, report=report)
