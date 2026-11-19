"""Coordinate workload preparation, mapping constraints, search, and compilation."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path

from .codegen import ScheduleCompiler, ScheduleCompileRequest, ScheduleCompileResult
from .common.artifacts import write_json
from .hardware.schema import GemminiSnapshot
from .mapping import MappingConstraintGenerator
from .search import SearchRequest, SearchResult, ZigzagSearchRunner
from .settings import Settings
from .workload import WorkloadBuilder, WorkloadBuildRequest, WorkloadBuildResult
from .workload.validation import WorkloadValidator


class PrepareRequest(WorkloadBuildRequest):
    mapping: Path | None = None
    snapshot: Path | None = None
    skip_zigzag: bool = False


class DsePipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings if settings is not None else Settings.load()
        self.builder = WorkloadBuilder(self.settings)
        self.mapping_generator = MappingConstraintGenerator()
        self.search_runner = ZigzagSearchRunner(self.settings)
        self.compiler = ScheduleCompiler(self.settings)

    def prepare_case(self, name: str, **kwargs) -> WorkloadBuildResult:
        return self.prepare(PrepareRequest(case=name, **kwargs))

    def prepare_mlir(self, path: Path, **kwargs) -> WorkloadBuildResult:
        return self.prepare(PrepareRequest(mlir=path, **kwargs))

    def prepare(self, request: PrepareRequest) -> WorkloadBuildResult:
        build_request = WorkloadBuildRequest.model_validate(
            request.model_dump(include=set(WorkloadBuildRequest.model_fields))
        )
        built = self.builder.build(build_request)
        out, report = built.output_dir, built.report
        report["status"] = "running"
        write_json(out / "validation.json", report)
        try:
            metadata = {}
            if request.mapping is not None:
                mode = "supplied"
                (out / "mapping.yaml").write_text(request.mapping.read_text())
            elif request.snapshot is not None:
                mode = "generated"
                snapshot_text = request.snapshot.read_text()
                snapshot = GemminiSnapshot.model_validate_json(snapshot_text)
                self.mapping_generator.write(built.workload, snapshot, out / "mapping.yaml")
                metadata = {
                    "mapping_backend": "gemmini_ws",
                    "snapshot_sha256": hashlib.sha256(snapshot_text.encode()).hexdigest(),
                }
            else:
                mode = "baseline"
                source = files("buddy_zigzag").joinpath("configs/mapping.yaml")
                (out / "mapping.yaml").write_text(source.read_text())
            manifest = json.loads((out / "manifest.json").read_text())
            manifest.update(
                mapping_mode=mode,
                mapping_sha256=hashlib.sha256((out / "mapping.yaml").read_bytes()).hexdigest(),
                **metadata,
            )
            write_json(out / "manifest.json", manifest)
            report["zigzag"] = (
                {"status": "skipped", "reason": "--skip-zigzag requested"}
                if request.skip_zigzag
                else WorkloadValidator().validate_zigzag(out / "workload.yaml", out / "mapping.yaml", built.workload)
            )
            report["status"] = "passed"
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            write_json(out / "validation.json", report)
            raise
        write_json(out / "validation.json", report)
        return built

    def search(self, request: SearchRequest) -> SearchResult:
        return self.search_runner.run(request)

    def compile(self, request: ScheduleCompileRequest) -> ScheduleCompileResult:
        return self.compiler.compile(request)
