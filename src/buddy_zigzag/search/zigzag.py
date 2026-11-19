"""Run ZigZag's search and expose its selected per-layer result files."""

import math

from ..common.artifacts import write_json
from ..settings import Settings
from .schema import SearchRequest, SearchResult


class SearchError(ValueError):
    """A search did not produce usable per-layer mappings."""


class ZigzagSearchRunner:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings if settings is not None else Settings.load()

    def run(self, request: SearchRequest) -> SearchResult:
        self.settings.configure_runtime()
        for path in (request.workload, request.hardware, request.mapping):
            if not path.is_file():
                raise SearchError(f"Search input is unavailable: {path}")
        out = (request.output_dir or request.workload.parent / "zigzag").resolve()
        out.mkdir(parents=True, exist_ok=True)
        report = {"status": "running", "request": request.model_dump(mode="json")}
        write_json(out / "search.json", report)
        try:
            from zigzag.api import get_hardware_performance_zigzag

            energy, latency, evaluations = get_hardware_performance_zigzag(
                workload=str(request.workload.resolve()),
                accelerator=str(request.hardware.resolve()),
                mapping=str(request.mapping.resolve()),
                opt=request.objective,
                dump_folder=str(out),
                nb_spatial_mappings_generated=request.spatial_candidates,
                lpf_limit=request.lpf_limit,
                loma_show_progress_bar=request.show_progress,
            )
            if not evaluations or not all(math.isfinite(value) for value in (energy, latency)):
                raise SearchError("ZigZag did not produce finite mapping costs")
            # SumStage returns the selected per-layer evaluations as extra info.
            # Derive paths from this run so unrelated files from earlier searches
            # cannot be returned as selected mappings.
            mappings = tuple(
                out / f"{evaluation.layer.name.replace('/', '_')}_complete.json"
                for _, layers in evaluations
                for evaluation, _ in layers
            )
            if not mappings or any(not path.is_file() for path in mappings):
                raise SearchError("ZigZag did not write per-layer complete mapping results")
            result = SearchResult(output_dir=out, energy=energy, latency=latency, mappings=mappings)
            report.update(status="passed", result=result.model_dump(mode="json"))
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            write_json(out / "search.json", report)
            raise
        write_json(out / "search.json", report)
        return result
