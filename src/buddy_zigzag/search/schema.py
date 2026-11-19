"""Inputs and persisted outputs of ZigZag mapping search."""

from pathlib import Path
from typing import Literal

from ..common.schema import PositiveInt, Schema


class SearchRequest(Schema):
    workload: Path
    hardware: Path
    mapping: Path
    output_dir: Path | None = None
    objective: Literal["energy", "latency", "EDP"] = "latency"
    spatial_candidates: PositiveInt = 1
    lpf_limit: PositiveInt = 4
    show_progress: bool = False


class SearchResult(Schema):
    output_dir: Path
    energy: float
    latency: float
    mappings: tuple[Path, ...]
