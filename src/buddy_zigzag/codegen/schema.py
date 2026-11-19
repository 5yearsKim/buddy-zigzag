"""Requests and results for schedule compilation."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..common.schema import PositiveInt, Schema
from ..schedule.schema import CpuSchedule, GemminiSchedule


class ScheduleCompileRequest(Schema):
    workload_dir: Path
    result: Path
    output_dir: Path | None = None
    serialize_spatial: bool = False
    verify: bool = False
    trials: Annotated[int, Field(strict=True, ge=1, le=100)] = 3
    target: Literal["cpu", "gemmini"] = "cpu"
    snapshot: Path | None = None
    gemmini_tiles: tuple[PositiveInt, PositiveInt, PositiveInt] | None = None

    @model_validator(mode="after")
    def validate_backend(self):
        if self.target == "gemmini":
            if self.snapshot is None:
                raise ValueError("Gemmini requires a resolved hardware --snapshot")
            if self.serialize_spatial:
                raise ValueError("Gemmini executes physical spatial lanes; omit --serialize-spatial")
        elif self.snapshot is not None or self.gemmini_tiles is not None:
            raise ValueError("Hardware snapshot and Gemmini tiles require --target gemmini")
        return self


class ScheduleCompileResult(Schema):
    output_dir: Path
    schedule: CpuSchedule | GemminiSchedule
    verification: dict
