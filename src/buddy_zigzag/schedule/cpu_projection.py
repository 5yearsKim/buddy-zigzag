"""Project ZigZag temporal and spatial factors onto a CPU loop nest."""

from math import prod
from pathlib import Path

from pydantic import ValidationError

from ..workload.schema import Layer
from .errors import ScheduleError
from .schema import CpuSchedule, build_loop_nest
from .zigzag_result import ZigzagMappingReader


class CpuScheduleProjector:
    def load(self, path: Path, layer: Layer, mlir_text: str, *, serialize_spatial: bool = False) -> CpuSchedule:
        values = ZigzagMappingReader().read(path, layer, mlir_text)
        spatial = {
            dim: prod(axis.get(dim, 1) for axis in values["spatial_by_axis"].values()) for dim in ("M", "N", "K")
        }
        flattened = [loop for level in values["temporal_by_operand"]["O"] for loop in level if loop.factor > 1]
        try:
            return CpuSchedule(
                **values,
                spatial_policy="serialize" if serialize_spatial else "reject",
                loop_nest=build_loop_nest(flattened, spatial),
            )
        except ValidationError as exc:
            raise ScheduleError(f"Unsupported or malformed ZigZag complete result: {exc}") from exc
