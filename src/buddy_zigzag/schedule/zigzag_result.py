"""Read per-layer ZigZag 3.10 complete JSON results without evaluating strings."""

import json
import re
from pathlib import Path

from pydantic import ValidationError

from ..common.artifacts import digest
from ..workload.schema import Layer
from .errors import ScheduleError
from .schema import LoopFactor

PAIR = re.compile(r"^\(([A-Z][A-Z0-9]*), ([0-9]+)\)$")


class ZigzagMappingReader:
    def read(self, path: Path, layer: Layer, mlir_text: str) -> dict:
        """Read common provenance and factors; the selected backend validates their use."""
        text = path.read_text()
        try:
            result = json.loads(text)
            inputs = result["inputs"]
            node = inputs["layer"]
            dimensions = dict(zip(layer.loop_dims, layer.loop_sizes))
            if node["name"] != layer.name or node["loop_dimensions"] != dimensions:
                raise ScheduleError("ZigZag result layer name or dimensions do not match the exported MLIR")

            def equation(value):
                return re.sub(r"\s+", "", value).replace("+=", "=")

            expected_precision = {name: layer.operands[name].tensor.precision for name in ("I", "W", "O")}
            expected_precision["O_final"] = layer.operands["O"].tensor.precision
            if (
                node["type"] != "Gemm"
                or node["operand_precision"] != expected_precision
                or equation(node["equation"]) != equation(layer.equation)
            ):
                raise ScheduleError("ZigZag result computation or precision does not match the exported MLIR")
            temporal = {}
            if set(inputs["temporal_mapping"]) != {"I", "W", "O"}:
                raise ScheduleError("Expected I, W, O temporal mappings in a per-layer complete result")
            for name, levels in inputs["temporal_mapping"].items():
                if not isinstance(levels, list) or not levels:
                    raise ScheduleError("Temporal mapping must contain memory-level lists")
                parsed = []
                for level in levels:
                    if not isinstance(level, list):
                        raise ScheduleError("Each memory level must be a loop list")
                    parsed.append([self._parse_factor(value) for value in level])
                temporal[name] = parsed
            # Use requested physical unroll factors. The serialized integer
            # internal mapping can under-cover non-divisible workloads after
            # ZigZag's fractional-utilization adjustments; it is not a tile size.
            spatial_axes = node["user_spatial_mapping"]
            if not isinstance(spatial_axes, dict) or any(not re.fullmatch(r"D[0-9]+", key) for key in spatial_axes):
                raise ScheduleError("Expected a selected spatial mapping keyed by hardware axes")
            spatial = {dim: 1 for dim in ("M", "N", "K")}
            for mapping in spatial_axes.values():
                if not isinstance(mapping, dict):
                    raise ScheduleError("Each hardware axis must map dimensions to integer factors")
                for dim, factor in mapping.items():
                    loop = LoopFactor(dimension=dim, factor=factor)
                    spatial[loop.dimension] *= loop.factor
            return dict(
                layer_id=layer.id,
                layer_name=layer.name,
                dimensions=dimensions,
                mlir_sha256=digest(mlir_text),
                result_sha256=digest(text),
                temporal_by_operand=temporal,
                spatial_by_axis=spatial_axes,
            )
        except (KeyError, TypeError, ValidationError, json.JSONDecodeError) as exc:
            raise ScheduleError(f"Unsupported or malformed ZigZag complete result: {exc}") from exc

    def _parse_factor(self, value: str) -> LoopFactor:
        if not isinstance(value, str) or (match := PAIR.fullmatch(value)) is None:
            raise ScheduleError(f"Invalid ZigZag loop factor: {value!r}")
        return LoopFactor(dimension=match[1], factor=int(match[2]))
