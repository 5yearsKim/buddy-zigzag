"""The validated contract between ZigZag results and compiler transforms."""

from math import prod
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..common.schema import NonNegativeInt, PositiveInt, Schema
from ..hardware.schema import GemminiSnapshot

Dimension = Literal["M", "N", "K"]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class LoopFactor(Schema):
    dimension: Dimension
    factor: PositiveInt


class LoopStep(LoopFactor):
    step: PositiveInt
    source: Literal["temporal", "serialized_spatial"]


def build_loop_nest(temporal: list[LoopFactor], spatial: dict[Dimension, int]) -> list[LoopStep]:
    # Temporal entries are innermost-first in ZigZag. Spatial dimensions execute
    # simultaneously there; this CPU projection serializes them in M,N,K order.
    factors = [(loop, "temporal") for loop in reversed(temporal) if loop.factor > 1]
    factors += [
        (LoopFactor(dimension=dim, factor=spatial[dim]), "serialized_spatial")
        for dim in ("M", "N", "K")
        if spatial[dim] > 1
    ]
    loops = []
    for index, (loop, source) in enumerate(factors):
        step = prod(inner.factor for inner, _ in factors[index + 1 :] if inner.dimension == loop.dimension)
        loops.append(LoopStep(dimension=loop.dimension, factor=loop.factor, step=step, source=source))
    return loops


class CpuSchedule(Schema):
    schema_version: Literal[1] = 1
    layer_id: NonNegativeInt
    layer_name: str
    mlir_sha256: Digest
    result_sha256: Digest
    dimensions: dict[Dimension, PositiveInt]
    temporal_by_operand: dict[Literal["I", "W", "O"], list[list[LoopFactor]]]
    spatial_by_axis: dict[str, dict[Dimension, PositiveInt]]
    spatial_policy: Literal["reject", "serialize"]
    spatial_mapping_source: Literal["inputs.layer.user_spatial_mapping"] = "inputs.layer.user_spatial_mapping"
    memory_placement: Literal["not_applied"] = "not_applied"
    execution_target: Literal["cpu_temporal_projection"] = "cpu_temporal_projection"
    loop_nest: list[LoopStep]

    @property
    def spatial_factors(self) -> dict[Dimension, int]:
        return {dim: prod(axis.get(dim, 1) for axis in self.spatial_by_axis.values()) for dim in ("M", "N", "K")}

    @model_validator(mode="after")
    def validate_coverage(self):
        if set(self.dimensions) != {"M", "N", "K"}:
            raise ValueError("Exactly M, N, K dimensions are required")
        if set(self.temporal_by_operand) != {"I", "W", "O"}:
            raise ValueError("Temporal mappings for I, W, O are required")
        flattened = {
            name: [loop for level in levels for loop in level if loop.factor > 1]
            for name, levels in self.temporal_by_operand.items()
        }
        temporal = flattened["O"]
        if any(loops != temporal for loops in flattened.values()):
            raise ValueError("Operand-specific temporal orders are unsupported; buffer scheduling is required")
        spatial = self.spatial_factors
        if self.spatial_policy == "reject" and any(factor > 1 for factor in spatial.values()):
            raise ValueError("Spatial mapping is present; explicitly enable spatial serialization for CPU verification")
        for dim, extent in self.dimensions.items():
            temporal_product = prod(loop.factor for loop in temporal if loop.dimension == dim)
            if temporal_product != (extent + spatial[dim] - 1) // spatial[dim]:
                raise ValueError(f"Mapping factors do not cover {dim} correctly")
        expected = build_loop_nest(temporal, spatial)
        if self.loop_nest != expected:
            raise ValueError("Loop nest disagrees with temporal/spatial factors")
        if any(loop.step >= 2**63 or loop.factor >= 2**63 for loop in self.loop_nest):
            raise ValueError("Loop factors exceed the supported index range")
        return self


class GemminiSchedule(Schema):
    schema_version: Literal[1] = 1
    execution_target: Literal["gemmini_ws_tile_projection"] = "gemmini_ws_tile_projection"
    layer_id: NonNegativeInt
    layer_name: str
    mlir_sha256: Digest
    result_sha256: Digest
    snapshot_sha256: Digest
    dimensions: dict[Dimension, PositiveInt]
    temporal_by_operand: dict[Literal["I", "W", "O"], list[list[LoopFactor]]]
    spatial_by_axis: dict[str, dict[Dimension, PositiveInt]]
    snapshot: GemminiSnapshot
    tiles: dict[Dimension, PositiveInt]
    requested_tiles: dict[Dimension, PositiveInt]
    tile_source: Literal["zigzag_memory_tiles", "explicit_override"]
    dataflow: Literal["WS"] = "WS"
    kernel_loop_order: tuple[Literal["M"], Literal["N"], Literal["K"]] = ("M", "N", "K")
    accumulation: Literal["resident_across_k"] = "resident_across_k"
    reuse: Literal["within_ws_kernel"] = "within_ws_kernel"
    memory_placement: Literal["gemmini_ws_kernel_managed"] = "gemmini_ws_kernel_managed"

    @model_validator(mode="after")
    def validate_hardware_schedule(self):
        hw = self.snapshot
        dim = hw.array.rows
        if hw.array.columns != dim or hw.array.tile_rows != 1 or hw.array.tile_columns != 1:
            raise ValueError("Gemmini scheduling requires a square array with 1x1 PE tiles")
        if hw.scratchpad.banks != 4 or hw.external_memory.shared or hw.external_memory.tilelink:
            raise ValueError("Buddy's initial Gemmini backend requires four internal scratchpad banks")
        if hw.dataflow not in ("WS", "BOTH"):
            raise ValueError("The selected hardware does not support WS")
        for _, dtype in hw.arithmetic:
            if (dtype.kind, dtype.bits, dtype.exponent_bits, dtype.significand_bits, dtype.recoded) != (
                "float",
                32,
                8,
                24,
                False,
            ):
                raise ValueError("Gemmini scheduling currently supports FP32 arithmetic only")
        if self.spatial_by_axis != {"D1": {"K": dim}, "D2": {"N": dim}}:
            raise ValueError(
                f"WS requires the physical mapping D1=K:{dim}, D2=N:{dim}; evaluate with an explicit WS mapping"
            )
        if any(set(values) != {"M", "N", "K"} for values in (self.dimensions, self.tiles, self.requested_tiles)):
            raise ValueError("Exactly M, N, K dimensions and tile sizes are required")
        if set(self.temporal_by_operand) != {"I", "W", "O"}:
            raise ValueError("Temporal mappings for I, W, O are required")
        for levels in self.temporal_by_operand.values():
            for axis, extent in self.dimensions.items():
                spatial = 1 if axis == "M" else dim
                count = prod(loop.factor for level in levels for loop in level if loop.dimension == axis)
                if count != (extent + spatial - 1) // spatial:
                    raise ValueError(f"Mapping factors do not cover {axis} correctly")
        blocks = {}
        for axis, tile in self.tiles.items():
            if tile % dim or tile > ((self.dimensions[axis] + dim - 1) // dim) * dim:
                raise ValueError("Gemmini tiles must be array multiples within the padded workload dimensions")
            blocks[axis] = tile // dim
            if blocks[axis] > 65535:
                raise ValueError("Gemmini tile exceeds the WS instruction's 16-bit block count")
        m, n, k = (blocks[axis] for axis in ("M", "N", "K"))
        if (m * k + k * n) * dim > hw.scratchpad.banks * hw.scratchpad.rows_per_bank // 2:
            raise ValueError("Gemmini tiles exceed double-buffered scratchpad capacity")
        if m * n * dim > hw.accumulator.banks * hw.accumulator.rows_per_bank // 2:
            raise ValueError("Gemmini tiles exceed double-buffered accumulator capacity")
        count = prod((self.dimensions[axis] + self.tiles[axis] - 1) // self.tiles[axis] for axis in ("M", "N"))
        if count > 4096:
            raise ValueError("Initial static Gemmini tiling supports at most 4096 output tiles")
        if count * ((self.dimensions["K"] + self.tiles["K"] - 1) // self.tiles["K"]) > 65536:
            raise ValueError("Initial Gemmini lowering supports at most 65536 WS kernel calls")
        return self
