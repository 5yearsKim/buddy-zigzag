"""Validated computation semantics, independent of hardware scheduling."""

import re
from math import prod
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, computed_field, model_validator

from ..common.schema import NonNegativeInt, PositiveInt, Schema

Dimension = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9]*$")]


class TensorSpec(Schema):
    shape: Annotated[tuple[PositiveInt, ...], Field(min_length=1)]
    dtype: str
    precision: PositiveInt

    @model_validator(mode="after")
    def check_precision(self):
        width = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}.get(self.dtype)
        match = re.fullmatch(r"[su]?i(\d+)", self.dtype)
        if match:
            width = int(match[1])
        if width is None or width != self.precision:
            raise ValueError("Element type and precision must agree")
        return self


class Operand(Schema):
    tensor: TensorSpec
    indices: Annotated[tuple[Dimension, ...], Field(min_length=1)]
    source_kind: Literal["input", "constant", "layer"]
    source_name: Annotated[str, Field(min_length=1)]
    source_id: NonNegativeInt | None = None

    @model_validator(mode="after")
    def check_source(self):
        if (self.source_kind == "layer") != (self.source_id is not None):
            raise ValueError("Only layer operands must carry a source_id")
        return self


class Layer(Schema):
    id: NonNegativeInt
    name: Annotated[str, Field(min_length=1)]
    mlir_op: str
    location: str
    loop_dims: Annotated[tuple[Dimension, ...], Field(min_length=1)]
    loop_sizes: Annotated[tuple[PositiveInt, ...], Field(min_length=1)]
    parallel_dims: tuple[Dimension, ...]
    reduction_dims: tuple[Dimension, ...]
    operands: dict[Literal["I", "W", "O"], Operand]

    @model_validator(mode="after")
    def check_dimensions(self):
        sizes = dict(zip(self.loop_dims, self.loop_sizes))
        if len(sizes) != len(self.loop_dims) or len(self.loop_sizes) != len(self.loop_dims):
            raise ValueError("Loop dimensions must be unique and match loop sizes")
        parallel, reduction = set(self.parallel_dims), set(self.reduction_dims)
        if (
            len(parallel) != len(self.parallel_dims)
            or len(reduction) != len(self.reduction_dims)
            or parallel & reduction
            or parallel | reduction != set(sizes)
        ):
            raise ValueError("Parallel and reduction dimensions must partition the loop dimensions")
        if set(self.operands) != {"I", "W", "O"}:
            raise ValueError("Exactly I, W, and O operands are required")
        for name, operand in self.operands.items():
            if not set(operand.indices) <= set(sizes):
                raise ValueError(f"{name} references an unknown loop dimension")
            if tuple(sizes[dim] for dim in operand.indices) != operand.tensor.shape:
                raise ValueError(f"{name} shape does not agree with its access expression")
        output = self.operands["O"]
        if output.source_kind != "layer" or output.source_id != self.id:
            raise ValueError("Output must identify its own layer as producer")
        if set(output.indices) != parallel:
            raise ValueError("Output accesses must cover the parallel dimensions")
        return self

    @computed_field
    @property
    def mac_count(self) -> int:
        return prod(self.loop_sizes)

    @computed_field
    @property
    def equation(self) -> str:
        def access(name):
            return name + "".join(f"[{dim.lower()}]" for dim in self.operands[name].indices)

        return f"{access('O')}+={access('I')}*{access('W')}"


class AuxiliaryOperation(Schema):
    operation: str
    location: str
    role: str


class Workload(Schema):
    schema_version: Literal[1] = 1
    layers: Annotated[list[Layer], Field(min_length=1)]
    auxiliary_operations: list[AuxiliaryOperation] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_producers(self):
        prior = {}
        for layer in self.layers:
            if layer.id in prior:
                raise ValueError(f"Duplicate layer ID {layer.id}")
            for name in ("I", "W"):
                operand = layer.operands[name]
                if operand.source_kind == "layer":
                    if operand.source_id not in prior:
                        raise ValueError(f"{layer.name}: producer must precede consumer")
                    if prior[operand.source_id].operands["O"].tensor != operand.tensor:
                        raise ValueError(f"{layer.name}: producer and consumer types differ")
            prior[layer.id] = layer
        return self

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class WorkloadCase(Schema):
    kind: Literal["matmul", "batch_matmul", "matmul_chain"]
    shapes: tuple[tuple[PositiveInt, ...], ...]
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    seed: NonNegativeInt = 0
    expected_macs: list[PositiveInt] | None = None
    expected_loops: list[dict[Dimension, PositiveInt]] | None = None

    @model_validator(mode="after")
    def check_shapes(self):
        count = 3 if self.kind == "matmul_chain" else 2
        rank = 3 if self.kind == "batch_matmul" else 2
        if len(self.shapes) != count or any(len(shape) != rank for shape in self.shapes):
            raise ValueError(f"{self.kind} requires {count} rank-{rank} input shapes")
        a, b = self.shapes[:2]
        if a[-1] != b[-2] or (rank == 3 and a[0] != b[0]):
            raise ValueError("Matmul input shapes have incompatible reduction or batch dimensions")
        if count == 3 and b[-1] != self.shapes[2][-2]:
            raise ValueError("Matmul chain has incompatible second reduction dimension")
        layers = 2 if count == 3 else 1
        if self.expected_macs is not None and len(self.expected_macs) != layers:
            raise ValueError("Expected MAC counts must match the number of layers")
        if self.expected_loops is not None and len(self.expected_loops) != layers:
            raise ValueError("Expected loop descriptions must match the number of layers")
        return self


class WorkloadBuildRequest(Schema):
    case: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")] | None = None
    mlir: Path | None = None
    cases: Path | None = None
    output_dir: Path | None = None

    @model_validator(mode="after")
    def check_source(self):
        if (self.case is None) == (self.mlir is None):
            raise ValueError("Provide exactly one of case or mlir")
        return self


class WorkloadBuildResult(Schema):
    output_dir: Path
    workload: Workload
    report: dict[str, Any]
