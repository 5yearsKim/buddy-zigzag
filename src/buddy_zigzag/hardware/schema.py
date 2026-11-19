"""Resolved Gemmini parameters and explicit analytical modeling assumptions."""

from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from ..common.schema import NonNegativeInt, PositiveInt, Schema

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]
Cost = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class SnapshotSource(Schema):
    config_name: Annotated[str, Field(min_length=1)]
    origin: Literal["elaboration", "resolved_config"]
    gemmini_revision: Revision | None
    chipyard_revision: Revision | None
    gemmini_scala_dirty: StrictBool | None
    gemmini_scala_sha256: Digest | None


class ArithmeticType(Schema):
    kind: Literal["float", "sint", "uint"]
    bits: PositiveInt
    exponent_bits: PositiveInt | None = None
    significand_bits: PositiveInt | None = None
    recoded: StrictBool = False

    @model_validator(mode="after")
    def validate_width(self):
        if self.kind == "float":
            if self.exponent_bits is None or self.significand_bits is None:
                raise ValueError("Floating types require exponent and significand widths")
            if self.bits != self.exponent_bits + self.significand_bits + int(self.recoded):
                raise ValueError("Floating type widths do not match the stored bit width")
        elif self.exponent_bits is not None or self.significand_bits is not None or self.recoded:
            raise ValueError("Integer types cannot carry floating-point format fields")
        return self


class ArithmeticParams(Schema):
    input: ArithmeticType
    weight: ArithmeticType
    accumulator: ArithmeticType
    spatial_input: ArithmeticType
    spatial_weight: ArithmeticType
    spatial_output: ArithmeticType


class ArrayParams(Schema):
    mesh_rows: PositiveInt
    mesh_columns: PositiveInt
    tile_rows: PositiveInt
    tile_columns: PositiveInt
    rows: PositiveInt
    columns: PositiveInt
    tile_latency: NonNegativeInt
    mesh_output_delay: NonNegativeInt

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.rows != self.mesh_rows * self.tile_rows or self.columns != self.mesh_columns * self.tile_columns:
            raise ValueError("Resolved array dimensions do not match mesh and tile dimensions")
        return self


class MemoryParams(Schema):
    banks: PositiveInt
    rows_per_bank: PositiveInt
    row_bits: PositiveInt
    capacity_bits: PositiveInt
    single_ported: StrictBool
    read_latency: PositiveInt
    sub_banks: Annotated[int, Field(strict=True)] | None = None

    @model_validator(mode="after")
    def validate_capacity(self):
        if self.capacity_bits != self.banks * self.rows_per_bank * self.row_bits:
            raise ValueError("Resolved capacity must equal banks * rows_per_bank * row_bits")
        if self.row_bits % 8:
            raise ValueError("Only byte-aligned memory rows are supported")
        if self.sub_banks is not None and self.sub_banks != -1 and self.sub_banks < 1:
            raise ValueError("Sub-bank count must be positive or -1 for unspecified")
        return self


class DMAParams(Schema):
    buswidth_bits: PositiveInt
    max_bytes: PositiveInt


class ExternalMemoryParams(Schema):
    shared: StrictBool
    tilelink: StrictBool


class GemminiSnapshot(Schema):
    schema_version: Literal[1]
    source: SnapshotSource
    array: ArrayParams
    arithmetic: ArithmeticParams
    scratchpad: MemoryParams
    accumulator: MemoryParams
    dma: DMAParams
    dataflow: Literal["OS", "WS", "BOTH"]
    external_memory: ExternalMemoryParams

    @model_validator(mode="after")
    def validate_memory_widths(self):
        if self.scratchpad.row_bits != self.array.columns * self.arithmetic.input.bits:
            raise ValueError("Scratchpad row width does not match columns * input precision")
        if self.accumulator.row_bits != self.array.columns * self.arithmetic.accumulator.bits:
            raise ValueError("Accumulator row width does not match columns * accumulator precision")
        if self.arithmetic.input.bits != self.arithmetic.weight.bits:
            raise ValueError("Gemmini input and weight widths must agree")
        if self.dma.buswidth_bits % 8:
            raise ValueError("DMA bus width must be byte-aligned")
        return self


class MemoryCost(Schema):
    read_energy_pj: Cost
    write_energy_pj: Cost
    area: Cost


class RegisterModel(MemoryCost):
    latency_cycles: PositiveInt


class ExternalMemoryModel(MemoryCost):
    capacity_bits: PositiveInt
    latency_cycles: PositiveInt
    bandwidth_bits_per_cycle: PositiveInt | None = None


class ComputeModel(Schema):
    energy_per_mac_pj: Cost
    area_per_mac: Cost
    systolic_dimensions: list[Literal["D1", "D2"]]

    @model_validator(mode="after")
    def unique_dimensions(self):
        if len(set(self.systolic_dimensions)) != len(self.systolic_dimensions):
            raise ValueError("Systolic dimensions must be unique")
        return self


class GemminiModelTemplate(Schema):
    schema_version: Literal[1]
    name: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")]
    dataflow: Literal["WS"]
    cost_status: Literal["placeholder", "characterized"]
    cost_source: Annotated[str, Field(min_length=1)]
    assumptions: Annotated[list[str], Field(min_length=1)]
    compute: ComputeModel
    weight_register: RegisterModel
    scratchpad: MemoryCost
    accumulator: MemoryCost
    external_memory: ExternalMemoryModel
    scratchpad_port_model: Literal["one_shared_row_per_cycle"]
    accumulator_port_model: Literal["one_row_per_direction_when_dual_ported"]


class HardwareExportResult(Schema):
    hardware_path: str
    manifest_path: str
    validation: dict
