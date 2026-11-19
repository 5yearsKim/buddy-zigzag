"""Mapping legality and spatial layout for Buddy's Gemmini WS matmul kernel."""

from ..hardware.schema import GemminiSnapshot
from ..workload.schema import Layer, Workload


class GemminiWsMatmulRule:
    dataflow = "WS"

    def generate(self, layer: Layer, workload: Workload, hardware: GemminiSnapshot) -> dict:
        array = hardware.array
        if array.rows != array.columns or array.tile_rows != 1 or array.tile_columns != 1:
            raise ValueError("Gemmini WS requires a square array with 1x1 PE tiles")
        if hardware.dataflow not in ("WS", "BOTH"):
            raise ValueError("The selected hardware does not support WS")
        if hardware.scratchpad.banks != 4 or hardware.external_memory.shared or hardware.external_memory.tilelink:
            raise ValueError("Buddy's Gemmini WS backend requires four internal scratchpad banks")
        for _, dtype in hardware.arithmetic:
            if (dtype.kind, dtype.bits, dtype.exponent_bits, dtype.significand_bits, dtype.recoded) != (
                "float",
                32,
                8,
                24,
                False,
            ):
                raise ValueError("Buddy's Gemmini WS backend requires FP32 hardware arithmetic")
        if layer.loop_dims != ("M", "N", "K") or layer.parallel_dims != ("M", "N") or layer.reduction_dims != ("K",):
            raise ValueError("Gemmini WS requires canonical M/N parallel loops and a K reduction")
        accesses = {"I": ("M", "K"), "W": ("K", "N"), "O": ("M", "N")}
        if any(operand.indices != accesses[name] for name, operand in layer.operands.items()):
            raise ValueError("Gemmini WS requires canonical, untransposed matrix accesses")
        if any(operand.tensor.dtype != "f32" for operand in layer.operands.values()):
            raise ValueError("Gemmini WS requires FP32 matrix operands")
        inputs = [layer.operands[name] for name in ("I", "W")]
        if any(operand.source_kind != "input" for operand in inputs) or {operand.source_name for operand in inputs} != {
            "arg0",
            "arg1",
        }:
            raise ValueError("Gemmini WS requires two distinct external matrix inputs, arg0 and arg1")
        if workload.outputs != [layer.operands["O"].source_name]:
            raise ValueError("Gemmini WS requires one returned matmul result")
        if (
            hardware.scratchpad.banks * hardware.scratchpad.rows_per_bank // 2 < 2 * array.rows
            or hardware.accumulator.banks * hardware.accumulator.rows_per_bank // 2 < array.rows
        ):
            raise ValueError("Gemmini memory cannot hold a minimum double-buffered matmul tile")
        return {
            "name": layer.name,
            "spatial_mapping": {"D1": [f"K, {array.rows}"], "D2": [f"N, {array.columns}"]},
            "memory_operand_links": {"O": "O", "I": "I1", "W": "I2"},
        }
