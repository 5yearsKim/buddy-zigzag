"""Project ZigZag memory tiles onto the supported Gemmini WS kernel."""

import json
from math import prod
from pathlib import Path

from ..exporter.schema import Layer
from ..hardware.schema import GemminiSnapshot
from .errors import ScheduleError
from .import_zigzag import ZigzagScheduleImporter
from .schema import GemminiSchedule
from .utils import digest


class GemminiScheduleImporter:
    def load(
        self,
        path: Path,
        layer: Layer,
        text: str,
        snapshot_path: Path,
        *,
        tiles: tuple[int, int, int] | None = None,
    ) -> GemminiSchedule:
        values = ZigzagScheduleImporter().read_mapping(path, layer, text)
        snapshot_text = snapshot_path.read_text()
        hw = GemminiSnapshot.model_validate_json(snapshot_text)
        accelerator = json.loads(path.read_text())["inputs"]["accelerator"]
        dim = hw.array.rows
        sizes = accelerator.get("mem_size_dict", {})
        if (
            accelerator.get("operational_array", {}).get("dimensions") != {"D1": dim, "D2": dim}
            or sizes.get("I1", [])[:1] != [hw.scratchpad.capacity_bits]
            or sizes.get("I2", [])[:2] != [32, hw.scratchpad.capacity_bits]
            or sizes.get("O", [])[:1] != [hw.accumulator.capacity_bits]
        ):
            raise ScheduleError("ZigZag accelerator array/memory structure does not match the Gemmini snapshot")
        mapping = values["temporal_by_operand"]
        if {name: len(levels) for name, levels in mapping.items()} != {"I": 2, "W": 3, "O": 2}:
            raise ScheduleError("Expected Gemmini scratchpad/weight-register/accumulator memory levels")

        def factor(operand, axis, count):
            return prod(loop.factor for level in mapping[operand][:count] for loop in level if loop.dimension == axis)

        requested = (
            dict(zip(("M", "N", "K"), tiles))
            if tiles
            else {
                "M": factor("O", "M", 1),
                "N": dim * factor("O", "N", 1),
                "K": dim * min(factor("I", "K", 1), factor("W", "K", 2)),
            }
        )
        selected = dict(requested)
        if tiles is None:
            # Array alignment and double buffering are Gemmini constraints that
            # ZigZag's aggregated memory model does not impose.
            blocks = {
                axis: min((size + dim - 1) // dim, (values["dimensions"][axis] + dim - 1) // dim)
                for axis, size in requested.items()
            }
            while True:
                m, n, k = (blocks[axis] for axis in ("M", "N", "K"))
                spad_ok = (m * k + k * n) * dim <= hw.scratchpad.capacity_bits // hw.scratchpad.row_bits // 2
                acc_ok = m * n * dim <= hw.accumulator.capacity_bits // hw.accumulator.row_bits // 2
                if spad_ok and acc_ok:
                    break
                candidates = [axis for axis in ("M", "N", "K") if blocks[axis] > 1 and (acc_ok or axis != "K")]
                if not candidates:
                    raise ScheduleError("Gemmini memory cannot hold the minimum double-buffered array tile")
                axis = max(candidates, key=blocks.get)
                blocks[axis] = (blocks[axis] + 1) // 2
            selected = {axis: count * dim for axis, count in blocks.items()}
        return GemminiSchedule(
            **values,
            snapshot=hw,
            snapshot_sha256=digest(snapshot_text),
            requested_tiles=requested,
            tiles=selected,
            tile_source="explicit_override" if tiles else "zigzag_memory_tiles",
        )
