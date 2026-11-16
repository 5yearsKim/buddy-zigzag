"""Static M/N Linalg tiles; Gemmini owns the complete K reduction."""

from ..exporter.identity import LAYER_ID
from .errors import ScheduleError
from .schema import GemminiSchedule
from .utils import digest


class GemminiScheduleApplier:
    def apply(self, text: str, schedule: GemminiSchedule) -> str:
        from buddy_mlir import ir
        from buddy_mlir.dialects import linalg, tensor

        if digest(text) != schedule.mlir_sha256:
            raise ScheduleError("Schedule was created for a different MLIR artifact")
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.parse(text)
            matches = []

            def find(op):
                if op.name == "linalg.matmul" and LAYER_ID in op.attributes:
                    if ir.IntegerAttr(op.attributes[LAYER_ID]).value == schedule.layer_id:
                        matches.append(op)
                return ir.WalkResult.ADVANCE

            module.operation.walk(find)
            if len(matches) != 1:
                raise ScheduleError("Expected exactly one original matmul with the schedule's persistent layer ID")
            target = matches[0]
            if str(ir.IntegerAttr(target.attributes[LAYER_ID]).type) != "i64":
                raise ScheduleError("Persistent layer IDs must be i64 attributes")
            a, b, output = target.operands
            dtype = ir.RankedTensorType(a.type).element_type
            i64 = ir.IntegerType.get_signless(64)
            hw = schedule.snapshot
            metadata = {
                "tile_m": schedule.tiles["M"],
                "tile_n": schedule.tiles["N"],
                "tile_k": schedule.tiles["K"],
                "array_dim": hw.array.rows,
                "bank_rows": hw.scratchpad.rows_per_bank,
                "acc_rows": hw.accumulator.banks * hw.accumulator.rows_per_bank,
                "dataflow": 1,
            }
            attrs = ir.DictAttr.get({key: ir.IntegerAttr.get(i64, value) for key, value in metadata.items()})
            m, n, k = (schedule.dimensions[axis] for axis in ("M", "N", "K"))

            def extract(value, offsets, sizes):
                return tensor.ExtractSliceOp(
                    ir.RankedTensorType.get(sizes, dtype), value, [], [], [], offsets, sizes, [1, 1]
                ).result

            # Static slices keep tail shapes and parent strides explicit. This
            # initial implementation deliberately bounds compile-time expansion.
            with ir.InsertionPoint(target), target.location:
                for row in range(0, m, schedule.tiles["M"]):
                    for col in range(0, n, schedule.tiles["N"]):
                        rows, cols = min(schedule.tiles["M"], m - row), min(schedule.tiles["N"], n - col)
                        lhs = extract(a, [row, 0], [rows, k])
                        rhs = extract(b, [0, col], [k, cols])
                        dest = extract(output, [row, col], [rows, cols])
                        result = linalg.matmul(lhs, rhs, outs=[dest])
                        result.owner.attributes["gemmini.schedule"] = attrs
                        result.owner.attributes[LAYER_ID] = target.attributes[LAYER_ID]
                        output = tensor.InsertSliceOp(
                            result, output, [], [], [], [row, col], [rows, cols], [1, 1]
                        ).result
            target.results[0].replace_all_uses_with(output)
            target.erase()
            if not module.operation.verify():
                raise ScheduleError("Gemmini-sized Linalg failed verification")
            return module.operation.get_asm(enable_debug_info=True)
