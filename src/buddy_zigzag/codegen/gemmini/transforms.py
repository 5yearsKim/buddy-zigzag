"""Runtime M/N tile loops with static tail shapes; Gemmini owns K."""

from ...common.artifacts import digest
from ...schedule.errors import ScheduleError
from ...schedule.schema import GemminiSchedule
from ...workload.identity import LAYER_ID


class GemminiScheduleApplier:
    def apply(self, text: str, schedule: GemminiSchedule) -> str:
        from buddy_mlir import ir
        from buddy_mlir.dialects import arith, linalg, scf, tensor

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
            tile_m, tile_n = (schedule.tiles[axis] for axis in ("M", "N"))
            full_m, full_n = m // tile_m * tile_m, n // tile_n * tile_n

            def split_offsets(values):
                # Only offsets depend on loop induction variables. Shapes and
                # parent row strides stay static for Buddy's Gemmini lowerer.
                dynamic = [value for value in values if not isinstance(value, int)]
                static = [value if isinstance(value, int) else ir.ShapedType.get_dynamic_size() for value in values]
                return dynamic, static

            def extract(value, offsets, sizes):
                dynamic, static = split_offsets(offsets)
                return tensor.ExtractSliceOp(
                    ir.RankedTensorType.get(sizes, dtype), value, dynamic, [], [], static, sizes, [1, 1]
                ).result

            def emit_tile(current, row, col, rows, cols):
                lhs = extract(a, [row, 0], [rows, k])
                rhs = extract(b, [0, col], [k, cols])
                dest = extract(current, [row, col], [rows, cols])
                result = linalg.matmul(lhs, rhs, outs=[dest])
                result.owner.attributes["gemmini.schedule"] = attrs
                result.owner.attributes[LAYER_ID] = target.attributes[LAYER_ID]
                dynamic, static = split_offsets([row, col])
                return tensor.InsertSliceOp(result, current, dynamic, [], [], static, [rows, cols], [1, 1]).result

            with ir.InsertionPoint(target), target.location:
                index_type = ir.IndexType.get()
                zero = arith.ConstantOp(index_type, 0).result
                m_end = arith.ConstantOp(index_type, full_m).result
                n_end = arith.ConstantOp(index_type, full_n).result
                m_step = arith.ConstantOp(index_type, tile_m).result
                n_step = arith.ConstantOp(index_type, tile_n).result

                def emit_row(current, row, rows):
                    if full_n:
                        loop = scf.ForOp(zero, n_end, n_step, iter_args=[current])
                        with ir.InsertionPoint(loop.body):
                            updated = emit_tile(loop.inner_iter_args[0], row, loop.induction_variable, rows, tile_n)
                            scf.YieldOp([updated])
                        current = loop.result
                    if full_n != n:
                        current = emit_tile(current, row, full_n, rows, n - full_n)
                    return current

                # Carry the output tensor through each loop so tile updates
                # accumulate in row-major order. K is kept whole in each matmul.
                if full_m:
                    loop = scf.ForOp(zero, m_end, m_step, iter_args=[output])
                    with ir.InsertionPoint(loop.body):
                        updated = emit_row(loop.inner_iter_args[0], loop.induction_variable, tile_m)
                        scf.YieldOp([updated])
                    output = loop.result
                # Separate right/bottom tails keep every tile shape constant:
                # at most four matmul bodies, regardless of the M/N tile count.
                if full_m != m:
                    output = emit_row(output, full_m, m - full_m)
            target.results[0].replace_all_uses_with(output)
            target.erase()
            if not module.operation.verify():
                raise ScheduleError("Gemmini-sized Linalg failed verification")
            return module.operation.get_asm(enable_debug_info=True)
