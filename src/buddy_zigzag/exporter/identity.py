"""Attach persistent identities before handing a workload to ZigZag."""

from .errors import ExportError

LAYER_ID = "buddy_zigzag.layer_id"


class LinalgIdentity:
    def annotate(self, text: str) -> str:
        from buddy_mlir import ir

        from .extract_mlir import MATMUL_OPS

        with ir.Context():
            try:
                module = ir.Module.parse(text)
            except ir.MLIRError as exc:
                raise ExportError(f"Invalid MLIR: {exc}") from exc
            seen = set()
            index = 0
            for fn in module.body.operations:
                if fn.operation.name != "func.func":
                    continue
                for block in fn.regions[0].blocks:
                    for view in block.operations:
                        op = view.operation
                        if op.name not in MATMUL_OPS:
                            continue
                        if LAYER_ID in op.attributes:
                            layer_id = ir.IntegerAttr(op.attributes[LAYER_ID]).value
                        else:
                            layer_id = index
                        if layer_id < 0 or layer_id in seen:
                            raise ExportError("Linalg layer IDs must be unique and nonnegative")
                        # Transform attribute matching is type-sensitive.
                        op.attributes[LAYER_ID] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), layer_id)
                        seen.add(layer_id)
                        index += 1
            return module.operation.get_asm(enable_debug_info=True)
