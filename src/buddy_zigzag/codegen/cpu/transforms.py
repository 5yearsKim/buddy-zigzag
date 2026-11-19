"""Generate and execute Transform dialect tiling on the original payload."""

from ...common.artifacts import digest
from ...schedule.errors import ScheduleError
from ...schedule.schema import CpuSchedule
from ...settings import Settings
from ...workload.identity import LAYER_ID
from ..utils import run_tool


class CpuScheduleApplier:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_transform(self, schedule: CpuSchedule) -> str:
        lines = [
            "module attributes {transform.with_named_sequence} {",
            "  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {",
            f"    %target = transform.structured.match attributes {{{LAYER_ID} = {schedule.layer_id} : i64}} "
            "in %root : (!transform.any_op) -> !transform.any_op",
        ]
        target = "%target"
        for index, loop in enumerate(schedule.loop_nest):
            sizes = [loop.step if dim == loop.dimension else 0 for dim in ("M", "N", "K")]
            values = ", ".join(map(str, sizes))
            lines += [
                f"    // {loop.source}: {loop.dimension} factor {loop.factor}, inner span {loop.step}",
                f"    %tile{index}, %loop{index} = transform.structured.tile_using_for {target} "
                f"tile_sizes [{values}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
            ]
            target = f"%tile{index}"
        lines += ["    transform.yield", "  }", "}"]
        return "\n".join(lines) + "\n"

    def apply(self, text: str, schedule: CpuSchedule) -> tuple[str, str]:
        from buddy_mlir import ir

        if digest(text) != schedule.mlir_sha256:
            raise ScheduleError("Schedule was created for a different MLIR artifact")
        if self.settings.buddy_mlir_build_dir is None:
            raise ScheduleError("BUDDY_MLIR_BUILD_DIR must be configured")
        script = self.generate_transform(schedule)
        with ir.Context():
            module = ir.Module.parse(text)
            matches = []
            for fn in module.body.operations:
                if fn.operation.name != "func.func":
                    continue
                for block in fn.regions[0].blocks:
                    for view in block.operations:
                        op = view.operation
                        if (
                            LAYER_ID in op.attributes
                            and ir.IntegerAttr(op.attributes[LAYER_ID]).value == schedule.layer_id
                        ):
                            matches.append(op)
            if len(matches) != 1 or matches[0].name != "linalg.matmul":
                raise ScheduleError("Expected exactly one original matmul with the schedule's persistent layer ID")
            if str(ir.IntegerAttr(matches[0].attributes[LAYER_ID]).type) != "i64":
                raise ScheduleError("Persistent layer IDs must be i64 attributes; rebuild this workload")
            transform = ir.Module.parse(script)
            # Clone: moving the root operation of an owning Module would leave
            # two Python Module owners and cause invalid native teardown.
            module.body.append(transform.operation.clone())
            combined = str(module)
        transformed = run_tool(
            self.settings.buddy_mlir_build_dir / "bin/buddy-opt",
            ["--transform-interpreter"],
            combined,
        )
        with ir.Context():
            module = ir.Module.parse(transformed)
            for view in list(module.body.operations):
                op = view.operation
                if op.name == "builtin.module" and "transform.with_named_sequence" in op.attributes:
                    op.erase()
            loops = []

            def collect_loops(op):
                if op.name == "scf.for":
                    loops.append(op)
                return ir.WalkResult.ADVANCE

            module.operation.walk(collect_loops, walk_order=ir.WalkOrder.PRE_ORDER)
            if len(loops) != len(schedule.loop_nest):
                raise ScheduleError("Transform did not produce the expected scheduled loop nest")
            for op, loop in zip(loops, schedule.loop_nest):
                step = op.operands[2].owner
                if step.name != "arith.constant" or ir.IntegerAttr(step.attributes["value"]).value != loop.step:
                    raise ScheduleError("Transformed loop step disagrees with the projected schedule")
                op.attributes["buddy_zigzag.loop_dimension"] = ir.StringAttr.get(loop.dimension)
                op.attributes["buddy_zigzag.loop_source"] = ir.StringAttr.get(loop.source)
            if not module.operation.verify():
                raise ScheduleError("Scheduled MLIR failed verification")
            return script, module.operation.get_asm(enable_debug_info=True)
