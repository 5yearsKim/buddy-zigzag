"""Serialize semantic workloads to the ZigZag 3.10 workload schema."""

from pathlib import Path

import yaml

from .schema import Workload


class ZigzagWorkloadWriter:
    """Serialize the validated workload using ZigZag operand conventions."""

    def to_data(self, workload: Workload) -> list[dict]:
        layers = []
        for layer in workload.layers:
            sources = {}
            for name in ("I", "W"):
                operand = layer.operands[name]
                if operand.source_kind == "layer":
                    sources[name] = operand.source_id
                elif operand.source_kind == "constant":
                    # ZigZag denotes constants using a self-reference. External
                    # function inputs have no producer entry, and are not constants.
                    sources[name] = layer.id
            layers.append(
                {
                    "id": layer.id,
                    "name": layer.name,
                    "operator_type": "Gemm",
                    "equation": layer.equation,
                    "dimension_relations": [],
                    "loop_dims": list(layer.loop_dims),
                    "loop_sizes": list(layer.loop_sizes),
                    "operand_precision": {
                        **{name: layer.operands[name].tensor.precision for name in ("I", "W", "O")},
                        "O_final": layer.operands["O"].tensor.precision,
                    },
                    "operand_source": sources,
                }
            )
        return layers

    def write(self, workload: Workload, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.to_data(workload), sort_keys=False))
