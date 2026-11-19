"""Select operation-specific rules and emit validated ZigZag mapping input."""

from pathlib import Path

import yaml

from ..hardware.schema import GemminiSnapshot
from ..workload.schema import Workload
from .gemmini_ws import GemminiWsMatmulRule
from .schema import GEMMINI_WS, BackendCapabilities


class MappingError(ValueError):
    """The workload/hardware/backend combination has no supported mapping."""


class MappingConstraintGenerator:
    def __init__(self):
        self._rules = {("gemmini_ws", "linalg.matmul"): GemminiWsMatmulRule()}

    def generate(
        self, workload: Workload, hardware: GemminiSnapshot, backend: BackendCapabilities = GEMMINI_WS
    ) -> list[dict]:
        if len(workload.layers) > backend.max_layers:
            raise MappingError(f"Backend {backend.name} supports at most {backend.max_layers} layer(s) per workload")
        names = [layer.name for layer in workload.layers]
        if len(names) != len(set(names)) or "default" in names:
            raise MappingError("Layer-specific mappings require unique names other than 'default'")
        mappings = [{"name": "default", "memory_operand_links": {"O": "O", "I": "I1", "W": "I2"}}]
        for layer in workload.layers:
            rule = self._rules.get((backend.name, layer.mlir_op))
            if layer.mlir_op not in backend.operations or rule is None:
                raise MappingError(f"{layer.name}: backend {backend.name} does not support {layer.mlir_op}")
            if backend.dataflow != rule.dataflow:
                raise MappingError(f"{layer.name}: the {backend.name} rule requires {rule.dataflow} dataflow")
            if any(operand.tensor.dtype not in backend.dtypes for operand in layer.operands.values()):
                raise MappingError(f"{layer.name}: backend {backend.name} supports only {', '.join(backend.dtypes)}")
            try:
                mappings.append(rule.generate(layer, workload, hardware))
            except ValueError as exc:
                raise MappingError(f"{layer.name}: {exc}") from exc
        from zigzag.parser.mapping_validator import MappingValidator

        validator = MappingValidator(mappings)
        if not validator.validate():
            raise MappingError(f"ZigZag rejected generated mapping: {validator.validator.errors}")
        return mappings

    def write(
        self, workload: Workload, hardware: GemminiSnapshot, path: Path, backend: BackendCapabilities = GEMMINI_WS
    ) -> None:
        mappings = self.generate(workload, hardware, backend)
        text = f"# Generated from the normalized workload and hardware snapshot for {backend.name}.\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + yaml.safe_dump(mappings, sort_keys=False))
