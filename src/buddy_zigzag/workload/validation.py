"""Check semantic invariants and exercise ZigZag's real workload parser."""

from .errors import WorkloadError
from .schema import Workload


class WorkloadValidator:
    """Check expectations, numerical semantics, and ZigZag compatibility."""

    def validate_semantics(self, workload: Workload, expected_macs: list[int] | None = None) -> dict:
        Workload.model_validate(workload.model_dump(round_trip=True))
        macs = [layer.mac_count for layer in workload.layers]
        if expected_macs is not None and macs != expected_macs:
            raise WorkloadError(f"MAC count mismatch: expected {expected_macs}, extracted {macs}")
        return {"status": "passed", "layer_count": len(macs), "macs_per_layer": macs, "total_macs": sum(macs)}

    def validate_zigzag(self, path, mapping_path, workload: Workload) -> dict:
        try:
            from zigzag.parser.mapping_validator import MappingValidator
            from zigzag.parser.workload_factory import WorkloadFactory
            from zigzag.parser.workload_validator import WorkloadValidator
            from zigzag.utils import open_yaml
        except ImportError as exc:
            raise WorkloadError(
                "ZigZag dependencies are unavailable; install the sibling zigzag checkout or use --skip-zigzag"
            ) from exc
        try:
            # These are the validators and factory used by WorkloadParserStage.
            # Call them directly because a stage also requires an evaluation pipeline.
            layers = WorkloadValidator(open_yaml(str(path)))
            mapping = MappingValidator(open_yaml(str(mapping_path)))
            if not layers.validate() or not mapping.validate():
                raise WorkloadError("ZigZag rejected the workload or mapping schema")
            parsed = WorkloadFactory(layers.normalized_data, mapping.normalized_data).create()
        except (ValueError, AssertionError, KeyError) as exc:
            raise WorkloadError(f"ZigZag workload parsing failed: {exc}") from exc
        actual = [int(node.total_mac_count) for node in parsed.layer_node_list]
        expected = [layer.mac_count for layer in workload.layers]
        if actual != expected:
            raise WorkloadError(f"ZigZag MAC counts differ: expected {expected}, got {actual}")
        return {"status": "passed", "macs_per_layer": actual, "edges": int(parsed.number_of_edges())}

    def validate_reference(self, workload: Workload, inputs: tuple, expected: tuple) -> dict:
        """Interpret extracted accesses and compare with Torch, without native JIT."""
        import torch

        values = {f"arg{i}": value for i, value in enumerate(inputs)}
        with torch.no_grad():
            for layer in workload.layers:
                try:
                    lhs = values[layer.operands["I"].source_name]
                    rhs = values[layer.operands["W"].source_name]
                except KeyError as exc:
                    raise WorkloadError("Reference checking requires inputs with known tensor values") from exc
                accesses = ["".join(dim.lower() for dim in layer.operands[name].indices) for name in ("I", "W", "O")]
                values[layer.operands["O"].source_name] = torch.einsum(
                    f"{accesses[0]},{accesses[1]}->{accesses[2]}", lhs, rhs
                )
            if len(workload.outputs) != len(expected):
                raise WorkloadError("Extracted output count differs from Torch")
            for name, reference in zip(workload.outputs, expected):
                try:
                    torch.testing.assert_close(values[name], reference)
                except AssertionError as exc:
                    raise WorkloadError(f"Extracted computation differs from Torch: {exc}") from exc
        return {"status": "passed", "method": "interpret extracted accesses using torch.einsum; compare to Torch model"}
