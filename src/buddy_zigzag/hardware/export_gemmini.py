"""Convert a resolved Gemmini snapshot and explicit assumptions to ZigZag YAML."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import yaml

from .schema import GemminiModelTemplate, GemminiSnapshot, HardwareExportResult, MemoryCost


class HardwareExportError(ValueError):
    """A hardware description is unsupported or inconsistent."""


class ZigzagHardwareValidator:
    def validate(self, data: dict) -> dict:
        from zigzag.parser.accelerator_factory import AcceleratorFactory
        from zigzag.parser.accelerator_validator import AcceleratorValidator

        validator = AcceleratorValidator(data)
        if not validator.validate():
            raise HardwareExportError(f"ZigZag rejected generated hardware: {validator.validator.errors}")
        accelerator = AcceleratorFactory(validator.normalized_data).create()
        return {"status": "passed", "accelerator": accelerator.name, "memory_levels": len(data["memories"])}


class GemminiHardwareExporter:
    def __init__(self):
        self.validator = ZigzagHardwareValidator()

    def convert(self, snapshot: GemminiSnapshot, model: GemminiModelTemplate) -> dict:
        for name, dtype in snapshot.arithmetic:
            if (dtype.kind, dtype.bits, dtype.exponent_bits, dtype.significand_bits, dtype.recoded) != (
                "float",
                32,
                8,
                24,
                False,
            ):
                raise HardwareExportError(
                    f"The initial converter supports FP32 only; {name} is {dtype.kind}/{dtype.bits}"
                )
        if snapshot.array.tile_rows != 1 or snapshot.array.tile_columns != 1:
            raise HardwareExportError("Combinational multi-PE tiles require a different modeling template")
        if snapshot.array.rows != snapshot.array.columns:
            raise HardwareExportError("The initial converter supports square Gemmini arrays")
        if snapshot.external_memory.shared or snapshot.external_memory.tilelink:
            raise HardwareExportError("External/shared scratchpad configurations require a different topology model")
        if snapshot.dataflow not in (model.dataflow, "BOTH"):
            raise HardwareExportError(f"Requested {model.dataflow} dataflow is unsupported by this configuration")
        bandwidth = model.external_memory.bandwidth_bits_per_cycle or snapshot.dma.buswidth_bits
        if bandwidth > snapshot.dma.buswidth_bits:
            raise HardwareExportError("External memory bandwidth cannot exceed the modeled DMA interface width")

        all_dims = ["D1", "D2"]
        memories = {
            "weight_register": self._memory(
                snapshot.arithmetic.weight.bits,
                model.weight_register.latency_cycles,
                model.weight_register,
                ["I2"],
                [],
                self._split_ports(["I2"], snapshot.arithmetic.weight.bits),
            ),
            "accumulator": self._memory(
                snapshot.accumulator.capacity_bits,
                snapshot.accumulator.read_latency,
                model.accumulator,
                ["O"],
                all_dims,
                self._shared_port(["O"], snapshot.accumulator.row_bits)
                if snapshot.accumulator.single_ported
                else self._split_ports(["O"], snapshot.accumulator.row_bits),
            ),
            "scratchpad": self._memory(
                snapshot.scratchpad.capacity_bits,
                snapshot.scratchpad.read_latency,
                model.scratchpad,
                ["I1", "I2"],
                all_dims,
                self._shared_port(["I1", "I2"], snapshot.scratchpad.row_bits),
            ),
            "external_memory": self._memory(
                model.external_memory.capacity_bits,
                model.external_memory.latency_cycles,
                model.external_memory,
                ["I1", "I2", "O"],
                all_dims,
                self._shared_port(["I1", "I2", "O"], bandwidth),
            ),
        }
        return {
            "name": model.name,
            "operational_array": {
                "dimensions": all_dims,
                "sizes": [snapshot.array.rows, snapshot.array.columns],
                "input_precision": [snapshot.arithmetic.input.bits, snapshot.arithmetic.weight.bits],
                "unit_energy": model.compute.energy_per_mac_pj,
                "unit_area": model.compute.area_per_mac,
                "systolic_dimensions": model.compute.systolic_dimensions,
            },
            "memories": memories,
            "dataflows": {"D1": [f"K, {snapshot.array.rows}"], "D2": [f"N, {snapshot.array.columns}"]},
        }

    def export(
        self, snapshot_path: Path, output_path: Path, *, template_path: Path | None = None
    ) -> HardwareExportResult:
        snapshot_text = snapshot_path.read_text()
        snapshot = GemminiSnapshot.model_validate_json(snapshot_text)
        resource = template_path or files("buddy_zigzag.hardware").joinpath("templates/gemmini_fp32.yaml")
        template_text = resource.read_text()
        model = GemminiModelTemplate.model_validate(yaml.safe_load(template_text))
        data = self.convert(snapshot, model)
        validation = self.validator.validate(data)

        def sha(value: str) -> str:
            return hashlib.sha256(value.encode()).hexdigest()

        manifest = {
            "schema_version": 1,
            "source": snapshot.source.model_dump(mode="json"),
            "snapshot_sha256": sha(snapshot_text),
            "template_sha256": sha(template_text),
            "resolved_parameters": snapshot.model_dump(mode="json"),
            "model": model.model_dump(mode="json"),
            "validation": validation,
            "limitations": {
                "calibrated_costs": model.cost_status == "characterized",
                "bank_conflicts_modeled": False,
                "gemmini_pipeline_timing_modeled": False,
                "soc_memory_system_modeled": False,
            },
        }
        output_path = output_path.resolve()
        manifest_path = output_path.with_suffix(".manifest.json")
        if output_path.suffix != ".yaml":
            raise HardwareExportError("The ZigZag hardware output must have a .yaml extension")
        sources = {snapshot_path.resolve()}
        if template_path:
            sources.add(template_path.resolve())
        if any(path in sources for path in (output_path, manifest_path)):
            raise HardwareExportError("Choose a separate output path to preserve source configuration files")
        comments = [
            "# Generated by GemminiHardwareExporter; analytical approximation, not an RTL-equivalent model.",
            f"# Configuration: {snapshot.source.config_name}; snapshot origin: {snapshot.source.origin}",
            f"# Snapshot SHA-256: {sha(snapshot_text)}",
            f"# Template SHA-256: {sha(template_text)}",
            f"# Energy/area cost status: {model.cost_status.upper()}",
            "# See the adjacent .manifest.json for resolved parameters, assumptions, and provenance.",
        ]
        comments += ["# Assumption: " + line for line in model.assumptions]
        comment_block = "\n".join(
            "# " + line.removeprefix("# ") for comment in comments for line in comment.splitlines()
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(comment_block + "\n" + yaml.safe_dump(data, sort_keys=False))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return HardwareExportResult(
            hardware_path=str(output_path),
            manifest_path=str(manifest_path),
            validation=validation,
        )

    def _memory(
        self, size: int, latency: int, costs: MemoryCost, operands: list[str], dimensions: list[str], ports: list
    ) -> dict:
        return {
            "size": size,
            "latency": latency,
            "r_cost": costs.read_energy_pj,
            "w_cost": costs.write_energy_pj,
            "area": costs.area,
            "auto_cost_extraction": False,
            "operands": operands,
            "served_dimensions": list(dimensions),
            "ports": ports,
        }

    def _directions(self, operand: str) -> tuple[str, ...]:
        return ("fh", "tl", "fl", "th") if operand == "O" else ("fh", "tl")

    def _shared_port(self, operands: list[str], bandwidth: int) -> list[dict]:
        return [
            {
                "name": "rw_port_1",
                "type": "read_write",
                "bandwidth_min": bandwidth,
                "bandwidth_max": bandwidth,
                "allocation": [
                    f"{operand}, {direction}" for operand in operands for direction in self._directions(operand)
                ],
            }
        ]

    def _split_ports(self, operands: list[str], bandwidth: int) -> list[dict]:
        return [
            {
                "name": f"{letter}_port_1",
                "type": kind,
                "bandwidth_min": bandwidth,
                "bandwidth_max": bandwidth,
                "allocation": [
                    f"{operand}, {direction}"
                    for operand in operands
                    for direction in self._directions(operand)
                    if direction in allowed
                ],
            }
            for letter, kind, allowed in (("r", "read", {"tl", "th"}), ("w", "write", {"fh", "fl"}))
        ]
