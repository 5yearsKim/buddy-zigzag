"""The snapshot fixture was emitted by the actual resolved Scala FP32 config."""

import json
import math
from importlib.resources import files
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from buddy_zigzag import GemminiHardwareExporter, ZigzagExporter
from buddy_zigzag.cli import main
from buddy_zigzag.hardware import GemminiModelTemplate, GemminiSnapshot, HardwareExportError

HERE = Path(__file__).parent
SNAPSHOT = HERE / "fixtures/hardware/fp32_parameters.json"


@pytest.fixture
def model():
    resource = files("buddy_zigzag.hardware").joinpath("templates/gemmini_fp32.yaml")
    return GemminiModelTemplate.model_validate(yaml.safe_load(resource.read_text()))


def test_scala_snapshot_translates_structural_parameters_and_records_assumptions(tmp_path):
    output = tmp_path / "hardware.yaml"
    result = GemminiHardwareExporter().export(SNAPSHOT, output)
    data = yaml.safe_load(output.read_text())
    assert result.validation == {"status": "passed", "accelerator": "gemmini_fp32_ws_approx", "memory_levels": 4}
    assert data["operational_array"]["sizes"] == [4, 4]
    assert data["operational_array"]["input_precision"] == [32, 32]
    assert data["dataflows"] == {"D1": ["K, 4"], "D2": ["N, 4"]}
    assert data["memories"]["scratchpad"]["size"] == 256 * 1024 * 8
    assert data["memories"]["accumulator"]["size"] == 64 * 1024 * 8
    assert data["memories"]["scratchpad"]["latency"] == 1
    assert data["memories"]["accumulator"]["latency"] == 2
    assert data["memories"]["weight_register"]["size"] == 32
    assert data["memories"]["weight_register"]["served_dimensions"] == []
    # Bandwidth is one modeled shared row, not an unsubstantiated banks multiplier.
    assert data["memories"]["scratchpad"]["ports"][0]["bandwidth_max"] == 128
    assert [port["type"] for port in data["memories"]["accumulator"]["ports"]] == ["read", "write"]
    assert data["memories"]["external_memory"]["ports"][0]["bandwidth_max"] == 128
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["source"]["origin"] == "resolved_config"
    assert manifest["source"]["gemmini_revision"]
    assert len(manifest["source"]["gemmini_scala_sha256"]) == 64
    assert manifest["resolved_parameters"]["array"]["tile_latency"] == 2
    assert manifest["model"]["cost_status"] == "placeholder"
    assert manifest["limitations"]["calibrated_costs"] is False
    assert "PLACEHOLDER" in output.read_text()
    assert "&id" not in output.read_text()


def test_changed_resolved_array_parameters_drive_the_generated_profile(model):
    data = json.loads(SNAPSHOT.read_text())
    data["array"].update(mesh_rows=8, mesh_columns=8, rows=8, columns=8)
    for name in ("scratchpad", "accumulator"):
        data[name]["row_bits"] *= 2
        data[name]["rows_per_bank"] //= 2
    snapshot = GemminiSnapshot.model_validate(data)
    hardware = GemminiHardwareExporter().convert(snapshot, model)
    assert hardware["operational_array"]["sizes"] == [8, 8]
    assert hardware["dataflows"] == {"D1": ["K, 8"], "D2": ["N, 8"]}
    assert hardware["memories"]["scratchpad"]["ports"][0]["bandwidth_max"] == 256
    assert hardware["memories"]["scratchpad"]["size"] == 256 * 1024 * 8


@pytest.mark.parametrize(
    "mutation", ["capacity", "dimensions", "precision", "boolean_bank_count", "negative_dma", "unknown"]
)
def test_inconsistent_or_malformed_snapshots_are_rejected(mutation):
    data = json.loads(SNAPSHOT.read_text())
    if mutation == "capacity":
        data["scratchpad"]["capacity_bits"] += 1
    elif mutation == "dimensions":
        data["array"]["rows"] += 1
    elif mutation == "precision":
        data["arithmetic"]["input"]["bits"] = 16
    elif mutation == "boolean_bank_count":
        data["scratchpad"]["banks"] = True
    elif mutation == "negative_dma":
        data["dma"]["buswidth_bits"] = -128
    else:
        data["scratchpad"]["capacity_bytes"] = 256 * 1024
    with pytest.raises(ValidationError):
        GemminiSnapshot.model_validate(data)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("integer", "FP32 only"),
        ("external", "External/shared"),
        ("os_only", "WS dataflow"),
        ("multi_pe_tile", "multi-PE tiles"),
    ],
)
def test_unsupported_hardware_families_are_explicit(model, mutation, message):
    data = json.loads(SNAPSHOT.read_text())
    if mutation == "integer":
        data["arithmetic"]["weight"] = {"kind": "sint", "bits": 32}
    elif mutation == "external":
        data["external_memory"]["shared"] = True
    elif mutation == "os_only":
        data["dataflow"] = "OS"
    else:
        data["array"].update(mesh_rows=2, mesh_columns=2, tile_rows=2, tile_columns=2)
    snapshot = GemminiSnapshot.model_validate(data)
    with pytest.raises(HardwareExportError, match=message):
        GemminiHardwareExporter().convert(snapshot, model)


def test_single_ported_accumulator_and_bandwidth_constraint(model):
    data = json.loads(SNAPSHOT.read_text())
    data["accumulator"].update(single_ported=True, sub_banks=1)
    snapshot = GemminiSnapshot.model_validate(data)
    profile = GemminiHardwareExporter().convert(snapshot, model)
    assert [port["type"] for port in profile["memories"]["accumulator"]["ports"]] == ["read_write"]
    updated = model.model_dump()
    updated["external_memory"]["bandwidth_bits_per_cycle"] = 256
    with pytest.raises(HardwareExportError, match="cannot exceed"):
        GemminiHardwareExporter().convert(snapshot, GemminiModelTemplate.model_validate(updated))


def test_template_cannot_silently_override_structural_parameters(model):
    data = model.model_dump()
    data["scratchpad"]["capacity_bits"] = 1
    with pytest.raises(ValidationError):
        GemminiModelTemplate.model_validate(data)


def test_invalid_snapshot_preserves_existing_profile(tmp_path):
    snapshot = tmp_path / "invalid.json"
    snapshot.write_text('{"schema_version": 99}')
    output = tmp_path / "hardware.yaml"
    output.write_text("existing profile")
    with pytest.raises(ValidationError):
        GemminiHardwareExporter().export(snapshot, output)
    assert output.read_text() == "existing profile"
    assert not output.with_suffix(".manifest.json").exists()


def test_source_cannot_be_overwritten_by_manifest(tmp_path):
    snapshot = tmp_path / "hardware.manifest.json"
    snapshot.write_text(SNAPSHOT.read_text())
    with pytest.raises(HardwareExportError, match="preserve source"):
        GemminiHardwareExporter().export(snapshot, tmp_path / "hardware.yaml")
    assert snapshot.read_text() == SNAPSHOT.read_text()


def test_hardware_export_cli(tmp_path):
    output = tmp_path / "generated.yaml"
    assert main(["hardware-export", "--snapshot", str(SNAPSHOT), "--out", str(output)]) == 0
    assert output.exists()
    assert output.with_suffix(".manifest.json").exists()


def test_hardware_export_cli_defaults_to_project_hw_config(tmp_path):
    assert main(["--env-file", str(tmp_path / ".env"), "hardware-export", "--snapshot", str(SNAPSHOT)]) == 0
    output = tmp_path / "hw_config/hardware.yaml"
    assert yaml.safe_load(output.read_text())["operational_array"]["sizes"] == [4, 4]
    assert output.with_suffix(".manifest.json").exists()
    assert not (tmp_path / "artifacts").exists()


def test_generated_profile_runs_a_real_zigzag_matmul(tmp_path):
    from zigzag.api import get_hardware_performance_zigzag

    output = tmp_path / "hardware.yaml"
    GemminiHardwareExporter().export(SNAPSHOT, output)
    workload = ZigzagExporter().export_case("matmul_small", output_dir=tmp_path / "workload")
    energy, cycles, results = get_hardware_performance_zigzag(
        workload=str(workload.output_dir / "workload.yaml"),
        accelerator=str(output),
        mapping=str(workload.output_dir / "mapping.yaml"),
        dump_folder=str(tmp_path / "evaluation"),
        nb_spatial_mappings_generated=1,
        lpf_limit=4,
        loma_show_progress_bar=False,
    )
    assert math.isfinite(energy) and energy > 0
    assert math.isfinite(cycles) and cycles > 0
    assert results
