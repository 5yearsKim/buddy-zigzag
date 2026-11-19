"""Workload-aware mapping legality and integration with the real ZigZag search."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from buddy_zigzag import (
    GEMMINI_WS,
    DsePipeline,
    GemminiHardwareExporter,
    MappingConstraintGenerator,
    ScheduleCompiler,
    SearchRequest,
)
from buddy_zigzag.cli import main
from buddy_zigzag.codegen import ScheduleCompileRequest
from buddy_zigzag.hardware import GemminiSnapshot
from buddy_zigzag.mapping import MappingError
from buddy_zigzag.workload.linalg import LinalgExtractor
from buddy_zigzag.workload.schema import Workload

HERE = Path(__file__).parent
SNAPSHOT = HERE / "fixtures/hardware/fp32_parameters.json"
MLIR = HERE / "fixtures/mlir/matmul.mlir"


@pytest.fixture
def workload():
    return LinalgExtractor().extract(MLIR.read_text())


@pytest.mark.parametrize("dim", [4, 8])
def test_mapping_uses_layer_name_and_actual_hardware_dimensions(workload, dim, tmp_path):
    values = json.loads(SNAPSHOT.read_text())
    values["array"].update(mesh_rows=dim, mesh_columns=dim, rows=dim, columns=dim)
    for name in ("scratchpad", "accumulator"):
        values[name]["row_bits"] = dim * 32
        values[name]["rows_per_bank"] = values[name]["capacity_bits"] // (values[name]["banks"] * dim * 32)
    hardware = GemminiSnapshot.model_validate(values)
    mapping = MappingConstraintGenerator().generate(workload, hardware, GEMMINI_WS)
    assert mapping[0] == {"name": "default", "memory_operand_links": {"O": "O", "I": "I1", "W": "I2"}}
    assert mapping[1]["name"] == workload.layers[0].name
    assert mapping[1]["spatial_mapping"] == {"D1": [f"K, {dim}"], "D2": [f"N, {dim}"]}
    path = tmp_path / "mapping.yaml"
    MappingConstraintGenerator().write(workload, hardware, path)
    assert yaml.safe_load(path.read_text()) == mapping


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("operation", "does not support"),
        ("dtype", "supports only f32"),
        ("layout", "untransposed"),
        ("constant", "external matrix inputs"),
        ("output", "returned matmul result"),
        ("reserved_name", "unique names"),
    ],
)
def test_unsupported_workload_is_rejected_before_writing(workload, mutation, message, tmp_path):
    data = workload.model_dump(round_trip=True)
    layer = data["layers"][0]
    if mutation == "operation":
        layer["mlir_op"] = "linalg.conv_2d_nhwc_hwcf"
    elif mutation == "dtype":
        for operand in layer["operands"].values():
            operand["tensor"].update(dtype="f16", precision=16)
    elif mutation == "layout":
        layer["operands"]["I"].update(indices=("K", "M"))
        layer["operands"]["I"]["tensor"]["shape"] = (5, 3)
    elif mutation == "constant":
        layer["operands"]["W"]["source_kind"] = "constant"
    elif mutation == "output":
        data["outputs"] = []
    else:
        layer["name"] = "default"
    changed = Workload.model_validate(data)
    path = tmp_path / "mapping.yaml"
    with pytest.raises(MappingError, match=message):
        MappingConstraintGenerator().write(changed, GemminiSnapshot.model_validate_json(SNAPSHOT.read_text()), path)
    assert not path.exists()


@pytest.mark.parametrize(
    "mutation,message",
    [("dataflow", "does not support WS"), ("banks", "four internal"), ("capacity", "double-buffered")],
)
def test_unsupported_hardware_is_rejected(workload, mutation, message):
    values = json.loads(SNAPSHOT.read_text())
    if mutation == "dataflow":
        values["dataflow"] = "OS"
    elif mutation == "banks":
        values["scratchpad"]["banks"] = 2
        values["scratchpad"]["capacity_bits"] //= 2
    else:
        values["accumulator"].update(rows_per_bank=1, capacity_bits=128)
    with pytest.raises(MappingError, match=message):
        MappingConstraintGenerator().generate(workload, GemminiSnapshot.model_validate(values))


@pytest.mark.parametrize("case,message", [("batch_matmul", "does not support"), ("matmul_chain", "at most 1")])
def test_preparation_reports_unsupported_backend_workloads(case, message, tmp_path):
    with pytest.raises(MappingError, match=message):
        DsePipeline().prepare_case(case, snapshot=SNAPSHOT, output_dir=tmp_path)
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["status"] == "failed"
    assert not (tmp_path / "mapping.yaml").exists()


def test_manual_mapping_takes_precedence_over_generation(tmp_path):
    supplied = tmp_path / "supplied.yaml"
    supplied.write_text("- name: default\n  memory_operand_links: {O: O, I: I1, W: I2}\n")
    output = tmp_path / "workload"
    DsePipeline().prepare_case("batch_matmul", snapshot=SNAPSHOT, mapping=supplied, output_dir=output)
    assert (output / "mapping.yaml").read_text() == supplied.read_text()
    assert json.loads((output / "manifest.json").read_text())["mapping_mode"] == "supplied"


def test_generated_mapping_runs_through_zigzag_and_gemmini_lowering(tmp_path):
    hardware_path = tmp_path / "hw_config/hardware.yaml"
    GemminiHardwareExporter().export(SNAPSHOT, hardware_path)
    output = tmp_path / "workload"
    assert main(["prepare-workload", "--case", "matmul_small", "--snapshot", str(SNAPSHOT), "--out", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["mapping_mode"] == "generated"
    assert manifest["mapping_backend"] == "gemmini_ws"
    assert manifest["snapshot_sha256"] == hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    assert manifest["mapping_sha256"] == hashlib.sha256((output / "mapping.yaml").read_bytes()).hexdigest()
    search_dir = output / "zigzag"
    search_dir.mkdir()
    stale = search_dir / "old_layer_complete.json"
    stale.write_text("{}")
    searched = DsePipeline().search(
        SearchRequest(
            workload=output / "workload.yaml",
            hardware=hardware_path,
            mapping=output / "mapping.yaml",
        )
    )
    assert searched.mappings == (search_dir / "matmul_0_complete.json",)
    assert stale.read_text() == "{}"
    report = json.loads((search_dir / "search.json").read_text())
    assert report["status"] == "passed"
    assert report["result"]["energy"] == searched.energy
    lowered = ScheduleCompiler().compile(
        ScheduleCompileRequest(workload_dir=output, result=searched.mappings[0], target="gemmini", snapshot=SNAPSHOT)
    )
    assert "llvm.riscv.loop.ws" in (lowered.output_dir / "06_lowered.ll").read_text()
