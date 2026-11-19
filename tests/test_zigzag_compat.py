import json
from pathlib import Path

import pytest
import yaml

from buddy_zigzag.cli import main

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "case, macs, edges",
    [
        ("matmul_rect", [262144], 0),
        ("matmul_small", [105], 0),
        ("matmul_256x512x1024", [134217728], 0),
        ("matmul_unit", [1], 0),
        ("batch_matmul", [210], 0),
        ("matmul_chain", [105, 84], 1),
    ],
)
def test_torch_buddy_export_and_real_zigzag_parser(tmp_path, case, macs, edges):
    pytest.importorskip("torch")
    pytest.importorskip("buddy.compiler.frontend")
    pytest.importorskip("zigzag.stages.parser.workload_parser")
    assert main(["prepare-workload", "--case", case, "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["status"] == "passed"
    assert report["semantics"]["macs_per_layer"] == macs
    assert report["reference"]["status"] == "passed"
    assert report["zigzag"] == {"status": "passed", "macs_per_layer": macs, "edges": edges}
    assert "linalg." in (tmp_path / "01_imported.mlir").read_text()
    assert json.loads((tmp_path / "manifest.json").read_text())["versions"]["zigzag-dse"] == "3.10.0"
    assert json.loads((tmp_path / "normalized.json").read_text())["schema_version"] == 1
    if case == "matmul_chain":
        data = yaml.safe_load((tmp_path / "workload.yaml").read_text())
        assert data[0]["operand_source"] == {}
        assert data[1]["operand_source"] == {"I": 0}
