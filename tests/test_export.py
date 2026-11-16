import json
from pathlib import Path

import pytest
import yaml

from buddy_zigzag.cli import main
from buddy_zigzag.exporter.emit_workload import ZigzagWorkloadWriter
from buddy_zigzag.exporter.errors import ExportError
from buddy_zigzag.exporter.extract_mlir import LinalgExtractor
from buddy_zigzag.exporter.normalize import matmul_dimensions
from buddy_zigzag.exporter.schema import TensorSpec
from buddy_zigzag.exporter.validate import ExportValidator

HERE = Path(__file__).parent
FIXTURE = HERE / "fixtures/mlir/matmul.mlir"


def test_rectangular_matmul_matches_reviewed_yaml():
    pytest.importorskip("buddy_mlir.ir")
    workload = LinalgExtractor().extract(FIXTURE.read_text())
    assert ZigzagWorkloadWriter().to_data(workload) == yaml.safe_load((HERE / "expected/matmul.yaml").read_text())
    layer = workload.layers[0]
    assert layer.mac_count == 105
    assert layer.parallel_dims == ("M", "N")
    assert layer.reduction_dims == ("K",)
    assert layer.operands["I"].source_kind == "input"
    assert layer.operands["W"].source_kind == "input"
    assert workload.outputs == ["matmul_0:O"]
    assert workload.auxiliary_operations[0].operation == "arith.constant"


def test_transpose_b_uses_actual_access_map():
    pytest.importorskip("buddy_mlir.ir")
    text = (
        FIXTURE.read_text()
        .replace("tensor<5x7xf32>", "tensor<7x5xf32>")
        .replace(
            "linalg.matmul ins",
            "linalg.matmul indexing_maps = ["
            "affine_map<(d0, d1, d2) -> (d0, d2)>, "
            "affine_map<(d0, d1, d2) -> (d1, d2)>, "
            "affine_map<(d0, d1, d2) -> (d0, d1)>] ins",
        )
    )
    workload = LinalgExtractor().extract(text)
    assert workload.layers[0].equation == "O[m][n]+=I[m][k]*W[n][k]"
    assert workload.layers[0].loop_sizes == (3, 7, 5)
    torch = pytest.importorskip("torch")
    a, b = torch.randn(3, 5), torch.randn(7, 5)
    assert ExportValidator().validate_reference(workload, (a, b), (a @ b.T,))["status"] == "passed"


def test_mixed_integer_accumulator_precision():
    pytest.importorskip("buddy_mlir.ir")
    text = FIXTURE.read_text().replace("3x5xf32", "3x5xi8").replace("5x7xf32", "5x7xi8")
    text = text.replace("3x7xf32", "3x7xi32").replace("dense<0.0>", "dense<0>")
    data = ZigzagWorkloadWriter().to_data(LinalgExtractor().extract(text))[0]
    assert data["operand_precision"] == {"I": 8, "W": 8, "O": 32, "O_final": 32}


def test_constant_weight_is_distinct_from_external_input():
    pytest.importorskip("buddy_mlir.ir")
    text = (
        FIXTURE.read_text()
        .replace("%a: tensor<3x5xf32>, %b: tensor<5x7xf32>", "%a: tensor<3x5xf32>")
        .replace("    %zero", "    %b = arith.constant dense<1.0> : tensor<5x7xf32>\n    %zero")
    )
    data = ZigzagWorkloadWriter().to_data(LinalgExtractor().extract(text))[0]
    assert data["operand_source"] == {"W": 0}


def test_zero_fill_is_recorded_and_supported():
    pytest.importorskip("buddy_mlir.ir")
    text = FIXTURE.read_text().replace(
        "%zero = arith.constant dense<0.0> : tensor<3x7xf32>",
        "%c = arith.constant 0.0 : f32\n"
        "    %empty = tensor.empty() : tensor<3x7xf32>\n"
        "    %zero = linalg.fill ins(%c : f32) outs(%empty : tensor<3x7xf32>) -> tensor<3x7xf32>",
    )
    workload = LinalgExtractor().extract(text)
    assert workload.layers[0].mac_count == 105
    assert [entry.operation for entry in workload.auxiliary_operations] == [
        "arith.constant",
        "tensor.empty",
        "linalg.fill",
    ]


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("dense<1.0>", "initialized with constant zero"),
        (None, "Unsupported operation"),
    ],
)
def test_unsupported_computation_is_rejected(replacement, message):
    pytest.importorskip("buddy_mlir.ir")
    text = FIXTURE.read_text()
    if replacement:
        text = text.replace("dense<0.0>", replacement)
    else:
        text = text.replace("    return %result", "    %neg = arith.negf %result : tensor<3x7xf32>\n    return %neg")
    with pytest.raises(ExportError, match=message):
        LinalgExtractor().extract(text)


def test_dynamic_shapes_are_rejected():
    pytest.importorskip("buddy_mlir.ir")
    text = """module {
      func.func @dynamic(%a: tensor<?x5xf32>, %b: tensor<5x7xf32>, %out: tensor<?x7xf32>) -> tensor<?x7xf32> {
        %r = linalg.matmul ins(%a, %b : tensor<?x5xf32>, tensor<5x7xf32>)
          outs(%out : tensor<?x7xf32>) -> tensor<?x7xf32>
        return %r : tensor<?x7xf32>
      }
    }"""
    with pytest.raises(ExportError, match="positive static"):
        LinalgExtractor().extract(text)


def test_mismatched_reduction_extents_are_rejected():
    specs = [
        TensorSpec(shape=(3, 5), dtype="f32", precision=32),
        TensorSpec(shape=(6, 7), dtype="f32", precision=32),
        TensorSpec(shape=(3, 7), dtype="f32", precision=32),
    ]
    with pytest.raises(ExportError, match="Inconsistent extent for K"):
        matmul_dimensions(specs, [(0, 2), (2, 1), (0, 1)], 3, False)


def test_independent_mac_expectation_catches_wrong_workload():
    pytest.importorskip("buddy_mlir.ir")
    with pytest.raises(ExportError, match="MAC count mismatch"):
        ExportValidator().validate_semantics(LinalgExtractor().extract(FIXTURE.read_text()), [106])


def test_failed_cli_writes_failure_report(tmp_path):
    pytest.importorskip("buddy_mlir.ir")
    invalid = tmp_path / "nonzero.mlir"
    invalid.write_text(FIXTURE.read_text().replace("dense<0.0>", "dense<2.0>"))
    out = tmp_path / "out"
    assert main(["export", "--mlir", str(invalid), "--out", str(out), "--skip-zigzag"]) == 1
    report = json.loads((out / "validation.json").read_text())
    assert report["status"] == "failed"
    assert "constant zero" in report["error"]


def test_skip_zigzag_is_explicit_in_report(tmp_path):
    pytest.importorskip("buddy_mlir.ir")
    assert main(["export", "--mlir", str(FIXTURE), "--out", str(tmp_path), "--skip-zigzag"]) == 0
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["semantics"]["status"] == "passed"
    assert report["zigzag"]["status"] == "skipped"
