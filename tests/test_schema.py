import pytest
from pydantic import ValidationError

from buddy_zigzag.workload.schema import Operand, TensorSpec, WorkloadBuildRequest, WorkloadCase


@pytest.mark.parametrize("shape", [(0, 5), (-1, 5), (True, 5), ("3", 5), ()])
def test_tensor_rejects_invalid_or_coerced_dimensions(shape):
    with pytest.raises(ValidationError):
        TensorSpec(shape=shape, dtype="f32", precision=32)


def test_tensor_rejects_inconsistent_precision():
    with pytest.raises(ValidationError, match="precision must agree"):
        TensorSpec(shape=(3, 5), dtype="f32", precision=16)


def test_operand_requires_a_producer_id_only_for_layers():
    tensor = TensorSpec(shape=(3, 5), dtype="f32", precision=32)
    with pytest.raises(ValidationError, match="source_id"):
        Operand(tensor=tensor, indices=("M", "K"), source_kind="layer", source_name="missing")
    with pytest.raises(ValidationError, match="source_id"):
        Operand(tensor=tensor, indices=("M", "K"), source_kind="input", source_name="arg0", source_id=0)


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "matmul", "shapes": [[3, 5], [6, 7]]},
        {"kind": "batch_matmul", "shapes": [[2, 3, 5], [4, 5, 7]]},
        {"kind": "matmul_chain", "shapes": [[3, 5], [5, 7], [8, 4]]},
        {"kind": "matmul", "shapes": [[3, 5], [5, 7]], "dtype": "typo"},
        {"kind": "matmul", "shapes": [[3, 5], [5, 7]], "expected_macs": [105, 1]},
        {"kind": "matmul", "shapes": [[3, 5], [5, 7]], "unknown_field": 1},
    ],
)
def test_case_validation_rejects_malformed_workloads(data):
    with pytest.raises(ValidationError):
        WorkloadCase.model_validate(data)


def test_export_request_requires_exactly_one_source():
    with pytest.raises(ValidationError, match="exactly one"):
        WorkloadBuildRequest()
    with pytest.raises(ValidationError, match="exactly one"):
        WorkloadBuildRequest(case="matmul_rect", mlir="model.mlir")
