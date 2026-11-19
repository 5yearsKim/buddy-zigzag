from pathlib import Path

import pytest

from buddy_zigzag import DsePipeline, WorkloadBuilder
from buddy_zigzag.workload.schema import Workload


def test_preparation_api_uses_packaged_mapping(tmp_path):
    pytest.importorskip("buddy_mlir.ir")
    fixture = Path(__file__).parent / "fixtures/mlir/matmul.mlir"
    result = DsePipeline().prepare_mlir(fixture, output_dir=tmp_path)
    assert result.output_dir == tmp_path
    assert result.report["status"] == "passed"
    assert result.workload.layers[0].mac_count == 105
    assert (tmp_path / "mapping.yaml").is_file()
    restored = Workload.model_validate(result.workload.model_dump(round_trip=True))
    assert restored.layers[0].equation == "O[m][n]+=I[m][k]*W[k][n]"


def test_workload_builder_is_independent_of_mapping_preparation(tmp_path):
    result = WorkloadBuilder().build_case("matmul_small", output_dir=tmp_path)
    assert result.report["status"] == "passed"
    assert result.report["reference"]["status"] == "passed"
    assert (tmp_path / "workload.yaml").is_file()
    assert (tmp_path / "01_imported.mlir").is_file()
    assert "zigzag" not in result.report
    assert not (tmp_path / "mapping.yaml").exists()
