from pathlib import Path

import pytest

from buddy_zigzag import ZigzagExporter
from buddy_zigzag.exporter.schema import Workload


def test_public_exporter_api_uses_packaged_mapping(tmp_path):
    pytest.importorskip("buddy_mlir.ir")
    fixture = Path(__file__).parent / "fixtures/mlir/matmul.mlir"
    result = ZigzagExporter().export_mlir(fixture, output_dir=tmp_path)
    assert result.output_dir == tmp_path
    assert result.report["status"] == "passed"
    assert result.workload.layers[0].mac_count == 105
    assert (tmp_path / "mapping.yaml").is_file()
    restored = Workload.model_validate(result.workload.model_dump(round_trip=True))
    assert restored.layers[0].equation == "O[m][n]+=I[m][k]*W[k][n]"
