import json
from pathlib import Path

import pytest

from buddy_zigzag import DsePipeline, ScheduleCompiler
from buddy_zigzag.cli import main
from buddy_zigzag.codegen import ScheduleCompileRequest
from buddy_zigzag.codegen.cpu.transforms import CpuScheduleApplier
from buddy_zigzag.schedule import CpuScheduleProjector
from buddy_zigzag.schedule.errors import ScheduleError
from buddy_zigzag.settings import Settings
from buddy_zigzag.workload.errors import WorkloadError
from buddy_zigzag.workload.identity import LinalgIdentity
from buddy_zigzag.workload.linalg import LinalgExtractor

HERE = Path(__file__).parent
RESULT = HERE / "fixtures/zigzag/matmul_small_complete.json"


def test_preparation_preserves_persistent_operation_ids(tmp_path):
    prepared = DsePipeline().prepare_case("matmul_chain", output_dir=tmp_path)
    text = (prepared.output_dir / "01_imported.mlir").read_text()
    assert "buddy_zigzag.layer_id = 0 : i64" in text
    assert "buddy_zigzag.layer_id = 1 : i64" in text
    again = LinalgIdentity().annotate(text)
    assert [layer.id for layer in LinalgExtractor().extract(again).layers] == [0, 1]
    narrower_id = text.replace("buddy_zigzag.layer_id = 0 : i64", "buddy_zigzag.layer_id = 0 : i32")
    assert "buddy_zigzag.layer_id = 0 : i64" in LinalgIdentity().annotate(narrower_id)
    with pytest.raises(WorkloadError, match="unique"):
        LinalgIdentity().annotate(text.replace("buddy_zigzag.layer_id = 1", "buddy_zigzag.layer_id = 0"))


def test_schedule_is_bound_to_exact_mlir(tmp_path):
    prepared = DsePipeline().prepare_case("matmul_small", output_dir=tmp_path)
    text = (tmp_path / "01_imported.mlir").read_text()
    schedule = CpuScheduleProjector().load(RESULT, prepared.workload.layers[0], text, serialize_spatial=True)
    with pytest.raises(ScheduleError, match="different MLIR"):
        CpuScheduleApplier(Settings.load()).apply(text + "\n", schedule)


def test_cli_failure_preserves_original_workload_and_writes_report(tmp_path):
    workload_dir = tmp_path / "workload"
    DsePipeline().prepare_case("matmul_small", output_dir=workload_dir)
    original = (workload_dir / "01_imported.mlir").read_text()
    assert main(["compile", "--workload-dir", str(workload_dir), "--result", str(RESULT)]) == 1
    report = json.loads((workload_dir / "lowered/validation.json").read_text())
    assert report["status"] == "failed"
    assert "spatial serialization" in report["error"]
    assert (workload_dir / "01_imported.mlir").read_text() == original


def test_modified_workload_is_rejected_before_compilation(tmp_path):
    DsePipeline().prepare_case("matmul_small", output_dir=tmp_path)
    path = tmp_path / "01_imported.mlir"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ScheduleError, match="no longer matches its manifest"):
        ScheduleCompiler().compile(ScheduleCompileRequest(workload_dir=tmp_path, result=RESULT, serialize_spatial=True))


def test_batch_workload_has_an_explicit_unsupported_error(tmp_path):
    DsePipeline().prepare_case("batch_matmul", output_dir=tmp_path)
    with pytest.raises(ScheduleError, match="unbatched"):
        ScheduleCompiler().compile(ScheduleCompileRequest(workload_dir=tmp_path, result=RESULT))
