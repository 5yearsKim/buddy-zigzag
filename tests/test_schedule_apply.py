import json
from pathlib import Path

import pytest

from buddy_zigzag import ZigzagExporter, ZigzagImporter
from buddy_zigzag.cli import main
from buddy_zigzag.exporter.errors import ExportError
from buddy_zigzag.exporter.extract_mlir import LinalgExtractor
from buddy_zigzag.exporter.identity import LinalgIdentity
from buddy_zigzag.importer import BuddyScheduleApplier, ImportRequest, ZigzagScheduleImporter
from buddy_zigzag.importer.errors import ScheduleError
from buddy_zigzag.settings import Settings

HERE = Path(__file__).parent
RESULT = HERE / "fixtures/zigzag/matmul_small_complete.json"


def test_export_preserves_persistent_operation_ids(tmp_path):
    exported = ZigzagExporter().export_case("matmul_chain", output_dir=tmp_path)
    text = (exported.output_dir / "imported.mlir").read_text()
    assert "buddy_zigzag.layer_id = 0 : i64" in text
    assert "buddy_zigzag.layer_id = 1 : i64" in text
    again = LinalgIdentity().annotate(text)
    assert [layer.id for layer in LinalgExtractor().extract(again).layers] == [0, 1]
    narrower_id = text.replace("buddy_zigzag.layer_id = 0 : i64", "buddy_zigzag.layer_id = 0 : i32")
    assert "buddy_zigzag.layer_id = 0 : i64" in LinalgIdentity().annotate(narrower_id)
    with pytest.raises(ExportError, match="unique"):
        LinalgIdentity().annotate(text.replace("buddy_zigzag.layer_id = 1", "buddy_zigzag.layer_id = 0"))


def test_schedule_is_bound_to_exact_mlir(tmp_path):
    exported = ZigzagExporter().export_case("matmul_small", output_dir=tmp_path)
    text = (tmp_path / "imported.mlir").read_text()
    schedule = ZigzagScheduleImporter().load(RESULT, exported.workload.layers[0], text, serialize_spatial=True)
    with pytest.raises(ScheduleError, match="different MLIR"):
        BuddyScheduleApplier(Settings.load()).apply(text + "\n", schedule)


def test_cli_failure_preserves_original_export_and_writes_report(tmp_path):
    export_dir = tmp_path / "export"
    ZigzagExporter().export_case("matmul_small", output_dir=export_dir)
    original = (export_dir / "imported.mlir").read_text()
    assert main(["import", "--export-dir", str(export_dir), "--result", str(RESULT)]) == 1
    report = json.loads((export_dir / "importer/validation.json").read_text())
    assert report["status"] == "failed"
    assert "spatial serialization" in report["error"]
    assert (export_dir / "imported.mlir").read_text() == original


def test_modified_export_is_rejected_before_compilation(tmp_path):
    ZigzagExporter().export_case("matmul_small", output_dir=tmp_path)
    path = tmp_path / "imported.mlir"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ScheduleError, match="no longer matches its manifest"):
        ZigzagImporter().run(ImportRequest(export_dir=tmp_path, result=RESULT, serialize_spatial=True))


def test_batch_workload_has_an_explicit_unsupported_error(tmp_path):
    ZigzagExporter().export_case("batch_matmul", output_dir=tmp_path)
    with pytest.raises(ScheduleError, match="unbatched"):
        ZigzagImporter().run(ImportRequest(export_dir=tmp_path, result=RESULT))
