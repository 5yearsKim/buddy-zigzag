"""Exercise real transformations, lowering, and native CPU execution."""

import json
from pathlib import Path

import pytest

from buddy_zigzag import ZigzagExporter, ZigzagImporter
from buddy_zigzag.cli import main
from buddy_zigzag.importer import ImportRequest

HERE = Path(__file__).parent
RESULT = HERE / "fixtures/zigzag/matmul_small_complete.json"


def test_real_zigzag_mapping_lowers_and_executes_with_partial_tiles(tmp_path):
    export_dir = tmp_path / "export"
    ZigzagExporter().export_case("matmul_small", output_dir=export_dir)
    result = ZigzagImporter().run(
        ImportRequest(
            export_dir=export_dir,
            result=RESULT,
            serialize_spatial=True,
            verify=True,
        )
    )
    assert result.verification["status"] == "passed"
    assert result.verification["trials"] == 3
    text = (result.output_dir / "scheduled.mlir").read_text()
    assert text.count("scf.for") == 5
    assert text.count("buddy_zigzag.loop_dimension") == 5
    assert "affine.min" in text  # K=5 and N=7 require clipped four-element tiles.
    assert "transform.named_sequence" not in text
    assert "transform.structured.tile_using_for" in (result.output_dir / "transform.mlir").read_text()
    assert "llvm.func" in (result.output_dir / "lowered.mlir").read_text()
    assert "define void @_mlir_ciface_forward" in (result.output_dir / "lowered.ll").read_text()
    assert "memref.dealloc" in (result.output_dir / "bufferized.mlir").read_text()


@pytest.mark.parametrize(
    "case, shape, temporal",
    [
        ("matmul_unit", {"M": 1, "N": 1, "K": 1}, []),
        (
            "matmul_rect",
            {"M": 64, "N": 128, "K": 32},
            [
                "(K, 4)",
                "(M, 8)",
                "(N, 16)",
                "(K, 8)",
                "(M, 8)",
                "(N, 8)",
            ],
        ),
    ],
)
def test_temporal_only_repeated_factors_and_unit_loops(tmp_path, case, shape, temporal):
    export_dir = tmp_path / "export"
    ZigzagExporter().export_case(case, output_dir=export_dir)
    # A synthetic, fully temporal candidate tests factor nesting independently
    # of the search engine's choices. No hardware cost claims use this fixture.
    data = json.loads(RESULT.read_text())
    data["inputs"]["layer"]["loop_dimensions"] = shape
    data["inputs"]["layer"]["user_spatial_mapping"] = {}
    data["inputs"]["temporal_mapping"] = {name: [temporal] for name in ("I", "W", "O")}
    path = tmp_path / "temporal.json"
    path.write_text(json.dumps(data))
    result = ZigzagImporter().run(ImportRequest(export_dir=export_dir, result=path, verify=True))
    assert result.verification["status"] == "passed"
    assert result.schedule.spatial_policy == "reject"
    assert result.verification["max_abs_error"] < 1e-3


def test_import_cli_produces_machine_code_and_verifies(tmp_path):
    ZigzagExporter().export_case("matmul_small", output_dir=tmp_path / "export")
    assert (
        main(
            [
                "import",
                "--export-dir",
                str(tmp_path / "export"),
                "--result",
                str(RESULT),
                "--serialize-spatial",
                "--verify",
                "--trials",
                "1",
                "--out",
                str(tmp_path / "code"),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "code/validation.json").read_text())
    assert report["native_verification"]["status"] == "passed"
