"""Mapping contract tests against a real ZigZag 3.10 complete-result fixture."""

import json
from pathlib import Path

import pytest

from buddy_zigzag.exporter.extract_mlir import LinalgExtractor
from buddy_zigzag.exporter.identity import LinalgIdentity
from buddy_zigzag.importer.errors import ScheduleError
from buddy_zigzag.importer.import_zigzag import ZigzagScheduleImporter

HERE = Path(__file__).parent
RESULT = HERE / "fixtures/zigzag/matmul_small_complete.json"


@pytest.fixture
def payload():
    text = LinalgIdentity().annotate((HERE / "fixtures/mlir/matmul.mlir").read_text())
    return text, LinalgExtractor().extract(text).layers[0]


def test_real_result_uses_physical_spatial_factors_and_retains_temporal_order(payload):
    text, layer = payload
    schedule = ZigzagScheduleImporter().load(RESULT, layer, text, serialize_spatial=True)
    assert schedule.spatial_factors == {"M": 1, "N": 4, "K": 4}
    # Integer internal factors in this real result are N=3, K=2. They cannot
    # cover the workload with its temporal factors and are not physical tiles.
    assert [(loop.dimension, loop.factor, loop.step) for loop in schedule.loop_nest] == [
        ("K", 2, 4),
        ("N", 2, 4),
        ("M", 3, 1),
        ("N", 4, 1),
        ("K", 4, 1),
    ]
    assert schedule.memory_placement == "not_applied"
    assert schedule.execution_target == "cpu_temporal_projection"


def test_spatial_serialization_requires_explicit_opt_in(payload):
    text, layer = payload
    with pytest.raises(ScheduleError, match="explicitly enable spatial serialization"):
        ZigzagScheduleImporter().load(RESULT, layer, text)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("dimensions", "dimensions do not match"),
        ("precision", "precision does not match"),
        ("name", "name or dimensions"),
        ("coverage", "do not cover K"),
        ("order", "Operand-specific temporal orders"),
        ("code", "Invalid ZigZag loop factor"),
        ("summary", "malformed ZigZag"),
    ],
)
def test_incompatible_or_unsafe_results_are_rejected(payload, tmp_path, mutation, message):
    text, layer = payload
    result = json.loads(RESULT.read_text())
    inputs = result["inputs"]
    if mutation == "dimensions":
        inputs["layer"]["loop_dimensions"]["K"] = 6
    elif mutation == "precision":
        inputs["layer"]["operand_precision"]["O"] = 16
    elif mutation == "name":
        inputs["layer"]["name"] = "matmul_99"
    elif mutation == "coverage":
        for levels in inputs["temporal_mapping"].values():
            levels[0][-1] = "(K, 3)"
    elif mutation == "order":
        inputs["temporal_mapping"]["I"][0].reverse()
    elif mutation == "code":
        inputs["temporal_mapping"]["O"][0][0] = "__import__('os').system('false')"
    elif mutation == "summary":
        inputs["layer"] = [0]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ScheduleError, match=message):
        ZigzagScheduleImporter().load(path, layer, text, serialize_spatial=True)
