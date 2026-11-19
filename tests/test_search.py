"""Search failures retain a report and preserve the prepared inputs."""

import json

import pytest

from buddy_zigzag import SearchRequest, ZigzagSearchRunner


def test_failed_search_records_the_error_and_preserves_inputs(tmp_path, monkeypatch):
    inputs = [tmp_path / name for name in ("workload.yaml", "hardware.yaml", "mapping.yaml")]
    for path in inputs:
        path.write_text("prepared input")

    def fail(**kwargs):
        raise RuntimeError("Search engine failed")

    monkeypatch.setattr("zigzag.api.get_hardware_performance_zigzag", fail)
    with pytest.raises(RuntimeError, match="Search engine failed"):
        ZigzagSearchRunner().run(SearchRequest(workload=inputs[0], hardware=inputs[1], mapping=inputs[2]))
    report = json.loads((tmp_path / "zigzag/search.json").read_text())
    assert report["status"] == "failed"
    assert report["error"] == "Search engine failed"
    assert all(path.read_text() == "prepared input" for path in inputs)
