"""Exercise project passes in the unmodified Buddy host, including rejection paths."""

from pathlib import Path

import pytest

from buddy_zigzag.codegen.gemmini.plugin import build_plugin
from buddy_zigzag.codegen.utils import run_tool
from buddy_zigzag.schedule.errors import ScheduleError
from buddy_zigzag.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures/gemmini"


@pytest.fixture(scope="module")
def plugin_tool():
    settings = Settings.load()
    plugin = build_plugin(settings)
    return settings.buddy_mlir_build_dir / "bin/buddy-opt", f"--load-pass-plugin={plugin}"


def test_project_conversion_preserves_destination_and_schedule(plugin_tool):
    tool, plugin = plugin_tool
    text = (FIXTURES / "linalg-to-gemmini-schedule.mlir").read_text()
    converted = run_tool(tool, [plugin, "--pass-pipeline=builtin.module(buddy-zigzag-convert-linalg-to-gemmini)"], text)
    assert "gemmini.tile_matmul %arg0 %arg1 %arg2 %arg2" in converted
    assert "gemmini.schedule" in converted
    assert "fullC = true" in converted
    assert "memref.alloc" not in converted
    assert "linalg.matmul" not in converted


def test_project_lowering_keeps_explicit_k_tiles(plugin_tool):
    from buddy_mlir import ir

    tool, plugin = plugin_tool
    text = (FIXTURES / "gemmini-explicit-tiles.mlir").read_text()
    lowered = run_tool(tool, [plugin, "--pass-pipeline=builtin.module(buddy-zigzag-lower-gemmini-tiles)"], text)
    assert lowered.count('"gemmini.intr.loop_ws"') == 2
    assert "gemmini.tile_matmul" not in lowered
    assert "llvm.fence" in lowered
    # Check the hardware configuration words too: the functional WS emulator
    # intentionally does not model the configuration instructions themselves.
    words = []
    with ir.Context():
        module = ir.Module.parse(lowered)

        def collect(op):
            if op.name == "gemmini.intr.config":
                words.append(tuple(ir.IntegerAttr(value.owner.attributes["value"]).value for value in op.operands))
            return ir.WalkResult.ADVANCE

        module.operation.walk(collect)
    assert words == [
        (4575657221408489476, 281474976710656),  # WS execute config, unit strides
        (2, 4575657221408423952),  # FP32 store, 4-element output row
        (4575657221408686337, 20),  # A load, 5-element row
        (4575657221408686345, 16),  # B load, 4-element row
        (4575657221408686353, 16),  # D load, 4-element row
    ]


@pytest.mark.parametrize(
    "option",
    ["dim=8", "bank_rows=2048", "acc_rows=2048"],
)
def test_project_lowering_rejects_different_hardware(plugin_tool, option):
    tool, plugin = plugin_tool
    text = (FIXTURES / "gemmini-explicit-tiles.mlir").read_text()
    with pytest.raises(ScheduleError, match="schedule disagrees with lowering hardware"):
        run_tool(tool, [plugin, f"--pass-pipeline=builtin.module(buddy-zigzag-lower-gemmini-tiles{{{option}}})"], text)


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("tile_k = 4", "tile_k = 3", "positive array multiples"),
        ("tile_k = 4", "tile_k = 65536", "memory capacity"),
        ("fullC = true", "fullC = false", "full-C FP32 WS"),
        ("fullC = true", "fullC = true, aScaleFactor = 2.0 : f32", "identity FP32 scaling"),
    ],
)
def test_project_lowering_rejects_invalid_contracts(plugin_tool, old, new, message):
    tool, plugin = plugin_tool
    text = (FIXTURES / "gemmini-explicit-tiles.mlir").read_text().replace(old, new)
    with pytest.raises(ScheduleError, match=message):
        run_tool(tool, [plugin, "--pass-pipeline=builtin.module(buddy-zigzag-lower-gemmini-tiles)"], text)


def test_plugin_requires_selected_llvm_build(tmp_path):
    with pytest.raises(ScheduleError, match="LLVM_MLIR_BUILD_DIR must be configured"):
        build_plugin(Settings(project_dir=tmp_path, llvm_mlir_build_dir=None, _env_file=None))
