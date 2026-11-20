import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from buddy_zigzag import DsePipeline, ScheduleCompiler
from buddy_zigzag.codegen import ScheduleCompileRequest
from buddy_zigzag.codegen.gemmini.transforms import GemminiScheduleApplier
from buddy_zigzag.schedule import GemminiSchedule, GemminiScheduleProjector
from buddy_zigzag.schedule.errors import ScheduleError

HERE = Path(__file__).parent
RESULT = HERE / "fixtures/zigzag/gemmini_matmul_small_complete.json"
SNAPSHOT = HERE / "fixtures/hardware/fp32_parameters.json"


@pytest.fixture
def source(tmp_path):
    prepared = DsePipeline().prepare_case("matmul_small", output_dir=tmp_path / "workload")
    text = (prepared.output_dir / "01_imported.mlir").read_text()
    return prepared, text


def load(source, *, path=RESULT, tiles=None):
    prepared, text = source
    return GemminiScheduleProjector().load(path, prepared.workload.layers[0], text, SNAPSHOT, tiles=tiles)


def prepare_matmul(tmp_path, m, n, k):
    """Build rectangular and boundary cases without running mapping search."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mlir = tmp_path / "matmul.mlir"
    mlir.write_text(f"""module {{
  func.func @forward(%a: tensor<{m}x{k}xf32>, %b: tensor<{k}x{n}xf32>) -> tensor<{m}x{n}xf32> {{
    %zero = arith.constant dense<0.0> : tensor<{m}x{n}xf32>
    %result = linalg.matmul ins(%a, %b : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
      outs(%zero : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    return %result : tensor<{m}x{n}xf32>
  }}
}}
""")
    prepared = DsePipeline().prepare_mlir(mlir, output_dir=tmp_path / "workload")
    mapping = json.loads(RESULT.read_text())
    mapping["inputs"]["layer"]["loop_dimensions"] = {"M": m, "N": n, "K": k}
    factors = [f"(M, {m})", f"(N, {(n + 3) // 4})", f"(K, {(k + 3) // 4})"]
    mapping["inputs"]["temporal_mapping"] = {"I": [factors, []], "W": [[], factors, []], "O": [factors, []]}
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping))
    return prepared, path


def test_memory_tiles_are_aligned_and_keep_spatial_execution(source):
    schedule = load(source)
    assert schedule.tiles == {"M": 4, "N": 8, "K": 8}
    assert schedule.dataflow == "WS"
    assert schedule.accumulation == "resident_across_k"
    assert schedule.execution_target == "gemmini_ws_tile_projection"
    assert schedule.spatial_by_axis == {"D1": {"K": 4}, "D2": {"N": 4}}


@pytest.mark.parametrize("tiles", [(1, 4, 4), (4, 0, 4), (4, 12, 4)])
def test_illegal_tile_overrides_are_rejected(source, tiles):
    with pytest.raises(ValidationError):
        load(source, tiles=tiles)


@pytest.mark.parametrize("change", ["spatial", "hardware", "coverage"])
def test_incompatible_results_are_rejected(source, tmp_path, change):
    result = json.loads(RESULT.read_text())
    if change == "spatial":
        result["inputs"]["layer"]["user_spatial_mapping"] = {"D1": {"M": 4}, "D2": {"N": 4}}
    elif change == "hardware":
        result["inputs"]["accelerator"]["mem_size_dict"]["I1"][0] = 1024
    else:
        result["inputs"]["temporal_mapping"]["I"] = [[], []]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    with pytest.raises((ScheduleError, ValidationError)):
        load(source, path=path)


def test_hardware_modes_and_capacity_are_validated(source):
    values = load(source).model_dump()
    values["snapshot"]["dataflow"] = "OS"
    with pytest.raises(ValidationError, match="does not support WS"):
        GemminiSchedule.model_validate(values)
    values = load(source).model_dump()
    values["snapshot"]["accumulator"].update(rows_per_bank=1, capacity_bits=128)
    with pytest.raises(ValidationError, match="accumulator capacity"):
        GemminiSchedule.model_validate(values)


def test_applier_preserves_full_k_and_static_tail_shapes(source):
    from buddy_mlir import ir

    schedule = load(source, tiles=(4, 4, 4))
    text = GemminiScheduleApplier().apply(source[1], schedule)
    shapes = []
    with ir.Context():
        module = ir.Module.parse(text)

        def collect(op):
            if op.name == "linalg.matmul":
                shapes.append([tuple(ir.RankedTensorType(value.type).shape) for value in op.operands])
            return ir.WalkResult.ADVANCE

        module.operation.walk(collect)
    assert shapes == [[(3, 5), (5, 4), (3, 4)], [(3, 5), (5, 3), (3, 3)]]
    assert "tile_k = 4 : i64" in text
    with pytest.raises(ScheduleError, match="different MLIR"):
        GemminiScheduleApplier().apply(source[1] + "\n", schedule)


@pytest.mark.parametrize(
    "case,tiles,shapes,counts,bodies",
    [
        ("matmul_small", (4, 4, 4), ((3, 5), (5, 7)), (4, 2, 2), 2),
        ("matmul_rect", (16, 32, 8), ((64, 32), (32, 128)), (64, 16, 16), 1),
    ],
)
def test_gemmini_pipeline_emits_ws_intrinsics_and_verifies_linalg(tmp_path, case, tiles, shapes, counts, bodies):
    prepared = DsePipeline().prepare_case(case, output_dir=tmp_path / "workload")
    result = ScheduleCompiler().compile(
        ScheduleCompileRequest(
            workload_dir=prepared.output_dir,
            result=HERE / f"fixtures/zigzag/gemmini_{case}_complete.json",
            target="gemmini",
            snapshot=SNAPSHOT,
            gemmini_tiles=tiles,
            verify=True,
        )
    )
    out = result.output_dir
    assert out == prepared.output_dir / "lowered"
    assert not (out / "cpu_reference.mlir").exists()
    gemmini = (out / "04_gemmini.mlir").read_text()
    assert gemmini.count("gemmini.tile_matmul") == bodies
    assert gemmini.count("fullC = true") == bodies
    if case == "matmul_rect":
        assert gemmini.count("scf.for") == 2
        assert "llvm.cond_br" in (out / "05_lowered.mlir").read_text()
    assert "linalg.matmul" not in gemmini
    assert "llvm.riscv" in (out / "06_lowered.ll").read_text()
    report = json.loads((out / "validation.json").read_text())
    assert report["gemmini_execution"]["status"] == "not_run"
    assert report["native_verification"]["status"] == "passed"
    assert "not executed" in report["native_verification"]["scope"]

    # Run the actual lowered address arithmetic with test callbacks replacing
    # WS instructions. This catches wrong parent strides, slice offsets, early
    # stores and lost K accumulation; it does not simulate Gemmini hardware.
    import numpy as np
    from gemmini_emulator import WsCommandEmulator

    from buddy_zigzag.settings import Settings

    rng = np.random.default_rng(42)
    a, b = [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
    output = np.full((shapes[0][0], shapes[1][1]), np.nan, dtype=np.float32)
    emulator = WsCommandEmulator(4)
    emulator.execute(
        (out / "05_lowered.mlir").read_text(),
        "forward",
        [a, b],
        output,
        Settings.load().llvm_mlir_build_dir / "lib/libmlir_c_runner_utils.so",
    )
    np.testing.assert_allclose(output, a @ b, rtol=1e-4, atol=1e-5)
    assert (emulator.calls, emulator.loads, emulator.stores) == counts


@pytest.mark.parametrize(
    "m,n,k,bodies",
    [(9, 11, 5, 4), (9, 8, 5, 2), (8, 11, 5, 2), (8, 8, 8, 1), (1, 1, 1, 1)],
)
def test_runtime_tiles_cover_full_tiles_and_all_boundaries(tmp_path, m, n, k, bodies):
    import numpy as np
    from gemmini_emulator import WsCommandEmulator

    from buddy_zigzag.settings import Settings

    prepared, mapping = prepare_matmul(tmp_path, m, n, k)
    compiled = ScheduleCompiler().compile(
        ScheduleCompileRequest(
            workload_dir=prepared.output_dir,
            result=mapping,
            target="gemmini",
            snapshot=SNAPSHOT,
            gemmini_tiles=(4, 4, 4),
            verify=True,
        )
    )
    scheduled = (compiled.output_dir / "02_scheduled.mlir").read_text()
    assert scheduled.count("linalg.matmul") == bodies
    assert "tensor<?" not in scheduled
    assert compiled.verification["status"] == "passed"

    rng = np.random.default_rng(42)
    a, b = [rng.standard_normal(shape).astype(np.float32) for shape in ((m, k), (k, n))]
    output = np.full((m, n), np.nan, dtype=np.float32)
    emulator = WsCommandEmulator(4)
    emulator.execute(
        (compiled.output_dir / "05_lowered.mlir").read_text(),
        "forward",
        [a, b],
        output,
        Settings.load().llvm_mlir_build_dir / "lib/libmlir_c_runner_utils.so",
    )
    np.testing.assert_allclose(output, a @ b, rtol=1e-4, atol=1e-5)
    output_tiles = ((m + 3) // 4) * ((n + 3) // 4)
    assert (emulator.calls, emulator.loads, emulator.stores) == (
        output_tiles * ((k + 3) // 4),
        output_tiles,
        output_tiles,
    )


def test_generated_tile_code_does_not_grow_with_m_n_iterations(tmp_path):
    lowered = []
    for extent in (32, 512):
        # The large case exceeds the previous 4096-output-tile expansion cap.
        prepared, mapping = prepare_matmul(tmp_path / str(extent), extent, extent, 5)
        compiled = ScheduleCompiler().compile(
            ScheduleCompileRequest(
                workload_dir=prepared.output_dir,
                result=mapping,
                target="gemmini",
                snapshot=SNAPSHOT,
                gemmini_tiles=(4, 4, 4),
            )
        )
        scheduled = (compiled.output_dir / "02_scheduled.mlir").read_text()
        assert scheduled.count("scf.for") == 2
        assert scheduled.count("linalg.matmul") == 1
        text = (compiled.output_dir / "05_lowered.mlir").read_text()
        assert text.count('"gemmini.intr.loop_ws"') == 2  # K expansion is unchanged.
        assert text.count("llvm.cond_br") == 2
        lowered.append(text)
    # Increasing both M/N tile counts 16x must not duplicate the command body.
    assert abs(len(lowered[1].splitlines()) - len(lowered[0].splitlines())) < 20


def test_remaining_k_expansion_is_still_bounded(source):
    values = load(source, tiles=(4, 4, 4)).model_dump()
    values["dimensions"]["K"] = 4 * 65536
    for levels in values["temporal_by_operand"].values():
        for level in levels:
            for loop in level:
                if loop["dimension"] == "K":
                    loop["factor"] = 65536
    # The two N shape variants would each emit 65536 K commands.
    with pytest.raises(ValidationError, match="static WS call sites"):
        GemminiSchedule.model_validate(values)


def test_backend_options_are_separate():
    with pytest.raises(ValidationError, match="requires a resolved hardware"):
        ScheduleCompileRequest(workload_dir=Path("export"), result=RESULT, target="gemmini")
    with pytest.raises(ValidationError, match="omit --serialize-spatial"):
        ScheduleCompileRequest(
            workload_dir=Path("export"), result=RESULT, target="gemmini", snapshot=SNAPSHOT, serialize_spatial=True
        )
