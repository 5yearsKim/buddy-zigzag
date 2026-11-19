# Buddy–ZigZag

Search for accelerator mappings with ZigZag, then apply them in Buddy-MLIR
and generate LLVM IR for CPU or Gemmini.

```text
Torch / saved Linalg → workload description ──→ workload.yaml ───────┐
                             │                                     │
Gemmini snapshot ────────────┼──→ mapping constraints → mapping.yaml ┼→ ZigZag search
       │                     │                                     │       ↓
       └→ hardware model ────┴──────────────────────→ hardware.yaml ─┘ selected mapping
                                                                           ↓
Original Linalg ───────────────────────────────────────────────→ target schedule
                                                                           ↓
                                                              tiled Linalg → LLVM IR
```

| Module | Input → output |
|---|---|
| `workload/` · `WorkloadBuilder` | Named Torch example or saved Linalg → preserved Linalg, structured workload, workload YAML |
| `hardware/` · `GemminiHardwareExporter` | Resolved Gemmini parameter JSON → ZigZag hardware YAML |
| `mapping/` · `MappingConstraintGenerator` | Structured workload + hardware snapshot + backend capabilities → mapping constraints YAML |
| `search/` · `ZigzagSearchRunner` | Workload, hardware, and mapping YAML → evaluated mappings, energy/latency estimates, result files |
| `schedule/` · target projectors | Selected ZigZag result + workload identity + target constraints → validated CPU or Gemmini schedule |
| `codegen/` · `ScheduleCompiler` | Original Linalg + selected result → scheduled Linalg and CPU/Gemmini LLVM IR |
| `pipeline.py` · `DsePipeline` | Coordinate workload/mapping preparation, search, and compilation |

`WorkloadBuilder` handles the computation independently of hardware. Dimension
normalization lives with Linalg analysis in `workload/linalg.py`, and persistent
operation IDs live in `workload/identity.py`. Pydantic validates workload,
hardware, mapping capabilities, and schedule data.

`DsePipeline.prepare()` calls the workload builder, selects or generates mapping
constraints, and checks the combined inputs against ZigZag's parser. Its `search()`
and `compile()` methods invoke the search runner and schedule compiler.
The compiler uses `schedule/` to translate the chosen mapping before applying
CPU or Gemmini transformations.

Each workload is saved under `artifacts/<case>/`: `01_imported.mlir` holds the
original Linalg, `zigzag/` holds search results, and `lowered/` holds the numbered
stages `02_scheduled.mlir`, `03_bufferized.mlir`, `04_gemmini.mlir`,
`05_lowered.mlir`, and `06_lowered.ll`. Gemmini verification does not save
separate CPU LLVM IR files.

Schedule compilation currently supports one FP32 matmul. Gemmini uses
weight-stationary (WS) tile projection with fixed kernel ordering. Maintain
`workload_config.h` manually, then use Bazel to run Buddy's `llc`, cross-compile
the static C harness/runtime, and link `program.riscv`. Follow
[simulation/README.md](simulation/README.md) to build and run through Chipyard
using Bazel and shell commands. The Python pipeline stops at LLVM IR generation.

Use `prepare-workload --snapshot <Gemmini JSON>` to generate layer-specific
mapping constraints after workload extraction. The current rule validates one
canonical FP32 matmul and Gemmini WS hardware, then maps K/N onto array
rows/columns. Unsupported operations, layouts, and backend combinations are
rejected. `--mapping` overrides generation; without either option, preparation
uses the packaged baseline mapping.

## Project layout

```text
src/buddy_zigzag/
    workload/       # Computation extraction, identity, validation, and YAML
    hardware/       # Gemmini parameters → hardware model
    mapping/        # Workload + hardware + backend → search constraints
    search/         # ZigZag search and cost evaluation
    schedule/       # Selected results → CPU/Gemmini schedule projections
    codegen/        # Target transformations and lowering
        cpu/
        gemmini/
    examples/       # Reproducible Torch models and case definitions
    common/         # Shared schema types, runtime errors, and artifact helpers
    pipeline.py     # DsePipeline orchestration and PrepareRequest
    settings.py     # Native runtime configuration
    cli.py          # Preparation, search, and compilation
simulation/         # Bazel build commands, local compiler inputs, reusable C sources, and terminal guide
MODULE.bazel        # Bazel module and local input selection
hw_config/          # Generated hardware YAML, manifests, and parameter snapshots; ignored by Git
third_party/        # Buddy-MLIR, ZigZag, Chipyard submodules
artifacts/          # Generated workloads, search results, and code; ignored by Git
docs/               # Installation, setup, and example workflow
tests/
```

## Python API

```python
from pathlib import Path

from buddy_zigzag import DsePipeline, ScheduleCompileRequest, SearchRequest

pipeline = DsePipeline()
snapshot = Path("tests/fixtures/hardware/fp32_parameters.json")
built = pipeline.prepare_case("matmul_small", snapshot=snapshot)
searched = pipeline.search(SearchRequest(
    workload=built.output_dir / "workload.yaml",
    hardware=Path("hw_config/hardware.yaml"),
    mapping=built.output_dir / "mapping.yaml",
))
compiled = pipeline.compile(ScheduleCompileRequest(
    workload_dir=built.output_dir,
    result=searched.mappings[0],
    target="gemmini",
    snapshot=snapshot,
))
```

Generate `hw_config/hardware.yaml` first as described in the quick start.
For computation-only preparation, use `WorkloadBuilder.build_case()` or
`build_mlir()`; this produces workload artifacts without mapping constraints.

## Migration from importer/exporter names

- Replace `ZigzagExporter.export_case()` / `export_mlir()` with
  `DsePipeline.prepare_case()` / `prepare_mlir()` when mapping preparation is needed.
- Replace `ZigzagImporter.run(ImportRequest(...))` with
  `ScheduleCompiler.compile(ScheduleCompileRequest(...))`, using `workload_dir`.
- Replace `MappingGenerator` with `MappingConstraintGenerator`.
- CLI commands `export`, `hardware-export`, and `import` are now
  `prepare-workload`, `prepare-hardware`, and `compile`; `--export-dir` is now
  `--workload-dir`. ZigZag search is available through `search`.
- Compilation manifests use `source_workload` instead of `source_export`.

## Get started

1. [Install the package](docs/installation.md).
2. [Set up native tools and hardware](docs/setup.md).
3. [Generate code from Torch](docs/quick_start.md).
4. [Build and run a bare-metal RTL simulation](docs/simulation.md).

For development: `uv run pytest -q` and `uv run ruff check src tests`.
