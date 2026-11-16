# Buddy–ZigZag

Search for accelerator mappings with ZigZag, then apply them in Buddy-MLIR
and generate LLVM IR for CPU or Gemmini.

```text
Torch → Buddy Linalg → workload YAML → ZigZag search
                                      ↓ selected mapping
                     tiled Linalg → Buddy lowering → LLVM IR
```

| Module | Input → output |
|---|---|
| `exporter/` · `ZigzagExporter` | Torch or saved Linalg → Linalg MLIR, ZigZag workload and mapping YAML |
| `hardware/` · `GemminiHardwareExporter` | Resolved Gemmini parameter JSON → ZigZag hardware YAML |
| `importer/` · `ZigzagImporter` | Original Linalg + ZigZag `*_complete.json` → scheduled Linalg and CPU/Gemmini LLVM IR |

ZigZag search runs through its Python API between export and import.
Pydantic validates workload, hardware, and schedule data.

The importer currently supports one FP32 matmul. Gemmini uses weight-stationary
(WS) tile projection with fixed kernel ordering. RISC-V linking and Chipyard
execution are a later step.

## Project layout

```text
src/buddy_zigzag/
    exporter/       # Torch/Linalg → ZigZag workload
    hardware/       # Gemmini parameters → hardware model
    importer/       # ZigZag mapping → scheduled and lowered code
    cli.py          # export, hardware-export, import
hardware/           # Hardware profiles and spatial mappings
third_party/        # Buddy-MLIR, ZigZag, Chipyard submodules
artifacts/          # Generated outputs; ignored by Git
docs/               # Installation, setup, and example workflow
tests/
```

## Get started

1. [Install the package](docs/installation.md).
2. [Set up native tools and hardware](docs/setup.md).
3. [Generate code from Torch](docs/quick_start.md).

For development: `uv run pytest -q` and `uv run ruff check src tests`.
