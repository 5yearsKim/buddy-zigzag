# Torch to generated code

Complete [installation](installation.md) and Buddy [setup](setup.md) first.
Run every command below from the project root. The example uses the bundled
4×4 FP32 Gemmini snapshot, so Chipyard setup is optional. Its energy/area costs
are placeholders for exercising the pipeline.

## 1. Export the Torch workload and hardware

`matmul_small` runs this Torch model with shapes `(3, 5)` and `(5, 7)`:

```python
import torch

class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b
```

```bash
uv run buddy-zigzag export --case matmul_small \
  --mapping hardware/gemmini_fp32_ws_mapping.yaml \
  --out artifacts/quick_start/workload
uv run buddy-zigzag hardware-export \
  --snapshot tests/fixtures/hardware/fp32_parameters.json
```

Buddy imports Torch to Linalg; the exporter writes ZigZag workload YAML.
The explicit spatial mapping binds array rows to K and columns to N.
Hardware export writes `hw_config/hardware.yaml` and
`hw_config/hardware.manifest.json` in the Git-ignored `hw_config/` directory.

## 2. Search for a mapping with ZigZag

```bash
uv run python - <<'PYTHON'
from zigzag.api import get_hardware_performance_zigzag

get_hardware_performance_zigzag(
    workload="artifacts/quick_start/workload/workload.yaml",
    accelerator="hw_config/hardware.yaml",
    mapping="artifacts/quick_start/workload/mapping.yaml",
    dump_folder="artifacts/quick_start/evaluation",
    nb_spatial_mappings_generated=1,
    lpf_limit=4,
    loma_show_progress_bar=False,
)
PYTHON
```

## 3. Import the mapping and generate Gemmini code

```bash
uv run buddy-zigzag import --target gemmini \
  --export-dir artifacts/quick_start/workload \
  --result artifacts/quick_start/evaluation/matmul_0_complete.json \
  --snapshot tests/fixtures/hardware/fp32_parameters.json \
  --gemmini-tiles 4 4 4 --verify
```

The importer applies M/N tiles to Linalg. Buddy lowers each tile to Gemmini WS
operations, retaining accumulation across K tiles. The final `lowered.ll`
contains Gemmini RISC-V intrinsics; linking and Chipyard execution come later.
`--verify` executes the tiled Linalg on CPU and compares it with Torch.

Key outputs under `artifacts/quick_start/`:

| File | Meaning |
|---|---|
| `workload/imported.mlir` | Original Buddy Linalg |
| `evaluation/matmul_0_complete.json` | Selected ZigZag mapping |
| `workload/gemmini_importer/schedule.json` | Validated tile projection and source hashes |
| `workload/gemmini_importer/scheduled.mlir` | Tiled Linalg |
| `workload/gemmini_importer/gemmini.mlir` | Gemmini operations |
| `workload/gemmini_importer/lowered.ll` | Generated LLVM IR |
| `workload/gemmini_importer/validation.json` | Compilation and CPU verification results |

For CPU code generation from the same mapping:

```bash
uv run buddy-zigzag import --target cpu \
  --export-dir artifacts/quick_start/workload \
  --result artifacts/quick_start/evaluation/matmul_0_complete.json \
  --serialize-spatial --verify
```

CPU output is in `workload/importer/`. To use your own hardware, replace the demo
snapshot and hardware YAML with the matching files generated in [setup](setup.md).
