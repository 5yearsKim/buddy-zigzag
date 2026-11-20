# Torch to generated code

Complete [installation](installation.md) and Buddy [setup](setup.md) first.
Run every command below from the project root. The example uses the bundled
4×4 FP32 Gemmini snapshot, so Chipyard setup is optional. Its energy/area costs
are placeholders for exercising the pipeline.

## 1. Prepare the Torch workload and hardware

`matmul_small` runs this Torch model with shapes `(3, 5)` and `(5, 7)`:

```python
import torch

class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b
```

```bash
uv run buddy-zigzag prepare-workload --case matmul_small \
  --snapshot tests/fixtures/hardware/fp32_parameters.json
uv run buddy-zigzag prepare-hardware \
  --snapshot tests/fixtures/hardware/fp32_parameters.json
```

Buddy imports Torch to Linalg; the workload builder writes ZigZag workload YAML.
With `--snapshot`, `MappingConstraintGenerator` inspects the extracted workload and checks
the current Gemmini backend's capabilities before generating `mapping.yaml`.
For supported FP32 matmul, array rows map to K and columns to N, with factors
derived from the snapshot. This produces a named entry for the matmul layer and
a baseline default entry required by ZigZag. Hardware export does not generate
workload mappings.
Hardware export writes `hw_config/hardware.yaml` and
`hw_config/hardware.manifest.json` in the Git-ignored `hw_config/` directory.

The initial mapping rule supports one canonical, unbatched FP32 matmul with two
external inputs. Other workloads require additional mapping and lowering rules.
Use `--mapping <file>` to supply an explicit override; it takes precedence over
automatic generation. Without `--snapshot` or `--mapping`, the packaged baseline
mapping is used. The preparation manifest records the mapping mode and mapping hash;
automatic generation also records the backend and snapshot hash.

## 2. Search for a mapping with ZigZag

```bash
uv run buddy-zigzag search \
  --workload-dir artifacts/matmul_small \
  --hardware hw_config/hardware.yaml
```

The command runs ZigZag through `ZigzagSearchRunner`. Its defaults use the
latency objective, one spatial candidate, and an LPF limit of four. Use
`--objective`, `--spatial-candidates`, and `--lpf-limit` to change these settings.
`zigzag/search.json` records the request, costs, and selected result paths.

## 3. Compile the selected mapping for Gemmini

```bash
uv run buddy-zigzag compile --target gemmini \
  --workload-dir artifacts/matmul_small \
  --result artifacts/matmul_small/zigzag/matmul_0_complete.json \
  --snapshot tests/fixtures/hardware/fp32_parameters.json \
  --gemmini-tiles 4 4 4 --verify
```

The schedule compiler emits runtime M/N tile loops with separate static boundary
shapes. Buddy lowers the tile bodies to Gemmini WS operations, retaining
accumulation across K tiles; K commands remain expanded at compile time.
In `02_scheduled.mlir`, look for `scf.for`; surviving loops become branches in
`05_lowered.mlir`. Single-iteration loops can disappear during canonicalization.
The final `06_lowered.ll`
contains Gemmini RISC-V intrinsics. Continue with the
[simulation guide](simulation.md) to link and execute it in Chipyard.
`--verify` executes the tiled Linalg on CPU and compares it with Torch, without
saving a separate CPU LLVM IR artifact. Omit it to skip CPU verification.

Key outputs under `artifacts/matmul_small/`:

| File | Meaning |
|---|---|
| `01_imported.mlir` | Original Buddy Linalg |
| `mapping.yaml` | Generated layer-specific spatial constraints and operand links |
| `zigzag/matmul_0_complete.json` | Selected ZigZag mapping |
| `lowered/schedule.json` | Validated tile projection and source hashes |
| `lowered/02_scheduled.mlir` | Tiled Linalg |
| `lowered/03_bufferized.mlir` | Linalg operating on memory buffers |
| `lowered/04_gemmini.mlir` | Gemmini operations |
| `lowered/05_lowered.mlir` | LLVM dialect and Gemmini intrinsics |
| `lowered/06_lowered.ll` | LLVM IR with Gemmini RISC-V intrinsics |
| `lowered/validation.json` | Compilation and optional CPU verification results |

## 4. Generate the larger matmul

The bundled `matmul_256x512x1024` case has `M=256`, `N=512`, `K=1024`:
input matrices are `(256, 1024)` and `(1024, 512)`. Prepare it with:

```bash
uv run buddy-zigzag prepare-workload --case matmul_256x512x1024 \
  --snapshot tests/fixtures/hardware/fp32_parameters.json
```

Run the ZigZag search above with `artifacts/matmul_256x512x1024` in place of
`artifacts/matmul_small`, then lower the result:

```bash
uv run buddy-zigzag compile --target gemmini \
  --workload-dir artifacts/matmul_256x512x1024 \
  --result artifacts/matmul_256x512x1024/zigzag/matmul_0_complete.json \
  --snapshot tests/fixtures/hardware/fp32_parameters.json
```

This uses tiles derived from the selected mapping, adjusted for Gemmini's memory
constraints. CPU LLVM lowering remains available through the Python API and
explicit `--target cpu` requests; the Gemmini workflow above emits only Gemmini
LLVM IR. To use your own hardware, replace the demo snapshot and hardware YAML
with the matching files generated in [setup](setup.md).
