# Build and run a Gemmini simulation

Once your workload has Gemmini LLVM IR and a workload header, run:

```bash
./simulation/run.sh artifacts/matmul_small
```

The script builds `program.riscv` with Bazel, copies it into the workload,
loads Chipyard's environment, updates the RTL simulator with Make, runs the
executable, and checks the numerical result. Existing build outputs are reused.
It prints `PASS — cycles: ...` on success and exits with a short error on failure.
Build and simulator output is shown in the terminal and saved in log files.

## Set up once

You need Bash, Bazel 7.4+ (or Bazelisk), Buddy LLVM with `+xgemmini` support,
and the RISC-V bare-metal GCC toolchain with `htif_nano.specs`.
For RTL execution, finish Chipyard setup so `third_party/chipyard/env.sh` exists,
and make Conda available to the script. Chipyard's environment must provide
Verilator and its other build dependencies.

The default compiler paths are:

- `third_party/buddy-mlir/llvm/build/bin/llc`
- `third_party/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc`

The workload must already contain `lowered/06_lowered.ll`. See
[workload preparation](../docs/simulation.md#prepare-llvm-ir-if-needed) if it is
missing. The script does not run mapping search or compile the workload to IR.

## Prepare the header once per workload

From the repository root:

```bash
mkdir -p artifacts/matmul_small/simulation
cp -n simulation/workload_config.h.in artifacts/matmul_small/simulation/workload_config.h
```

`workload_config.h.in` is a template. Edit the copied `workload_config.h` and
replace every `@WORKLOAD_...@` placeholder before running the script:

| Setting | Where to find the value |
|---|---|
| `WORKLOAD_M`, `WORKLOAD_N`, `WORKLOAD_K` | Dimensions in `lowered/schedule.json` |
| `WORKLOAD_ENTRY` | C wrapper symbol in `lowered/06_lowered.ll` |
| `WORKLOAD_ARG0_IS_LHS` | `1` if the wrapper takes LHS first; `0` if inputs are reversed |
| `WORKLOAD_ATOL`, `WORKLOAD_RTOL` | Numerical comparison tolerances; both default to `1e-4` |

For `matmul_small`, use `M=3`, `N=7`, `K=5`,
`ENTRY=_mlir_ciface_forward`, and `ARG0_IS_LHS=1`. For example,
`#define WORKLOAD_M @WORKLOAD_M@` becomes `#define WORKLOAD_M 3`.
The tolerances already have defaults and need no substitution.

The header stays manual. Neither the script nor Bazel overwrites it. Update it
when the dimensions or kernel interface change. The reusable harness currently
checks one FP32 matrix multiplication against a scalar reference.

## Run

```bash
# Build, update the simulator, run, and verify.
./simulation/run.sh artifacts/matmul_small

# Build and copy only the executable; no Conda or simulator needed.
./simulation/run.sh artifacts/matmul_small --build-only

# Reuse an existing executable and simulator; skip both builds.
./simulation/run.sh artifacts/matmul_small --run-only
```

Use the normal command after changing LLVM IR, the header, C sources, or RTL.
`--run-only` deliberately skips freshness checks; it requires both an existing
`simulation/program.riscv` and the simulator for the selected config.

Defaults are `--config FPGemminiRocketConfig`, `--jobs 16`, and
`--timeout-cycles 10000000`. For example:

```bash
./simulation/run.sh artifacts/matmul_small --jobs 8 --timeout-cycles 20000000
./simulation/run.sh --help
```

Absolute paths work too. All relative paths, including environment overrides,
are resolved from the repository root, so the script can be called from another
directory using its full path. The positional workload always takes precedence
over any old `BUDDY_WORKLOAD_DIR` selection in your shell.

Set these variables only if the defaults do not match your installation:

| Variable | Purpose |
|---|---|
| `BUDDY_LLVM_DIR` | Buddy LLVM build directory containing `bin/llc` |
| `RISCV_TOOLCHAIN_DIR` | Toolchain directory containing `bin/riscv64-unknown-elf-gcc` |
| `BUDDY_LOWERED_DIR` | Alternate directory containing `06_lowered.ll` |
| `CHIPYARD_DIR` | Alternate Chipyard checkout; also sets the default toolchain location |

These are shell environment variables; the script does not read Python's `.env`.
If Conda is activated but absent from PATH, the script also accepts `CONDA_EXE`
from that activated environment. You do not need to source Chipyard manually.

## Results and logs

A passing run needs a successful simulator exit and exactly one complete
`BZ_SIM_RESULT` record with `"status":"passed"` and `"mismatches":0`.
The script checks the fixed record format emitted by `main.c`; missing,
malformed, duplicate, or failed records cause a nonzero exit.
The cycle count covers the kernel call, including its allocation, copies,
and synchronization. Scalar reference verification runs afterward.

All paths below are relative to your workload:

| Path | Contents |
|---|---|
| `simulation/program.riscv` | Executable copied from Bazel |
| `simulation/program-build.log` | Executable build output |
| `simulation/simulator-build.log` | RTL build output |
| `simulation/simulation.log` | Latest simulator run and numerical result |
| `simulation/chipyard/` | Additional Chipyard output |

A stage's log is replaced when that stage runs again. A failing run never uses
an older result to report success. Build-only and run-only leave skipped stages'
logs intact.

`FPGemminiRocketConfig` is the FP32 config matching the bundled 4x4 snapshot.
Select the same hardware used for mapping and compilation. RTL elaboration can
regenerate Gemmini's parameter header and other files under Chipyard. Hardware
JSON export is a separate project-owned command, described in
[setup](../docs/setup.md#generate-hardware-configuration); no source hook is added
to Gemmini. Keep the snapshot paired with the selected simulator configuration.
The script does not verify hardware identity.

Developer checks are described in [docs/simulation.md](../docs/simulation.md#optional-developer-checks).
