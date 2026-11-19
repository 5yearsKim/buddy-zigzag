# Bare-metal Gemmini build with Bazel

Maintain `workload_config.h` manually. Bazel handles the remaining executable
build without Python:

```text
06_lowered.ll ── Buddy llc ── kernel.o ──────────────┐
main.c + workload_config.h ── RISC-V GCC ── main.o ──┼── HTIF link ── program.riscv
memref_runtime.c ── RISC-V GCC ── memref_runtime.o ──┘
```

`main.c`, `memref_runtime.c`, and `memref_runtime.h` are reusable C sources
kept directly in this directory.
The static harness initializes deterministic FP32 inputs, creates MLIR memref
descriptors, calls the kernel, measures cycles, and checks every output against
a scalar reference. The memory runtime implements strided `memrefCopy` for
rank 0 through 64, including scalar, empty, and negative-stride cases.

## 1. Select the workload

Run from the root of your GitHub checkout, `buddy_zigzag/`, using the same
terminal for the following steps. `PRJ_DIRECTORY` below records that directory.
Bazel 7.4 or newer (or Bazelisk selecting such a version) must be available on
PATH. The lowered Gemmini LLVM IR must already exist.

```bash
PRJ_DIRECTORY="${PWD}"
WORKLOAD_NAME="matmul_small"
WORKLOAD_DIR="${PRJ_DIRECTORY}/artifacts/${WORKLOAD_NAME}"
export BUDDY_WORKLOAD_DIR="${WORKLOAD_DIR}"

mkdir -p "${WORKLOAD_DIR}/simulation"
```

For a custom compiler output directory, also export `BUDDY_LOWERED_DIR` to the
absolute directory containing `06_lowered.ll`.

## 2. Write workload_config.h manually

The following values match the default workload. For a different workload,
change the dimensions, entry point, and argument order to match its kernel.

```bash
cat > "${WORKLOAD_DIR}/simulation/workload_config.h" <<'EOF'
#ifndef BUDDY_ZIGZAG_WORKLOAD_CONFIG_H
#define BUDDY_ZIGZAG_WORKLOAD_CONFIG_H

#define WORKLOAD_M 3
#define WORKLOAD_N 7
#define WORKLOAD_K 5
#define WORKLOAD_ENTRY _mlir_ciface_forward
#define WORKLOAD_ARG0_IS_LHS 1
#define WORKLOAD_ATOL 1e-4
#define WORKLOAD_RTOL 1e-4

#endif
EOF
```

The dimensions are recorded in `lowered/schedule.json`; the C wrapper is
specified in `lowered/06_lowered.ll`. `WORKLOAD_ARG0_IS_LHS` is `1` when the
first input argument is the left matrix and `0` when inputs are reversed.
`WORKLOAD_ATOL` and `WORKLOAD_RTOL` are the numerical comparison tolerances.
Bazel reads this header and never generates or overwrites it.

## 3. Select the existing compiler installations

These paths match this checkout and are also the repository rule's defaults:

```bash
export BUDDY_LLVM_DIR="${PRJ_DIRECTORY}/third_party/buddy-mlir/llvm/build"
export RISCV_TOOLCHAIN_DIR="${PRJ_DIRECTORY}/third_party/chipyard/.conda-env/riscv-tools"
```

`BUDDY_LLVM_DIR/bin/llc` must support `+xgemmini`.
`RISCV_TOOLCHAIN_DIR/bin/riscv64-unknown-elf-gcc` must provide
`htif_nano.specs`, the HTIF startup code, and the bare-metal C library.
These variables are read from the shell, not Python's `.env` file.

## 4. Build the RISC-V executable

```bash
bazel build //simulation:program --jobs=16 \
  --repo_env=BUDDY_WORKLOAD_DIR="${BUDDY_WORKLOAD_DIR}" \
  --repo_env=BUDDY_LLVM_DIR="${BUDDY_LLVM_DIR}" \
  --repo_env=RISCV_TOOLCHAIN_DIR="${RISCV_TOOLCHAIN_DIR}"
```

Bazel runs separate actions for LLVM lowering, each C source, and linking.
It tracks the LLVM IR, both headers, reusable C sources, rule definitions,
and declared local compiler/runtime files. Changes to those inputs invalidate
the affected actions. Switching `BUDDY_WORKLOAD_DIR` selects another workload.
If using `BUDDY_LOWERED_DIR`, pass `--repo_env=BUDDY_LOWERED_DIR` as well.

Outputs:

```text
bazel-bin/simulation/program/
├── kernel.o
├── main.o
├── memref_runtime.o
└── program.riscv
```

Preserve the executable for this workload before building another one:

```bash
cp --remove-destination bazel-bin/simulation/program/program.riscv \
  "${WORKLOAD_DIR}/simulation/program.riscv"
file "${WORKLOAD_DIR}/simulation/program.riscv"
```

`--remove-destination` replaces an earlier read-only copy of Bazel's output.
Expected: a 64-bit little-endian RISC-V ELF executable. Use
`bazel build -s --jobs=16 //simulation:program` to print action commands on a rebuild.
Bazel owns intermediate objects; the earlier manual `llc` command and Python
`build-simulation` command are no longer needed.

## 5. From program.riscv: initialize the Chipyard environment

If the executable is already built and copied, start here. Run from the
`buddy_zigzag/` repository root; the commands also work in a fresh terminal.
Initialize Conda in your terminal before sourcing Chipyard's environment.
Set `WORKLOAD_NAME` to
the workload you built, or keep an existing `BUDDY_WORKLOAD_DIR` selection.
Paste only the commands inside the code blocks.

```bash
PRJ_DIRECTORY="${PWD}"
WORKLOAD_NAME="${WORKLOAD_NAME:-matmul_small}"
WORKLOAD_DIR="${BUDDY_WORKLOAD_DIR:-${PRJ_DIRECTORY}/artifacts/${WORKLOAD_NAME}}"
CHIPYARD_DIR="${PRJ_DIRECTORY}/third_party/chipyard"
SIM_CONFIG="FPGemminiRocketConfig"

file "${WORKLOAD_DIR}/simulation/program.riscv"

command -v conda
source "${CHIPYARD_DIR}/env.sh"
command -v verilator
```

The file check should identify an RV64 ELF executable, and the tool lookup
should print a Verilator path. Chipyard's `env.sh` must have been generated
for your checkout by its setup procedure.

`FPGemminiRocketConfig` uses the FP32 Gemmini configuration matching the bundled
4x4 snapshot. Use the same hardware configuration used to produce the workload
mapping and lowered code. RTL elaboration can regenerate Gemmini's C parameter
header and other files under Chipyard.

## 6. Build the RTL simulator

Run Step 5 first in the same terminal so `WORKLOAD_DIR`, `CHIPYARD_DIR`, and
`SIM_CONFIG` are defined. This step is separate from the Bazel executable
build and uses 16 parallel jobs. The commands below also export an optional
hardware snapshot during elaboration.

```bash
(
  set -eo pipefail
  : "${WORKLOAD_DIR:?Run Step 5 first in this terminal}"
  : "${CHIPYARD_DIR:?Run Step 5 first in this terminal}"
  : "${SIM_CONFIG:?Run Step 5 first in this terminal}"

  mkdir -p "${WORKLOAD_DIR}/simulation"
  export GEMMINI_CONFIG_NAME="${SIM_CONFIG}"
  export GEMMINI_CONFIG_JSON_DIR="${WORKLOAD_DIR}/simulation/rtl-config/${SIM_CONFIG}"
  mkdir -p "${GEMMINI_CONFIG_JSON_DIR}"

  make -C "${CHIPYARD_DIR}/sims/verilator" \
    CONFIG="${SIM_CONFIG}" -j16 2>&1 | \
    tee "${WORKLOAD_DIR}/simulation/simulator-build.log"
)
```

To build without exporting a hardware snapshot, replace the two `export`
lines and the following `mkdir` with `unset GEMMINI_CONFIG_JSON_DIR` inside
the block. This also clears any snapshot destination inherited from your shell.

If elaboration fails with `java.nio.file.AccessDeniedException: /simulation`,
check `GEMMINI_CONFIG_JSON_DIR`. Exporting it while `WORKLOAD_DIR` is empty
produces `/simulation/rtl-config/...`, where your account cannot create
directories. Run Step 5 and then rerun the complete block above to recompute
the destination; setting `WORKLOAD_DIR` alone does not update an earlier export.
Use a writable workload directory, or disable snapshot export as described above.

Continue when `make` exits successfully. If that exact simulator is already
built, this step can be skipped; `make` also reuses up-to-date build outputs.
The expected simulator is:

```text
third_party/chipyard/sims/verilator/simulator-chipyard.harness-FPGemminiRocketConfig
```

Actual elaboration exports `gemmini-0.json` under `GEMMINI_CONFIG_JSON_DIR`.
If cached elaboration is reused, a new snapshot may not be written. Keep any
elaborated snapshot paired with the simulator that produced it; compare its
hardware parameters with the snapshot recorded in `lowered/schedule.json`
when checking hardware identity. This is a manual check; there is no Python
simulation runner or automatic hardware comparison.

## 7. Run program.riscv on the RTL model

```bash
(
  set -o pipefail
  make -C "${CHIPYARD_DIR}/sims/verilator" \
    CONFIG="${SIM_CONFIG}" \
    run-binary-fast \
    BINARY="${WORKLOAD_DIR}/simulation/program.riscv" \
    LOADMEM=1 TIMEOUT_CYCLES=10000000 \
    output_dir="${WORKLOAD_DIR}/simulation/chipyard" 2>&1 | \
    tee "${WORKLOAD_DIR}/simulation/simulation.log"
)
```

This runs the selected executable in the Rocket/Gemmini Verilator model.
`LOADMEM=1` loads the ELF directly into the simulated memory.
`TIMEOUT_CYCLES` limits simulated cycles; increase it if your workload needs
more than ten million cycles. The `pipefail` setting preserves failures from
`make` while also showing and saving the output through `tee`.

## 8. Check numerical results and cycles

```bash
rg '^BZ_SIM_RESULT ' "${WORKLOAD_DIR}/simulation/simulation.log"
```

Expected output has this form:

```text
BZ_SIM_RESULT {"status":"passed","cycles":12345,"mismatches":0}
```

The cycle value above is illustrative. Success requires both a successful
simulator exit and a passing record with zero mismatches. Missing output, a
failed status, nonzero mismatches, or a cycle timeout means the run has not
passed. The reported cycles measure the kernel call including allocation,
copies, and synchronization; scalar reference verification runs afterward.

Files to inspect:

| Path under the workload directory | Contents |
|---|---|
| `simulation/program.riscv` | Executable copied from Bazel |
| `simulation/simulator-build.log` | RTL build command output |
| `simulation/simulation.log` | Complete run output, including the numerical JSON record |
| `simulation/chipyard/` | Additional Chipyard simulator logs |
| `simulation/rtl-config/<config>/gemmini-0.json` | Hardware parameters when exported during elaboration |

The C harness prints the JSON record into `simulation.log`. The shell commands
above handle execution, and the log files listed here hold the run results.

## 9. Optional waveforms

```bash
make -C "${CHIPYARD_DIR}/sims/verilator" \
  CONFIG="${SIM_CONFIG}" debug -j16

make -C "${CHIPYARD_DIR}/sims/verilator" \
  CONFIG="${SIM_CONFIG}" \
  run-binary-debug \
  BINARY="${WORKLOAD_DIR}/simulation/program.riscv" \
  LOADMEM=1 TIMEOUT_CYCLES=10000000 \
  output_dir="${WORKLOAD_DIR}/simulation/chipyard"
```

Chipyard writes the waveform and additional logs under
`${WORKLOAD_DIR}/simulation/chipyard/`.

## Build files

`BUILD.bazel` contains four ordinary `genrule` targets: `kernel_object`,
`main_object`, `runtime_object`, and `program`. Each contains the shell command
for that stage. `local_inputs.bzl` exposes the selected workload and existing
compiler files as one local repository. `MODULE.bazel` enables that repository.
There is no custom toolchain registration or `ToolchainInfo` layer.

Each command invokes the selected compiler directly. The `local = True` setting
allows GCC and LLVM to use their installed support files and host libraries.
This build uses local Linux compiler installations and does not download them.

Developer tests for the C runtime, Bazel ELF build, and native harness ABI are
in `tests/test_simulation.py`. These checks are separate from the build target
and do not launch Chipyard RTL.
