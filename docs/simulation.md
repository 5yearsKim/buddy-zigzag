# Build and run generated Gemmini code

The executable build uses Bazel; `workload_config.h` is maintained manually.
Start with the [terminal steps in simulation/README.md](../simulation/README.md).
They cover selecting a workload, writing the header, running Bazel, and
executing the ELF through Chipyard without Python.

Run commands from the project root. Simulator compilation can regenerate
Gemmini's parameter header and other elaboration outputs under Chipyard.

## Prepare LLVM IR if needed

Skip this section when the corresponding workload and Gemmini compilation
already exist. These are the original project preparation stages, before the
Bazel build starts.

```bash
WORKLOAD_NAME="matmul_small"
WORKLOAD_DIR="${PWD}/artifacts/${WORKLOAD_NAME}"

uv run buddy-zigzag prepare-workload --case "${WORKLOAD_NAME}" \
  --snapshot tests/fixtures/hardware/fp32_parameters.json
uv run buddy-zigzag prepare-hardware \
  --snapshot tests/fixtures/hardware/fp32_parameters.json
uv run buddy-zigzag search \
  --workload-dir "${WORKLOAD_DIR}" --hardware hw_config/hardware.yaml
uv run buddy-zigzag compile --target gemmini \
  --workload-dir "${WORKLOAD_DIR}" \
  --result "${WORKLOAD_DIR}/zigzag/matmul_0_complete.json" \
  --snapshot tests/fixtures/hardware/fp32_parameters.json \
  --gemmini-tiles 4 4 4
```

Then follow the Bazel build instructions. No Python builder or `build.json`
is required. Bazel produces `bazel-bin/simulation/program/program.riscv`;
copy it to `${WORKLOAD_DIR}/simulation/program.riscv` to preserve it per workload.
Rebuild and recopy after changing LLVM IR, C sources, or the manual header.

## Run program.riscv in Chipyard

Continue with [Step 5 in simulation/README.md](../simulation/README.md#5-from-programriscv-initialize-the-chipyard-environment).
The instructions initialize the shell environment, build the FP32 Verilator
simulator, run the executable, and inspect the pass/fail record and cycle count.
They also cover logs and optional waveforms. Chipyard execution uses shell
commands directly; the Python simulation package, API, and CLI command have
been removed.

## Optional developer checks

```bash
uv run pytest -q tests/test_simulation.py
uv run ruff check src tests
```

The tests cover strided/scalar/empty memref copies, a Bazel RISC-V ELF build,
and the static harness against a native CPU kernel. The Bazel test skips when
Bazel or the compiler installations are missing. The tests do not build or
execute Chipyard RTL.
