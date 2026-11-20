# Build and run generated Gemmini code

The executable build uses Bazel; `workload_config.h` is maintained manually.
Start with the [short guide in simulation/README.md](../simulation/README.md).
After preparing the header, `simulation/run.sh` builds the executable with
Bazel, updates the Chipyard simulator, runs it, and checks numerical results.

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

Then prepare `simulation/workload_config.h` using the template in the short
guide and run:

```bash
./simulation/run.sh "${WORKLOAD_DIR}"
```

The script copies Bazel's executable to the workload's `simulation/program.riscv`.
Rerun it after changing LLVM IR, C sources, or the manual header.

## Run program.riscv in Chipyard

The normal script invocation handles the build and run. To reuse an existing
executable and simulator without rebuilding either:

```bash
./simulation/run.sh "${WORKLOAD_DIR}" --run-only
```

See the [simulation guide](../simulation/README.md#results-and-logs) for results
and log locations. Chipyard execution uses Bash and Make; there is no Python
simulation package or runner.

## Optional developer checks

```bash
uv run pytest -q tests/test_simulation.py tests/test_simulation_runner.py
uv run ruff check src tests
```

The tests cover memref copies, a Bazel RISC-V ELF build, the native harness,
and runner orchestration with stub tools. The Bazel test skips when Bazel or
the compiler installations are missing. The tests do not execute Chipyard RTL.
