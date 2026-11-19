# Setup

Run commands from the project root unless a block changes directory.
For the existing workspace, reuse the current builds and skip the fresh-build
sections. Keep the old `../buddy-mlir`, `../zigzag`, and `../chipyard` compatibility
symlinks: existing builds and environments contain those paths.

## Environment

The CLI reads `.env` automatically. Start from `.env.example`; paths are relative
to that file, and process environment variables take precedence.

| Setting | Purpose |
|---|---|
| `BUDDY_MLIR_BUILD_DIR` | Buddy tools and Python bindings |
| `LLVM_MLIR_BUILD_DIR` | LLVM/MLIR tools and runner libraries |
| `BUDDY_ZIGZAG_PYTHON` | Interpreter used to build Buddy; checked for Python ABI compatibility |
| `ZIGZAG_SOURCE_DIR` | ZigZag checkout |
| `PYTHONPATH` | Additional Python import paths |
| `ARTIFACTS_DIR` | Generated output directory |

## Fresh Buddy/LLVM build

Requires a C/C++17 toolchain, FlatBuffers compiler/headers, and NUMA development
headers. See [Buddy's prerequisites](../third_party/buddy-mlir/README.md#dependencies).
This block creates Buddy's separate Python environment, matching `.env.example`.
Skip it when reusing this workspace's binaries.

```bash
git -C third_party/buddy-mlir submodule update --init llvm
(
  cd third_party/buddy-mlir
  uv venv --python 3.13 .venv
  uv pip install --python .venv/bin/python -r requirements.txt \
    -r llvm/mlir/python/requirements.txt cmake ninja
  source .venv/bin/activate

  cmake -G Ninja -S llvm/llvm -B llvm/build \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_ENABLE_RUNTIMES=openmp -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
    -DPython_EXECUTABLE="$PWD/.venv/bin/python"
  cmake --build llvm/build --parallel 2

  cmake -G Ninja -S . -B build \
    -DMLIR_DIR="$PWD/llvm/build/lib/cmake/mlir" \
    -DLLVM_DIR="$PWD/llvm/build/lib/cmake/llvm" \
    -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
    -DPython_EXECUTABLE="$PWD/.venv/bin/python"
  cmake --build build --parallel 2
)
```

## Fresh Chipyard setup

Needed to export parameters from your own Gemmini configuration or build and
run the RTL simulator described in the [simulation guide](simulation.md).
With Conda installed, initialize Chipyard and its optional Gemmini dependency:

```bash
git submodule update --init third_party/chipyard
(
  cd third_party/chipyard
  ./build-setup.sh --use-lean-conda
  ./scripts/init-submodules-no-riscv-tools.sh --gemmini
)
```

This installs Chipyard's tools and prepares Scala sources. The quick start uses
a bundled snapshot and does not require this step.

## Generate hardware configuration

With Chipyard prepared and the Gemmini JSON exporter present, export the resolved
`GemminiFPConfigs.FP32DefaultConfig` parameters:

```bash
mkdir -p hw_config
(
  bridge_root="$PWD"
  cd third_party/chipyard
  source env.sh
  sbt "gemmini/runMain gemmini.GemminiConfigExportMain --output $bridge_root/hw_config/fp32_parameters.json"
)
uv run buddy-zigzag prepare-hardware \
  --snapshot hw_config/fp32_parameters.json
```

The default outputs are `hw_config/hardware.yaml` and
`hw_config/hardware.manifest.json`; use `--out` to choose another path.
The generated `hw_config/` directory is ignored by Git. This snapshot resolves
the Scala configuration object; it does not elaborate a complete SoC.
To capture actual elaboration parameters, set `GEMMINI_CONFIG_NAME` and
`GEMMINI_CONFIG_JSON_DIR` during a Chipyard build.

The current model supports square FP32 arrays with 1×1 PE tiles and WS dataflow.
Energy/area costs are placeholders. When changing the array size, regenerate the
hardware YAML and prepare workloads with `--snapshot` pointing to the same JSON.
`MappingConstraintGenerator` derives the K/N spatial factors from those resolved dimensions.

Continue with the [quick start](quick_start.md).
