#!/usr/bin/env bash
# Build a prepared workload with Bazel, then run it in Chipyard's RTL simulator.
# Keep workload_config.h manual; this script only coordinates existing tools.
set -Eeuo pipefail

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: simulation/run.sh WORKLOAD_DIR [options]

Build the executable, update the simulator, run, and check the result.
Relative paths are resolved from the buddy_zigzag repository root.

Options:
  --build-only          Build and copy program.riscv; no Chipyard needed
  --run-only            Reuse the existing executable and simulator
  --config NAME         Chipyard config (default: FPGemminiRocketConfig)
  --jobs N              Parallel build jobs (default: 16)
  --timeout-cycles N    Simulation cycle limit (default: 10000000)
  -h, --help            Show this help

Examples:
  ./simulation/run.sh artifacts/matmul_small
  ./simulation/run.sh artifacts/matmul_small --build-only
  ./simulation/run.sh artifacts/matmul_small --run-only

Optional environment overrides:
  BUDDY_LLVM_DIR, RISCV_TOOLCHAIN_DIR, BUDDY_LOWERED_DIR, CHIPYARD_DIR
EOF
}

# Resolve paths before loading Chipyard, which changes the shell environment.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
repo_path() {
  if [[ $1 = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$repo_root" "$1"
  fi
}

workload=''
mode=all
config=FPGemminiRocketConfig
jobs=16
timeout_cycles=10000000
while (( $# )); do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --build-only|--run-only)
      [[ $mode = all ]] || die 'Please choose either --build-only or --run-only.'
      mode=${1#--}
      shift
      ;;
    --config|--jobs|--timeout-cycles)
      (( $# >= 2 )) && [[ -n $2 && $2 != --* ]] || die "Please add a value after $1."
      case $1 in
        --config) config=$2 ;;
        --jobs) jobs=$2 ;;
        --timeout-cycles) timeout_cycles=$2 ;;
      esac
      shift 2
      ;;
    --*) die "Unknown option: $1. Try --help." ;;
    -*) die "Unknown option: $1. Try --help." ;;
    *)
      [[ -z $workload ]] || die 'Please select just one workload directory.'
      workload=$(repo_path "$1")
      shift
      ;;
  esac
done

[[ -n $workload ]] || die 'Please choose a workload, e.g. artifacts/matmul_small. Try --help.'
[[ -d $workload ]] || die "Workload directory not found: $workload"
[[ $jobs =~ ^[1-9][0-9]*$ ]] || die 'Please use a positive whole number for --jobs.'
[[ $timeout_cycles =~ ^[1-9][0-9]*$ ]] || die 'Please use a positive whole number for --timeout-cycles.'
[[ $config =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die 'Please use a simple Chipyard class name for --config.'
workload=$(cd -- "$workload" && pwd -P)
sim_dir=$workload/simulation
program=$sim_dir/program.riscv
chipyard_dir=$(repo_path "${CHIPYARD_DIR:-third_party/chipyard}")
simulator=$chipyard_dir/sims/verilator/simulator-chipyard.harness-$config

# Check everything needed by this mode before starting expensive builds.
if [[ $mode != run-only ]]; then
  lowered_dir=$(repo_path "${BUDDY_LOWERED_DIR:-$workload/lowered}")
  llvm_dir=$(repo_path "${BUDDY_LLVM_DIR:-third_party/buddy-mlir/llvm/build}")
  riscv_dir=$(repo_path "${RISCV_TOOLCHAIN_DIR:-$chipyard_dir/.conda-env/riscv-tools}")
  [[ -f $lowered_dir/06_lowered.ll ]] || die 'LLVM IR is missing. Please compile this workload first.'
  [[ -f $sim_dir/workload_config.h ]] || die 'Please add simulation/workload_config.h using simulation/workload_config.h.in.'
  # Catch an unfilled template before starting Bazel or compiling C sources.
  header_text=$(< "$sim_dir/workload_config.h")
  [[ ! $header_text =~ @WORKLOAD_[A-Z0-9_]+@ ]] \
    || die 'Please fill in the @WORKLOAD_...@ values in simulation/workload_config.h.'
  [[ -x $llvm_dir/bin/llc ]] || die 'Buddy llc was not found. Please check BUDDY_LLVM_DIR.'
  [[ -x $riscv_dir/bin/riscv64-unknown-elf-gcc ]] || die 'RISC-V GCC was not found. Please check RISCV_TOOLCHAIN_DIR.'
  bazel=$(command -v bazel || command -v bazelisk) || die 'Bazel was not found. Please install Bazel or Bazelisk.'
else
  [[ -f $program ]] || die 'program.riscv is missing. Please run without --run-only first.'
  [[ -x $simulator ]] || die "The $config simulator is missing. Please run without --run-only first."
fi

if [[ $mode != build-only ]]; then
  [[ -f $chipyard_dir/env.sh ]] || die 'Chipyard env.sh is missing. Please finish Chipyard setup first.'
  [[ -f $chipyard_dir/sims/verilator/Makefile ]] || die 'Chipyard Makefile is missing. Please check CHIPYARD_DIR.'
  command -v make >/dev/null || die 'Make was not found. Please install it first.'
  # An activated Conda shell may provide CONDA_EXE without putting conda on PATH.
  if ! command -v conda >/dev/null; then
    if [[ -n ${CONDA_EXE:-} && -x $CONDA_EXE ]]; then
      export PATH="$(dirname -- "$CONDA_EXE"):$PATH"
    else
      die 'Conda was not found. Please activate your Chipyard environment first.'
    fi
  fi
fi
mkdir -p -- "$sim_dir" || die "Please choose a writable workload directory: $workload"

if [[ $mode != run-only ]]; then
  printf 'Building program.riscv...\n'
  # Pass every input explicitly so an older shell selection cannot leak in.
  # Clear the lowered override by supplying the resolved directory each time.
  if ! (
    cd -- "$repo_root" || exit 1
    "$bazel" build //simulation:program --jobs="$jobs" \
      --repo_env=BUDDY_WORKLOAD_DIR="$workload" \
      --repo_env=BUDDY_LOWERED_DIR="$lowered_dir" \
      --repo_env=BUDDY_LLVM_DIR="$llvm_dir" \
      --repo_env=RISCV_TOOLCHAIN_DIR="$riscv_dir"
  ) 2>&1 | tee "$sim_dir/program-build.log"; then
    die "Executable build failed. See: $sim_dir/program-build.log"
  fi
  # Removing the old copy also handles Bazel's read-only output permissions.
  cp --remove-destination -- "$repo_root/bazel-bin/simulation/program/program.riscv" "$program" \
    || die 'Could not copy program.riscv. Please check workload directory permissions.'
  if [[ $mode = build-only ]]; then
    printf 'Built: %s\n' "$program"
    exit 0
  fi
fi

# Keep Conda activation and Chipyard variables inside a subshell. Disable
# nounset while sourcing third-party setup, but still stop on setup failures.
(
  printf 'Loading Chipyard environment...\n'
  trap 'die "Could not load Chipyard. Please check: $chipyard_dir/env.sh"' ERR
  set +u
  # shellcheck source=/dev/null
  source "$chipyard_dir/env.sh"
  set -u
  trap - ERR
  command -v verilator >/dev/null || die 'Verilator was not found. Please check your Chipyard environment.'

  # Hardware snapshots are exported explicitly by our project-owned Scala
  # script. Stock Gemmini needs no source hook during simulator elaboration.
  unset GEMMINI_CONFIG_NAME GEMMINI_CONFIG_JSON_DIR

  if [[ $mode != run-only ]]; then
    printf 'Updating %s simulator...\n' "$config"
    if ! make -C "$chipyard_dir/sims/verilator" CONFIG="$config" -j"$jobs" \
      2>&1 | tee "$sim_dir/simulator-build.log"; then
      die "Simulator build failed. See: $sim_dir/simulator-build.log"
    fi
    [[ -x $simulator ]] || die "Simulator not found after building: $simulator"
  fi

  printf 'Running %s...\n' "$program"
  # The normal path just updated the simulator; run-only explicitly reuses it.
  # BREAK_SIM_PREREQ prevents Make from unexpectedly rebuilding RTL here.
  if ! make -C "$chipyard_dir/sims/verilator" CONFIG="$config" \
    run-binary-fast BREAK_SIM_PREREQ=1 BINARY="$program" \
    LOADMEM=1 TIMEOUT_CYCLES="$timeout_cycles" output_dir="$sim_dir/chipyard" \
    2>&1 | tee "$sim_dir/simulation.log"; then
    die "Simulation failed or timed out. See: $sim_dir/simulation.log"
  fi
)

# Match the exact record format emitted by main.c, allowing whitespace.
# Keep this in Bash so the runner does not need Python or a JSON utility.
# Require one complete record; a successful Make exit alone is not a pass.
result_pattern='^BZ_SIM_RESULT[[:space:]]+\{[[:space:]]*'
result_pattern+='"status"[[:space:]]*:[[:space:]]*"(passed|failed)"[[:space:]]*,[[:space:]]*'
result_pattern+='"cycles"[[:space:]]*:[[:space:]]*(0|[1-9][0-9]*)[[:space:]]*,[[:space:]]*'
result_pattern+='"mismatches"[[:space:]]*:[[:space:]]*(0|[1-9][0-9]*)[[:space:]]*\}[[:space:]]*$'
records=0
while IFS= read -r line || [[ -n $line ]]; do
  [[ $line = BZ_SIM_RESULT* ]] || continue
  [[ $line =~ $result_pattern ]] || die "The result record is incomplete. See: $sim_dir/simulation.log"
  status=${BASH_REMATCH[1]}
  cycles=${BASH_REMATCH[2]}
  mismatches=${BASH_REMATCH[3]}
  records=$((records + 1))
done < "$sim_dir/simulation.log"
[[ $records = 1 ]] || die "Expected one result record; found $records. See: $sim_dir/simulation.log"
[[ $status = passed && $mismatches = 0 ]] || die "Numerical check failed ($mismatches mismatches). See: $sim_dir/simulation.log"
printf 'PASS — cycles: %s\nLog: %s\n' "$cycles" "$sim_dir/simulation.log"
