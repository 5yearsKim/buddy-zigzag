"""Check runner failures and mode selection without compiling Chipyard RTL."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "simulation/run.sh"
PASSED = 'BZ_SIM_RESULT {"status":"passed","cycles":12345,"mismatches":0}'


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + body + "\n")
    path.chmod(0o755)


@pytest.fixture
def runner(tmp_path):
    # Use an isolated checkout so stub builds cannot touch real Bazel outputs.
    root = tmp_path / "checkout"
    script = root / "simulation/run.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(SOURCE, script)
    workload = root / "artifacts/example"
    (workload / "lowered").mkdir(parents=True)
    (workload / "lowered/06_lowered.ll").touch()
    (workload / "simulation").mkdir()
    (workload / "simulation/workload_config.h").write_text("/* user header */\n")

    llvm = root / "third_party/buddy-mlir/llvm/build"
    chipyard = root / "third_party/chipyard"
    executable(llvm / "bin/llc", "exit 0")
    executable(chipyard / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc", "exit 0")
    verilator = chipyard / "sims/verilator"
    verilator.mkdir(parents=True)
    (verilator / "Makefile").touch()
    simulator = verilator / "simulator-chipyard.harness-FPGemminiRocketConfig"
    executable(simulator, "exit 0")
    # This unset-variable read checks that Chipyard setup is sourced without -u.
    (chipyard / "env.sh").write_text('echo "${OPTIONAL_CHIPYARD_VARIABLE}" >/dev/null\n')

    tools = tmp_path / "tools"
    executable(tools / "conda", "exit 0")
    executable(tools / "verilator", "exit 0")
    executable(
        tools / "bazel",
        """printf 'bazel %s\\n' "$*" >> "$CALLS"
if [[ ${FAIL_STAGE:-} = bazel ]]; then exit 3; fi
mkdir -p bazel-bin/simulation/program
printf 'fresh ELF' > bazel-bin/simulation/program/program.riscv
chmod 444 bazel-bin/simulation/program/program.riscv
""",
    )
    executable(
        tools / "make",
        """printf 'make %s\\n' "$*" >> "$CALLS"
printf 'snapshot %s\\n' "${GEMMINI_CONFIG_JSON_DIR:-unset}" >> "$CALLS"
if [[ " $* " = *' run-binary-fast '* ]]; then
  printf '%s\\n' "$RECORD"
  if [[ ${FAIL_STAGE:-} = run ]]; then exit 4; fi
else
  if [[ ${FAIL_STAGE:-} = simulator ]]; then exit 5; fi
fi
""",
    )
    env = os.environ.copy()
    # Do not let a developer's toolchain or workload selections affect tests.
    for key in (
        "BUDDY_LLVM_DIR",
        "RISCV_TOOLCHAIN_DIR",
        "BUDDY_LOWERED_DIR",
        "CHIPYARD_DIR",
        "BASH_ENV",
        "SHELLOPTS",
        "FAIL_STAGE",
        "OPTIONAL_CHIPYARD_VARIABLE",
    ):
        env.pop(key, None)
    env.update(
        PATH=str(tools) + os.pathsep + env["PATH"],
        CALLS=str(tmp_path / "calls"),
        RECORD=PASSED,
        # Stale values must not select a different workload or snapshot output.
        BUDDY_WORKLOAD_DIR="/old/workload",
        GEMMINI_CONFIG_JSON_DIR="/old/snapshot",
    )

    def run(*args, **overrides):
        return subprocess.run(
            [str(script), *args],
            cwd=tmp_path,  # Relative paths should still resolve from the repo.
            env=env | overrides,
            text=True,
            capture_output=True,
            timeout=10,
        )

    return run, root, workload, simulator, Path(env["CALLS"])


def test_default_builds_runs_and_preserves_manual_header(runner):
    run, root, workload, _, calls = runner
    result = run("artifacts/example", "--jobs", "3", "--timeout-cycles", "321")
    assert result.returncode == 0, result.stderr
    assert "PASS — cycles: 12345" in result.stdout
    assert (workload / "simulation/program.riscv").read_text() == "fresh ELF"
    assert (workload / "simulation/workload_config.h").read_text() == "/* user header */\n"
    commands = calls.read_text()
    assert f"--repo_env=BUDDY_WORKLOAD_DIR={workload}" in commands
    assert "--jobs=3" in commands
    assert "-j3" in commands
    assert "TIMEOUT_CYCLES=321" in commands
    assert "BREAK_SIM_PREREQ=1" in commands
    assert commands.count("make ") == 2
    assert "/old/" not in commands
    assert "snapshot unset" in commands
    for name in ("program-build.log", "simulator-build.log", "simulation.log"):
        assert (workload / "simulation" / name).is_file()


def test_build_only_replaces_read_only_elf_without_chipyard(runner):
    run, root, workload, simulator, calls = runner
    (simulator.parents[2] / "env.sh").unlink()
    program = workload / "simulation/program.riscv"
    program.write_text("old ELF")
    program.chmod(0o444)
    result = run(str(workload), "--build-only")
    assert result.returncode == 0, result.stderr
    assert program.read_text() == "fresh ELF"
    assert "make " not in calls.read_text()
    assert "PASS" not in result.stdout


def test_run_only_needs_no_compilers_or_ir(runner):
    run, root, workload, _, calls = runner
    (workload / "simulation/program.riscv").write_text("existing ELF")
    (workload / "lowered/06_lowered.ll").unlink()
    (workload / "simulation/workload_config.h").unlink()
    shutil.rmtree(root / "third_party/buddy-mlir")
    result = run("artifacts/example", "--run-only")
    assert result.returncode == 0, result.stderr
    commands = calls.read_text()
    assert "bazel " not in commands
    assert commands.count("make ") == 1
    assert "run-binary-fast BREAK_SIM_PREREQ=1" in commands


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "Please choose a workload"),
        (["artifacts/missing"], "Workload directory not found"),
        (["artifacts/example", "--build-only", "--run-only"], "Please choose either"),
        (["artifacts/example", "--jobs"], "Please add a value"),
        (["artifacts/example", "--jobs", "0"], "positive whole number"),
        (["artifacts/example", "--timeout-cycles", "-1"], "positive whole number"),
        (["artifacts/example", "--unknown"], "Unknown option"),
        (["artifacts/example", "--config", "Bad/Config"], "simple Chipyard class name"),
    ],
)
def test_bad_arguments_do_not_start_builds(runner, args, message):
    run, _, _, _, calls = runner
    result = run(*args)
    assert result.returncode != 0
    assert message in result.stderr
    assert "unbound variable" not in result.stderr
    assert not calls.exists()


def test_missing_header_gives_help_before_building(runner):
    run, _, workload, _, calls = runner
    (workload / "simulation/workload_config.h").unlink()
    result = run("artifacts/example")
    assert result.returncode != 0
    assert "Please add simulation/workload_config.h" in result.stderr
    assert not calls.exists()


def test_unfilled_template_gives_help_before_building(runner):
    run, _, workload, _, calls = runner
    template = SOURCE.with_name("workload_config.h.in").read_text()
    header = workload / "simulation/workload_config.h"
    header.write_text(template)
    result = run("artifacts/example")
    assert result.returncode != 0
    assert "Please fill in the @WORKLOAD_...@ values" in result.stderr
    assert header.read_text() == template
    assert not calls.exists()


@pytest.mark.parametrize("missing", ["program", "simulator"])
def test_run_only_requires_both_existing_outputs(runner, missing):
    run, _, workload, simulator, calls = runner
    if missing == "simulator":
        (workload / "simulation/program.riscv").touch()
        simulator.unlink()
    result = run("artifacts/example", "--run-only")
    assert result.returncode != 0
    assert "Please run without --run-only first" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize(
    ("stage", "message", "log"),
    [
        ("bazel", "Executable build failed", "program-build.log"),
        ("simulator", "Simulator build failed", "simulator-build.log"),
        ("run", "Simulation failed or timed out", "simulation.log"),
    ],
)
def test_command_failures_cannot_report_success(runner, stage, message, log):
    run, _, workload, _, _ = runner
    result = run("artifacts/example", FAIL_STAGE=stage)
    assert result.returncode != 0
    assert message in result.stderr
    assert str(workload / "simulation" / log) in result.stderr
    assert "PASS" not in result.stdout


@pytest.mark.parametrize(
    "record",
    [
        "",
        "BZ_SIM_RESULT {}",
        'BZ_SIM_RESULT {"status":"failed","cycles":1,"mismatches":0}',
        'BZ_SIM_RESULT {"status":"passed","cycles":1,"mismatches":2}',
        'BZ_SIM_RESULT {"status":"passed","cycles":-1,"mismatches":0}',
        'BZ_SIM_RESULT {"status":"passed","cycles":01,"mismatches":0}',
        PASSED + "\n" + PASSED,
    ],
)
def test_bad_results_do_not_reuse_old_pass(runner, record):
    run, _, workload, _, _ = runner
    (workload / "simulation/simulation.log").write_text(PASSED + "\n")
    result = run("artifacts/example", RECORD=record)
    assert result.returncode != 0
    assert "simulation.log" in result.stderr
    assert "PASS" not in result.stdout


def test_chipyard_setup_failure_stops_before_make(runner):
    run, root, _, _, calls = runner
    (root / "third_party/chipyard/env.sh").write_text("false\nprintf 'setup continued\\n'\n")
    result = run("artifacts/example")
    assert result.returncode != 0
    assert "Could not load Chipyard" in result.stderr
    assert "setup continued" not in result.stdout
    assert "make " not in calls.read_text()
