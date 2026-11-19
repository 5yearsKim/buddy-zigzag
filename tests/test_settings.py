import sys
from types import SimpleNamespace

import pytest

from buddy_zigzag.common.errors import ConfigurationError
from buddy_zigzag.settings import Settings


def test_env_paths_resolve_from_file_and_environment_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("BUDDY_MLIR_BUILD_DIR", raising=False)
    monkeypatch.delenv("BUDDY_ZIGZAG_PYTHON", raising=False)
    env = tmp_path / ".env"
    env.write_text("BUDDY_MLIR_BUILD_DIR=../buddy/build\nARTIFACTS_DIR=from_file\n")
    monkeypatch.setenv("ARTIFACTS_DIR", "from_environment")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    settings = Settings.load(env)
    assert settings.buddy_mlir_build_dir == (tmp_path.parent / "buddy/build").resolve()
    assert settings.artifacts_dir == tmp_path / "from_environment"


def test_runtime_exposes_configured_paths_without_shell_setup(tmp_path, monkeypatch):
    for name in ("BUDDY_MLIR_BUILD_DIR", "LLVM_MLIR_BUILD_DIR", "BUDDY_ZIGZAG_PYTHON", "PYTHONPATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    settings = Settings(
        _env_file=None,
        project_dir=tmp_path,
        buddy_mlir_build_dir="buddy/build",
        llvm_mlir_build_dir="llvm/build",
        pythonpath="extra",
        zigzag_source_dir="zigzag",
    )
    settings.configure_runtime()
    assert str(tmp_path / "buddy/build/python_packages") in sys.path
    assert str(tmp_path / "zigzag") in sys.path
    assert str(tmp_path / "extra") in sys.path


def test_mismatched_native_python_version_has_a_clear_error(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, buddy_zigzag_python=tmp_path / "python")
    monkeypatch.setattr("buddy_zigzag.settings.subprocess.run", lambda *a, **kw: SimpleNamespace(stdout="0.0\n"))
    with pytest.raises(ConfigurationError, match="Buddy requires Python 0.0"):
        settings.configure_runtime()
