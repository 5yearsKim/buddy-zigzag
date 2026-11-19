"""Environment configuration for local native compiler builds."""

import os
import subprocess
import sys
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .common.errors import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    project_dir: Path = Field(default_factory=Path.cwd, exclude=True)
    buddy_mlir_build_dir: Path | None = None
    llvm_mlir_build_dir: Path | None = None
    buddy_zigzag_python: Path | None = None
    zigzag_source_dir: Path | None = None
    pythonpath: str = ""
    artifacts_dir: Path = Path("artifacts")

    @model_validator(mode="after")
    def resolve_paths(self):
        self.project_dir = self.project_dir.resolve()
        for name in (
            "buddy_mlir_build_dir",
            "llvm_mlir_build_dir",
            "buddy_zigzag_python",
            "zigzag_source_dir",
            "artifacts_dir",
        ):
            path = getattr(self, name)
            if path is not None:
                setattr(self, name, (self.project_dir / path).resolve())
        return self

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        path = (env_file or Path(".env")).resolve()
        return cls(_env_file=path, project_dir=path.parent)

    def configure_runtime(self) -> None:
        """Expose configured compiler bindings before importing native modules."""
        if self.buddy_zigzag_python is not None:
            try:
                result = subprocess.run(
                    [
                        str(self.buddy_zigzag_python),
                        "-c",
                        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ConfigurationError(
                    f"Cannot inspect Buddy's reference interpreter: {self.buddy_zigzag_python}"
                ) from exc
            current = f"{sys.version_info.major}.{sys.version_info.minor}"
            if result.stdout.strip() != current:
                raise ConfigurationError(
                    f"Buddy requires Python {result.stdout.strip()}; current Python is {current}. "
                    "Select a matching uv interpreter."
                )
        paths = [self.project_dir / part for part in self.pythonpath.split(os.pathsep) if part]
        if self.buddy_mlir_build_dir is not None:
            paths.append(self.buddy_mlir_build_dir / "python_packages")
        if self.zigzag_source_dir is not None:
            paths.append(self.zigzag_source_dir)
        for path in reversed(list(dict.fromkeys(str(path.resolve()) for path in paths))):
            if path not in sys.path:
                sys.path.insert(0, path)
        for name in ("buddy_mlir_build_dir", "llvm_mlir_build_dir"):
            value = getattr(self, name)
            if value is not None:
                os.environ[name.upper()] = str(value)
