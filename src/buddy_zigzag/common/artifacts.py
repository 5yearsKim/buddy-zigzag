"""Small artifact and environment metadata utilities."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..settings import Settings


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def versions(settings: "Settings") -> dict:
    data = {"python": platform.python_version()}
    for name in ("buddy-zigzag", "torch", "PyYAML", "pydantic", "zigzag-dse"):
        try:
            data[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            data[name] = None
    for name, path in {
        "buddy": settings.buddy_mlir_build_dir.parent if settings.buddy_mlir_build_dir else None,
        "zigzag": settings.zigzag_source_dir,
    }.items():
        try:
            if path is None:
                raise OSError("Source path is not configured")
            result = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
            data[f"{name}_commit"] = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            data[f"{name}_commit"] = None
    return data


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
