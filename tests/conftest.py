"""Configure native compiler bindings through the same settings as the CLI."""

from pathlib import Path

from buddy_zigzag.settings import Settings

Settings.load(Path(__file__).resolve().parents[1] / ".env").configure_runtime()
