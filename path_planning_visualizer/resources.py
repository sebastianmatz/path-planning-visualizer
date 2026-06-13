from __future__ import annotations

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def asset_path(name: str) -> str:
    """Return the absolute path to a bundled asset (e.g. an example maze)."""
    return str(ASSETS_DIR / name)
