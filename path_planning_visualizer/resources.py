from __future__ import annotations

import sys
from pathlib import Path


def _assets_dir() -> Path:
    """Locate the bundled ``assets`` directory in both source and frozen runs.

    Under PyInstaller the package source is not on disk; data files are unpacked
    under ``sys._MEIPASS`` (mirroring their bundled layout), so resolve from there
    when frozen and from this module's directory otherwise.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "path_planning_visualizer" / "assets"
    return Path(__file__).resolve().parent / "assets"


ASSETS_DIR = _assets_dir()


def asset_path(name: str) -> str:
    """Return the absolute path to a bundled asset (e.g. an example maze)."""
    return str(ASSETS_DIR / name)
