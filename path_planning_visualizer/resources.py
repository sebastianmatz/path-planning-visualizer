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


def list_maps() -> list[str]:
    """Return the bundled example-map filenames (``maze*.png``), naturally sorted.

    Works both from source and from a frozen build (the assets are unpacked under
    ``sys._MEIPASS``). ``maze.png`` sorts before ``maze 2.png`` etc.
    """
    if not ASSETS_DIR.is_dir():
        return []
    names = [p.name for p in ASSETS_DIR.glob("maze*.png")]

    def _key(name: str) -> tuple[int, str]:
        stem = name[:-4]  # drop ".png"
        digits = "".join(c for c in stem if c.isdigit())
        return (int(digits) if digits else 0, name.lower())

    return sorted(names, key=_key)
