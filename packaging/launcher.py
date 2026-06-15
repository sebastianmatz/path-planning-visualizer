"""Frozen-app entry point for PyInstaller.

Uses an absolute import (not a package-relative one) because PyInstaller runs this
file as the top-level ``__main__`` script, where relative imports would fail.
"""
from __future__ import annotations

from path_planning_visualizer.app import main

if __name__ == "__main__":
    main()
