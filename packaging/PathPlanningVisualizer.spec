# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for a single-file Windows build of Path Planning Visualizer.

Build (from the repo root):
    pyinstaller --noconfirm packaging/PathPlanningVisualizer.spec
Output:
    dist/PathPlanningVisualizer.exe
"""
import os

from PyInstaller.utils.hooks import copy_metadata

# Resolve paths from the spec location (SPECPATH = packaging/) so the build does
# not depend on the working directory it is invoked from.
ROOT = os.path.dirname(SPECPATH)

# Bundle the example mazes (mirrored under the same package-relative path so
# resources.asset_path resolves them via sys._MEIPASS at runtime) and the package
# metadata so importlib.metadata can report the real __version__ in the frozen app.
datas = [(os.path.join(ROOT, "path_planning_visualizer", "assets"), "path_planning_visualizer/assets")]
datas += copy_metadata("path-planning-visualizer")


a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PathPlanningVisualizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,  # GUI app: no console window (errors still surface via the dialog)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
