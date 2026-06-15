"""Headless integration tests that drive the real MainWindow.

These exercise the GUI orchestration end to end (off-thread planner build ->
stepping -> found path), the map editor, and the save/load round-trip, rather
than testing planners in isolation. Headless via QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

import path_planning_visualizer.gui.main_window as mw
from path_planning_visualizer.geometry import line_collision_free, make_distance_field
from path_planning_visualizer.gui.main_window import MainWindow
from path_planning_visualizer.mapping import image_to_occupancy, occupancy_to_image


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _spin_until(app, predicate, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)
    return predicate()


def _select_algo(win: MainWindow, name: str) -> None:
    combo = win.algo_combo
    for i in range(combo.count()):
        if combo.itemData(i, Qt.ItemDataRole.UserRole) == name:
            combo.setCurrentIndex(i)
            assert win._get_selected_algo_name() == name
            return
    raise AssertionError(f"algorithm {name!r} not found in the combo box")


def _set_map(win: MainWindow, occ: np.ndarray) -> None:
    win.occ = occ
    win.clearance_field = make_distance_field(occ)
    win.canvas.set_image(win._occ_to_pixmap(occ))  # resets start/goal/overlay


def test_end_to_end_plan_finds_collision_free_path(qapp):
    win = MainWindow()
    try:
        # Small map with a partial vertical wall, so a path must route around it.
        occ = np.zeros((60, 60), dtype=bool)
        occ[20:40, 30] = True
        _set_map(win, occ)
        win.canvas.start = (5, 30)
        win.canvas.goal = (55, 30)
        _select_algo(win, "A*")

        # Off-thread build via the single GUI chokepoint, then spin until ready.
        ready = []
        win._build_planner_async(lambda: ready.append(True))
        assert _spin_until(qapp, lambda: bool(ready) and win.planner is not None)
        assert win.running_algo_name == "A*"

        for _ in range(100000):
            if win.planner.done:
                break
            win._do_one_step()
        assert win.planner.done
        assert win.planner.found_path

        path = win.planner.extract_path()
        assert len(path) >= 2
        assert path[0] == (5, 30)
        assert path[-1] == (55, 30)
        for a, b in zip(path, path[1:], strict=False):
            assert line_collision_free(a, b, occ)

        # The GUI's own metrics path should produce a positive length.
        metrics = win._get_path_metrics(path)
        assert metrics.length_px > 0
    finally:
        win.close()


def test_step_button_stays_enabled_after_async_build(qapp):
    # Regression: the first Step triggers an off-thread build, which entered the
    # "preparing" state and disabled Step/Run; leaving that state must re-enable
    # them, otherwise the Step button is dead after a single click.
    win = MainWindow()
    try:
        occ = np.zeros((60, 60), dtype=bool)
        _set_map(win, occ)
        _select_algo(win, "RRT")
        win.canvas.start = (5, 5)
        win.canvas.goal = (55, 55)
        win._on_point_picked("goal", (55, 55))  # enables the playback buttons
        assert win.btn_step.isEnabled()

        # First click: drives the async build path.
        win.btn_step.click()
        assert _spin_until(
            qapp,
            lambda: win.planner is not None
            and not (win._builder is not None and win._builder.isRunning()),
        )
        assert win.btn_step.isEnabled(), "Step button must stay clickable after the build"
        assert win.btn_run.isEnabled()
        first_iter = win.planner.iteration

        # Second click must actually step again (button was live).
        win.btn_step.click()
        qapp.processEvents()
        assert win.planner.iteration > first_iter
        assert win.btn_step.isEnabled()
    finally:
        win.close()


def test_new_map_paint_and_endpoint_validation(qapp):
    win = MainWindow()
    try:
        win.new_map()
        assert win.occ is not None
        assert not win.occ.any()  # blank map: all free

        # Place a start, then paint an obstacle over it and confirm it is dropped.
        win.canvas.start = (50, 50)
        win.canvas.goal = (60, 60)
        win._on_paint((50, 50), erase=False)
        assert win.occ[50, 50]  # brush drew an obstacle here
        assert win.canvas.start is None  # buried marker was invalidated
        assert win.canvas.goal == (60, 60)  # untouched marker survives

        # Erasing clears obstacle pixels again.
        win._on_paint((50, 50), erase=True)
        assert not win.occ[50, 50]
    finally:
        win.close()


def test_save_map_round_trips(qapp, monkeypatch, tmp_path):
    win = MainWindow()
    try:
        occ = np.zeros((40, 40), dtype=bool)
        occ[10:20, 10:20] = True
        win.occ = occ

        out = tmp_path / "saved_map.png"
        monkeypatch.setattr(
            mw.QFileDialog, "getSaveFileName",
            lambda *a, **k: (str(out), "PNG image (*.png)"),
        )
        win.save_map()
        assert out.exists()

        # Reload through the same convention the GUI uses and confirm a round-trip.
        on_disk = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(image_to_occupancy(on_disk), occ)
        assert np.array_equal(on_disk, occupancy_to_image(occ))
    finally:
        win.close()
