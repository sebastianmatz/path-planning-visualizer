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


def test_rrt_star_displayed_tree_matches_actual_tree(qapp):
    # Regression: RRT* rewires (a node's parent can change), so the displayed tree
    # must be redrawn from the current parent structure each step. Appending edges
    # would leave rewired-away edges on screen (stale) and never show the new
    # rewired edges (missing) — here ~half the tree was wrong before the fix.
    win = MainWindow()
    try:
        occ = np.zeros((80, 80), dtype=bool)
        _set_map(win, occ)
        win.canvas.start = (5, 5)
        win.canvas.goal = (75, 75)
        _select_algo(win, "RRT*")
        ready = []
        win._build_planner_async(lambda: ready.append(True))
        assert _spin_until(qapp, lambda: bool(ready) and win.planner is not None)

        for _ in range(400):
            if win.planner.done:
                break
            win._do_one_step()

        p = win.planner
        true_edges = {
            (p.nodes[par], p.nodes[i]) for i, par in enumerate(p.parent) if par != -1
        }
        drawn = set(win.canvas.current_tree_edges)
        assert len(true_edges) > 50  # rewiring actually exercised at scale
        assert drawn == true_edges  # no stale edges, no missing edges
        assert win.canvas._appended_edges == []  # RRT* must not accumulate edges
    finally:
        win.close()


def _assert_search_tree_matches_came_from(qapp, algo):
    win = MainWindow()
    try:
        occ = np.zeros((60, 60), dtype=bool)
        occ[0:40, 30] = True
        _set_map(win, occ)
        win.canvas.start = (5, 5)
        win.canvas.goal = (55, 55)
        _select_algo(win, algo)
        ready = []
        win._build_planner_async(lambda: ready.append(True))
        assert _spin_until(qapp, lambda: bool(ready) and win.planner is not None)
        for _ in range(4000):
            if win.planner.done:
                break
            win._do_one_step()
        p = win.planner
        gs, half = p.grid_size, p.grid_size // 2
        true_edges = {
            ((par[0] * gs + half, par[1] * gs + half), (n[0] * gs + half, n[1] * gs + half))
            for n, par in p.came_from.items()
        }
        assert set(win.canvas.current_tree_edges) == true_edges
        assert len(true_edges) > 10
        assert win.canvas._appended_edges == []
    finally:
        win.close()


def test_search_tree_visualization_matches_came_from(qapp):
    # Regression: A*/Dijkstra relax nodes still in the open set, so the displayed
    # search tree must be redrawn from came_from each step (not one accumulated
    # relaxation edge per pop), otherwise it is incomplete and partly stale.
    _assert_search_tree_matches_came_from(qapp, "A*")
    _assert_search_tree_matches_came_from(qapp, "Dijkstra")


def test_sbl_displayed_tree_is_collision_free_and_current(qapp):
    # Regression: SBL is lazy and attempts cross-tree bridges (some in collision);
    # the displayed tree must be the authoritative (collision-free) milestone tree,
    # not accumulated attempt edges.
    win = MainWindow()
    try:
        occ = np.zeros((60, 60), dtype=bool)
        occ[0:40, 30] = True
        _set_map(win, occ)
        win.canvas.start = (5, 5)
        win.canvas.goal = (55, 55)
        _select_algo(win, "SBL")
        ready = []
        win._build_planner_async(lambda: ready.append(True))
        assert _spin_until(qapp, lambda: bool(ready) and win.planner is not None)
        for _ in range(5000):
            if win.planner.done:
                break
            win._do_one_step()
        p = win.planner
        true_edges = {
            (p.nodes[n.parent].point, n.point) for n in p.nodes if n.parent not in (None, -1)
        }
        drawn = set(win.canvas.current_tree_edges)
        assert drawn == true_edges
        for a, b in drawn:
            assert line_collision_free(a, b, occ)  # no through-wall attempt edges shown
    finally:
        win.close()


def test_prm_samples_drawn_as_node_dots_not_edges(qapp):
    # Regression: PRM milestones are shown as node dots, not tiny "marker edges"
    # (which weren't real edges and could clip walls).
    win = MainWindow()
    try:
        occ = np.zeros((50, 50), dtype=bool)
        _set_map(win, occ)
        win.canvas.start = (2, 2)
        win.canvas.goal = (47, 47)
        _select_algo(win, "PRM")
        ready = []
        win._build_planner_async(lambda: ready.append(True))
        assert _spin_until(qapp, lambda: bool(ready) and win.planner is not None)
        for _ in range(3000):
            win._do_one_step()
            if len(win.canvas.node_markers) > 5:
                break
        assert len(win.canvas.node_markers) > 5  # milestones recorded as dots
    finally:
        win.close()


def test_prm_drag_requery_reuses_roadmap(qapp):
    # Roadmap planners: after completion, dragging start/goal re-solves on the SAME
    # learned roadmap (no re-sampling), and an invalid drop reverts.
    win = MainWindow()
    try:
        occ = np.zeros((70, 70), dtype=bool)
        occ[0:48, 35] = True
        _set_map(win, occ)
        win.canvas.start = (5, 5)
        win.canvas.goal = (65, 65)
        _select_algo(win, "sPRM")
        ready = []
        win._build_planner_async(lambda: ready.append(True))
        assert _spin_until(qapp, lambda: bool(ready) and win.planner is not None)
        for _ in range(40000):
            if win.planner.done:
                break
            win._do_one_step()
        assert win.planner.found_path
        assert win.canvas.drag_markers_enabled  # drag-to-re-query enabled after completion

        roadmap = win.planner.roadmap_size
        new_goal = (65, 5)
        win.canvas.goal = new_goal  # the drag moved the marker
        win._on_marker_dragged("goal", new_goal)

        assert win.planner.roadmap_size == roadmap  # roadmap reused, not rebuilt
        path = win.planner.extract_path()
        assert path and path[-1] == new_goal
        for a, b in zip(path, path[1:], strict=False):
            assert line_collision_free(a, b, occ)

        # An invalid drop (onto the wall) reverts to the committed endpoint.
        win.canvas.goal = (35, 10)
        win._on_marker_dragged("goal", (35, 10))
        assert win.canvas.goal == win.planner.goal == new_goal
    finally:
        win.close()


def test_prm_requery_recovers_after_failed_query(qapp):
    # Regression: a failed re-query (dragging to a spot with no path) must not
    # permanently block re-querying — dragging back to a solvable spot must recompute.
    win = MainWindow()
    try:
        occ = np.zeros((60, 60), dtype=bool)
        occ[:, 30] = True  # full wall: two disconnected regions
        _set_map(win, occ)
        win.canvas.start = (10, 30)
        win.canvas.goal = (50, 30)  # opposite region -> build finds no path
        _select_algo(win, "sPRM")
        ready = []
        win._build_planner_async(lambda: ready.append(True))
        assert _spin_until(qapp, lambda: bool(ready) and win.planner is not None)
        for _ in range(40000):
            if win.planner.done:
                break
            win._do_one_step()

        # No path at build, but a roadmap exists -> drag-to-re-query is still available.
        assert not win.planner.found_path
        assert win.canvas.drag_markers_enabled
        assert win._requery_active()

        # Drag the goal into the start's region -> a path is found.
        win.canvas.goal = (15, 30)
        win._on_marker_dragged("goal", (15, 30))
        assert win.planner.found_path and len(win.canvas._final_path) >= 2

        # Drag to the other region -> no path; eligibility must persist.
        win.canvas.goal = (50, 30)
        win._on_marker_dragged("goal", (50, 30))
        assert not win.planner.found_path
        assert win._requery_active()

        # Drag back to a solvable spot -> the path recomputes (the bug was it didn't).
        win.canvas.goal = (20, 40)
        win._on_marker_dragged("goal", (20, 40))
        assert win.planner.found_path and len(win.canvas._final_path) >= 2
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
