"""Unit tests for the headless benchmark harness."""

from __future__ import annotations

import numpy as np
import pytest

from path_planning_visualizer import benchmark as bm


def test_nearest_free_moves_off_obstacle():
    occ = np.zeros((10, 10), dtype=bool)
    occ[5, 5] = True
    p = bm._nearest_free(occ, (5, 5))
    assert not occ[p[1], p[0]]  # returns a free cell


def test_nearest_free_keeps_free_cell():
    occ = np.zeros((10, 10), dtype=bool)
    assert bm._nearest_free(occ, (3, 4)) == (3, 4)


def test_resolve_names_all_and_unknown():
    avail = ["A*", "RRT", "RRT*"]
    assert bm._resolve("all", avail, "planner") == avail
    assert bm._resolve("A*, RRT", avail, "planner") == ["A*", "RRT"]
    with pytest.raises(SystemExit):
        bm._resolve("Nope", avail, "planner")


def test_builtin_maps_have_free_endpoints():
    for name, factory in bm.BUILTIN_MAPS.items():
        occ, start, goal = factory()
        assert not occ[start[1], start[0]], f"{name}: start on obstacle"
        assert not occ[goal[1], goal[0]], f"{name}: goal on obstacle"


def test_benchmark_runs_and_reports_success():
    results = bm.benchmark(["A*", "RRT"], ["open"], seeds=1, max_steps=100_000)
    assert all(results["open"]["A*"].success)        # A* is complete on an open map
    res_rrt = results["open"]["RRT"]
    assert len(res_rrt.success) == 1 and any(res_rrt.success)
    # a successful run records a finite length
    assert results["open"]["A*"].length and results["open"]["A*"].length[0] > 0
