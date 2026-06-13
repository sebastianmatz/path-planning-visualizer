"""Evidence tests for the academic claims the planners make.

These go beyond soundness/completeness and assert the *quality* properties the
papers promise:

* anytime planners (RRT* with the shrinking RGG radius, uncapped BIT*) improve
  their incumbent solution monotonically and converge near the optimum;
* asymptotically-optimal batch planners (FMT*, PRM) return near-optimal paths;
* the shrinking-radius RRT* is deterministic under a fixed seed;
* the trajectory optimizers (STOMP, TrajOpt, ITOMP, GPMP) converge to a sound
  path on a problem a local optimizer can actually solve.

The reference cost is A* on the induced grid.  Because the continuous planners
can take Euclidean shortcuts the grid cannot, they typically come in *below* the
grid cost, so a generous upper bound of ``1.3x`` is comfortably non-flaky.
"""

from __future__ import annotations

import numpy as np
import pytest

import path_planning_visualizer as ppv

WALL_START = (8, 8)
WALL_GOAL = (52, 8)
OPT_FACTOR = 1.3  # near-optimal upper bound vs the grid reference


def wall_map() -> np.ndarray:
    occ = np.zeros((60, 60), dtype=bool)
    occ[0:45, 30] = True
    occ[0:45, 31] = True
    return occ


def easy_map():
    """Open map with one small obstacle on the straight line (local-solvable)."""
    occ = np.zeros((60, 60), dtype=bool)
    occ[25:35, 28:32] = True
    return occ, (8, 30), (52, 30)


def _run(planner, max_steps=400_000):
    steps = 0
    while not planner.done and steps < max_steps:
        planner.step_once()
        steps += 1
    return planner


def _segments_free(path, occ) -> bool:
    return all(ppv.line_collision_free(path[i], path[i + 1], occ) for i in range(len(path) - 1))


@pytest.fixture()
def reference_cost() -> float:
    occ = wall_map()
    a = _run(ppv.AStarPlanner(occ, WALL_START, WALL_GOAL, grid_size=5, allow_diagonal=True))
    assert a.found_path
    return ppv.compute_path_length(a.extract_path())


def test_rrt_star_anytime_improves(reference_cost):
    """Adaptive RRT* incumbent and returned path are monotone and near-optimal."""
    occ = wall_map()
    p = ppv.RRTStarPlanner(occ, WALL_START, WALL_GOAL, step_size=10, goal_tolerance=14,
                           search_radius=0, max_iters=4000, seed=1)
    incumbents = []
    extracted = []
    while not p.done:
        p.step_once()
        if np.isfinite(p.best_goal_cost):
            incumbents.append(p.best_goal_cost)
            extracted.append(ppv.compute_path_length(p.extract_path()))

    assert p.found_path
    assert len(incumbents) > 1
    assert all(incumbents[i + 1] <= incumbents[i] + 1e-6 for i in range(len(incumbents) - 1)), \
        "RRT* incumbent cost increased between steps"
    # Regression guard: the actually returned path must also improve monotonically
    # (it used to rise after a stale-incumbent goal reconnection).
    assert all(extracted[i + 1] <= extracted[i] + 1e-6 for i in range(len(extracted) - 1)), \
        "RRT* extract_path() cost increased between steps"
    assert _segments_free(p.extract_path(), occ)
    assert p.best_goal_cost <= OPT_FACTOR * reference_cost


def test_bit_star_anytime_improves(reference_cost):
    """Uncapped BIT* incumbent is monotone non-increasing and near-optimal."""
    occ = wall_map()
    p = ppv.BITStarPlanner(occ, WALL_START, WALL_GOAL, batch_size=200, max_iters=5000,
                           step_size=18.0, seed=1)
    incumbents = []
    while not p.done:
        p.step_once()
        if np.isfinite(p.best_cost):
            incumbents.append(p.best_cost)

    assert p.found_path
    assert len(incumbents) > 1
    assert all(incumbents[i + 1] <= incumbents[i] + 1e-6 for i in range(len(incumbents) - 1)), \
        "BIT* incumbent cost increased between steps"
    assert _segments_free(p.extract_path(), occ)
    assert p.best_cost <= OPT_FACTOR * reference_cost


def test_bit_star_cap_is_optional():
    """The visualization cap must still produce a sound path when enabled."""
    occ = wall_map()
    p = _run(ppv.BITStarPlanner(occ, WALL_START, WALL_GOAL, batch_size=200, max_iters=5000,
                                step_size=18.0, cap_edges_to_step=True, seed=1))
    assert p.found_path
    assert _segments_free(p.extract_path(), occ)


@pytest.mark.parametrize("name,kwargs", [
    ("FMT*", dict(num_samples=800, seed=1)),
    ("PRM", dict(num_samples=600, k_neighbors=15, max_edge_dist=40, seed=1)),
])
def test_batch_planners_near_optimal(name, kwargs, reference_cost):
    occ = wall_map()
    p = _run(ppv.AVAILABLE_PLANNERS[name](occ, WALL_START, WALL_GOAL, **kwargs))
    assert p.found_path
    cost = ppv.compute_path_length(p.extract_path())
    assert _segments_free(p.extract_path(), occ)
    assert cost <= OPT_FACTOR * reference_cost


def test_rrt_star_adaptive_is_deterministic():
    occ = wall_map()
    a = _run(ppv.RRTStarPlanner(occ, WALL_START, WALL_GOAL, step_size=10, goal_tolerance=14,
                                search_radius=0, max_iters=3000, seed=7))
    b = _run(ppv.RRTStarPlanner(occ, WALL_START, WALL_GOAL, step_size=10, goal_tolerance=14,
                                search_radius=0, max_iters=3000, seed=7))
    assert a.found_path == b.found_path
    assert a.extract_path() == b.extract_path()


@pytest.mark.parametrize("name,kwargs", [
    ("STOMP", dict(num_points=40, max_iters=300, seed=1)),
    ("TrajOpt", dict(num_points=40, max_iters=300)),
    ("ITOMP", dict(num_points=30, max_iters=600, replan_interval=10, seed=1)),
    ("GPMP", dict(num_points=25, max_iters=200, sigma=8.0)),
])
def test_trajectory_optimizers_converge(name, kwargs):
    """Each reworked optimizer reaches a sound goal path on a local-solvable map."""
    occ, start, goal = easy_map()
    p = _run(ppv.AVAILABLE_PLANNERS[name](occ, start, goal, **kwargs))
    assert p.found_path, f"{name} failed to converge on a local-solvable map"
    path = p.extract_path()
    assert _segments_free(path, occ), f"{name} returned a path that clips an obstacle"
    assert ppv.dist(path[0], start) <= 2.0
    assert ppv.dist(path[-1], goal) <= 2.0


@pytest.mark.parametrize("name,kwargs", [
    ("STOMP", dict(num_points=30, max_iters=120, seed=3)),
    ("ITOMP", dict(num_points=25, max_iters=200, replan_interval=10, seed=3)),
    ("GPMP", dict(num_points=20, max_iters=120, sigma=8.0)),
])
def test_seeded_optimizers_are_deterministic(name, kwargs):
    occ, start, goal = easy_map()
    a = _run(ppv.AVAILABLE_PLANNERS[name](occ, start, goal, **kwargs))
    b = _run(ppv.AVAILABLE_PLANNERS[name](occ, start, goal, **kwargs))
    assert a.extract_path() == b.extract_path()
