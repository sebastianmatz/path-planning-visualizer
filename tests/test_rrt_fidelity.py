"""Paper-fidelity tests for RRT (LaValle 1998, GENERATE_RRT).

- SELECT_INPUT / NEW_STATE implement the holonomic f=u, ||u||<=1 model;
- every tree edge spans at most one dt step;
- the goal-bias extension + goal-region stop return a start-rooted path;
- the planner halts at the K vertex budget.
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.geometry import (
    dist,
    integrate_holonomic_state,
    line_collision_free,
    select_holonomic_input,
)
from path_planning_visualizer.planners.rrt import RRTPlanner


def _run(p, n: int = 400000):
    steps = 0
    while not p.done and steps < n:
        p.step_once()
        steps += 1
    return p


def _path_is_collision_free(path, occ) -> bool:
    return all(
        line_collision_free(path[i], path[i + 1], occ, samples=64)
        for i in range(len(path) - 1)
    )


def test_holonomic_select_input_and_new_state():
    # Far target: unit-magnitude control, one dt step.
    u = select_holonomic_input((0.0, 0.0), (100.0, 0.0), 10.0)
    assert np.isclose(np.hypot(*u), 1.0)
    assert np.allclose(integrate_holonomic_state((0.0, 0.0), u, 10.0), (10.0, 0.0))

    # Within dt: control magnitude < 1, lands exactly on the target.
    u2 = select_holonomic_input((0.0, 0.0), (5.0, 0.0), 10.0)
    assert np.hypot(*u2) < 1.0
    assert np.allclose(integrate_holonomic_state((0.0, 0.0), u2, 10.0), (5.0, 0.0))


def test_rrt_edges_respect_delta_t():
    occ = np.zeros((100, 100), dtype=bool)
    p = _run(RRTPlanner(occ, (50, 50), (95, 95), delta_t=10.0, goal_bias=0.0,
                        goal_region_radius=3.0, max_vertices=150, seed=1))
    for i in range(1, len(p.nodes)):
        parent = p.parent[i]
        assert parent >= 0
        assert dist(p.nodes[i], p.nodes[parent]) <= 10.0 + 2.0  # dt + pixel rounding


def test_rrt_goal_bias_reaches_goal_region():
    occ = np.zeros((100, 100), dtype=bool)
    p = _run(RRTPlanner(occ, (10, 50), (90, 50), delta_t=10.0, goal_bias=1.0,
                        goal_region_radius=12.0, max_vertices=2000, seed=1))
    assert p.found_path
    path = p.extract_path()
    assert path[0] == (10, 50)
    assert dist(path[-1], (90, 50)) <= 12.0


def test_rrt_goal_requires_line_of_sight():
    # A vertex inside the goal region but on the far side of a wall must not count
    # as reaching the goal (otherwise the reported path appears to cross the wall).
    occ = np.zeros((100, 100), dtype=bool)
    occ[:, 50] = True  # sealed vertical wall at column x=50
    p = RRTPlanner(occ, (10, 50), (55, 50), delta_t=8.0, goal_bias=0.0,
                   goal_region_radius=20.0, collision_samples=80, max_vertices=10, seed=0)
    assert dist((49, 50), (55, 50)) <= 20.0   # within the goal region...
    assert not p._goal_reached((49.0, 50.0))  # ...but blocked: no line of sight
    assert p._goal_reached((52.0, 50.0))      # same side as the goal: accepted


def test_rrt_does_not_report_path_through_wall():
    # Start and goal separated by a fully sealed wall: the goal is unreachable, so
    # RRT must not declare success (the old goal-region rule accepted a vertex
    # within radius on the wrong side of the wall).
    occ = np.zeros((100, 100), dtype=bool)
    occ[:, 50] = True
    p = _run(RRTPlanner(occ, (10, 50), (55, 50), delta_t=8.0, goal_bias=0.3,
                        goal_region_radius=20.0, collision_samples=80,
                        max_vertices=1500, seed=1))
    assert not p.found_path
    assert p.extract_path() == []


def test_rrt_found_path_is_collision_free_and_reaches_goal():
    # Vertical wall with a gap: any valid route must pass through the gap, and the
    # returned path must be collision-free and actually end at the goal.
    occ = np.zeros((100, 100), dtype=bool)
    occ[:, 50] = True
    occ[45:56, 50] = False  # gap at y in [45, 55]
    p = _run(RRTPlanner(occ, (10, 50), (90, 50), delta_t=8.0, goal_bias=0.2,
                        goal_region_radius=10.0, collision_samples=80,
                        max_vertices=8000, seed=3))
    assert p.found_path
    path = p.extract_path()
    assert path[0] == (10, 50)
    assert path[-1] == (90, 50)  # the path now actually reaches the goal
    assert _path_is_collision_free(path, occ)


def test_rrt_respects_vertex_budget_K():
    occ = np.zeros((100, 100), dtype=bool)
    K = 50
    p = _run(RRTPlanner(occ, (50, 50), (95, 5), delta_t=8.0, goal_bias=0.0,
                        goal_region_radius=2.0, max_vertices=K, seed=2))
    assert p.done
    assert len(p.nodes) <= K
