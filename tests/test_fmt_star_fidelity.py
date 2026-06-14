"""Paper-fidelity tests for FMT* (Janson et al. 2015).

- obstacle-free runs return a near-optimal (~straight-line) path, i.e. the
  shortest path over the disk graph (Theorem 3.2);
- the planner terminates when the goal is *popped* as the lowest-cost node in
  V_open (Alg. 1 line 9.2), not the instant the goal is connected.
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.geometry import compute_path_length, dist
from path_planning_visualizer.planners.fmt_star import FMTStarPlanner


def _run(p, n: int = 400000):
    steps = 0
    while not p.done and steps < n:
        p.step_once()
        steps += 1
    return p


def test_obstacle_free_path_is_near_optimal():
    occ = np.zeros((60, 60), dtype=bool)
    start, goal = (5, 30), (55, 30)
    p = _run(FMTStarPlanner(occ, start, goal, num_samples=800, seed=1))
    assert p.found_path
    path = p.extract_path()
    cost = compute_path_length(path)
    euclid = dist(start, goal)
    # Disk-graph shortest path: close to the straight line for a dense sample set.
    assert cost <= 1.3 * euclid


def test_terminates_on_goal_pop_not_on_goal_connect():
    occ = np.zeros((60, 60), dtype=bool)
    occ[20:40, 30] = True  # a small wall, still locally solvable
    p = FMTStarPlanner(occ, (5, 30), (55, 30), num_samples=800, seed=1)

    goal_in_open_before_done = False
    steps = 0
    while not p.done and steps < 400000:
        p.step_once()
        steps += 1
        if not p.done and p.goal_idx in p.V_open:
            goal_in_open_before_done = True

    assert p.found_path
    # The goal was connected (in V_open) and the planner kept running until it was
    # popped as the minimum -- the paper's termination, not "stop on connect".
    assert goal_in_open_before_done
