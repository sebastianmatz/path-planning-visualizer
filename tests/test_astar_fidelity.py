"""Paper-fidelity tests for A* (Hart, Nilsson & Raphael 1968).

- the evaluation function is f = g + h (Eq 2);
- the heuristic is consistent (h(n) <= c(n,m) + h(m), Eq 5) and admissible
  (h(start) <= the optimal path cost, Theorem 1).
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.planners.astar import AStarPlanner


def _wall_map() -> np.ndarray:
    occ = np.zeros((60, 60), dtype=bool)
    occ[0:45, 30] = True
    occ[0:45, 31] = True
    return occ


def _run(p, n: int = 100000):
    steps = 0
    while not p.done and steps < n:
        p.step_once()
        steps += 1
    return p


def test_evaluation_is_g_plus_h():
    p = _run(AStarPlanner(_wall_map(), (8, 8), (52, 8), grid_size=5, allow_diagonal=True))
    assert p.found_path
    for node, f in p.f_score.items():
        assert abs(f - (p.g_score[node] + p._heuristic(node))) < 1e-9


def test_heuristic_is_consistent():
    p = AStarPlanner(np.zeros((100, 100), dtype=bool), (2, 2), (98, 98),
                     grid_size=5, allow_diagonal=True)
    for n in [(2, 2), (5, 7), (10, 3), (14, 14), (0, 18), (9, 1)]:
        for m, cost in p._get_neighbors(n):
            assert p._heuristic(n) <= cost + p._heuristic(m) + 1e-9


def test_heuristic_is_admissible():
    p = _run(AStarPlanner(_wall_map(), (8, 8), (52, 8), grid_size=5, allow_diagonal=True))
    assert p.found_path
    # h(start) must lower-bound the optimal cost found to the goal.
    assert p._heuristic(p.start_grid) <= p.g_score[p.goal_grid] + 1e-9
