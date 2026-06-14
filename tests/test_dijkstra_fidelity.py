"""Paper-fidelity tests for Dijkstra's algorithm (Dijkstra 1959, Problem 2).

- nodes are finalized (closed) in non-decreasing distance from the start
  (the paper's "added to A in order of increasing distance from P");
- the reconstructed path's cost equals the finalized goal distance dist[Q].
"""

from __future__ import annotations

import math

import numpy as np

from path_planning_visualizer.planners.dijkstra import DijkstraPlanner


def _wall_map() -> np.ndarray:
    occ = np.zeros((60, 60), dtype=bool)
    occ[0:45, 30] = True
    occ[0:45, 31] = True
    return occ


def test_nodes_are_closed_in_increasing_distance():
    p = DijkstraPlanner(_wall_map(), (8, 8), (52, 8), grid_size=5, allow_diagonal=True)
    closed_order = []
    prev = set()
    steps = 0
    while not p.done and steps < 100000:
        p.step_once()
        steps += 1
        newly = p.closed_set - prev
        for node in newly:
            closed_order.append(p.dist[node])
        prev = set(p.closed_set)

    assert p.found_path
    assert len(closed_order) > 1
    # Step 2 always finalizes the minimum-distance frontier node.
    assert all(closed_order[i] <= closed_order[i + 1] + 1e-9
               for i in range(len(closed_order) - 1))


def test_path_cost_equals_finalized_goal_distance():
    p = DijkstraPlanner(_wall_map(), (8, 8), (52, 8), grid_size=5, allow_diagonal=True)
    while not p.done:
        p.step_once()
    assert p.found_path

    gs = p.grid_size
    total = 0.0
    for (ax, ay), (bx, by) in zip(p.path_grid, p.path_grid[1:]):
        total += gs * math.hypot(ax - bx, ay - by)
    assert abs(total - p.dist[p.goal_grid]) < 1e-6
