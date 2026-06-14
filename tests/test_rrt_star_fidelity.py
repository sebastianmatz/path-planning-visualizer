"""Paper-fidelity tests for RRT* (Karaman & Frazzoli 2011, Algorithm 6).

- the connection radius is the shrinking RGG radius capped at eta = step_size,
  and is adaptive only when search_radius = 0;
- ChooseParent + Rewire + cost propagation keep the paper's cost recursion
  Cost(v) = Cost(Parent(v)) + ||v - Parent(v)|| exact at every node;
- the parent pointers form an acyclic tree.
(Anytime monotonicity / near-optimality are covered by test_optimality.py.)
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.geometry import dist
from path_planning_visualizer.planners.rrt_star import RRTStarPlanner


def _wall_map() -> np.ndarray:
    occ = np.zeros((60, 60), dtype=bool)
    occ[0:45, 30] = True
    occ[0:45, 31] = True
    return occ


def _run(p, n: int = 400000):
    steps = 0
    while not p.done and steps < n:
        p.step_once()
        steps += 1
    return p


def test_connection_radius_capped_and_adaptive():
    occ = np.zeros((100, 100), dtype=bool)
    auto = RRTStarPlanner(occ, (5, 5), (95, 95), step_size=18, search_radius=0)
    assert auto.adaptive_radius
    assert 0.0 < auto._connection_radius() <= 18.0  # capped at eta = step_size

    fixed = RRTStarPlanner(occ, (5, 5), (95, 95), step_size=18, search_radius=30)
    assert not fixed.adaptive_radius


def test_cost_recursion_holds_at_every_node():
    p = _run(RRTStarPlanner(_wall_map(), (8, 8), (52, 8), step_size=10,
                            goal_tolerance=14, search_radius=0, max_iters=2500, seed=1))
    for i in range(len(p.nodes)):
        parent = p.parent[i]
        if parent == -1:
            assert p.cost[i] == 0.0  # root
            continue
        expected = p.cost[parent] + dist(p.nodes[i], p.nodes[parent])
        assert abs(p.cost[i] - expected) < 1e-6


def test_tree_is_acyclic():
    p = _run(RRTStarPlanner(_wall_map(), (8, 8), (52, 8), step_size=10,
                            goal_tolerance=14, search_radius=0, max_iters=2500, seed=2))
    for start in range(len(p.nodes)):
        seen = set()
        i = start
        while i != -1:
            assert i not in seen, "parent pointers contain a cycle"
            seen.add(i)
            i = p.parent[i]
