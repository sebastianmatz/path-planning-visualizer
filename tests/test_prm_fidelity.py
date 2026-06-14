"""Paper-fidelity tests for PRM (Kavraki et al. 1996), sPRM variant.

- N_c = candidate neighbors within max_edge_dist, sorted by increasing distance,
  capped at k_neighbors (the paper's maxdist + maxneighbors scheme);
- every roadmap edge is collision-free (the straight-line local planner Delta);
- the roadmap keeps cycles (sPRM), i.e. more edges than a spanning forest.
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.geometry import line_collision_free
from path_planning_visualizer.planners.prm import ClassicPRMPlanner, PRMPlanner


def _run(p, n: int = 400000):
    steps = 0
    while not p.done and steps < n:
        p.step_once()
        steps += 1
    return p


def test_candidate_neighbors_within_maxdist_sorted_capped():
    occ = np.zeros((100, 100), dtype=bool)
    p = PRMPlanner(occ, (2, 2), (98, 98), k_neighbors=3, max_edge_dist=20)
    p.nodes = [(50, 50), (55, 50), (60, 50), (50, 60), (50, 90), (90, 90)]

    cands = p._candidate_neighbors((50, 50), list(range(1, len(p.nodes))))
    dists = [d for d, _ in cands]
    # within maxdist: idx1 (5), idx2 (10), idx3 (10); idx4 (40) and idx5 (~57) excluded.
    assert len(cands) == 3
    assert all(d <= 20.0 for d in dists)
    assert dists == sorted(dists)

    # maxneighbors cap.
    p.k_neighbors = 2
    assert len(p._candidate_neighbors((50, 50), list(range(1, len(p.nodes))))) == 2


def test_roadmap_edges_are_collision_free():
    occ = np.zeros((60, 60), dtype=bool)
    occ[20:40, 20:40] = True
    p = _run(PRMPlanner(occ, (5, 5), (55, 55), num_samples=300, k_neighbors=10,
                        max_edge_dist=25, seed=1))
    for a, neighbors in p.edges.items():
        for b, _ in neighbors:
            assert line_collision_free(p.nodes[a], p.nodes[b], occ)


def test_roadmap_keeps_cycles_sprm():
    occ = np.zeros((60, 60), dtype=bool)  # open space -> dense connectivity
    p = _run(PRMPlanner(occ, (2, 2), (58, 58), num_samples=200, k_neighbors=10,
                        max_edge_dist=30, seed=1))
    total_edges = sum(len(e) for e in p.edges.values()) // 2
    # A spanning forest over the roadmap would have < roadmap_size edges; sPRM
    # keeps cycles, so there are strictly more.
    assert total_edges > p.roadmap_size


def test_classic_prm_is_a_cycle_free_forest():
    """The original Kavraki 1996 'PRM' skips same-component edges -> a forest."""
    occ = np.zeros((60, 60), dtype=bool)  # open space: sPRM here is dense with cycles
    p = _run(ClassicPRMPlanner(occ, (2, 2), (58, 58), num_samples=200, k_neighbors=10,
                               max_edge_dist=30, seed=1))
    assert p.found_path
    total_edges = sum(len(e) for e in p.edges.values()) // 2
    # A forest over all nodes has edges = nodes - components < nodes (acyclic).
    assert total_edges < len(p.nodes)
