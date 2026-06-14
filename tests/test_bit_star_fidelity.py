"""Paper-fidelity tests for BIT* (Gammell et al. 2015).

Guards the verification-pass changes:
- seeded determinism (locks the queue/tie-break refactor);
- Alg. 2 line 4: only vertices added during the current batch enqueue
  vertex-vertex (rewiring) edges.

(Anytime monotonicity / near-optimality are covered by test_optimality.py.)
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.planners.bit_star import BITStarPlanner


def _wall_map() -> np.ndarray:
    occ = np.zeros((60, 60), dtype=bool)
    occ[0:45, 30] = True
    occ[0:45, 31] = True
    return occ


def _run(p, n: int = 6000):
    steps = 0
    while not p.done and steps < n:
        p.step_once()
        steps += 1
    return p


def test_bit_star_deterministic_under_seed():
    occ = _wall_map()
    a = _run(BITStarPlanner(occ, (8, 8), (52, 8), batch_size=200, max_iters=5000, step_size=18.0, seed=1))
    b = _run(BITStarPlanner(occ, (8, 8), (52, 8), batch_size=200, max_iters=5000, step_size=18.0, seed=1))
    assert a.found_path == b.found_path
    assert a.extract_path() == b.extract_path()


def test_old_vertices_do_not_enqueue_vertex_vertex_edges():
    """Alg. 2 line 4: an old vertex (index < v_old_count) must add no kind-1 edges."""
    occ = _wall_map()
    p = BITStarPlanner(occ, (8, 8), (52, 8), batch_size=150, max_iters=20000, step_size=18.0, seed=1)
    for _ in range(20000):
        p.step_once()
        if p.batch_count >= 2 and len(p.V) > 5:
            break
    assert p.batch_count >= 2 and len(p.V) > 5

    old_idx = 0  # the start vertex is always in V_old
    assert old_idx < p.v_old_count
    p.vertex_expanded_batch.pop(old_idx, None)
    p._expand_vertex(old_idx)

    # Q_E tuples are (key, g_T, counter, parent_idx, target_kind, target_idx).
    kind1_from_old = [e for e in p.Q_E if e[3] == old_idx and e[4] == 1]
    assert kind1_from_old == []
