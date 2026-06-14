"""Paper-fidelity tests for SBL (Sánchez & Latombe 2001).

- TEST-SEGMENT refines a segment dyadically (kappa 0->k) and marks it safe exactly
  when 2^(-kappa)*lambda < epsilon;
- TEST-PATH orders segments by 2^(-kappa)*lambda (safe segments sink to -1);
- the random shortcut optimizer keeps endpoints and never lengthens the path.
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.geometry import compute_path_length
from path_planning_visualizer.planners.sbl import SBLPlanner


def _open(**kw) -> SBLPlanner:
    occ = np.zeros((200, 200), dtype=bool)
    return SBLPlanner(occ, (5, 5), (190, 190), **kw)


def test_segment_dyadic_refinement_marks_safe_at_resolution():
    p = _open(lazy_resolution=4)
    nid = p._add_node(0, (5, 60), tree_id=0)  # L_inf length 55 from (5,5)
    key = p._edge_key(0, nid)

    assert p.edge_states[key].kappa == 0
    assert abs(p._segment_score(key) - 55.0) < 1e-6  # 2^0 * 55

    for _ in range(10):
        assert p._test_segment_once(key) is None  # open space -> never collides
        if p.edge_states[key].safe:
            break

    # smallest k with 55/2^k < 4 is k = 4  (55/16 = 3.4375)
    assert p.edge_states[key].safe is True
    assert p.edge_states[key].kappa == 4
    assert p._segment_score(key) == -1.0


def test_test_path_orders_by_collision_likelihood():
    p = _open(lazy_resolution=4)
    long_id = p._add_node(0, (5, 60), tree_id=0)   # length 55
    short_id = p._add_node(0, (5, 20), tree_id=0)  # length 15
    k_long = p._edge_key(0, long_id)
    k_short = p._edge_key(0, short_id)

    # longer (more likely to hide a collision) is tested first
    assert p._segment_score(k_long) > p._segment_score(k_short)

    for _ in range(10):
        p._test_segment_once(k_long)
        if p.edge_states[k_long].safe:
            break
    # once safe it sinks below any unresolved segment
    assert p._segment_score(k_long) == -1.0
    assert p._segment_score(k_short) > p._segment_score(k_long)


def test_shortcut_optimizer_keeps_endpoints_and_never_lengthens():
    p = _open()
    path = [(5, 5), (5, 80), (80, 80), (80, 5)]
    opt = p._optimize_solution_path(path)

    assert opt[0] == path[0]
    assert opt[-1] == path[-1]
    assert compute_path_length(opt) <= compute_path_length(path) + 1e-6
