"""Paper-fidelity tests for RRT-Connect (Kuffner & LaValle 2000).

Checks the parts made paper-exact:
- EXTEND lands exactly on the target when within step_size (Reached, Fig. 2);
- EXTEND advances ~step_size toward a far target (Advanced);
- CONNECT reaches a far target exactly via a chain of EXTENDs (Fig. 5);
- RANDOM_CONFIG samples over the whole space C (occupied configs are returned).
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.geometry import dist
from path_planning_visualizer.planners.rrt_connect import RRTConnectPlanner


def _planner(occ=None, **kw):
    if occ is None:
        occ = np.zeros((40, 40), dtype=bool)
    return RRTConnectPlanner(occ, (5, 5), (35, 35), **kw)


def test_extend_lands_exactly_on_near_target():
    p = _planner(step_size=10)
    target = (12, 5)  # distance 7 from start (5,5) < step_size
    status, idx, edge, rejected = p._extend(p.nodes_a, p.parent_a, p._index_a, target)
    assert status == 'reached'
    assert p.nodes_a[idx] == target  # landed exactly on the target


def test_extend_advances_toward_far_target():
    p = _planner(step_size=10)
    target = (35, 5)  # distance 30 from start > step_size
    status, idx, edge, rejected = p._extend(p.nodes_a, p.parent_a, p._index_a, target)
    assert status == 'advanced'
    assert abs(dist((5, 5), p.nodes_a[idx]) - 10) <= 1.5  # stepped ~step_size


def test_connect_reaches_far_target_exactly():
    p = _planner(step_size=8)
    target = (33, 5)  # far, collision-free corridor from start
    status, idx, edges, rejected = p._connect(p.nodes_a, p.parent_a, p._index_a, target)
    assert status == 'reached'
    assert p.nodes_a[idx] == target   # CONNECT lands exactly on the target
    assert len(edges) >= 2            # required several EXTEND steps


def test_random_config_samples_over_whole_space():
    occ = np.zeros((40, 40), dtype=bool)
    occ[10:30, 10:30] = True  # 25% of the map is occupied
    p = _planner(occ=occ)
    samples = [p._sample() for _ in range(2000)]
    assert all(0 <= x < 40 and 0 <= y < 40 for x, y in samples)
    # Uniform over C (no rejection): some samples land on occupied cells.
    assert any(occ[y, x] for x, y in samples)
