"""Paper-fidelity tests for BiTRRT (Devaurs, Siméon & Cortés 2013).

- an accepted uphill transition divides T by 2^((c_j-c_i)/(0.1*costRange)) (Alg 2 line 4);
- a rejected uphill transition multiplies T by 2^(T_rate) (Alg 2 line 6);
- attemptLink only fires within 10*delta and refinement is thresholded at delta.
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.planners.bitrrt import BiTRRTPlanner


def _planner(**kw) -> BiTRRTPlanner:
    occ = np.zeros((60, 60), dtype=bool)
    return BiTRRTPlanner(occ, (5, 5), (55, 55), **kw)


def test_accepted_uphill_decreases_T_base2():
    p = _planner(range=18.0, temp_change_factor=0.1)
    p.temp = 100.0
    p.best_cost = 0.0
    p.worst_cost = 10.0  # costRange = 10
    mc = 5.0             # exp(-5/100) = 0.951 > 0.5 -> accept
    assert p._transition_test(mc) is True
    expected = 100.0 / (2.0 ** (mc / (0.1 * 10.0)))  # /2^5
    assert abs(p.temp - expected) < 1e-6


def test_rejected_uphill_increases_T_base2():
    p = _planner(range=18.0, temp_change_factor=0.1)
    p.temp = 1.0
    mc = 5.0             # exp(-5/1) = 0.0067 <= 0.5 -> reject
    assert p._transition_test(mc) is False
    expected = 1.0 * (2.0 ** p.temp_change_factor_param)
    assert abs(p.temp - expected) < 1e-6


def test_connection_range_and_refinement_threshold_use_delta():
    p = _planner(range=20.0)
    assert abs(p.connection_range - 10.0 * p.max_distance) < 1e-9   # attemptLink within 10*delta
    assert abs(p.frontier_threshold - p.max_distance) < 1e-9        # refinement threshold = delta
