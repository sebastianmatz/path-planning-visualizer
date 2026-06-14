"""Paper-fidelity tests for STOMP (Kalakrishnan et al. 2011).

Checks the parts made paper-exact:
- the shared signed distance field (negative inside obstacles);
- the per-timestep rollout probabilities of Eq. 11;
- the obstacle cost max(eps - d, 0) * ||x_dot|| on the signed field (Eq. 13).

(Sound + deterministic convergence is covered by test_optimality.py.)
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.planners._trajectory import signed_distance_field
from path_planning_visualizer.planners.stomp import STOMPPlanner


def test_signed_distance_field_has_correct_sign():
    occ = np.zeros((40, 40), dtype=bool)
    occ[15:25, 15:25] = True
    sdf = signed_distance_field(occ)

    assert sdf[20, 20] < 0.0          # deep inside the block
    assert sdf[5, 5] > 0.0            # far in free space
    assert sdf[14, 20] > 0.0          # free pixel just outside the block
    # Penetration depth grows toward the block center.
    assert sdf[20, 20] < sdf[16, 20] <= 0.0


def test_rollout_probabilities_match_eq11():
    occ = np.zeros((30, 30), dtype=bool)
    p = STOMPPlanner(occ, (2, 2), (27, 27), num_points=10, h=10.0)

    # Column 0: distinct costs; column 1: all tied -> uniform.
    S = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    P = p._probabilities(S)

    # Each timestep's probabilities sum to one.
    assert np.allclose(P.sum(axis=0), 1.0)

    # Column 0 matches the paper's exponent exp(-h (S - min)/(max - min)).
    expected = np.exp(-10.0 * np.array([0.0, 1.0, 2.0]) / 2.0)
    expected = expected / expected.sum()
    assert np.allclose(P[:, 0], expected)

    # Tied costs -> uniform weighting.
    assert np.allclose(P[:, 1], 1.0 / 3.0)


def test_obstacle_cost_uses_signed_distance_and_velocity():
    occ = np.zeros((60, 60), dtype=bool)
    occ[20:40, 20:40] = True
    p = STOMPPlanner(occ, (5, 5), (55, 55), epsilon=10.0)

    # Far from the block (clearance >= eps): zero obstacle cost.
    free = np.array([[5.0, 5.0], [10.0, 5.0], [15.0, 5.0], [20.0, 5.0]])
    assert float(np.sum(p._state_cost_along(free))) == 0.0

    # Inside the block: positive cost (signed distance is negative there).
    inside = np.array([[28.0, 30.0], [30.0, 30.0], [32.0, 30.0]])
    assert float(np.sum(p._state_cost_along(inside))) > 0.0

    # Velocity term: same clearance, larger spacing -> larger cost (Eq. 13 ||x_dot||).
    slow = np.array([[18.0, 30.0], [18.0, 31.0], [18.0, 32.0]])
    fast = np.array([[18.0, 25.0], [18.0, 30.0], [18.0, 35.0]])
    assert float(np.sum(p._state_cost_along(fast))) > float(np.sum(p._state_cost_along(slow)))
