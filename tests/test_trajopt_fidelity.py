"""Paper-fidelity tests for TrajOpt (Schulman et al. 2013).

- the smoothness objective is the sum of squared displacements (Eq. 5);
- the collision penalty uses a true signed distance field (negative inside);
- the trust-region step is rejected when actual/predicted <= accept_ratio.
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.planners._trajectory import sdf_query
from path_planning_visualizer.planners.trajopt import TrajOptPlanner


def _block_map() -> np.ndarray:
    occ = np.zeros((60, 60), dtype=bool)
    occ[20:40, 20:40] = True
    return occ


def test_smoothness_is_sum_of_squared_displacements():
    p = TrajOptPlanner(np.zeros((60, 60), dtype=bool), (5, 30), (55, 30), num_points=6)
    theta = np.array([[10.0, 30.0], [20.0, 35.0], [30.0, 25.0], [40.0, 30.0]])
    f, _ = p._smoothness(theta)
    full = np.vstack([[5.0, 30.0], theta, [55.0, 30.0]])
    disp = np.diff(full, axis=0)
    assert abs(f - float(np.sum(disp * disp))) < 1e-6


def test_collision_penalty_uses_signed_distance():
    p = TrajOptPlanner(_block_map(), (5, 5), (55, 55), d_safe=10.0)
    d_inside, _ = sdf_query(p.dist_field, p.grad_x, p.grad_y, 30, 30)
    d_free, _ = sdf_query(p.dist_field, p.grad_x, p.grad_y, 5, 5)
    assert d_inside < 0.0       # signed: negative deep inside the obstacle
    assert d_free > 10.0        # clearly outside, beyond d_safe


def test_step_rejected_when_ratio_below_accept_ratio():
    # An impossible acceptance threshold means no trust-region step is ever taken.
    p = TrajOptPlanner(_block_map(), (8, 30), (52, 30), num_points=40,
                       max_iters=50, accept_ratio=10.0)
    theta0 = p.trajectory[1:-1].copy()
    p.step_once()
    assert np.allclose(p.trajectory[1:-1], theta0)
