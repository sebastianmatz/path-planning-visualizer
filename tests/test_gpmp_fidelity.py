"""Paper-fidelity tests for GPMP (Mukadam, Yan & Boots 2016, ICRA).

- the GP-prior precision K^-1 = B^T Q^-1 B is symmetric and block-tridiagonal (Eq 13);
- the SDF is signed (negative inside obstacles);
- one step applies the covariant gradient update xi <- xi - (1/eta) K grad U (Eq 24).
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.planners._trajectory import sdf_query
from path_planning_visualizer.planners.gpmp import GPMPPlanner


def _block_map() -> np.ndarray:
    occ = np.zeros((60, 60), dtype=bool)
    occ[20:40, 20:40] = True
    return occ


def test_gp_prior_precision_is_symmetric_block_tridiagonal():
    p = GPMPPlanner(np.zeros((60, 60), dtype=bool), (5, 30), (55, 30), num_points=6)
    K = p.Kinv_prior
    assert np.allclose(K, K.T)
    # Interior states 1 and 3 (blocks 0 and 2) are two apart -> no direct coupling.
    assert np.allclose(K[0:4, 8:12], 0.0)
    assert not np.allclose(K[0:4, 4:8], 0.0)  # consecutive states ARE coupled


def test_uses_signed_distance_field():
    p = GPMPPlanner(_block_map(), (5, 5), (55, 55), num_points=20)
    d_inside, _ = sdf_query(p.dist_field, p.grad_x, p.grad_y, 30, 30)
    assert d_inside < 0.0


def test_step_is_covariant_gradient_update():
    p = GPMPPlanner(_block_map(), (8, 30), (52, 30), num_points=12, eta=5.0)
    states0 = p.states.copy()

    g = p._gradient(p.states)
    expected = (-(1.0 / p.eta) * np.linalg.solve(p.Kinv_prior, g)).reshape(-1, 4)
    pos_norm = float(np.linalg.norm(expected[:, :2]))
    if pos_norm > p.max_step:
        expected *= p.max_step / pos_norm

    p.step_once()
    applied = p.states[1:-1] - states0[1:-1]
    assert np.allclose(applied, expected, atol=1e-6)
