"""Paper-fidelity tests for ITOMP (Park, Pan & Manocha 2012).

- the smoothness metric is acceleration-based (Eq. 6): a straight line has ~zero AΘ+c;
- the SDF is signed (negative inside obstacles, Eq. 8);
- the static obstacle cost (Eq. 8) is zero with clearance >= eps, positive inside;
- the receding execution horizon advances over iterations (Figs. 2-3).
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.planners._trajectory import sdf_query
from path_planning_visualizer.planners.itomp import ITOMPPlanner


def _block_map() -> np.ndarray:
    occ = np.zeros((60, 60), dtype=bool)
    occ[20:40, 20:40] = True
    return occ


def test_smoothness_metric_is_acceleration():
    p = ITOMPPlanner(np.zeros((60, 60), dtype=bool), (5, 30), (55, 30), num_points=10)
    theta = p.trajectory[1:-1]
    # A straight line (the free-space init) has zero acceleration.
    assert np.allclose(p.A @ theta + p.c, 0.0, atol=1e-6)


def test_uses_signed_distance_field():
    p = ITOMPPlanner(_block_map(), (5, 5), (55, 55), num_points=20)
    d_inside, _ = sdf_query(p.dist_field, p.grad_x, p.grad_y, 30, 30)
    assert d_inside < 0.0


def test_static_obstacle_cost_form():
    p = ITOMPPlanner(_block_map(), (5, 5), (55, 55), num_points=5)
    p.trajectory = np.array([[5.0, 5.0], [15.0, 5.0], [30.0, 5.0], [45.0, 5.0], [55.0, 5.0]])
    assert p._static_obstacle_cost() == 0.0  # clearance >= eps everywhere
    p.trajectory = np.array([[5.0, 30.0], [20.0, 30.0], [30.0, 30.0], [40.0, 30.0], [55.0, 30.0]])
    assert p._static_obstacle_cost() > 0.0   # passes through the block


def test_execution_horizon_advances():
    p = ITOMPPlanner(_block_map(), (8, 30), (52, 30), num_points=20,
                     replan_interval=5, max_iters=2000)
    start_exec = p.exec_idx
    for _ in range(60):
        if p.done:
            break
        p.step_once()
    assert p.exec_idx > start_exec
