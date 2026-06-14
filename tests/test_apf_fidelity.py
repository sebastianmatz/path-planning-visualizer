"""Paper-fidelity tests for APF (Khatib 1986).

Checks the defining properties of the artificial potential field:
- attractive force is the parabolic-well linear force F = -k(x - x_goal);
- the FIRAS repulsive force is zero beyond the influence limit and points away
  from obstacles within it;
- with no obstacles the descent is a straight line to the goal;
- pure APF terminates (stalls at a local minimum) instead of spinning forever.

Planners are built directly from kwargs, so the suite runs headless.
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.planners.apf import APFPlanner


def _run(planner, max_steps: int = 20000):
    steps = 0
    while not planner.done and steps < max_steps:
        planner.step_once()
        steps += 1
    return steps


def test_attractive_force_is_linear_in_goal_distance():
    occ = np.zeros((40, 40), dtype=bool)
    p = APFPlanner(occ, (5, 5), (35, 35), goal_gain=2.0)

    p.pos = np.array([10.0, 10.0])
    f_far = p._attractive_force()
    assert np.allclose(f_far, 2.0 * np.array([25.0, 25.0]))

    p.pos = np.array([20.0, 20.0])
    f_near = p._attractive_force()
    assert np.allclose(f_near, 2.0 * np.array([15.0, 15.0]))

    # Linear in distance: halving the error halves the force magnitude.
    assert np.linalg.norm(f_far) > np.linalg.norm(f_near)
    assert np.isclose(np.linalg.norm(f_far) / np.linalg.norm(f_near), 25.0 / 15.0)

    # Vanishes at the goal.
    p.pos = np.array([35.0, 35.0])
    assert np.allclose(p._attractive_force(), 0.0)


def test_repulsion_only_within_influence_and_points_away():
    occ = np.zeros((40, 40), dtype=bool)
    occ[:, 20] = True  # vertical wall
    p = APFPlanner(occ, (5, 10), (35, 10), obstacle_gain=100.0, obstacle_dist=10)

    # Far from the wall (distance 15 > rho_0 = 10): no repulsion.
    p.pos = np.array([5.0, 10.0])
    assert np.allclose(p._repulsive_force(), 0.0)

    # Close to the wall (distance 4 < rho_0): repulsion points away (-x).
    p.pos = np.array([16.0, 10.0])
    f_rep = p._repulsive_force()
    assert np.linalg.norm(f_rep) > 0.0
    assert f_rep[0] < 0.0  # pushed away from the wall at x = 20


def test_free_space_descent_is_straight():
    occ = np.zeros((60, 60), dtype=bool)
    p = APFPlanner(occ, (5, 30), (55, 30), goal_gain=1.0, step_size=5.0)
    _run(p)
    assert p.found_path
    # With no obstacles the parabolic attractor drives a straight line.
    ys = [y for _, y in p.extract_path()]
    assert max(abs(y - 30) for y in ys) <= 2


def test_pure_apf_terminates_in_a_trap():
    # Vertical wall with a gap only at the very bottom: a classic local-minimum
    # trap for a straight goal pull. Pure APF must stop, not spin forever.
    occ = np.zeros((60, 60), dtype=bool)
    occ[0:50, 30] = True
    occ[0:50, 31] = True
    p = APFPlanner(occ, (8, 8), (52, 8), max_iters=3000, enable_escape=False)
    steps = _run(p, max_steps=5000)
    assert p.done
    assert steps < 5000  # terminated on its own, no infinite spin
