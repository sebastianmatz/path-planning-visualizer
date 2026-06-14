"""Paper-fidelity tests for PSO (Kennedy & Eberhart 1995, sec. 3.6).

- the defaults are the exact 1995 form (w=1.0 momentum, c1=c2=2.0, Vmax clamp);
- with safeguards off, one velocity update equals
  v <- w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x), clamped to +/- Vmax;
- the off-by-default safeguards toggle changes the swarm dynamics.
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.planners.pso import PSOPlanner


def _planner(**kw) -> PSOPlanner:
    occ = np.zeros((40, 40), dtype=bool)
    params = dict(num_particles=1, num_points=5, max_iters=50, seed=1)
    params.update(kw)
    return PSOPlanner(occ, (5, 5), (35, 35), **params)


def test_default_params_are_exact_1995():
    p = _planner()
    assert p.w_inertia == 1.0          # full momentum, no inertia weight
    assert p.c1 == 2.0 and p.c2 == 2.0  # acceleration constants
    assert p.vmax == 20.0               # Vmax clamp
    assert p.enable_safeguards is False


def test_velocity_update_matches_1995_formula():
    p = _planner()  # safeguards off, single particle

    # distinct pbest / gbest / position so all three terms are exercised
    setup = np.random.default_rng(123)
    p.particles[0] = setup.uniform(5, 35, (p.num_points, 2))
    p.p_best[0] = setup.uniform(5, 35, (p.num_points, 2))
    p.g_best = setup.uniform(5, 35, (p.num_points, 2))

    v_prev = p.velocities[0].copy()
    x_prev = p.particles[0].copy()
    pbest = p.p_best[0].copy()
    gbest = p.g_best.copy()

    # reproduce the exact r1, r2 the step will draw next
    clone = np.random.default_rng()
    clone.bit_generator.state = p.rng.bit_generator.state
    r1 = clone.random((p.num_points, 2))
    r2 = clone.random((p.num_points, 2))

    expected = (
        p.w_inertia * v_prev
        + p.c1 * r1 * (pbest - x_prev)
        + p.c2 * r2 * (gbest - x_prev)
    )
    expected = np.clip(expected, -p.vmax, p.vmax)
    expected[0] = 0.0   # start is pinned
    expected[-1] = 0.0  # goal is pinned

    p.step_once()
    assert np.allclose(p.velocities[0], expected)


def test_safeguards_toggle_changes_dynamics():
    off = _planner(num_particles=20, num_points=12, enable_safeguards=False, seed=7)
    on = _planner(num_particles=20, num_points=12, enable_safeguards=True, seed=7)

    # identical initial swarm from the same seed
    assert np.allclose(off.velocities, on.velocities)

    off.step_once()
    on.step_once()

    # adaptive inertia / social gain make the safeguarded update diverge
    assert not np.allclose(off.velocities, on.velocities)
