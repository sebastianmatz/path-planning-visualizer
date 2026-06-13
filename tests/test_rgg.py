"""Guards the shared RGG-radius helper against the previous inline formulas.

``FMT*`` and ``BIT*`` were refactored to call ``rgg_radius`` instead of computing
the shrinking connection radius inline.  These tests reproduce the *old*
expressions verbatim and assert the helper returns byte-identical values, so the
refactor provably did not change those (paper-faithful) planners' behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from path_planning_visualizer.planners._rgg import rgg_radius, unit_ball_volume


def _unit_ball_volume_old(dimension: int) -> float:
    if dimension == 0:
        return 1.0
    if dimension == 1:
        return 2.0
    return 2.0 * np.pi / dimension * _unit_ball_volume_old(dimension - 2)


def _fmt_radius_old(num_samples: int, free_space_volume: float) -> float:
    dimension = 2
    n = max(2, num_samples)
    ubv = _unit_ball_volume_old(dimension)
    free_volume = max(1.0, free_space_volume)
    gamma = 2.0 * np.power(1.0 / dimension, 1.0 / dimension) * np.power(free_volume / ubv, 1.0 / dimension)
    return float(1.1 * gamma * np.power(np.log(n) / n, 1.0 / dimension))


def _bit_radius_old(q: int, free_space_volume: float) -> float:
    dimension = 2
    q = max(2, q)
    ubv = _unit_ball_volume_old(dimension)
    free_volume = max(1.0, free_space_volume)
    gamma = 2.0 * np.power(1.0 + 1.0 / dimension, 1.0 / dimension) * np.power(free_volume / ubv, 1.0 / dimension)
    return float(1.1 * gamma * np.power(np.log(q) / q, 1.0 / dimension))


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_unit_ball_volume_matches_old(dim):
    assert unit_ball_volume(dim) == _unit_ball_volume_old(dim)


@pytest.mark.parametrize("n", [1, 2, 10, 200, 400, 5000])
@pytest.mark.parametrize("free", [0.0, 1.0, 100.0, 3500.0, 1_000_000.0])
def test_fmt_radius_equivalence(n, free):
    assert rgg_radius(n, free, plus_one=False) == _fmt_radius_old(n, free)


@pytest.mark.parametrize("q", [1, 2, 10, 200, 600, 5000])
@pytest.mark.parametrize("free", [0.0, 1.0, 100.0, 3500.0, 1_000_000.0])
def test_bit_radius_equivalence(q, free):
    assert rgg_radius(q, free, plus_one=True) == _bit_radius_old(q, free)


def test_eta_caps_radius():
    # Early on (small n) the unconstrained radius is large; eta must clamp it.
    big = rgg_radius(3, 1_000_000.0, plus_one=True)
    assert big > 18.0
    assert rgg_radius(3, 1_000_000.0, plus_one=True, eta=18.0) == 18.0
    # When the formula is already below eta, eta has no effect.
    small = rgg_radius(5000, 100.0, plus_one=True)
    assert small < 18.0
    assert rgg_radius(5000, 100.0, plus_one=True, eta=18.0) == small
