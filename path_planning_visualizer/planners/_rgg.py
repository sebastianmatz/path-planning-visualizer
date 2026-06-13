"""Shared Random-Geometric-Graph (RGG) connection-radius math.

Several asymptotically-optimal sampling planners (``FMT*``, ``BIT*``, ``RRT*``)
share the same shrinking connection radius derived from the measure of the free
space.  The radius has the form

    r(n) = inflation * gamma * (log(n) / n) ** (1 / d)

where ``gamma`` is a constant determined by the dimension and the free-space
volume.  The leading constant differs slightly between formulations:

* ``RRG`` / ``RRT*`` / ``BIT*`` (Karaman & Frazzoli 2011, Gammell et al. 2015)
  use ``2 * (1 + 1/d) ** (1/d)``.
* ``FMT*`` (Janson et al. 2015) uses ``2 * (1/d) ** (1/d)``.

Both are captured here via the ``plus_one`` flag so the three planners can share
one implementation instead of duplicating the formula (and the unit-ball-volume
recursion) in each module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def unit_ball_volume(dimension: int) -> float:
    """Volume of the unit ball in ``dimension`` dimensions (Lebesgue measure)."""
    if dimension == 0:
        return 1.0
    if dimension == 1:
        return 2.0
    return 2.0 * np.pi / dimension * unit_ball_volume(dimension - 2)


def rgg_radius(
    n: float,
    free_volume: float,
    *,
    dim: int = 2,
    plus_one: bool = True,
    inflation: float = 1.1,
    eta: Optional[float] = None,
) -> float:
    """Shrinking RGG connection radius for ``n`` samples.

    Args:
        n: Number of samples (vertices) the radius is computed for.
        free_volume: Lebesgue measure of the free configuration space.
        dim: Configuration-space dimension (2 for this 2D tool).
        plus_one: ``True`` for the RRG/RRT*/BIT* constant ``2*(1+1/d)^(1/d)``,
            ``False`` for the FMT* constant ``2*(1/d)^(1/d)``.
        inflation: Safety factor on the threshold radius (``1.1`` in the cited
            implementations).
        eta: Optional steering range; when given the result is capped at ``eta``
            (RRT* connects only within its steering parameter).

    Returns:
        The connection radius.
    """
    n_eff = max(2.0, float(n))
    base = (1.0 + 1.0 / dim) if plus_one else (1.0 / dim)
    zeta = unit_ball_volume(dim)
    free = max(1.0, float(free_volume))
    gamma = 2.0 * np.power(base, 1.0 / dim) * np.power(free / zeta, 1.0 / dim)
    r = inflation * gamma * np.power(np.log(n_eff) / n_eff, 1.0 / dim)
    if eta is not None:
        r = min(r, float(eta))
    return float(r)
