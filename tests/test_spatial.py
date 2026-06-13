"""Exactness tests for the GridIndex spatial helper.

GridIndex must return results identical to a brute-force argmin / radius scan,
because the planners rely on that equivalence to stay deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from path_planning_visualizer.planners._spatial import GridIndex


def brute_nearest(xs, ys, qx, qy) -> int:
    best_idx, best_d2 = -1, float("inf")
    for idx in range(len(xs)):
        d2 = (xs[idx] - qx) ** 2 + (ys[idx] - qy) ** 2
        if d2 < best_d2 or (d2 == best_d2 and idx < best_idx):
            best_d2, best_idx = d2, idx
    return best_idx


def brute_within(xs, ys, qx, qy, r):
    r2 = r * r
    return [i for i in range(len(xs)) if (xs[i] - qx) ** 2 + (ys[i] - qy) ** 2 <= r2]


@pytest.mark.parametrize("cell_size", [1.0, 5.0, 18.0, 50.0])
def test_nearest_matches_bruteforce(cell_size):
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 200, size=(400, 2))
    idx = GridIndex(cell_size)
    xs, ys = [], []
    for x, y in pts:
        idx.add(float(x), float(y))
        xs.append(float(x))
        ys.append(float(y))

    queries = rng.uniform(-10, 210, size=(300, 2))
    for qx, qy in queries:
        assert idx.nearest(float(qx), float(qy)) == brute_nearest(xs, ys, qx, qy)


@pytest.mark.parametrize("cell_size", [1.0, 7.0, 25.0])
@pytest.mark.parametrize("radius", [3.0, 15.0, 60.0])
def test_within_matches_bruteforce(cell_size, radius):
    rng = np.random.default_rng(11)
    pts = rng.uniform(0, 150, size=(350, 2))
    idx = GridIndex(cell_size)
    xs, ys = [], []
    for x, y in pts:
        idx.add(float(x), float(y))
        xs.append(float(x))
        ys.append(float(y))

    for qx, qy in rng.uniform(0, 150, size=(120, 2)):
        assert idx.within(float(qx), float(qy), radius) == brute_within(xs, ys, qx, qy, radius)


def test_nearest_ties_pick_lowest_index():
    idx = GridIndex(10.0)
    idx.add(5.0, 5.0)   # index 0
    idx.add(5.0, 5.0)   # index 1 (identical point)
    assert idx.nearest(5.0, 5.0) == 0


def test_empty_index_returns_minus_one():
    assert GridIndex(10.0).nearest(0.0, 0.0) == -1
