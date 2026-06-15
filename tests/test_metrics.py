"""Unit tests for the path-quality metrics."""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.metrics import (
    PathMetrics,
    compute_path_mean_clearance,
    compute_path_metrics,
    compute_path_min_clearance,
    compute_path_smoothness,
)


def test_straight_line_smoothness_is_zero():
    path = [(0, 0), (10, 0), (20, 0), (30, 0)]
    assert compute_path_smoothness(path) < 1e-9


def test_zigzag_has_positive_smoothness():
    path = [(0, 0), (10, 10), (20, 0), (30, 10), (40, 0)]
    assert compute_path_smoothness(path) > 0.1


def test_collinear_two_point_path_smoothness_is_negligible():
    assert compute_path_smoothness([(0, 0), (5, 5)]) < 1e-9


def test_clearance_on_constant_field():
    field = np.full((20, 20), 3.0, dtype=np.float64)
    path = [(2, 2), (15, 15)]
    assert abs(compute_path_min_clearance(path, field) - 3.0) < 1e-9
    assert abs(compute_path_mean_clearance(path, field) - 3.0) < 1e-9


def test_min_clearance_picks_the_low_spot():
    field = np.full((20, 20), 3.0, dtype=np.float64)
    field[5, 8] = 1.0  # a pixel on the horizontal path y=5
    path = [(2, 5), (15, 5)]
    assert abs(compute_path_min_clearance(path, field) - 1.0) < 1e-9
    # the single low pixel pulls the mean below the constant, but above the min
    mean = compute_path_mean_clearance(path, field)
    assert 1.0 < mean < 3.0


def test_metrics_without_clearance_field():
    m = compute_path_metrics([(0, 0), (3, 4)], None)
    assert isinstance(m, PathMetrics)
    assert abs(m.length_px - 5.0) < 1e-9
    assert m.min_clearance_px is None and m.mean_clearance_px is None
    assert m.smoothness is not None


def test_metrics_with_clearance_field_populates_all():
    field = np.full((20, 20), 2.0, dtype=np.float64)
    m = compute_path_metrics([(1, 1), (10, 1)], field)
    assert m.min_clearance_px is not None and m.mean_clearance_px is not None
    assert abs(m.min_clearance_px - 2.0) < 1e-9
