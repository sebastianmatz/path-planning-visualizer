from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .geometry import (
    compute_path_length,
    iter_path_pixels,
    resample_path_points,
    segment_points,
)
from .types import Point


def compute_path_min_clearance(path: List[Point], clearance_field: np.ndarray) -> float:
    """Return the minimum obstacle clearance along a path in pixels."""
    if len(path) < 2:
        return 0.0

    min_clearance = float("inf")
    for i in range(len(path) - 1):
        for x, y in segment_points(path[i], path[i + 1]):
            min_clearance = min(min_clearance, float(clearance_field[y, x]))

    return 0.0 if min_clearance == float("inf") else min_clearance


def compute_path_mean_clearance(path: List[Point], clearance_field: np.ndarray) -> float:
    """Return the mean obstacle clearance along a path in pixels."""
    pixels = iter_path_pixels(path)
    if not pixels:
        return 0.0
    values = [float(clearance_field[y, x]) for x, y in pixels]
    return float(np.mean(values))


def compute_path_smoothness(path: List[Point], spacing: float = 4.0) -> float:
    """Return an average squared turning-angle smoothness score in rad^2.

    Lower is smoother. The path is resampled first so the metric is less
    dependent on the original waypoint density.
    """
    samples = resample_path_points(path, spacing=spacing)
    if len(samples) < 3:
        return 0.0

    turn_angles: List[float] = []
    for i in range(1, len(samples) - 1):
        p_prev = np.array(samples[i - 1], dtype=np.float64)
        p_curr = np.array(samples[i], dtype=np.float64)
        p_next = np.array(samples[i + 1], dtype=np.float64)

        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue

        cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        turn_angles.append(float(np.arccos(cos_angle)))

    if not turn_angles:
        return 0.0
    return float(np.mean(np.square(turn_angles)))


@dataclass(frozen=True)
class PathMetrics:
    """Summary metrics for a final path."""

    length_px: float
    min_clearance_px: Optional[float] = None
    mean_clearance_px: Optional[float] = None
    smoothness: Optional[float] = None


def compute_path_metrics(path: List[Point], clearance_field: Optional[np.ndarray]) -> PathMetrics:
    """Compute the main path-quality metrics used in the UI."""
    length_px = compute_path_length(path)
    if clearance_field is None:
        return PathMetrics(length_px=length_px, smoothness=compute_path_smoothness(path))

    return PathMetrics(
        length_px=length_px,
        min_clearance_px=compute_path_min_clearance(path, clearance_field),
        mean_clearance_px=compute_path_mean_clearance(path, clearance_field),
        smoothness=compute_path_smoothness(path),
    )
