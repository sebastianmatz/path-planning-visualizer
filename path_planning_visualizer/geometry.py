from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .types import FloatPoint, OccupancyGrid, Point


def dist(a: Point, b: Point) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        a: First point (x, y)
        b: Second point (x, y)
        
    Returns:
        Euclidean distance between a and b
    """
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def l1_dist(a: Point, b: Point) -> float:
    """Calculate Manhattan distance between two points."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def linf_dist(a: Point, b: Point) -> float:
    """Calculate Chebyshev / L-infinity distance between two points."""
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


def round_point(p: FloatPoint) -> Point:
    """Round a continuous state to the occupancy-grid pixel used by the UI."""
    return (int(round(p[0])), int(round(p[1])))


def select_holonomic_input(from_state: FloatPoint, to_state: FloatPoint, delta_t: float) -> FloatPoint:
    """Return the paper-style holonomic control that best moves toward ``to_state``.

    This specializes LaValle's ``SELECT_INPUT`` step to the simple holonomic model
    ``x_dot = u`` with ``||u|| <= 1`` over a fixed integration interval ``delta_t``.
    """
    dx = float(to_state[0] - from_state[0])
    dy = float(to_state[1] - from_state[1])
    distance = float(np.hypot(dx, dy))
    if distance <= 1e-12 or delta_t <= 1e-12:
        return (0.0, 0.0)
    if distance <= delta_t:
        return (dx / delta_t, dy / delta_t)
    return (dx / distance, dy / distance)


def integrate_holonomic_state(state: FloatPoint, control: FloatPoint, delta_t: float) -> FloatPoint:
    """Integrate the holonomic state equation ``x_dot = u`` for one fixed step."""
    return (
        float(state[0] + delta_t * control[0]),
        float(state[1] + delta_t * control[1]),
    )


def steer(from_pt: Point, to_pt: Point, step: float) -> Point:
    """Move from from_pt towards to_pt by at most step distance.
    
    Args:
        from_pt: Starting point
        to_pt: Target point
        step: Maximum distance to move
        
    Returns:
        New point, moved at most step distance towards to_pt
    """
    d = dist(from_pt, to_pt)
    if d <= step:
        return (int(round(to_pt[0])), int(round(to_pt[1])))
    ux = (to_pt[0] - from_pt[0]) / d
    uy = (to_pt[1] - from_pt[1]) / d
    return (int(round(from_pt[0] + ux * step)), int(round(from_pt[1] + uy * step)))


def clamp_point(p: Point, w: int, h: int) -> Point:
    """Clamp point coordinates to image boundaries.
    
    Args:
        p: Point to clamp
        w: Image width
        h: Image height
        
    Returns:
        Point with coordinates clamped to [0, w-1] x [0, h-1]
    """
    x = int(np.clip(p[0], 0, w - 1))
    y = int(np.clip(p[1], 0, h - 1))
    return (x, y)


def make_distance_field(occ: OccupancyGrid) -> np.ndarray:
    """Euclidean distance (in pixels) from every free cell to the nearest obstacle.

    Shared helper for the many planners that need an obstacle-clearance /
    distance field over the occupancy grid.
    """
    free_space = (~occ).astype(np.uint8)
    return cv2.distanceTransform(free_space, cv2.DIST_L2, 5)


def bilinear_sample_scalar(field: np.ndarray, x: float, y: float) -> float:
    """Sample a 2D scalar field continuously with bilinear interpolation."""
    h, w = field.shape
    x = float(np.clip(x, 0.0, w - 1))
    y = float(np.clip(y, 0.0, h - 1))

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx = x - x0
    ty = y - y0

    v00 = float(field[y0, x0])
    v10 = float(field[y0, x1])
    v01 = float(field[y1, x0])
    v11 = float(field[y1, x1])

    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def bilinear_sample_vector(
    field_x: np.ndarray,
    field_y: np.ndarray,
    x: float,
    y: float,
) -> np.ndarray:
    """Sample a 2D vector field continuously with bilinear interpolation."""
    return np.array(
        [
            bilinear_sample_scalar(field_x, x, y),
            bilinear_sample_scalar(field_y, x, y),
        ],
        dtype=np.float64,
    )


def bilinear_sample_scalar_batch(
    field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a 2D scalar field at many points at once.

    Vectorized equivalent of :func:`bilinear_sample_scalar` (identical
    interpolation, identical edge clamping) for arrays of coordinates. Used by the
    trajectory optimizers' per-waypoint distance/gradient lookups.
    """
    h, w = field.shape
    xs = np.clip(np.asarray(xs, dtype=np.float64), 0.0, w - 1)
    ys = np.clip(np.asarray(ys, dtype=np.float64), 0.0, h - 1)

    x0 = np.floor(xs).astype(np.intp)
    y0 = np.floor(ys).astype(np.intp)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = xs - x0
    ty = ys - y0

    v00 = field[y0, x0]
    v10 = field[y0, x1]
    v01 = field[y1, x0]
    v11 = field[y1, x1]

    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def segment_points(a: Point, b: Point, samples: Optional[int] = None) -> List[Point]:
    """Return every integer cell the segment ``a``->``b`` passes through.

    Uses an exact grid (voxel) traversal, so no cell is skipped. This matters at
    obstacle corners: fixed-rate sampling (the old ``max(dx, dy)`` scheme) can round
    *around* the corner cell of a diagonal edge and report a colliding edge as clear.
    The traversal walks cell boundary to cell boundary and never misses a touched
    cell. ``samples`` is accepted for backward compatibility but ignored — the
    traversal is already complete and exact regardless of segment length.
    """
    x0, y0 = int(round(a[0])), int(round(a[1]))
    x1, y1 = int(round(b[0])), int(round(b[1]))
    cells: List[Point] = [(x0, y0)]
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return cells

    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    adx, ady = abs(dx), abs(dy)
    # Parametric distance (t in [0, 1]) to the next cell boundary, and across one
    # full cell, on each axis. Cells are centered on integers, so the first boundary
    # is half a cell away.
    t_max_x = (0.5 / adx) if adx else float("inf")
    t_max_y = (0.5 / ady) if ady else float("inf")
    t_delta_x = (1.0 / adx) if adx else float("inf")
    t_delta_y = (1.0 / ady) if ady else float("inf")

    x, y = x0, y0
    guard = adx + ady + 2  # upper bound on the number of steps
    while (x != x1 or y != y1) and guard > 0:
        guard -= 1
        if t_max_x < t_max_y:
            x += step_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_x:
            y += step_y
            t_max_y += t_delta_y
        else:  # exact diagonal crossing through a lattice corner: advance both
            x += step_x
            y += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        cells.append((x, y))
    return cells


def line_collision_free(
    a: Point, 
    b: Point, 
    occ: OccupancyGrid, 
    samples: Optional[int] = None
) -> bool:
    """Check if line segment from a to b is collision-free.
    
    Uses adaptive rasterization to cover long segments robustly.
    
    Args:
        a: Start point of line segment
        b: End point of line segment
        occ: Occupancy grid (True = obstacle)
        samples: Number of points to check along the line
        
    Returns:
        True if the entire line is collision-free
    """
    h, w = occ.shape
    for x, y in segment_points(a, b, samples=samples):
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        if occ[y, x]:
            return False
    return True


def compute_path_length(path: List[Point]) -> float:
    """Return geometric path length in pixels."""
    if len(path) < 2:
        return 0.0
    return float(sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1)))


def iter_path_pixels(path: List[Point]) -> List[Point]:
    """Return deduplicated rasterized pixels along a path."""
    if not path:
        return []
    if len(path) == 1:
        return [path[0]]

    pts: List[Point] = []
    for i in range(len(path) - 1):
        for p in segment_points(path[i], path[i + 1]):
            if not pts or pts[-1] != p:
                pts.append(p)
    return pts


def _resample_to_targets(
    pts: np.ndarray,
    deltas: np.ndarray,
    seg_lens: np.ndarray,
    cum: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Interpolate ``pts`` (a polyline) at the given cumulative-arclength ``targets``.

    Shared core of the resample helpers; walks the segments once and linearly
    interpolates each target position. Returns an ``(len(targets), 2)`` array.
    """
    resampled = np.zeros((len(targets), 2), dtype=np.float64)
    seg_idx = 0
    for i, t in enumerate(targets):
        while seg_idx < len(seg_lens) - 1 and t > cum[seg_idx + 1]:
            seg_idx += 1
        seg_len = seg_lens[seg_idx]
        if seg_len <= 1e-6:
            resampled[i] = pts[seg_idx]
            continue
        local_t = (t - cum[seg_idx]) / seg_len
        resampled[i] = pts[seg_idx] + local_t * deltas[seg_idx]
    return resampled


def resample_path_points(path: List[Point], spacing: float = 4.0) -> List[Tuple[float, float]]:
    """Resample a path polyline with approximately uniform spacing."""
    if not path:
        return []
    if len(path) == 1:
        return [(float(path[0][0]), float(path[0][1]))]

    pts = np.array(path, dtype=np.float64)
    deltas = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    total_len = float(np.sum(seg_lens))
    if total_len <= 1e-6:
        return [(float(path[0][0]), float(path[0][1]))]

    sample_count = max(2, int(np.ceil(total_len / max(1e-6, spacing))) + 1)
    targets = np.linspace(0.0, total_len, sample_count)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    resampled = _resample_to_targets(pts, deltas, seg_lens, cum, targets)

    return [(float(p[0]), float(p[1])) for p in resampled]


def resample_float_path_points(
    path: List[Tuple[float, float]],
    spacing: float = 4.0,
) -> List[Tuple[float, float]]:
    """Resample a floating-point polyline with approximately uniform spacing."""
    if not path:
        return []
    if len(path) == 1:
        return [(float(path[0][0]), float(path[0][1]))]

    pts = np.array(path, dtype=np.float64)
    deltas = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    total_len = float(np.sum(seg_lens))
    if total_len <= 1e-6:
        return [(float(path[0][0]), float(path[0][1]))]

    sample_count = max(2, int(np.ceil(total_len / max(1e-6, spacing))) + 1)
    targets = np.linspace(0.0, total_len, sample_count)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    resampled = _resample_to_targets(pts, deltas, seg_lens, cum, targets)

    return [(float(p[0]), float(p[1])) for p in resampled]


def smooth_float_polyline(
    path: List[Tuple[float, float]],
    spacing: float = 3.0,
    iterations: int = 2,
) -> List[Tuple[float, float]]:
    """Return a smoother display-only version of a floating-point polyline."""
    if not path:
        return []
    if len(path) == 1:
        return [(float(path[0][0]), float(path[0][1]))]

    smoothed = resample_float_path_points(path, spacing=max(1.0, spacing))
    if len(smoothed) < 3:
        return smoothed

    for _ in range(max(0, iterations)):
        candidate: List[Tuple[float, float]] = [smoothed[0]]
        for i in range(len(smoothed) - 1):
            p0 = np.array(smoothed[i], dtype=np.float64)
            p1 = np.array(smoothed[i + 1], dtype=np.float64)
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            candidate.append((float(q[0]), float(q[1])))
            candidate.append((float(r[0]), float(r[1])))
        candidate.append(smoothed[-1])
        smoothed = candidate

    return smoothed


def resample_float_path_fixed_count(
    path: List[Tuple[float, float]],
    sample_count: int,
) -> List[Tuple[float, float]]:
    """Resample a floating-point polyline to an exact number of samples."""
    if not path:
        return []
    sample_count = max(1, int(sample_count))
    if len(path) == 1 or sample_count == 1:
        p = (float(path[0][0]), float(path[0][1]))
        return [p for _ in range(sample_count)]

    pts = np.array(path, dtype=np.float64)
    deltas = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    total_len = float(np.sum(seg_lens))
    if total_len <= 1e-6:
        p = (float(path[0][0]), float(path[0][1]))
        return [p for _ in range(sample_count)]

    targets = np.linspace(0.0, total_len, sample_count)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    resampled = _resample_to_targets(pts, deltas, seg_lens, cum, targets)

    return [(float(p[0]), float(p[1])) for p in resampled]


def blend_float_paths(
    path_a: List[Tuple[float, float]],
    path_b: List[Tuple[float, float]],
    alpha: float,
) -> List[Tuple[float, float]]:
    """Blend two float polylines by resampling them to a shared waypoint count."""
    if not path_a:
        return list(path_b)
    if not path_b:
        return list(path_a)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    sample_count = max(24, len(path_a), len(path_b))
    a_pts = np.array(resample_float_path_fixed_count(path_a, sample_count), dtype=np.float64)
    b_pts = np.array(resample_float_path_fixed_count(path_b, sample_count), dtype=np.float64)
    blended = (1.0 - alpha) * a_pts + alpha * b_pts
    return [(float(p[0]), float(p[1])) for p in blended]


def shortcut_path(path: List[Point], occ: OccupancyGrid, max_passes: int = 2) -> List[Point]:
    """Greedily shortcut a polyline while keeping it collision-free."""
    if len(path) < 3:
        return list(path)

    shortened = list(path)
    for _ in range(max_passes):
        improved = False
        new_path = [shortened[0]]
        i = 0

        while i < len(shortened) - 1:
            next_idx = i + 1
            for j in range(len(shortened) - 1, i + 1, -1):
                if line_collision_free(shortened[i], shortened[j], occ):
                    next_idx = j
                    break

            if next_idx > i + 1:
                improved = True
            new_path.append(shortened[next_idx])
            i = next_idx

        shortened = new_path
        if not improved:
            break

    return shortened


def float_polyline_collision_free(
    path: List[Tuple[float, float]],
    occ: OccupancyGrid,
) -> bool:
    """Check a float polyline by rasterizing each segment onto the occupancy grid."""
    if len(path) < 2:
        return True

    for i in range(len(path) - 1):
        a = (int(round(path[i][0])), int(round(path[i][1])))
        b = (int(round(path[i + 1][0])), int(round(path[i + 1][1])))
        if not line_collision_free(a, b, occ):
            return False
    return True


def smooth_display_path(
    path: List[Point],
    occ: OccupancyGrid,
    spacing: float = 3.0,
    iterations: int = 2,
) -> List[Tuple[float, float]]:
    """Return a denser, cleaner display-only polyline while keeping it collision-free."""
    if not path:
        return []
    if len(path) == 1:
        return [(float(path[0][0]), float(path[0][1]))]

    smoothed = resample_path_points(path, spacing=max(1.0, spacing))
    if len(smoothed) < 3:
        return smoothed

    for _ in range(max(0, iterations)):
        candidate: List[Tuple[float, float]] = [smoothed[0]]
        for i in range(len(smoothed) - 1):
            p0 = np.array(smoothed[i], dtype=np.float64)
            p1 = np.array(smoothed[i + 1], dtype=np.float64)
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            candidate.append((float(q[0]), float(q[1])))
            candidate.append((float(r[0]), float(r[1])))
        candidate.append(smoothed[-1])

        if float_polyline_collision_free(candidate, occ):
            smoothed = candidate
        else:
            break

    return smoothed
