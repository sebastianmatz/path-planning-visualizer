"""Shared building blocks for the trajectory-optimization planners.

``STOMP``, ``TrajOpt``, ``ITOMP`` and ``GPMP`` all represent a path as a fixed
number of waypoints and share the same primitives: a straight-line / escape
initialization, a finite-difference smoothness metric ``R = AᵀA`` (sum of
squared accelerations), and a signed-distance-field (SDF) lookup with its
spatial gradient.  Centralizing them here keeps the four planners consistent and
removes the near-duplicate code that previously lived in each module.

``CHOMP`` deliberately keeps its own (verified) implementation and does not use
these helpers.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..geometry import make_distance_field


def straight_line(start: Tuple[int, int], goal: Tuple[int, int], n: int) -> np.ndarray:
    """``(n, 2)`` straight-line trajectory from ``start`` to ``goal`` inclusive."""
    s = np.asarray(start, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return (1.0 - ts) * s + ts * g


def _has_internal_collision(traj: np.ndarray, occ: np.ndarray) -> bool:
    h, w = occ.shape
    for i in range(1, len(traj) - 1):
        x = int(np.clip(traj[i, 0], 0, w - 1))
        y = int(np.clip(traj[i, 1], 0, h - 1))
        if occ[y, x]:
            return True
    return False


def escape_init(traj: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                occ: np.ndarray) -> np.ndarray:
    """Bend a colliding straight line off the obstacle with a sine bump.

    If the straight-line initialization passes through an obstacle, try a
    half-sine perpendicular perturbation on each side and return the first
    collision-free one; if neither is clear, return the last attempt (the
    optimizer then takes over).  Returns ``traj`` unchanged when already free.
    """
    if not _has_internal_collision(traj, occ):
        return traj

    h, w = occ.shape
    n = len(traj)
    dx = float(goal[0] - start[0])
    dy = float(goal[1] - start[1])
    path_len = np.hypot(dx, dy) + 1e-6
    perp = np.array([-dy / path_len, dx / path_len])
    amplitude = min(h, w) * 0.3

    test = traj
    for sign in (1.0, -1.0):
        test = traj.copy()
        for i in range(1, n - 1):
            t = i / (n - 1)
            offset = sign * amplitude * np.sin(np.pi * t)
            test[i] += offset * perp
        if not _has_internal_collision(test, occ):
            return test
    return test


def fd_acceleration_matrix(n_internal: int) -> np.ndarray:
    """Second-difference (acceleration) finite-difference matrix ``A``.

    ``A`` acts on the ``n_internal`` interior waypoints; row ``i`` encodes
    ``x[i-1] - 2 x[i] + x[i+1]`` with the (fixed) endpoints contributing zero.
    """
    a = np.zeros((n_internal, n_internal), dtype=np.float64)
    for i in range(n_internal):
        a[i, i] = -2.0
        if i > 0:
            a[i, i - 1] = 1.0
        if i < n_internal - 1:
            a[i, i + 1] = 1.0
    return a


def smoothness_hessian(n_internal: int, reg: float = 1e-6) -> np.ndarray:
    """Smoothness metric ``R = AᵀA`` (+ tiny regularization) over interior points."""
    a = fd_acceleration_matrix(n_internal)
    return a.T @ a + reg * np.eye(n_internal, dtype=np.float64)


def signed_distance_field(occ: np.ndarray) -> np.ndarray:
    """True signed distance field: positive in free space, negative inside obstacles.

    ``d(x) = dist_to_obstacle(x) - dist_inside_obstacle(x)`` (the signed Euclidean
    distance transform used by CHOMP/STOMP), so the sign and penetration depth are
    correct when a trajectory passes through an obstacle.
    """
    dist_outside = make_distance_field(occ)
    inside = cv2.distanceTransform((occ > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return dist_outside - inside


def make_sdf(occ: np.ndarray, signed: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance field and its Sobel gradient (``dist``, ``grad_x``, ``grad_y``).

    With ``signed=True`` the field is the true signed distance (negative inside
    obstacles); otherwise it is the unsigned clearance field. ``grad`` points
    toward increasing clearance (away from obstacles).
    """
    dist_field = signed_distance_field(occ) if signed else make_distance_field(occ)
    grad_x = cv2.Sobel(dist_field, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(dist_field, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    return dist_field, grad_x, grad_y


def sdf_query(dist_field: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray,
              x: float, y: float) -> Tuple[float, np.ndarray]:
    """Clearance distance and (unit) clearance gradient at ``(x, y)``."""
    h, w = dist_field.shape
    ix = int(np.clip(x, 0, w - 1))
    iy = int(np.clip(y, 0, h - 1))
    d = float(dist_field[iy, ix])
    g = np.array([grad_x[iy, ix], grad_y[iy, ix]], dtype=np.float64)
    norm = float(np.linalg.norm(g))
    if norm > 1e-9:
        g = g / norm
    return d, g


def sdf_query_batch(dist_field: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray,
                    xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`sdf_query` over arrays of points.

    Identical nearest-pixel semantics: returns ``(d, g)`` with ``d`` of shape
    ``(n,)`` and unit clearance gradients ``g`` of shape ``(n, 2)`` (rows with a
    ~zero gradient are left un-normalized, exactly like the scalar version).
    Coordinates are clamped to the grid (lower bound 0, so ``int`` truncation and
    ``astype`` floor agree).
    """
    h, w = dist_field.shape
    ix = np.clip(np.asarray(xs, dtype=np.float64), 0, w - 1).astype(np.intp)
    iy = np.clip(np.asarray(ys, dtype=np.float64), 0, h - 1).astype(np.intp)
    d = dist_field[iy, ix].astype(np.float64)
    g = np.stack((grad_x[iy, ix], grad_y[iy, ix]), axis=1).astype(np.float64)
    norms = np.linalg.norm(g, axis=1)
    mask = norms > 1e-9
    g[mask] /= norms[mask, None]
    return d, g
