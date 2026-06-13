"""Occupancy-grid I/O and editing helpers.

Pure NumPy utilities (no Qt) shared by map loading, the interactive map editor,
and map saving. Keeping them here means the load and save paths agree on the
same grayscale<->occupancy convention, and the brush logic is unit-testable
without constructing any widgets.

Convention: an occupancy grid is a boolean array, ``True`` = obstacle, indexed
``occ[y, x]``. In images, bright pixels (``> 127``) are free space.
"""

from __future__ import annotations

import numpy as np


def image_to_occupancy(gray: np.ndarray) -> np.ndarray:
    """Threshold a grayscale image to a boolean occupancy grid.

    Bright pixels are treated as free. If the image is mostly dark, the polarity
    is flipped so the majority is still treated as free (matching the loader's
    auto-inversion heuristic).
    """
    free = gray > 127
    if float(free.mean()) < 0.5:
        free = ~free
    return ~free


def occupancy_to_image(occ: np.ndarray) -> np.ndarray:
    """Render an occupancy grid as a uint8 grayscale image (free=255, obstacle=0)."""
    return np.where(occ, np.uint8(0), np.uint8(255)).astype(np.uint8)


def blank_occupancy(height: int, width: int) -> np.ndarray:
    """Return an all-free occupancy grid of the given size."""
    return np.zeros((int(height), int(width)), dtype=bool)


def paint_disk(occ: np.ndarray, cx: int, cy: int, radius: int, obstacle: bool) -> None:
    """Set a bounds-clipped filled disk in ``occ`` to obstacle/free in place.

    Used as the editor brush: ``obstacle=True`` draws walls, ``False`` erases.
    """
    h, w = occ.shape
    r = max(0, int(radius))
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    occ[y0:y1, x0:x1][mask] = obstacle
