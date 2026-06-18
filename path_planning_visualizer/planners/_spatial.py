"""Lightweight uniform-grid spatial index for nearest-neighbor queries.

Replaces the O(n) "rebuild a numpy array and scan" pattern used by the
sampling-based planners. It is dependency-free and, by design, returns results
that are *identical* to the previous brute-force code:

- ``nearest`` breaks ties by lowest index, reproducing ``np.argmin``.
- ``within`` returns indices in ascending order, reproducing ``np.where``.

Points are never moved after insertion (the planners only append nodes; RRT*
rewiring changes parents, not coordinates), so the index never needs updates.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple


class GridIndex:
    """Uniform-grid index over 2D points, addressed by insertion index."""

    def __init__(self, cell_size: float) -> None:
        self.cell_size = max(1.0, float(cell_size))
        self._cells: Dict[Tuple[int, int], List[int]] = {}
        self.xs: List[float] = []
        self.ys: List[float] = []

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.cell_size)), int(math.floor(y / self.cell_size)))

    def add(self, x: float, y: float) -> int:
        """Insert a point and return its index (matches the caller's node index)."""
        idx = len(self.xs)
        self.xs.append(float(x))
        self.ys.append(float(y))
        self._cells.setdefault(self._cell(x, y), []).append(idx)
        return idx

    def __len__(self) -> int:
        return len(self.xs)

    def nearest(self, x: float, y: float) -> int:
        """Return the index of the closest point (exact; ties -> lowest index)."""
        total = len(self.xs)
        if total == 0:
            return -1

        qcx, qcy = self._cell(x, y)
        best_idx = -1
        best_d2 = float("inf")
        examined = 0
        k = 0
        while True:
            for cx, cy in self._ring(qcx, qcy, k):
                bucket = self._cells.get((cx, cy))
                if not bucket:
                    continue
                for idx in bucket:
                    examined += 1
                    dx = self.xs[idx] - x
                    dy = self.ys[idx] - y
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2 or (d2 == best_d2 and idx < best_idx):
                        best_d2 = d2
                        best_idx = idx

            # Every still-unexamined point sits in a ring at Chebyshev distance
            # >= k+1, i.e. at Euclidean distance >= k * cell_size. Stop once the
            # current best cannot be beaten, or once all points are seen.
            if best_idx != -1 and best_d2 <= (k * self.cell_size) ** 2:
                break
            if examined >= total:
                break
            k += 1
        return best_idx

    def within(self, x: float, y: float, radius: float) -> List[int]:
        """Return indices with distance <= radius, ascending (matches np.where)."""
        if radius < 0 or not self.xs:
            return []
        r2 = radius * radius
        cx0, cy0 = self._cell(x - radius, y - radius)
        cx1, cy1 = self._cell(x + radius, y + radius)

        found: List[int] = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                bucket = self._cells.get((cx, cy))
                if not bucket:
                    continue
                for idx in bucket:
                    dx = self.xs[idx] - x
                    dy = self.ys[idx] - y
                    if dx * dx + dy * dy <= r2:
                        found.append(idx)
        found.sort()
        return found

    @staticmethod
    def _ring(qcx: int, qcy: int, k: int):
        """Yield cell coords at Chebyshev distance exactly ``k`` from (qcx, qcy).

        Generates the ring's ~8k perimeter cells directly in O(k); the previous
        version scanned the full (2k+1)^2 box with an abs/max filter (O(k^2) per
        ring), which made nearest-neighbour queries on large/sparse maps explode to
        ~O(K^3) (hundreds of millions of abs/max calls). The cell *set* is identical,
        and nearest()'s result is order-independent (min distance, ties by index).
        """
        if k == 0:
            yield (qcx, qcy)
            return
        for dx in range(-k, k + 1):  # bottom + top rows (full width, includes corners)
            yield (qcx + dx, qcy - k)
            yield (qcx + dx, qcy + k)
        for dy in range(-k + 1, k):  # left + right columns (corners already emitted)
            yield (qcx - k, qcy + dy)
            yield (qcx + k, qcy + dy)
