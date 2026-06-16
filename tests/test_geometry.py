"""Collision-geometry regressions, especially obstacle-corner grazing.

The collision check rasterizes a segment to grid cells. Fixed-rate sampling
(``max(dx, dy)`` points) could round *around* the corner cell of a diagonal edge
and report a colliding edge as clear — which let BIT* (the one planner relying on
the default sampling) return paths that clipped wall corners on real maps. The
rasterization is now an exact grid traversal that never skips a touched cell.
"""
from __future__ import annotations

import numpy as np

from path_planning_visualizer.geometry import line_collision_free, segment_points


def test_segment_points_includes_grazed_corner_cell():
    # The exact edge that slipped through BIT*'s check: it crosses wall cell (30,39).
    cells = segment_points((29, 39), (40, 46))
    assert (30, 39) in cells


def test_segment_points_has_no_gaps():
    # Consecutive traversed cells must be adjacent (no skipped cell on any segment).
    for a, b in [((0, 0), (40, 46)), ((3, 50), (57, 2)), ((10, 10), (10, 40)), ((5, 5), (40, 6))]:
        cells = segment_points(a, b)
        for (x0, y0), (x1, y1) in zip(cells, cells[1:], strict=False):
            assert abs(x1 - x0) <= 1 and abs(y1 - y0) <= 1


def test_line_collision_free_catches_corner_graze():
    occ = np.zeros((60, 60), dtype=bool)
    occ[0:40, 30] = True  # wall column x=30 for y in [0,40)
    # Diagonal edge that grazes the top corner of the wall at (30,39).
    assert not line_collision_free((29, 39), (40, 46), occ)
    # ...regardless of an explicit (now-ignored) sample count.
    assert not line_collision_free((29, 39), (40, 46), occ, samples=4)


def test_line_collision_free_allows_genuinely_clear_diagonal():
    # A diagonal whose obstacle lies off the line must stay passable (no over-blocking).
    occ = np.zeros((5, 5), dtype=bool)
    occ[0, 1] = True  # obstacle at (1,0); the line (0,0)->(2,2) does not touch it
    assert line_collision_free((0, 0), (2, 2), occ)


def test_line_collision_free_blocks_straight_through_wall():
    occ = np.zeros((60, 60), dtype=bool)
    occ[0:40, 30] = True
    assert not line_collision_free((20, 10), (40, 10), occ)  # horizontal through the wall
    assert line_collision_free((20, 50), (40, 50), occ)      # below the wall (gap) -> clear
