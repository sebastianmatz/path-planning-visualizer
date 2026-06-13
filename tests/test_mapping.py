"""Tests for the pure occupancy-grid map helpers (no Qt)."""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.mapping import (
    blank_occupancy,
    image_to_occupancy,
    occupancy_to_image,
    paint_disk,
)


def test_blank_occupancy_shape_and_free():
    occ = blank_occupancy(40, 60)
    assert occ.shape == (40, 60)
    assert occ.dtype == bool
    assert not occ.any()


def test_paint_disk_draws_filled_disk():
    occ = blank_occupancy(50, 50)
    paint_disk(occ, 25, 25, 5, obstacle=True)
    # Centre and a point within the radius are obstacles; a far corner is free.
    assert occ[25, 25]
    assert occ[25, 29]  # 4px to the right, within radius 5
    assert not occ[25, 31]  # 6px away, outside radius 5
    assert not occ[0, 0]
    # Disk is roughly pi r^2 in area.
    painted = int(occ.sum())
    assert 60 <= painted <= 100  # pi*25 ~= 78


def test_paint_disk_erases():
    occ = np.ones((30, 30), dtype=bool)
    paint_disk(occ, 15, 15, 4, obstacle=False)
    assert not occ[15, 15]
    assert occ[0, 0]  # untouched obstacle remains


def test_paint_disk_clips_to_bounds():
    occ = blank_occupancy(20, 20)
    # Centre near a corner with a large radius must not raise and must clip.
    paint_disk(occ, 1, 1, 10, obstacle=True)
    assert occ[0, 0]
    assert occ.shape == (20, 20)
    # Fully out-of-bounds centre is a no-op.
    before = occ.copy()
    paint_disk(occ, 100, 100, 5, obstacle=True)
    assert np.array_equal(occ, before)


def test_occupancy_image_roundtrip():
    occ = blank_occupancy(30, 40)
    paint_disk(occ, 10, 10, 6, obstacle=True)
    paint_disk(occ, 25, 20, 4, obstacle=True)
    img = occupancy_to_image(occ)
    assert img.dtype == np.uint8
    assert img[occ].max() == 0          # obstacles are black
    assert img[~occ].min() == 255       # free space is white
    # Majority-free grid round-trips exactly through the loader threshold.
    assert np.array_equal(image_to_occupancy(img), occ)


def test_image_to_occupancy_inverts_mostly_dark():
    # Mostly-dark image: dark is treated as free, the bright minority as obstacle.
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[0:3, 0:3] = 255
    occ = image_to_occupancy(gray)
    assert occ[0, 0]          # bright minority -> obstacle
    assert not occ[10, 10]    # dark majority -> free
