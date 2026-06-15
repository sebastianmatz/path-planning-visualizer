"""Light smoke tests for the ImageCanvas widget (no pixel assertions)."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication

from path_planning_visualizer.gui.canvas import ImageCanvas


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _pixmap(size: int = 20) -> QPixmap:
    img = QImage(size, size, QImage.Format.Format_RGB888)
    img.fill(Qt.GlobalColor.white)
    return QPixmap.fromImage(img)


def test_set_image_resets_state(qapp):
    c = ImageCanvas()
    c.set_image(_pixmap())
    assert c.start is None and c.goal is None
    assert c.pick_mode == "start"
    assert c.current_path == [] and c.current_tree_edges == []


def test_clear_helpers_do_not_crash(qapp):
    c = ImageCanvas()
    c.set_image(_pixmap())
    c.clear_path()
    c.clear_current_tree()
    c.clear_reference_path()
    c.reset_overlay()
    assert c.current_path == []


def test_optimizer_animation_profile_toggle(qapp):
    c = ImageCanvas()
    c.set_image(_pixmap())
    c.set_optimizer_animation_profile(live=True)
    c.set_optimizer_animation_profile(live=False)
