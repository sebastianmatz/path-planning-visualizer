"""Light smoke tests for the ImageCanvas widget (no pixel assertions)."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication

from path_planning_visualizer.gui.canvas import ImageCanvas


class _Evt:
    """Minimal stand-in for QMouseEvent (only what the handlers read)."""

    def __init__(self, x: float, y: float, button=Qt.MouseButton.LeftButton):
        self._p = QPointF(x, y)
        self._b = button

    def button(self):
        return self._b

    def buttons(self):
        return self._b

    def position(self):
        return self._p


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


def test_set_appended_edges_replaces_and_invalidates_layer(qapp):
    c = ImageCanvas()
    c.set_image(_pixmap(40))
    c.draw_edge((1, 1), (2, 2))
    c.draw_edge((3, 3), (4, 4))
    c.set_appended_edges([((5, 5), (6, 6))])
    assert c._appended_edges == [((5, 5), (6, 6))]
    assert c._tree_layer is None  # layer invalidated so it rebuilds from the new set


def test_marker_drag_interaction(qapp):
    c = ImageCanvas()
    c.resize(200, 200)
    c.set_image(_pixmap(40))
    c._recompute_geometry()
    c.start = (5, 5)
    c.goal = (30, 30)
    c.pick_mode = None  # post-completion state (start/goal already placed)
    captured = []
    c.on_marker_dragged = lambda which, pt: captured.append((which, pt))

    gx, gy = c._image_to_canvas((30, 30))
    # Drag disabled: pressing the marker does nothing.
    c.mousePressEvent(_Evt(gx, gy))
    assert c._dragging is None

    # Enable drag mode: press grabs the goal marker, move follows, release fires callback.
    c.drag_markers_enabled = True
    c.mousePressEvent(_Evt(gx, gy))
    assert c._dragging == "goal"
    nx, ny = c._image_to_canvas((10, 10))
    c.mouseMoveEvent(_Evt(nx, ny))
    assert c.goal == (10, 10)
    c.mouseReleaseEvent(_Evt(nx, ny))
    assert c._dragging is None
    assert captured == [("goal", (10, 10))]
