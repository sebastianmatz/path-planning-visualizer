from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple, Union


from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

from ..types import Point, Edge
from ..geometry import blend_float_paths, resample_float_path_fixed_count


class ImageCanvas(QLabel):
    """Canvas for displaying the map and visualization.
    
    Handles image display, mouse interaction for setting start/goal,
    and overlay rendering for algorithm visualization.
    
    Attributes:
        base_pixmap: Original map image
        overlay: Transparent layer for drawing algorithm visualization
        scale: Current display scale factor
        offset_x: X offset for centered display
        offset_y: Y offset for centered display
        start: Start point (if set)
        goal: Goal point (if set)
        pick_mode: Current point picking mode ('start', 'goal', or None)
        on_point_picked: Callback for when a point is picked
        is_point_valid: Callback to validate point selection
        highlights: Temporary highlights for visualization
        current_path: Current best path for live display
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        # Image state
        self.base_pixmap: Optional[QPixmap] = None
        self.overlay: Optional[QPixmap] = None
        self.scale: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0
        self._cached_disp_size: Optional[Tuple[int, int]] = None
        
        # Point selection state
        self.start: Optional[Point] = None
        self.goal: Optional[Point] = None
        self.pick_mode: Optional[str] = "start"

        # Map-editing state
        self.edit_mode: bool = False

        # Callbacks
        self.on_point_picked: Optional[Callable[[str, Point], None]] = None
        self.is_point_valid: Optional[Callable[[Point], bool]] = None
        self.on_paint: Optional[Callable[[Point, bool], None]] = None  # (point, is_erase)
        
        # Visualization state
        self.highlights: List[Tuple[int, int, int]] = []  # (x, y, alpha)
        self.rejected_highlights: List[Tuple[int, int, int]] = []  # (x, y, alpha)
        self.edge_highlights: List[Tuple[int, int, int, int, int]] = []  # (x1, y1, x2, y2, alpha)
        self.current_tree_edges: List[Edge] = []
        self.current_path: List[Tuple[float, float]] = []
        self.previous_path: List[Tuple[float, float]] = []
        self.reference_path: List[Tuple[float, float]] = []
        self.reference_path_color = QColor(130, 170, 255, 120)
        self.current_path_style: str = "default"
        self.current_path_focus: Optional[Tuple[float, float]] = None
        self.animated_path: List[Tuple[float, float]] = []
        self.path_animation_from: List[Tuple[float, float]] = []
        self.path_animation_to: List[Tuple[float, float]] = []
        self.path_animation_start_time: float = 0.0
        self.path_animation_duration_s: float = 0.14
        self.path_animation_timer = QTimer(self)
        self.path_animation_timer.setInterval(16)
        self.path_animation_timer.timeout.connect(self._advance_path_animation)
        self.path_history: List[Tuple[List[Tuple[float, float]], str, int]] = []

    def _make_pen(self, color: Union[Qt.GlobalColor, QColor], width: int) -> QPen:
        """Create a pen with rounded joins for cleaner path rendering."""
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return pen

    def _draw_polyline(
        self,
        painter: QPainter,
        path: List[Tuple[Union[int, float], Union[int, float]]],
        color: Union[Qt.GlobalColor, QColor],
        width: int,
        pen_style: Qt.PenStyle = Qt.PenStyle.SolidLine,
    ) -> None:
        """Draw a path with anti-aliasing and rounded joins."""
        if len(path) < 2:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = self._make_pen(color, width)
        pen.setStyle(pen_style)
        painter.setPen(pen)
        for i in range(len(path) - 1):
            p1 = QPointF(float(path[i][0]), float(path[i][1]))
            p2 = QPointF(float(path[i + 1][0]), float(path[i + 1][1]))
            painter.drawLine(p1, p2)

    def _draw_live_path(self, painter: QPainter) -> None:
        """Draw the current live path with an algorithm-specific visual style."""
        if len(self.reference_path) >= 2:
            self._draw_polyline(
                painter,
                self.reference_path,
                self.reference_path_color,
                2,
                pen_style=Qt.PenStyle.DashLine,
            )

        render_path = self.current_path
        if len(render_path) < 2:
            return

        if self.current_path_style == "optimizer":
            history_color = QColor(85, 200, 255, 70)
            prev_color = QColor(210, 245, 255, 135)
            connector_color = QColor(65, 210, 255, 110)
            main_glow = QColor(35, 185, 255, 70)
            main_core = QColor(235, 248, 255, 245)
        elif self.current_path_style == "optimizer_post":
            history_color = QColor(255, 155, 215, 78)
            prev_color = QColor(255, 232, 244, 145)
            connector_color = QColor(255, 120, 205, 110)
            main_glow = QColor(255, 95, 185, 78)
            main_core = QColor(255, 246, 250, 245)
        else:
            self._draw_polyline(painter, render_path, Qt.GlobalColor.yellow, 4)
            return

        # Recent full trajectories remain visible briefly as contour lines.
        for history_path, history_style, alpha in self.path_history:
            if len(history_path) < 2 or alpha <= 0:
                continue
            if history_style != self.current_path_style:
                continue
            self._draw_polyline(
                painter,
                history_path,
                QColor(history_color.red(), history_color.green(), history_color.blue(), min(alpha, 120)),
                2,
            )

        # Show the immediately previous full trajectory as a dashed comparison path.
        if len(self.previous_path) >= 2:
            self._draw_polyline(
                painter,
                self.previous_path,
                prev_color,
                2,
                pen_style=Qt.PenStyle.DashLine,
            )

            # Visualize the whole-trajectory deformation between the last and current iteration.
            sample_count = max(14, min(40, max(len(self.previous_path), len(render_path))))
            prev_samples = resample_float_path_fixed_count(self.previous_path, sample_count)
            curr_samples = resample_float_path_fixed_count(render_path, sample_count)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(self._make_pen(connector_color, 1))
            painter.setBrush(QBrush(QColor(connector_color.red(), connector_color.green(), connector_color.blue(), 95)))
            for (ax, ay), (bx, by) in zip(prev_samples[1:-1], curr_samples[1:-1]):
                painter.drawLine(QPointF(float(ax), float(ay)), QPointF(float(bx), float(by)))
                painter.drawEllipse(QPointF(float(bx), float(by)), 1.6, 1.6)

        # Current full trajectory
        self._draw_polyline(painter, render_path, main_glow, 8)
        self._draw_polyline(painter, render_path, main_core, 3)
    
    def set_image(self, qpix: QPixmap):
        self.base_pixmap = qpix
        self.overlay = QPixmap(qpix.size())
        self.overlay.fill(Qt.GlobalColor.transparent)
        self.path_animation_timer.stop()
        self.start = None
        self.goal = None
        self.pick_mode = "start"
        self.highlights = []
        self.rejected_highlights = []
        self.edge_highlights = []
        self.current_tree_edges = []
        self.current_path = []
        self.previous_path = []
        self.reference_path = []
        self.current_path_style = "default"
        self.current_path_focus = None
        self.animated_path = []
        self.path_animation_from = []
        self.path_animation_to = []
        self.path_history = []
        self._cached_disp_size = None  # Recompute on new image
        self._update_display()
    
    def reset_overlay(self):
        if self.base_pixmap is None:
            return
        self.overlay = QPixmap(self.base_pixmap.size())
        self.overlay.fill(Qt.GlobalColor.transparent)
        self.path_animation_timer.stop()
        self.highlights = []
        self.rejected_highlights = []
        self.edge_highlights = []
        self.current_tree_edges = []
        self.current_path = []
        self.previous_path = []
        self.reference_path = []
        self.current_path_style = "default"
        self.current_path_focus = None
        self.animated_path = []
        self.path_animation_from = []
        self.path_animation_to = []
        self.path_history = []
        self._update_display()
    
    def _update_display(self):
        if self.base_pixmap is None:
            self.setText("Load an image first.")
            return
        
        w, h = self.width(), self.height()
        bw, bh = self.base_pixmap.width(), self.base_pixmap.height()
        
        s = min(w / bw, h / bh)
        self.scale = s
        disp_w, disp_h = int(bw * s), int(bh * s)
        
        # Cache display size - only recompute on resize, not during animation
        if self._cached_disp_size is None:
            self._cached_disp_size = (disp_w, disp_h)
        else:
            # Use cached size to prevent jitter during animation
            disp_w, disp_h = self._cached_disp_size
        
        self.offset_x = (w - disp_w) // 2
        self.offset_y = (h - disp_h) // 2
        
        composed = QPixmap(bw, bh)
        composed.fill(Qt.GlobalColor.transparent)
        painter = QPainter(composed)
        painter.drawPixmap(0, 0, self.base_pixmap)

        # Draw live tree edges for planners with rewiring/history-sensitive trees.
        if self.current_tree_edges:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(self._make_pen(QColor(0, 0, 255, 100), 1))
            for a, b in self.current_tree_edges:
                painter.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))

        painter.drawPixmap(0, 0, self.overlay)

        # Draw current optimizer/path estimate on top of the tree.
        self._draw_live_path(painter)
        
        # Draw edge highlights
        for (x1, y1, x2, y2, alpha) in self.edge_highlights:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(self._make_pen(QColor(0, 255, 255, alpha), 2))
            painter.drawLine(QPointF(float(x1), float(y1)), QPointF(float(x2), float(y2)))
        
        # Draw rejected highlights
        for (x, y, alpha) in self.rejected_highlights:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 165, 0, alpha)))
            painter.drawEllipse(QPointF(x, y), 4, 4)
        
        # Draw node highlights
        for (x, y, alpha) in self.highlights:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 50, 50, alpha)))
            painter.drawEllipse(QPointF(x, y), 5, 5)
        
        painter.end()
        self.setPixmap(composed.scaled(disp_w, disp_h, Qt.AspectRatioMode.KeepAspectRatio, 
                                       Qt.TransformationMode.SmoothTransformation))
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Recompute display size on resize
        self._cached_disp_size = None
        self._update_display()
    
    def _canvas_to_image(self, x: float, y: float) -> Optional[Point]:
        """Map a canvas-widget coordinate to an image (occupancy) pixel."""
        if self.base_pixmap is None or self.scale <= 0:
            return None
        ix = (x - self.offset_x) / self.scale
        iy = (y - self.offset_y) / self.scale
        if ix < 0 or iy < 0 or ix >= self.base_pixmap.width() or iy >= self.base_pixmap.height():
            return None
        return (int(ix), int(iy))

    def mousePressEvent(self, event):
        if self.base_pixmap is None:
            return

        # Map editing: left draws obstacles, right erases.
        if self.edit_mode:
            if self.on_paint is None:
                return
            if event.button() == Qt.MouseButton.LeftButton:
                erase = False
            elif event.button() == Qt.MouseButton.RightButton:
                erase = True
            else:
                return
            p = self._canvas_to_image(event.position().x(), event.position().y())
            if p is not None:
                self.on_paint(p, erase)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        p = self._canvas_to_image(event.position().x(), event.position().y())
        if p is None:
            return

        # Check if point is on obstacle - if so, ignore click silently
        if self.is_point_valid and not self.is_point_valid(p):
            return

        if self.pick_mode == "start":
            self.start = p
            self.pick_mode = "goal"
            if self.on_point_picked:
                self.on_point_picked("start", p)
            self._draw_marker(p, "start")
        elif self.pick_mode == "goal":
            self.goal = p
            self.pick_mode = None
            if self.on_point_picked:
                self.on_point_picked("goal", p)
            self._draw_marker(p, "goal")

    def mouseMoveEvent(self, event):
        # Drag-painting while editing: left held = draw, right held = erase.
        if not self.edit_mode or self.on_paint is None or self.base_pixmap is None:
            return
        buttons = event.buttons()
        if buttons & Qt.MouseButton.LeftButton:
            erase = False
        elif buttons & Qt.MouseButton.RightButton:
            erase = True
        else:
            return
        p = self._canvas_to_image(event.position().x(), event.position().y())
        if p is not None:
            self.on_paint(p, erase)

    def update_base_image(self, qpix: QPixmap) -> None:
        """Swap the base map image in place, keeping start/goal and the overlay.

        Used by the map editor so painting a stroke does not wipe the markers or
        the current visualization (unlike ``set_image``).
        """
        self.base_pixmap = qpix
        self._update_display()

    def _draw_marker(self, p, kind):
        if self.overlay is None:
            return
        painter = QPainter(self.overlay)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = Qt.GlobalColor.green if kind == "start" else Qt.GlobalColor.red
        painter.setPen(self._make_pen(color, 4))
        painter.drawEllipse(QPointF(p[0], p[1]), 6, 6)
        painter.end()
        self._update_display()
    
    def draw_edge(self, a, b, color=Qt.GlobalColor.blue):
        if self.overlay is None:
            return
        painter = QPainter(self.overlay)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self._make_pen(color, 1))
        painter.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))
        painter.end()
        self.highlights.append((b[0], b[1], 255))
        self.edge_highlights.append((a[0], a[1], b[0], b[1], 255))
        self._update_display()

    def highlight_edge(self, a: Point, b: Point, color: Optional[QColor] = None):
        self.highlights.append((b[0], b[1], 255))
        self.edge_highlights.append((a[0], a[1], b[0], b[1], 255))
        self._update_display()

    def set_current_tree_edges(self, edges: List[Edge]):
        self.current_tree_edges = list(edges)
        self._update_display()

    def clear_current_tree(self):
        self.current_tree_edges = []
        self._update_display()

    def set_reference_path(
        self,
        path: List[Tuple[float, float]],
        color: Optional[QColor] = None,
    ) -> None:
        self.reference_path = list(path)
        if color is not None:
            self.reference_path_color = QColor(color)
        self._update_display()

    def clear_reference_path(self) -> None:
        self.reference_path = []
        self._update_display()

    def _advance_path_animation(self) -> None:
        """Advance the live path tween used for optimizer-style planners."""
        if len(self.path_animation_from) < 2 or len(self.path_animation_to) < 2:
            self.path_animation_timer.stop()
            self.animated_path = list(self.current_path)
            self._update_display()
            return

        elapsed = time.perf_counter() - self.path_animation_start_time
        alpha = min(1.0, elapsed / max(1e-6, self.path_animation_duration_s))
        eased = alpha * alpha * (3.0 - 2.0 * alpha)
        self.animated_path = blend_float_paths(self.path_animation_from, self.path_animation_to, eased)
        self._update_display()

        if alpha >= 1.0:
            self.path_animation_timer.stop()
            self.animated_path = list(self.current_path)
    
    def add_rejected_highlight(self, point):
        self.rejected_highlights.append((point[0], point[1], 255))
    
    def fade_highlights(self, fade_amount=25):
        self.highlights = [(x, y, a - fade_amount) for x, y, a in self.highlights if a - fade_amount > 0]
        self.rejected_highlights = [(x, y, a - fade_amount) for x, y, a in self.rejected_highlights if a - fade_amount > 0]
        edge_fade = fade_amount * 3
        self.edge_highlights = [(x1, y1, x2, y2, a - edge_fade) for x1, y1, x2, y2, a in self.edge_highlights if a - edge_fade > 0]
        self.path_history = [
            (path, style, alpha - fade_amount)
            for path, style, alpha in self.path_history
            if alpha - fade_amount > 0
        ]
    
    def clear_path(self):
        """Clear the current path (for RRT* live updates)."""
        self.path_animation_timer.stop()
        self.current_path = []
        self.previous_path = []
        self.current_path_style = "default"
        self.current_path_focus = None
        self.animated_path = []
        self.path_animation_from = []
        self.path_animation_to = []
        self.path_history = []

    def set_current_path(
        self,
        path: List[Tuple[float, float]],
        style: str = "default",
        focus_point: Optional[Tuple[float, float]] = None,
    ) -> None:
        next_path = list(path)
        current_draw_path = self.animated_path if len(self.animated_path) >= 2 else self.current_path
        if current_draw_path and len(current_draw_path) >= 2:
            self.previous_path = list(current_draw_path)
        else:
            self.previous_path = []

        if style in {"optimizer", "optimizer_post"} and current_draw_path and len(current_draw_path) >= 2:
            self.path_history.append((list(current_draw_path), style, 190))
            self.path_history = self.path_history[-18:]

        self.current_path = next_path
        self.current_path_style = style
        if focus_point is None:
            self.current_path_focus = None
        else:
            self.current_path_focus = (float(focus_point[0]), float(focus_point[1]))

        # Optimizer paths are shown as discrete full-trajectory iterations rather
        # than morphing continuously between them.
        self.path_animation_timer.stop()
        self.path_animation_from = []
        self.path_animation_to = []
        self.animated_path = list(next_path)
        self._update_display()

    def set_optimizer_animation_profile(self, live: bool) -> None:
        """Switch between interactive and process-heavy optimizer animation profiles."""
        if live:
            self.path_animation_duration_s = 0.10
        else:
            self.path_animation_duration_s = 0.16
    
    def draw_path(self, path, permanent=False, color=Qt.GlobalColor.yellow):
        """Draw path. If permanent=True, draw to overlay. Otherwise set as current_path for live display."""
        if len(path) < 2:
            return
        if permanent:
            # Draw permanently to overlay (for final path)
            if self.overlay is None:
                return
            painter = QPainter(self.overlay)
            self._draw_polyline(painter, list(path), color, 4)
            painter.end()
        else:
            self.set_current_path(list(path), style="default")
            return
        self._update_display()
