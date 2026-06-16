from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple, Union

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

from ..geometry import blend_float_paths, resample_float_path_fixed_count
from ..types import Edge, Point


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

    # Cap on the transient glow highlights drawn per frame; older ones have faded
    # and their edges are already in the permanent tree layer (keeps paint O(1)).
    _MAX_HIGHLIGHTS = 240

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        # Image state. The scene is drawn vectorially in paintEvent: the base map
        # is scaled on draw and all overlays (tree, path, markers) are painted in
        # image coordinates under a scale transform with *cosmetic* (screen-space)
        # pen widths, so the visualization stays crisp at any map resolution.
        self.base_pixmap: Optional[QPixmap] = None
        self.scale: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.disp_w: int = 0
        self.disp_h: int = 0
        
        # Point selection state
        self.start: Optional[Point] = None
        self.goal: Optional[Point] = None
        self.pick_mode: Optional[str] = "start"

        # Map-editing state
        self.edit_mode: bool = False

        # Drag-to-re-query state (roadmap planners: drag start/goal after completion)
        self.drag_markers_enabled: bool = False
        self._dragging: Optional[str] = None  # 'start' or 'goal' while dragging

        # Callbacks
        self.on_point_picked: Optional[Callable[[str, Point], None]] = None
        self.is_point_valid: Optional[Callable[[Point], bool]] = None
        self.on_paint: Optional[Callable[[Point, bool], None]] = None  # (point, is_erase)
        self.on_marker_dragged: Optional[Callable[[str, Point], None]] = None  # (which, new_point) on release
        self.on_marker_dragging: Optional[Callable[[str, Point], None]] = None  # (which, point) live, per move
        
        # Visualization state
        self.highlights: List[Tuple[int, int, int]] = []  # (x, y, alpha)
        self.rejected_highlights: List[Tuple[int, int, int]] = []  # (x, y, alpha)
        self.node_markers: List[Point] = []  # persistent roadmap milestones (PRM), drawn as dots
        self.edge_highlights: List[Tuple[int, int, int, int, int]] = []  # (x1, y1, x2, y2, alpha)
        self.current_tree_edges: List[Edge] = []        # replaced each step (rewiring planners)
        self._appended_edges: List[Edge] = []           # accumulated incrementally (draw_edge)
        self._appended_edge_color: QColor = QColor(0, 0, 255, 130)
        # Accumulated tree edges are baked into a persistent display-resolution
        # layer so paintEvent blits it in O(1) instead of re-stroking every edge
        # (which is O(edges) and freezes the UI once the tree gets large).
        self._tree_layer: Optional[QPixmap] = None
        self._tree_layer_size: Tuple[int, int] = (0, 0)
        self._tree_layer_count: int = 0
        self._tree_layer_color: QColor = QColor(self._appended_edge_color)
        self._final_path: List[Tuple[float, float]] = []         # permanent found path (draw_path)
        self._final_path_color: QColor = QColor(Qt.GlobalColor.yellow)
        self.legend_optimized: bool = False                      # show the "Optimized" legend row
        self.legend_optimized_color: QColor = QColor(255, 105, 180)
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

    def _make_pen(self, color: Union[Qt.GlobalColor, QColor], width: float) -> QPen:
        """Create a cosmetic pen with rounded joins.

        Cosmetic means ``width`` is in *device* (screen) pixels regardless of the
        painter's scale transform, so line thickness is constant across map sizes.
        """
        pen = QPen(color, width)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return pen

    def _recompute_geometry(self) -> None:
        """Update scale/offset/display size from the current widget and map size."""
        if self.base_pixmap is None:
            return
        w, h = max(1, self.width()), max(1, self.height())
        bw, bh = self.base_pixmap.width(), self.base_pixmap.height()
        s = min(w / bw, h / bh)
        self.scale = s
        self.disp_w, self.disp_h = int(bw * s), int(bh * s)
        self.offset_x = (w - self.disp_w) // 2
        self.offset_y = (h - self.disp_h) // 2

    def _ensure_tree_layer(self) -> Optional[QPixmap]:
        """Return the display-resolution pixmap holding all accumulated tree edges.

        Edges are baked once (incrementally, only the newly appended ones) onto a
        persistent layer drawn with the same scale transform + cosmetic 1-px pen as
        before, so the tree looks identical while ``paintEvent`` can blit it in O(1)
        instead of re-stroking every edge each repaint.
        """
        if self.base_pixmap is None or self.disp_w <= 0 or self.disp_h <= 0:
            return None
        size = (self.disp_w, self.disp_h)
        n = len(self._appended_edges)

        rebuild = (
            self._tree_layer is None
            or self._tree_layer_size != size
            or self._tree_layer_count > n                       # edges were cleared/reset
            or self._tree_layer_color != self._appended_edge_color  # recolor all edges
        )
        if rebuild:
            layer = QPixmap(size[0], size[1])
            layer.fill(Qt.GlobalColor.transparent)
            self._tree_layer = layer
            self._tree_layer_size = size
            self._tree_layer_color = QColor(self._appended_edge_color)
            self._tree_layer_count = 0
        elif self._tree_layer_count == n:
            return self._tree_layer  # already up to date

        if n > self._tree_layer_count:
            painter = QPainter(self._tree_layer)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.scale(self.scale, self.scale)
            painter.setPen(self._make_pen(self._appended_edge_color, 1.0))
            for a, b in self._appended_edges[self._tree_layer_count:]:
                painter.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))
            painter.end()
            self._tree_layer_count = n

        return self._tree_layer

    def _dot(self, painter: QPainter, x: float, y: float, screen_r: float, color: QColor) -> None:
        """Filled circle of a fixed *screen* radius, drawn under the scale transform."""
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        r = screen_r / self.scale if self.scale > 0 else screen_r
        painter.drawEllipse(QPointF(float(x), float(y)), r, r)

    def _ring(self, painter: QPainter, x: float, y: float, screen_r: float,
              color: Union[Qt.GlobalColor, QColor], screen_w: float) -> None:
        """Hollow circle of a fixed *screen* radius (start/goal markers)."""
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(self._make_pen(color, screen_w))
        r = screen_r / self.scale if self.scale > 0 else screen_r
        painter.drawEllipse(QPointF(float(x), float(y)), r, r)

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
            # Solid, full-width: used to keep the original found path visible (and
            # looking unchanged) while an optimizer evolves over it.
            self._draw_polyline(
                painter,
                self.reference_path,
                self.reference_path_color,
                3,
            )

        render_path = self.animated_path if len(self.animated_path) >= 2 else self.current_path
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
            self._draw_polyline(painter, render_path, Qt.GlobalColor.yellow, 3)
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
            dot_r = 1.6 / self.scale if self.scale > 0 else 1.6
            for (ax, ay), (bx, by) in zip(prev_samples[1:-1], curr_samples[1:-1], strict=False):
                painter.drawLine(QPointF(float(ax), float(ay)), QPointF(float(bx), float(by)))
                painter.drawEllipse(QPointF(float(bx), float(by)), dot_r, dot_r)

        # Current full trajectory
        self._draw_polyline(painter, render_path, main_glow, 8)
        self._draw_polyline(painter, render_path, main_core, 3)
    
    def _clear_overlays(self) -> None:
        """Reset all transient visualization state (edges, paths, highlights)."""
        self.path_animation_timer.stop()
        self.legend_optimized = False
        self.highlights = []
        self.rejected_highlights = []
        self.edge_highlights = []
        self.node_markers = []
        self.current_tree_edges = []
        self._appended_edges = []
        self._tree_layer = None
        self._tree_layer_count = 0
        self._final_path = []
        self.current_path = []
        self.previous_path = []
        self.reference_path = []
        self.current_path_style = "default"
        self.current_path_focus = None
        self.animated_path = []
        self.path_animation_from = []
        self.path_animation_to = []
        self.path_history = []

    def set_image(self, qpix: QPixmap):
        self.base_pixmap = qpix
        self.start = None
        self.goal = None
        self.pick_mode = "start"
        self._clear_overlays()
        self._recompute_geometry()
        self.update()

    def reset_overlay(self):
        if self.base_pixmap is None:
            return
        self._clear_overlays()  # keeps start/goal (unlike set_image)
        self.update()
    
    def _update_display(self):
        """Schedule a repaint (drawing happens in paintEvent)."""
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0xEC, 0xEC, 0xEC))

        if self.base_pixmap is None:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load an image first.")
            painter.end()
            return

        self._recompute_geometry()
        s = self.scale

        # Base map, scaled on draw. Crisp (nearest) when enlarging an occupancy
        # grid so walls stay sharp; smooth only when shrinking a large image.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, s < 1.0)
        painter.drawPixmap(
            QRectF(self.offset_x, self.offset_y, self.disp_w, self.disp_h),
            self.base_pixmap,
            QRectF(self.base_pixmap.rect()),
        )

        # Accumulated tree edges: blitted from a persistent display-resolution layer
        # (device space, before the scale transform) so this stays O(1) regardless of
        # how large the tree grows.
        tree_layer = self._ensure_tree_layer()
        if tree_layer is not None and self._appended_edges:
            painter.drawPixmap(self.offset_x, self.offset_y, tree_layer)

        # Overlays are drawn in image coordinates under a scale transform; cosmetic
        # pens and screen-radius dots keep their size constant across map sizes.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(s, s)

        # Tree edges replaced each step (rewiring planners) are drawn live.
        if self.current_tree_edges:
            painter.setPen(self._make_pen(QColor(0, 0, 255, 130), 1.0))
            for a, b in self.current_tree_edges:
                painter.drawLine(QPointF(float(a[0]), float(a[1])), QPointF(float(b[0]), float(b[1])))

        # Roadmap milestones (e.g. PRM samples): persistent dots, drawn under the
        # tree/path so connections render on top of their endpoints.
        for (x, y) in self.node_markers:
            self._dot(painter, x, y, 2.0, QColor(70, 110, 200, 200))

        # Permanent / found path, then the live path estimate on top.
        if len(self._final_path) >= 2:
            self._draw_polyline(painter, self._final_path, self._final_path_color, 3)
        self._draw_live_path(painter)

        # Bounded recent-activity glow: highlights/edge-highlights are short-lived
        # fade effects whose edges already live permanently in the tree layer, so a
        # fast run that appends them quicker than they fade must not let these loops
        # grow O(edges) (that is what froze the UI). Keep only the freshest few.
        if len(self.edge_highlights) > self._MAX_HIGHLIGHTS:
            self.edge_highlights = self.edge_highlights[-self._MAX_HIGHLIGHTS:]
        if len(self.highlights) > self._MAX_HIGHLIGHTS:
            self.highlights = self.highlights[-self._MAX_HIGHLIGHTS:]
        if len(self.rejected_highlights) > self._MAX_HIGHLIGHTS:
            self.rejected_highlights = self.rejected_highlights[-self._MAX_HIGHLIGHTS:]

        # Edge highlights (fading), then rejected and node dots (fixed screen size).
        for (x1, y1, x2, y2, alpha) in self.edge_highlights:
            painter.setPen(self._make_pen(QColor(0, 255, 255, alpha), 2))
            painter.drawLine(QPointF(float(x1), float(y1)), QPointF(float(x2), float(y2)))
        for (x, y, alpha) in self.rejected_highlights:
            self._dot(painter, x, y, 2.5, QColor(255, 165, 0, alpha))
        for (x, y, alpha) in self.highlights:
            self._dot(painter, x, y, 2.5, QColor(255, 50, 50, alpha))

        # Start / goal markers always on top.
        if self.start is not None:
            self._ring(painter, self.start[0], self.start[1], 7.0, Qt.GlobalColor.green, 2.5)
        if self.goal is not None:
            self._ring(painter, self.goal[0], self.goal[1], 7.0, Qt.GlobalColor.red, 2.5)

        painter.restore()
        self._draw_legend(painter)  # device-space overlay, on top of everything
        painter.end()

    def _draw_legend(self, painter: QPainter) -> None:
        """Horizontal color key, placed in the margin *below* the map so it does not
        obstruct it (falls back to the widget's bottom edge if there is no margin).
        Swatches use the exact map colors on a dark chip so all of them are legible.
        """
        items = [
            ("dot", QColor(Qt.GlobalColor.green), "Start"),
            ("dot", QColor(Qt.GlobalColor.red), "Goal"),
            ("line", QColor(0, 0, 255), "Tree"),
            ("line", QColor(Qt.GlobalColor.yellow), "Path"),
        ]
        if self.legend_optimized:  # only once an optimized (CHOMP) path exists
            items.append(("line", self.legend_optimized_color, "Optimized"))

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        fm = painter.fontMetrics()
        sw, gap, item_gap, pad, h = 16, 5, 16, 8, 22
        widths = [sw + gap + fm.horizontalAdvance(label) for _, _, label in items]
        w = pad * 2 + sum(widths) + item_gap * (len(items) - 1)

        ww, wh = self.width(), self.height()
        x = self.offset_x + (self.disp_w - w) / 2.0     # centered under the map
        x = max(4.0, min(x, ww - w - 4.0))
        y = self.offset_y + self.disp_h + 6.0           # just below the map
        if y + h > wh - 4:                              # no bottom margin -> bottom edge
            y = wh - h - 4.0

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(90, 90, 90), 1))
        painter.setBrush(QBrush(QColor(45, 45, 45, 215)))
        painter.drawRoundedRect(QRectF(x, y, w, h), 6, 6)

        cx = x + pad
        cy = y + h / 2.0
        for (kind, color, label), iw in zip(items, widths, strict=True):
            if kind == "dot":
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(cx + sw / 2.0, cy), 4, 4)
            else:
                pen = QPen(color, 3)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QPointF(cx, cy), QPointF(cx + sw, cy))
            painter.setPen(QColor(235, 235, 235))
            painter.drawText(
                QRectF(cx + sw + gap, y, iw - sw - gap + 2, h),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                label,
            )
            cx += iw + item_gap

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recompute_geometry()
        self.update()
    
    def _canvas_to_image(self, x: float, y: float) -> Optional[Point]:
        """Map a canvas-widget coordinate to an image (occupancy) pixel."""
        if self.base_pixmap is None:
            return None
        self._recompute_geometry()  # ensure scale/offset are current for clicks
        if self.scale <= 0:
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

        # Drag-to-re-query: if enabled (roadmap planner finished), grab a marker.
        if self.drag_markers_enabled:
            which = self._marker_at(event.position().x(), event.position().y())
            if which is not None:
                self._dragging = which
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
        # Dragging a start/goal marker (roadmap re-query): follow the cursor live.
        if self._dragging is not None and self.base_pixmap is not None:
            p = self._clamp_to_image(event.position().x(), event.position().y())
            if self._dragging == "start":
                self.start = p
            else:
                self.goal = p
            self.update()
            if self.on_marker_dragging is not None:
                self.on_marker_dragging(self._dragging, p)  # live path update
            return

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

    def mouseReleaseEvent(self, event):
        # Finish a marker drag: hand the dropped position to the re-query callback.
        if self._dragging is None:
            return
        which = self._dragging
        self._dragging = None
        point = self.start if which == "start" else self.goal
        if self.on_marker_dragged is not None and point is not None:
            self.on_marker_dragged(which, point)

    def _image_to_canvas(self, p: Point) -> Tuple[float, float]:
        return (self.offset_x + p[0] * self.scale, self.offset_y + p[1] * self.scale)

    def _marker_at(self, sx: float, sy: float) -> Optional[str]:
        """Return 'start'/'goal' if a screen point is within grab range of a marker."""
        self._recompute_geometry()
        grab = 12.0
        for which, pt in (("start", self.start), ("goal", self.goal)):
            if pt is None:
                continue
            cx, cy = self._image_to_canvas(pt)
            if (sx - cx) ** 2 + (sy - cy) ** 2 <= grab * grab:
                return which
        return None

    def _clamp_to_image(self, x: float, y: float) -> Point:
        """Map a screen point to an image pixel, clamped to the map bounds."""
        self._recompute_geometry()
        s = self.scale if self.scale > 0 else 1.0
        ix = int((x - self.offset_x) / s)
        iy = int((y - self.offset_y) / s)
        ix = max(0, min(ix, self.base_pixmap.width() - 1))
        iy = max(0, min(iy, self.base_pixmap.height() - 1))
        return (ix, iy)

    def update_base_image(self, qpix: QPixmap) -> None:
        """Swap the base map image in place, keeping start/goal and the overlays.

        Used by the map editor so painting a stroke does not wipe the markers or
        the current visualization (unlike ``set_image``).
        """
        self.base_pixmap = qpix
        self._recompute_geometry()
        self.update()

    def _draw_marker(self, p, kind):
        # Markers are rendered from self.start / self.goal in paintEvent; this just
        # requests a repaint (kept for API compatibility with the main window).
        self.update()

    def set_appended_edges(self, edges: List[Edge], color: Optional[QColor] = None):
        """Replace the accumulated tree/roadmap edges wholesale and rebuild the layer.

        Used to redraw a roadmap after a re-query (drag start/goal): the stale previous
        query connections are dropped and the current graph is re-baked once.
        """
        self._appended_edges = list(edges)
        if color is not None:
            self._appended_edge_color = QColor(color)
        self._tree_layer = None
        self._tree_layer_count = 0
        self.update()

    def draw_edge(self, a, b, color=Qt.GlobalColor.blue):
        self._appended_edges.append((a, b))
        self._appended_edge_color = QColor(color)
        self.highlights.append((b[0], b[1], 255))
        self.edge_highlights.append((a[0], a[1], b[0], b[1], 255))
        self.update()

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

    def add_node_marker(self, point: Point):
        """Record a persistent roadmap milestone (drawn as a small dot)."""
        self.node_markers.append((int(point[0]), int(point[1])))
        self.update()
    
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
        animate: bool = False,
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

        # Optimizer iterations are shown as discrete frames; only an explicit
        # animate=True (e.g. the final converged frame) morphs smoothly to the new
        # path, so the optimization doesn't visibly "jump" at the end.
        if animate and len(current_draw_path) >= 2 and len(next_path) >= 2:
            self.path_animation_from = list(current_draw_path)
            self.path_animation_to = list(next_path)
            self.animated_path = list(current_draw_path)
            self.path_animation_start_time = time.perf_counter()
            self.path_animation_timer.start()
        else:
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
        """Draw a path. ``permanent=True`` keeps it as the final/found path;
        otherwise it is shown as the live ``current_path``."""
        if len(path) < 2:
            return
        if permanent:
            self._final_path = list(path)
            self._final_path_color = QColor(color)
            self.update()
        else:
            self.set_current_path(list(path), style="default")

    def clear_final_path(self) -> None:
        """Drop the permanent/found path (e.g. before showing it as a reference)."""
        self._final_path = []
        self.update()
