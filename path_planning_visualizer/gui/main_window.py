from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap, QStandardItem
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..types import Point, OccupancyGrid
from ..geometry import make_distance_field
from ..mapping import (
    blank_occupancy,
    image_to_occupancy,
    occupancy_to_image,
    paint_disk,
)
from ..metrics import PathMetrics, compute_path_metrics
from ..resources import asset_path
from .canvas import ImageCanvas
from .worker import PlannerBuilder
from ..planners.base import BasePlanner, StepResult
from ..planners.registry import (
    ALGORITHM_GROUPS,
    ALGORITHM_INFO,
    ANYTIME_ALGOS,
    AVAILABLE_PLANNERS,
    SAMPLING_BASED_ALGOS,
)
from ..planners.chomp import CHOMPPlanner
from ..planners.pso import PSOPlanner
from ..planners.gpmp import GPMPPlanner
from ..planners.bit_star import BITStarPlanner


class MainWindow(QMainWindow):
    """Main application window for path planning visualization.
    
    Provides the complete GUI including:
    - Map display canvas
    - Algorithm selection and parameters
    - Playback controls (step, run, pause)
    - Status display
    
    Attributes:
        canvas: Image canvas for visualization
        algo_combo: Algorithm selection dropdown
        params_stack: Stacked widget for algorithm parameters
        params_widgets: Dictionary of parameter widgets by algorithm name
        occ: Current occupancy grid
        planner: Current planner instance
        running_algo_name: Name of currently running algorithm
        timer: Timer for animation playback
        is_playing: Whether animation is currently playing
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Path Planning Visualizer {__version__}")
        
        self._setup_canvas()
        self._setup_algorithm_controls()
        self._setup_playback_controls()
        self._setup_status_display()
        self._setup_layout()
        self._setup_state()
        self._connect_signals()
        self._try_load_default_maze()
    
    def _setup_canvas(self) -> None:
        """Initialize the image canvas."""
        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(800, 600)
        self.canvas.on_point_picked = self._on_point_picked
        self.canvas.is_point_valid = self._is_point_on_free_space
    
    def _setup_algorithm_controls(self) -> None:
        """Initialize algorithm selection and parameters."""
        # Algorithm selection with grouped dropdown
        self.algo_combo = QComboBox()
        self._populate_algo_combo()
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)
        
        # Stacked widget for algorithm-specific parameters
        self.params_stack = QStackedWidget()
        self.params_widgets: Dict[str, QWidget] = {}
        for name, planner_class in AVAILABLE_PLANNERS.items():
            widget = planner_class.get_params_widget()
            self.params_widgets[name] = widget
            self.params_stack.addWidget(widget)
        
        # Algorithm info label
        self.lbl_algo_info = QLabel()
        self.lbl_algo_info.setWordWrap(True)
        self.lbl_algo_info.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_algo_info.setStyleSheet("color: #555; font-size: 11px;")
        self._update_algo_info()
    
    def _setup_playback_controls(self) -> None:
        """Initialize playback control buttons and speed slider."""
        # Control buttons
        self.btn_load = QPushButton("Load Image")
        self.btn_reset = QPushButton("Reset")
        self.btn_step = QPushButton("Step")
        self.btn_run = QPushButton("Run")
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setFixedWidth(130)

        # Map-editing controls (collapsed by default behind the "Map Tools" toggle)
        self.btn_edit = QPushButton("Edit Map")
        self.btn_edit.setCheckable(True)
        self.btn_edit.setToolTip("Draw obstacles: left-drag adds walls, right-drag erases")
        self.btn_new_map = QPushButton("New Map")
        self.btn_save_map = QPushButton("Save Map")
        self.btn_save_map.setEnabled(False)
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 80)
        self.spin_brush.setValue(8)
        self.spin_brush.setToolTip("Brush radius in pixels")

        self.btn_map_tools = QPushButton("▸ Map Tools")
        self.btn_map_tools.setCheckable(True)
        self.btn_map_tools.setToolTip("Show or hide the map editing tools")

        for btn in [self.btn_step, self.btn_run, self.btn_pause]:
            btn.setEnabled(False)
        
        # Speed slider: 1-999 = steps/sec, 1000 = MAX (unlimited)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 1000)
        self.speed_slider.setValue(1000)  # Start at max speed
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(100)
        self.speed_label = QLabel("MAX")
        self.speed_label.setFixedWidth(90)
        self.speed_slider.valueChanged.connect(self._update_speed_label)
    
    def _setup_status_display(self) -> None:
        """Initialize status display labels."""
        self.lbl_algorithm = QLabel("-")
        self.lbl_iteration = QLabel("-")
        self.lbl_status_state = QLabel("Idle")
        self.lbl_path_length = QLabel("-")
        self.lbl_min_clearance = QLabel("-")
        self.lbl_mean_clearance = QLabel("-")
        self.lbl_smoothness = QLabel("-")
        self.lbl_stopwatch = QLabel("-")
        self.lbl_total_compute_time = QLabel("-")
        self.lbl_info = QLabel("Load an image, then click START and GOAL.")
        self.lbl_info.setWordWrap(True)
        # Reserve a stable height (~3 lines) and top-align: the info line changes
        # every step during optimization (e.g. the CHOMP status), and letting it
        # rewrap to a different line count would reflow the panel and make the
        # Animation Speed / Status boxes jump.
        self.lbl_info.setMinimumHeight(self.lbl_info.fontMetrics().lineSpacing() * 3)
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lbl_path_length.setToolTip("Geometric path length in pixels. Lower is usually better.")
        self.lbl_min_clearance.setToolTip("Smallest obstacle distance anywhere along the path in pixels. Higher is safer.")
        self.lbl_mean_clearance.setToolTip("Average obstacle distance along the full path in pixels. Higher means the path stays farther from walls overall.")
        self.lbl_smoothness.setToolTip("Average squared turning angle in rad^2 on a uniformly resampled path. Lower is smoother.")
        self.lbl_stopwatch.setToolTip("Planner compute time until the first valid path is found")
        self.lbl_total_compute_time.setToolTip("Accumulated planner compute time for the current run")
        
        # Style for status labels
        status_labels = [
            self.lbl_algorithm, self.lbl_iteration, 
            self.lbl_status_state, self.lbl_path_length, self.lbl_min_clearance,
            self.lbl_mean_clearance, self.lbl_smoothness, self.lbl_stopwatch,
            self.lbl_total_compute_time
        ]
        for lbl in status_labels:
            lbl.setStyleSheet("font-weight: bold;")
    
    def _setup_layout(self) -> None:
        """Arrange all widgets in the window layout."""
        # Algorithm group box
        algo_box = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout()
        algo_layout.addWidget(self.algo_combo)
        algo_layout.addWidget(self.lbl_algo_info)
        algo_box.setLayout(algo_layout)
        
        # Parameters group box
        params_box = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        params_layout.addWidget(self.params_stack)
        params_box.setLayout(params_layout)
        
        # Speed group box
        speed_box = QGroupBox("Animation Speed")
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        speed_box.setLayout(speed_layout)
        
        # Status group box
        status_box = QGroupBox("Status")
        status_layout = QFormLayout()
        status_layout.addRow("Algorithm:", self.lbl_algorithm)
        status_layout.addRow("Iteration:", self.lbl_iteration)
        status_layout.addRow("State:", self.lbl_status_state)
        status_layout.addRow("Path length:", self.lbl_path_length)
        status_layout.addRow("Min clearance:", self.lbl_min_clearance)
        status_layout.addRow("Mean clearance:", self.lbl_mean_clearance)
        status_layout.addRow("Smoothness:", self.lbl_smoothness)
        status_layout.addRow("Compute time to first path:", self.lbl_stopwatch)
        status_layout.addRow("Total compute time:", self.lbl_total_compute_time)
        status_layout.addRow(self.lbl_info)
        status_box.setLayout(status_layout)
        
        # Button rows
        controls_row1 = QHBoxLayout()
        controls_row1.addWidget(self.btn_load)
        controls_row1.addWidget(self.btn_reset)
        controls_row1.addWidget(self.btn_map_tools)

        # Map-editing tools live in a panel that is collapsed by default and shown
        # only when the "Map Tools" toggle is expanded.
        self.map_tools_container = QWidget()
        map_tools_row = QHBoxLayout()
        map_tools_row.setContentsMargins(0, 0, 0, 0)
        map_tools_row.addWidget(self.btn_edit)
        map_tools_row.addWidget(QLabel("Brush:"))
        map_tools_row.addWidget(self.spin_brush)
        map_tools_row.addWidget(self.btn_new_map)
        map_tools_row.addWidget(self.btn_save_map)
        self.map_tools_container.setLayout(map_tools_row)
        self.map_tools_container.setVisible(False)

        controls_row2 = QHBoxLayout()
        controls_row2.addWidget(self.btn_step)
        controls_row2.addWidget(self.btn_run)
        controls_row2.addWidget(self.btn_pause)

        # Left panel
        left = QVBoxLayout()
        left.addLayout(controls_row1)
        left.addWidget(self.map_tools_container)
        left.addLayout(controls_row2)
        left.addWidget(algo_box)
        left.addWidget(params_box)
        left.addWidget(speed_box)
        left.addWidget(status_box)
        left.addStretch(1)
        
        # Main layout
        root = QHBoxLayout()
        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setMaximumWidth(350)
        root.addWidget(left_wrap, 0)
        root.addWidget(self.canvas, 1)
        
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
    
    def _setup_state(self) -> None:
        """Initialize application state variables."""
        self.occ: Optional[OccupancyGrid] = None
        self.clearance_field: Optional[np.ndarray] = None
        self.planner: Optional[BasePlanner] = None
        self.running_algo_name: Optional[str] = None
        self.stopwatch_start: Optional[float] = None
        self.stopwatch_stopped: bool = False
        self.solve_elapsed: float = 0.0
        self.time_to_first_path: Optional[float] = None
        self._metrics_cache_key: Optional[Tuple[Point, ...]] = None
        self._metrics_cache_value: Optional[PathMetrics] = None
        
        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self._run_tick)
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self._fade_tick)
        # Debounced clearance-field recompute while painting the map
        self._clearance_timer = QTimer()
        self._clearance_timer.setSingleShot(True)
        self._clearance_timer.timeout.connect(self._recompute_clearance)

        # Off-thread planner construction
        self._builder: Optional[PlannerBuilder] = None
        self._build_generation: int = 0
        
        # Playback state
        self.steps_per_tick: int = 60
        self.is_playing: bool = False
        
        # CHOMP optimization state
        self.last_found_path: Optional[List[Point]] = None
        self.last_found_algo: Optional[str] = None
        self.optimizing_from_sampling: bool = False

    def _reset_solver_metrics(self) -> None:
        """Reset runtime metrics for a fresh planning attempt."""
        self.stopwatch_start = None
        self.stopwatch_stopped = False
        self.solve_elapsed = 0.0
        self.time_to_first_path = None
        self._metrics_cache_key = None
        self._metrics_cache_value = None
        self.lbl_stopwatch.setText("-")
        self.lbl_stopwatch.setStyleSheet("font-weight: bold;")
        self.lbl_total_compute_time.setText("-")
        self.lbl_total_compute_time.setStyleSheet("font-weight: bold;")
        self._clear_path_metrics_labels()

    def _record_solver_time(self, elapsed: float) -> None:
        """Accumulate compute time and freeze first-path timing once available."""
        self.solve_elapsed += elapsed
        if self.planner is not None and self.planner.found_path and self.time_to_first_path is None:
            self.time_to_first_path = self.solve_elapsed
            self.stopwatch_stopped = True
        self._update_stopwatch_label()

    def _update_stopwatch_label(self) -> None:
        """Refresh compute-time labels from accumulated solver time."""
        if self.solve_elapsed > 0:
            self.lbl_total_compute_time.setText(f"{self.solve_elapsed:.3f}s")
            total_color = "blue" if self.planner is not None and not self.planner.done and self.is_playing else "black"
            self.lbl_total_compute_time.setStyleSheet(f"font-weight: bold; color: {total_color};")
        else:
            self.lbl_total_compute_time.setText("-")
            self.lbl_total_compute_time.setStyleSheet("font-weight: bold;")

        if self.time_to_first_path is not None:
            self.lbl_stopwatch.setText(f"{self.time_to_first_path:.3f}s")
            self.lbl_stopwatch.setStyleSheet("font-weight: bold; color: green;")
            return

        if self.planner is not None and not self.planner.done and (self.is_playing or self.solve_elapsed > 0):
            self.lbl_stopwatch.setText(f"{self.solve_elapsed:.3f}s")
            self.lbl_stopwatch.setStyleSheet("font-weight: bold; color: blue;")
            return

        self.lbl_stopwatch.setText("-")
        self.lbl_stopwatch.setStyleSheet("font-weight: bold;")

    def _clear_path_metrics_labels(self) -> None:
        """Reset displayed path-quality metrics."""
        self.lbl_path_length.setText("-")
        self.lbl_min_clearance.setText("-")
        self.lbl_mean_clearance.setText("-")
        self.lbl_smoothness.setText("-")

    def _set_path_metrics_labels(self, metrics: Optional[PathMetrics]) -> None:
        """Display path-quality metrics in the status panel."""
        if metrics is None:
            self._clear_path_metrics_labels()
            return

        self.lbl_path_length.setText(f"{metrics.length_px:.1f}px")
        self.lbl_min_clearance.setText(
            "-" if metrics.min_clearance_px is None else f"{metrics.min_clearance_px:.1f}px"
        )
        self.lbl_mean_clearance.setText(
            "-" if metrics.mean_clearance_px is None else f"{metrics.mean_clearance_px:.1f}px"
        )
        self.lbl_smoothness.setText(
            "-" if metrics.smoothness is None else f"{metrics.smoothness:.3f} rad^2"
        )

    def _get_path_metrics(self, path: List[Point]) -> PathMetrics:
        """Compute path metrics with a small cache for repeated UI refreshes."""
        key = tuple(path)
        if key != self._metrics_cache_key:
            self._metrics_cache_key = key
            self._metrics_cache_value = compute_path_metrics(path, self.clearance_field)
        return self._metrics_cache_value
    
    def _connect_signals(self) -> None:
        """Connect all button signals to their handlers."""
        self.btn_load.clicked.connect(self.load_image)
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_step.clicked.connect(self.step_once)
        self.btn_run.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_edit.toggled.connect(self._on_edit_toggled)
        self.btn_new_map.clicked.connect(self.new_map)
        self.btn_save_map.clicked.connect(self.save_map)
        self.btn_map_tools.toggled.connect(self._on_map_tools_toggled)
    
    def _populate_algo_combo(self):
        """Populate algorithm dropdown with grouped items."""
        model = self.algo_combo.model()
        
        first_algo = None
        for group_name, algos in ALGORITHM_GROUPS:
            # Add group header (disabled, styled)
            header = QStandardItem(f"--- {group_name} ---")
            header.setEnabled(False)
            header_font = QFont()
            header_font.setBold(True)
            header.setFont(header_font)
            header.setForeground(QColor(100, 100, 100))
            model.appendRow(header)
            
            # Add algorithms in this group
            for algo in algos:
                if algo in AVAILABLE_PLANNERS:
                    item = QStandardItem(f"    {algo}")
                    item.setData(algo, Qt.ItemDataRole.UserRole)  # Store actual name
                    model.appendRow(item)
                    if first_algo is None:
                        first_algo = algo
        
        # Select first algorithm
        if first_algo:
            for i in range(self.algo_combo.count()):
                if self.algo_combo.itemData(i, Qt.ItemDataRole.UserRole) == first_algo:
                    self.algo_combo.setCurrentIndex(i)
                    break
    
    def _get_selected_algo_name(self) -> str:
        """Get the actual algorithm name from the combo box (without indent/formatting)."""
        idx = self.algo_combo.currentIndex()
        # Try to get from UserRole first
        name = self.algo_combo.itemData(idx, Qt.ItemDataRole.UserRole)
        if name:
            return name
        # Fallback: strip whitespace from text
        return self.algo_combo.currentText().strip()
    
    def _update_algo_info(self):
        """Update the algorithm info label with description and paper citation."""
        algo_name = self._get_selected_algo_name()
        if algo_name in ALGORITHM_INFO:
            desc, paper = ALGORITHM_INFO[algo_name]
            self.lbl_algo_info.setText(f"{desc}<br><b>Paper:</b> {paper}")
        else:
            self.lbl_algo_info.setText("")
    
    def _on_algo_changed(self, name: str):
        """Switch parameter widget when algorithm changes."""
        # Extract actual algo name (without formatting)
        actual_name = name.strip()
        if actual_name.startswith('---'):
            # This is a group header, skip
            return
        
        if actual_name in self.params_widgets:
            self.params_stack.setCurrentWidget(self.params_widgets[actual_name])
        
        # If algorithm changed while a planner exists, invalidate it so a new one is created
        if self.running_algo_name is not None and self.running_algo_name != actual_name:
            # Reset the canvas overlay to clear old visualization
            if self.canvas.start is not None and self.canvas.goal is not None:
                start = self.canvas.start
                goal = self.canvas.goal
                self.canvas.reset_overlay()
                self.canvas.start = start
                self.canvas.goal = goal
                if start:
                    self.canvas._draw_marker(start, kind="start")
                if goal:
                    self.canvas._draw_marker(goal, kind="goal")
            
            self.planner = None
            self.running_algo_name = None
        
        # Update label only if no algorithm is running
        if self.running_algo_name is None:
            self.lbl_algorithm.setText(actual_name)
        
        # Update algorithm info box
        self._update_algo_info()
    
    def _update_status_display(self, state: Optional[str] = None, info: Optional[str] = None):
        """Update the status info box."""
        # Algorithm name - show running algorithm, not dropdown selection
        if self.running_algo_name is not None:
            self.lbl_algorithm.setText(self.running_algo_name)
        else:
            self.lbl_algorithm.setText(self._get_selected_algo_name())
        
        # Iteration count
        if self.planner is not None:
            self.lbl_iteration.setText(str(self.planner.iteration))
        else:
            self.lbl_iteration.setText("-")
        
        # State (Idle, Running, Paused, Done, Found, No Path)
        if state is not None:
            self.lbl_status_state.setText(state)
            # Color coding
            if state == "Found":
                self.lbl_status_state.setStyleSheet("font-weight: bold; color: green;")
            elif state == "No Path":
                self.lbl_status_state.setStyleSheet("font-weight: bold; color: red;")
            elif state == "Running":
                self.lbl_status_state.setStyleSheet("font-weight: bold; color: blue;")
            else:
                self.lbl_status_state.setStyleSheet("font-weight: bold; color: black;")
        
        # Path-quality metrics
        if self.planner is not None and self.planner.found_path:
            path = self.planner.extract_path()
            self._set_path_metrics_labels(self._get_path_metrics(path))
        else:
            self._clear_path_metrics_labels()
        
        # Info message
        if info is not None:
            self.lbl_info.setText(info)
    
    def _get_current_planner_class(self) -> type:
        return AVAILABLE_PLANNERS[self._get_selected_algo_name()]
    
    def _get_current_params_widget(self) -> QWidget:
        return self.params_widgets[self._get_selected_algo_name()]
    
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open maze image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self._load_image_from_path(path)
    
    def _try_load_default_maze(self):
        maze_path = asset_path("maze.png")
        if os.path.exists(maze_path):
            self._load_image_from_path(maze_path)
    
    def _load_image_from_path(self, path: str):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            QMessageBox.critical(self, "Error", "Could not read image.")
            return
        
        self.occ = image_to_occupancy(gray)
        self.clearance_field = make_distance_field(self.occ)
        
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.canvas.set_image(QPixmap.fromImage(qimg.copy()))
        
        self._update_status_display(state="Idle", info="Click START then GOAL on a free (white) pixel.")
        self._set_buttons_enabled(False)
        self.btn_save_map.setEnabled(True)
        self._reset_solver_metrics()
        self._cancel_pending_build()
        self.planner = None
        self.running_algo_name = None  # Clear running algorithm name

    # ------------------------------------------------------------------ map editor
    def _occ_to_pixmap(self, occ: OccupancyGrid) -> QPixmap:
        """Render an occupancy grid to a QPixmap (free=white, obstacle=black)."""
        img = occupancy_to_image(occ)
        h, w = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def new_map(self):
        """Create a blank free map and enter edit mode."""
        if self.occ is not None:
            h, w = self.occ.shape
        else:
            h, w = 400, 400
        self.occ = blank_occupancy(h, w)
        self.clearance_field = make_distance_field(self.occ)
        self.canvas.set_image(self._occ_to_pixmap(self.occ))  # resets start/goal/overlay
        self._cancel_pending_build()
        self.planner = None
        self.running_algo_name = None
        self._reset_solver_metrics()
        self.btn_save_map.setEnabled(True)
        if not self.btn_edit.isChecked():
            self.btn_edit.setChecked(True)  # triggers _on_edit_toggled
        else:
            self._on_edit_toggled(True)
        self._update_status_display(state="Editing", info="Blank map. Left-drag to draw walls.")

    def save_map(self):
        """Save the current occupancy grid as a PNG (free=white, obstacle=black)."""
        if self.occ is None:
            QMessageBox.information(self, "Info", "No map to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save map", "", "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if not cv2.imwrite(path, occupancy_to_image(self.occ)):
            QMessageBox.critical(self, "Error", "Could not write image.")
            return
        self._update_status_display(info=f"Saved map to {os.path.basename(path)}")

    def _on_map_tools_toggled(self, expanded: bool):
        """Show or hide the collapsible map-editing tools panel."""
        self.map_tools_container.setVisible(expanded)
        self.btn_map_tools.setText(("▾ " if expanded else "▸ ") + "Map Tools")

    def _on_edit_toggled(self, checked: bool):
        """Enter/leave map-editing mode."""
        self.canvas.edit_mode = checked
        if checked:
            if not self.btn_map_tools.isChecked():
                self.btn_map_tools.setChecked(True)  # reveal the tools panel
            self.pause()
            self.canvas.on_paint = self._on_paint
            self.canvas.pick_mode = None  # suspend start/goal picking while editing
            self._set_buttons_enabled(False)
            self._update_status_display(
                state="Editing",
                info="Left-drag draws walls, right-drag erases. Untoggle Edit Map when done.",
            )
        else:
            self.canvas.on_paint = None
            if self.canvas.start is None:
                self.canvas.pick_mode = "start"
            elif self.canvas.goal is None:
                self.canvas.pick_mode = "goal"
            else:
                self.canvas.pick_mode = None
            ready = (
                self.occ is not None
                and self.canvas.start is not None
                and self.canvas.goal is not None
            )
            self._set_buttons_enabled(ready)
            info = "Press Step or Run to start." if ready else "Click START then GOAL."
            self._update_status_display(state="Idle", info=info)

    def _on_paint(self, p: Point, erase: bool):
        """Brush callback from the canvas: draw/erase obstacles into the grid."""
        if self.occ is None:
            return
        paint_disk(self.occ, p[0], p[1], self.spin_brush.value(), obstacle=not erase)
        self.canvas.update_base_image(self._occ_to_pixmap(self.occ))
        # The map changed, so any existing or in-flight planner is stale.
        self._cancel_pending_build()
        self.planner = None
        self.running_algo_name = None
        self._validate_endpoints_after_edit()
        # Recompute the (relatively expensive) distance field once painting settles.
        self._clearance_timer.start(120)

    def _validate_endpoints_after_edit(self):
        """Drop start/goal markers that an edit has buried under an obstacle."""
        changed = False
        if self.canvas.start is not None and self.occ[self.canvas.start[1], self.canvas.start[0]]:
            self.canvas.start = None
            changed = True
        if self.canvas.goal is not None and self.occ[self.canvas.goal[1], self.canvas.goal[0]]:
            self.canvas.goal = None
            changed = True
        if changed:
            self.canvas.reset_overlay()
            if self.canvas.start is not None:
                self.canvas._draw_marker(self.canvas.start, "start")
            if self.canvas.goal is not None:
                self.canvas._draw_marker(self.canvas.goal, "goal")

    def _recompute_clearance(self):
        if self.occ is not None:
            self.clearance_field = make_distance_field(self.occ)

    def reset_all(self):
        self.pause()
        self._cancel_pending_build()
        if self.canvas.base_pixmap is not None:
            self.canvas.set_image(self.canvas.base_pixmap)
        self.planner = None
        self.running_algo_name = None  # Clear running algorithm name
        self._reset_solver_metrics()
        info = "Click START then GOAL." if self.occ is not None else "Load an image first."
        self._update_status_display(state="Idle", info=info)
        self._set_buttons_enabled(False)
    
    def _set_buttons_enabled(self, enabled: bool):
        self.btn_step.setEnabled(enabled)
        self.btn_run.setEnabled(enabled)
        self.btn_pause.setEnabled(False)
        if not enabled:
            self.btn_pause.setText("Pause")
    
    def _is_point_on_free_space(self, p: Tuple[int, int]) -> bool:
        """Check if point is on free space (white). Returns False for obstacles."""
        if self.occ is None:
            return False
        return self.occ[p[1], p[0]] == 0
    
    def _on_point_picked(self, which: str, p: Tuple[int, int]):
        if which == "start":
            self._update_status_display(state="Idle", info="Now click GOAL.")
        else:
            self._update_status_display(state="Ready", info="Press Step or Run to start.")
            self._set_buttons_enabled(True)
    
    def _set_preparing_state(self, preparing: bool):
        """Toggle the 'building a planner off-thread' UI state."""
        if preparing:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.btn_step.setEnabled(False)
            self.btn_run.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_edit.setEnabled(False)
            self._update_status_display(state="Preparing", info="Preparing planner...")
        else:
            QApplication.restoreOverrideCursor()
            self.btn_edit.setEnabled(True)

    def _cancel_pending_build(self):
        """Invalidate any in-flight off-thread build (its result is discarded)."""
        self._build_generation += 1
        if self._builder is not None:
            self._set_preparing_state(False)
            self._builder = None

    def _build_planner_async(self, continuation, force_new: bool = False):
        """Ensure a planner exists, building it off the GUI thread if needed.

        ``continuation`` runs on the GUI thread once a planner is ready (or
        immediately if a usable one already exists). Heavy planner construction
        (PRM/FMT*/BIT* on large maps) therefore no longer blocks the UI.
        """
        if self.planner is not None and not force_new:
            continuation()
            return
        if self.occ is None or self.canvas.start is None or self.canvas.goal is None:
            return
        if self._builder is not None and self._builder.isRunning():
            return  # a build is already in flight

        start, goal = self.canvas.start, self.canvas.goal
        if self.occ[start[1], start[0]] or self.occ[goal[1], goal[0]]:
            QMessageBox.critical(self, "Error", "Start or goal is on an obstacle.")
            return

        algo_name = self._get_selected_algo_name()
        planner_class = AVAILABLE_PLANNERS[algo_name]
        params = self.params_widgets[algo_name].get_params()  # read widgets on GUI thread
        occ = self.occ

        def build():
            return planner_class(occ, start, goal, **params)

        self._build_generation += 1
        gen = self._build_generation
        self._set_preparing_state(True)

        builder = PlannerBuilder(build, self)
        self._builder = builder

        def on_ready(planner):
            if gen != self._build_generation:
                return  # stale (map edited or a newer build started)
            self._builder = None
            self._set_preparing_state(False)
            self.planner = planner
            self.running_algo_name = algo_name
            self._reset_solver_metrics()
            self.optimizing_from_sampling = False
            self.btn_pause.setText("Pause")
            self.btn_pause.setEnabled(False)
            continuation()

        def on_failed(message):
            if gen != self._build_generation:
                return
            self._builder = None
            self._set_preparing_state(False)
            self.planner = None
            QMessageBox.critical(self, "Error", message)

        builder.result_ready.connect(on_ready)
        builder.build_failed.connect(on_failed)
        builder.finished.connect(builder.deleteLater)
        builder.start()
    
    def _is_anytime_algorithm(self) -> bool:
        """Check if current algorithm is anytime (continues improving after finding first path)."""
        return self._get_selected_algo_name() in ANYTIME_ALGOS

    def _is_sampling_based_algo(self, algo_name: str) -> bool:
        """Check if given algorithm is sampling-based (eligible for CHOMP optimization)."""
        return algo_name in SAMPLING_BASED_ALGOS

    def _can_offer_chomp(self) -> bool:
        return (
            self.planner is not None
            and self.planner.found_path
            and self.running_algo_name is not None
            and self._is_sampling_based_algo(self.running_algo_name)
            and not self.optimizing_from_sampling
        )

    def _offer_chomp_if_available(self):
        if self._can_offer_chomp():
            self.btn_pause.setText("CHOMP Optimize")
            self.btn_pause.setEnabled(True)
        else:
            self.btn_pause.setText("Pause")
    
    def _prepare_restart(self):
        """Reset the overlay (keeping start/goal markers) and bump the seed.

        The actual (possibly off-thread) planner construction is done by the
        caller via ``_build_planner_async(..., force_new=True)``.
        """
        start = self.canvas.start
        goal = self.canvas.goal

        # Increment seed in the current params widget for a different result
        params_widget = self._get_current_params_widget()
        if hasattr(params_widget, 'spin_seed'):
            current_seed = params_widget.spin_seed.value()
            params_widget.spin_seed.setValue(current_seed + 1)

        # Reset overlay but keep base image
        self.canvas.reset_overlay()

        # Restore start and goal
        self.canvas.start = start
        self.canvas.goal = goal
        self.canvas.pick_mode = None

        # Redraw start and goal markers
        if start:
            self.canvas._draw_marker(start, kind="start")
        if goal:
            self.canvas._draw_marker(goal, kind="goal")

        self.planner = None  # force a fresh build
        self.optimizing_from_sampling = False

    def step_once(self):
        # If planner is done, restart it (new seed), then step once it is built.
        if self.planner is not None and self.planner.done:
            self._prepare_restart()
            self._build_planner_async(self._do_one_step, force_new=True)
            return
        self._build_planner_async(self._do_one_step)

    def _do_one_step(self):
        if self.planner is None:
            return
        step_start = time.perf_counter()
        result = self.planner.step_once()
        self._record_solver_time(time.perf_counter() - step_start)
        self._handle_step_result(result)
        self._check_done()
        if not self.planner.done:
            self._update_status_display(state="Stepping", info=self.planner.get_status())
        if not self.fade_timer.isActive():
            self.fade_timer.start(50)

    def play(self):
        # If Re-Run clicked while running (anytime algorithm), restart
        if self.is_playing and self._is_anytime_algorithm():
            self.timer.stop()
            self._prepare_restart()
            self.btn_run.setText("Run")  # Reset text temporarily
            self._build_planner_async(self._begin_playback, force_new=True)
            return
        # If planner is done, restart it
        if self.planner is not None and self.planner.done:
            self._prepare_restart()
            self._build_planner_async(self._begin_playback, force_new=True)
            return
        self._build_planner_async(self._begin_playback)

    def _begin_playback(self):
        if self.planner is None:
            return
        self.canvas.set_optimizer_animation_profile(live=not isinstance(self.planner, CHOMPPlanner))
        self.is_playing = True
        self._set_running_state()
        self._update_status_display(state="Running", info="Algorithm is running...")
        self._update_stopwatch_label()
        speed = self.speed_slider.value()
        interval_ms = self._compute_timer_interval_ms(speed)
        self.timer.start(interval_ms)
    
    def _set_running_state(self):
        # For anytime algorithms, allow Re-Run while running
        if self._is_anytime_algorithm():
            self.btn_run.setText("Re-Run")
            self.btn_run.setEnabled(True)
        else:
            self.btn_run.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_pause.setEnabled(True)
        self.btn_step.setEnabled(False)
    
    def _on_pause_clicked(self):
        if self.btn_pause.text() == "CHOMP Optimize":
            self.optimize_with_chomp()
        else:
            self.pause()

    def pause(self):
        self.timer.stop()
        self.is_playing = False
        self.btn_run.setText("Run")  # Reset button text
        if self.occ is not None and self.canvas.start is not None and self.canvas.goal is not None:
            self._set_buttons_enabled(True)
        self.btn_pause.setEnabled(False)
        if self.planner is not None and not self.planner.done:
            self._update_status_display(state="Paused", info=self.planner.get_status())

        # If a sampling-based planner has a path, offer CHOMP on the same Pause button
        if self._can_offer_chomp():
            path = self.planner.extract_path()
            if path:
                self.canvas.clear_path()
                self.canvas.draw_path(path, permanent=True, color=Qt.GlobalColor.yellow)
                self.last_found_path = list(path)
                self.last_found_algo = self.running_algo_name
                self._offer_chomp_if_available()

    def optimize_with_chomp(self):
        """Run CHOMP to optimize the last sampling-based path."""
        if self.occ is None or self.canvas.start is None or self.canvas.goal is None:
            return
        if not self._can_offer_chomp():
            QMessageBox.information(self, "Info", "Find a path with a sampling-based planner first.")
            return

        base_path = self.last_found_path or (self.planner.extract_path() if self.planner else None)
        if base_path is None or len(base_path) < 2:
            QMessageBox.information(self, "Info", "No valid path to optimize.")
            return
        base_path = list(base_path)

        # Stop any running timers
        self.timer.stop()
        self.is_playing = False

        chomp_params = self.params_widgets['CHOMP'].get_params()

        # Keep the originally found path visible while CHOMP optimizes over it.
        self.canvas.clear_current_tree()
        self.canvas.clear_path()
        self.canvas.clear_reference_path()
        self.canvas.set_optimizer_animation_profile(live=True)

        self.planner = CHOMPPlanner(
            self.occ,
            self.canvas.start,
            self.canvas.goal,
            init_trajectory=base_path,
            **chomp_params,
        )
        self._reset_solver_metrics()
        self.running_algo_name = "CHOMP"
        self.optimizing_from_sampling = True

        # Start optimization immediately
        self.is_playing = True
        self._set_running_state()
        self._update_status_display(state="Running", info="CHOMP optimizing...")
        speed = self.speed_slider.value()
        interval_ms = self._compute_timer_interval_ms(speed)
        self.timer.start(interval_ms)
    
    def _update_speed_label(self, value: int):
        if value >= 1000:
            self.speed_label.setText("MAX")
        else:
            self.speed_label.setText(f"{value} steps/sec")
        if self.is_playing and self.timer.isActive():
            interval_ms = self._compute_timer_interval_ms(value)
            self.timer.setInterval(interval_ms)

    def _compute_timer_interval_ms(self, speed_value: int) -> int:
        """Return a playback interval suited to the current planner."""
        if isinstance(self.planner, CHOMPPlanner):
            if speed_value >= 1000:
                return 8
            return max(8, 1000 // max(1, speed_value))
        return 1 if speed_value >= 1000 else max(1, 1000 // speed_value)
    
    def _fade_tick(self):
        self.canvas.fade_highlights(fade_amount=30)
        self.canvas._update_display()
        if not (
            self.canvas.highlights
            or self.canvas.rejected_highlights
            or self.canvas.edge_highlights
            or self.canvas.path_history
        ):
            self.fade_timer.stop()
    
    def _run_tick(self):
        if self.planner is None:
            return
        
        # MAX mode: faster fading to keep display clean
        speed = self.speed_slider.value()
        is_chomp = isinstance(self.planner, CHOMPPlanner)
        if is_chomp:
            self.canvas.fade_highlights(fade_amount=18 if self.is_playing else 30)
        elif speed >= 1000 and not is_chomp:
            self.canvas.fade_highlights(fade_amount=120)  # Much faster fade in MAX mode
        else:
            self.canvas.fade_highlights(fade_amount=30 if self.is_playing else 60)
        
        # MAX mode: run many steps per tick for maximum speed, but update display periodically
        if is_chomp:
            num_steps = 1
            display_interval = 1
            tick_time_budget = None
        elif speed >= 1000:
            if isinstance(self.planner, PSOPlanner):
                num_steps = 500  # PSO benefits from larger batches
                display_interval = 100
                tick_time_budget = 0.020  # Faster while still responsive
            else:
                num_steps = 200  # Upper bound in MAX mode
                display_interval = 25  # Update display every N steps
                tick_time_budget = 0.012  # Keep GUI responsive (~12ms work per tick)
        else:
            num_steps = 1 if self.is_playing else self.steps_per_tick
            display_interval = num_steps  # Always update in normal mode
            tick_time_budget = None

        tick_start = time.perf_counter()
        
        for i in range(num_steps):
            step_start = time.perf_counter()
            result = self.planner.step_once()
            self._record_solver_time(time.perf_counter() - step_start)
            self._handle_step_result(result)
            
            # In MAX mode, update display periodically for visual feedback
            if speed >= 1000 and not is_chomp and (i + 1) % display_interval == 0:
                self.canvas.fade_highlights(fade_amount=120)  # Extra fade during updates
                self.canvas._update_display()
                self._update_stopwatch_label()
                self._update_status_display(state="Running", info=self.planner.get_status())
                QApplication.processEvents()  # Allow UI to refresh
            
            if self.planner.done:
                break

            # In MAX mode, do not monopolize the UI thread with long batches.
            if tick_time_budget is not None and (time.perf_counter() - tick_start) >= tick_time_budget:
                break
        
        self.canvas._update_display()
        self._check_done()
    
    def _handle_step_result(self, result: StepResult):
        edge_color = QColor(160, 32, 240) if self.optimizing_from_sampling else Qt.GlobalColor.blue
        is_gpmp = isinstance(self.planner, GPMPPlanner)
        is_chomp = isinstance(self.planner, CHOMPPlanner)
        is_bitstar = isinstance(self.planner, BITStarPlanner)
        # Handle multiple edges (for batch algorithms like FMT*)
        if result.edges and not is_gpmp and not is_chomp and not is_bitstar:
            for edge in result.edges:
                self.canvas.draw_edge(edge[0], edge[1], color=edge_color)
        elif result.edge and not is_gpmp and not is_chomp and not is_bitstar:
            self.canvas.draw_edge(result.edge[0], result.edge[1], color=edge_color)
        elif is_bitstar:
            if self.planner is not None and hasattr(self.planner, "extract_tree_edges"):
                self.canvas.set_current_tree_edges(self.planner.extract_tree_edges())
        if result.rejected_point:
            self.canvas.add_rejected_highlight(result.rejected_point)
            self.canvas._update_display()
        if (is_gpmp or is_chomp) and self.planner is not None:
            display_path = self.planner.extract_display_path()
            if display_path:
                style = "optimizer_post" if self.optimizing_from_sampling else "optimizer"
                self.canvas.set_current_path(display_path, style=style, focus_point=None)
        # For RRT*: always keep the current best path visible
        if self.planner is not None and self.planner.found_path:
            path = self.planner.extract_path()
            if path:
                if not self.optimizing_from_sampling and not is_gpmp and not is_chomp:
                    display_path = self.planner.extract_display_path() if hasattr(self.planner, "extract_display_path") else path
                    self.canvas.set_current_path(list(display_path), style="default")
    
    def _check_done(self):
        if self.planner is None:
            return
        
        if not self.planner.done:
            # Still running - update status
            state = "Running" if self.is_playing else "Paused"
            self._update_status_display(state=state, info=self.planner.get_status())
            return
        
        self.pause()
        if not self.fade_timer.isActive():
            self.fade_timer.start(50)
        
        if not self.planner.found_path:
            self.canvas.clear_current_tree()
            self.canvas.clear_path()
            self._update_stopwatch_label()
            self._update_status_display(state="No Path", info=self.planner.get_status())
            self.btn_pause.setEnabled(False)
            self.btn_pause.setText("Pause")
            return
        
        path = self.planner.extract_path()
        if isinstance(self.planner, BITStarPlanner):
            self.canvas.clear_current_tree()
        self.canvas.clear_path()  # Clear live path
        self.canvas.clear_reference_path()
        display_path = self.planner.extract_display_path() if hasattr(self.planner, "extract_display_path") else path
        if self.optimizing_from_sampling:
            self.canvas.draw_path(display_path, permanent=True, color=QColor(255, 105, 180))
        else:
            self.canvas.draw_path(display_path, permanent=True, color=Qt.GlobalColor.yellow)
            self.last_found_path = list(path)
            self.last_found_algo = self.running_algo_name
        self._update_stopwatch_label()
        self._update_status_display(state="Found", info=self.planner.get_status())

        if self.optimizing_from_sampling:
            self.btn_pause.setEnabled(False)
            self.btn_pause.setText("Pause")
            self.optimizing_from_sampling = False
        else:
            self._offer_chomp_if_available()
