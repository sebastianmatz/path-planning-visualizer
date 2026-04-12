"""
Path Planning Visualizer Beta (0.1.0b3)
=============================

Interactive desktop application for exploring and comparing path-planning
algorithms on occupancy-grid maps.

Supported algorithms:
- Sampling-Based: RRT, RRT-Connect, BiTRRT, KPIECE, RRT*, PRM, SBL, FMT*, BIT*
- Graph Search: A*, Dijkstra
- Potential Field: APF
- Trajectory Optimization: CHOMP, STOMP, TrajOpt, ITOMP, GPMP
- Metaheuristic: PSO, Genetic

Usage:
    python path_planning_visualizer.py
"""

from __future__ import annotations

import heapq
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import cv2
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QStandardItem,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


# =============================================================================
# Type Aliases
# =============================================================================

Point = Tuple[int, int]
FloatPoint = Tuple[float, float]
Edge = Tuple[Point, Point]
OccupancyGrid = NDArray[np.bool_]

# =============================================================================
# Utility Functions
# =============================================================================

def dist(a: Point, b: Point) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        a: First point (x, y)
        b: Second point (x, y)
        
    Returns:
        Euclidean distance between a and b
    """
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def l1_dist(a: Point, b: Point) -> float:
    """Calculate Manhattan distance between two points."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def linf_dist(a: Point, b: Point) -> float:
    """Calculate Chebyshev / L-infinity distance between two points."""
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


def round_point(p: FloatPoint) -> Point:
    """Round a continuous state to the occupancy-grid pixel used by the UI."""
    return (int(round(p[0])), int(round(p[1])))


def select_holonomic_input(from_state: FloatPoint, to_state: FloatPoint, delta_t: float) -> FloatPoint:
    """Return the paper-style holonomic control that best moves toward ``to_state``.

    This specializes LaValle's ``SELECT_INPUT`` step to the simple holonomic model
    ``x_dot = u`` with ``||u|| <= 1`` over a fixed integration interval ``delta_t``.
    """
    dx = float(to_state[0] - from_state[0])
    dy = float(to_state[1] - from_state[1])
    distance = float(np.hypot(dx, dy))
    if distance <= 1e-12 or delta_t <= 1e-12:
        return (0.0, 0.0)
    if distance <= delta_t:
        return (dx / delta_t, dy / delta_t)
    return (dx / distance, dy / distance)


def integrate_holonomic_state(state: FloatPoint, control: FloatPoint, delta_t: float) -> FloatPoint:
    """Integrate the holonomic state equation ``x_dot = u`` for one fixed step."""
    return (
        float(state[0] + delta_t * control[0]),
        float(state[1] + delta_t * control[1]),
    )


def steer(from_pt: Point, to_pt: Point, step: float) -> Point:
    """Move from from_pt towards to_pt by at most step distance.
    
    Args:
        from_pt: Starting point
        to_pt: Target point
        step: Maximum distance to move
        
    Returns:
        New point, moved at most step distance towards to_pt
    """
    d = dist(from_pt, to_pt)
    if d <= step:
        return (int(to_pt[0]), int(to_pt[1]))
    ux = (to_pt[0] - from_pt[0]) / d
    uy = (to_pt[1] - from_pt[1]) / d
    return (int(from_pt[0] + ux * step), int(from_pt[1] + uy * step))


def clamp_point(p: Point, w: int, h: int) -> Point:
    """Clamp point coordinates to image boundaries.
    
    Args:
        p: Point to clamp
        w: Image width
        h: Image height
        
    Returns:
        Point with coordinates clamped to [0, w-1] x [0, h-1]
    """
    x = int(np.clip(p[0], 0, w - 1))
    y = int(np.clip(p[1], 0, h - 1))
    return (x, y)


def segment_points(a: Point, b: Point, samples: Optional[int] = None) -> List[Point]:
    """Return rasterized sample points along a segment.

    The default sampling is adaptive to segment length so collision checks do not
    become looser on long segments.
    """
    dx = abs(int(b[0]) - int(a[0]))
    dy = abs(int(b[1]) - int(a[1]))
    adaptive_samples = max(dx, dy)
    total_samples = max(1, adaptive_samples, samples or 0)

    pts: List[Point] = []
    for i in range(total_samples + 1):
        t = i / total_samples
        x = int(round(a[0] + t * (b[0] - a[0])))
        y = int(round(a[1] + t * (b[1] - a[1])))
        p = (x, y)
        if not pts or pts[-1] != p:
            pts.append(p)
    return pts


def line_collision_free(
    a: Point, 
    b: Point, 
    occ: OccupancyGrid, 
    samples: Optional[int] = None
) -> bool:
    """Check if line segment from a to b is collision-free.
    
    Uses adaptive rasterization to cover long segments robustly.
    
    Args:
        a: Start point of line segment
        b: End point of line segment
        occ: Occupancy grid (True = obstacle)
        samples: Number of points to check along the line
        
    Returns:
        True if the entire line is collision-free
    """
    h, w = occ.shape
    for x, y in segment_points(a, b, samples=samples):
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        if occ[y, x]:
            return False
    return True


def compute_path_length(path: List[Point]) -> float:
    """Return geometric path length in pixels."""
    if len(path) < 2:
        return 0.0
    return float(sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1)))


def compute_path_min_clearance(path: List[Point], clearance_field: np.ndarray) -> float:
    """Return the minimum obstacle clearance along a path in pixels."""
    if len(path) < 2:
        return 0.0

    min_clearance = float("inf")
    for i in range(len(path) - 1):
        for x, y in segment_points(path[i], path[i + 1]):
            min_clearance = min(min_clearance, float(clearance_field[y, x]))

    return 0.0 if min_clearance == float("inf") else min_clearance


def iter_path_pixels(path: List[Point]) -> List[Point]:
    """Return deduplicated rasterized pixels along a path."""
    if not path:
        return []
    if len(path) == 1:
        return [path[0]]

    pts: List[Point] = []
    for i in range(len(path) - 1):
        for p in segment_points(path[i], path[i + 1]):
            if not pts or pts[-1] != p:
                pts.append(p)
    return pts


def compute_path_mean_clearance(path: List[Point], clearance_field: np.ndarray) -> float:
    """Return the mean obstacle clearance along a path in pixels."""
    pixels = iter_path_pixels(path)
    if not pixels:
        return 0.0
    values = [float(clearance_field[y, x]) for x, y in pixels]
    return float(np.mean(values))


def resample_path_points(path: List[Point], spacing: float = 4.0) -> List[Tuple[float, float]]:
    """Resample a path polyline with approximately uniform spacing."""
    if not path:
        return []
    if len(path) == 1:
        return [(float(path[0][0]), float(path[0][1]))]

    pts = np.array(path, dtype=np.float64)
    deltas = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    total_len = float(np.sum(seg_lens))
    if total_len <= 1e-6:
        return [(float(path[0][0]), float(path[0][1]))]

    sample_count = max(2, int(np.ceil(total_len / max(1e-6, spacing))) + 1)
    targets = np.linspace(0.0, total_len, sample_count)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    resampled = np.zeros((sample_count, 2), dtype=np.float64)

    seg_idx = 0
    for i, t in enumerate(targets):
        while seg_idx < len(seg_lens) - 1 and t > cum[seg_idx + 1]:
            seg_idx += 1
        seg_len = seg_lens[seg_idx]
        if seg_len <= 1e-6:
            resampled[i] = pts[seg_idx]
            continue
        local_t = (t - cum[seg_idx]) / seg_len
        resampled[i] = pts[seg_idx] + local_t * deltas[seg_idx]

    return [(float(p[0]), float(p[1])) for p in resampled]


def compute_path_smoothness(path: List[Point], spacing: float = 4.0) -> float:
    """Return an average squared turning-angle smoothness score in rad^2.

    Lower is smoother. The path is resampled first so the metric is less
    dependent on the original waypoint density.
    """
    samples = resample_path_points(path, spacing=spacing)
    if len(samples) < 3:
        return 0.0

    turn_angles: List[float] = []
    for i in range(1, len(samples) - 1):
        p_prev = np.array(samples[i - 1], dtype=np.float64)
        p_curr = np.array(samples[i], dtype=np.float64)
        p_next = np.array(samples[i + 1], dtype=np.float64)

        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue

        cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        turn_angles.append(float(np.arccos(cos_angle)))

    if not turn_angles:
        return 0.0
    return float(np.mean(np.square(turn_angles)))


def compute_path_metrics(path: List[Point], clearance_field: Optional[np.ndarray]) -> PathMetrics:
    """Compute the main path-quality metrics used in the UI."""
    length_px = compute_path_length(path)
    if clearance_field is None:
        return PathMetrics(length_px=length_px, smoothness=compute_path_smoothness(path))

    return PathMetrics(
        length_px=length_px,
        min_clearance_px=compute_path_min_clearance(path, clearance_field),
        mean_clearance_px=compute_path_mean_clearance(path, clearance_field),
        smoothness=compute_path_smoothness(path),
    )


def shortcut_path(path: List[Point], occ: OccupancyGrid, max_passes: int = 2) -> List[Point]:
    """Greedily shortcut a polyline while keeping it collision-free."""
    if len(path) < 3:
        return list(path)

    shortened = list(path)
    for _ in range(max_passes):
        improved = False
        new_path = [shortened[0]]
        i = 0

        while i < len(shortened) - 1:
            next_idx = i + 1
            for j in range(len(shortened) - 1, i + 1, -1):
                if line_collision_free(shortened[i], shortened[j], occ):
                    next_idx = j
                    break

            if next_idx > i + 1:
                improved = True
            new_path.append(shortened[next_idx])
            i = next_idx

        shortened = new_path
        if not improved:
            break

    return shortened


def float_polyline_collision_free(
    path: List[Tuple[float, float]],
    occ: OccupancyGrid,
) -> bool:
    """Check a float polyline by rasterizing each segment onto the occupancy grid."""
    if len(path) < 2:
        return True

    for i in range(len(path) - 1):
        a = (int(round(path[i][0])), int(round(path[i][1])))
        b = (int(round(path[i + 1][0])), int(round(path[i + 1][1])))
        if not line_collision_free(a, b, occ):
            return False
    return True


def smooth_display_path(
    path: List[Point],
    occ: OccupancyGrid,
    spacing: float = 3.0,
    iterations: int = 2,
) -> List[Tuple[float, float]]:
    """Return a denser, cleaner display-only polyline while keeping it collision-free."""
    if not path:
        return []
    if len(path) == 1:
        return [(float(path[0][0]), float(path[0][1]))]

    smoothed = resample_path_points(path, spacing=max(1.0, spacing))
    if len(smoothed) < 3:
        return smoothed

    for _ in range(max(0, iterations)):
        candidate: List[Tuple[float, float]] = [smoothed[0]]
        for i in range(len(smoothed) - 1):
            p0 = np.array(smoothed[i], dtype=np.float64)
            p1 = np.array(smoothed[i + 1], dtype=np.float64)
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            candidate.append((float(q[0]), float(q[1])))
            candidate.append((float(r[0]), float(r[1])))
        candidate.append(smoothed[-1])

        if float_polyline_collision_free(candidate, occ):
            smoothed = candidate
        else:
            break

    return smoothed


# =============================================================================
# Step Result Data Class
# =============================================================================

@dataclass
class StepResult:
    """Result of a single planning step.
    
    Attributes:
        edge: Single edge added this step (from, to)
        edges: Multiple edges for batch algorithms
        rejected_point: Point that was rejected (collision)
        done: Whether planning is complete
        found_path: Whether a valid path was found
        path_improved: Whether path quality improved this step (for anytime algorithms)
    """
    edge: Optional[Edge] = None
    edges: Optional[List[Edge]] = None
    rejected_point: Optional[Point] = None
    done: bool = False
    found_path: bool = False
    path_improved: bool = False


@dataclass(frozen=True)
class PathMetrics:
    """Summary metrics for a final path."""

    length_px: float
    min_clearance_px: Optional[float] = None
    mean_clearance_px: Optional[float] = None
    smoothness: Optional[float] = None


@dataclass
class SBLSegmentState:
    """Lazy collision-check state for an SBL segment."""

    kappa: int = 0
    safe: bool = False


@dataclass
class SBLNode:
    """Milestone node used by SBL."""

    point: Point
    tree_id: int
    parent: Optional[int] = None
    children: Set[int] = field(default_factory=set)


# =============================================================================
# Abstract Base Class for Path Planners
# =============================================================================

class BasePlanner(ABC):
    """Abstract base class for all path planning algorithms.
    
    This class defines the interface that all planners must implement.
    Subclasses provide specific planning algorithms.
    
    Attributes:
        occ: Occupancy grid (True = obstacle)
        h: Grid height
        w: Grid width
        start: Start position
        goal: Goal position
        done: Whether planning is complete
        found_path: Whether a valid path was found
        iteration: Current iteration count
    """
    
    name: str = "Base Planner"
    description: str = "Abstract base class"
    
    def __init__(
        self, 
        occ: OccupancyGrid, 
        start: Point, 
        goal: Point
    ) -> None:
        """Initialize the planner.
        
        Args:
            occ: Occupancy grid (True = obstacle)
            start: Start position
            goal: Goal position
        """
        self.occ = occ
        self.h, self.w = occ.shape
        self.start = start
        self.goal = goal
        self.done = False
        self.found_path = False
        self.iteration = 0
    
    def is_free(self, p: Point) -> bool:
        """Check if a point is in free space.
        
        Args:
            p: Point to check
            
        Returns:
            True if point is within bounds and not on an obstacle
        """
        x, y = p
        return (0 <= x < self.w) and (0 <= y < self.h) and (not self.occ[y, x])
    
    @abstractmethod
    def step_once(self) -> StepResult:
        """Execute one step of the algorithm.
        
        Returns:
            StepResult containing information about this step
        """
        pass
    
    @abstractmethod
    def extract_path(self) -> List[Point]:
        """Extract the found path.
        
        Returns:
            List of points from start to goal, or empty if no path found
        """
        pass
    
    @abstractmethod
    def get_status(self) -> str:
        """Return current status string for display.
        
        Returns:
            Human-readable status string
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_params_widget() -> QWidget:
        """Return a widget for configuring this planner's parameters.
        
        Returns:
            QWidget for parameter configuration
        """
        pass
    
    @staticmethod
    @abstractmethod
    def create_from_params(
        occ: OccupancyGrid, 
        start: Point, 
        goal: Point, 
        params_widget: QWidget
    ) -> 'BasePlanner':
        """Create a planner instance from the params widget.
        
        Args:
            occ: Occupancy grid
            start: Start position
            goal: Goal position
            params_widget: Widget containing parameter values
            
        Returns:
            New planner instance configured with widget parameters
        """
        pass

# =============================================================================
# RRT Implementation
# =============================================================================

class RRTParamsWidget(QWidget):
    """Widget for RRT parameterization."""

    def __init__(self) -> None:
        super().__init__()
        layout = QFormLayout()

        self.spin_delta_t = QDoubleSpinBox()
        self.spin_delta_t.setRange(0.5, 200.0)
        self.spin_delta_t.setSingleStep(0.5)
        self.spin_delta_t.setValue(18.0)
        self.spin_delta_t.setToolTip(
            "Fixed integration interval Delta t for the holonomic NEW_STATE step"
        )

        self.spin_goal_radius = QDoubleSpinBox()
        self.spin_goal_radius.setRange(1.0, 200.0)
        self.spin_goal_radius.setSingleStep(1.0)
        self.spin_goal_radius.setValue(20.0)
        self.spin_goal_radius.setToolTip(
            "Goal-region radius for the single-query adaptation of the paper's tree generator"
        )

        self.spin_goal_bias = QDoubleSpinBox()
        self.spin_goal_bias.setRange(0.0, 1.0)
        self.spin_goal_bias.setSingleStep(0.01)
        self.spin_goal_bias.setDecimals(3)
        self.spin_goal_bias.setValue(0.05)
        self.spin_goal_bias.setToolTip(
            "OMPL-style probability of sampling the exact goal state; 0.05 is the typical default"
        )

        self.spin_col = QSpinBox()
        self.spin_col.setRange(10, 500)
        self.spin_col.setValue(80)
        self.spin_col.setToolTip("Raster samples used to validate each accepted edge on the occupancy grid")

        self.spin_max_vertices = QSpinBox()
        self.spin_max_vertices.setRange(2, 200000)
        self.spin_max_vertices.setValue(25000)
        self.spin_max_vertices.setToolTip(
            "Vertex budget K from GENERATE_RRT(x_init, K, Delta t), including the root"
        )

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Delta t:", self.spin_delta_t)
        layout.addRow("Goal region radius:", self.spin_goal_radius)
        layout.addRow("Goal bias:", self.spin_goal_bias)
        layout.addRow("Collision samples:", self.spin_col)
        layout.addRow("Vertex budget K:", self.spin_max_vertices)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> Dict[str, Union[int, float]]:
        """Get all parameter values as a dictionary."""
        return {
            'delta_t': self.spin_delta_t.value(),
            'goal_region_radius': self.spin_goal_radius.value(),
            'goal_bias': self.spin_goal_bias.value(),
            'collision_samples': self.spin_col.value(),
            'max_vertices': self.spin_max_vertices.value(),
            'seed': self.spin_seed.value(),
        }

class RRTPlanner(BasePlanner):
    """RRT adapted to a 2D holonomic occupancy-grid world.

    The implementation follows the structure of LaValle's
    ``GENERATE_RRT(x_init, K, Delta t)``:
    - ``RANDOM_STATE`` samples uniformly from the bounded workspace
    - ``NEAREST_NEIGHBOR`` uses Euclidean distance in the continuous state
    - ``SELECT_INPUT`` and ``NEW_STATE`` are specialized to ``x_dot = u`` with
      ``||u|| <= 1`` over a fixed ``Delta t``
    - an optional OMPL-style ``goal_bias`` can replace a random sample with the
      exact goal state
    - the UI's single-query path-planning interpretation terminates when a
      vertex enters a goal region around the clicked goal point
    """

    name = "RRT"
    description = "Rapidly-exploring Random Tree implementation after LaValle (1998) with optional OMPL-style goal bias"

    def __init__(
        self,
        occ: OccupancyGrid,
        start: Point,
        goal: Point,
        delta_t: float = 18.0,
        goal_region_radius: float = 20.0,
        goal_bias: float = 0.05,
        collision_samples: int = 80,
        max_vertices: int = 25000,
        seed: int = 1
    ) -> None:
        super().__init__(occ, start, goal)

        self.delta_t = float(max(1e-6, delta_t))
        self.goal_region_radius = float(max(1.0, goal_region_radius))
        self.goal_bias = float(np.clip(goal_bias, 0.0, 1.0))
        self.collision_samples = int(collision_samples)
        self.max_vertices = int(max(2, max_vertices))

        self.rng = np.random.default_rng(seed)
        self.states: List[FloatPoint] = [(float(start[0]), float(start[1]))]
        self.nodes: List[Point] = [start]
        self.node_set: Set[Point] = {start}
        self.parent: List[int] = [-1]
        self.controls: List[Optional[FloatPoint]] = [None]
        self.goal_idx: Optional[int] = None
        if self._goal_reached(self.states[0]):
            self.goal_idx = 0
            self.found_path = True
            self.done = True

    def _sample_state(self) -> FloatPoint:
        """Sample uniformly from the bounded workspace, as in RANDOM_STATE."""
        if self.rng.random() < self.goal_bias:
            return (float(self.goal[0]), float(self.goal[1]))
        return (
            float(self.rng.uniform(0.0, max(0.0, self.w - 1))),
            float(self.rng.uniform(0.0, max(0.0, self.h - 1))),
        )

    def _nearest(self, state: FloatPoint) -> int:
        """Find the nearest existing continuous state to ``state``."""
        pts = np.array(self.states, dtype=np.float64)
        dx = pts[:, 0] - state[0]
        dy = pts[:, 1] - state[1]
        return int(np.argmin(dx * dx + dy * dy))

    def _goal_reached(self, state: FloatPoint) -> bool:
        goal_state = (float(self.goal[0]), float(self.goal[1]))
        return float(np.hypot(state[0] - goal_state[0], state[1] - goal_state[1])) <= self.goal_region_radius

    def _attempt_extension(
        self,
        sample_state: FloatPoint,
    ) -> Tuple[Optional[Point], Optional[Point], Optional[int], Optional[FloatPoint], Optional[FloatPoint]]:
        """Attempt one paper-style EXTEND step toward ``sample_state``."""
        i_near = self._nearest(sample_state)
        x_near = self.states[i_near]
        u = select_holonomic_input(x_near, sample_state, self.delta_t)
        if abs(u[0]) <= 1e-12 and abs(u[1]) <= 1e-12:
            return None, round_point(sample_state), None, None, None

        x_new = integrate_holonomic_state(x_near, u, self.delta_t)
        q_near = self.nodes[i_near]
        q_new = round_point(x_new)

        # Do not insert null motions caused by discretizing the continuous state.
        if q_new == q_near:
            return None, q_new, None, None, None
        if q_new in self.node_set:
            return None, q_new, None, None, None
        if not self.is_free(q_new):
            return None, q_new, None, None, None
        if not line_collision_free(q_near, q_new, self.occ, samples=self.collision_samples):
            return None, q_new, None, None, None
        return q_near, q_new, i_near, u, x_new

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if len(self.nodes) >= self.max_vertices:
            self.done = True
            return StepResult(done=True, found_path=False)

        self.iteration += 1
        sample_state = self._sample_state()
        q_near, q_new, i_near, u, x_new = self._attempt_extension(sample_state)
        if q_near is None or q_new is None or i_near is None or u is None or x_new is None:
            return StepResult(rejected_point=q_new if q_new is not None else round_point(sample_state))

        self.nodes.append(q_new)
        self.node_set.add(q_new)
        self.parent.append(i_near)
        self.states.append(x_new)
        self.controls.append(u)

        if self._goal_reached(x_new):
            self.goal_idx = len(self.nodes) - 1
            self.done = True
            self.found_path = True
            return StepResult(edge=(q_near, q_new), done=True, found_path=True)

        if len(self.nodes) >= self.max_vertices:
            self.done = True
            return StepResult(edge=(q_near, q_new), done=True, found_path=False)

        return StepResult(edge=(q_near, q_new))

    def extract_path(self) -> List[Point]:
        if self.goal_idx is None:
            return []
        path: List[Point] = []
        i = self.goal_idx
        while i != -1:
            path.append(self.nodes[i])
            i = self.parent[i]
        path.reverse()
        return path

    def extract_display_path(self) -> List[Tuple[float, float]]:
        """Return a smoothed display-only version of the current path."""
        path = self.extract_path()
        if len(path) < 3:
            return [(float(p[0]), float(p[1])) for p in path]
        spacing = max(2.0, min(4.0, self.delta_t / 4.0))
        return smooth_display_path(path, self.occ, spacing=spacing, iterations=1)

    def get_status(self) -> str:
        return (
            f"RRT: iter {self.iteration}, vertices {len(self.nodes)}/{self.max_vertices}, "
            f"Delta t {self.delta_t:.1f}, goal radius {self.goal_region_radius:.1f}, "
            f"goal bias {self.goal_bias:.2f}"
        )

    @staticmethod
    def get_params_widget() -> QWidget:
        return RRTParamsWidget()

    @staticmethod
    def create_from_params(
        occ: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        params_widget: QWidget
    ) -> 'RRTPlanner':
        params = params_widget.get_params()
        return RRTPlanner(occ, start, goal, **params)

# ============================================================================
# RRT-Connect Implementation
# ============================================================================

class RRTConnectParamsWidget(QWidget):
    """Widget for RRT-Connect parameter configuration."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 200)
        self.spin_step.setValue(18)
        self.spin_step.setToolTip("Distance for each tree expansion step")
        
        self.spin_col = QSpinBox()
        self.spin_col.setRange(10, 500)
        self.spin_col.setValue(80)
        self.spin_col.setToolTip("Number of samples for collision checking along edges")
        
        self.spin_maxit = QSpinBox()
        self.spin_maxit.setRange(100, 200000)
        self.spin_maxit.setValue(25000)
        self.spin_maxit.setToolTip("Maximum number of iterations")
        
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Step size:", self.spin_step)
        layout.addRow("Collision samples:", self.spin_col)
        layout.addRow("Max iterations:", self.spin_maxit)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'step_size': self.spin_step.value(),
            'collision_samples': self.spin_col.value(),
            'max_iters': self.spin_maxit.value(),
            'seed': self.spin_seed.value(),
        }

class RRTConnectPlanner(BasePlanner):
    """RRT-Connect: Bidirectional RRT that grows trees from start and goal."""
    
    name = "RRT-Connect"
    description = "Bidirectional RRT - grows two trees and connects them"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 step_size: int = 18, collision_samples: int = 80, 
                 max_iters: int = 25000, seed: int = 1):
        super().__init__(occ, start, goal)
        
        self.step_size = step_size
        self.collision_samples = collision_samples
        self.max_iters = max_iters
        
        self.rng = np.random.default_rng(seed)
        
        # Tree from start (tree_a)
        self.nodes_a = [start]
        self.parent_a = [-1]
        
        # Tree from goal (tree_b)
        self.nodes_b = [goal]
        self.parent_b = [-1]
        
        # Connection point indices (when trees connect)
        self.connect_idx_a = None
        self.connect_idx_b = None
        
        # Track which tree is currently extending (swap each iteration)
        self.swap_trees = False
    
    def _nearest(self, nodes: List[Tuple[int, int]], p: Tuple[int, int]) -> int:
        """Find index of nearest node to point p in given tree."""
        pts = np.array(nodes, dtype=np.float32)
        dx = pts[:, 0] - p[0]
        dy = pts[:, 1] - p[1]
        return int(np.argmin(dx * dx + dy * dy))
    
    def _sample(self) -> Tuple[int, int]:
        """Sample a random point in free space."""
        for _ in range(100):
            p = (int(self.rng.integers(0, self.w)), int(self.rng.integers(0, self.h)))
            if self.is_free(p):
                return p
        return (int(self.rng.integers(0, self.w)), int(self.rng.integers(0, self.h)))
    
    def _extend(self, nodes: List[Tuple[int, int]], parent: List[int], 
                target: Tuple[int, int]) -> Tuple[str, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Extend tree towards target.
        Returns: (status, new_point, rejected_point)
        status: 'reached' if target reached, 'advanced' if extended, 'trapped' if blocked
        """
        i_near = self._nearest(nodes, target)
        q_near = nodes[i_near]
        q_new = clamp_point(steer(q_near, target, self.step_size), self.w, self.h)
        
        if not self.is_free(q_new):
            return ('trapped', None, q_new)
        if not line_collision_free(q_near, q_new, self.occ, samples=self.collision_samples):
            return ('trapped', None, q_new)
        
        nodes.append(q_new)
        parent.append(i_near)
        
        if dist(q_new, target) < self.step_size:
            return ('reached', q_new, None)
        return ('advanced', q_new, None)
    
    def _connect(self, nodes: List[Tuple[int, int]], parent: List[int],
                 target: Tuple[int, int]) -> Tuple[str, List[Tuple[Tuple[int,int], Tuple[int,int]]], Optional[Tuple[int, int]]]:
        """
        Try to connect tree to target point by repeatedly extending.
        Returns: (status, edges_added, last_rejected)
        """
        edges = []
        last_rejected = None
        
        while True:
            i_near = self._nearest(nodes, target)
            q_near = nodes[i_near]
            q_new = clamp_point(steer(q_near, target, self.step_size), self.w, self.h)
            
            if not self.is_free(q_new):
                return ('trapped', edges, q_new)
            if not line_collision_free(q_near, q_new, self.occ, samples=self.collision_samples):
                return ('trapped', edges, q_new)
            
            nodes.append(q_new)
            parent.append(i_near)
            edges.append((q_near, q_new))
            
            if dist(q_new, target) < self.step_size:
                return ('reached', edges, None)
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=False)
        
        self.iteration += 1
        
        # Alternate which tree extends and which connects
        if self.swap_trees:
            nodes_extend, parent_extend = self.nodes_b, self.parent_b
            nodes_connect, parent_connect = self.nodes_a, self.parent_a
        else:
            nodes_extend, parent_extend = self.nodes_a, self.parent_a
            nodes_connect, parent_connect = self.nodes_b, self.parent_b
        
        # Sample random point and extend first tree
        q_rand = self._sample()
        status, q_new, rejected = self._extend(nodes_extend, parent_extend, q_rand)
        
        if status == 'trapped':
            self.swap_trees = not self.swap_trees
            return StepResult(rejected_point=rejected)
        
        # Try to connect second tree to the new point
        connect_status, connect_edges, connect_rejected = self._connect(
            nodes_connect, parent_connect, q_new
        )
        
        # Determine the edge to visualize (from extend step)
        i_new = len(nodes_extend) - 1
        i_parent = parent_extend[i_new]
        edge = (nodes_extend[i_parent], q_new)
        
        if connect_status == 'reached':
            # Trees connected!
            if self.swap_trees:
                self.connect_idx_b = len(self.nodes_b) - 1
                self.connect_idx_a = len(self.nodes_a) - 1
            else:
                self.connect_idx_a = len(self.nodes_a) - 1
                self.connect_idx_b = len(self.nodes_b) - 1
            
            self.done = True
            self.found_path = True
            return StepResult(edge=edge, done=True, found_path=True)
        
        self.swap_trees = not self.swap_trees
        return StepResult(edge=edge, rejected_point=connect_rejected)
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if self.connect_idx_a is None or self.connect_idx_b is None:
            return []
        
        # Path from start to connection point
        path_a = []
        i = self.connect_idx_a
        while i != -1:
            path_a.append(self.nodes_a[i])
            i = self.parent_a[i]
        path_a.reverse()
        
        # Path from connection point to goal
        path_b = []
        i = self.connect_idx_b
        while i != -1:
            path_b.append(self.nodes_b[i])
            i = self.parent_b[i]
        
        # Combine paths (skip duplicate connection point)
        return path_a + path_b[1:] if len(path_b) > 1 else path_a + path_b
    
    def get_status(self) -> str:
        total_nodes = len(self.nodes_a) + len(self.nodes_b)
        return f"RRT-Connect: iter {self.iteration}/{self.max_iters}, nodes {total_nodes} (A:{len(self.nodes_a)}, B:{len(self.nodes_b)})"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return RRTConnectParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'RRTConnectPlanner':
        params = params_widget.get_params()
        return RRTConnectPlanner(occ, start, goal, **params)

# ============================================================================
# BiTRRT Implementation
# ============================================================================


@dataclass
class BiTRRTMotion:
    """Single motion/node in a BiTRRT exploration tree."""

    point: Point
    parent: Optional[int]
    state_cost: float


class BiTRRTParamsWidget(QWidget):
    """Widget for BiTRRT parameter configuration."""

    def __init__(self) -> None:
        super().__init__()
        layout = QFormLayout()

        self.spin_range = QDoubleSpinBox()
        self.spin_range.setRange(1.0, 500.0)
        self.spin_range.setSingleStep(1.0)
        self.spin_range.setValue(24.0)
        self.spin_range.setToolTip("Maximum expansion range per tree extension")

        self.spin_temp_change = QDoubleSpinBox()
        self.spin_temp_change.setRange(0.001, 2.0)
        self.spin_temp_change.setSingleStep(0.01)
        self.spin_temp_change.setDecimals(3)
        self.spin_temp_change.setValue(0.10)
        self.spin_temp_change.setToolTip(
            "OMPL-style temperature increase factor parameter; the actual multiplier is exp(value)"
        )

        self.spin_init_temp = QDoubleSpinBox()
        self.spin_init_temp.setRange(0.001, 100000.0)
        self.spin_init_temp.setDecimals(3)
        self.spin_init_temp.setValue(100.0)
        self.spin_init_temp.setToolTip("Initial transition-test temperature")

        self.spin_frontier_threshold = QDoubleSpinBox()
        self.spin_frontier_threshold.setRange(0.0, 1000.0)
        self.spin_frontier_threshold.setDecimals(3)
        self.spin_frontier_threshold.setSpecialValueText("auto")
        self.spin_frontier_threshold.setValue(0.0)
        self.spin_frontier_threshold.setToolTip(
            "Distance threshold for frontier vs refinement expansion; 0 uses OMPL-style auto scaling"
        )

        self.spin_frontier_ratio = QDoubleSpinBox()
        self.spin_frontier_ratio.setRange(0.01, 10.0)
        self.spin_frontier_ratio.setSingleStep(0.01)
        self.spin_frontier_ratio.setDecimals(3)
        self.spin_frontier_ratio.setValue(0.10)
        self.spin_frontier_ratio.setToolTip(
            "Maximum allowed ratio of non-frontier to frontier expansions"
        )

        self.chk_cost_threshold = QCheckBox("Enable")
        self.chk_cost_threshold.setToolTip(
            "Enable an upper bound on accepted transition costs"
        )

        self.spin_cost_threshold = QDoubleSpinBox()
        self.spin_cost_threshold.setRange(0.0, 1000.0)
        self.spin_cost_threshold.setSingleStep(0.1)
        self.spin_cost_threshold.setDecimals(3)
        self.spin_cost_threshold.setValue(25.0)
        self.spin_cost_threshold.setEnabled(False)
        self.spin_cost_threshold.setToolTip(
            "Maximum motion cost accepted by the transition test when enabled"
        )
        self.chk_cost_threshold.toggled.connect(self.spin_cost_threshold.setEnabled)

        cost_threshold_widget = QWidget()
        cost_threshold_layout = QHBoxLayout(cost_threshold_widget)
        cost_threshold_layout.setContentsMargins(0, 0, 0, 0)
        cost_threshold_layout.addWidget(self.chk_cost_threshold)
        cost_threshold_layout.addWidget(self.spin_cost_threshold)

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 200000)
        self.spin_max_iters.setValue(25000)
        self.spin_max_iters.setToolTip("Maximum number of planning iterations")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Range:", self.spin_range)
        layout.addRow("Temp change factor:", self.spin_temp_change)
        layout.addRow("Initial temperature:", self.spin_init_temp)
        layout.addRow("Frontier threshold:", self.spin_frontier_threshold)
        layout.addRow("Frontier node ratio:", self.spin_frontier_ratio)
        layout.addRow("Cost threshold:", cost_threshold_widget)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'range': self.spin_range.value(),
            'temp_change_factor': self.spin_temp_change.value(),
            'init_temperature': self.spin_init_temp.value(),
            'frontier_threshold': self.spin_frontier_threshold.value(),
            'frontier_node_ratio': self.spin_frontier_ratio.value(),
            'cost_threshold': self.spin_cost_threshold.value() if self.chk_cost_threshold.isChecked() else float('inf'),
            'max_iters': self.spin_max_iters.value(),
            'seed': self.spin_seed.value(),
        }


class BiTRRTPlanner(BasePlanner):
    """BiTRRT - Bidirectional Transition-based Rapidly-exploring Random Trees."""

    name = "BiTRRT"
    description = "Bidirectional cost-aware RRT with transition tests and frontier control"

    FAILED = 0
    ADVANCED = 1
    SUCCESS = 2

    def __init__(
        self,
        occ: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        range: float = 18.0,
        temp_change_factor: float = 0.10,
        init_temperature: float = 100.0,
        frontier_threshold: float = 0.0,
        frontier_node_ratio: float = 0.10,
        cost_threshold: float = float('inf'),
        max_iters: int = 25000,
        seed: int = 1,
    ) -> None:
        super().__init__(occ, start, goal)

        self.max_distance = float(max(1.0, range))
        self.temp_change_factor_param = float(temp_change_factor)
        self.temp_change_multiplier = float(np.exp(temp_change_factor))
        self.init_temperature = float(max(1e-6, init_temperature))
        self.temp = self.init_temperature
        self.max_iters = int(max_iters)
        self.frontier_node_ratio = float(max(0.01, frontier_node_ratio))
        self.cost_threshold = float(cost_threshold)
        self.rng = np.random.default_rng(seed)

        free_space = (~occ).astype(np.uint8)
        self.clearance_field = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        # Clearance-derived adaptation of OMPL's generic state cost / mechanical-work setup.
        self.cost_field = 100.0 / (1.0 + self.clearance_field.astype(np.float64))

        self.max_extent = float(np.hypot(max(1, self.w - 1), max(1, self.h - 1)))
        self.frontier_threshold = (
            float(frontier_threshold)
            if frontier_threshold > 0.0
            else max(1e-6, 0.01 * self.max_extent)
        )
        self.connection_range = max(1e-6, 0.10 * self.max_extent)

        start_cost = self._state_cost(start)
        goal_cost = self._state_cost(goal)
        self.tree_start: List[BiTRRTMotion] = [BiTRRTMotion(start, None, start_cost)]
        self.tree_goal: List[BiTRRTMotion] = [BiTRRTMotion(goal, None, goal_cost)]

        self.best_cost = 0.0
        self.worst_cost = max(start_cost, goal_cost, 0.0)
        self.frontier_count = 1
        self.nonfrontier_count = 1

        self.grow_start_tree = True
        self.connection_start_idx: Optional[int] = None
        self.connection_goal_idx: Optional[int] = None

    def _state_cost(self, point: Point) -> float:
        x, y = point
        return float(self.cost_field[y, x])

    def _motion_cost(self, from_pt: Point, to_pt: Point) -> float:
        pixels = segment_points(from_pt, to_pt)
        if len(pixels) < 2:
            return 0.0

        total = 0.0
        prev_cost = self._state_cost(pixels[0])
        for p in pixels[1:]:
            current_cost = self._state_cost(p)
            if current_cost > prev_cost:
                total += current_cost - prev_cost
            prev_cost = current_cost
        return total

    def _sample_uniform(self) -> Point:
        return (
            int(self.rng.integers(0, self.w)),
            int(self.rng.integers(0, self.h)),
        )

    def _nearest_index(self, motions: List[BiTRRTMotion], point: Point) -> int:
        pts = np.array([motion.point for motion in motions], dtype=np.float32)
        dx = pts[:, 0] - point[0]
        dy = pts[:, 1] - point[1]
        return int(np.argmin(dx * dx + dy * dy))

    def _add_motion(self, point: Point, motions: List[BiTRRTMotion], parent: Optional[int]) -> int:
        state_cost = self._state_cost(point)
        motions.append(BiTRRTMotion(point=point, parent=parent, state_cost=state_cost))
        self.worst_cost = max(self.worst_cost, state_cost)
        self.best_cost = min(self.best_cost, state_cost)
        return len(motions) - 1

    def _transition_test(self, motion_cost: float) -> bool:
        if motion_cost >= self.cost_threshold:
            return False
        if motion_cost < 1e-4:
            return True

        transition_probability = float(np.exp(-motion_cost / max(self.temp, 1e-9)))
        if transition_probability > 0.5:
            cost_range = self.worst_cost - self.best_cost
            if abs(cost_range) > 1e-4:
                self.temp /= float(np.exp(motion_cost / (0.1 * cost_range)))
            return True

        self.temp *= self.temp_change_multiplier
        return False

    def _min_expansion_control(self, distance_from_nearest: float) -> bool:
        if distance_from_nearest > self.frontier_threshold:
            self.frontier_count += 1
            return True

        if (self.nonfrontier_count / max(1, self.frontier_count)) > self.frontier_node_ratio:
            return False

        self.nonfrontier_count += 1
        return True

    def _extend_tree(
        self,
        nearest_idx: int,
        motions: List[BiTRRTMotion],
        target: Point,
        tree_is_start: bool,
    ) -> Tuple[int, Optional[int], Optional[Edge], Optional[Point]]:
        q_near = motions[nearest_idx].point
        d = dist(q_near, target)
        reach = d <= self.max_distance
        q_new = clamp_point(target if reach else steer(q_near, target, self.max_distance), self.w, self.h)

        if q_new == q_near or not self.is_free(q_new):
            return self.FAILED, None, None, q_new

        if not line_collision_free(q_near, q_new, self.occ):
            return self.FAILED, None, None, q_new

        motion_cost = self._motion_cost(q_near, q_new) if tree_is_start else self._motion_cost(q_new, q_near)
        extension_distance = dist(q_near, q_new)
        if not self._transition_test(motion_cost):
            return self.FAILED, None, None, q_new
        if not self._min_expansion_control(extension_distance):
            return self.FAILED, None, None, q_new

        new_idx = self._add_motion(q_new, motions, nearest_idx)
        edge = (q_near, q_new)
        return (self.SUCCESS if reach else self.ADVANCED), new_idx, edge, None

    def _connect_trees(
        self,
        source_idx: int,
        source_motions: List[BiTRRTMotion],
        target_motions: List[BiTRRTMotion],
        target_tree_is_start: bool,
    ) -> Tuple[bool, List[Edge], Optional[int], Optional[Point]]:
        source_point = source_motions[source_idx].point
        nearest_idx = self._nearest_index(target_motions, source_point)
        nearest_point = target_motions[nearest_idx].point
        if dist(nearest_point, source_point) > self.connection_range:
            return False, [], None, None

        connect_edges: List[Edge] = []
        current_nearest_idx = nearest_idx
        while True:
            result, new_idx, edge, rejected = self._extend_tree(
                current_nearest_idx, target_motions, source_point, target_tree_is_start
            )
            if edge is not None:
                connect_edges.append(edge)
            if result == self.ADVANCED and new_idx is not None:
                current_nearest_idx = new_idx
                continue
            if result == self.SUCCESS and new_idx is not None:
                return True, connect_edges, new_idx, None
            return False, connect_edges, None, rejected

    def _backtrack_path(self, motions: List[BiTRRTMotion], index: int) -> List[Point]:
        path: List[Point] = []
        visited: Set[int] = set()
        current: Optional[int] = index
        while current is not None and current not in visited:
            visited.add(current)
            path.append(motions[current].point)
            current = motions[current].parent
        return path

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=False)

        self.iteration += 1
        q_rand = self._sample_uniform()

        if self.grow_start_tree:
            grow_tree = self.tree_start
            grow_is_start = True
            other_tree = self.tree_goal
            other_is_start = False
        else:
            grow_tree = self.tree_goal
            grow_is_start = False
            other_tree = self.tree_start
            other_is_start = True

        nearest_idx = self._nearest_index(grow_tree, q_rand)
        extend_result, new_idx, extend_edge, rejected_point = self._extend_tree(
            nearest_idx, grow_tree, q_rand, grow_is_start
        )

        edges: List[Edge] = []
        if extend_edge is not None:
            edges.append(extend_edge)

        final_rejected = rejected_point
        if extend_result != self.FAILED and new_idx is not None:
            connected, connect_edges, other_idx, connect_rejected = self._connect_trees(
                new_idx, grow_tree, other_tree, other_is_start
            )
            edges.extend(connect_edges)
            if connect_rejected is not None:
                final_rejected = connect_rejected

            if connected and other_idx is not None:
                if grow_is_start:
                    self.connection_start_idx = new_idx
                    self.connection_goal_idx = other_idx
                else:
                    self.connection_start_idx = other_idx
                    self.connection_goal_idx = new_idx
                self.done = True
                self.found_path = True

                if len(edges) > 1:
                    result = StepResult(edges=edges, done=True, found_path=True)
                elif len(edges) == 1:
                    result = StepResult(edge=edges[0], done=True, found_path=True)
                else:
                    result = StepResult(done=True, found_path=True)
                self.grow_start_tree = not self.grow_start_tree
                return result

        self.grow_start_tree = not self.grow_start_tree
        if len(edges) > 1:
            return StepResult(edges=edges, rejected_point=final_rejected)
        if len(edges) == 1:
            return StepResult(edge=edges[0], rejected_point=final_rejected)
        return StepResult(rejected_point=final_rejected)

    def extract_path(self) -> List[Tuple[int, int]]:
        if self.connection_start_idx is None or self.connection_goal_idx is None:
            return []

        path_start = self._backtrack_path(self.tree_start, self.connection_start_idx)
        path_start.reverse()
        path_goal = self._backtrack_path(self.tree_goal, self.connection_goal_idx)

        if path_goal and path_start and path_start[-1] == path_goal[0]:
            return path_start + path_goal[1:]
        return path_start + path_goal

    def get_status(self) -> str:
        total_states = len(self.tree_start) + len(self.tree_goal)
        status = "FOUND" if self.found_path else "searching"
        threshold = "inf" if not np.isfinite(self.cost_threshold) else f"{self.cost_threshold:.2f}"
        return (
            f"BiTRRT: iter {self.iteration}/{self.max_iters}, states {total_states} "
            f"(S:{len(self.tree_start)}, G:{len(self.tree_goal)}), temp {self.temp:.3f}, "
            f"f/nf {self.frontier_count}/{self.nonfrontier_count}, cost<= {threshold}, {status}"
        )

    @staticmethod
    def get_params_widget() -> QWidget:
        return BiTRRTParamsWidget()

    @staticmethod
    def create_from_params(
        occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], params_widget: QWidget
    ) -> 'BiTRRTPlanner':
        params = params_widget.get_params()
        return BiTRRTPlanner(occ, start, goal, **params)


# ============================================================================
# KPIECE Implementation
# ============================================================================


@dataclass
class KPIECEMotion:
    """Single motion/node in a KPIECE exploration tree."""

    start: Point
    end: Point
    parent: Optional[int]
    cell_coord: Tuple[int, int]


@dataclass
class KPIECECell:
    """Single projected grid cell used by KPIECE."""

    coord: Tuple[int, int]
    motion_indices: List[int] = field(default_factory=list)
    coverage: float = 0.0
    selections: int = 1
    score: float = 1.0
    creation_iteration: int = 1
    importance: float = 0.0
    border: bool = True
    neighbor_count: int = 0


class KPIECEParamsWidget(QWidget):
    """Widget for KPIECE parameter configuration."""

    def __init__(self) -> None:
        super().__init__()
        layout = QFormLayout()

        self.spin_range = QDoubleSpinBox()
        self.spin_range.setRange(1.0, 500.0)
        self.spin_range.setSingleStep(1.0)
        self.spin_range.setValue(18.0)
        self.spin_range.setToolTip(
            "Maximum local expansion radius used when sampling around the selected motion"
        )

        self.spin_goal_bias = QDoubleSpinBox()
        self.spin_goal_bias.setRange(0.0, 1.0)
        self.spin_goal_bias.setSingleStep(0.01)
        self.spin_goal_bias.setDecimals(3)
        self.spin_goal_bias.setValue(0.02)
        self.spin_goal_bias.setToolTip(
            "Optional goal-directed sampling probability. Small nonzero values often help in this geometric 2D adaptation."
        )

        self.spin_goal_tol = QSpinBox()
        self.spin_goal_tol.setRange(1, 200)
        self.spin_goal_tol.setValue(24)
        self.spin_goal_tol.setToolTip(
            "Distance threshold for snapping a newly added state to the goal"
        )

        self.spin_border_fraction = QDoubleSpinBox()
        self.spin_border_fraction.setRange(0.0, 1.0)
        self.spin_border_fraction.setSingleStep(0.01)
        self.spin_border_fraction.setDecimals(3)
        self.spin_border_fraction.setValue(0.80)
        self.spin_border_fraction.setToolTip(
            "Probability of expanding from a border / exterior cell rather than an interior cell"
        )

        self.spin_progress_alpha = QDoubleSpinBox()
        self.spin_progress_alpha.setRange(0.001, 2.0)
        self.spin_progress_alpha.setSingleStep(0.01)
        self.spin_progress_alpha.setDecimals(3)
        self.spin_progress_alpha.setValue(0.10)
        self.spin_progress_alpha.setToolTip(
            "Positive progress offset alpha used in P = alpha + beta * (coverage increase / simulated distance)"
        )

        self.spin_progress_beta = QDoubleSpinBox()
        self.spin_progress_beta.setRange(0.0, 5.0)
        self.spin_progress_beta.setSingleStep(0.05)
        self.spin_progress_beta.setDecimals(3)
        self.spin_progress_beta.setValue(0.90)
        self.spin_progress_beta.setToolTip(
            "Progress scaling beta used in the paper-style score penalty"
        )

        self.spin_min_valid = QDoubleSpinBox()
        self.spin_min_valid.setRange(0.01, 1.0)
        self.spin_min_valid.setSingleStep(0.01)
        self.spin_min_valid.setDecimals(3)
        self.spin_min_valid.setValue(0.20)
        self.spin_min_valid.setToolTip(
            "Minimum valid fraction required to keep a partial edge when collision stops a motion"
        )

        self.spin_cell_size = QSpinBox()
        self.spin_cell_size.setRange(2, 200)
        self.spin_cell_size.setValue(28)
        self.spin_cell_size.setToolTip(
            "Projected grid cell size in pixels for the single-level KPIECE discretization"
        )

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 200000)
        self.spin_max_iters.setValue(25000)
        self.spin_max_iters.setToolTip("Maximum number of planning iterations")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Range:", self.spin_range)
        layout.addRow("Goal bias:", self.spin_goal_bias)
        layout.addRow("Goal tolerance:", self.spin_goal_tol)
        layout.addRow("Border fraction:", self.spin_border_fraction)
        layout.addRow("Progress alpha:", self.spin_progress_alpha)
        layout.addRow("Progress beta:", self.spin_progress_beta)
        layout.addRow("Min valid fraction:", self.spin_min_valid)
        layout.addRow("Cell size:", self.spin_cell_size)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'range': self.spin_range.value(),
            'goal_bias': self.spin_goal_bias.value(),
            'goal_tolerance': self.spin_goal_tol.value(),
            'border_fraction': self.spin_border_fraction.value(),
            'progress_alpha': self.spin_progress_alpha.value(),
            'progress_beta': self.spin_progress_beta.value(),
            'min_valid_path_fraction': self.spin_min_valid.value(),
            'cell_size': self.spin_cell_size.value(),
            'max_iters': self.spin_max_iters.value(),
            'seed': self.spin_seed.value(),
        }


class KPIECEPlanner(BasePlanner):
    """KPIECE - projection-guided tree planner with border-cell exploration."""

    name = "KPIECE"
    description = "Projection-grid planner with state sampling along motions, border-cell preference, and progress-based score penalties"

    def __init__(
        self,
        occ: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        range: float = 24.0,
        goal_bias: float = 0.02,
        goal_tolerance: int = 24,
        border_fraction: float = 0.80,
        progress_alpha: float = 0.10,
        progress_beta: float = 0.90,
        min_valid_path_fraction: float = 0.20,
        cell_size: int = 28,
        max_iters: int = 25000,
        seed: int = 1,
    ) -> None:
        super().__init__(occ, start, goal)

        self.max_distance = float(max(1.0, range))
        self.goal_bias = float(np.clip(goal_bias, 0.0, 1.0))
        self.goal_tolerance = float(max(1, goal_tolerance))
        self.border_fraction = float(np.clip(border_fraction, 0.0, 1.0))
        self.progress_alpha = float(max(1e-6, progress_alpha))
        self.progress_beta = float(max(0.0, progress_beta))
        self.min_valid_path_fraction = float(np.clip(min_valid_path_fraction, 1e-6, 1.0))
        self.cell_size = int(max(2, cell_size))
        self.max_iters = int(max_iters)
        self.rng = np.random.default_rng(seed)

        self.motions: List[KPIECEMotion] = []
        self.cells: Dict[Tuple[int, int], KPIECECell] = {}
        self.point_set: Set[Point] = set()
        self.goal_idx: Optional[int] = None
        self.closest_goal_idx: Optional[int] = None
        self.closest_goal_dist = float('inf')
        self.border_cell_count = 0

        start_idx, _ = self._add_motion(start, start, parent=None)
        self.closest_goal_idx = start_idx
        self.closest_goal_dist = dist(start, goal)

        if start == goal:
            self.goal_idx = start_idx
            self.found_path = True
            self.done = True

    def _project(self, point: Point) -> Tuple[int, int]:
        return (int(point[0] // self.cell_size), int(point[1] // self.cell_size))

    @staticmethod
    def _neighbor_coords(coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        cx, cy = coord
        return [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]

    def _compute_importance(self, cell: KPIECECell) -> None:
        creation_term = float(np.log(max(2, cell.creation_iteration)))
        neighbor_term = float(max(1, cell.neighbor_count))
        coverage_term = max(1.0, cell.coverage)
        selection_term = float(max(1, cell.selections))
        cell.importance = creation_term * cell.score / (selection_term * neighbor_term * coverage_term)

    def _recompute_cells(self, coords: Set[Tuple[int, int]]) -> None:
        for coord in coords:
            cell = self.cells.get(coord)
            if cell is None:
                continue
            prev_border = cell.border
            cell.neighbor_count = sum(1 for ncoord in self._neighbor_coords(coord) if ncoord in self.cells)
            cell.border = cell.neighbor_count < 4
            if cell.border != prev_border:
                self.border_cell_count += 1 if cell.border else -1
            self._compute_importance(cell)

    def _add_motion(
        self, start: Point, end: Point, parent: Optional[int], coord: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, float]:
        coord = self._project(end) if coord is None else coord
        motion_idx = len(self.motions)
        self.motions.append(KPIECEMotion(start=start, end=end, parent=parent, cell_coord=coord))
        self.point_set.add(end)

        cell = self.cells.get(coord)
        affected_coords: Set[Tuple[int, int]] = {coord}
        if cell is None:
            cell = KPIECECell(coord=coord, creation_iteration=max(1, self.iteration))
            self.cells[coord] = cell
            self.border_cell_count += 1
            affected_coords.update(self._neighbor_coords(coord))

        coverage_delta = 1.0 if parent is None else max(1e-6, dist(start, end))
        cell.coverage += coverage_delta
        cell.motion_indices.append(motion_idx)

        self._recompute_cells(affected_coords)

        goal_dist = dist(end, self.goal)
        if goal_dist < self.closest_goal_dist:
            self.closest_goal_dist = goal_dist
            self.closest_goal_idx = motion_idx

        return motion_idx, coverage_delta

    def _select_cell(self) -> KPIECECell:
        all_cells = [cell for cell in self.cells.values() if cell.motion_indices]
        border_cells = [cell for cell in all_cells if cell.border]
        interior_cells = [cell for cell in all_cells if not cell.border]
        border_probability = max(
            self.border_fraction,
            len(border_cells) / max(1, len(all_cells)),
        )

        if border_cells and (not interior_cells or self.rng.random() < border_probability):
            pool = border_cells
        else:
            pool = interior_cells if interior_cells else all_cells

        chosen = max(
            pool,
            key=lambda cell: (
                cell.importance,
                cell.creation_iteration,
                -cell.selections,
                len(cell.motion_indices),
            ),
        )
        if chosen.score < np.finfo(np.float64).eps:
            for cell in self.cells.values():
                cell.score += 1.0
                self._compute_importance(cell)
            chosen = max(
                pool,
                key=lambda cell: (
                    cell.importance,
                    cell.creation_iteration,
                    -cell.selections,
                    len(cell.motion_indices),
                ),
            )
        chosen.selections += 1
        self._compute_importance(chosen)
        return chosen

    def _select_motion_index(self, cell: KPIECECell) -> int:
        if len(cell.motion_indices) == 1:
            return cell.motion_indices[0]

        sigma = max(1.0, len(cell.motion_indices) / 3.0)
        offset = min(int(abs(self.rng.normal(0.0, sigma))), len(cell.motion_indices) - 1)
        return cell.motion_indices[-1 - offset]

    def _select_state_on_motion(self, motion_idx: int) -> Point:
        motion = self.motions[motion_idx]
        if motion.start == motion.end:
            return motion.end
        t = float(self.rng.random())
        x = int(round(motion.start[0] + t * (motion.end[0] - motion.start[0])))
        y = int(round(motion.start[1] + t * (motion.end[1] - motion.start[1])))
        return clamp_point((x, y), self.w, self.h)

    def _sample_target(self, from_pt: Point) -> Point:
        if self.rng.random() < self.goal_bias:
            target = self.goal
        else:
            angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
            radius = float(self.max_distance * np.sqrt(self.rng.random()))
            target = (
                int(round(from_pt[0] + radius * np.cos(angle))),
                int(round(from_pt[1] + radius * np.sin(angle))),
            )

        target = clamp_point(target, self.w, self.h)
        if dist(from_pt, target) > self.max_distance:
            target = clamp_point(steer(from_pt, target, self.max_distance), self.w, self.h)
        return target

    def _split_segment_by_cells(
        self, start: Point, end: Point
    ) -> List[Tuple[Point, Point, Tuple[int, int]]]:
        points = segment_points(start, end)
        if len(points) < 2:
            return []

        segments: List[Tuple[Point, Point, Tuple[int, int]]] = []
        current_start = points[0]
        current_cell = self._project(points[0])
        for point in points[1:]:
            point_cell = self._project(point)
            if point_cell != current_cell:
                if current_start != point:
                    segments.append((current_start, point, current_cell))
                current_start = point
                current_cell = point_cell
        if current_start != points[-1]:
            segments.append((current_start, points[-1], current_cell))
        return segments

    def _partial_extension(self, from_pt: Point, target: Point) -> Tuple[Optional[Point], float]:
        if target == from_pt:
            return None, 0.0

        total_dist = dist(from_pt, target)
        if total_dist <= 1e-6:
            return None, 0.0

        last_valid = from_pt
        for point in segment_points(from_pt, target):
            if point == from_pt:
                continue
            if not self.is_free(point):
                break
            last_valid = point

        if last_valid == from_pt:
            return None, 0.0

        moved_dist = dist(from_pt, last_valid)
        valid_fraction = moved_dist / total_dist
        if last_valid != target and valid_fraction < self.min_valid_path_fraction:
            return None, moved_dist
        return last_valid, moved_dist

    def _apply_progress_penalty(self, cell: KPIECECell, coverage_delta: float, simulated_distance: float) -> None:
        if simulated_distance <= 1e-6:
            progress = self.progress_alpha
        else:
            progress = self.progress_alpha + self.progress_beta * (coverage_delta / simulated_distance)
        if progress < 1.0:
            cell.score *= progress
            self._compute_importance(cell)

    def _add_motion_chain(
        self, parent_idx: int, start: Point, end: Point
    ) -> Tuple[Optional[int], List[Edge], float]:
        segments = self._split_segment_by_cells(start, end)
        if not segments:
            return None, [], 0.0

        total_coverage_delta = 0.0
        edges: List[Edge] = []
        current_parent = parent_idx
        last_idx: Optional[int] = None
        for seg_start, seg_end, seg_cell in segments:
            if seg_start == seg_end:
                continue
            motion_idx, coverage_delta = self._add_motion(seg_start, seg_end, current_parent, coord=seg_cell)
            total_coverage_delta += coverage_delta
            edges.append((seg_start, seg_end))
            current_parent = motion_idx
            last_idx = motion_idx
        return last_idx, edges, total_coverage_delta

    def _connect_goal(self, from_idx: int) -> Optional[Tuple[int, List[Edge]]]:
        from_pt = self.motions[from_idx].end
        if dist(from_pt, self.goal) > self.goal_tolerance:
            return None
        if not line_collision_free(from_pt, self.goal, self.occ):
            return None

        if from_pt == self.goal:
            self.goal_idx = from_idx
            self.done = True
            self.found_path = True
            return (from_idx, [])

        goal_idx, goal_edges, _ = self._add_motion_chain(from_idx, from_pt, self.goal)
        if goal_idx is None:
            return None

        self.goal_idx = goal_idx
        self.done = True
        self.found_path = True
        return (goal_idx, goal_edges)

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=False)

        self.iteration += 1
        cell = self._select_cell()
        motion_idx = self._select_motion_index(cell)
        source = self._select_state_on_motion(motion_idx)
        target = self._sample_target(source)
        attempted_distance = max(dist(source, target), 1e-6)
        new_point, traveled = self._partial_extension(source, target)

        if new_point is None or new_point in self.point_set:
            self._apply_progress_penalty(cell, coverage_delta=0.0, simulated_distance=attempted_distance)
            return StepResult(rejected_point=target)

        new_idx, edges, coverage_delta = self._add_motion_chain(motion_idx, source, new_point)
        self._apply_progress_penalty(cell, coverage_delta=coverage_delta, simulated_distance=max(traveled, attempted_distance))
        if new_idx is None:
            return StepResult(rejected_point=target)

        goal_result = self._connect_goal(new_idx)
        if goal_result is not None:
            _, goal_edges = goal_result
            if goal_edges:
                return StepResult(edges=edges + goal_edges, done=True, found_path=True)
            return StepResult(edges=edges, done=True, found_path=True)

        if len(edges) == 1:
            return StepResult(edge=edges[0])
        return StepResult(edges=edges)

    def extract_path(self) -> List[Point]:
        if self.goal_idx is None:
            return []

        chain: List[Tuple[int, Point]] = []
        current: Optional[int] = self.goal_idx
        target = self.motions[self.goal_idx].end
        while current is not None:
            chain.append((current, target))
            motion = self.motions[current]
            target = motion.start
            current = motion.parent

        chain.reverse()
        root_motion = self.motions[chain[0][0]]
        path: List[Point] = [root_motion.start]
        for _, target_state in chain:
            if path[-1] != target_state:
                path.append(target_state)
        return path

    def get_status(self) -> str:
        status = "FOUND" if self.found_path else f"closest {self.closest_goal_dist:.1f}px"
        return (
            f"KPIECE: iter {self.iteration}/{self.max_iters}, motions {len(self.motions)}, "
            f"cells {len(self.cells)} (border {self.border_cell_count}), {status}"
        )

    @staticmethod
    def get_params_widget() -> QWidget:
        return KPIECEParamsWidget()

    @staticmethod
    def create_from_params(
        occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], params_widget: QWidget
    ) -> 'KPIECEPlanner':
        params = params_widget.get_params()
        return KPIECEPlanner(occ, start, goal, **params)

# ============================================================================
# RRT* Implementation
# ============================================================================

class RRTStarParamsWidget(QWidget):
    """Widget for RRT* parameter configuration."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 200)
        self.spin_step.setValue(18)
        self.spin_step.setToolTip("Distance for each tree expansion step")
        
        self.spin_goal_rate = QDoubleSpinBox()
        self.spin_goal_rate.setRange(0.0, 1.0)
        self.spin_goal_rate.setSingleStep(0.01)
        self.spin_goal_rate.setValue(0.10)
        self.spin_goal_rate.setToolTip("Probability of sampling the goal directly")
        
        self.spin_goal_tol = QSpinBox()
        self.spin_goal_tol.setRange(1, 200)
        self.spin_goal_tol.setValue(20)
        self.spin_goal_tol.setToolTip("Distance threshold to consider goal reached")
        
        self.spin_search_radius = QSpinBox()
        self.spin_search_radius.setRange(10, 500)
        self.spin_search_radius.setValue(50)
        self.spin_search_radius.setToolTip("Radius for finding nearby nodes for rewiring")
        
        self.spin_col = QSpinBox()
        self.spin_col.setRange(10, 500)
        self.spin_col.setValue(80)
        self.spin_col.setToolTip("Number of samples for collision checking along edges")
        
        self.spin_maxit = QSpinBox()
        self.spin_maxit.setRange(100, 200000)
        self.spin_maxit.setValue(25000)
        self.spin_maxit.setToolTip("Maximum number of iterations")
        
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Step size:", self.spin_step)
        layout.addRow("Goal sample rate:", self.spin_goal_rate)
        layout.addRow("Goal tolerance:", self.spin_goal_tol)
        layout.addRow("Search radius:", self.spin_search_radius)
        layout.addRow("Collision samples:", self.spin_col)
        layout.addRow("Max iterations:", self.spin_maxit)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'step_size': self.spin_step.value(),
            'goal_sample_rate': self.spin_goal_rate.value(),
            'goal_tolerance': self.spin_goal_tol.value(),
            'search_radius': self.spin_search_radius.value(),
            'collision_samples': self.spin_col.value(),
            'max_iters': self.spin_maxit.value(),
            'seed': self.spin_seed.value(),
        }

class RRTStarPlanner(BasePlanner):
    """RRT* - Rewiring-based RRT variant with incremental path improvement."""
    
    name = "RRT*"
    description = "Rewiring-based sampling planner with incremental path improvement"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 step_size: int = 18, goal_sample_rate: float = 0.10, goal_tolerance: int = 20,
                 search_radius: int = 50, collision_samples: int = 80, 
                 max_iters: int = 25000, seed: int = 1):
        super().__init__(occ, start, goal)
        
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
        self.search_radius = search_radius
        self.collision_samples = collision_samples
        self.max_iters = max_iters
        
        self.rng = np.random.default_rng(seed)
        self.nodes = [start]
        self.parent = [-1]
        self.cost = [0.0]  # Cost from start to each node
        self.goal_idx = None
        self.best_goal_cost = float('inf')
    
    def _nearest(self, p: Tuple[int, int]) -> int:
        """Find index of nearest node to point p."""
        pts = np.array(self.nodes, dtype=np.float32)
        dx = pts[:, 0] - p[0]
        dy = pts[:, 1] - p[1]
        return int(np.argmin(dx * dx + dy * dy))
    
    def _near(self, p: Tuple[int, int], radius: float) -> List[int]:
        """Find all nodes within radius of point p."""
        pts = np.array(self.nodes, dtype=np.float32)
        dx = pts[:, 0] - p[0]
        dy = pts[:, 1] - p[1]
        dists = np.sqrt(dx * dx + dy * dy)
        return list(np.where(dists <= radius)[0])
    
    def _sample(self) -> Tuple[int, int]:
        """Sample a random point, biased towards goal."""
        if self.rng.random() < self.goal_sample_rate:
            return self.goal
        for _ in range(100):
            p = (int(self.rng.integers(0, self.w)), int(self.rng.integers(0, self.h)))
            if self.is_free(p):
                return p
        return (int(self.rng.integers(0, self.w)), int(self.rng.integers(0, self.h)))
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        q_rand = self._sample()
        i_near = self._nearest(q_rand)
        q_near = self.nodes[i_near]
        q_new = clamp_point(steer(q_near, q_rand, self.step_size), self.w, self.h)
        
        # Check if new point is valid
        if not self.is_free(q_new):
            return StepResult(rejected_point=q_new)
        if not line_collision_free(q_near, q_new, self.occ, samples=self.collision_samples):
            return StepResult(rejected_point=q_new)
        
        # Find nearby nodes for potential better parent
        near_indices = self._near(q_new, self.search_radius)
        
        # Choose best parent (lowest cost)
        best_parent = i_near
        best_cost = self.cost[i_near] + dist(q_near, q_new)
        
        for i in near_indices:
            potential_cost = self.cost[i] + dist(self.nodes[i], q_new)
            if potential_cost < best_cost:
                if line_collision_free(self.nodes[i], q_new, self.occ, samples=self.collision_samples):
                    best_parent = i
                    best_cost = potential_cost
        
        # Add new node with best parent
        new_idx = len(self.nodes)
        self.nodes.append(q_new)
        self.parent.append(best_parent)
        self.cost.append(best_cost)
        
        edge = (self.nodes[best_parent], q_new)
        
        # Rewire: check if nearby nodes can be improved through new node
        # NEVER rewire the goal node (goal_idx) - its parent is managed separately
        # Also prevent cycles by not rewiring ancestors of the new node
        ancestors = set()
        p = best_parent
        while p != -1:
            ancestors.add(p)
            p = self.parent[p]
        
        for i in near_indices:
            if i == best_parent:
                continue
            if self.goal_idx is not None and i == self.goal_idx:
                continue  # Don't rewire goal node
            if i in ancestors:
                continue  # Don't rewire ancestors (would create cycle)
            potential_cost = best_cost + dist(q_new, self.nodes[i])
            if potential_cost < self.cost[i]:
                if line_collision_free(q_new, self.nodes[i], self.occ, samples=self.collision_samples):
                    self.parent[i] = new_idx
                    self._update_costs(i, potential_cost)
        
        # Track if path was improved this step
        path_improved = False
        
        # Check if goal reached
        goal_dist = dist(q_new, self.goal)
        if goal_dist <= self.goal_tolerance:
            if line_collision_free(q_new, self.goal, self.occ, samples=self.collision_samples):
                goal_cost = best_cost + goal_dist
                if goal_cost < self.best_goal_cost:
                    # Check: new_idx must not have goal_idx as ancestor (would create cycle)
                    creates_cycle = False
                    if self.goal_idx is not None:
                        p = new_idx
                        while p != -1:
                            if p == self.goal_idx:
                                creates_cycle = True
                                break
                            p = self.parent[p]
                    
                    if not creates_cycle:
                        # Found better path to goal
                        if self.goal_idx is None:
                            # First time reaching goal - add goal node
                            self.nodes.append(self.goal)
                            self.parent.append(new_idx)
                            self.cost.append(goal_cost)
                            self.goal_idx = len(self.nodes) - 1
                        else:
                            # Update existing goal connection
                            self.parent[self.goal_idx] = new_idx
                            self.cost[self.goal_idx] = goal_cost
                        self.best_goal_cost = goal_cost
                        self.found_path = True
                        path_improved = True
        
        return StepResult(edge=edge, path_improved=path_improved)
    
    def _update_costs(self, idx: int, new_cost: float):
        """Update costs iteratively after rewiring (avoid recursion limit)."""
        stack = [(idx, new_cost)]
        while stack:
            current_idx, current_cost = stack.pop()
            self.cost[current_idx] = current_cost
            # Find children and add to stack
            for i, p in enumerate(self.parent):
                if p == current_idx:
                    child_cost = current_cost + dist(self.nodes[current_idx], self.nodes[i])
                    stack.append((i, child_cost))
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if self.goal_idx is None:
            return []
        path = []
        i = self.goal_idx
        visited = set()  # Prevent infinite loop from bad parent pointers
        while i != -1 and i not in visited:
            visited.add(i)
            path.append(self.nodes[i])
            i = self.parent[i]
        path.reverse()
        return path
    
    def get_status(self) -> str:
        cost_str = f", path cost: {self.best_goal_cost:.1f}" if self.found_path else ""
        return f"RRT*: iter {self.iteration}/{self.max_iters}, nodes {len(self.nodes)}{cost_str}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return RRTStarParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'RRTStarPlanner':
        params = params_widget.get_params()
        return RRTStarPlanner(occ, start, goal, **params)

# ============================================================================
# CHOMP Planner (Covariant Hamiltonian Optimization for Motion Planning)
# ============================================================================

class CHOMPParamsWidget(QWidget):
    """Parameter widget for CHOMP planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 500)
        self.spin_num_points.setValue(50)
        self.spin_num_points.setToolTip("Number of waypoints in trajectory")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(10, 50000)
        self.spin_max_iters.setValue(1000)
        self.spin_max_iters.setToolTip("Maximum optimization iterations")
        
        self.spin_learning_rate = QDoubleSpinBox()
        self.spin_learning_rate.setRange(0.001, 10.0)
        self.spin_learning_rate.setSingleStep(0.1)
        self.spin_learning_rate.setValue(1.0)
        self.spin_learning_rate.setToolTip("Gradient descent step size")
        
        self.spin_smoothness_weight = QDoubleSpinBox()
        self.spin_smoothness_weight.setRange(0.0, 100.0)
        self.spin_smoothness_weight.setSingleStep(0.1)
        self.spin_smoothness_weight.setValue(1.0)
        self.spin_smoothness_weight.setToolTip("Weight for curvature reduction and smoothness")
        
        self.spin_obstacle_weight = QDoubleSpinBox()
        self.spin_obstacle_weight.setRange(0.0, 1000.0)
        self.spin_obstacle_weight.setSingleStep(1.0)
        self.spin_obstacle_weight.setValue(100.0)
        self.spin_obstacle_weight.setToolTip("Weight for obstacle avoidance")
        
        self.spin_obstacle_epsilon = QSpinBox()
        self.spin_obstacle_epsilon.setRange(1, 100)
        self.spin_obstacle_epsilon.setValue(20)
        self.spin_obstacle_epsilon.setToolTip("Distance field epsilon (obstacle influence range)")

        self.spin_path_length_weight = QDoubleSpinBox()
        self.spin_path_length_weight.setRange(0.0, 10.0)
        self.spin_path_length_weight.setSingleStep(0.05)
        self.spin_path_length_weight.setValue(0.0)
        self.spin_path_length_weight.setToolTip("Optional penalty for overly long detours")
        
        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Learning rate:", self.spin_learning_rate)
        layout.addRow("Smoothness weight:", self.spin_smoothness_weight)
        layout.addRow("Obstacle weight:", self.spin_obstacle_weight)
        layout.addRow("Obstacle epsilon:", self.spin_obstacle_epsilon)
        layout.addRow("Path length weight:", self.spin_path_length_weight)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'learning_rate': self.spin_learning_rate.value(),
            'smoothness_weight': self.spin_smoothness_weight.value(),
            'obstacle_weight': self.spin_obstacle_weight.value(),
            'obstacle_epsilon': self.spin_obstacle_epsilon.value(),
            'path_length_weight': self.spin_path_length_weight.value(),
        }


class CHOMPPlanner(BasePlanner):
    """CHOMP - Covariant Hamiltonian Optimization for Motion Planning.
    
    Optimizes a trajectory by minimizing:
    - Smoothness cost (minimize acceleration/curvature)
    - Obstacle cost (stay away from obstacles)
    """
    
    name = "CHOMP"
    description = "Optimization-based trajectory smoother"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 50, max_iters: int = 500, learning_rate: float = 1.0,
                 smoothness_weight: float = 1.0, obstacle_weight: float = 100.0,
                 obstacle_epsilon: int = 20, path_length_weight: float = 0.2,
                 init_trajectory: Optional[List[Tuple[int, int]]] = None):
        super().__init__(occ, start, goal)
        
        self.num_points = num_points
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.smoothness_weight = smoothness_weight
        self.obstacle_weight = obstacle_weight
        self.obstacle_epsilon = obstacle_epsilon
        self.path_length_weight = path_length_weight
        
        # Random generator for perturbations
        self.rng = np.random.default_rng(42)
        
        # Compute distance field from obstacles first
        self.dist_field = self._compute_distance_field()
        self.grad_x, self.grad_y = self._compute_gradient_field()
        
        # Initialize trajectory - use provided trajectory if available
        if init_trajectory is not None and len(init_trajectory) >= 2:
            self.trajectory = self._initialize_from_path(init_trajectory)
        else:
            self.trajectory = self._initialize_trajectory()
        
        # Track optimization state
        self.total_cost = float('inf')
        self.obs_cost = float('inf')
        self.converged = False
        self.best_trajectory = self.trajectory.copy()
        self.best_cost = float('inf')
        self.best_valid_trajectory: Optional[np.ndarray] = None
        self.best_valid_cost = float('inf')
        self.best_valid_length = float('inf')
        self.path_length = self._trajectory_length(self.trajectory)
        initial_total_cost, initial_obs_cost, initial_smooth_cost, _ = self._evaluate_trajectory(self.trajectory)
        self.total_cost = initial_total_cost
        self.obs_cost = initial_obs_cost
        self.smooth_cost = initial_smooth_cost
        self.best_cost = initial_total_cost
        self._check_path_validity()
        self._store_best_valid_trajectory(initial_total_cost)
        
    def _compute_distance_field(self) -> np.ndarray:
        """Compute signed distance field from obstacles."""
        free_space = (self.occ == 0).astype(np.uint8)
        dist_from_obstacle = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        
        obstacle_space = (self.occ > 0).astype(np.uint8)
        dist_inside = cv2.distanceTransform(obstacle_space, cv2.DIST_L2, 5)
        
        sdf = dist_from_obstacle - dist_inside
        return sdf
    
    def _compute_gradient_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient of distance field."""
        grad_x = cv2.Sobel(self.dist_field, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        grad_y = cv2.Sobel(self.dist_field, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        return grad_x, grad_y
    
    def _initialize_trajectory(self) -> np.ndarray:
        """Initialize trajectory - with random perturbation to help escape obstacles."""
        trajectory = np.zeros((self.num_points, 2), dtype=np.float64)
        
        # Start with straight line
        for i in range(self.num_points):
            t = i / (self.num_points - 1)
            trajectory[i, 0] = self.start[0] + t * (self.goal[0] - self.start[0])
            trajectory[i, 1] = self.start[1] + t * (self.goal[1] - self.start[1])
        
        # Check if straight line goes through obstacles
        has_collision = False
        for i in range(self.num_points):
            ix = int(np.clip(trajectory[i, 0], 0, self.w - 1))
            iy = int(np.clip(trajectory[i, 1], 0, self.h - 1))
            if self.dist_field[iy, ix] < 0:
                has_collision = True
                break
        
        # If collision, add sinusoidal perturbation perpendicular to path
        if has_collision:
            dx = self.goal[0] - self.start[0]
            dy = self.goal[1] - self.start[1]
            length = np.sqrt(dx**2 + dy**2) + 1e-6
            # Perpendicular direction
            perp_x = -dy / length
            perp_y = dx / length
            
            # Add sine wave perturbation (try both directions)
            amplitude = min(self.h, self.w) * 0.3
            best_traj = trajectory.copy()
            best_obs_cost = float('inf')
            
            for sign in [1, -1]:
                test_traj = trajectory.copy()
                for i in range(1, self.num_points - 1):
                    t = i / (self.num_points - 1)
                    offset = sign * amplitude * np.sin(np.pi * t)
                    test_traj[i, 0] += perp_x * offset
                    test_traj[i, 1] += perp_y * offset
                
                # Clamp to bounds
                test_traj[:, 0] = np.clip(test_traj[:, 0], 0, self.w - 1)
                test_traj[:, 1] = np.clip(test_traj[:, 1], 0, self.h - 1)
                
                # Evaluate obstacle cost
                obs_cost = 0
                for i in range(self.num_points):
                    ix = int(test_traj[i, 0])
                    iy = int(test_traj[i, 1])
                    d = self.dist_field[iy, ix]
                    if d < self.obstacle_epsilon:
                        obs_cost += self.obstacle_epsilon - d
                
                if obs_cost < best_obs_cost:
                    best_obs_cost = obs_cost
                    best_traj = test_traj.copy()
            
            trajectory = best_traj
        
        return trajectory

    def _resample_path(self, path: List[Tuple[int, int]], num_points: int) -> Optional[np.ndarray]:
        """Resample a polyline path to a fixed number of points."""
        if path is None or len(path) < 2:
            return None

        pts = np.array(path, dtype=np.float64)
        deltas = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(deltas, axis=1)
        total_len = float(np.sum(seg_lens))
        if total_len <= 1e-6:
            return None

        cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
        targets = np.linspace(0.0, total_len, num_points)
        resampled = np.zeros((num_points, 2), dtype=np.float64)

        seg_idx = 0
        for i, t in enumerate(targets):
            while seg_idx < len(seg_lens) - 1 and t > cum[seg_idx + 1]:
                seg_idx += 1
            seg_len = seg_lens[seg_idx]
            if seg_len <= 1e-6:
                resampled[i] = pts[seg_idx]
                continue
            local_t = (t - cum[seg_idx]) / seg_len
            resampled[i] = pts[seg_idx] + local_t * deltas[seg_idx]

        resampled[0] = np.array(self.start, dtype=np.float64)
        resampled[-1] = np.array(self.goal, dtype=np.float64)
        return resampled

    def _initialize_from_path(self, path: List[Tuple[int, int]]) -> np.ndarray:
        """Initialize trajectory from a provided path."""
        resampled = self._resample_path(path, self.num_points)
        if resampled is None:
            return self._initialize_trajectory()
        return resampled

    def _trajectory_length(self, trajectory: Optional[np.ndarray] = None) -> float:
        """Return the geometric length of a trajectory polyline."""
        traj = self.trajectory if trajectory is None else trajectory
        if len(traj) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

    def _trajectory_point_to_pixel(self, point: np.ndarray) -> Point:
        """Convert a floating-point waypoint to the nearest valid pixel."""
        x = int(np.clip(np.rint(point[0]), 0, self.w - 1))
        y = int(np.clip(np.rint(point[1]), 0, self.h - 1))
        return (x, y)

    def _rounded_trajectory_point(self, index: int) -> Point:
        """Return a trajectory waypoint rounded to the nearest pixel."""
        return self._trajectory_point_to_pixel(self.trajectory[index])

    def _is_local_trajectory_segment_valid(self, trajectory: np.ndarray, index: int) -> bool:
        """Check local segment validity around a trajectory waypoint."""
        prev_point = self._trajectory_point_to_pixel(trajectory[index - 1])
        current_point = self._trajectory_point_to_pixel(trajectory[index])
        next_point = self._trajectory_point_to_pixel(trajectory[index + 1])
        return (
            line_collision_free(prev_point, current_point, self.occ)
            and line_collision_free(current_point, next_point, self.occ)
        )

    def _polish_valid_trajectory(self, trajectory: np.ndarray, passes: int = 20, alpha: float = 0.3) -> np.ndarray:
        """Apply a light collision-checked smoothing pass to an already valid trajectory."""
        polished = trajectory.copy()
        n = len(polished)
        if n < 3:
            return polished

        for _ in range(passes):
            improved = False
            for i in range(1, n - 1):
                candidate = polished.copy()
                candidate[i] = (1.0 - alpha) * polished[i] + alpha * 0.5 * (polished[i - 1] + polished[i + 1])
                if self._is_local_trajectory_segment_valid(candidate, i):
                    old_bend = np.linalg.norm(polished[i - 1] - 2 * polished[i] + polished[i + 1])
                    new_bend = np.linalg.norm(candidate[i - 1] - 2 * candidate[i] + candidate[i + 1])
                    if new_bend <= old_bend + 1e-6:
                        polished[i] = candidate[i]
                        improved = True
            if not improved:
                break

        return polished

    def _evaluate_trajectory(self, trajectory: np.ndarray) -> Tuple[float, float, float, float]:
        """Return total, obstacle, smoothness, and path-length costs for a trajectory."""
        n = len(trajectory)
        total_smooth_cost = 0.0
        total_obs_cost = 0.0

        for i in range(1, n - 1):
            x, y = trajectory[i]
            accel_x = trajectory[i + 1, 0] - 2 * x + trajectory[i - 1, 0]
            accel_y = trajectory[i + 1, 1] - 2 * y + trajectory[i - 1, 1]
            total_smooth_cost += accel_x ** 2 + accel_y ** 2

            obs_cost, _, _ = self._get_obstacle_cost_and_grad(x, y)
            total_obs_cost += obs_cost

        avg_smooth_cost = total_smooth_cost / max(1, n - 2)
        avg_obs_cost = total_obs_cost / max(1, n - 2)
        avg_path_segment_length = self._trajectory_length(trajectory) / max(1, n - 1)
        total_cost = (
            self.smoothness_weight * avg_smooth_cost
            + self.obstacle_weight * avg_obs_cost
            + self.path_length_weight * avg_path_segment_length
        )
        return total_cost, avg_obs_cost, avg_smooth_cost, avg_path_segment_length

    def _store_best_valid_trajectory(self, total_cost: float) -> bool:
        """Keep the best collision-free trajectory seen so far."""
        if not self.found_path:
            return False

        current_length = self._trajectory_length(self.trajectory)
        better = (
            self.best_valid_trajectory is None
            or current_length < self.best_valid_length - 1.0
            or (
                abs(current_length - self.best_valid_length) <= 1.0
                and total_cost < self.best_valid_cost - 1e-6
            )
        )
        if better:
            self.best_valid_trajectory = self.trajectory.copy()
            self.best_valid_cost = total_cost
            self.best_valid_length = current_length
            return True
        return False

    def _finalize_best_trajectory(self) -> None:
        """Restore the best valid trajectory if one was found."""
        if self.best_valid_trajectory is not None:
            self.trajectory = self._polish_valid_trajectory(self.best_valid_trajectory.copy())
            self._check_path_validity()
            if not self.found_path:
                self.trajectory = self.best_valid_trajectory.copy()
                self.found_path = True
            self.found_path = True
            return

        self.trajectory = self.best_trajectory.copy()
        self._check_path_validity()
    
    def _get_obstacle_cost_and_grad(self, x: float, y: float) -> Tuple[float, float, float]:
        """Get obstacle cost and gradient at a point."""
        ix, iy = int(np.clip(x, 0, self.w - 1)), int(np.clip(y, 0, self.h - 1))
        
        d = self.dist_field[iy, ix]
        
        if d < 0:  # Inside obstacle - strong repulsion
            cost = self.obstacle_epsilon - d
            # Gradient points away from obstacle (towards free space)
            grad_x = -self.grad_x[iy, ix] 
            grad_y = -self.grad_y[iy, ix]
            # Normalize and amplify
            norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
            grad_x = grad_x / norm * 2.0
            grad_y = grad_y / norm * 2.0
        elif d < self.obstacle_epsilon:  # Near obstacle
            cost = 0.5 * (d - self.obstacle_epsilon) ** 2 / self.obstacle_epsilon
            factor = (d - self.obstacle_epsilon) / self.obstacle_epsilon
            grad_x = factor * self.grad_x[iy, ix]
            grad_y = factor * self.grad_y[iy, ix]
        else:  # Far from obstacle
            cost = 0.0
            grad_x = 0.0
            grad_y = 0.0
        
        return cost, grad_x, grad_y
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        if self.iteration >= self.max_iters:
            self.done = True
            self._finalize_best_trajectory()
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        n = self.num_points
        
        # Compute gradients for all internal points (not start/goal)
        grad = np.zeros((n, 2), dtype=np.float64)

        for i in range(1, n - 1):  # Skip start and goal
            x, y = self.trajectory[i]
            
            # Smoothness gradient (using finite differences of acceleration)
            smooth_grad_x = 2 * (2 * x - self.trajectory[i-1, 0] - self.trajectory[i+1, 0])
            smooth_grad_y = 2 * (2 * y - self.trajectory[i-1, 1] - self.trajectory[i+1, 1])

            # Obstacle gradient and cost
            _, obs_grad_x, obs_grad_y = self._get_obstacle_cost_and_grad(x, y)

            prev_vec = self.trajectory[i] - self.trajectory[i - 1]
            next_vec = self.trajectory[i] - self.trajectory[i + 1]
            prev_len = np.linalg.norm(prev_vec) + 1e-6
            next_len = np.linalg.norm(next_vec) + 1e-6
            length_grad_x = prev_vec[0] / prev_len + next_vec[0] / next_len
            length_grad_y = prev_vec[1] / prev_len + next_vec[1] / next_len

            # Combine gradients
            grad[i, 0] = (
                self.smoothness_weight * smooth_grad_x
                + self.obstacle_weight * obs_grad_x
                + self.path_length_weight * length_grad_x
            )
            grad[i, 1] = (
                self.smoothness_weight * smooth_grad_y
                + self.obstacle_weight * obs_grad_y
                + self.path_length_weight * length_grad_y
            )
        
        # Adaptive learning rate - reduce if oscillating
        lr = self.learning_rate
        if self.iteration > 50:
            lr *= 0.5  # Reduce learning rate later for fine-tuning
        
        # Gradient descent update with gradient clipping
        grad_norm = np.sqrt(np.sum(grad ** 2))
        if grad_norm > 1e-6:
            max_grad_norm = 50.0  # Clip gradients to prevent instability
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
        
        # Update trajectory
        self.trajectory[1:-1] -= lr * grad[1:-1]
        
        # Clamp to image bounds
        self.trajectory[:, 0] = np.clip(self.trajectory[:, 0], 0, self.w - 1)
        self.trajectory[:, 1] = np.clip(self.trajectory[:, 1], 0, self.h - 1)

        total_cost, avg_obs_cost, avg_smooth_cost, avg_path_segment_length = self._evaluate_trajectory(self.trajectory)
        self.path_length = avg_path_segment_length * max(1, n - 1)

        if total_cost < self.best_cost:
            self.best_cost = total_cost
            self.best_trajectory = self.trajectory.copy()

        # Check if current path is valid (for live visualization)
        path_improved = False
        if self.iteration % 10 == 0 or avg_obs_cost < 0.5:
            self._check_path_validity()
            path_improved = self._store_best_valid_trajectory(total_cost)
        
        # Check convergence
        cost_change = abs(self.total_cost - total_cost) if self.total_cost != float('inf') else float('inf')
        self.total_cost = total_cost
        self.obs_cost = avg_obs_cost
        self.smooth_cost = avg_smooth_cost
        
        # Convergence: small cost change and low obstacle cost
        if cost_change < 0.01 and avg_obs_cost < 0.5 and self.iteration > 20:
            self.converged = True
            self.done = True
            self._finalize_best_trajectory()
        
        # Create edge for visualization
        point_idx = (self.iteration % (n - 1))
        if point_idx < n - 1:
            p1 = self._rounded_trajectory_point(point_idx)
            p2 = self._rounded_trajectory_point(point_idx + 1)
            edge = (p1, p2)
        else:
            edge = None
        
        return StepResult(edge=edge, path_improved=path_improved)
    
    def _check_path_validity(self):
        """Check if the final trajectory is collision-free."""
        for i in range(len(self.trajectory) - 1):
            p1 = self._rounded_trajectory_point(i)
            p2 = self._rounded_trajectory_point(i + 1)
            if not line_collision_free(p1, p2, self.occ):
                self.found_path = False
                return
        self.found_path = True
    
    def extract_path(self) -> List[Tuple[int, int]]:
        """Extract the current trajectory as a path."""
        return [self._rounded_trajectory_point(i) for i in range(len(self.trajectory))]

    def extract_display_path(self) -> List[Tuple[float, float]]:
        """Extract the floating-point trajectory for rendering."""
        return [(float(p[0]), float(p[1])) for p in self.trajectory]
    
    def get_status(self) -> str:
        status = "converged" if self.converged else ("FOUND" if self.found_path else "optimizing")
        smooth_str = f"smooth: {getattr(self, 'smooth_cost', 0):.1f}"
        obs_str = f"obs: {self.obs_cost:.1f}"
        return f"CHOMP: iter {self.iteration}/{self.max_iters}, {smooth_str}, {obs_str}, {status}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return CHOMPParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'CHOMPPlanner':
        params = params_widget.get_params()
        return CHOMPPlanner(occ, start, goal, **params)


# ============================================================================
# STOMP - Stochastic Trajectory Optimization for Motion Planning
# ============================================================================

class STOMPParamsWidget(QWidget):
    """Parameters widget for STOMP planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 200)
        self.spin_num_points.setValue(50)
        self.spin_num_points.setToolTip("Number of waypoints in trajectory")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(10, 20000)
        self.spin_max_iters.setValue(500)
        self.spin_max_iters.setToolTip("Maximum optimization iterations")
        
        self.spin_num_rollouts = QSpinBox()
        self.spin_num_rollouts.setRange(5, 100)
        self.spin_num_rollouts.setValue(20)
        self.spin_num_rollouts.setToolTip("Number of noisy trajectory samples per iteration")
        
        self.spin_noise_std = QDoubleSpinBox()
        self.spin_noise_std.setRange(0.1, 50.0)
        self.spin_noise_std.setSingleStep(1.0)
        self.spin_noise_std.setValue(10.0)
        self.spin_noise_std.setToolTip("Standard deviation of exploration noise")
        
        self.spin_smoothness_weight = QDoubleSpinBox()
        self.spin_smoothness_weight.setRange(0.0, 100.0)
        self.spin_smoothness_weight.setSingleStep(0.1)
        self.spin_smoothness_weight.setValue(0.1)
        self.spin_smoothness_weight.setToolTip("Weight for smoothness cost")
        
        self.spin_obstacle_weight = QDoubleSpinBox()
        self.spin_obstacle_weight.setRange(0.0, 1000.0)
        self.spin_obstacle_weight.setSingleStep(10.0)
        self.spin_obstacle_weight.setValue(100.0)
        self.spin_obstacle_weight.setToolTip("Weight for obstacle cost")
        
        self.spin_temperature = QDoubleSpinBox()
        self.spin_temperature.setRange(0.1, 100.0)
        self.spin_temperature.setSingleStep(1.0)
        self.spin_temperature.setValue(10.0)
        self.spin_temperature.setToolTip("Temperature for probability weighting (lower = more greedy)")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Num rollouts:", self.spin_num_rollouts)
        layout.addRow("Noise std:", self.spin_noise_std)
        layout.addRow("Smoothness weight:", self.spin_smoothness_weight)
        layout.addRow("Obstacle weight:", self.spin_obstacle_weight)
        layout.addRow("Temperature:", self.spin_temperature)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'num_rollouts': self.spin_num_rollouts.value(),
            'noise_std': self.spin_noise_std.value(),
            'smoothness_weight': self.spin_smoothness_weight.value(),
            'obstacle_weight': self.spin_obstacle_weight.value(),
            'temperature': self.spin_temperature.value(),
            'seed': self.spin_seed.value(),
        }


class STOMPPlanner(BasePlanner):
    """STOMP - Stochastic Trajectory Optimization for Motion Planning.
    
    Uses stochastic sampling to optimize trajectories:
    1. Generate K noisy trajectories by adding Gaussian noise
    2. Evaluate cost of each trajectory
    3. Compute probability-weighted average of updates
    4. Update trajectory with weighted combination
    """
    
    name = "STOMP"
    description = "Approximate / experimental stochastic trajectory optimization"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 50, max_iters: int = 300, num_rollouts: int = 20,
                 noise_std: float = 10.0, smoothness_weight: float = 0.1,
                 obstacle_weight: float = 100.0, temperature: float = 10.0, seed: int = 42):
        super().__init__(occ, start, goal)
        
        self.num_points = num_points
        self.max_iters = max_iters
        self.num_rollouts = num_rollouts
        self.noise_std = noise_std
        self.smoothness_weight = smoothness_weight
        self.obstacle_weight = obstacle_weight
        self.temperature = temperature
        
        # Random generator
        self.rng = np.random.default_rng(seed)
        
        # Compute distance field from obstacles
        self.dist_field = self._compute_distance_field()
        
        # Initialize trajectory with collision-avoiding initialization
        self.trajectory = self._initialize_trajectory()
        
        # Track optimization state
        self.total_cost = float('inf')
        self.obs_cost = float('inf')
        self.smooth_cost = 0.0
        self.converged = False
        self.best_trajectory = self.trajectory.copy()
        self.best_cost = float('inf')
        
        # Pre-compute smoothing matrix (finite difference matrix for acceleration)
        self._compute_smoothing_matrix()
    
    def _compute_distance_field(self) -> np.ndarray:
        """Compute distance field from obstacles."""
        free_space = (self.occ == 0).astype(np.uint8)
        dist_from_obstacle = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        return dist_from_obstacle
    
    def _compute_smoothing_matrix(self):
        """Pre-compute the smoothing matrix R for trajectory smoothing."""
        n = self.num_points - 2  # Internal points only
        if n <= 0:
            self.R_inv = np.eye(1)
            return
        
        # Finite difference matrix for acceleration (second derivative)
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = -2
            if i > 0:
                A[i, i-1] = 1
            if i < n - 1:
                A[i, i+1] = 1
        
        # R = A^T * A (metric for smoothness)
        R = A.T @ A + 1e-6 * np.eye(n)  # Regularization
        
        # Inverse for weighted averaging
        self.R_inv = np.linalg.inv(R)
        # Normalize rows
        row_sums = np.sum(self.R_inv, axis=1, keepdims=True)
        self.R_inv = self.R_inv / (row_sums + 1e-6)
    
    def _initialize_trajectory(self) -> np.ndarray:
        """Initialize trajectory, avoiding obstacles if possible."""
        trajectory = np.zeros((self.num_points, 2), dtype=np.float64)
        
        # Linear interpolation from start to goal
        for i in range(self.num_points):
            t = i / (self.num_points - 1)
            trajectory[i, 0] = self.start[0] + t * (self.goal[0] - self.start[0])
            trajectory[i, 1] = self.start[1] + t * (self.goal[1] - self.start[1])
        
        # Check for collisions and add perturbation if needed
        has_collision = False
        for i in range(1, self.num_points - 1):
            x, y = int(trajectory[i, 0]), int(trajectory[i, 1])
            x = np.clip(x, 0, self.w - 1)
            y = np.clip(y, 0, self.h - 1)
            if self.occ[y, x] > 0:
                has_collision = True
                break
        
        if has_collision:
            # Add sinusoidal perturbation perpendicular to path
            dx = self.goal[0] - self.start[0]
            dy = self.goal[1] - self.start[1]
            path_len = np.sqrt(dx*dx + dy*dy) + 1e-6
            
            # Perpendicular direction
            perp_x = -dy / path_len
            perp_y = dx / path_len
            
            # Determine which side to go (check both)
            amplitude = min(self.h, self.w) * 0.3
            
            for sign in [1, -1]:
                test_traj = trajectory.copy()
                for i in range(1, self.num_points - 1):
                    t = i / (self.num_points - 1)
                    offset = sign * amplitude * np.sin(np.pi * t)
                    test_traj[i, 0] += offset * perp_x
                    test_traj[i, 1] += offset * perp_y
                
                # Check if this side is better
                collision_free = True
                for i in range(1, self.num_points - 1):
                    x = int(np.clip(test_traj[i, 0], 0, self.w - 1))
                    y = int(np.clip(test_traj[i, 1], 0, self.h - 1))
                    if self.occ[y, x] > 0:
                        collision_free = False
                        break
                
                if collision_free:
                    trajectory = test_traj
                    break
            else:
                # Neither side is completely free, use the one with less collision
                trajectory = test_traj  # Use last tested
        
        return trajectory
    
    def _compute_trajectory_cost(self, traj: np.ndarray) -> Tuple[float, float, float]:
        """Compute total cost of a trajectory."""
        n = len(traj)
        
        # Obstacle cost
        obs_cost = 0.0
        for i in range(n):
            x = int(np.clip(traj[i, 0], 0, self.w - 1))
            y = int(np.clip(traj[i, 1], 0, self.h - 1))
            
            # High cost inside obstacles
            if self.occ[y, x] > 0:
                obs_cost += 1000.0
            else:
                # Cost increases near obstacles
                dist = self.dist_field[y, x]
                if dist < 10:
                    obs_cost += (10 - dist) ** 2
        
        # Smoothness cost (sum of squared accelerations)
        smooth_cost = 0.0
        for i in range(1, n - 1):
            accel_x = traj[i+1, 0] - 2 * traj[i, 0] + traj[i-1, 0]
            accel_y = traj[i+1, 1] - 2 * traj[i, 1] + traj[i-1, 1]
            smooth_cost += accel_x ** 2 + accel_y ** 2
        
        # Normalize
        obs_cost /= n
        smooth_cost /= max(1, n - 2)
        
        total = self.obstacle_weight * obs_cost + self.smoothness_weight * smooth_cost
        return total, obs_cost, smooth_cost
    
    def _compute_point_cost(self, x: float, y: float) -> float:
        """Compute cost at a single point."""
        ix = int(np.clip(x, 0, self.w - 1))
        iy = int(np.clip(y, 0, self.h - 1))
        
        if self.occ[iy, ix] > 0:
            return 1000.0
        
        dist = self.dist_field[iy, ix]
        if dist < 10:
            return (10 - dist) ** 2
        return 0.0
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        if self.iteration >= self.max_iters:
            self.done = True
            self.trajectory = self.best_trajectory.copy()
            self._check_path_validity()
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        n = self.num_points
        n_internal = n - 2  # Only internal points are modified
        
        if n_internal <= 0:
            self.done = True
            return StepResult(done=True, found_path=False)
        
        # Generate noisy rollouts
        noisy_trajectories = []
        costs = []
        
        # Adaptive noise - reduce over time
        current_noise = self.noise_std * (1.0 - 0.5 * self.iteration / self.max_iters)
        
        for k in range(self.num_rollouts):
            # Create noisy trajectory
            noise = self.rng.normal(0, current_noise, (n_internal, 2))
            
            # Apply smoothing to noise using R_inv
            if n_internal == self.R_inv.shape[0]:
                smooth_noise_x = self.R_inv @ noise[:, 0]
                smooth_noise_y = self.R_inv @ noise[:, 1]
                noise[:, 0] = smooth_noise_x
                noise[:, 1] = smooth_noise_y
            
            noisy_traj = self.trajectory.copy()
            noisy_traj[1:-1] += noise
            
            # Clamp to bounds
            noisy_traj[:, 0] = np.clip(noisy_traj[:, 0], 0, self.w - 1)
            noisy_traj[:, 1] = np.clip(noisy_traj[:, 1], 0, self.h - 1)
            
            # Compute cost
            total_cost, obs_cost, smooth_cost = self._compute_trajectory_cost(noisy_traj)
            
            noisy_trajectories.append(noisy_traj)
            costs.append(total_cost)
        
        # Convert to arrays
        costs = np.array(costs)
        
        # Compute probabilities using exponential weighting
        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / self.temperature)
        probabilities = exp_costs / (np.sum(exp_costs) + 1e-10)
        
        # Compute weighted average update
        delta = np.zeros((n_internal, 2))
        for k in range(self.num_rollouts):
            diff = noisy_trajectories[k][1:-1] - self.trajectory[1:-1]
            delta += probabilities[k] * diff
        
        # Update trajectory
        self.trajectory[1:-1] += delta
        
        # Clamp to bounds
        self.trajectory[:, 0] = np.clip(self.trajectory[:, 0], 0, self.w - 1)
        self.trajectory[:, 1] = np.clip(self.trajectory[:, 1], 0, self.h - 1)
        
        # Compute current cost
        self.total_cost, self.obs_cost, self.smooth_cost = self._compute_trajectory_cost(self.trajectory)
        
        # Track best
        if self.total_cost < self.best_cost:
            self.best_cost = self.total_cost
            self.best_trajectory = self.trajectory.copy()
        
        # Check path validity periodically
        path_improved = False
        if self.iteration % 5 == 0 or self.obs_cost < 1.0:
            old_found = self.found_path
            self._check_path_validity()
            if self.found_path and not old_found:
                path_improved = True
                self.best_trajectory = self.trajectory.copy()
        
        # Check convergence
        if self.obs_cost < 0.5 and self.iteration > 20:
            # Verify path is valid
            self._check_path_validity()
            if self.found_path:
                self.converged = True
                self.done = True
        
        # Create edge for visualization (show current segment being optimized)
        point_idx = self.iteration % (n - 1)
        if point_idx < n - 1:
            p1 = (int(self.trajectory[point_idx, 0]), int(self.trajectory[point_idx, 1]))
            p2 = (int(self.trajectory[point_idx + 1, 0]), int(self.trajectory[point_idx + 1, 1]))
            edge = (p1, p2)
        else:
            edge = None
        
        return StepResult(edge=edge, path_improved=path_improved)
    
    def _check_path_validity(self):
        """Check if current trajectory is collision-free."""
        for i in range(len(self.trajectory) - 1):
            p1 = (int(self.trajectory[i, 0]), int(self.trajectory[i, 1]))
            p2 = (int(self.trajectory[i + 1, 0]), int(self.trajectory[i + 1, 1]))
            if not line_collision_free(p1, p2, self.occ):
                self.found_path = False
                return
        self.found_path = True
    
    def extract_path(self) -> List[Tuple[int, int]]:
        """Extract current trajectory as path."""
        return [(int(p[0]), int(p[1])) for p in self.trajectory]
    
    def get_status(self) -> str:
        status = "converged" if self.converged else ("FOUND" if self.found_path else "optimizing")
        return f"STOMP: iter {self.iteration}/{self.max_iters}, obs: {self.obs_cost:.1f}, smooth: {self.smooth_cost:.1f}, {status}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return STOMPParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'STOMPPlanner':
        params = params_widget.get_params()
        return STOMPPlanner(occ, start, goal, **params)


# ============================================================================
# PRM - Probabilistic Roadmap
# ============================================================================

class PRMParamsWidget(QWidget):
    """Parameters widget for PRM planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_samples = QSpinBox()
        self.spin_num_samples.setRange(50, 10000)
        self.spin_num_samples.setValue(500)
        self.spin_num_samples.setToolTip("Number of random samples to generate")
        
        self.spin_k_neighbors = QSpinBox()
        self.spin_k_neighbors.setRange(3, 50)
        self.spin_k_neighbors.setValue(15)
        self.spin_k_neighbors.setToolTip("Number of nearest neighbors to connect")
        
        self.spin_max_edge_dist = QSpinBox()
        self.spin_max_edge_dist.setRange(10, 500)
        self.spin_max_edge_dist.setValue(100)
        self.spin_max_edge_dist.setToolTip("Maximum edge length")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Num samples:", self.spin_num_samples)
        layout.addRow("K neighbors:", self.spin_k_neighbors)
        layout.addRow("Max edge dist:", self.spin_max_edge_dist)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_samples': self.spin_num_samples.value(),
            'k_neighbors': self.spin_k_neighbors.value(),
            'max_edge_dist': self.spin_max_edge_dist.value(),
            'seed': self.spin_seed.value(),
        }


class PRMPlanner(BasePlanner):
    """PRM - Probabilistic Roadmap planner.
    
    Two phases:
    1. Learning phase: Build a roadmap of collision-free random samples
    2. Query phase: Connect start/goal to the roadmap and search the graph
    """
    
    name = "PRM"
    description = "Two-phase probabilistic roadmap with query-time start/goal connection"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                  num_samples: int = 500, k_neighbors: int = 10, max_edge_dist: int = 100, seed: int = 42):
        super().__init__(occ, start, goal)
        
        self.num_samples = int(max(1, num_samples))
        self.k_neighbors = int(max(1, k_neighbors))
        self.max_edge_dist = float(max(1.0, max_edge_dist))
        self.rng = np.random.default_rng(seed)

        # Roadmap is learned independently of the query.
        self.nodes: List[Point] = []
        self.edges: Dict[int, List[Tuple[int, float]]] = {}
        self.roadmap_size = 0
        self.sample_pool: List[Point] = self._build_sample_pool()

        # Query nodes are added only after roadmap learning.
        self.start_idx: Optional[int] = None
        self.goal_idx: Optional[int] = None

        # Phase tracking
        self.phase = "sampling"  # sampling -> connecting -> query_start -> query_goal -> searching -> done
        self.sample_idx = 0
        self.connect_idx = 0
        self.search_open: List[Tuple[float, float, int]] = []  # (f_cost, g_cost, node)
        self.search_closed: Set[int] = set()
        self.search_parent: Dict[int, int] = {}
        self.search_g: Dict[int, float] = {}
        self.current_path: List[int] = []

    def _build_sample_pool(self) -> List[Point]:
        """Pre-sample unique free roadmap nodes, excluding query configurations."""
        free_y, free_x = np.where(~self.occ)
        free_points = [
            (int(x), int(y))
            for x, y in zip(free_x.tolist(), free_y.tolist())
            if (int(x), int(y)) not in {self.start, self.goal}
        ]
        if not free_points:
            return []

        sample_count = min(self.num_samples, len(free_points))
        chosen = self.rng.choice(len(free_points), size=sample_count, replace=False)
        return [free_points[int(i)] for i in chosen]

    def _add_node(self, point: Point) -> int:
        node_idx = len(self.nodes)
        self.nodes.append(point)
        self.edges[node_idx] = []
        return node_idx

    def _edge_exists(self, a: int, b: int) -> bool:
        return any(neighbor == b for neighbor, _ in self.edges[a])

    def _candidate_neighbors(self, point: Point, candidate_indices: List[int]) -> List[Tuple[float, int]]:
        distances: List[Tuple[float, int]] = []
        for idx in candidate_indices:
            other = self.nodes[idx]
            d = dist(point, other)
            if d <= self.max_edge_dist:
                distances.append((d, idx))
        distances.sort(key=lambda item: item[0])
        return distances[:self.k_neighbors]

    def _connect_node(self, node_idx: int, candidate_indices: List[int]) -> List[Edge]:
        node = self.nodes[node_idx]
        added_edges: List[Edge] = []
        for edge_dist, neighbor_idx in self._candidate_neighbors(node, candidate_indices):
            if neighbor_idx == node_idx or self._edge_exists(node_idx, neighbor_idx):
                continue
            if not line_collision_free(node, self.nodes[neighbor_idx], self.occ):
                continue
            self.edges[node_idx].append((neighbor_idx, edge_dist))
            self.edges[neighbor_idx].append((node_idx, edge_dist))
            added_edges.append((node, self.nodes[neighbor_idx]))
        return added_edges
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        # If already found path, just return done
        if self.found_path:
            self.done = True
            return StepResult(done=True, found_path=True)
        
        self.iteration += 1
        
        if self.phase == "sampling":
            return self._sampling_step()
        elif self.phase == "connecting":
            return self._connecting_step()
        elif self.phase == "query_start":
            return self._query_connect_step(self.start)
        elif self.phase == "query_goal":
            return self._query_connect_step(self.goal)
        elif self.phase == "searching":
            return self._searching_step()
        
        return StepResult(done=True)
    
    def _sampling_step(self) -> StepResult:
        """Sample random points in free space."""
        if self.sample_idx >= len(self.sample_pool):
            self.roadmap_size = len(self.nodes)
            self.phase = "connecting"
            return StepResult()

        x, y = self.sample_pool[self.sample_idx]
        self._add_node((x, y))
        self.sample_idx += 1
        return StepResult(edge=((x - 2, y - 2), (x + 2, y + 2)))  # Small marker
    
    def _connecting_step(self) -> StepResult:
        """Connect roadmap nodes to nearby roadmap neighbors."""
        if self.connect_idx >= self.roadmap_size:
            self.phase = "query_start"
            return StepResult()

        candidate_indices = [idx for idx in range(self.roadmap_size) if idx != self.connect_idx]
        edges = self._connect_node(self.connect_idx, candidate_indices)
        self.connect_idx += 1
        return StepResult(edge=edges[0] if len(edges) == 1 else None, edges=edges if len(edges) > 1 else None)

    def _query_connect_step(self, query_point: Point) -> StepResult:
        """Attach a query configuration to the learned roadmap."""
        if self.roadmap_size == 0:
            self.phase = "done"
            self.done = True
            return StepResult(done=True, found_path=False)

        node_idx = self._add_node(query_point)
        # Query nodes connect to the previously built roadmap and, once available,
        # to earlier query nodes as well (e.g. direct start-goal visibility).
        edges = self._connect_node(node_idx, list(range(node_idx)))

        if query_point == self.start:
            self.start_idx = node_idx
            self.phase = "query_goal"
        else:
            self.goal_idx = node_idx
            if not edges or self.start_idx is None:
                self.phase = "done"
                self.done = True
                return StepResult(done=True, found_path=False)
            self.phase = "searching"
            self.search_open = []
            self.search_closed = set()
            self.search_parent = {}
            self.search_g = {self.start_idx: 0.0}
            heapq.heappush(self.search_open, (self._heuristic(self.start_idx), 0.0, self.start_idx))

        return StepResult(edge=edges[0] if len(edges) == 1 else None, edges=edges if len(edges) > 1 else None)
    
    def _searching_step(self) -> StepResult:
        """A* search on the roadmap."""
        if self.start_idx is None or self.goal_idx is None:
            self.phase = "done"
            self.done = True
            return StepResult(done=True, found_path=False)

        if not self.search_open:
            self.phase = "done"
            self.done = True
            return StepResult(done=True, found_path=False)
        
        _, current_g, current = heapq.heappop(self.search_open)

        if current == self.goal_idx:
            self.current_path = [current]
            while current in self.search_parent:
                current = self.search_parent[current]
                self.current_path.append(current)
            self.current_path.reverse()
            self.found_path = True
            self.phase = "done"
            self.done = True
            return StepResult(done=True, found_path=True)
        
        if current in self.search_closed:
            return StepResult()
        
        self.search_closed.add(current)
        
        edge = None
        for neighbor, cost in self.edges[current]:
            if neighbor in self.search_closed:
                continue

            tentative_g = current_g + cost
            if tentative_g < self.search_g.get(neighbor, float('inf')):
                self.search_parent[neighbor] = current
                self.search_g[neighbor] = tentative_g
                h_cost = self._heuristic(neighbor)
                heapq.heappush(self.search_open, (tentative_g + h_cost, tentative_g, neighbor))
                edge = (self.nodes[current], self.nodes[neighbor])
        
        return StepResult(edge=edge)
    
    def _heuristic(self, node_idx: int) -> float:
        """Euclidean distance to goal."""
        node = self.nodes[node_idx]
        return np.sqrt((node[0] - self.goal[0])**2 + (node[1] - self.goal[1])**2)
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if not self.current_path:
            return []
        return [self.nodes[i] for i in self.current_path]
    
    def get_status(self) -> str:
        total_edges = sum(len(e) for e in self.edges.values()) // 2
        return (
            f"PRM: {self.phase}, roadmap nodes: {self.roadmap_size}, "
            f"total nodes: {len(self.nodes)}, edges: {total_edges}"
        )
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return PRMParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'PRMPlanner':
        params = params_widget.get_params()
        return PRMPlanner(occ, start, goal, **params)


# ============================================================================
# SBL - Single-query Bi-directional Lazy collision checking
# ============================================================================

class SBLParamsWidget(QWidget):
    """Parameters widget for the SBL planner."""

    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.spin_maxit = QSpinBox()
        self.spin_maxit.setRange(100, 200000)
        self.spin_maxit.setValue(12000)
        self.spin_maxit.setToolTip("Maximum number of milestone expansion iterations")

        self.spin_rho = QSpinBox()
        self.spin_rho.setRange(2, 400)
        self.spin_rho.setValue(45)
        self.spin_rho.setToolTip("SBL distance threshold rho for local expansion and tree connection")

        self.spin_resolution = QSpinBox()
        self.spin_resolution.setRange(1, 50)
        self.spin_resolution.setValue(4)
        self.spin_resolution.setToolTip("Lazy segment resolution epsilon in pixels")

        self.spin_candidates = QSpinBox()
        self.spin_candidates.setRange(1, 20)
        self.spin_candidates.setValue(6)
        self.spin_candidates.setToolTip("Maximum number of shrinking-neighborhood candidates per expansion")

        self.spin_grid_cells = QSpinBox()
        self.spin_grid_cells.setRange(2, 50)
        self.spin_grid_cells.setValue(10)
        self.spin_grid_cells.setToolTip("Spatial indexing resolution per tree (cells per axis)")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Max iterations:", self.spin_maxit)
        layout.addRow("Rho:", self.spin_rho)
        layout.addRow("Lazy resolution:", self.spin_resolution)
        layout.addRow("Candidates:", self.spin_candidates)
        layout.addRow("Grid cells:", self.spin_grid_cells)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'max_iters': self.spin_maxit.value(),
            'rho': self.spin_rho.value(),
            'lazy_resolution': self.spin_resolution.value(),
            'max_candidates': self.spin_candidates.value(),
            'grid_cells': self.spin_grid_cells.value(),
            'seed': self.spin_seed.value(),
        }


class SBLPlanner(BasePlanner):
    """SBL - Single-query bi-directional planner with lazy collision checking."""

    name = "SBL"
    description = "Bi-directional lazy roadmap planner with deferred collision checking and a lightweight post-optimizer"

    def __init__(
        self,
        occ: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_iters: int = 12000,
        rho: int = 45,
        lazy_resolution: int = 4,
        max_candidates: int = 6,
        grid_cells: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__(occ, start, goal)

        self.max_iters = max_iters
        self.rho = float(rho)
        self.lazy_resolution = float(lazy_resolution)
        self.max_candidates = max_candidates
        self.grid_cells = grid_cells
        self.rng = np.random.default_rng(seed)

        self.nodes: List[SBLNode] = [
            SBLNode(point=start, tree_id=0, parent=None),
            SBLNode(point=goal, tree_id=1, parent=None),
        ]
        self.tree_nodes: Dict[int, Set[int]] = {0: {0}, 1: {1}}
        self.cell_maps: Dict[int, Dict[Tuple[int, int], Set[int]]] = {0: {}, 1: {}}
        self.edge_states: Dict[frozenset[int], SBLSegmentState] = {}
        self._add_to_cell_map(0)
        self._add_to_cell_map(1)

        self.raw_solution_path: List[Point] = []
        self.solution_path: List[Point] = []
        self.lazy_checks = 0
        self.transfer_count = 0
        self.bridge_attempts = 0
        self.last_collision_point: Optional[Point] = None

    def _edge_key(self, a: int, b: int) -> frozenset[int]:
        return frozenset((a, b))

    def _cell_for_point(self, p: Point) -> Tuple[int, int]:
        cx = min(int(p[0] * self.grid_cells / max(1, self.w)), self.grid_cells - 1)
        cy = min(int(p[1] * self.grid_cells / max(1, self.h)), self.grid_cells - 1)
        return (cx, cy)

    def _add_to_cell_map(self, node_id: int) -> None:
        node = self.nodes[node_id]
        cell = self._cell_for_point(node.point)
        self.cell_maps[node.tree_id].setdefault(cell, set()).add(node_id)

    def _remove_from_cell_map(self, node_id: int, tree_id: Optional[int] = None) -> None:
        actual_tree = self.nodes[node_id].tree_id if tree_id is None else tree_id
        cell = self._cell_for_point(self.nodes[node_id].point)
        ids = self.cell_maps[actual_tree].get(cell)
        if ids is None:
            return
        ids.discard(node_id)
        if not ids:
            del self.cell_maps[actual_tree][cell]

    def _move_node_between_trees(self, node_id: int, new_tree: int) -> None:
        old_tree = self.nodes[node_id].tree_id
        if old_tree == new_tree:
            return
        self._remove_from_cell_map(node_id, tree_id=old_tree)
        self.tree_nodes[old_tree].discard(node_id)
        self.nodes[node_id].tree_id = new_tree
        self.tree_nodes[new_tree].add(node_id)
        self._add_to_cell_map(node_id)

    def _pick_node_by_density(self, tree_id: int) -> int:
        non_empty_cells = [cell for cell, ids in self.cell_maps[tree_id].items() if ids]
        if not non_empty_cells:
            return 0 if tree_id == 0 else 1
        cell = non_empty_cells[int(self.rng.integers(0, len(non_empty_cells)))]
        ids = list(self.cell_maps[tree_id][cell])
        return ids[int(self.rng.integers(0, len(ids)))]

    def _random_tree_node(self, tree_id: int) -> Optional[int]:
        ids = list(self.tree_nodes[tree_id])
        if not ids:
            return None
        return ids[int(self.rng.integers(0, len(ids)))]

    def _sample_near(self, center: Point, radius: float) -> Point:
        radius = max(1.0, radius)
        iradius = max(1, int(np.ceil(radius)))
        for _ in range(40):
            dx = int(self.rng.integers(-iradius, iradius + 1))
            dy = int(self.rng.integers(-iradius, iradius + 1))
            p = clamp_point((center[0] + dx, center[1] + dy), self.w, self.h)
            if p != center and linf_dist(center, p) <= radius + 1e-6:
                return p
        return clamp_point((center[0] + iradius, center[1]), self.w, self.h)

    def _add_node(self, parent_id: int, point: Point, tree_id: int) -> int:
        node_id = len(self.nodes)
        self.nodes.append(SBLNode(point=point, tree_id=tree_id, parent=parent_id))
        self.nodes[parent_id].children.add(node_id)
        self.tree_nodes[tree_id].add(node_id)
        self._add_to_cell_map(node_id)
        self.edge_states[self._edge_key(parent_id, node_id)] = SBLSegmentState()
        return node_id

    def _expand_tree(self) -> Tuple[Optional[int], Optional[Edge]]:
        tree_id = int(self.rng.integers(0, 2))
        for _ in range(60):
            parent_id = self._pick_node_by_density(tree_id)
            parent_point = self.nodes[parent_id].point
            for i in range(1, self.max_candidates + 1):
                q = self._sample_near(parent_point, self.rho / i)
                if q == parent_point or not self.is_free(q):
                    continue
                node_id = self._add_node(parent_id, q, tree_id)
                return node_id, (parent_point, q)
        return None, None

    def _chain_root_to_node(self, node_id: int) -> List[int]:
        chain: List[int] = []
        current: Optional[int] = node_id
        while current is not None:
            chain.append(current)
            current = self.nodes[current].parent
        chain.reverse()
        return chain

    def _segment_length(self, edge_key: frozenset[int]) -> float:
        a, b = tuple(edge_key)
        return linf_dist(self.nodes[a].point, self.nodes[b].point)

    def _segment_new_points(self, a: Point, b: Point, level: int) -> List[Point]:
        denom = 2 ** level
        pts: List[Point] = []
        for odd in range(1, denom, 2):
            t = odd / denom
            x = int(round(a[0] + t * (b[0] - a[0])))
            y = int(round(a[1] + t * (b[1] - a[1])))
            p = clamp_point((x, y), self.w, self.h)
            if p != a and p != b and (not pts or pts[-1] != p):
                pts.append(p)
        return pts

    def _test_segment_once(self, edge_key: frozenset[int]) -> Optional[Point]:
        state = self.edge_states.setdefault(edge_key, SBLSegmentState())
        a_id, b_id = tuple(edge_key)
        a = self.nodes[a_id].point
        b = self.nodes[b_id].point
        next_level = state.kappa + 1
        self.lazy_checks += 1

        for p in self._segment_new_points(a, b, next_level):
            if not self.is_free(p):
                return p

        state.kappa = next_level
        if self._segment_length(edge_key) / (2 ** next_level) < self.lazy_resolution:
            if not line_collision_free(a, b, self.occ):
                for p in segment_points(a, b):
                    if not self.is_free(p):
                        return p
                return a
            state.safe = True
        return None

    def _segment_score(self, edge_key: frozenset[int]) -> float:
        state = self.edge_states[edge_key]
        if state.safe:
            return -1.0
        return self._segment_length(edge_key) / (2 ** state.kappa)

    def _build_solution_path(self, start_chain: List[int], goal_chain: List[int]) -> List[Point]:
        path = [self.nodes[node_id].point for node_id in start_chain]
        path.extend(self.nodes[node_id].point for node_id in reversed(goal_chain))

        deduped: List[Point] = []
        for p in path:
            if not deduped or deduped[-1] != p:
                deduped.append(p)
        return deduped

    def _point_at_arclength(self, path: List[Point], arclength: float) -> Tuple[int, Point]:
        """Return a point sampled along a polyline at the given arclength."""
        if len(path) < 2:
            return 0, path[0]

        remaining = max(0.0, arclength)
        for seg_idx in range(len(path) - 1):
            a = path[seg_idx]
            b = path[seg_idx + 1]
            seg_len = dist(a, b)
            if seg_len <= 1e-9:
                continue
            if remaining <= seg_len or seg_idx == len(path) - 2:
                t = float(np.clip(remaining / seg_len, 0.0, 1.0))
                x = int(round(a[0] + t * (b[0] - a[0])))
                y = int(round(a[1] + t * (b[1] - a[1])))
                return seg_idx, clamp_point((x, y), self.w, self.h)
            remaining -= seg_len
        return len(path) - 2, path[-1]

    def _splice_shortcut(
        self,
        path: List[Point],
        first_seg_idx: int,
        first_point: Point,
        second_seg_idx: int,
        second_point: Point,
    ) -> List[Point]:
        """Replace a subpath by a direct shortcut segment."""
        rebuilt: List[Point] = []
        for p in path[: first_seg_idx + 1]:
            if not rebuilt or rebuilt[-1] != p:
                rebuilt.append(p)
        if not rebuilt or rebuilt[-1] != first_point:
            rebuilt.append(first_point)
        if rebuilt[-1] != second_point:
            rebuilt.append(second_point)
        for p in path[second_seg_idx + 1:]:
            if rebuilt[-1] != p:
                rebuilt.append(p)
        return rebuilt

    def _optimize_solution_path(self, path: List[Point]) -> List[Point]:
        """Apply the SBL-style lightweight random path optimizer."""
        if len(path) < 3:
            return list(path)

        optimized = list(path)
        for _ in range(16):
            total_len = compute_path_length(optimized)
            if total_len <= 1e-6:
                break

            s1 = float(self.rng.uniform(0.0, total_len))
            s2 = float(self.rng.uniform(0.0, total_len))
            if s1 > s2:
                s1, s2 = s2, s1
            if s2 - s1 < 4.0:
                continue

            first_seg_idx, q1 = self._point_at_arclength(optimized, s1)
            second_seg_idx, q2 = self._point_at_arclength(optimized, s2)
            if q1 == q2:
                continue
            if not line_collision_free(q1, q2, self.occ):
                continue

            optimized = self._splice_shortcut(optimized, first_seg_idx, q1, second_seg_idx, q2)

        return optimized

    def _candidate_path_segments(
        self,
        start_chain: List[int],
        goal_chain: List[int],
        bridge_key: frozenset[int],
    ) -> List[frozenset[int]]:
        segments: List[frozenset[int]] = []
        for i in range(1, len(start_chain)):
            segments.append(self._edge_key(start_chain[i - 1], start_chain[i]))
        segments.append(bridge_key)
        for i in range(len(goal_chain) - 1, 0, -1):
            segments.append(self._edge_key(goal_chain[i], goal_chain[i - 1]))
        return segments

    def _test_candidate_path(
        self,
        start_chain: List[int],
        goal_chain: List[int],
        bridge_key: frozenset[int],
    ) -> Tuple[Optional[frozenset[int]], Optional[Point]]:
        path_segments = self._candidate_path_segments(start_chain, goal_chain, bridge_key)
        for seg in path_segments:
            self.edge_states.setdefault(seg, SBLSegmentState())

        unresolved = [seg for seg in path_segments if not self.edge_states[seg].safe]
        while unresolved:
            seg = max(unresolved, key=self._segment_score)
            collision_point = self._test_segment_once(seg)
            if collision_point is not None:
                return seg, collision_point
            unresolved = [key for key in path_segments if not self.edge_states[key].safe]
        return None, None

    def _find_edge_index(self, chain: List[int], edge_key: frozenset[int]) -> Optional[int]:
        for i in range(1, len(chain)):
            if self._edge_key(chain[i - 1], chain[i]) == edge_key:
                return i
        return None

    def _collect_subtree(self, root_id: int) -> Set[int]:
        stack = [root_id]
        subtree: Set[int] = set()
        while stack:
            node_id = stack.pop()
            if node_id in subtree:
                continue
            subtree.add(node_id)
            stack.extend(self.nodes[node_id].children)
        return subtree

    def _transfer_subtree_after_collision(
        self,
        chain: List[int],
        edge_index: int,
        opposite_bridge: int,
        new_tree: int,
        bridge_key: frozenset[int],
    ) -> None:
        parent = chain[edge_index - 1]
        child = chain[edge_index]
        collision_key = self._edge_key(parent, child)

        self.nodes[parent].children.discard(child)
        if self.nodes[child].parent == parent:
            self.nodes[child].parent = None
        self.edge_states.pop(collision_key, None)

        chain_sub = chain[edge_index:]  # child -> ... -> bridge endpoint on this side
        subtree_nodes = self._collect_subtree(child)

        for node_id in subtree_nodes:
            self._move_node_between_trees(node_id, new_tree)

        for i in range(len(chain_sub) - 1):
            old_parent = chain_sub[i]
            old_child = chain_sub[i + 1]
            self.nodes[old_parent].children.discard(old_child)

        rev_chain = list(reversed(chain_sub))  # bridge endpoint -> ... -> child
        bridge_side = rev_chain[0]
        self.nodes[bridge_side].parent = opposite_bridge
        self.nodes[opposite_bridge].children.add(bridge_side)
        self.edge_states.setdefault(bridge_key, SBLSegmentState())

        for i in range(1, len(rev_chain)):
            node_id = rev_chain[i]
            new_parent = rev_chain[i - 1]
            self.nodes[node_id].parent = new_parent
            self.nodes[new_parent].children.add(node_id)

        self.transfer_count += 1

    def _handle_path_collision(
        self,
        collision_key: frozenset[int],
        start_chain: List[int],
        goal_chain: List[int],
        bridge_key: frozenset[int],
    ) -> None:
        self.raw_solution_path = []
        self.solution_path = []
        if collision_key == bridge_key:
            self.edge_states.pop(bridge_key, None)
            return

        start_idx = self._find_edge_index(start_chain, collision_key)
        if start_idx is not None:
            self._transfer_subtree_after_collision(
                start_chain,
                start_idx,
                opposite_bridge=goal_chain[-1],
                new_tree=1,
                bridge_key=bridge_key,
            )
            return

        goal_idx = self._find_edge_index(goal_chain, collision_key)
        if goal_idx is not None:
            self._transfer_subtree_after_collision(
                goal_chain,
                goal_idx,
                opposite_bridge=start_chain[-1],
                new_tree=0,
                bridge_key=bridge_key,
            )
            return

        a_id, b_id = tuple(collision_key)
        if self.nodes[a_id].parent == b_id:
            self.nodes[b_id].children.discard(a_id)
            self.nodes[a_id].parent = None
        elif self.nodes[b_id].parent == a_id:
            self.nodes[a_id].children.discard(b_id)
            self.nodes[b_id].parent = None
        self.edge_states.pop(collision_key, None)

    def _attempt_connection(self, new_node_id: int) -> Optional[StepResult]:
        this_tree = self.nodes[new_node_id].tree_id
        other_tree = 1 - this_tree
        new_point = self.nodes[new_node_id].point
        same_cell = self._cell_for_point(new_point)

        candidate_ids: List[int] = []
        same_cell_ids = list(self.cell_maps[other_tree].get(same_cell, set()))
        if same_cell_ids:
            best_same_cell = min(same_cell_ids, key=lambda node_id: linf_dist(new_point, self.nodes[node_id].point))
            candidate_ids.append(best_same_cell)

        random_other = self._random_tree_node(other_tree)
        if random_other is not None and random_other not in candidate_ids:
            candidate_ids.append(random_other)

        attempted_edges: List[Edge] = []
        last_rejected_point: Optional[Point] = None
        for other_id in candidate_ids:
            other_point = self.nodes[other_id].point
            if linf_dist(new_point, other_point) >= self.rho:
                continue

            self.bridge_attempts += 1
            if this_tree == 0:
                start_bridge, goal_bridge = new_node_id, other_id
            else:
                start_bridge, goal_bridge = other_id, new_node_id

            start_chain = self._chain_root_to_node(start_bridge)
            goal_chain = self._chain_root_to_node(goal_bridge)
            bridge_key = self._edge_key(start_bridge, goal_bridge)
            self.edge_states.setdefault(bridge_key, SBLSegmentState())

            collision_key, collision_point = self._test_candidate_path(start_chain, goal_chain, bridge_key)
            bridge_edge = (self.nodes[start_bridge].point, self.nodes[goal_bridge].point)

            if collision_key is None:
                self.raw_solution_path = self._build_solution_path(start_chain, goal_chain)
                self.solution_path = self._optimize_solution_path(self.raw_solution_path)
                self.found_path = True
                self.done = True
                return StepResult(edge=bridge_edge, done=True, found_path=True)

            attempted_edges.append(bridge_edge)
            last_rejected_point = collision_point
            self.last_collision_point = collision_point
            self._handle_path_collision(collision_key, start_chain, goal_chain, bridge_key)
            if collision_key == bridge_key:
                continue
            return StepResult(edge=bridge_edge, rejected_point=collision_point)

        if attempted_edges:
            if len(attempted_edges) == 1:
                return StepResult(edge=attempted_edges[0], rejected_point=last_rejected_point)
            return StepResult(edges=attempted_edges, rejected_point=last_rejected_point)
        return None

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=False)

        self.iteration += 1

        new_node_id, expansion_edge = self._expand_tree()
        if new_node_id is None:
            return StepResult()

        connect_result = self._attempt_connection(new_node_id)
        edges: List[Edge] = [expansion_edge] if expansion_edge is not None else []

        if connect_result is not None:
            if connect_result.edge is not None:
                edges.append(connect_result.edge)
            if connect_result.edges:
                edges.extend(connect_result.edges)
            if len(edges) > 1:
                connect_result.edges = edges
                connect_result.edge = None
            elif len(edges) == 1 and connect_result.edge is None:
                connect_result.edge = edges[0]
            return connect_result

        if len(edges) == 1:
            return StepResult(edge=edges[0])
        if edges:
            return StepResult(edges=edges)
        return StepResult()

    def extract_path(self) -> List[Tuple[int, int]]:
        return list(self.solution_path)

    def get_status(self) -> str:
        if self.found_path:
            return (
                f"SBL: iter {self.iteration}/{self.max_iters}, "
                f"T0={len(self.tree_nodes[0])}, T1={len(self.tree_nodes[1])}, "
                f"checks {self.lazy_checks}, FOUND"
            )
        return (
            f"SBL: iter {self.iteration}/{self.max_iters}, "
            f"T0={len(self.tree_nodes[0])}, T1={len(self.tree_nodes[1])}, "
            f"checks {self.lazy_checks}, transfers {self.transfer_count}"
        )

    @staticmethod
    def get_params_widget() -> QWidget:
        return SBLParamsWidget()

    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'SBLPlanner':
        params = params_widget.get_params()
        return SBLPlanner(occ, start, goal, **params)


# ============================================================================
# A* - A-Star Search
# ============================================================================

class AStarParamsWidget(QWidget):
    """Parameters widget for A* planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_grid_size = QSpinBox()
        self.spin_grid_size.setRange(1, 20)
        self.spin_grid_size.setValue(5)
        self.spin_grid_size.setToolTip("Grid cell size (lower = finer but slower)")
        
        self.check_diagonal = QCheckBox()
        self.check_diagonal.setChecked(True)
        self.check_diagonal.setToolTip("Allow diagonal movement")
        
        layout.addRow("Grid size:", self.spin_grid_size)
        layout.addRow("Diagonal:", self.check_diagonal)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'grid_size': self.spin_grid_size.value(),
            'allow_diagonal': self.check_diagonal.isChecked(),
        }


class AStarPlanner(BasePlanner):
    """A* - Classic heuristic search algorithm."""
    
    name = "A*"
    description = "Grid-based heuristic search"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 grid_size: int = 5, allow_diagonal: bool = True):
        super().__init__(occ, start, goal)
        
        self.grid_size = grid_size
        self.allow_diagonal = allow_diagonal
        
        # Convert to grid coordinates
        self.grid_w = int(np.ceil(self.w / grid_size))
        self.grid_h = int(np.ceil(self.h / grid_size))
        self.start_grid = (
            min(start[0] // grid_size, self.grid_w - 1),
            min(start[1] // grid_size, self.grid_h - 1),
        )
        self.goal_grid = (
            min(goal[0] // grid_size, self.grid_w - 1),
            min(goal[1] // grid_size, self.grid_h - 1),
        )
        
        # Pre-compute grid occupancy
        self.grid_occ = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                # Check if any cell in this grid block is occupied
                y1, y2 = gy * grid_size, min((gy + 1) * grid_size, self.h)
                x1, x2 = gx * grid_size, min((gx + 1) * grid_size, self.w)
                if np.any(self.occ[y1:y2, x1:x2] > 0):
                    self.grid_occ[gy, gx] = True
        
        # A* data structures
        import heapq
        self.open_set: List[Tuple[float, Tuple[int, int]]] = []
        self.came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.g_score: Dict[Tuple[int, int], float] = {self.start_grid: 0}
        self.f_score: Dict[Tuple[int, int], float] = {self.start_grid: self._heuristic(self.start_grid)}
        heapq.heappush(self.open_set, (self.f_score[self.start_grid], self.start_grid))
        self.closed_set: Set[Tuple[int, int]] = set()
        self.path_grid: List[Tuple[int, int]] = []
        
    def _heuristic(self, pos: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        dx = abs(pos[0] - self.goal_grid[0])
        dy = abs(pos[1] - self.goal_grid[1])
        if self.allow_diagonal:
            return self.grid_size * np.sqrt(dx*dx + dy*dy)
        return self.grid_size * (dx + dy)
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighbors with costs."""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if self.allow_diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                if self.grid_occ[ny, nx]:
                    continue

                # Block diagonal corner cutting through occupied orthogonal neighbors.
                if dx != 0 and dy != 0:
                    side_a = (pos[0] + dx, pos[1])
                    side_b = (pos[0], pos[1] + dy)
                    if self.grid_occ[side_a[1], side_a[0]] or self.grid_occ[side_b[1], side_b[0]]:
                        continue

                cost = self.grid_size * np.sqrt(dx*dx + dy*dy)
                neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def step_once(self) -> StepResult:
        import heapq
        
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        if not self.open_set:
            self.done = True
            return StepResult(done=True, found_path=False)
        
        _, current = heapq.heappop(self.open_set)
        
        if current == self.goal_grid:
            # Reconstruct path
            self.path_grid = [current]
            while current in self.came_from:
                current = self.came_from[current]
                self.path_grid.append(current)
            self.path_grid.reverse()
            self.found_path = True
            self.done = True
            return StepResult(done=True, found_path=True)
        
        if current in self.closed_set:
            return StepResult()
        
        self.closed_set.add(current)
        
        edge = None
        for neighbor, cost in self._get_neighbors(current):
            if neighbor in self.closed_set:
                continue
            
            tentative_g = self.g_score.get(current, float('inf')) + cost
            
            if tentative_g < self.g_score.get(neighbor, float('inf')):
                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g
                self.f_score[neighbor] = tentative_g + self._heuristic(neighbor)
                heapq.heappush(self.open_set, (self.f_score[neighbor], neighbor))
                
                # Create edge for visualization
                p1 = (current[0] * self.grid_size + self.grid_size // 2,
                      current[1] * self.grid_size + self.grid_size // 2)
                p2 = (neighbor[0] * self.grid_size + self.grid_size // 2,
                      neighbor[1] * self.grid_size + self.grid_size // 2)
                edge = (p1, p2)
        
        return StepResult(edge=edge)
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if not self.path_grid:
            return []
        return [(gx * self.grid_size + self.grid_size // 2,
                 gy * self.grid_size + self.grid_size // 2) for gx, gy in self.path_grid]
    
    def get_status(self) -> str:
        return f"A*: explored {len(self.closed_set)}, open {len(self.open_set)}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return AStarParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'AStarPlanner':
        params = params_widget.get_params()
        return AStarPlanner(occ, start, goal, **params)


# ============================================================================
# Dijkstra's Algorithm
# ============================================================================

class DijkstraParamsWidget(QWidget):
    """Parameters widget for Dijkstra planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_grid_size = QSpinBox()
        self.spin_grid_size.setRange(1, 20)
        self.spin_grid_size.setValue(5)
        self.spin_grid_size.setToolTip("Grid cell size")
        
        self.check_diagonal = QCheckBox()
        self.check_diagonal.setChecked(True)
        self.check_diagonal.setToolTip("Allow diagonal movement")
        
        layout.addRow("Grid size:", self.spin_grid_size)
        layout.addRow("Diagonal:", self.check_diagonal)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'grid_size': self.spin_grid_size.value(),
            'allow_diagonal': self.check_diagonal.isChecked(),
        }


class DijkstraPlanner(BasePlanner):
    """Dijkstra's Algorithm - Grid-based uniform-cost pathfinding."""
    
    name = "Dijkstra"
    description = "Grid-based uniform-cost search"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 grid_size: int = 5, allow_diagonal: bool = True):
        super().__init__(occ, start, goal)
        
        self.grid_size = grid_size
        self.allow_diagonal = allow_diagonal
        
        # Convert to grid coordinates
        self.grid_w = int(np.ceil(self.w / grid_size))
        self.grid_h = int(np.ceil(self.h / grid_size))
        self.start_grid = (
            min(start[0] // grid_size, self.grid_w - 1),
            min(start[1] // grid_size, self.grid_h - 1),
        )
        self.goal_grid = (
            min(goal[0] // grid_size, self.grid_w - 1),
            min(goal[1] // grid_size, self.grid_h - 1),
        )
        
        # Pre-compute grid occupancy
        self.grid_occ = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                y1, y2 = gy * grid_size, min((gy + 1) * grid_size, self.h)
                x1, x2 = gx * grid_size, min((gx + 1) * grid_size, self.w)
                if np.any(self.occ[y1:y2, x1:x2] > 0):
                    self.grid_occ[gy, gx] = True
        
        # Dijkstra data structures
        import heapq
        self.open_set: List[Tuple[float, Tuple[int, int]]] = []
        self.came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.dist: Dict[Tuple[int, int], float] = {self.start_grid: 0}
        heapq.heappush(self.open_set, (0, self.start_grid))
        self.closed_set: Set[Tuple[int, int]] = set()
        self.path_grid: List[Tuple[int, int]] = []
        
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if self.allow_diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                if self.grid_occ[ny, nx]:
                    continue

                if dx != 0 and dy != 0:
                    side_a = (pos[0] + dx, pos[1])
                    side_b = (pos[0], pos[1] + dy)
                    if self.grid_occ[side_a[1], side_a[0]] or self.grid_occ[side_b[1], side_b[0]]:
                        continue

                cost = self.grid_size * np.sqrt(dx*dx + dy*dy)
                neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def step_once(self) -> StepResult:
        import heapq
        
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        if not self.open_set:
            self.done = True
            return StepResult(done=True, found_path=False)
        
        current_dist, current = heapq.heappop(self.open_set)
        
        if current == self.goal_grid:
            self.path_grid = [current]
            while current in self.came_from:
                current = self.came_from[current]
                self.path_grid.append(current)
            self.path_grid.reverse()
            self.found_path = True
            self.done = True
            return StepResult(done=True, found_path=True)
        
        if current in self.closed_set:
            return StepResult()
        
        self.closed_set.add(current)
        
        edge = None
        for neighbor, cost in self._get_neighbors(current):
            if neighbor in self.closed_set:
                continue
            
            new_dist = current_dist + cost
            
            if new_dist < self.dist.get(neighbor, float('inf')):
                self.came_from[neighbor] = current
                self.dist[neighbor] = new_dist
                heapq.heappush(self.open_set, (new_dist, neighbor))
                
                p1 = (current[0] * self.grid_size + self.grid_size // 2,
                      current[1] * self.grid_size + self.grid_size // 2)
                p2 = (neighbor[0] * self.grid_size + self.grid_size // 2,
                      neighbor[1] * self.grid_size + self.grid_size // 2)
                edge = (p1, p2)
        
        return StepResult(edge=edge)
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if not self.path_grid:
            return []
        return [(gx * self.grid_size + self.grid_size // 2,
                 gy * self.grid_size + self.grid_size // 2) for gx, gy in self.path_grid]
    
    def get_status(self) -> str:
        return f"Dijkstra: explored {len(self.closed_set)}, open {len(self.open_set)}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return DijkstraParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'DijkstraPlanner':
        params = params_widget.get_params()
        return DijkstraPlanner(occ, start, goal, **params)


# ============================================================================
# APF - Artificial Potential Field
# ============================================================================

class APFParamsWidget(QWidget):
    """Parameters widget for APF planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_step_size = QDoubleSpinBox()
        self.spin_step_size.setRange(0.5, 20.0)
        self.spin_step_size.setSingleStep(0.5)
        self.spin_step_size.setValue(5.0)
        self.spin_step_size.setToolTip("Step size for gradient descent")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 50000)
        self.spin_max_iters.setValue(5000)
        self.spin_max_iters.setToolTip("Maximum iterations")
        
        self.spin_goal_gain = QDoubleSpinBox()
        self.spin_goal_gain.setRange(0.1, 100.0)
        self.spin_goal_gain.setSingleStep(1.0)
        self.spin_goal_gain.setValue(5.0)
        self.spin_goal_gain.setToolTip("Attractive force gain")
        
        self.spin_obstacle_gain = QDoubleSpinBox()
        self.spin_obstacle_gain.setRange(1.0, 10000.0)
        self.spin_obstacle_gain.setSingleStep(100.0)
        self.spin_obstacle_gain.setValue(1000.0)
        self.spin_obstacle_gain.setToolTip("Repulsive force gain")
        
        self.spin_obstacle_dist = QSpinBox()
        self.spin_obstacle_dist.setRange(5, 100)
        self.spin_obstacle_dist.setValue(30)
        self.spin_obstacle_dist.setToolTip("Obstacle influence distance")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducible escape perturbations")
        
        layout.addRow("Step size:", self.spin_step_size)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Goal gain:", self.spin_goal_gain)
        layout.addRow("Obstacle gain:", self.spin_obstacle_gain)
        layout.addRow("Obstacle dist:", self.spin_obstacle_dist)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'step_size': self.spin_step_size.value(),
            'max_iters': self.spin_max_iters.value(),
            'goal_gain': self.spin_goal_gain.value(),
            'obstacle_gain': self.spin_obstacle_gain.value(),
            'obstacle_dist': self.spin_obstacle_dist.value(),
            'seed': self.spin_seed.value(),
        }


class APFPlanner(BasePlanner):
    """APF - Artificial Potential Field planner.
    
    Uses attractive force from goal and repulsive forces from obstacles.
    """
    
    name = "APF"
    description = "Artificial Potential Field"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 step_size: float = 5.0, max_iters: int = 2000,
                 goal_gain: float = 5.0, obstacle_gain: float = 1000.0,
                 obstacle_dist: int = 30, seed: int = 42):
        super().__init__(occ, start, goal)
        
        self.step_size = step_size
        self.max_iters = max_iters
        self.goal_gain = goal_gain
        self.obstacle_gain = obstacle_gain
        self.obstacle_dist = obstacle_dist
        self.rng = np.random.default_rng(seed)
        
        # Current position
        self.pos = np.array([float(start[0]), float(start[1])])
        self.path: List[Tuple[int, int]] = [start]
        
        # Compute distance field
        free_space = (self.occ == 0).astype(np.uint8)
        self.dist_field = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        
        # Goal tolerance
        self.goal_tolerance = 10.0
        
        # Track if stuck
        self.stuck_counter = 0
        self.last_pos = self.pos.copy()
        
    def _attractive_force(self) -> np.ndarray:
        """Compute attractive force towards goal."""
        diff = np.array([self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]])
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(2)
        return self.goal_gain * diff / dist
    
    def _repulsive_force(self) -> np.ndarray:
        """Compute repulsive force from obstacles."""
        x, y = int(np.clip(self.pos[0], 0, self.w - 1)), int(np.clip(self.pos[1], 0, self.h - 1))
        dist = self.dist_field[y, x]
        
        if dist >= self.obstacle_dist or dist < 1e-6:
            return np.zeros(2)
        
        # Compute gradient of distance field
        grad_x = 0.0
        grad_y = 0.0
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                if dx != 0:
                    grad_x += dx * self.dist_field[ny, nx]
                if dy != 0:
                    grad_y += dy * self.dist_field[ny, nx]
        
        grad = np.array([grad_x, grad_y])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            grad = grad / grad_norm
        
        # Repulsive force magnitude
        force_mag = self.obstacle_gain * (1.0 / dist - 1.0 / self.obstacle_dist) / (dist * dist)
        
        return force_mag * grad
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=self.found_path)
        
        # Check if reached goal
        dist_to_goal = np.sqrt((self.pos[0] - self.goal[0])**2 + (self.pos[1] - self.goal[1])**2)
        if dist_to_goal < self.goal_tolerance:
            self.path.append(self.goal)
            self.found_path = True
            self.done = True
            return StepResult(done=True, found_path=True)
        
        # Compute total force
        f_att = self._attractive_force()
        f_rep = self._repulsive_force()
        force = f_att + f_rep
        
        # Add random perturbation if stuck
        if self.stuck_counter > 20:
            force += self.rng.normal(0.0, 5.0, 2)
            self.stuck_counter = 0
        
        # Normalize and step
        force_norm = np.linalg.norm(force)
        if force_norm > 1e-6:
            direction = force / force_norm
            new_pos = self.pos + direction * self.step_size
        else:
            new_pos = self.pos
        
        # Clamp to bounds
        new_pos[0] = np.clip(new_pos[0], 0, self.w - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.h - 1)
        
        # Check collision
        new_x, new_y = int(new_pos[0]), int(new_pos[1])
        if self.occ[new_y, new_x] > 0:
            # Stuck in obstacle, try random direction
            self.stuck_counter += 5
            return StepResult(rejected_point=(new_x, new_y))
        
        # Check if making progress
        move_dist = np.linalg.norm(new_pos - self.last_pos)
        if move_dist < 0.5:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        old_pos = (int(self.pos[0]), int(self.pos[1]))
        self.pos = new_pos
        self.last_pos = self.pos.copy()
        new_point = (int(self.pos[0]), int(self.pos[1]))
        
        if new_point != self.path[-1]:
            self.path.append(new_point)
        
        return StepResult(edge=(old_pos, new_point))
    
    def extract_path(self) -> List[Tuple[int, int]]:
        return self.path
    
    def get_status(self) -> str:
        dist = np.sqrt((self.pos[0] - self.goal[0])**2 + (self.pos[1] - self.goal[1])**2)
        status = "FOUND" if self.found_path else ("stuck" if self.stuck_counter > 10 else "moving")
        return f"APF: iter {self.iteration}, dist: {dist:.0f}, {status}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return APFParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'APFPlanner':
        params = params_widget.get_params()
        return APFPlanner(occ, start, goal, **params)


# ============================================================================
# FMT* - Fast Marching Tree
# ============================================================================

class FMTStarParamsWidget(QWidget):
    """Parameters widget for FMT* planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_samples = QSpinBox()
        self.spin_num_samples.setRange(50, 5000)
        self.spin_num_samples.setValue(400)
        self.spin_num_samples.setToolTip("Number of random samples (higher = more robust, slower)")
        
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.0, 300.0)  # 0 = auto
        self.spin_radius.setSingleStep(10.0)
        self.spin_radius.setValue(0.0)  # Auto by default
        self.spin_radius.setToolTip("Connection radius (0 = auto-compute FMT* radius for the 2D free space)")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Num samples:", self.spin_num_samples)
        layout.addRow("Radius (0=auto):", self.spin_radius)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_samples': self.spin_num_samples.value(),
            'radius': self.spin_radius.value() if self.spin_radius.value() > 0 else None,
            'seed': self.spin_seed.value(),
        }


class FMTStarPlanner(BasePlanner):
    """FMT* - Fast Marching Tree algorithm.
    
    Samples points first, then grows a tree using dynamic-programming-style
    wavefront expansion with lazy collision checking.
    """
    
    name = "FMT*"
    description = "Fast Marching Tree with uniform free-space sampling and lazy wavefront expansion"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_samples: int = 400, radius: Optional[float] = None, seed: int = 42):
        super().__init__(occ, start, goal)
        
        self.num_samples = int(max(50, num_samples))
        self.rng = np.random.default_rng(seed)
        self.samples: List[np.ndarray] = [np.array(start, dtype=float)]
        self._sample_points_uniform()
        self.samples.append(np.array(goal, dtype=float))
        self.goal_idx = len(self.samples) - 1

        self.num_nodes = len(self.samples)
        self.free_space_volume = float(np.count_nonzero(~self.occ))
        self.radius = self._compute_connection_radius(radius)

        self.V_open: Set[int] = {0}
        self.V_closed: Set[int] = set()
        self.V_unvisited: Set[int] = set(range(1, self.num_nodes))

        self.parent: Dict[int, int] = {0: -1}
        self.cost: Dict[int, float] = {0: 0.0}
        self.open_heap: List[Tuple[float, int]] = [(0.0, 0)]

        self.neighbors: Dict[int, List[Tuple[int, float]]] = self._precompute_neighbors()
        self.collision_cache: Dict[Tuple[int, int], bool] = {}
        self.collision_checks = 0

    def _sample_points_uniform(self) -> None:
        """Sample free states uniformly from the occupancy grid."""
        attempts = 0
        seen: Set[Point] = {self.start, self.goal}
        while len(self.samples) < self.num_samples + 1 and attempts < self.num_samples * 50:
            attempts += 1
            x = int(self.rng.integers(0, self.w))
            y = int(self.rng.integers(0, self.h))
            point = (x, y)
            if point in seen or self.occ[y, x]:
                continue
            seen.add(point)
            self.samples.append(np.array(point, dtype=float))

    @staticmethod
    def _unit_ball_volume(dimension: int) -> float:
        if dimension == 0:
            return 1.0
        if dimension == 1:
            return 2.0
        return 2.0 * np.pi / dimension * FMTStarPlanner._unit_ball_volume(dimension - 2)

    def _compute_connection_radius(self, radius: Optional[float]) -> float:
        if radius is not None and radius > 0:
            return float(radius)

        dimension = 2
        n = max(2, self.num_samples)
        unit_ball_volume = self._unit_ball_volume(dimension)
        free_volume = max(1.0, self.free_space_volume)
        gamma = 2.0 * np.power(1.0 / dimension, 1.0 / dimension) * np.power(free_volume / unit_ball_volume, 1.0 / dimension)
        return float(1.1 * gamma * np.power(np.log(n) / n, 1.0 / dimension))

    def _precompute_neighbors(self) -> Dict[int, List[Tuple[int, float]]]:
        coords = np.array(self.samples, dtype=np.float64)
        neighborhoods: Dict[int, List[Tuple[int, float]]] = {}
        for idx in range(self.num_nodes):
            deltas = coords - coords[idx]
            dists = np.linalg.norm(deltas, axis=1)
            mask = (dists > 0.0) & (dists <= self.radius)
            nbrs = [(int(j), float(dists[j])) for j in np.where(mask)[0]]
            nbrs.sort(key=lambda item: item[1])
            neighborhoods[idx] = nbrs
        return neighborhoods

    def _neighbors_in_set(self, idx: int, candidate_set: Set[int]) -> List[Tuple[int, float]]:
        return [(other_idx, dist) for other_idx, dist in self.neighbors[idx] if other_idx in candidate_set]

    def _collision_free_indices(self, a_idx: int, b_idx: int) -> bool:
        key = (a_idx, b_idx) if a_idx <= b_idx else (b_idx, a_idx)
        cached = self.collision_cache.get(key)
        if cached is not None:
            return cached

        a = self.samples[a_idx]
        b = self.samples[b_idx]
        free = line_collision_free((int(a[0]), int(a[1])), (int(b[0]), int(b[1])), self.occ)
        self.collision_cache[key] = free
        self.collision_checks += 1
        return free
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        while self.open_heap and self.open_heap[0][1] not in self.V_open:
            heapq.heappop(self.open_heap)

        if not self.open_heap:
            self.done = True
            return StepResult(done=True, found_path=False)
        
        _, z = heapq.heappop(self.open_heap)
        
        z_neighbors = self._neighbors_in_set(z, self.V_unvisited)

        edges: List[Edge] = []
        H_new: List[int] = []

        for x, _ in z_neighbors:
            x_open_neighbors = self._neighbors_in_set(x, self.V_open)
            if not x_open_neighbors:
                continue

            y_min = None
            c_min = float('inf')
            for y, dist_yx in x_open_neighbors:
                potential_cost = self.cost[y] + dist_yx
                if potential_cost < c_min:
                    c_min = potential_cost
                    y_min = y

            if y_min is None:
                continue

            if not self._collision_free_indices(y_min, x):
                continue

            self.parent[x] = y_min
            self.cost[x] = c_min
            H_new.append(x)

            p1 = self.samples[y_min]
            p2 = self.samples[x]
            edges.append(((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))))

        for x in H_new:
            self.V_unvisited.discard(x)
            self.V_open.add(x)
            heapq.heappush(self.open_heap, (self.cost[x], x))

        # In FMT*, z stays in the open wavefront while its neighbors are
        # processed, then moves to closed after the batch update.
        self.V_open.remove(z)
        self.V_closed.add(z)

        if self.goal_idx in H_new:
            self.found_path = True
            self.done = True
            return StepResult(edge=edges[0] if len(edges) == 1 else None, edges=edges if edges else None, done=True, found_path=True)

        return StepResult(edge=edges[0] if len(edges) == 1 else None, edges=edges if edges else None)
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if self.goal_idx not in self.parent:
            return []
        
        path: List[Tuple[int, int]] = []
        current = self.goal_idx
        while current != -1:
            pos = self.samples[current]
            path.append((int(pos[0]), int(pos[1])))
            current = self.parent.get(current, -1)
        path.reverse()
        return path
    
    def get_status(self) -> str:
        status = "FOUND" if self.found_path else "searching"
        return (
            f"FMT*: open {len(self.V_open)}, closed {len(self.V_closed)}, "
            f"unvisited {len(self.V_unvisited)}, cc {self.collision_checks}, {status}"
        )
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return FMTStarParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'FMTStarPlanner':
        params = params_widget.get_params()
        return FMTStarPlanner(occ, start, goal, **params)


# ============================================================================
# BIT* - Batch Informed Trees
# ============================================================================

class BITStarParamsWidget(QWidget):
    """Parameters widget for BIT* planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(50, 2000)
        self.spin_batch_size.setValue(200)
        self.spin_batch_size.setToolTip("Samples per batch")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 50000)
        self.spin_max_iters.setValue(10000)
        self.spin_max_iters.setToolTip("Maximum iterations")
        
        self.spin_rewire_radius = QDoubleSpinBox()
        self.spin_rewire_radius.setRange(0.0, 300.0)
        self.spin_rewire_radius.setSingleStep(5.0)
        self.spin_rewire_radius.setValue(0.0)
        self.spin_rewire_radius.setToolTip("Connection / rewiring radius (0 = auto)")

        self.spin_step_size = QDoubleSpinBox()
        self.spin_step_size.setRange(4.0, 100.0)
        self.spin_step_size.setSingleStep(2.0)
        self.spin_step_size.setValue(26.0)
        self.spin_step_size.setToolTip("Maximum local BIT* connection length")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Batch size:", self.spin_batch_size)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Rewire radius:", self.spin_rewire_radius)
        layout.addRow("Step size:", self.spin_step_size)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'batch_size': self.spin_batch_size.value(),
            'max_iters': self.spin_max_iters.value(),
            'rewire_radius': self.spin_rewire_radius.value() if self.spin_rewire_radius.value() > 0 else None,
            'step_size': self.spin_step_size.value(),
            'seed': self.spin_seed.value(),
        }


class BITStarPlanner(BasePlanner):
    """BIT* - Batch Informed Trees.
    
    BIT* uses batches of samples and processes edges in order of potential
    solution quality. It combines the benefits of RRT* (anytime, asymptotic
    optimality) with graph-based search (ordered edge processing).
    """
    
    name = "BIT*"
    description = "Batch-informed tree search with ordered vertex/edge queues and rewiring in a 2D occupancy-grid adaptation"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 batch_size: int = 200, max_iters: int = 10000, rewire_radius: Optional[float] = None,
                 step_size: float = 26.0, seed: int = 42):
        super().__init__(occ, start, goal)
        
        self.batch_size = int(max(1, batch_size))
        self.max_iters = int(max(1, max_iters))
        self.step_size = float(max(4.0, step_size))
        self.rng = np.random.default_rng(seed)

        self.goal_state = np.array(goal, dtype=float)
        self.start_state = np.array(start, dtype=float)
        self.free_space_volume = float(np.count_nonzero(~self.occ))

        # Tree vertices
        self.V: List[np.ndarray] = [self.start_state.copy()]
        self.parent: Dict[int, Optional[int]] = {0: None}
        self.children: Dict[int, Set[int]] = {0: set()}
        self.g_cost: Dict[int, float] = {0: 0.0}

        # Unconnected samples, including a single goal sample.
        self.X_samples: List[Optional[np.ndarray]] = [self.goal_state.copy()]
        self.sample_point_keys: Set[Point] = {self.goal}

        # Ordered search queues
        self.Q_V: List[Tuple[float, int, int]] = []
        self.Q_E: List[Tuple[float, int, int, int, int]] = []
        self._queue_counter = 0
        self.vertex_expanded_batch: Dict[int, int] = {}

        # Solution tracking
        self.best_cost = float('inf')
        self.goal_idx_in_V: Optional[int] = None
        
        # Informed set parameters
        self.c_min = np.linalg.norm(np.array(goal) - np.array(start))
        self.x_center = (np.array(start) + np.array(goal)) / 2
        self.C = self._rotation_to_world()

        # Batch / RGG tracking
        self.batch_count = 0
        self.user_radius = rewire_radius
        self.r = self._compute_radius(rewire_radius)
        self.edge_collision_checks = 0
        self._new_batch()
    
    def _rotation_to_world(self) -> np.ndarray:
        """Compute rotation matrix from ellipse frame to world frame."""
        diff = np.array(self.goal) - np.array(self.start)
        angle = np.arctan2(diff[1], diff[0])
        return np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    
    def _sample_uniform(self) -> np.ndarray:
        """Sample uniformly in configuration space."""
        x = int(self.rng.integers(0, self.w))
        y = int(self.rng.integers(0, self.h))
        return np.array([x, y], dtype=float)
    
    def _sample_ellipse(self) -> np.ndarray:
        """Sample uniformly in the informed set (ellipse)."""
        if self.best_cost >= float('inf'):
            return self._sample_uniform()
        
        c_best = self.best_cost
        # Semi-axes of the ellipse
        r1 = c_best / 2.0
        r2_sq = c_best**2 - self.c_min**2
        if r2_sq <= 0:
            r2 = 1.0
        else:
            r2 = np.sqrt(r2_sq) / 2.0
        
        # Sample in unit disk
        theta = self.rng.uniform(0, 2 * np.pi)
        rho = np.sqrt(self.rng.uniform(0, 1))
        
        # Scale to ellipse
        x_ell = np.array([r1 * rho * np.cos(theta), r2 * rho * np.sin(theta)])
        
        # Rotate and translate
        x_world = self.C @ x_ell + self.x_center
        
        # Clamp to bounds and discretize to the grid used by the environment.
        x_world[0] = np.clip(np.round(x_world[0]), 0, self.w - 1)
        x_world[1] = np.clip(np.round(x_world[1]), 0, self.h - 1)
        return x_world.astype(float)
    
    def _g_hat(self, v_idx: int) -> float:
        """Estimated cost from start to vertex v."""
        return self.g_cost.get(v_idx, float('inf'))
    
    def _h_hat(self, x: np.ndarray) -> float:
        """Estimated cost from x to goal."""
        return np.linalg.norm(x - self.goal_state)
    
    def _f_hat(self, v_idx: int, x: np.ndarray) -> float:
        """Estimated total cost through edge (v, x)."""
        v = self.V[v_idx]
        g_v = self._g_hat(v_idx)
        c_vx = np.linalg.norm(v - x)  # Edge cost
        h_x = self._h_hat(x)
        return g_v + c_vx + h_x

    def _point_key(self, x: np.ndarray) -> Point:
        return (int(round(float(x[0]))), int(round(float(x[1]))))

    @staticmethod
    def _unit_ball_volume(dimension: int) -> float:
        if dimension == 0:
            return 1.0
        if dimension == 1:
            return 2.0
        return 2.0 * np.pi / dimension * BITStarPlanner._unit_ball_volume(dimension - 2)

    def _compute_radius(self, radius: Optional[float]) -> float:
        if radius is not None and radius > 0:
            return float(radius)

        dimension = 2
        q = max(2, len(self.V) + self.num_unconnected_samples())
        unit_ball_volume = self._unit_ball_volume(dimension)
        free_volume = max(1.0, self.free_space_volume)
        gamma = 2.0 * np.power(1.0 + 1.0 / dimension, 1.0 / dimension) * np.power(free_volume / unit_ball_volume, 1.0 / dimension)
        return float(1.1 * gamma * np.power(np.log(q) / q, 1.0 / dimension))

    def num_unconnected_samples(self) -> int:
        return sum(1 for x in self.X_samples if x is not None)

    def _sample_free(self) -> Optional[np.ndarray]:
        for _ in range(500):
            x_new = self._sample_ellipse()
            ix, iy = self._point_key(x_new)
            if self.occ[iy, ix]:
                continue
            if (ix, iy) == self.start or (ix, iy) in self.sample_point_keys:
                continue
            self.sample_point_keys.add((ix, iy))
            return np.array([ix, iy], dtype=float)
        return None

    def _vertex_key(self, v_idx: int) -> float:
        return self._g_hat(v_idx) + self._h_hat(self.V[v_idx])

    def _edge_key(self, parent_idx: int, target: np.ndarray) -> float:
        return self._f_hat(parent_idx, target)

    def _push_vertex(self, v_idx: int) -> None:
        key = self._vertex_key(v_idx)
        self._queue_counter += 1
        heapq.heappush(self.Q_V, (key, self._queue_counter, v_idx))

    def _push_edge(self, key: float, parent_idx: int, target_kind: int, target_idx: int) -> None:
        self._queue_counter += 1
        heapq.heappush(self.Q_E, (key, self._queue_counter, parent_idx, target_kind, target_idx))

    def _iter_active_samples(self):
        for idx, sample in enumerate(self.X_samples):
            if sample is not None:
                yield idx, sample

    def _near_samples(self, v_idx: int) -> List[Tuple[int, np.ndarray, float]]:
        v = self.V[v_idx]
        near: List[Tuple[int, np.ndarray, float]] = []
        edge_radius = min(self.r, self.step_size)
        for idx, sample in self._iter_active_samples():
            d = float(np.linalg.norm(v - sample))
            if d <= edge_radius:
                near.append((idx, sample, d))
        return near

    def _near_vertices(self, v_idx: int) -> List[Tuple[int, float]]:
        v = self.V[v_idx]
        near: List[Tuple[int, float]] = []
        edge_radius = min(self.r, self.step_size)
        for idx, other in enumerate(self.V):
            if idx == v_idx:
                continue
            d = float(np.linalg.norm(v - other))
            if d <= edge_radius:
                near.append((idx, d))
        return near

    def _is_ancestor(self, ancestor_idx: int, v_idx: int) -> bool:
        current = self.parent.get(v_idx)
        while current is not None:
            if current == ancestor_idx:
                return True
            current = self.parent.get(current)
        return False

    def _set_parent(self, child_idx: int, parent_idx: Optional[int]) -> None:
        old_parent = self.parent.get(child_idx)
        if old_parent is not None:
            self.children.setdefault(old_parent, set()).discard(child_idx)
        self.parent[child_idx] = parent_idx
        if parent_idx is not None:
            self.children.setdefault(parent_idx, set()).add(child_idx)
        self.children.setdefault(child_idx, set())

    def _propagate_cost_delta(self, root_idx: int, delta: float) -> None:
        stack = [root_idx]
        while stack:
            current = stack.pop()
            self.g_cost[current] = self.g_cost.get(current, float('inf')) + delta
            self._push_vertex(current)
            stack.extend(self.children.get(current, ()))

    def _is_collision_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        self.edge_collision_checks += 1
        return line_collision_free(self._point_key(a), self._point_key(b), self.occ)

    def _sample_lower_bound(self, x: np.ndarray) -> float:
        return np.linalg.norm(x - self.start_state) + self._h_hat(x)

    def _prune(self) -> None:
        if not np.isfinite(self.best_cost):
            return

        # Drop samples that can provably not improve the incumbent.
        for idx, sample in enumerate(self.X_samples):
            if sample is None:
                continue
            if self._sample_lower_bound(sample) >= self.best_cost:
                self.X_samples[idx] = None

        # Rebuild ordered queues against the incumbent.
        new_qv: List[Tuple[float, int, int]] = []
        for _, _, v_idx in self.Q_V:
            if self._vertex_key(v_idx) < self.best_cost:
                self._queue_counter += 1
                heapq.heappush(new_qv, (self._vertex_key(v_idx), self._queue_counter, v_idx))
        self.Q_V = new_qv

        new_qe: List[Tuple[float, int, int, int, int]] = []
        for _, _, parent_idx, target_kind, target_idx in self.Q_E:
            if target_kind == 0:
                if target_idx >= len(self.X_samples) or self.X_samples[target_idx] is None:
                    continue
                key = self._edge_key(parent_idx, self.X_samples[target_idx])
            else:
                if target_idx >= len(self.V):
                    continue
                key = self._edge_key(parent_idx, self.V[target_idx])
            if key < self.best_cost:
                self._queue_counter += 1
                heapq.heappush(new_qe, (key, self._queue_counter, parent_idx, target_kind, target_idx))
        self.Q_E = new_qe
    
    def _new_batch(self):
        """Add a new batch of samples."""
        self.batch_count += 1

        for _ in range(self.batch_size):
            x_new = self._sample_free()
            if x_new is not None:
                self.X_samples.append(x_new)

        self.r = self._compute_radius(self.user_radius)
        self.Q_V = []
        self.Q_E = []
        self.vertex_expanded_batch = {}
        for v_idx in range(len(self.V)):
            if self._vertex_key(v_idx) < self.best_cost:
                self._push_vertex(v_idx)
    
    def _expand_vertex(self, v_idx: int):
        """Expand vertex by adding edges to nearby samples."""
        if self.vertex_expanded_batch.get(v_idx) == self.batch_count:
            return
        self.vertex_expanded_batch[v_idx] = self.batch_count

        v = self.V[v_idx]
        g_v = self._g_hat(v_idx)

        if self._vertex_key(v_idx) >= self.best_cost:
            return

        for x_idx, x, dist_vx in self._near_samples(v_idx):
            f_est = g_v + dist_vx + self._h_hat(x)
            if f_est < self.best_cost:
                self._push_edge(f_est, v_idx, 0, x_idx)

        for w_idx, dist_vw in self._near_vertices(v_idx):
            if self._is_ancestor(w_idx, v_idx):
                continue
            g_new = g_v + dist_vw
            if g_new + self._h_hat(self.V[w_idx]) >= self.best_cost:
                continue
            if g_new + 1e-9 >= self.g_cost.get(w_idx, float('inf')):
                continue
            self._push_edge(g_new + self._h_hat(self.V[w_idx]), v_idx, 1, w_idx)
    
    def _is_goal(self, x: np.ndarray) -> bool:
        """Check if x is the goal."""
        return np.linalg.norm(x - np.array(self.goal)) < 1.0
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=self.found_path)
        
        while True:
            while True:
                best_vertex = self._best_vertex_cost()
                best_edge = self._best_edge_cost()
                if not self.Q_V or best_vertex > best_edge:
                    break
                _, _, v_idx = heapq.heappop(self.Q_V)
                if v_idx >= len(self.V):
                    continue
                if self._vertex_key(v_idx) >= self.best_cost:
                    continue
                if self.vertex_expanded_batch.get(v_idx) == self.batch_count:
                    continue
                self._expand_vertex(v_idx)

            if not self.Q_E:
                if self.num_unconnected_samples() == 0:
                    self.done = True
                    return StepResult(done=True, found_path=self.found_path)
                self._new_batch()
                return StepResult()

            _, _, parent_idx, target_kind, target_idx = heapq.heappop(self.Q_E)
            if parent_idx >= len(self.V):
                continue

            parent_state = self.V[parent_idx]
            if target_kind == 0:
                if target_idx >= len(self.X_samples):
                    continue
                x = self.X_samples[target_idx]
                if x is None:
                    continue

                edge_cost = float(np.linalg.norm(parent_state - x))
                g_new = self._g_hat(parent_idx) + edge_cost
                f_new = g_new + self._h_hat(x)
                if f_new >= self.best_cost:
                    continue

                p1 = self._point_key(parent_state)
                p2 = self._point_key(x)
                if not self._is_collision_free(parent_state, x):
                    return StepResult(rejected_point=p2)

                new_v_idx = len(self.V)
                self.V.append(x.copy())
                self.g_cost[new_v_idx] = g_new
                self._set_parent(new_v_idx, parent_idx)
                self.X_samples[target_idx] = None
                self._push_vertex(new_v_idx)

                path_improved = False
                if self._is_goal(x) and g_new < self.best_cost:
                    self.best_cost = g_new
                    self.goal_idx_in_V = new_v_idx
                    self.found_path = True
                    path_improved = True
                    self._prune()

                return StepResult(edge=(p1, p2), path_improved=path_improved)

            if target_idx >= len(self.V):
                continue
            if self._is_ancestor(target_idx, parent_idx) or target_idx == parent_idx:
                continue

            child_state = self.V[target_idx]
            edge_cost = float(np.linalg.norm(parent_state - child_state))
            new_cost = self._g_hat(parent_idx) + edge_cost
            current_cost = self.g_cost.get(target_idx, float('inf'))
            if new_cost + 1e-9 >= current_cost:
                continue
            if new_cost + self._h_hat(child_state) >= self.best_cost:
                continue

            p1 = self._point_key(parent_state)
            p2 = self._point_key(child_state)
            if not self._is_collision_free(parent_state, child_state):
                return StepResult(rejected_point=p2)

            delta = new_cost - current_cost
            self._set_parent(target_idx, parent_idx)
            self._propagate_cost_delta(target_idx, delta)

            path_improved = False
            goal_updated = (
                self.goal_idx_in_V is not None
                and (self.goal_idx_in_V == target_idx or self._is_ancestor(target_idx, self.goal_idx_in_V))
            )
            if goal_updated and self.goal_idx_in_V is not None:
                goal_cost = self.g_cost.get(self.goal_idx_in_V, float('inf'))
                if goal_cost < self.best_cost:
                    self.best_cost = goal_cost
                    self.found_path = True
                    path_improved = True
                    self._prune()

            if self.goal_idx_in_V == target_idx and new_cost < self.best_cost:
                self.best_cost = new_cost
                self.found_path = True
                path_improved = True
                self._prune()

            return StepResult(edge=(p1, p2), path_improved=path_improved)
    
    def _best_vertex_cost(self) -> float:
        """Get the best potential cost through any vertex in Q_V."""
        while self.Q_V:
            key, _, v_idx = self.Q_V[0]
            if v_idx >= len(self.V):
                heapq.heappop(self.Q_V)
                continue
            current_key = self._vertex_key(v_idx)
            if abs(current_key - key) > 1e-9 or self.vertex_expanded_batch.get(v_idx) == self.batch_count:
                heapq.heappop(self.Q_V)
                continue
            return current_key
        return float('inf')
    
    def _best_edge_cost(self) -> float:
        """Get the best potential cost in edge queue."""
        while self.Q_E:
            key, _, parent_idx, target_kind, target_idx = self.Q_E[0]
            if parent_idx >= len(self.V):
                heapq.heappop(self.Q_E)
                continue

            if target_kind == 0:
                if target_idx >= len(self.X_samples) or self.X_samples[target_idx] is None:
                    heapq.heappop(self.Q_E)
                    continue
                current_key = self._edge_key(parent_idx, self.X_samples[target_idx])
            else:
                if target_idx >= len(self.V):
                    heapq.heappop(self.Q_E)
                    continue
                current_key = self._edge_key(parent_idx, self.V[target_idx])

            if abs(current_key - key) > 1e-9:
                heapq.heappop(self.Q_E)
                continue
            return current_key
        return float('inf')
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if self.goal_idx_in_V is None:
            return []
        
        path = []
        current = self.goal_idx_in_V
        while current is not None:
            pos = self.V[current]
            path.append((int(pos[0]), int(pos[1])))
            current = self.parent.get(current)
        path.reverse()
        return path

    def extract_display_path(self) -> List[Tuple[float, float]]:
        path = self.extract_path()
        if len(path) < 3:
            return [(float(p[0]), float(p[1])) for p in path]
        spacing = max(2.0, min(4.0, self.step_size / 4.0))
        return smooth_display_path(path, self.occ, spacing=spacing, iterations=1)

    def extract_tree_edges(self) -> List[Edge]:
        edges: List[Edge] = []
        for child_idx, parent_idx in self.parent.items():
            if parent_idx is None:
                continue
            if child_idx >= len(self.V) or parent_idx >= len(self.V):
                continue
            parent_pt = self._point_key(self.V[parent_idx])
            child_pt = self._point_key(self.V[child_idx])
            edges.append((parent_pt, child_pt))
        return edges
    
    def get_status(self) -> str:
        cost_str = f"{self.best_cost:.1f}" if self.best_cost < float('inf') else "inf"
        return (
            f"BIT*: batch {self.batch_count}, V={len(self.V)}, X={self.num_unconnected_samples()}, "
            f"Qv={len(self.Q_V)}, Qe={len(self.Q_E)}, cc={self.edge_collision_checks}, cost: {cost_str}"
        )
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return BITStarParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'BITStarPlanner':
        params = params_widget.get_params()
        return BITStarPlanner(occ, start, goal, **params)


# ============================================================================
# TrajOpt - Trajectory Optimization via Sequential Convex Programming
# ============================================================================

class TrajOptParamsWidget(QWidget):
    """Parameters widget for TrajOpt planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 200)
        self.spin_num_points.setValue(50)
        self.spin_num_points.setToolTip("Number of waypoints")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(10, 20000)
        self.spin_max_iters.setValue(1000)
        self.spin_max_iters.setToolTip("Maximum iterations")
        
        self.spin_trust_region = QDoubleSpinBox()
        self.spin_trust_region.setRange(1.0, 100.0)
        self.spin_trust_region.setSingleStep(5.0)
        self.spin_trust_region.setValue(20.0)
        self.spin_trust_region.setToolTip("Trust region size")
        
        self.spin_collision_weight = QDoubleSpinBox()
        self.spin_collision_weight.setRange(1.0, 1000.0)
        self.spin_collision_weight.setSingleStep(10.0)
        self.spin_collision_weight.setValue(100.0)
        self.spin_collision_weight.setToolTip("Collision penalty weight")
        
        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Trust region:", self.spin_trust_region)
        layout.addRow("Collision weight:", self.spin_collision_weight)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'trust_region': self.spin_trust_region.value(),
            'collision_weight': self.spin_collision_weight.value(),
        }


class TrajOptPlanner(BasePlanner):
    """TrajOpt - Sequential Convex Programming for trajectory optimization.
    
    Iteratively linearizes constraints and solves convex subproblems.
    Uses penalty method for collision avoidance.
    """
    
    name = "TrajOpt"
    description = "Approximate / experimental sequential convex-style optimization"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 50, max_iters: int = 500, trust_region: float = 20.0,
                 collision_weight: float = 100.0):
        super().__init__(occ, start, goal)
        
        self.num_points = num_points
        self.max_iters = max_iters
        self.trust_region = trust_region
        self.collision_weight = collision_weight
        
        # Compute signed distance field
        free_space = (self.occ == 0).astype(np.uint8)
        self.dist_field = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        
        # Initialize straight-line trajectory
        self.trajectory = np.zeros((num_points, 2), dtype=np.float64)
        for i in range(num_points):
            t = i / (num_points - 1)
            self.trajectory[i, 0] = start[0] + t * (goal[0] - start[0])
            self.trajectory[i, 1] = start[1] + t * (goal[1] - start[1])
        
        # Check for collisions and add perturbation
        self._initialize_with_perturbation()
        
        self.best_trajectory = self.trajectory.copy()
        self.best_cost = float('inf')
        self.converged = False
        self.total_cost = float('inf')
        self.collision_cost = float('inf')
        
    def _initialize_with_perturbation(self):
        """Add sinusoidal perturbation if trajectory has collisions."""
        has_collision = False
        for i in range(self.num_points):
            x, y = int(self.trajectory[i, 0]), int(self.trajectory[i, 1])
            x, y = np.clip(x, 0, self.w-1), np.clip(y, 0, self.h-1)
            if self.occ[y, x] > 0:
                has_collision = True
                break
        
        if has_collision:
            dx = self.goal[0] - self.start[0]
            dy = self.goal[1] - self.start[1]
            path_len = np.sqrt(dx*dx + dy*dy) + 1e-6
            perp_x, perp_y = -dy / path_len, dx / path_len
            amplitude = min(self.h, self.w) * 0.3
            
            for sign in [1, -1]:
                test_traj = self.trajectory.copy()
                for i in range(1, self.num_points - 1):
                    t = i / (self.num_points - 1)
                    offset = sign * amplitude * np.sin(np.pi * t)
                    test_traj[i, 0] += offset * perp_x
                    test_traj[i, 1] += offset * perp_y
                
                collision_free = True
                for i in range(self.num_points):
                    x = int(np.clip(test_traj[i, 0], 0, self.w - 1))
                    y = int(np.clip(test_traj[i, 1], 0, self.h - 1))
                    if self.occ[y, x] > 0:
                        collision_free = False
                        break
                
                if collision_free:
                    self.trajectory = test_traj
                    break
    
    def _get_collision_cost_and_gradient(self, x: float, y: float) -> Tuple[float, float, float]:
        """Get collision cost and gradient at a point."""
        ix = int(np.clip(x, 0, self.w - 1))
        iy = int(np.clip(y, 0, self.h - 1))
        
        dist = self.dist_field[iy, ix]
        
        if self.occ[iy, ix] > 0:
            # Inside obstacle
            return 100.0, 0.0, 0.0
        
        safety_margin = 10.0
        if dist < safety_margin:
            cost = (safety_margin - dist) ** 2
            
            # Numerical gradient
            grad_x, grad_y = 0.0, 0.0
            delta = 1.0
            for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = int(np.clip(x + ddx * delta, 0, self.w - 1))
                ny = int(np.clip(y + ddy * delta, 0, self.h - 1))
                d = self.dist_field[ny, nx]
                if ddx != 0:
                    grad_x += ddx * (safety_margin - d) * (-1)
                if ddy != 0:
                    grad_y += ddy * (safety_margin - d) * (-1)
            
            return cost, grad_x, grad_y
        
        return 0.0, 0.0, 0.0
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        if self.iteration >= self.max_iters:
            self.done = True
            self.trajectory = self.best_trajectory.copy()
            self._check_path_validity()
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        n = self.num_points
        
        # Compute gradient for each waypoint
        grad = np.zeros((n, 2), dtype=np.float64)
        total_smooth = 0.0
        total_collision = 0.0
        
        for i in range(1, n - 1):
            x, y = self.trajectory[i]
            
            # Smoothness gradient
            smooth_grad_x = 2 * (2 * x - self.trajectory[i-1, 0] - self.trajectory[i+1, 0])
            smooth_grad_y = 2 * (2 * y - self.trajectory[i-1, 1] - self.trajectory[i+1, 1])
            
            accel_x = self.trajectory[i+1, 0] - 2 * x + self.trajectory[i-1, 0]
            accel_y = self.trajectory[i+1, 1] - 2 * y + self.trajectory[i-1, 1]
            total_smooth += accel_x ** 2 + accel_y ** 2
            
            # Collision gradient
            coll_cost, coll_grad_x, coll_grad_y = self._get_collision_cost_and_gradient(x, y)
            total_collision += coll_cost
            
            grad[i, 0] = smooth_grad_x + self.collision_weight * coll_grad_x
            grad[i, 1] = smooth_grad_y + self.collision_weight * coll_grad_y
        
        # Trust region constraint
        grad_norm = np.sqrt(np.sum(grad ** 2))
        if grad_norm > self.trust_region:
            grad = grad * (self.trust_region / grad_norm)
        
        # Adaptive step size
        step_size = 0.5 * (1.0 - 0.5 * self.iteration / self.max_iters)
        
        # Update trajectory
        self.trajectory[1:-1] -= step_size * grad[1:-1]
        
        # Clamp to bounds
        self.trajectory[:, 0] = np.clip(self.trajectory[:, 0], 0, self.w - 1)
        self.trajectory[:, 1] = np.clip(self.trajectory[:, 1], 0, self.h - 1)
        
        # Compute total cost
        self.collision_cost = total_collision / max(1, n - 2)
        smooth_cost = total_smooth / max(1, n - 2)
        self.total_cost = smooth_cost + self.collision_weight * self.collision_cost
        
        # Track best
        if self.total_cost < self.best_cost:
            self.best_cost = self.total_cost
            self.best_trajectory = self.trajectory.copy()
        
        # Check validity periodically
        path_improved = False
        if self.iteration % 10 == 0 or self.collision_cost < 0.5:
            old_found = self.found_path
            self._check_path_validity()
            if self.found_path and not old_found:
                path_improved = True
        
        # Check convergence
        if self.collision_cost < 0.1 and self.iteration > 20:
            self._check_path_validity()
            if self.found_path:
                self.converged = True
                self.done = True
        
        # Visualization edge
        idx = self.iteration % (n - 1)
        p1 = (int(self.trajectory[idx, 0]), int(self.trajectory[idx, 1]))
        p2 = (int(self.trajectory[idx + 1, 0]), int(self.trajectory[idx + 1, 1]))
        
        return StepResult(edge=(p1, p2), path_improved=path_improved)
    
    def _check_path_validity(self):
        """Check if trajectory is collision-free."""
        for i in range(len(self.trajectory) - 1):
            p1 = (int(np.rint(self.trajectory[i, 0])), int(np.rint(self.trajectory[i, 1])))
            p2 = (int(np.rint(self.trajectory[i + 1, 0])), int(np.rint(self.trajectory[i + 1, 1])))
            if not line_collision_free(p1, p2, self.occ):
                self.found_path = False
                return
        self.found_path = True
    
    def extract_path(self) -> List[Tuple[int, int]]:
        return [(int(np.rint(p[0])), int(np.rint(p[1]))) for p in self.trajectory]
    
    def get_status(self) -> str:
        status = "converged" if self.converged else ("FOUND" if self.found_path else "optimizing")
        return f"TrajOpt: iter {self.iteration}, collision: {self.collision_cost:.1f}, {status}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return TrajOptParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'TrajOptPlanner':
        params = params_widget.get_params()
        return TrajOptPlanner(occ, start, goal, **params)


# ============================================================================
# PSO - Particle Swarm Optimization
# ============================================================================

class PSOParamsWidget(QWidget):
    """Parameters widget for PSO planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_particles = QSpinBox()
        self.spin_num_particles.setRange(10, 200)
        self.spin_num_particles.setValue(30)
        self.spin_num_particles.setToolTip("Number of particles")
        
        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(5, 100)
        self.spin_num_points.setValue(30)
        self.spin_num_points.setToolTip("Waypoints per path")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(50, 5000)
        self.spin_max_iters.setValue(1200)
        self.spin_max_iters.setToolTip("Maximum iterations")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Particles:", self.spin_num_particles)
        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iters:", self.spin_max_iters)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_particles': self.spin_num_particles.value(),
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'seed': self.spin_seed.value(),
        }


class PSOPlanner(BasePlanner):
    """PSO - Particle Swarm Optimization for path planning.
    
    Each particle represents a complete path. Particles move through the search
    space influenced by their best known position and the swarm's best position.
    """
    
    name = "PSO"
    description = "Particle Swarm Optimization"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_particles: int = 30, num_points: int = 20, max_iters: int = 500,
                 w: float = 0.5, c1: float = 1.5, c2: float = 1.5, seed: int = 42):
        super().__init__(occ, start, goal)
        
        self.num_particles = num_particles
        self.num_points = num_points
        self.max_iters = max_iters
        self.w_inertia = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.rng = np.random.default_rng(seed)
        
        # Initialize particles (each is a path with num_points waypoints)
        # Shape: (num_particles, num_points, 2)
        self.particles = np.zeros((num_particles, num_points, 2))
        self.velocities = np.zeros((num_particles, num_points, 2))
        
        start_arr = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)
        
        for i in range(num_particles):
            # Initialize with straight line + noise
            for j in range(self.num_points):
                t = j / (self.num_points - 1)
                base = start_arr + t * (goal_arr - start_arr)
                noise = self.rng.normal(0, 30, 2) if 0 < j < self.num_points - 1 else 0
                self.particles[i, j] = base + noise
            # Fix start and goal
            self.particles[i, 0] = start_arr
            self.particles[i, -1] = goal_arr
        
        # Initialize velocities
        self.velocities = self.rng.normal(0, 5, (num_particles, num_points, 2))
        self.velocities[:, 0] = 0  # Don't move start
        self.velocities[:, -1] = 0  # Don't move goal
        
        # Personal best for each particle
        self.p_best = self.particles.copy()
        self.p_best_cost = np.full(num_particles, float('inf'))
        self.p_best_valid = np.zeros(num_particles, dtype=bool)
        
        # Global best
        self.g_best = self.particles[0].copy()
        self.g_best_cost = float('inf')
        
        # Distance field for obstacle cost - MUST be initialized before _evaluate_path
        free_space = (self.occ == 0).astype(np.uint8)
        self.dist_field = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)

        # Collision/clearance tuning
        self.collision_penalty = 1e7
        self.clearance_distance = 18.0
        self.clearance_weight = 30.0
        self.display_smoothing_passes = 1
        self.g_best_is_valid = False
        self.early_exit_on_collision = True
        self.social_gain_when_invalid = 0.05

        # Diversification settings to escape repeated invalid corridors.
        self.mutation_std = 24.0
        self.mutation_fraction = 0.50
        self.random_immigrant_fraction = 0.20
        self.stagnation_window = 14
        self.no_improve_iters = 0
        self.valid_particle_count = 0
        self.zero_valid_iters = 0
        self.zero_valid_restart_window = 180
        self.restart_fraction = 0.55

        # Adaptive inertia: broad exploration early, stable convergence later.
        self.w_max = 0.90
        self.w_min = 0.35
        
        # Evaluate initial positions
        for i in range(num_particles):
            cost, is_valid = self._evaluate_path(self.particles[i])
            self.p_best_cost[i] = cost
            self.p_best_valid[i] = is_valid
            self._maybe_update_global_best(self.particles[i], cost, is_valid)
            if np.array_equal(self.g_best, self.particles[i]):
                self.best_particle_idx = i

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return ((angle + np.pi) % (2.0 * np.pi)) - np.pi

    def _segment_collision_samples(self, p1: np.ndarray, p2: np.ndarray) -> int:
        """Adaptive samples based on segment length."""
        seg_len = float(np.linalg.norm(p2 - p1))
        return int(np.clip(np.ceil(seg_len * 0.75), 8, 80))

    def _segment_grid_points(self, p1: np.ndarray, p2: np.ndarray) -> List[Point]:
        """Rasterize a segment and return all visited grid points for robust collision checks."""
        x0 = int(np.clip(round(p1[0]), 0, self.w - 1))
        y0 = int(np.clip(round(p1[1]), 0, self.h - 1))
        x1 = int(np.clip(round(p2[0]), 0, self.w - 1))
        y1 = int(np.clip(round(p2[1]), 0, self.h - 1))

        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy)))
        if steps == 0:
            return [(x0, y0)]

        pts: List[Point] = []
        for i in range(steps + 1):
            t = i / steps
            x = int(np.clip(round(x0 + t * dx), 0, self.w - 1))
            y = int(np.clip(round(y0 + t * dy), 0, self.h - 1))
            if not pts or pts[-1] != (x, y):
                pts.append((x, y))
        return pts

    def _segment_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        for x, y in self._segment_grid_points(p1, p2):
            if self.occ[y, x]:
                return False
        return True

    def _segment_cost_and_collision(self, p1: np.ndarray, p2: np.ndarray, samples: int) -> Tuple[float, bool]:
        """Compute segment clearance/collision cost in a single sampling pass."""
        cost = 0.0
        for x, y in self._segment_grid_points(p1, p2):

            if self.occ[y, x]:
                return self.collision_penalty, True

            d = self.dist_field[y, x]
            if d < self.clearance_distance:
                delta = self.clearance_distance - d
                cost += self.clearance_weight * delta * delta
        return cost, False

    def _maybe_update_global_best(self, candidate: np.ndarray, cost: float, is_valid: bool) -> None:
        """Prefer valid global best paths; allow invalid only as fallback until a valid one exists."""
        if is_valid:
            if (not self.g_best_is_valid) or (cost < self.g_best_cost):
                self.g_best = candidate.copy()
                self.g_best_cost = cost
                self.g_best_is_valid = True
            return

        if (not self.g_best_is_valid) and (cost < self.g_best_cost):
            self.g_best = candidate.copy()
            self.g_best_cost = cost

    def _point_obstacle_penalty(self, x: int, y: int) -> float:
        """Penalty at a single grid point based on obstacle proximity."""
        if self.occ[y, x] > 0:
            return 1000.0

        dist = self.dist_field[y, x]
        if dist < 10:
            return float((10 - dist) * 5)
        return 0.0

    def _segment_obstacle_cost(self, p1: np.ndarray, p2: np.ndarray, samples: int = 12) -> float:
        """Accumulate obstacle penalty along a segment to catch crossings between waypoints."""
        cost = 0.0
        for k in range(samples + 1):
            t = k / samples
            pt = (1.0 - t) * p1 + t * p2
            x = int(np.clip(pt[0], 0, self.w - 1))
            y = int(np.clip(pt[1], 0, self.h - 1))
            cost += self._point_obstacle_penalty(x, y)
        return cost
    
    def _evaluate_path(self, path: np.ndarray) -> Tuple[float, bool]:
        """Evaluate path cost (lower is better) and validity."""
        # Path length cost
        length_cost = 0.0
        for i in range(len(path) - 1):
            length_cost += np.linalg.norm(path[i+1] - path[i])
        
        # Obstacle/clearance cost and segment collision accounting
        obstacle_cost = 0.0
        collision_segments = 0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            samples = self._segment_collision_samples(p1, p2)
            seg_cost, seg_collision = self._segment_cost_and_collision(p1, p2, samples)
            obstacle_cost += seg_cost
            if seg_collision:
                collision_segments += 1
                if self.early_exit_on_collision:
                    remaining = (len(path) - 2) - i
                    obstacle_cost += remaining * self.collision_penalty
                    break
        
        # Smoothness cost
        smooth_cost = 0.0
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
                continue
            raw_diff = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle_diff = self._wrap_angle(raw_diff)
            smooth_cost += abs(angle_diff) * 10

        if collision_segments > 0:
            obstacle_cost += 0.0

        total_cost = length_cost + obstacle_cost + smooth_cost
        return total_cost, collision_segments == 0

    def _mutate_particle(self, idx: int, std: float) -> None:
        """Apply strong random perturbation to internal waypoints to explore new corridors."""
        if self.num_points <= 2:
            return

        self.particles[idx, 1:-1] += self.rng.normal(0.0, std, (self.num_points - 2, 2))
        self.particles[idx, :, 0] = np.clip(self.particles[idx, :, 0], 0, self.w - 1)
        self.particles[idx, :, 1] = np.clip(self.particles[idx, :, 1], 0, self.h - 1)
        self.particles[idx, 0] = np.array(self.start, dtype=float)
        self.particles[idx, -1] = np.array(self.goal, dtype=float)

        self.velocities[idx] = 0.0

    def _randomize_particle(self, idx: int, base_std: float = 42.0) -> None:
        """Reinitialize a particle as random line+noise immigrant to escape swarm collapse."""
        start_arr = np.array(self.start, dtype=float)
        goal_arr = np.array(self.goal, dtype=float)
        for j in range(self.num_points):
            t = j / (self.num_points - 1)
            base = start_arr + t * (goal_arr - start_arr)
            noise = self.rng.normal(0.0, base_std, 2) if 0 < j < self.num_points - 1 else 0.0
            self.particles[idx, j] = base + noise

        self.particles[idx, :, 0] = np.clip(self.particles[idx, :, 0], 0, self.w - 1)
        self.particles[idx, :, 1] = np.clip(self.particles[idx, :, 1], 0, self.h - 1)
        self.particles[idx, 0] = start_arr
        self.particles[idx, -1] = goal_arr
        self.velocities[idx] = self.rng.normal(0.0, 8.0, (self.num_points, 2))
        self.velocities[idx, 0] = 0.0
        self.velocities[idx, -1] = 0.0

    def _inject_diversity(self) -> None:
        """Mutate worst particles when the swarm stagnates or has no valid global best."""
        if self.num_particles <= 2:
            return

        valid_ratio = self.valid_particle_count / max(1, self.num_particles)

        # Adaptive mutation: stronger without valid particles, weaker once valid paths exist.
        if valid_ratio <= 0.0:
            mutation_fraction = min(0.70, self.mutation_fraction + 0.15)
            std = self.mutation_std * 1.7
        elif valid_ratio < 0.2:
            mutation_fraction = self.mutation_fraction
            std = self.mutation_std * 1.2
        else:
            mutation_fraction = max(0.20, self.mutation_fraction - 0.20)
            std = max(10.0, self.mutation_std * 0.6)

        mutate_count = max(1, int(self.num_particles * mutation_fraction))
        worst_indices = np.argsort(self.p_best_cost)[-mutate_count:]

        for idx in worst_indices:
            if idx == getattr(self, 'best_particle_idx', -1):
                continue
            self._mutate_particle(int(idx), std)

        if not self.g_best_is_valid:
            immigrant_count = max(1, int(self.num_particles * self.random_immigrant_fraction))
            immigrants = np.argsort(self.p_best_cost)[-immigrant_count:]
            for idx in immigrants:
                if idx == getattr(self, 'best_particle_idx', -1):
                    continue
                self._randomize_particle(int(idx))

    def _restart_swarm_worst(self) -> None:
        """Hard restart of worst particles after long zero-valid stagnation."""
        if self.num_particles <= 2:
            return

        restart_count = max(1, int(self.num_particles * self.restart_fraction))
        worst_indices = np.argsort(self.p_best_cost)[-restart_count:]

        for idx in worst_indices:
            if idx == getattr(self, 'best_particle_idx', -1):
                continue
            j = int(idx)
            self._randomize_particle(j, base_std=50.0)
            self.p_best[j] = self.particles[j].copy()
            self.p_best_cost[j] = float('inf')
            self.p_best_valid[j] = False

        self.no_improve_iters = 0

    def _smooth_path_for_display(self, path: np.ndarray) -> np.ndarray:
        """Optional light smoothing while preserving collision-free segments."""
        smoothed = path.copy()
        n = len(smoothed)
        if n <= 2:
            return smoothed

        for _ in range(self.display_smoothing_passes):
            for i in range(1, n - 1):
                candidate = 0.25 * smoothed[i - 1] + 0.5 * smoothed[i] + 0.25 * smoothed[i + 1]
                candidate[0] = np.clip(candidate[0], 0, self.w - 1)
                candidate[1] = np.clip(candidate[1], 0, self.h - 1)

                old_point = smoothed[i].copy()
                smoothed[i] = candidate
                if (not self._segment_collision_free(smoothed[i - 1], smoothed[i]) or
                        not self._segment_collision_free(smoothed[i], smoothed[i + 1])):
                    smoothed[i] = old_point

        smoothed[0] = np.array(self.start, dtype=float)
        smoothed[-1] = np.array(self.goal, dtype=float)
        return smoothed
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if self.iteration >= self.max_iters:
            self._check_best_path()
            self.done = True
            return StepResult(done=True, found_path=self.found_path)

        self.iteration += 1
        prev_best_cost = self.g_best_cost
        prev_best_valid = self.g_best_is_valid
        valid_count = 0
        progress = min(1.0, self.iteration / max(1, self.max_iters))
        inertia = self.w_max - (self.w_max - self.w_min) * progress
        valid_ratio = self.valid_particle_count / max(1, self.num_particles)

        # Keep social pull very low while no valid particles exist.
        if valid_ratio <= 0.0:
            social_gain_dynamic = self.social_gain_when_invalid
        elif valid_ratio < 0.25:
            social_gain_dynamic = self.social_gain_when_invalid + (valid_ratio / 0.25) * (0.60 - self.social_gain_when_invalid)
        else:
            social_gain_dynamic = 1.0
        
        # Update velocities and positions for each particle
        for i in range(self.num_particles):
            r1 = self.rng.random((self.num_points, 2))
            r2 = self.rng.random((self.num_points, 2))
            
            # Velocity update
            cognitive = self.c1 * r1 * (self.p_best[i] - self.particles[i])
            social_gain = 1.0 if self.g_best_is_valid else social_gain_dynamic
            social = self.c2 * social_gain * r2 * (self.g_best - self.particles[i])
            self.velocities[i] = inertia * self.velocities[i] + cognitive + social
            
            # Clamp velocity
            self.velocities[i] = np.clip(self.velocities[i], -20, 20)
            self.velocities[i, 0] = 0  # Don't move start
            self.velocities[i, -1] = 0  # Don't move goal
            
            # Position update
            self.particles[i] += self.velocities[i]
            
            # Clamp to bounds
            self.particles[i, :, 0] = np.clip(self.particles[i, :, 0], 0, self.w - 1)
            self.particles[i, :, 1] = np.clip(self.particles[i, :, 1], 0, self.h - 1)
            
            # Evaluate
            cost, is_valid = self._evaluate_path(self.particles[i])
            if is_valid:
                valid_count += 1
            
            # Update personal best
            if is_valid:
                better = (not self.p_best_valid[i]) or (cost < self.p_best_cost[i])
            else:
                better = (not self.p_best_valid[i]) and (cost < self.p_best_cost[i])

            if better:
                self.p_best_cost[i] = cost
                self.p_best[i] = self.particles[i].copy()
                self.p_best_valid[i] = is_valid

            old_best_cost = self.g_best_cost
            self._maybe_update_global_best(self.particles[i], cost, is_valid)
            if self.g_best_cost != old_best_cost and np.array_equal(self.g_best, self.particles[i]):
                self.best_particle_idx = i

        improved = (self.g_best_is_valid and not prev_best_valid) or (self.g_best_cost < prev_best_cost)
        if improved:
            self.no_improve_iters = 0
        else:
            self.no_improve_iters += 1

        self.valid_particle_count = valid_count

        if valid_count == 0:
            self.zero_valid_iters += 1
        else:
            self.zero_valid_iters = 0

        if (not self.g_best_is_valid and self.iteration % 4 == 0) or (self.no_improve_iters >= self.stagnation_window):
            self._inject_diversity()
            if self.no_improve_iters >= self.stagnation_window:
                self.no_improve_iters = 0

        if self.zero_valid_iters >= self.zero_valid_restart_window:
            self._restart_swarm_worst()
            self.zero_valid_iters = 0
        
        # Check if best path is collision-free
        self.found_path = self.g_best_is_valid
        if self.iteration % 10 == 0:
            self._check_best_path()

        if not self.g_best_is_valid:
            # While searching, visualize exploratory particle motion so progress remains visible.
            vis_particle_idx = self.iteration % self.num_particles
            vis_seg_idx = self.iteration % (self.num_points - 1)

            p1_arr = self.particles[vis_particle_idx, vis_seg_idx]
            p2_arr = self.particles[vis_particle_idx, vis_seg_idx + 1]
            p1 = (int(p1_arr[0]), int(p1_arr[1]))
            p2 = (int(p2_arr[0]), int(p2_arr[1]))

            if self._segment_collision_free(p1_arr, p2_arr):
                return StepResult(edge=(p1, p2))

            mid = (int(round((p1[0] + p2[0]) * 0.5)), int(round((p1[1] + p2[1]) * 0.5)))
            return StepResult(rejected_point=mid)
        
        # Visualization: show best particle's path segment
        idx = self.iteration % (self.num_points - 1)
        p1 = (int(self.g_best[idx, 0]), int(self.g_best[idx, 1]))
        p2 = (int(self.g_best[idx + 1, 0]), int(self.g_best[idx + 1, 1]))

        if not self._segment_collision_free(self.g_best[idx], self.g_best[idx + 1]):
            return StepResult()
        
        return StepResult(edge=(p1, p2))
    
    def _check_best_path(self):
        """Check if best path is collision-free."""
        if not self.g_best_is_valid:
            self.found_path = False
            return

        path = self.g_best
        for i in range(len(path) - 1):
            if not self._segment_collision_free(path[i], path[i + 1]):
                self.found_path = False
                return
        self.found_path = True
    
    def extract_path(self) -> List[Tuple[int, int]]:
        path = self.g_best
        if self.done and self.found_path:
            path = self._smooth_path_for_display(path)
        return [(int(p[0]), int(p[1])) for p in path]
    
    def get_status(self) -> str:
        status = "FOUND" if self.found_path else "searching"
        return (
            f"PSO: iter {self.iteration}, best_cost: {self.g_best_cost:.0f}, "
            f"valid_particles: {self.valid_particle_count}/{self.num_particles}, {status}"
        )
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return PSOParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'PSOPlanner':
        params = params_widget.get_params()
        return PSOPlanner(occ, start, goal, **params)


# ============================================================================
# Genetic Algorithm
# ============================================================================

class GeneticParamsWidget(QWidget):
    """Parameters widget for Genetic Algorithm planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_pop_size = QSpinBox()
        self.spin_pop_size.setRange(10, 200)
        self.spin_pop_size.setValue(80)
        self.spin_pop_size.setToolTip("Population size")
        
        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(5, 100)
        self.spin_num_points.setValue(35)
        self.spin_num_points.setToolTip("Waypoints per path")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(50, 5000)
        self.spin_max_iters.setValue(2000)
        self.spin_max_iters.setToolTip("Maximum generations")
        
        self.spin_mutation_rate = QDoubleSpinBox()
        self.spin_mutation_rate.setRange(0.01, 0.5)
        self.spin_mutation_rate.setSingleStep(0.05)
        self.spin_mutation_rate.setValue(0.2)
        self.spin_mutation_rate.setToolTip("Mutation rate")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Population:", self.spin_pop_size)
        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Generations:", self.spin_max_iters)
        layout.addRow("Mutation rate:", self.spin_mutation_rate)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'pop_size': self.spin_pop_size.value(),
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'mutation_rate': self.spin_mutation_rate.value(),
            'seed': self.spin_seed.value(),
        }


class GeneticPlanner(BasePlanner):
    """Genetic Algorithm for path planning.
    
    Uses selection, crossover, and mutation to evolve a population of paths.
    """
    
    name = "Genetic"
    description = "Genetic Algorithm"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 pop_size: int = 50, num_points: int = 20, max_iters: int = 500,
                 mutation_rate: float = 0.1, seed: int = 42):
        super().__init__(occ, start, goal)
        
        self.pop_size = pop_size
        self.num_points = num_points
        self.max_iters = max_iters
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        self.max_mutation_rate = 0.45
        self.rng = np.random.default_rng(seed)
        
        # Initialize population
        self.population = np.zeros((pop_size, num_points, 2))
        start_arr = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)
        
        for i in range(pop_size):
            for j in range(num_points):
                t = j / (num_points - 1)
                base = start_arr + t * (goal_arr - start_arr)
                noise = self.rng.normal(0, 40, 2) if 0 < j < num_points - 1 else 0
                self.population[i, j] = base + noise
            self.population[i, 0] = start_arr
            self.population[i, -1] = goal_arr

        # Distance field must be ready before fitness evaluation.
        free_space = (self.occ == 0).astype(np.uint8)
        self.dist_field = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)

        # Collision/clearance penalties for segment-aware fitness.
        self.collision_penalty = 3000.0
        self.clearance_distance = 14.0
        self.clearance_weight = 20.0
        self.valid_individuals = 0
        self.no_valid_generations = 0
        self.random_immigrant_fraction = 0.20
        self.hard_restart_window = 18
        self.hard_restart_fraction = 0.60
        
        # Fitness scores
        self.fitness = np.zeros(pop_size)
        self._evaluate_population()
        
        # Best individual
        best_idx = np.argmax(self.fitness)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def _segment_collision_samples(self, p1: np.ndarray, p2: np.ndarray) -> int:
        seg_len = float(np.linalg.norm(p2 - p1))
        return int(np.clip(np.ceil(seg_len * 0.75), 8, 80))

    def _segment_cost_and_collision(self, p1: np.ndarray, p2: np.ndarray, samples: int) -> Tuple[float, bool]:
        penalty = 0.0
        for k in range(samples + 1):
            t = k / samples
            x = int(np.clip(round(p1[0] + t * (p2[0] - p1[0])), 0, self.w - 1))
            y = int(np.clip(round(p1[1] + t * (p2[1] - p1[1])), 0, self.h - 1))

            if self.occ[y, x]:
                return self.collision_penalty, True

            d = self.dist_field[y, x]
            if d < self.clearance_distance:
                delta = self.clearance_distance - d
                penalty += self.clearance_weight * delta * delta
        return penalty, False

    def _is_path_valid(self, path: np.ndarray) -> bool:
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            samples = self._segment_collision_samples(p1, p2)
            p1i = (int(p1[0]), int(p1[1]))
            p2i = (int(p2[0]), int(p2[1]))
            if not line_collision_free(p1i, p2i, self.occ, samples=samples):
                return False
        return True

    def _repair_individual(self, individual: np.ndarray, max_passes: int = 2, tries_per_segment: int = 8) -> np.ndarray:
        """Local collision repair: nudges offending waypoints sideways to open free segments."""
        repaired = individual.copy()
        n = len(repaired)
        if n <= 2:
            return repaired

        for _ in range(max_passes):
            changed = False
            for i in range(n - 1):
                p1 = repaired[i]
                p2 = repaired[i + 1]
                if self._is_segment_free(p1, p2):
                    continue

                # Move a non-fixed endpoint of the offending segment.
                if 0 < i + 1 < n - 1:
                    move_idx = i + 1
                    anchor = p1
                elif 0 < i < n - 1:
                    move_idx = i
                    anchor = p2
                else:
                    continue

                base = repaired[move_idx].copy()
                direction = base - anchor
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    direction = np.array([1.0, 0.0])
                    norm = 1.0
                direction = direction / norm
                perp = np.array([-direction[1], direction[0]])

                found = False
                for _try in range(tries_per_segment):
                    step = 8.0 + 3.0 * _try
                    sign = -1.0 if (_try % 2) else 1.0
                    jitter = self.rng.normal(0.0, 5.0, 2)
                    candidate = base + sign * perp * step + jitter
                    candidate[0] = np.clip(candidate[0], 0, self.w - 1)
                    candidate[1] = np.clip(candidate[1], 0, self.h - 1)
                    repaired[move_idx] = candidate

                    left_ok = True
                    right_ok = True
                    if move_idx > 0:
                        left_ok = self._is_segment_free(repaired[move_idx - 1], repaired[move_idx])
                    if move_idx < n - 1:
                        right_ok = self._is_segment_free(repaired[move_idx], repaired[move_idx + 1])

                    if left_ok and right_ok:
                        found = True
                        changed = True
                        break

                if not found:
                    repaired[move_idx] = base

            if not changed:
                break

        repaired[0] = np.array(self.start, dtype=float)
        repaired[-1] = np.array(self.goal, dtype=float)
        return repaired

    def _is_segment_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        samples = self._segment_collision_samples(p1, p2)
        p1i = (int(p1[0]), int(p1[1]))
        p2i = (int(p2[0]), int(p2[1]))
        return line_collision_free(p1i, p2i, self.occ, samples=samples)
    
    def _evaluate_individual(self, path: np.ndarray) -> float:
        """Evaluate fitness (higher is better)."""
        # Path length cost (shorter is better)
        length = 0.0
        obstacle_penalty = 0.0
        collision_segments = 0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            length += np.linalg.norm(p2 - p1)
            samples = self._segment_collision_samples(p1, p2)
            seg_penalty, seg_collision = self._segment_cost_and_collision(p1, p2, samples)
            obstacle_penalty += seg_penalty
            if seg_collision:
                collision_segments += 1
                # Early abort on invalid segment to speed up evaluation.
                break

        # Waypoint proximity penalty
        for point in path:
            x, y = int(np.clip(point[0], 0, self.w - 1)), int(np.clip(point[1], 0, self.h - 1))
            if self.occ[y, x] > 0:
                obstacle_penalty += self.collision_penalty
            else:
                dist = self.dist_field[y, x]
                if dist < 5:
                    obstacle_penalty += (5 - dist) * 10

        if collision_segments > 0:
            obstacle_penalty += collision_segments * self.collision_penalty
        
        # Smoothness (less turning is better)
        smooth_penalty = 0.0
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0.1 and n2 > 0.1:
                cos_angle = np.dot(v1, v2) / (n1 * n2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                smooth_penalty += angle * 5

        total_cost = length + obstacle_penalty + smooth_penalty

        # Strongly prefer fully valid paths, but preserve gradient among invalid ones.
        secondary = 5000.0 / (1.0 + total_cost)

        # Lexicographic fitness shaping:
        # 1) Any fully valid path dominates any invalid path.
        # 2) Invalid paths still keep useful ranking among themselves.
        if collision_segments == 0:
            return 1_000_000.0 + secondary

        invalid_factor = 1.0 + 5.0 * collision_segments
        return secondary / invalid_factor
    
    def _evaluate_population(self):
        """Evaluate fitness of entire population."""
        valid_count = 0
        for i in range(self.pop_size):
            self.fitness[i] = self._evaluate_individual(self.population[i])
            if self._is_path_valid(self.population[i]):
                valid_count += 1
        self.valid_individuals = valid_count
    
    def _select_parents(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tournament selection."""
        def tournament():
            candidates = self.rng.choice(self.pop_size, size=3, replace=False)
            winner = candidates[np.argmax(self.fitness[candidates])]
            return self.population[winner]
        
        return tournament(), tournament()
    
    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Single-point crossover."""
        child = np.zeros_like(p1)
        crossover_point = self.rng.integers(1, self.num_points - 1)
        child[:crossover_point] = p1[:crossover_point]
        child[crossover_point:] = p2[crossover_point:]
        # Ensure start and goal are correct
        child[0] = p1[0]
        child[-1] = p1[-1]
        return child
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        mutated = individual.copy()
        for i in range(1, self.num_points - 1):
            if self.rng.random() < self.mutation_rate:
                std = 22 if self.valid_individuals == 0 else 12
                mutated[i] += self.rng.normal(0, std, 2)
                mutated[i, 0] = np.clip(mutated[i, 0], 0, self.w - 1)
                mutated[i, 1] = np.clip(mutated[i, 1], 0, self.h - 1)
        return mutated

    def _random_individual(self) -> np.ndarray:
        start_arr = np.array(self.start, dtype=float)
        goal_arr = np.array(self.goal, dtype=float)
        ind = np.zeros((self.num_points, 2), dtype=float)
        for j in range(self.num_points):
            t = j / (self.num_points - 1)
            base = start_arr + t * (goal_arr - start_arr)
            noise = self.rng.normal(0, 45, 2) if 0 < j < self.num_points - 1 else 0
            ind[j] = base + noise
        ind[:, 0] = np.clip(ind[:, 0], 0, self.w - 1)
        ind[:, 1] = np.clip(ind[:, 1], 0, self.h - 1)
        ind[0] = start_arr
        ind[-1] = goal_arr
        return ind
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        if self.iteration >= self.max_iters:
            self._check_best_path()
            self.done = True
            return StepResult(done=True, found_path=self.found_path)
        
        # Create new population
        new_population = np.zeros_like(self.population)
        
        # Elitism: keep best individual
        best_idx = np.argmax(self.fitness)
        new_population[0] = self.population[best_idx].copy()
        
        # Generate rest through selection, crossover, mutation
        for i in range(1, self.pop_size):
            p1, p2 = self._select_parents()
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            # Try local repair to turn near-miss children into valid paths.
            if self.valid_individuals == 0 or self.rng.random() < 0.35:
                child = self._repair_individual(child)
            new_population[i] = child
        
        self.population = new_population

        # If no valid individuals for several generations, inject random immigrants.
        if self.valid_individuals == 0:
            self.no_valid_generations += 1
        else:
            self.no_valid_generations = 0

        if self.no_valid_generations >= 10:
            immigrant_count = max(1, int(self.pop_size * self.random_immigrant_fraction))
            for i in range(self.pop_size - immigrant_count, self.pop_size):
                new_population[i] = self._repair_individual(self._random_individual())
            self.population = new_population
            self.no_valid_generations = 5

        # Hard restart if validity is stuck at zero for too long.
        if self.no_valid_generations >= self.hard_restart_window:
            restart_count = max(1, int(self.pop_size * self.hard_restart_fraction))
            for i in range(self.pop_size - restart_count, self.pop_size):
                new_population[i] = self._repair_individual(self._random_individual())
            # Keep elite and reset stagnation counter.
            new_population[0] = self.best_individual.copy()
            self.population = new_population
            self.no_valid_generations = 0

        # Adaptive mutation schedule.
        if self.valid_individuals == 0:
            self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate + 0.02)
        else:
            self.mutation_rate = max(self.base_mutation_rate, self.mutation_rate - 0.01)

        self._evaluate_population()
        
        # Update best
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_individual = self.population[best_idx].copy()
        
        # Check if best path is valid
        self._check_best_path()
        
        # Visualization
        idx = self.iteration % (self.num_points - 1)
        p1 = (int(self.best_individual[idx, 0]), int(self.best_individual[idx, 1]))
        p2 = (int(self.best_individual[idx + 1, 0]), int(self.best_individual[idx + 1, 1]))
        
        return StepResult(edge=(p1, p2))
    
    def _check_best_path(self):
        """Check if best path is collision-free."""
        path = self.best_individual
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            samples = self._segment_collision_samples(p1, p2)
            p1i = (int(p1[0]), int(p1[1]))
            p2i = (int(p2[0]), int(p2[1]))
            if not line_collision_free(p1i, p2i, self.occ, samples=samples):
                self.found_path = False
                return
        self.found_path = True
    
    def extract_path(self) -> List[Tuple[int, int]]:
        return [(int(p[0]), int(p[1])) for p in self.best_individual]
    
    def get_status(self) -> str:
        status = "FOUND" if self.found_path else "evolving"
        return (
            f"GA: gen {self.iteration}, fitness: {self.best_fitness:.2f}, "
            f"valid: {self.valid_individuals}/{self.pop_size}, mut: {self.mutation_rate:.2f}, {status}"
        )
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return GeneticParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'GeneticPlanner':
        params = params_widget.get_params()
        return GeneticPlanner(occ, start, goal, **params)


# ============================================================================
# ITOMP - Incremental Trajectory Optimization for Motion Planning
# ============================================================================

class ITOMPParamsWidget(QWidget):
    """Parameters widget for ITOMP planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 100)
        self.spin_num_points.setValue(30)
        self.spin_num_points.setToolTip("Trajectory waypoints")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 10000)
        self.spin_max_iters.setValue(1000)
        self.spin_max_iters.setToolTip("Maximum iterations")
        
        self.spin_replan_interval = QSpinBox()
        self.spin_replan_interval.setRange(5, 100)
        self.spin_replan_interval.setValue(20)
        self.spin_replan_interval.setToolTip("Replanning interval")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iters:", self.spin_max_iters)
        layout.addRow("Replan interval:", self.spin_replan_interval)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'replan_interval': self.spin_replan_interval.value(),
            'seed': self.spin_seed.value(),
        }


class ITOMPPlanner(BasePlanner):
    """ITOMP - Incremental Trajectory Optimization for Motion Planning.
    
    Combines trajectory optimization with incremental replanning.
    Optimizes trajectory while executing, allowing for reactive behavior.
    """
    
    name = "ITOMP"
    description = "Approximate / experimental incremental trajectory optimization"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 30, max_iters: int = 1000, replan_interval: int = 20,
                 learning_rate: float = 0.3, seed: int = 42):
        super().__init__(occ, start, goal)
        
        self.num_points = num_points
        self.max_iters = max_iters
        self.replan_interval = replan_interval
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        
        # Initialize trajectory as straight line
        self.trajectory = np.zeros((num_points, 2))
        for i in range(num_points):
            t = i / (num_points - 1)
            self.trajectory[i] = np.array(start) * (1 - t) + np.array(goal) * t
        
        # Distance field
        free_space = (self.occ == 0).astype(np.uint8)
        self.dist_field = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        
        # Current execution index (simulates robot progress)
        self.exec_idx = 0
        self.converged = False
        
        # Cost tracking
        self.total_cost = float('inf')
    
    def _compute_obstacle_gradient(self, point: np.ndarray) -> np.ndarray:
        """Compute gradient pushing away from obstacles."""
        x, y = int(np.clip(point[0], 0, self.w - 1)), int(np.clip(point[1], 0, self.h - 1))
        dist = self.dist_field[y, x]
        
        if dist > 20:
            return np.zeros(2)
        
        # Numerical gradient of distance field
        grad = np.zeros(2)
        for dx, dy, dim in [(-1, 0, 0), (1, 0, 0), (0, -1, 1), (0, 1, 1)]:
            nx, ny = np.clip(x + dx, 0, self.w - 1), np.clip(y + dy, 0, self.h - 1)
            grad[dim] += self.dist_field[ny, nx] - dist
        
        # Push away from obstacles (in direction of increasing distance)
        if dist < 5:
            return grad * 50  # Strong push near obstacles
        else:
            return grad * (20 - dist)
    
    def _compute_smoothness_gradient(self, idx: int) -> np.ndarray:
        """Compute gradient for smoothness."""
        if idx <= 0 or idx >= self.num_points - 1:
            return np.zeros(2)
        
        # Acceleration minimization
        accel = self.trajectory[idx - 1] - 2 * self.trajectory[idx] + self.trajectory[idx + 1]
        return accel * 0.5
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        if self.iteration >= self.max_iters:
            self._check_path_validity()
            self.done = True
            return StepResult(done=True, found_path=self.found_path)
        
        # Optimize trajectory points (only future points from exec_idx)
        start_opt = max(1, self.exec_idx)
        end_opt = self.num_points - 1
        
        for i in range(start_opt, end_opt):
            # Obstacle avoidance gradient
            obs_grad = self._compute_obstacle_gradient(self.trajectory[i])
            
            # Smoothness gradient
            smooth_grad = self._compute_smoothness_gradient(i)
            
            # Goal attraction (slight pull toward straight-line path)
            t = i / (self.num_points - 1)
            target = np.array(self.start) * (1 - t) + np.array(self.goal) * t
            goal_grad = (target - self.trajectory[i]) * 0.02
            
            # Combined update
            update = obs_grad + smooth_grad + goal_grad
            self.trajectory[i] += self.learning_rate * update
            
            # Clamp to bounds
            self.trajectory[i, 0] = np.clip(self.trajectory[i, 0], 0, self.w - 1)
            self.trajectory[i, 1] = np.clip(self.trajectory[i, 1], 0, self.h - 1)
        
        # Simulate execution progress (increment every replan_interval iterations)
        if self.iteration % self.replan_interval == 0:
            self.exec_idx = min(self.exec_idx + 1, self.num_points - 2)
        
        # Check path validity
        self._check_path_validity()
        
        # Convergence check
        if self.found_path and self.exec_idx >= self.num_points - 2:
            self.converged = True
            self.done = True
        
        # Visualization
        idx = self.iteration % (self.num_points - 1)
        p1 = (int(self.trajectory[idx, 0]), int(self.trajectory[idx, 1]))
        p2 = (int(self.trajectory[idx + 1, 0]), int(self.trajectory[idx + 1, 1]))
        
        return StepResult(edge=(p1, p2))
    
    def _check_path_validity(self):
        """Check if trajectory is collision-free."""
        for i in range(len(self.trajectory) - 1):
            p1 = (int(self.trajectory[i, 0]), int(self.trajectory[i, 1]))
            p2 = (int(self.trajectory[i + 1, 0]), int(self.trajectory[i + 1, 1]))
            if not line_collision_free(p1, p2, self.occ):
                self.found_path = False
                return
        self.found_path = True
    
    def extract_path(self) -> List[Tuple[int, int]]:
        return [(int(p[0]), int(p[1])) for p in self.trajectory]
    
    def get_status(self) -> str:
        status = "FOUND" if self.found_path else "optimizing"
        progress = self.exec_idx / (self.num_points - 1) * 100
        return f"ITOMP: iter {self.iteration}, exec: {progress:.0f}%, {status}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return ITOMPParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'ITOMPPlanner':
        params = params_widget.get_params()
        return ITOMPPlanner(occ, start, goal, **params)


# ============================================================================
# GPMP - Gaussian Process Motion Planning
# ============================================================================

class GPMPParamsWidget(QWidget):
    """Parameters widget for GPMP planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 100)
        self.spin_num_points.setValue(25)
        self.spin_num_points.setToolTip("Trajectory waypoints")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 10000)
        self.spin_max_iters.setValue(800)
        self.spin_max_iters.setToolTip("Maximum iterations")
        
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 50.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(6.0)
        self.spin_sigma.setToolTip("GP prior sigma. Higher values make the prior softer.")
        
        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iters:", self.spin_max_iters)
        layout.addRow("Prior sigma:", self.spin_sigma)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'sigma': self.spin_sigma.value(),
        }


class GPMPPlanner(BasePlanner):
    """GPMP - Gaussian Process Motion Planning.
    
    Models trajectory as samples from a Gaussian Process prior,
    then optimizes using MAP estimation with obstacle factors.
    """
    
    name = "GPMP"
    description = "Approximate / experimental GP-based local trajectory optimization"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 25, max_iters: int = 800, sigma: float = 10.0,
                 obstacle_weight: float = 100.0):
        super().__init__(occ, start, goal)
        
        self.num_points = num_points
        self.max_iters = max_iters
        self.sigma = sigma
        self.obstacle_weight = obstacle_weight
        self.epsilon = 14.0
        self.learning_rate = 0.2
        self.interp_samples = 4
        self.prior_weight = 1.0 / max(1e-6, self.sigma ** 2)
        
        # Distance field
        free_space = (self.occ == 0).astype(np.uint8)
        self.dist_field = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        self.grad_x = cv2.Sobel(self.dist_field, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        self.grad_y = cv2.Sobel(self.dist_field, cv2.CV_64F, 0, 1, ksize=3) / 8.0

        self.mean_trajectory = np.zeros((num_points, 2), dtype=np.float64)
        start_vec = np.array(start, dtype=np.float64)
        goal_vec = np.array(goal, dtype=np.float64)
        for i in range(num_points):
            t = i / (num_points - 1)
            self.mean_trajectory[i] = start_vec * (1.0 - t) + goal_vec * t

        self.trajectory = self.mean_trajectory.copy()
        self.prior_precision = self._build_prior_precision()
        self.prior_system = self.prior_precision[1:-1, 1:-1] + 1e-6 * np.eye(self.num_points - 2, dtype=np.float64)
        
        self.converged = False
        self.best_trajectory = self.trajectory.copy()
        self.best_cost = float("inf")
        self.best_valid_trajectory: Optional[np.ndarray] = None
        self.best_valid_cost = float("inf")
        self.prev_total_cost: Optional[float] = None
        self.obs_cost = float("inf")
        self.prior_cost = float("inf")
        self._check_path_validity()
        initial_total_cost = self._trajectory_cost(self.trajectory)
        if self.found_path:
            self.best_valid_trajectory = self.trajectory.copy()
            self.best_valid_cost = initial_total_cost
        self.best_cost = initial_total_cost
    
    def _build_prior_precision(self) -> np.ndarray:
        """Build a banded precision approximation for the GP prior."""
        n = self.num_points
        P = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n - 1):
            P[i, i] += 2.0
            P[i, i - 1] -= 1.0
            P[i, i + 1] -= 1.0
        P[0, 0] = 1.0
        P[-1, -1] = 1.0
        return P

    def _obstacle_cost_and_grad(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute obstacle cost and gradient."""
        x, y = int(np.clip(point[0], 0, self.w - 1)), int(np.clip(point[1], 0, self.h - 1))
        dist = self.dist_field[y, x]

        if dist >= self.epsilon:
            return 0.0, np.zeros(2)

        sdf_grad = np.array([self.grad_x[y, x], self.grad_y[y, x]], dtype=np.float64)
        grad_norm = float(np.linalg.norm(sdf_grad))
        if grad_norm > 1e-6:
            sdf_grad = sdf_grad / grad_norm

        cost = 0.5 * self.obstacle_weight * (self.epsilon - dist) ** 2
        grad = -self.obstacle_weight * (self.epsilon - dist) * sdf_grad
        return cost, grad

    def _compute_interpolated_obstacle_factors(self) -> Tuple[float, np.ndarray]:
        """Evaluate obstacle factors on interpolated states and project them back."""
        grad = np.zeros_like(self.trajectory)
        total_cost = 0.0

        for i in range(self.num_points - 1):
            p0 = self.trajectory[i]
            p1 = self.trajectory[i + 1]

            for j in range(1, self.interp_samples + 1):
                tau = j / (self.interp_samples + 1)
                point = (1.0 - tau) * p0 + tau * p1
                cost, point_grad = self._obstacle_cost_and_grad(point)
                if cost <= 0.0:
                    continue
                total_cost += cost
                grad[i] += (1.0 - tau) * point_grad
                grad[i + 1] += tau * point_grad

        return total_cost, grad

    def _compute_prior_cost(self, trajectory: Optional[np.ndarray] = None) -> float:
        """Compute the GP prior cost around the deterministic mean trajectory."""
        traj = self.trajectory if trajectory is None else trajectory
        diff = traj - self.mean_trajectory
        total_cost = 0.0
        for dim in range(2):
            total_cost += 0.5 * float(diff[:, dim] @ (self.prior_precision @ diff[:, dim]))
        return self.prior_weight * total_cost

    def _precondition_obstacle_gradient(self, obstacle_grad: np.ndarray) -> np.ndarray:
        """Approximate the GP covariance action K_w * g from the paper's update rule."""
        if self.num_points <= 2:
            return np.zeros_like(obstacle_grad)

        covariant_grad = np.zeros_like(obstacle_grad)
        for dim in range(2):
            covariant_grad[1:-1, dim] = np.linalg.solve(self.prior_system, obstacle_grad[1:-1, dim])
        return covariant_grad

    def _trajectory_cost(self, trajectory: np.ndarray) -> float:
        """Compute the GPMP objective for selecting the best iterate."""
        old_traj = self.trajectory
        self.trajectory = trajectory
        obstacle_cost, _ = self._compute_interpolated_obstacle_factors()
        prior_cost = self._compute_prior_cost(trajectory)
        self.trajectory = old_traj
        return prior_cost + obstacle_cost

    def _finalize_best_trajectory(self) -> None:
        """Restore the best valid trajectory if one exists."""
        if self.best_valid_trajectory is not None:
            self.trajectory = self.best_valid_trajectory.copy()
            self.found_path = True
            return
        self.trajectory = self.best_trajectory.copy()
        self._check_path_validity()
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        if self.iteration >= self.max_iters:
            self._finalize_best_trajectory()
            self.done = True
            return StepResult(done=True, found_path=self.found_path)
        
        _, obs_grad = self._compute_interpolated_obstacle_factors()
        total_grad = self.prior_weight * (self.trajectory - self.mean_trajectory)
        total_grad += self._precondition_obstacle_gradient(obs_grad)

        total_grad[0] = 0.0
        total_grad[-1] = 0.0
        grad_norm = float(np.linalg.norm(total_grad))
        if grad_norm > 25.0:
            total_grad *= 25.0 / grad_norm

        step_size = self.learning_rate * (0.995 ** min(self.iteration, 400))
        
        # Update (don't move start and goal)
        for i in range(1, self.num_points - 1):
            self.trajectory[i] -= step_size * total_grad[i]
            self.trajectory[i, 0] = np.clip(self.trajectory[i, 0], 0, self.w - 1)
            self.trajectory[i, 1] = np.clip(self.trajectory[i, 1], 0, self.h - 1)
        
        # Check validity
        self._check_path_validity()
        obs_cost, _ = self._compute_interpolated_obstacle_factors()
        prior_cost = self._compute_prior_cost()
        total_cost = prior_cost + obs_cost
        self.obs_cost = obs_cost
        self.prior_cost = prior_cost
        if total_cost < self.best_cost:
            self.best_cost = total_cost
            self.best_trajectory = self.trajectory.copy()
        if self.found_path and total_cost < self.best_valid_cost:
            self.best_valid_cost = total_cost
            self.best_valid_trajectory = self.trajectory.copy()
        
        # Convergence
        cost_change = (
            abs(self.prev_total_cost - total_cost)
            if self.prev_total_cost is not None
            else float("inf")
        )
        self.prev_total_cost = total_cost
        if self.found_path and cost_change < 1e-3 and grad_norm < 1.0 and self.iteration > 40:
            self.converged = True
            self._finalize_best_trajectory()
            self.done = True

        return StepResult()
    
    def _check_path_validity(self):
        """Check if trajectory is collision-free."""
        for i in range(len(self.trajectory) - 1):
            p1 = (int(self.trajectory[i, 0]), int(self.trajectory[i, 1]))
            p2 = (int(self.trajectory[i + 1, 0]), int(self.trajectory[i + 1, 1]))
            if not line_collision_free(p1, p2, self.occ):
                self.found_path = False
                return
        self.found_path = True
    
    def extract_path(self) -> List[Tuple[int, int]]:
        return [(int(p[0]), int(p[1])) for p in self.trajectory]

    def extract_display_path(self) -> List[Tuple[float, float]]:
        return [(float(p[0]), float(p[1])) for p in self.trajectory]
    
    def get_status(self) -> str:
        status = "converged" if self.converged else ("FOUND" if self.found_path else "optimizing")
        return f"GPMP: iter {self.iteration}, prior: {self.prior_cost:.1f}, obs: {self.obs_cost:.1f}, {status}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return GPMPParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'GPMPPlanner':
        params = params_widget.get_params()
        return GPMPPlanner(occ, start, goal, **params)


# =============================================================================
# Planner Registry
# =============================================================================

# Algorithms grouped by category for UI organization
ALGORITHM_GROUPS: List[Tuple[str, List[str]]] = [
    ('Sampling-Based', ['RRT', 'RRT-Connect', 'BiTRRT', 'KPIECE', 'RRT*', 'PRM', 'SBL', 'FMT*', 'BIT*']),
    ('Graph Search', ['A*', 'Dijkstra']),
    ('Potential Field', ['APF']),
    ('Trajectory Optimization', ['CHOMP']),
    ('Approximate / Experimental', ['STOMP', 'TrajOpt', 'ITOMP', 'GPMP', 'PSO', 'Genetic']),
]

# Register all available planners here
AVAILABLE_PLANNERS: Dict[str, Type[BasePlanner]] = {
    'RRT': RRTPlanner,
    'RRT-Connect': RRTConnectPlanner,
    'BiTRRT': BiTRRTPlanner,
    'KPIECE': KPIECEPlanner,
    'RRT*': RRTStarPlanner,
    'PRM': PRMPlanner,
    'SBL': SBLPlanner,
    'FMT*': FMTStarPlanner,
    'BIT*': BITStarPlanner,
    'A*': AStarPlanner,
    'Dijkstra': DijkstraPlanner,
    'APF': APFPlanner,
    'CHOMP': CHOMPPlanner,
    'STOMP': STOMPPlanner,
    'TrajOpt': TrajOptPlanner,
    'ITOMP': ITOMPPlanner,
    'GPMP': GPMPPlanner,
    'PSO': PSOPlanner,
    'Genetic': GeneticPlanner,
}

# Sampling-based algorithms that can be optimized with CHOMP
SAMPLING_BASED_ALGOS: Set[str] = {
    'RRT', 'RRT-Connect', 'BiTRRT', 'KPIECE', 'RRT*', 'PRM', 'SBL', 'FMT*', 'BIT*'
}

# Anytime algorithms that continue improving after finding first path
ANYTIME_ALGOS: Set[str] = {'RRT*', 'BIT*'}

# Algorithm descriptions and paper citations
ALGORITHM_INFO: Dict[str, Tuple[str, str]] = {
    'RRT': (
        "Rapidly-exploring Random Tree implementation after LaValle (1998) for a 2D occupancy-grid setting with configurable goal bias.",
        "LaValle, 1998"
    ),
    'RRT-Connect': (
        "Bidirectional RRT growing two trees from start and goal, connecting when they meet.",
        "Kuffner & LaValle, 2000"
    ),
    'BiTRRT': (
        "OMPL-style bidirectional Transition-based RRT with transition tests, frontier control, and a clearance-derived cost map adaptation for 2D occupancy grids.",
        "Devaurs et al., 2013 / OMPL"
    ),
    'KPIECE': (
        "Single-level geometric KPIECE adaptation using a 2D projection grid, border-cell preference, half-normal motion selection within cells, state sampling along motions, and progress-based score penalties.",
        "Sucan & Kavraki, 2008"
    ),
    'RRT*': (
        "RRT variant with rewiring and incremental path improvement. This UI shows a practical fixed-radius implementation rather than a proof-oriented asymptotic-optimality setup.",
        "Karaman & Frazzoli, 2011"
    ),
    'PRM': (
        "Probabilistic Roadmap with a query-independent learning phase and a query phase that connects start and goal to the learned roadmap before graph search.",
        "Kavraki et al., 1996"
    ),
    'SBL': (
        "Single-query bi-directional lazy roadmap planner using an L-infinity neighborhood metric, deferred segment validation, and the paper's lightweight random path optimizer.",
        "Sanchez & Latombe, 2001"
    ),
    'FMT*': (
        "Fast Marching Tree using uniform free-space sampling, open-set parent selection, and one-shot lazy collision checking per candidate connection in a 2D occupancy-grid adaptation.",
        "Janson et al., 2015"
    ),
    'BIT*': (
        "Batch Informed Trees with ordered vertex and edge queues, rewiring, informed batch sampling, and incumbent-based pruning in a 2D occupancy-grid adaptation.",
        "Gammell et al., 2015"
    ),
    'A*': (
        "Classic heuristic graph search on the induced occupancy grid. Optimal only with respect to that grid discretization.",
        "Hart et al., 1968"
    ),
    'Dijkstra': (
        "Uniform-cost graph search on the induced occupancy grid. Optimal only with respect to that grid discretization.",
        "Dijkstra, 1959"
    ),
    'APF': (
        "Artificial Potential Field. Goal attracts, obstacles repel. Fast but can get stuck.",
        "Khatib, 1986"
    ),
    'CHOMP': (
        "Covariant Hamiltonian Optimization. Gradient-based trajectory optimization.",
        "Zucker et al., 2013"
    ),
    'STOMP': (
        "Approximate / experimental STOMP-style stochastic optimizer using noisy rollout averaging for visualization.",
        "Kalakrishnan et al., 2011"
    ),
    'TrajOpt': (
        "Approximate / experimental TrajOpt-style trust-region optimizer rather than a full sequential convex solver.",
        "Schulman et al., 2014"
    ),
    'ITOMP': (
        "Approximate / experimental ITOMP-style incremental smoother for visual replanning demonstrations.",
        "Park et al., 2012"
    ),
    'GPMP': (
        "Approximate / experimental GPMP-style deterministic local optimizer with GP prior and interpolated obstacle factors.",
        "Mukadam et al., 2016"
    ),
    'PSO': (
        "Experimental waypoint-path optimizer based on Particle Swarm Optimization.",
        "Kennedy & Eberhart, 1995"
    ),
    'Genetic': (
        "Experimental waypoint-path optimizer based on a Genetic Algorithm.",
        "Holland, 1975"
    ),
}

# =============================================================================
# GUI Components
# =============================================================================

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
        
        # Callbacks
        self.on_point_picked: Optional[Callable[[str, Point], None]] = None
        self.is_point_valid: Optional[Callable[[Point], bool]] = None
        
        # Visualization state
        self.highlights: List[Tuple[int, int, int]] = []  # (x, y, alpha)
        self.rejected_highlights: List[Tuple[int, int, int]] = []  # (x, y, alpha)
        self.edge_highlights: List[Tuple[int, int, int, int, int]] = []  # (x1, y1, x2, y2, alpha)
        self.current_tree_edges: List[Edge] = []
        self.current_path: List[Tuple[float, float]] = []

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
    ) -> None:
        """Draw a path with anti-aliasing and rounded joins."""
        if len(path) < 2:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self._make_pen(color, width))
        for i in range(len(path) - 1):
            p1 = QPointF(float(path[i][0]), float(path[i][1]))
            p2 = QPointF(float(path[i + 1][0]), float(path[i + 1][1]))
            painter.drawLine(p1, p2)
    
    def set_image(self, qpix: QPixmap):
        self.base_pixmap = qpix
        self.overlay = QPixmap(qpix.size())
        self.overlay.fill(Qt.GlobalColor.transparent)
        self.start = None
        self.goal = None
        self.pick_mode = "start"
        self.highlights = []
        self.rejected_highlights = []
        self.edge_highlights = []
        self.current_tree_edges = []
        self.current_path = []
        self._cached_disp_size = None  # Recompute on new image
        self._update_display()
    
    def reset_overlay(self):
        if self.base_pixmap is None:
            return
        self.overlay = QPixmap(self.base_pixmap.size())
        self.overlay.fill(Qt.GlobalColor.transparent)
        self.highlights = []
        self.rejected_highlights = []
        self.edge_highlights = []
        self.current_tree_edges = []
        self.current_path = []
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

        # Draw current path on top of the tree for a cleaner readable solution view.
        if len(self.current_path) >= 2:
            self._draw_polyline(painter, self.current_path, Qt.GlobalColor.yellow, 4)
        
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
    
    def mousePressEvent(self, event):
        if self.base_pixmap is None or event.button() != Qt.MouseButton.LeftButton:
            return
        
        x, y = event.position().x(), event.position().y()
        ix = (x - self.offset_x) / self.scale
        iy = (y - self.offset_y) / self.scale
        
        if ix < 0 or iy < 0 or ix >= self.base_pixmap.width() or iy >= self.base_pixmap.height():
            return
        
        p = (int(ix), int(iy))
        
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
    
    def add_rejected_highlight(self, point):
        self.rejected_highlights.append((point[0], point[1], 255))
    
    def fade_highlights(self, fade_amount=25):
        self.highlights = [(x, y, a - fade_amount) for x, y, a in self.highlights if a - fade_amount > 0]
        self.rejected_highlights = [(x, y, a - fade_amount) for x, y, a in self.rejected_highlights if a - fade_amount > 0]
        edge_fade = fade_amount * 3
        self.edge_highlights = [(x1, y1, x2, y2, a - edge_fade) for x1, y1, x2, y2, a in self.edge_highlights if a - edge_fade > 0]
    
    def clear_path(self):
        """Clear the current path (for RRT* live updates)."""
        self.current_path = []
    
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
            # Set as current path for live display
            self.current_path = list(path)
        self._update_display()

# =============================================================================
# Main Window
# =============================================================================

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
        self.setWindowTitle("Path Planning Visualizer Beta (0.1.0b3)")
        
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
        
        controls_row2 = QHBoxLayout()
        controls_row2.addWidget(self.btn_step)
        controls_row2.addWidget(self.btn_run)
        controls_row2.addWidget(self.btn_pause)
        
        # Left panel
        left = QVBoxLayout()
        left.addLayout(controls_row1)
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
    
    def _update_status_display(self, state: str = None, info: str = None):
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        maze_path = os.path.join(script_dir, "assets", "maze.png")
        if os.path.exists(maze_path):
            self._load_image_from_path(maze_path)
    
    def _load_image_from_path(self, path: str):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            QMessageBox.critical(self, "Error", "Could not read image.")
            return
        
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        if np.mean(bw == 255) < 0.5:
            bw = 255 - bw
        self.occ = (bw == 0)
        free_space = (~self.occ).astype(np.uint8)
        self.clearance_field = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.canvas.set_image(QPixmap.fromImage(qimg.copy()))
        
        self._update_status_display(state="Idle", info="Click START then GOAL on a free (white) pixel.")
        self._set_buttons_enabled(False)
        self._reset_solver_metrics()
        self.planner = None
        self.running_algo_name = None  # Clear running algorithm name

    def reset_all(self):
        self.pause()
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
    
    def _ensure_planner(self, force_new: bool = False) -> bool:
        """Ensure planner exists. If force_new=True, create a new one even if one exists."""
        if self.planner is not None and not force_new:
            return True
        if self.occ is None or self.canvas.start is None or self.canvas.goal is None:
            return False
        
        try:
            algo_name = self._get_selected_algo_name()
            planner_class = AVAILABLE_PLANNERS[algo_name]
            params_widget = self.params_widgets[algo_name]
            self.planner = planner_class.create_from_params(
                self.occ, self.canvas.start, self.canvas.goal, params_widget
            )
            self.running_algo_name = algo_name  # Save the running algorithm name
            self._reset_solver_metrics()
            self.btn_pause.setText("Pause")
            self.btn_pause.setEnabled(False)
            self.optimizing_from_sampling = False
            if not self.planner.is_free(self.canvas.start):
                raise ValueError("Start on obstacle.")
            if not self.planner.is_free(self.canvas.goal):
                raise ValueError("Goal on obstacle.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.planner = None
            return False
        return True
    
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
    
    def _restart_planner(self):
        """Reset overlay (keeping start/goal markers) and create a new planner with new seed."""
        # Save start and goal
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
        
        # Create new planner
        self._ensure_planner(force_new=True)
        self.btn_pause.setText("Pause")
        self.btn_pause.setEnabled(False)
        self.optimizing_from_sampling = False
    
    def step_once(self):
        # If planner is done, restart it
        if self.planner is not None and self.planner.done:
            self._restart_planner()
        
        if not self._ensure_planner():
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
            self._restart_planner()
            self.btn_run.setText("Run")  # Reset text temporarily
        # If planner is done, restart it
        elif self.planner is not None and self.planner.done:
            self._restart_planner()
        
        if not self._ensure_planner():
            return
        self.is_playing = True
        self._set_running_state()
        self._update_status_display(state="Running", info="Algorithm is running...")
        self._update_stopwatch_label()
        # MAX mode (1000) uses minimal interval, normal mode uses 1000/speed
        speed = self.speed_slider.value()
        interval_ms = 1 if speed >= 1000 else max(1, 1000 // speed)
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
        base_path = shortcut_path(list(base_path), self.occ)

        # Stop any running timers
        self.timer.stop()
        self.is_playing = False

        chomp_params = self.params_widgets['CHOMP'].get_params()
        # Bias CHOMP toward smoother trajectories when optimizing an existing path
        chomp_params['num_points'] = max(
            chomp_params.get('num_points', 50),
            min(90, max(50, len(base_path) * 2))
        )
        # Keep CHOMP Optimize quick for interactive use.
        chomp_params['max_iters'] = 400
        chomp_params['smoothness_weight'] = chomp_params.get('smoothness_weight', 1.0) * 1.5
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
        interval_ms = 1 if speed >= 1000 else max(1, 1000 // speed)
        self.timer.start(interval_ms)
    
    def _update_speed_label(self, value: int):
        if value >= 1000:
            self.speed_label.setText("MAX")
        else:
            self.speed_label.setText(f"{value} steps/sec")
        if self.is_playing and self.timer.isActive():
            # MAX mode uses minimal interval
            interval_ms = 1 if value >= 1000 else max(1, 1000 // value)
            self.timer.setInterval(interval_ms)
    
    def _fade_tick(self):
        self.canvas.fade_highlights(fade_amount=30)
        self.canvas._update_display()
        if not (self.canvas.highlights or self.canvas.rejected_highlights or self.canvas.edge_highlights):
            self.fade_timer.stop()
    
    def _run_tick(self):
        if self.planner is None:
            return
        
        # MAX mode: faster fading to keep display clean
        speed = self.speed_slider.value()
        if speed >= 1000:
            self.canvas.fade_highlights(fade_amount=120)  # Much faster fade in MAX mode
        else:
            self.canvas.fade_highlights(fade_amount=30 if self.is_playing else 60)
        
        # MAX mode: run many steps per tick for maximum speed, but update display periodically
        if speed >= 1000:
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
            if speed >= 1000 and (i + 1) % display_interval == 0:
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
        is_bitstar = isinstance(self.planner, BITStarPlanner)
        # Handle multiple edges (for batch algorithms like FMT*)
        if result.edges and not is_gpmp and not is_bitstar:
            for edge in result.edges:
                self.canvas.draw_edge(edge[0], edge[1], color=edge_color)
        elif result.edge and not is_gpmp and not is_bitstar:
            self.canvas.draw_edge(result.edge[0], result.edge[1], color=edge_color)
        elif is_bitstar:
            if self.planner is not None and hasattr(self.planner, "extract_tree_edges"):
                self.canvas.set_current_tree_edges(self.planner.extract_tree_edges())
        if result.rejected_point:
            self.canvas.add_rejected_highlight(result.rejected_point)
            self.canvas._update_display()
        if is_gpmp and self.planner is not None:
            display_path = self.planner.extract_display_path()
            if display_path:
                self.canvas.current_path = list(display_path)
        # For RRT*: always keep the current best path visible
        if self.planner is not None and self.planner.found_path:
            path = self.planner.extract_path()
            if path:
                if not self.optimizing_from_sampling and not is_gpmp:
                    display_path = self.planner.extract_display_path() if hasattr(self.planner, "extract_display_path") else path
                    self.canvas.current_path = list(display_path)
    
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

# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for the Path Planning Visualizer application.
    
    Creates and shows the main window, then runs the Qt event loop.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
