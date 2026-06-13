from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)

from ..types import Point, Edge
from ..geometry import (
    clamp_point,
    dist,
    line_collision_free,
    segment_points,
    steer,
)
from .base import BasePlanner, StepResult


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
