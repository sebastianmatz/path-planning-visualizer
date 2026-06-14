from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QSpinBox,
    QWidget,
)

from ..types import Point, Edge
from ..geometry import (
    clamp_point,
    dist,
    line_collision_free,
    make_distance_field,
    segment_points,
    steer,
)
from .base import BasePlanner, StepResult
from ._spatial import GridIndex


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
        # Reject-step temperature increase T <- T * 2^(T_rate) (Alg. 2 line 6).
        self.temp_change_multiplier = float(np.power(2.0, temp_change_factor))
        self.init_temperature = float(max(1e-6, init_temperature))
        self.temp = self.init_temperature
        self.max_iters = int(max_iters)
        self.frontier_node_ratio = float(max(0.01, frontier_node_ratio))
        self.cost_threshold = float(cost_threshold)
        self.rng = np.random.default_rng(seed)

        self.clearance_field = make_distance_field(occ)
        # Clearance-derived adaptation of OMPL's generic state cost / mechanical-work setup.
        self.cost_field = 100.0 / (1.0 + self.clearance_field.astype(np.float64))

        self.max_extent = float(np.hypot(max(1, self.w - 1), max(1, self.h - 1)))
        # refinementControl threshold = the step size delta (Alg. 3 line 1).
        self.frontier_threshold = (
            float(frontier_threshold)
            if frontier_threshold > 0.0
            else self.max_distance
        )
        # attemptLink is only tried when the trees are within 10*delta (Alg. 5 line 1).
        self.connection_range = max(1e-6, 10.0 * self.max_distance)

        start_cost = self._state_cost(start)
        goal_cost = self._state_cost(goal)
        self.tree_start: List[BiTRRTMotion] = [BiTRRTMotion(start, None, start_cost)]
        self.tree_goal: List[BiTRRTMotion] = [BiTRRTMotion(goal, None, goal_cost)]

        # One spatial index per tree (lockstep with tree_start / tree_goal).
        cell = max(1.0, self.max_distance)
        self._index_start = GridIndex(cell)
        self._index_start.add(start[0], start[1])
        self._index_goal = GridIndex(cell)
        self._index_goal.add(goal[0], goal[1])

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

    def _tree_index(self, motions: List[BiTRRTMotion]) -> GridIndex:
        return self._index_start if motions is self.tree_start else self._index_goal

    def _nearest_index(self, motions: List[BiTRRTMotion], point: Point) -> int:
        return self._tree_index(motions).nearest(point[0], point[1])

    def _add_motion(self, point: Point, motions: List[BiTRRTMotion], parent: Optional[int]) -> int:
        state_cost = self._state_cost(point)
        motions.append(BiTRRTMotion(point=point, parent=parent, state_cost=state_cost))
        self._tree_index(motions).add(point[0], point[1])
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
            # Accept uphill: decrease T by 2^((c_j - c_i)/(0.1*costRange)) (Alg. 2 line 4).
            cost_range = self.worst_cost - self.best_cost
            if abs(cost_range) > 1e-4:
                self.temp /= float(np.power(2.0, motion_cost / (0.1 * cost_range)))
            return True

        self.temp *= self.temp_change_multiplier
        return False

    def _min_expansion_control(self, distance_from_nearest: float) -> bool:
        # refinementControl (Alg. 3): an extension shorter than delta is a refinement
        # node; reject it once refinement nodes exceed rho * total nodes.
        if distance_from_nearest > self.frontier_threshold:
            self.frontier_count += 1
            return True

        total = self.frontier_count + self.nonfrontier_count
        if self.nonfrontier_count > self.frontier_node_ratio * total:
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

        # attemptLink (Alg. 5): extend the target tree toward source along flat/downhill
        # slopes only (no transition test), merging if it reaches source.
        connect_edges: List[Edge] = []
        current_nearest_idx = nearest_idx
        while True:
            q_near = target_motions[current_nearest_idx].point
            reach = dist(q_near, source_point) <= self.max_distance
            q_new = clamp_point(
                source_point if reach else steer(q_near, source_point, self.max_distance),
                self.w, self.h,
            )
            if q_new == q_near or not self.is_free(q_new):
                return False, connect_edges, None, q_new
            if not line_collision_free(q_near, q_new, self.occ):
                return False, connect_edges, None, q_new
            # Downhill-only junction (Alg. 5 line 3): reject if the cost increases.
            if self._state_cost(q_new) > self._state_cost(q_near) + 1e-9:
                return False, connect_edges, None, q_new
            new_idx = self._add_motion(q_new, target_motions, current_nearest_idx)
            connect_edges.append((q_near, q_new))
            if reach:
                return True, connect_edges, new_idx, None
            current_nearest_idx = new_idx

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
