from __future__ import annotations

from typing import List, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)

from ..geometry import (
    make_distance_field,
)
from .base import BasePlanner, StepResult


class APFParamsWidget(QWidget):
    """Parameters widget for APF planner."""

    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.spin_step_size = QDoubleSpinBox()
        self.spin_step_size.setRange(0.5, 20.0)
        self.spin_step_size.setSingleStep(0.5)
        self.spin_step_size.setValue(5.0)
        self.spin_step_size.setToolTip("Maximum speed V_max (per-step displacement cap; Khatib Eq. 16-17)")

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(100, 50000)
        self.spin_max_iters.setValue(5000)
        self.spin_max_iters.setToolTip("Maximum iterations")

        self.spin_goal_gain = QDoubleSpinBox()
        self.spin_goal_gain.setRange(0.01, 100.0)
        self.spin_goal_gain.setSingleStep(0.1)
        self.spin_goal_gain.setValue(1.0)
        self.spin_goal_gain.setToolTip("Attractive stiffness k in F_att = -k(x - x_goal) (Khatib Eq. 12)")

        self.spin_obstacle_gain = QDoubleSpinBox()
        self.spin_obstacle_gain.setRange(1.0, 10000.0)
        self.spin_obstacle_gain.setSingleStep(100.0)
        self.spin_obstacle_gain.setValue(1000.0)
        self.spin_obstacle_gain.setToolTip("Repulsive gain eta in the FIRAS force (Khatib Eq. 20)")

        self.spin_obstacle_dist = QSpinBox()
        self.spin_obstacle_dist.setRange(5, 100)
        self.spin_obstacle_dist.setValue(30)
        self.spin_obstacle_dist.setToolTip("Obstacle influence limit rho_0 (FIRAS); no repulsion beyond it")

        self.chk_escape = QCheckBox("Enable local-minimum escape (non-paper)")
        self.chk_escape.setChecked(False)
        self.chk_escape.setToolTip(
            "Off (default): pure APF; the robot stalls at local minima as Khatib documents. "
            "On: add a stochastic kick to try to escape (a heuristic not in the paper)."
        )

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for the optional escape perturbations")

        layout.addRow("Max speed (V_max):", self.spin_step_size)
        layout.addRow("Max iterations:", self.spin_max_iters)
        layout.addRow("Attractive gain (k):", self.spin_goal_gain)
        layout.addRow("Repulsive gain (eta):", self.spin_obstacle_gain)
        layout.addRow("Influence dist (rho_0):", self.spin_obstacle_dist)
        layout.addRow("", self.chk_escape)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'step_size': self.spin_step_size.value(),
            'max_iters': self.spin_max_iters.value(),
            'goal_gain': self.spin_goal_gain.value(),
            'obstacle_gain': self.spin_obstacle_gain.value(),
            'obstacle_dist': self.spin_obstacle_dist.value(),
            'enable_escape': self.chk_escape.isChecked(),
            'seed': self.spin_seed.value(),
        }


class APFPlanner(BasePlanner):
    """APF - Artificial Potential Field planner (Khatib 1986).

    Faithful 2D holonomic point-robot specialization of Khatib's artificial
    potential field:

    - attractive force from a parabolic well, ``F_att = -k (x - x_goal)`` (Eq. 12);
    - FIRAS repulsive force ``eta (1/rho - 1/rho_0) (1/rho^2) dRho`` within the
      influence limit ``rho_0`` (Eq. 20);
    - the resultant force is integrated with Khatib's velocity saturation: the
      step is the force itself, magnitude-capped at ``V_max`` (Eqs. 15-17).

    Pure APF stalls at local minima (Khatib, Sec. 10); when that happens the
    planner stops and reports no path. An optional, non-paper stochastic escape
    can be enabled.
    """

    name = "APF"
    description = "Artificial Potential Field"

    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 step_size: float = 5.0, max_iters: int = 5000,
                 goal_gain: float = 1.0, obstacle_gain: float = 1000.0,
                 obstacle_dist: int = 30, enable_escape: bool = False, seed: int = 42):
        super().__init__(occ, start, goal)

        self.v_max = float(max(1e-3, step_size))
        self.max_iters = max_iters
        self.goal_gain = float(goal_gain)
        self.obstacle_gain = float(obstacle_gain)
        self.obstacle_dist = float(obstacle_dist)
        self.enable_escape = bool(enable_escape)
        self.rng = np.random.default_rng(seed)

        # Current position
        self.pos = np.array([float(start[0]), float(start[1])])
        self.path: List[Tuple[int, int]] = [start]

        # Shortest-distance field rho(x) (distance to the nearest obstacle).
        self.dist_field = make_distance_field(self.occ)

        # Goal-region radius and equilibrium / no-progress detection.
        self.goal_tolerance = 10.0
        self.force_eps = 1e-3       # below this the resultant force is an equilibrium
        self.no_progress_limit = 30
        self.no_progress_count = 0
        self.last_pos = self.pos.copy()

    def _attractive_force(self) -> np.ndarray:
        """Parabolic-well attractive force F_att = -k (x - x_goal) (Khatib Eq. 12)."""
        diff = np.array([self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]])
        return self.goal_gain * diff

    def _distance_gradient(self, x: int, y: int) -> np.ndarray:
        """Central-difference gradient of the distance field (points away from obstacles)."""
        xm, xp = max(x - 1, 0), min(x + 1, self.w - 1)
        ym, yp = max(y - 1, 0), min(y + 1, self.h - 1)
        gx = (self.dist_field[y, xp] - self.dist_field[y, xm]) / max(1, xp - xm)
        gy = (self.dist_field[yp, x] - self.dist_field[ym, x]) / max(1, yp - ym)
        return np.array([gx, gy])

    def _repulsive_force(self) -> np.ndarray:
        """FIRAS repulsive force within the influence limit (Khatib Eqs. 19-20)."""
        x = int(np.clip(self.pos[0], 0, self.w - 1))
        y = int(np.clip(self.pos[1], 0, self.h - 1))
        rho = float(self.dist_field[y, x])

        if rho >= self.obstacle_dist or rho < 1e-6:
            return np.zeros(2)

        # dRho/dx as the unit outward normal (||grad rho|| = 1 for a Euclidean field).
        grad = self._distance_gradient(x, y)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-9:
            return np.zeros(2)
        normal = grad / grad_norm

        magnitude = self.obstacle_gain * (1.0 / rho - 1.0 / self.obstacle_dist) / (rho * rho)
        return magnitude * normal

    def _finish_no_path(self) -> StepResult:
        self.done = True
        self.found_path = False
        return StepResult(done=True, found_path=False)

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        self.iteration += 1

        if self.iteration >= self.max_iters:
            return self._finish_no_path()

        # Reached the goal region?
        dist_to_goal = float(np.hypot(self.pos[0] - self.goal[0], self.pos[1] - self.goal[1]))
        if dist_to_goal < self.goal_tolerance:
            self.path.append(self.goal)
            self.found_path = True
            self.done = True
            return StepResult(done=True, found_path=True)

        # Resultant artificial force (Khatib Eq. 11).
        force = self._attractive_force() + self._repulsive_force()
        force_norm = float(np.linalg.norm(force))

        # Stable equilibrium away from the goal: pure APF stalls here.
        if force_norm < self.force_eps:
            if self.enable_escape:
                force = self.rng.normal(0.0, self.v_max, 2)
                force_norm = float(np.linalg.norm(force))
            else:
                return self._finish_no_path()

        # Velocity saturation: step along the force, capped at V_max (Eqs. 15-17).
        if force_norm > self.v_max:
            velocity = force * (self.v_max / force_norm)
        else:
            velocity = force
        new_pos = self.pos + velocity

        # Clamp to bounds.
        new_pos[0] = np.clip(new_pos[0], 0, self.w - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.h - 1)

        # A step into an obstacle pixel is rejected (discretization); count it as
        # lack of progress so pure APF can give up at a trap.
        new_x, new_y = int(new_pos[0]), int(new_pos[1])
        if self.occ[new_y, new_x] > 0:
            self.no_progress_count += 5
            if not self.enable_escape and self.no_progress_count > self.no_progress_limit:
                return self._finish_no_path()
            return StepResult(rejected_point=(new_x, new_y))

        # Progress bookkeeping (oscillation / channel stall detection).
        move_dist = float(np.linalg.norm(new_pos - self.last_pos))
        if move_dist < 0.5:
            self.no_progress_count += 1
        else:
            self.no_progress_count = max(0, self.no_progress_count - 1)

        if self.no_progress_count > self.no_progress_limit:
            if self.enable_escape:
                new_pos = self.pos + self.rng.normal(0.0, self.v_max, 2)
                new_pos[0] = np.clip(new_pos[0], 0, self.w - 1)
                new_pos[1] = np.clip(new_pos[1], 0, self.h - 1)
                self.no_progress_count = 0
            else:
                return self._finish_no_path()

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
        dist = float(np.hypot(self.pos[0] - self.goal[0], self.pos[1] - self.goal[1]))
        if self.found_path:
            status = "FOUND"
        elif self.done:
            status = "stuck (local minimum)"
        else:
            status = "descending"
        return f"APF: iter {self.iteration}, dist: {dist:.0f}, {status}"

    @staticmethod
    def get_params_widget() -> QWidget:
        return APFParamsWidget()

    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'APFPlanner':
        params = params_widget.get_params()
        return APFPlanner(occ, start, goal, **params)
