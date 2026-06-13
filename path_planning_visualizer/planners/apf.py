from __future__ import annotations

from typing import List, Tuple

import numpy as np

from PyQt6.QtWidgets import (
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
        self.dist_field = make_distance_field(self.occ)
        
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
