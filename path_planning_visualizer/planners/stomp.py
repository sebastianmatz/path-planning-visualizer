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
    line_collision_free,
    make_distance_field,
)
from ._trajectory import straight_line, escape_init, fd_acceleration_matrix
from .base import BasePlanner, StepResult


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
    description = "Stochastic Trajectory Optimization for Motion Planning (Kalakrishnan et al. 2011)"
    
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
        return make_distance_field(self.occ)
    
    def _compute_smoothing_matrix(self):
        """Pre-compute the STOMP control matrices (Kalakrishnan et al. 2011).

        ``R = AᵀA`` is the finite-difference acceleration (control) cost metric.
        ``R⁻¹`` is the noise covariance, so exploration noise is smooth and keeps
        the endpoints fixed.  ``M`` is ``R⁻¹`` with each column scaled so its
        largest element is ``1/N``; multiplying the per-timestep update by ``M``
        keeps the trajectory smooth while leaving the endpoints unchanged.
        """
        n = self.num_points - 2  # Internal points only
        if n <= 0:
            self.R = np.eye(1)
            self.R_inv = np.eye(1)
            self.M = np.eye(1)
            self.noise_factor = np.eye(1)
            return

        A = fd_acceleration_matrix(n)
        self.R = A.T @ A + 1e-6 * np.eye(n)
        self.R_inv = np.linalg.inv(self.R)

        # M: scale each column of R_inv so its maximum element equals 1/N.
        col_max = np.max(np.abs(self.R_inv), axis=0, keepdims=True)
        self.M = self.R_inv / (n * (col_max + 1e-12))

        # Cholesky factor of the (SPD) noise covariance R_inv, so that
        # L @ z (z ~ N(0, I)) is distributed as N(0, R_inv).
        self.noise_factor = np.linalg.cholesky(self.R_inv)
    
    def _initialize_trajectory(self) -> np.ndarray:
        """Straight-line init, bent off obstacles if it collides."""
        trajectory = straight_line(self.start, self.goal, self.num_points)
        return escape_init(trajectory, self.start, self.goal, self.occ)
    
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
        
        K = self.num_rollouts

        # Annealed exploration magnitude (keeps the STOMP structure intact).
        scale = self.noise_std * (1.0 - 0.5 * self.iteration / self.max_iters)

        # Smooth, endpoint-preserving exploration noise: eps ~ N(0, scale^2 R^-1)
        # sampled per axis via the cached Cholesky factor of R^-1.
        z = self.rng.standard_normal((K, n_internal, 2))
        eps = np.empty_like(z)
        for k in range(K):
            eps[k, :, 0] = scale * (self.noise_factor @ z[k, :, 0])
            eps[k, :, 1] = scale * (self.noise_factor @ z[k, :, 1])

        # Per-timestep state cost S[k, i] of each noisy rollout.
        S = np.zeros((K, n_internal))
        for k in range(K):
            for i in range(n_internal):
                x = self.trajectory[i + 1, 0] + eps[k, i, 0]
                y = self.trajectory[i + 1, 1] + eps[k, i, 1]
                S[k, i] = self._compute_point_cost(x, y)

        # Per-timestep probabilities: softmax over rollouts at each waypoint.
        S_min = np.min(S, axis=0, keepdims=True)
        exp_S = np.exp(-(S - S_min) / self.temperature)
        P = exp_S / (np.sum(exp_S, axis=0, keepdims=True) + 1e-10)  # (K, n_internal)

        # Probability-weighted noise per timestep, then project with M so the
        # update stays smooth and leaves the endpoints fixed.
        delta_tilde = np.zeros((n_internal, 2))
        for dim in range(2):
            delta_tilde[:, dim] = np.sum(P * eps[:, :, dim], axis=0)
        delta = self.M @ delta_tilde

        # Update interior waypoints and clamp to bounds.
        self.trajectory[1:-1] += delta
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
        
        # Create edge for visualization (point_idx is always in [0, n-2]).
        point_idx = self.iteration % (n - 1)
        p1 = (int(self.trajectory[point_idx, 0]), int(self.trajectory[point_idx, 1]))
        p2 = (int(self.trajectory[point_idx + 1, 0]), int(self.trajectory[point_idx + 1, 1]))
        edge = (p1, p2)
        
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
