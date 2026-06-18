from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..geometry import line_collision_free
from ._trajectory import escape_init, fd_acceleration_matrix, signed_distance_field, straight_line
from .base import BasePlanner, StepResult


class STOMPPlanner(BasePlanner):
    """STOMP - Stochastic Trajectory Optimization for Motion Planning (Kalakrishnan et al. 2011).

    Faithful 2D point-robot specialization of the Table I algorithm:

    1. sample ``K`` noisy rollouts ``theta + eps``, ``eps ~ N(0, R^-1)``;
    2. per-timestep state cost ``S = q_o`` and per-timestep probabilities from the
       paper's exponent ``exp(-h (S - min)/(max - min))`` (Eq. 11);
    3. probability-weighted noise per timestep, projected with ``M`` (R^-1 scaled
       so each column's max is 1/N) to keep updates smooth and endpoints fixed;
    4. obstacle cost ``q_o = max(eps - d, 0) * ||x_dot||`` on a signed distance
       field (Eq. 13); the convergence metric is ``Q = sum q + 1/2 theta^T R theta``.
    """

    name = "STOMP"
    description = "Stochastic Trajectory Optimization for Motion Planning (Kalakrishnan et al. 2011)"

    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 50, max_iters: int = 500, num_rollouts: int = 20,
                 noise_std: float = 10.0, epsilon: float = 10.0, h: float = 10.0, seed: int = 42):
        super().__init__(occ, start, goal)

        self.num_points = num_points
        self.max_iters = max_iters
        self.num_rollouts = num_rollouts
        self.noise_std = noise_std
        self.epsilon = float(epsilon)
        self.cost_h = float(h)  # cost sensitivity (named to avoid shadowing BasePlanner's grid height self.h)

        self.rng = np.random.default_rng(seed)

        # Signed distance field: negative inside obstacles, positive in free space.
        self.sdf = signed_distance_field(self.occ)

        # Straight-line init, bent off obstacles if it collides.
        self.trajectory = escape_init(
            straight_line(self.start, self.goal, self.num_points), self.start, self.goal, self.occ
        )

        # Control matrices: R = A^T A, noise covariance R^-1, and the M projection.
        self._compute_smoothing_matrix()

        # Optimization state.
        self.total_cost = float('inf')
        self.obs_cost = float('inf')
        self.smooth_cost = 0.0
        self.converged = False
        self.best_traj = self.trajectory.copy()
        self.best_cost = float('inf')
        self.best_valid_traj: Optional[np.ndarray] = None
        self.best_valid_cost = float('inf')
        self.stall = 0
        self.patience = 40

        q0, obs0, ctrl0 = self._compute_trajectory_cost(self.trajectory)
        self.total_cost, self.obs_cost, self.smooth_cost = q0, obs0, ctrl0
        self.best_cost = q0
        if self._is_collision_free(self.trajectory):
            self.best_valid_traj = self.trajectory.copy()
            self.best_valid_cost = q0
            self.found_path = True

    def _compute_smoothing_matrix(self):
        """Precompute R = A^T A, the noise covariance R^-1, and the M projection."""
        n = self.num_points - 2  # internal points only
        if n <= 0:
            self.R = np.eye(1)
            self.R_inv = np.eye(1)
            self.M = np.eye(1)
            self.noise_factor = np.eye(1)
            return

        A = fd_acceleration_matrix(n)
        self.R = A.T @ A + 1e-6 * np.eye(n)
        self.R_inv = np.linalg.inv(self.R)

        # M: scale each column of R_inv so its largest element equals 1/N.
        col_max = np.max(np.abs(self.R_inv), axis=0, keepdims=True)
        self.M = self.R_inv / (n * (col_max + 1e-12))

        # Cholesky factor of R_inv: L @ z (z ~ N(0, I)) ~ N(0, R_inv).
        self.noise_factor = np.linalg.cholesky(self.R_inv)

    def _state_cost_along(self, traj: np.ndarray) -> np.ndarray:
        """Per-timestep obstacle cost q_o = max(eps - d, 0) * ||x_dot|| (Eq. 13).

        Returns one value per interior waypoint (signed distance d; ``r_b = 0``).
        """
        n = len(traj)
        idx = np.arange(1, n - 1)
        xs = np.clip(traj[idx, 0], 0, self.w - 1).astype(np.intp)
        ys = np.clip(traj[idx, 1], 0, self.h - 1).astype(np.intp)
        d = self.sdf[ys, xs]
        hinge = np.maximum(self.epsilon - d, 0.0)
        velocity = np.linalg.norm((traj[idx + 1] - traj[idx - 1]) * 0.5, axis=1)
        return hinge * velocity

    def _probabilities(self, S: np.ndarray) -> np.ndarray:
        """Per-timestep rollout probabilities P(theta_k, i) from Eq. 11.

        ``S`` is ``(K, n_internal)``; returns probabilities of the same shape that
        sum to one over the K rollouts at each timestep. Uniform where costs tie.
        """
        s_min = np.min(S, axis=0, keepdims=True)
        s_max = np.max(S, axis=0, keepdims=True)
        denom = np.where((s_max - s_min) > 1e-10, s_max - s_min, 1.0)
        exponent = -self.cost_h * (S - s_min) / denom
        P = np.exp(exponent)
        return P / np.sum(P, axis=0, keepdims=True)

    def _compute_trajectory_cost(self, traj: np.ndarray) -> Tuple[float, float, float]:
        """Q = sum_i q_o(theta_i) + 1/2 theta^T R theta (Eq. 1 / Table I step 6)."""
        obs = float(np.sum(self._state_cost_along(traj)))
        accel = traj[2:] - 2.0 * traj[1:-1] + traj[:-2]
        control = 0.5 * float(np.sum(accel * accel))
        return obs + control, obs, control

    def _is_collision_free(self, traj: np.ndarray) -> bool:
        for i in range(len(traj) - 1):
            p1 = (int(traj[i, 0]), int(traj[i, 1]))
            p2 = (int(traj[i + 1, 0]), int(traj[i + 1, 1]))
            if not line_collision_free(p1, p2, self.occ):
                return False
        return True

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        n = self.num_points
        n_internal = n - 2
        if n_internal <= 0:
            self.done = True
            return StepResult(done=True, found_path=False)

        if self.iteration >= self.max_iters:
            self._finalize()
            self.done = True
            return StepResult(done=True, found_path=self.found_path)

        self.iteration += 1
        K = self.num_rollouts

        # Smooth, endpoint-preserving exploration noise eps ~ N(0, noise_std^2 R^-1),
        # sampled per axis via the cached Cholesky factor (fixed magnitude, per paper).
        z = self.rng.standard_normal((K, n_internal, 2))
        eps = np.empty_like(z)
        for k in range(K):
            eps[k, :, 0] = self.noise_std * (self.noise_factor @ z[k, :, 0])
            eps[k, :, 1] = self.noise_std * (self.noise_factor @ z[k, :, 1])

        # Noisy rollouts (interior perturbed, endpoints fixed) and their per-timestep cost.
        S = np.zeros((K, n_internal))
        for k in range(K):
            noisy = self.trajectory.copy()
            noisy[1:-1] += eps[k]
            noisy[:, 0] = np.clip(noisy[:, 0], 0, self.w - 1)
            noisy[:, 1] = np.clip(noisy[:, 1], 0, self.h - 1)
            S[k] = self._state_cost_along(noisy)

        # Per-timestep probabilities (Eq. 11), probability-weighted noise, M projection.
        P = self._probabilities(S)
        delta_tilde = np.zeros((n_internal, 2))
        for dim in range(2):
            delta_tilde[:, dim] = np.sum(P * eps[:, :, dim], axis=0)
        delta = self.M @ delta_tilde

        self.trajectory[1:-1] += delta
        self.trajectory[:, 0] = np.clip(self.trajectory[:, 0], 0, self.w - 1)
        self.trajectory[:, 1] = np.clip(self.trajectory[:, 1], 0, self.h - 1)

        # Convergence metric Q and best-trajectory bookkeeping.
        self.total_cost, self.obs_cost, self.smooth_cost = self._compute_trajectory_cost(self.trajectory)
        if self.total_cost < self.best_cost:
            self.best_cost = self.total_cost
            self.best_traj = self.trajectory.copy()

        path_improved = False
        if self._is_collision_free(self.trajectory):
            if self.best_valid_traj is None or self.total_cost < self.best_valid_cost - 1e-9:
                self.best_valid_traj = self.trajectory.copy()
                self.best_valid_cost = self.total_cost
                path_improved = not self.found_path
                self.found_path = True
                self.stall = 0
            else:
                self.stall += 1
        else:
            self.stall += 1

        # Converge once a valid trajectory has stopped improving (Q stabilized).
        if self.found_path and self.iteration > 20 and self.stall > self.patience:
            self.converged = True
            self._finalize()
            self.done = True

        # Visualization edge.
        point_idx = self.iteration % (n - 1)
        p1 = (int(self.trajectory[point_idx, 0]), int(self.trajectory[point_idx, 1]))
        p2 = (int(self.trajectory[point_idx + 1, 0]), int(self.trajectory[point_idx + 1, 1]))
        return StepResult(edge=(p1, p2), path_improved=path_improved)

    def _finalize(self):
        """Return the best collision-free rollout result (or the lowest-cost one)."""
        if self.best_valid_traj is not None:
            self.trajectory = self.best_valid_traj.copy()
            self.found_path = True
        else:
            self.trajectory = self.best_traj.copy()
            self.found_path = self._is_collision_free(self.trajectory)

    def extract_path(self) -> List[Tuple[int, int]]:
        return [(int(p[0]), int(p[1])) for p in self.trajectory]

    def extract_display_path(self) -> List[Tuple[float, float]]:
        """Float trajectory for clean single-curve rendering (no accumulated edges)."""
        return [(float(p[0]), float(p[1])) for p in self.trajectory]

    def get_status(self) -> str:
        status = "converged" if self.converged else ("FOUND" if self.found_path else "optimizing")
        return f"STOMP: iter {self.iteration}/{self.max_iters}, obs: {self.obs_cost:.1f}, ctrl: {self.smooth_cost:.1f}, {status}"

