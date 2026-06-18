from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..geometry import line_collision_free
from ._trajectory import (
    escape_init,
    fd_acceleration_matrix,
    make_sdf,
    sdf_query_batch,
    straight_line,
)
from .base import BasePlanner, StepResult


class ITOMPPlanner(BasePlanner):
    """ITOMP - Incremental Trajectory Optimization for Motion Planning.

    ITOMP (Park et al. 2012) optimizes a fixed-horizon timed trajectory while it
    is being executed, re-optimizing the not-yet-executed remainder at each
    control step.  On this static 2D grid (no dynamic obstacles) the faithful
    reduction is a CHOMP-style covariant-gradient optimizer applied to the
    *active suffix* ``[exec_idx, n-1)``, advanced by a receding execution
    horizon.  The smoothness metric ``A^T A`` (sum of squared accelerations)
    acts as the Riemannian metric, so the obstacle gradient is preconditioned by
    its inverse before the update -- the covariant step that keeps the
    re-optimized trajectory smooth.
    """

    name = "ITOMP"
    description = "Incremental covariant trajectory optimization over a receding execution horizon (Park et al. 2012)"

    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 30, max_iters: int = 1000, replan_interval: int = 20,
                 learning_rate: float = 0.5, seed: int = 42):
        super().__init__(occ, start, goal)

        self.num_points = num_points
        self.max_iters = max_iters
        self.replan_interval = max(1, replan_interval)
        self.learning_rate = learning_rate
        self.epsilon = 14.0
        self.obstacle_weight = 50.0
        self.smooth_weight = 0.1

        # True signed distance field and gradient (negative inside obstacles; Eq. 8).
        self.dist_field, self.grad_x, self.grad_y = make_sdf(self.occ, signed=True)

        # Straight-line init, bent off obstacles if it collides.
        self.trajectory = escape_init(
            straight_line(start, goal, num_points), start, goal, self.occ
        )

        # Smoothness metric over interior points: accel = A theta + c.
        self.n_int = max(0, num_points - 2)
        if self.n_int > 0:
            self.A = fd_acceleration_matrix(self.n_int)
            self.K = self.A.T @ self.A + 1e-6 * np.eye(self.n_int)
            self.c = np.zeros((self.n_int, 2), dtype=np.float64)
            self.c[0] = np.asarray(start, dtype=np.float64)
            self.c[-1] = np.asarray(goal, dtype=np.float64)
        else:
            self.A = np.zeros((0, 0))
            self.K = np.zeros((0, 0))
            self.c = np.zeros((0, 2))

        # Current execution index (simulates robot progress).
        self.exec_idx = 0
        self.converged = False
        self.total_cost = float('inf')
        self._check_path_validity()

    def _static_obstacle_cost(self) -> float:
        """ITOMP static obstacle cost (Eq. 8): sum of max(eps - d, 0) * ||x_dot||."""
        if self.n_int <= 0:
            return 0.0
        full = self.trajectory
        interior = full[1:-1]
        d, _ = sdf_query_batch(self.dist_field, self.grad_x, self.grad_y,
                               interior[:, 0], interior[:, 1])
        hinge = np.maximum(self.epsilon - d, 0.0)
        vel = (full[2:] - full[:-2]) * 0.5  # central-difference velocity at each interior point
        speed = np.linalg.norm(vel, axis=1)
        return float(np.sum(hinge * speed))

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        self.iteration += 1

        if self.iteration >= self.max_iters or self.n_int <= 0:
            self._check_path_validity()
            self.done = True
            return StepResult(done=True, found_path=self.found_path)

        theta = self.trajectory[1:-1]

        # Active suffix: interior points the robot has not yet executed.
        active = [j for j in range(self.n_int) if (j + 1) > self.exec_idx]
        if active:
            idx = np.array(active)

            # Obstacle cost gradient: ITOMP Eq. 8 hinge max(eps - d, 0) * ||x_dot||
            # on the signed distance d. The gradient of the (linear) hinge where it
            # is active (d < eps) is -||x_dot|| * grad(d), pushing away from obstacles.
            full = self.trajectory
            g = np.zeros((self.n_int, 2), dtype=np.float64)
            pts = theta[idx]
            d_arr, normals = sdf_query_batch(self.dist_field, self.grad_x, self.grad_y,
                                             pts[:, 0], pts[:, 1])
            vel = (full[2:] - full[:-2]) * 0.5  # workspace velocity at each interior point
            vmag = np.linalg.norm(vel[idx], axis=1)
            active_hinge = (d_arr < self.epsilon).astype(np.float64)  # 0 where d >= eps
            g[idx] = -self.obstacle_weight * (vmag * active_hinge)[:, None] * normals

            # Smoothness gradient (acceleration energy).
            accel = self.A @ theta + self.c
            g += self.smooth_weight * (2.0 * (self.A.T @ accel))

            # Covariant step: precondition by the smoothness metric restricted
            # to the active block, then descend.
            k_sub = self.K[np.ix_(idx, idx)]
            g_cov = np.linalg.solve(k_sub, g[idx])
            theta[idx] -= self.learning_rate * g_cov
            theta[idx, 0] = np.clip(theta[idx, 0], 0, self.w - 1)
            theta[idx, 1] = np.clip(theta[idx, 1], 0, self.h - 1)

        # Advance the execution horizon every replan_interval iterations.
        if self.iteration % self.replan_interval == 0:
            self.exec_idx = min(self.exec_idx + 1, self.num_points - 2)

        self._check_path_validity()
        if self.found_path and self.exec_idx >= self.num_points - 2:
            self.converged = True
            self.done = True

        idx_viz = self.iteration % (self.num_points - 1)
        p1 = (int(self.trajectory[idx_viz, 0]), int(self.trajectory[idx_viz, 1]))
        p2 = (int(self.trajectory[idx_viz + 1, 0]), int(self.trajectory[idx_viz + 1, 1]))
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

    def extract_display_path(self) -> List[Tuple[float, float]]:
        """Float trajectory for clean single-curve rendering (no accumulated edges)."""
        return [(float(p[0]), float(p[1])) for p in self.trajectory]

    def get_status(self) -> str:
        status = "FOUND" if self.found_path else "optimizing"
        progress = self.exec_idx / (self.num_points - 1) * 100
        return f"ITOMP: iter {self.iteration}, exec: {progress:.0f}%, {status}"

