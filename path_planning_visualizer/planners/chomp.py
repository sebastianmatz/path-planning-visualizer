from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..geometry import (
    _resample_to_targets,
    bilinear_sample_scalar,
    bilinear_sample_scalar_batch,
    bilinear_sample_vector,
    line_collision_free,
    make_distance_field,
)
from ..types import Point
from .base import BasePlanner, StepResult


class CHOMPPlanner(BasePlanner):
    """CHOMP - Covariant Hamiltonian Optimization for Motion Planning.
    
    This implementation specializes CHOMP to a 2D point robot on an
    occupancy-grid map using:
    - A signed distance field in workspace
    - A covariant preconditioned update with a velocity-based smoothness metric
    - The point-robot form of CHOMP's obstacle functional with tangent projection
      and curvature terms
    """
    
    name = "CHOMP"
    description = "Covariant trajectory optimization for a 2D point robot"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 50, max_iters: int = 500, learning_rate: float = 0.3,
                 smoothness_weight: float = 1.0, obstacle_weight: float = 100.0,
                 obstacle_epsilon: int = 20, path_length_weight: float = 0.0,
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
        self._build_smoothness_system()
        
        # Initialize trajectory - use provided trajectory if available
        if init_trajectory is not None and len(init_trajectory) >= 2:
            self.trajectory = self._initialize_from_path(init_trajectory)
        else:
            self.trajectory = self._initialize_trajectory()
        
        # Track optimization state
        self.total_cost = float('inf')
        self.obs_cost = float('inf')
        self.converged = False
        self.iters_since_improvement = 0
        self.stagnation_limit = 40  # stop if the best cost stalls (e.g. an obstacle-term limit cycle)
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
        dist_from_obstacle = make_distance_field(self.occ)

        obstacle_space = (self.occ > 0).astype(np.uint8)
        dist_inside = cv2.distanceTransform(obstacle_space, cv2.DIST_L2, 5)
        
        sdf = dist_from_obstacle - dist_inside
        return sdf
    
    def _compute_gradient_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient of distance field."""
        grad_x = cv2.Sobel(self.dist_field, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        grad_y = cv2.Sobel(self.dist_field, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        return grad_x, grad_y

    def _build_smoothness_system(self) -> None:
        """Build the discrete CHOMP metric for the free waypoints.

        We use the squared-velocity prior from the original formulation, which
        yields a tridiagonal quadratic form over the free waypoints. Its inverse
        acts as the CHOMP preconditioner.
        """
        self.num_free = max(0, self.num_points - 2)
        if self.num_free == 0:
            self.velocity_diff_matrix = np.zeros((0, 0), dtype=np.float64)
            self.smoothness_matrix = np.zeros((0, 0), dtype=np.float64)
            self.metric_matrix = np.zeros((0, 0), dtype=np.float64)
            self.boundary_term = np.zeros((0, 2), dtype=np.float64)
            return

        m = self.num_free
        diff = np.zeros((m + 1, m), dtype=np.float64)
        diff[0, 0] = 1.0
        for i in range(1, m):
            diff[i, i - 1] = -1.0
            diff[i, i] = 1.0
        diff[m, m - 1] = -1.0

        boundary = np.zeros((m + 1, 2), dtype=np.float64)
        boundary[0] = -np.array(self.start, dtype=np.float64)
        boundary[m] = np.array(self.goal, dtype=np.float64)

        self.velocity_diff_matrix = diff
        self.smoothness_matrix = diff.T @ diff
        self.metric_matrix = self.smoothness_matrix + 1e-6 * np.eye(m, dtype=np.float64)
        self.boundary_term = diff.T @ boundary

    def _solve_metric(self, rhs: np.ndarray) -> np.ndarray:
        """Apply the CHOMP metric inverse to a waypoint-space vector."""
        if self.num_free == 0:
            return np.zeros_like(rhs)
        return np.linalg.solve(self.metric_matrix, rhs)
    
    def _initialize_trajectory(self) -> np.ndarray:
        """Initialize trajectory - with random perturbation to help escape obstacles."""
        trajectory = np.zeros((self.num_points, 2), dtype=np.float64)
        
        # Start with straight line
        for i in range(self.num_points):
            t = i / (self.num_points - 1)
            trajectory[i, 0] = self.start[0] + t * (self.goal[0] - self.start[0])
            trajectory[i, 1] = self.start[1] + t * (self.goal[1] - self.start[1])
        
        # Check if the straight-line initialization is already collision free.
        has_collision = not self._trajectory_collision_free(trajectory)
        
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
                
                # Evaluate the CHOMP obstacle functional and keep the lower-cost seed.
                _, obs_cost, _, _ = self._evaluate_trajectory(test_traj)
                
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
        resampled = _resample_to_targets(pts, deltas, seg_lens, cum, targets)

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

    def _trajectory_collision_free(self, trajectory: np.ndarray) -> bool:
        """Check a floating-point trajectory against the occupancy grid."""
        if len(trajectory) < 2:
            return True
        for i in range(len(trajectory) - 1):
            p1 = self._trajectory_point_to_pixel(trajectory[i])
            p2 = self._trajectory_point_to_pixel(trajectory[i + 1])
            if not line_collision_free(p1, p2, self.occ):
                return False
        return True

    def _trajectory_point_to_pixel(self, point: np.ndarray) -> Point:
        """Convert a floating-point waypoint to the nearest valid pixel.

        Uses plain Python round/clamp rather than ``np.rint``/``np.clip`` on scalars:
        this is called per waypoint in every per-iteration collision check, and numpy
        scalar ops carry large per-call overhead (they dominated the CHOMP profile).
        """
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x = 0 if x < 0 else (self.w - 1 if x >= self.w else x)
        y = 0 if y < 0 else (self.h - 1 if y >= self.h else y)
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

    def _sample_workspace_distance_and_gradient(self, x: float, y: float) -> Tuple[float, np.ndarray]:
        """Sample the signed distance field and its gradient at a continuous point."""
        distance = bilinear_sample_scalar(self.dist_field, x, y)
        gradient = bilinear_sample_vector(self.grad_x, self.grad_y, x, y)
        return distance, gradient

    def _workspace_cost_and_gradient(self, x: float, y: float) -> Tuple[float, np.ndarray]:
        """Return CHOMP's workspace potential c(x) and its gradient."""
        distance, distance_grad = self._sample_workspace_distance_and_gradient(x, y)
        epsilon = float(self.obstacle_epsilon)

        if distance < 0.0:
            cost = -distance + 0.5 * epsilon
            cost_grad = -distance_grad
        elif distance <= epsilon:
            cost = 0.5 * (distance - epsilon) ** 2 / epsilon
            cost_grad = ((distance - epsilon) / epsilon) * distance_grad
        else:
            cost = 0.0
            cost_grad = np.zeros(2, dtype=np.float64)

        return float(cost), cost_grad

    def _compute_smoothness_cost_and_grad(self, trajectory: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return the discrete squared-velocity prior and its gradient."""
        if self.num_free == 0:
            return 0.0, np.zeros((0, 2), dtype=np.float64)

        segments = np.diff(trajectory, axis=0)
        free_points = trajectory[1:-1]
        smooth_cost = 0.5 * float(np.sum(segments ** 2))
        smooth_grad = self.smoothness_matrix @ free_points + self.boundary_term
        return smooth_cost, smooth_grad

    def _compute_obstacle_terms(
        self,
        trajectory: np.ndarray,
        compute_gradients: bool,
    ) -> Tuple[float, np.ndarray, float, np.ndarray]:
        """Return CHOMP obstacle/length functionals and their gradients.

        Vectorized over the free waypoints: this is the per-waypoint Ratliff (2009)
        obstacle functional ``Sigma c(q)*||q_dot||`` with gradient
        ``||q_dot|| * (P grad(c) - c * kappa)`` (``P = I - t t^T`` the velocity
        projector, ``kappa`` the curvature) -- identical math to the original
        per-waypoint loop, evaluated as array operations.
        """
        n = len(trajectory)
        m = max(0, n - 2)
        empty = np.zeros((m, 2), dtype=np.float64)
        if m == 0:
            return 0.0, empty, 0.0, empty.copy()

        q_prev = trajectory[:-2]
        q_curr = trajectory[1:-1]
        q_next = trajectory[2:]

        velocity = 0.5 * (q_next - q_prev)
        speed_safe = np.maximum(np.linalg.norm(velocity, axis=1), 1e-6)   # (m,)
        tangent = velocity / speed_safe[:, None]                          # (m, 2)

        # Workspace cost c(x) and its gradient at each waypoint (three-case form,
        # Ratliff et al. 2009 Sec. II-D) over the signed distance field.
        x, y = q_curr[:, 0], q_curr[:, 1]
        distance = bilinear_sample_scalar_batch(self.dist_field, x, y)    # (m,)
        dist_grad = np.stack(
            (bilinear_sample_scalar_batch(self.grad_x, x, y),
             bilinear_sample_scalar_batch(self.grad_y, x, y)),
            axis=1,
        )                                                                # (m, 2)

        epsilon = float(self.obstacle_epsilon)
        cost = np.zeros(m, dtype=np.float64)
        cost_grad = np.zeros((m, 2), dtype=np.float64)
        inside = distance < 0.0
        near = (~inside) & (distance <= epsilon)
        cost[inside] = -distance[inside] + 0.5 * epsilon
        cost_grad[inside] = -dist_grad[inside]
        d_near = distance[near]
        cost[near] = 0.5 * (d_near - epsilon) ** 2 / epsilon
        cost_grad[near] = ((d_near - epsilon) / epsilon)[:, None] * dist_grad[near]

        obs_cost = float(np.sum(cost * speed_safe))
        length_cost = float(np.sum(speed_safe))

        if not compute_gradients:
            return obs_cost, empty, length_cost, empty.copy()

        acceleration = q_next - 2.0 * q_curr + q_prev
        # projector @ v = v - t (t . v), applied to acceleration and to grad(c).
        proj_accel = acceleration - tangent * np.sum(tangent * acceleration, axis=1)[:, None]
        curvature = proj_accel / (speed_safe ** 2)[:, None]
        proj_grad = cost_grad - tangent * np.sum(tangent * cost_grad, axis=1)[:, None]

        obs_grad = speed_safe[:, None] * (proj_grad - cost[:, None] * curvature)
        length_grad = -speed_safe[:, None] * curvature

        return float(obs_cost), obs_grad, float(length_cost), length_grad

    def _evaluate_trajectory(self, trajectory: np.ndarray) -> Tuple[float, float, float, float]:
        """Return total, obstacle, smoothness, and path-length costs for a trajectory."""
        smooth_cost, _ = self._compute_smoothness_cost_and_grad(trajectory)
        obs_cost, _, path_length_cost, _ = self._compute_obstacle_terms(
            trajectory,
            compute_gradients=False,
        )
        total_cost = (
            self.smoothness_weight * smooth_cost
            + self.obstacle_weight * obs_cost
            + self.path_length_weight * path_length_cost
        )
        return total_cost, obs_cost, smooth_cost, path_length_cost

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
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        if self.iteration >= self.max_iters:
            self.done = True
            self._finalize_best_trajectory()
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        n = self.num_points
        previous_total_cost = self.total_cost

        smooth_cost, smooth_grad = self._compute_smoothness_cost_and_grad(self.trajectory)
        obs_cost, obs_grad, path_length_cost, length_grad = self._compute_obstacle_terms(
            self.trajectory,
            compute_gradients=True,
        )
        total_cost = (
            self.smoothness_weight * smooth_cost
            + self.obstacle_weight * obs_cost
            + self.path_length_weight * path_length_cost
        )

        # CHOMP covariant gradient step (Ratliff et al. 2009, Sec. II-A):
        #   xi <- xi - (1/lambda) A^-1 g,   g = grad(f_prior + f_obs).
        # learning_rate plays the role of 1/lambda; a single total-step-norm cap is
        # the only stability guard (no line search).
        g = (
            self.smoothness_weight * smooth_grad
            + self.obstacle_weight * obs_grad
            + self.path_length_weight * length_grad
        )
        accepted_update = np.zeros_like(g)
        accepted_total_cost = total_cost
        accepted_obs_cost = obs_cost
        accepted_smooth_cost = smooth_cost

        if self.num_free > 0:
            delta = -self.learning_rate * self._solve_metric(g)
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > 25.0:
                delta *= 25.0 / delta_norm

            self.trajectory[1:-1] += delta
            self.trajectory[1:-1, 0] = np.clip(self.trajectory[1:-1, 0], 0, self.w - 1)
            self.trajectory[1:-1, 1] = np.clip(self.trajectory[1:-1, 1], 0, self.h - 1)

            accepted_update = delta
            accepted_total_cost, accepted_obs_cost, accepted_smooth_cost, _ = self._evaluate_trajectory(self.trajectory)

        self.path_length = self._trajectory_length(self.trajectory)

        if accepted_total_cost < self.best_cost - 1e-6:
            self.best_cost = accepted_total_cost
            self.best_trajectory = self.trajectory.copy()
            self.iters_since_improvement = 0
        else:
            self.iters_since_improvement += 1

        path_improved = False
        if self.iteration % 5 == 0:
            self._check_path_validity()
            path_improved = self._store_best_valid_trajectory(accepted_total_cost)

        cost_change = abs(previous_total_cost - accepted_total_cost) if previous_total_cost != float('inf') else float('inf')
        update_norm = float(np.linalg.norm(accepted_update))
        self.total_cost = accepted_total_cost
        self.obs_cost = accepted_obs_cost
        self.smooth_cost = accepted_smooth_cost

        step_settled = cost_change < 8e-4 or update_norm < 8e-3
        stalled = self.iters_since_improvement >= self.stagnation_limit  # caught in a limit cycle
        if (
            self.iteration > 40
            and self.best_valid_trajectory is not None
            and (step_settled or stalled)
        ):
            self.converged = True
            self.done = True
            self._finalize_best_trajectory()
        
        # Create edge for visualization (point_idx is always in [0, n-2]).
        point_idx = self.iteration % (n - 1)
        p1 = self._rounded_trajectory_point(point_idx)
        p2 = self._rounded_trajectory_point(point_idx + 1)
        edge = (p1, p2)
        
        return StepResult(edge=edge, path_improved=path_improved)
    
    def _check_path_validity(self):
        """Check if the final trajectory is collision-free."""
        self.found_path = self._trajectory_collision_free(self.trajectory)
    
    def extract_path(self) -> List[Tuple[int, int]]:
        """Extract the current trajectory as a path."""
        return [self._rounded_trajectory_point(i) for i in range(len(self.trajectory))]

    def extract_display_path(self) -> List[Tuple[float, float]]:
        """The optimized trajectory itself — the genuine CHOMP output, unsmoothed.

        CHOMP already minimizes a smoothness objective, so the trajectory is smooth by
        construction; no extra display smoothing is applied (it would alter the shown
        result away from what the optimizer actually produced).
        """
        return [(float(p[0]), float(p[1])) for p in self.trajectory]
    
    def get_status(self) -> str:
        status = "converged" if self.converged else ("FOUND" if self.found_path else "optimizing")
        smooth_str = f"smooth: {getattr(self, 'smooth_cost', 0):.1f}"
        obs_str = f"obs: {self.obs_cost:.1f}"
        return f"CHOMP: iter {self.iteration}/{self.max_iters}, {smooth_str}, {obs_str}, {status}"
    
    