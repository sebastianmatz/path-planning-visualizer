from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..geometry import line_collision_free
from ._trajectory import (
    escape_init,
    make_sdf,
    sdf_query_batch,
    straight_line,
)
from .base import BasePlanner, StepResult


class TrajOptPlanner(BasePlanner):
    """TrajOpt - trajectory optimization by Sequential Convex Optimization.

    Implements the penalty / trust-region SCO loop of Schulman et al. (2014) for
    a 2D point robot:

    * the (convex, quadratic) objective is the sum of squared accelerations;
    * the non-convex collision constraint ``sd(x) >= d_safe`` is convexified by
      linearizing the signed distance, ``sd(x) ~ sd(x0) + grad_sd . dx``, and
      penalized with an l1 hinge ``mu * max(0, d_safe - sd_lin)``;
    * each iteration solves the resulting convex subproblem inside a box trust
      region, then accepts/rejects the step and grows/shrinks the trust region
      from the ratio of true to predicted merit improvement;
    * when the inner loop converges but the trajectory is still infeasible the
      penalty coefficient ``mu`` is increased (the outer penalty loop).
    """

    name = "TrajOpt"
    description = "Sequential convex optimization with l1 penalties and trust regions (Schulman et al. 2014)"

    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 50, max_iters: int = 500, trust_region: float = 20.0,
                 collision_weight: float = 100.0, d_safe: float = 10.0, accept_ratio: float = 0.1):
        super().__init__(occ, start, goal)

        self.num_points = num_points
        self.max_iters = max_iters
        self.trust_region = float(trust_region)
        self.collision_weight = float(collision_weight)
        self.d_safe = float(d_safe)
        self.accept_ratio = float(accept_ratio)  # step acceptance parameter c (Alg. 1)

        # True signed distance field and its gradient (negative inside obstacles).
        self.dist_field, self.grad_x, self.grad_y = make_sdf(self.occ, signed=True)

        # Straight-line init, bent off obstacles if it collides.
        self.trajectory = escape_init(
            straight_line(start, goal, num_points), start, goal, self.occ
        )

        # SCO state.
        self.mu = self.collision_weight          # current penalty coefficient
        self.trust_size = self.trust_region      # current trust-region box size
        self.mu_scale = 10.0
        self.mu_max = 1e7
        self.trust_expand = 2.0
        self.trust_shrink = 0.5
        self.trust_min = 1e-2
        self.trust_max = max(50.0, 2.0 * self.trust_region)

        # Smoothness objective (Eq. 5): sum of squared displacements
        # ||theta_{t+1} - theta_t||^2. A is the first-difference operator over the
        # n_int+1 segments; the fixed endpoints are folded into c. disp = A theta + c,
        # f = ||disp||^2, Hessian Hs = 2 A^T A is constant.
        self.n_int = max(0, num_points - 2)
        if self.n_int > 0:
            m = self.n_int
            A = np.zeros((m + 1, m), dtype=np.float64)
            A[0, 0] = 1.0
            for i in range(1, m):
                A[i, i - 1] = -1.0
                A[i, i] = 1.0
            A[m, m - 1] = -1.0
            self.A = A
            self.Hs = 2.0 * (A.T @ A) + 1e-6 * np.eye(m)
            self.c = np.zeros((m + 1, 2), dtype=np.float64)
            self.c[0] = -np.asarray(start, dtype=np.float64)
            self.c[-1] = np.asarray(goal, dtype=np.float64)
        else:
            self.A = np.zeros((0, 0))
            self.Hs = np.zeros((0, 0))
            self.c = np.zeros((0, 2))

        self.converged = False
        self.total_cost = float('inf')
        self.collision_cost = float('inf')
        self.best_valid_trajectory: Optional[np.ndarray] = None
        self._check_path_validity()
        if self.found_path:
            self.best_valid_trajectory = self.trajectory.copy()

    def _distances_and_normals(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Batched nearest-pixel SDF lookup over the interior waypoints (identical to
        # the per-point sdf_query it replaces).
        return sdf_query_batch(self.dist_field, self.grad_x, self.grad_y,
                               theta[:, 0], theta[:, 1])

    def _smoothness(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        disp = self.A @ theta + self.c  # consecutive displacements (Eq. 5)
        f = float(np.sum(disp * disp))
        grad = 2.0 * (self.A.T @ disp)
        return f, grad

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if self.iteration >= self.max_iters or self.n_int <= 0:
            self._finalize()
            self.done = True
            return StepResult(done=True, found_path=self.found_path)

        self.iteration += 1

        theta = self.trajectory[1:-1].copy()

        # --- current merit -------------------------------------------------
        f_smooth0, grad_smooth = self._smoothness(theta)
        d0, normals = self._distances_and_normals(theta)
        hinge0 = np.maximum(0.0, self.d_safe - d0)
        merit0 = f_smooth0 + self.mu * float(np.sum(hinge0))

        # --- convex subproblem: Newton step on smoothness + linearized hinge
        active = d0 < self.d_safe
        grad_coll = np.zeros((self.n_int, 2), dtype=np.float64)
        grad_coll[active] = -self.mu * normals[active]
        grad_total = grad_smooth + grad_coll
        delta = -np.linalg.solve(self.Hs, grad_total)
        np.clip(delta, -self.trust_size, self.trust_size, out=delta)

        theta_new = theta + delta
        theta_new[:, 0] = np.clip(theta_new[:, 0], 0, self.w - 1)
        theta_new[:, 1] = np.clip(theta_new[:, 1], 0, self.h - 1)
        delta_eff = theta_new - theta

        # --- model (predicted) vs true merit at the proposed step ----------
        f_smooth_new, _ = self._smoothness(theta_new)  # smoothness is exact
        d_lin = d0 + np.sum(normals * delta_eff, axis=1)
        model_merit = f_smooth_new + self.mu * float(np.sum(np.maximum(0.0, self.d_safe - d_lin)))
        predicted = merit0 - model_merit

        d_new, _ = self._distances_and_normals(theta_new)
        hinge_new = np.maximum(0.0, self.d_safe - d_new)
        true_merit = f_smooth_new + self.mu * float(np.sum(hinge_new))
        actual = merit0 - true_merit

        ratio = actual / predicted if predicted > 1e-12 else (1.0 if actual > 1e-9 else -1.0)

        # --- accept / reject + trust-region update (Alg. 1 lines 6-10) ------
        # Accept and expand iff TrueImprove/ModelImprove > c, else reject + shrink.
        accepted = ratio > self.accept_ratio
        if accepted:
            self.trajectory[1:-1] = theta_new
            cur_hinge = hinge_new
            self.trust_size = min(self.trust_max, self.trust_size * self.trust_expand)
        else:
            cur_hinge = hinge0
            self.trust_size = max(self.trust_min, self.trust_size * self.trust_shrink)

        max_viol = float(np.max(cur_hinge)) if self.n_int > 0 else 0.0
        self.collision_cost = float(np.mean(cur_hinge))
        self.total_cost = (true_merit if accepted else merit0)

        self._check_path_validity()
        if self.found_path:
            self.best_valid_trajectory = self.trajectory.copy()

        # --- outer penalty loop / convergence ------------------------------
        inner_converged = self.trust_size <= self.trust_min * 2.0 or abs(predicted) < 1e-4
        if inner_converged:
            if max_viol > 1e-2 and self.mu < self.mu_max:
                self.mu *= self.mu_scale
                self.trust_size = self.trust_region
            elif self.found_path:
                self.converged = True
                self._finalize()
                self.done = True

        # Visualization edge.
        idx = self.iteration % (self.num_points - 1)
        p1 = (int(round(self.trajectory[idx, 0])), int(round(self.trajectory[idx, 1])))
        p2 = (int(round(self.trajectory[idx + 1, 0])), int(round(self.trajectory[idx + 1, 1])))
        return StepResult(edge=(p1, p2), path_improved=(accepted and self.found_path))

    def _finalize(self):
        """Restore the best collision-free trajectory found, if any."""
        if self.best_valid_trajectory is not None:
            self.trajectory = self.best_valid_trajectory.copy()
            self.found_path = True
        else:
            self._check_path_validity()

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
        return (f"TrajOpt: iter {self.iteration}, mu {self.mu:.0f}, trust {self.trust_size:.2f}, "
                f"viol {self.collision_cost:.2f}, {status}")

