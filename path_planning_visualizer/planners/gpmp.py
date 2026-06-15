from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..geometry import line_collision_free
from ._trajectory import escape_init, make_sdf, sdf_query_batch, straight_line
from .base import BasePlanner, StepResult


def _phi(dt: float) -> np.ndarray:
    """Constant-velocity state-transition matrix for a [x, y, vx, vy] state."""
    return np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _q(dt: float, qc: float) -> np.ndarray:
    """Process-noise covariance of the constant-velocity LTI GP over ``dt``."""
    a = (dt ** 3) / 3.0 * qc
    b = (dt ** 2) / 2.0 * qc
    c = dt * qc
    return np.array([
        [a, 0.0, b, 0.0],
        [0.0, a, 0.0, b],
        [b, 0.0, c, 0.0],
        [0.0, b, 0.0, c],
    ], dtype=np.float64)


class GPMPPlanner(BasePlanner):
    """GPMP - Gaussian Process Motion Planning (Mukadam, Yan & Boots 2016).

    The trajectory is a set of GP support states ``[x, y, vx, vy]`` drawn from a
    constant-velocity LTI Gaussian-process prior:

    * **GP prior** connects consecutive states through the LTI model
      ``e_i = theta_{i+1} - Phi theta_i`` weighted by the process-noise
      information ``Q^{-1}`` (a block-tridiagonal prior precision);
    * **obstacle cost** penalizes states *and* GP-interpolated intermediate
      states whose signed distance falls below a safety margin, using the SDF
      gradient;
    * optimization is the paper's **covariant gradient update**
      ``xi <- xi - (1/eta) K grad U`` (Eq. 24): the cost-functional gradient
      (GP-prior + obstacle terms) preconditioned by the GP covariance
      ``K = (B^T Q^-1 B)^-1``, with a per-step position-step cap.
    """

    name = "GPMP"
    description = "Gaussian Process Motion Planning: an LTI GP prior with GP interpolation, optimized by the covariant gradient update (Mukadam, Yan & Boots 2016)"

    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 25, max_iters: int = 200, sigma: float = 6.0,
                 obstacle_weight: float = 1.0, dt: float = 1.0, eta: float = 5.0):
        super().__init__(occ, start, goal)

        self.num_points = int(max(3, num_points))
        self.max_iters = max_iters
        self.qc = float(sigma) ** 2          # GP process-noise spectral density
        self.obstacle_weight = float(obstacle_weight)
        self.dt = float(dt)
        self.eta = float(max(1e-6, eta))     # regularization; 1/eta is the covariant step size
        self.epsilon = 14.0
        self.max_step = 20.0                  # per-iteration position-step cap (stability guard)
        self.interp_taus = [0.25, 0.5, 0.75]

        # True signed distance field and gradient (negative inside obstacles).
        self.dist_field, self.grad_x, self.grad_y = make_sdf(self.occ, signed=True)

        # GP matrices (uniform spacing -> same for every segment).
        self.Phi = _phi(self.dt)
        self.Q = _q(self.dt, self.qc)
        self.Q_inv = np.linalg.inv(self.Q)
        self._build_interpolators()

        # GP-prior precision K^-1 = B^T Q^-1 B over the free states (constant). It
        # is the Riemannian metric for the covariant gradient update (Eqs. 23-24).
        self.Kinv_prior = self._prior_precision()

        # Support states: positions from a (possibly bent) straight line, with a
        # constant-velocity initialization; endpoints have zero velocity.
        self.states = self._init_states()
        self.start_state = self.states[0].copy()
        self.goal_state = self.states[-1].copy()

        self.converged = False
        self.prior_cost = float('inf')
        self.obs_cost = float('inf')
        self.total_cost = self._total_cost(self.states)
        self.best_valid_trajectory: Optional[np.ndarray] = None
        self._check_path_validity()
        if self.found_path:
            self.best_valid_trajectory = self.states[:, :2].copy()

    # ------------------------------------------------------------------ setup
    def _build_interpolators(self) -> None:
        """Precompute GP interpolation matrices Lambda(tau), Psi(tau)."""
        self.lambdas: List[np.ndarray] = []
        self.psis: List[np.ndarray] = []
        for tau in self.interp_taus:
            ta = tau * self.dt
            phi_ta = _phi(ta)
            phi_rest = _phi(self.dt - ta)
            q_ta = _q(ta, self.qc)
            psi = q_ta @ phi_rest.T @ self.Q_inv
            lam = phi_ta - psi @ self.Phi
            self.lambdas.append(lam)
            self.psis.append(psi)

    def _init_states(self) -> np.ndarray:
        n = self.num_points
        pos = straight_line(self.start, self.goal, n)
        pos = escape_init(pos, self.start, self.goal, self.occ)
        states = np.zeros((n, 4), dtype=np.float64)
        states[:, :2] = pos
        # Constant-velocity guess from consecutive positions; endpoints rest.
        vel = np.zeros((n, 2), dtype=np.float64)
        vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * self.dt)
        states[:, 2:] = vel
        return states

    # ------------------------------------------------------------------- cost
    def _residuals_batch(self, pos_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Whitened obstacle residuals and d(residual)/d(position) for many points.

        Vectorized form of the per-point residual: ``r = w*(eps - d)`` and
        ``dr/dpos = -w*normal`` where ``d < eps`` (signed distance), else ``0``.
        """
        d, normal = sdf_query_batch(self.dist_field, self.grad_x, self.grad_y,
                                    pos_xy[:, 0], pos_xy[:, 1])
        w = np.sqrt(self.obstacle_weight)
        active = d < self.epsilon
        r = np.where(active, w * (self.epsilon - d), 0.0)
        drdp = np.where(active[:, None], -w * normal, 0.0)
        return r, drdp

    def _obstacle_residuals(self, states: np.ndarray):
        """Residuals (r, dr/dpos) at the support states and the GP-interpolated states.

        Returns ``(r_sup, drdp_sup, r_int, drdp_int)`` indexed so that ``r_sup[i-1]``
        is the residual at support state ``i`` and ``r_int[i, k]`` at the ``k``-th
        interpolated state on segment ``i``. The SDF is queried in two batched calls.
        """
        n = self.num_points
        nk = len(self.interp_taus)
        r_sup, drdp_sup = self._residuals_batch(states[1:-1, :2])

        interp_pos = np.empty((n - 1, nk, 2), dtype=np.float64)
        for i in range(n - 1):
            for k in range(nk):
                inter = self.lambdas[k] @ states[i] + self.psis[k] @ states[i + 1]
                interp_pos[i, k] = inter[:2]
        r_int, drdp_int = self._residuals_batch(interp_pos.reshape(-1, 2))
        return r_sup, drdp_sup, r_int.reshape(n - 1, nk), drdp_int.reshape(n - 1, nk, 2)

    def _total_cost(self, states: np.ndarray) -> float:
        prior = 0.0
        for i in range(self.num_points - 1):
            e = states[i + 1] - self.Phi @ states[i]
            prior += 0.5 * float(e @ self.Q_inv @ e)

        r_sup, _, r_int, _ = self._obstacle_residuals(states)
        obs = 0.5 * float(np.sum(r_sup * r_sup)) + 0.5 * float(np.sum(r_int * r_int))

        self.prior_cost = prior
        self.obs_cost = obs
        return prior + obs

    # -------------------------------------------------------------- GN system
    def _free_index(self, i: int) -> int:
        """Variable-block offset for free (interior) state ``i``; -1 if fixed."""
        if i <= 0 or i >= self.num_points - 1:
            return -1
        return (i - 1) * 4

    def _prior_precision(self) -> np.ndarray:
        """Constant GP-prior precision K^-1 = B^T Q^-1 B over the free states (Eq. 13)."""
        d = (self.num_points - 2) * 4
        kinv = np.zeros((d, d), dtype=np.float64)
        I4 = np.eye(4)
        for i in range(self.num_points - 1):
            blocks = [(i, -self.Phi), (i + 1, I4)]
            for (ia, Ja) in blocks:
                va = self._free_index(ia)
                if va < 0:
                    continue
                JaT_W = Ja.T @ self.Q_inv
                for (ib, Jb) in blocks:
                    vb = self._free_index(ib)
                    if vb < 0:
                        continue
                    kinv[va:va + 4, vb:vb + 4] += JaT_W @ Jb
        kinv += 1e-6 * np.eye(d)
        return kinv

    def _gradient(self, states: np.ndarray) -> np.ndarray:
        """Cost-functional gradient grad U = grad F_obs + grad F_gp (Eq. 25)."""
        d = (self.num_points - 2) * 4
        g = np.zeros(d, dtype=np.float64)

        def add_factor(blocks, residual, weight):
            for (ia, Ja) in blocks:
                va = self._free_index(ia)
                if va < 0:
                    continue
                g[va:va + 4] += (Ja.T @ weight) @ residual

        # GP prior gradient grad F_gp = K^-1 (xi - mu), via the transition residuals.
        I4 = np.eye(4)
        for i in range(self.num_points - 1):
            e = states[i + 1] - self.Phi @ states[i]
            add_factor([(i, -self.Phi), (i + 1, I4)], e, self.Q_inv)

        # Obstacle residuals (support + GP-interpolated) from one batched SDF pass.
        r_sup, drdp_sup, r_int, drdp_int = self._obstacle_residuals(states)

        # Obstacle gradient grad F_obs at the support states (scalar hinge residual).
        w1 = np.array([[1.0]])
        for i in range(1, self.num_points - 1):
            r = r_sup[i - 1]
            if r == 0.0:
                continue
            J = np.zeros((1, 4))
            J[0, :2] = drdp_sup[i - 1]
            add_factor([(i, J)], np.array([r]), w1)

        # ... and at the GP-interpolated states (Eq. 32-36).
        for i in range(self.num_points - 1):
            for k in range(len(self.interp_taus)):
                r = r_int[i, k]
                if r == 0.0:
                    continue
                Jp = np.zeros((1, 4))
                Jp[0, :2] = drdp_int[i, k]
                add_factor([(i, Jp @ self.lambdas[k]), (i + 1, Jp @ self.psis[k])], np.array([r]), w1)

        return g

    # ------------------------------------------------------------------- step
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if self.iteration >= self.max_iters or self.num_points <= 2:
            self._finalize()
            self.done = True
            return StepResult(done=True, found_path=self.found_path)

        self.iteration += 1

        cost0 = self._total_cost(self.states)

        # GPMP covariant gradient update (Eq. 24/33): xi <- xi - (1/eta) K grad U,
        # i.e. precondition the cost gradient by the GP covariance K = (K^-1)^-1.
        g = self._gradient(self.states)
        delta = -(1.0 / self.eta) * np.linalg.solve(self.Kinv_prior, g)
        delta = delta.reshape(-1, 4)

        # Stability guard: cap the per-iteration position change.
        pos_norm = float(np.linalg.norm(delta[:, :2]))
        if pos_norm > self.max_step:
            delta *= self.max_step / pos_norm

        self.states[1:-1] += delta
        self.states[1:-1, 0] = np.clip(self.states[1:-1, 0], 0, self.w - 1)
        self.states[1:-1, 1] = np.clip(self.states[1:-1, 1], 0, self.h - 1)

        self.total_cost = self._total_cost(self.states)
        self._check_path_validity()
        if self.found_path:
            self.best_valid_trajectory = self.states[:, :2].copy()

        # Convergence: negligible change in cost or a vanishing step.
        if (abs(cost0 - self.total_cost) < 1e-4 or pos_norm < 1e-3) and self.found_path:
            self.converged = True
            self._finalize()
            self.done = True

        return StepResult(path_improved=self.found_path)

    def _finalize(self):
        if self.best_valid_trajectory is not None:
            self.states[:, :2] = self.best_valid_trajectory
            self.found_path = True
        else:
            self._check_path_validity()

    def _check_path_validity(self):
        pos = self.states[:, :2]
        for i in range(len(pos) - 1):
            p1 = (int(pos[i, 0]), int(pos[i, 1]))
            p2 = (int(pos[i + 1, 0]), int(pos[i + 1, 1]))
            if not line_collision_free(p1, p2, self.occ):
                self.found_path = False
                return
        self.found_path = True

    def extract_path(self) -> List[Tuple[int, int]]:
        return [(int(p[0]), int(p[1])) for p in self.states[:, :2]]

    def extract_display_path(self) -> List[Tuple[float, float]]:
        return [(float(p[0]), float(p[1])) for p in self.states[:, :2]]

    def get_status(self) -> str:
        status = "converged" if self.converged else ("FOUND" if self.found_path else "optimizing")
        return (f"GPMP: iter {self.iteration}, prior {self.prior_cost:.1f}, "
                f"obs {self.obs_cost:.1f}, {status}")

