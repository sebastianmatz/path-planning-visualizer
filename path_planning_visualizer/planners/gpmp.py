from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)

from ._trajectory import make_sdf, sdf_query, straight_line, escape_init
from ..geometry import line_collision_free
from .base import BasePlanner, StepResult


class GPMPParamsWidget(QWidget):
    """Parameters widget for GPMP planner."""

    def __init__(self):
        super().__init__()
        layout = QFormLayout()

        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(10, 100)
        self.spin_num_points.setValue(25)
        self.spin_num_points.setToolTip("Number of GP support states")

        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(50, 10000)
        self.spin_max_iters.setValue(200)
        self.spin_max_iters.setToolTip("Maximum Gauss-Newton iterations")

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 50.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(6.0)
        self.spin_sigma.setToolTip("GP process-noise scale. Higher = softer prior.")

        layout.addRow("Support states:", self.spin_num_points)
        layout.addRow("Max iters:", self.spin_max_iters)
        layout.addRow("Prior sigma:", self.spin_sigma)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'sigma': self.spin_sigma.value(),
        }


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
    """GPMP - Gaussian Process Motion Planning (GPMP2 style, Mukadam et al. 2016).

    The trajectory is a set of GP support states ``[x, y, vx, vy]`` drawn from a
    constant-velocity LTI Gaussian-process prior.  Planning is MAP inference on a
    factor graph:

    * **GP prior factors** connect consecutive states through the LTI model
      ``e_i = theta_{i+1} - Phi theta_i`` weighted by the process-noise
      information ``Q^{-1}`` (a block-tridiagonal prior precision);
    * **obstacle factors** penalize states *and* GP-interpolated intermediate
      states whose signed distance falls below a safety margin, using the SDF
      gradient as the Jacobian;
    * optimization is **Gauss-Newton with Levenberg-Marquardt damping**: each
      iteration linearizes all factors, solves the (dense here, sparse in
      general) normal equations, and accepts the step if the objective drops.
    """

    name = "GPMP"
    description = "Gaussian Process Motion Planning with an LTI GP prior and Gauss-Newton MAP inference (Mukadam et al. 2016)"

    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_points: int = 25, max_iters: int = 200, sigma: float = 6.0,
                 obstacle_weight: float = 1.0, dt: float = 1.0):
        super().__init__(occ, start, goal)

        self.num_points = int(max(3, num_points))
        self.max_iters = max_iters
        self.qc = float(sigma) ** 2          # GP process-noise spectral density
        self.obstacle_weight = float(obstacle_weight)
        self.dt = float(dt)
        self.epsilon = 14.0
        self.interp_taus = [0.25, 0.5, 0.75]

        # SDF and gradient.
        self.dist_field, self.grad_x, self.grad_y = make_sdf(self.occ)

        # GP matrices (uniform spacing -> same for every segment).
        self.Phi = _phi(self.dt)
        self.Q = _q(self.dt, self.qc)
        self.Q_inv = np.linalg.inv(self.Q)
        self._build_interpolators()

        # Support states: positions from a (possibly bent) straight line, with a
        # constant-velocity initialization; endpoints have zero velocity.
        self.states = self._init_states()
        self.start_state = self.states[0].copy()
        self.goal_state = self.states[-1].copy()

        # Levenberg-Marquardt state.
        self.lm_lambda = 1e-2
        self.lm_lambda_min = 1e-6
        self.lm_lambda_max = 1e8
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
    def _obstacle_residual(self, pos: np.ndarray) -> Tuple[float, np.ndarray]:
        """Whitened obstacle residual and d(residual)/d(position)."""
        d, normal = sdf_query(self.dist_field, self.grad_x, self.grad_y, pos[0], pos[1])
        if d >= self.epsilon:
            return 0.0, np.zeros(2)
        w = np.sqrt(self.obstacle_weight)
        r = w * (self.epsilon - d)
        # d r / d pos = w * d(eps - d)/d pos = -w * grad(d) = -w * normal
        return r, -w * normal

    def _total_cost(self, states: np.ndarray) -> float:
        prior = 0.0
        for i in range(self.num_points - 1):
            e = states[i + 1] - self.Phi @ states[i]
            prior += 0.5 * float(e @ self.Q_inv @ e)

        obs = 0.0
        for i in range(1, self.num_points - 1):
            r, _ = self._obstacle_residual(states[i, :2])
            obs += 0.5 * r * r
        for i in range(self.num_points - 1):
            for k in range(len(self.interp_taus)):
                inter = self.lambdas[k] @ states[i] + self.psis[k] @ states[i + 1]
                r, _ = self._obstacle_residual(inter[:2])
                obs += 0.5 * r * r

        self.prior_cost = prior
        self.obs_cost = obs
        return prior + obs

    # -------------------------------------------------------------- GN system
    def _free_index(self, i: int) -> int:
        """Variable-block offset for free (interior) state ``i``; -1 if fixed."""
        if i <= 0 or i >= self.num_points - 1:
            return -1
        return (i - 1) * 4

    def _assemble(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = (self.num_points - 2) * 4
        H = np.zeros((d, d), dtype=np.float64)
        g = np.zeros(d, dtype=np.float64)

        def add_factor(blocks, residual, weight):
            # blocks: list of (state_index, jacobian wrt that 4-vector state)
            for (ia, Ja) in blocks:
                va = self._free_index(ia)
                if va < 0:
                    continue
                JaT_W = Ja.T @ weight
                g[va:va + 4] += JaT_W @ residual
                for (ib, Jb) in blocks:
                    vb = self._free_index(ib)
                    if vb < 0:
                        continue
                    H[va:va + 4, vb:vb + 4] += JaT_W @ Jb

        # GP prior factors (4-D residual, weight Q_inv).
        I4 = np.eye(4)
        for i in range(self.num_points - 1):
            e = states[i + 1] - self.Phi @ states[i]
            add_factor([(i, -self.Phi), (i + 1, I4)], e, self.Q_inv)

        # Obstacle factors at support states (scalar residual, unit weight).
        w1 = np.array([[1.0]])
        for i in range(1, self.num_points - 1):
            r, drdp = self._obstacle_residual(states[i, :2])
            if r == 0.0:
                continue
            J = np.zeros((1, 4))
            J[0, :2] = drdp
            add_factor([(i, J)], np.array([r]), w1)

        # Obstacle factors at GP-interpolated states.
        for i in range(self.num_points - 1):
            for k in range(len(self.interp_taus)):
                inter = self.lambdas[k] @ states[i] + self.psis[k] @ states[i + 1]
                r, drdp = self._obstacle_residual(inter[:2])
                if r == 0.0:
                    continue
                Jp = np.zeros((1, 4))
                Jp[0, :2] = drdp
                Ja = Jp @ self.lambdas[k]
                Jb = Jp @ self.psis[k]
                add_factor([(i, Ja), (i + 1, Jb)], np.array([r]), w1)

        return H, g

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
        H, g = self._assemble(self.states)
        d = H.shape[0]

        # Levenberg-Marquardt: try damped Gauss-Newton steps until improvement.
        accepted = False
        for _ in range(8):
            try:
                delta = np.linalg.solve(H + self.lm_lambda * np.eye(d), -g)
            except np.linalg.LinAlgError:
                self.lm_lambda = min(self.lm_lambda_max, self.lm_lambda * 10.0)
                continue

            candidate = self.states.copy()
            candidate[1:-1] += delta.reshape(-1, 4)
            candidate[1:-1, 0] = np.clip(candidate[1:-1, 0], 0, self.w - 1)
            candidate[1:-1, 1] = np.clip(candidate[1:-1, 1], 0, self.h - 1)

            cost1 = self._total_cost(candidate)
            if cost1 < cost0:
                self.states = candidate
                self.lm_lambda = max(self.lm_lambda_min, self.lm_lambda * 0.5)
                accepted = True
                break
            self.lm_lambda = min(self.lm_lambda_max, self.lm_lambda * 10.0)

        self.total_cost = self._total_cost(self.states)
        self._check_path_validity()
        if self.found_path:
            self.best_valid_trajectory = self.states[:, :2].copy()

        # Convergence: no accepted step (LM saturated) or negligible change.
        if not accepted or abs(cost0 - self.total_cost) < 1e-4:
            if self.found_path or self.lm_lambda >= self.lm_lambda_max:
                self.converged = True
                self._finalize()
                self.done = True

        return StepResult(path_improved=(accepted and self.found_path))

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
                f"obs {self.obs_cost:.1f}, lambda {self.lm_lambda:.1e}, {status}")

    @staticmethod
    def get_params_widget() -> QWidget:
        return GPMPParamsWidget()

    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'GPMPPlanner':
        params = params_widget.get_params()
        return GPMPPlanner(occ, start, goal, **params)
