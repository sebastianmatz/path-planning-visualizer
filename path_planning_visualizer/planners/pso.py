from __future__ import annotations

from typing import List, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)

from ..types import Point
from ..geometry import (
    make_distance_field,
)
from .base import BasePlanner, StepResult


class PSOParamsWidget(QWidget):
    """Parameters widget for PSO planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_num_particles = QSpinBox()
        self.spin_num_particles.setRange(10, 200)
        self.spin_num_particles.setValue(30)
        self.spin_num_particles.setToolTip("Number of particles")
        
        self.spin_num_points = QSpinBox()
        self.spin_num_points.setRange(5, 100)
        self.spin_num_points.setValue(30)
        self.spin_num_points.setToolTip("Waypoints per path")
        
        self.spin_max_iters = QSpinBox()
        self.spin_max_iters.setRange(50, 5000)
        self.spin_max_iters.setValue(1200)
        self.spin_max_iters.setToolTip("Maximum iterations")

        self.chk_safeguards = QCheckBox("Enable safeguards (non-1995)")
        self.chk_safeguards.setChecked(False)
        self.chk_safeguards.setToolTip(
            "Off = exact Kennedy & Eberhart (1995) update: v += 2*r1*(pbest-x) + 2*r2*(gbest-x), "
            "Vmax clamp, full momentum (w=1.0).\n"
            "On = non-1995 robustness: adaptive inertia/social gain, diversity injection, "
            "random immigrants, and swarm restart for cluttered maps."
        )

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Random seed for reproducibility")

        layout.addRow("Particles:", self.spin_num_particles)
        layout.addRow("Waypoints:", self.spin_num_points)
        layout.addRow("Max iters:", self.spin_max_iters)
        layout.addRow(self.chk_safeguards)
        layout.addRow("Seed:", self.spin_seed)

        self.setLayout(layout)

    def get_params(self) -> dict:
        return {
            'num_particles': self.spin_num_particles.value(),
            'num_points': self.spin_num_points.value(),
            'max_iters': self.spin_max_iters.value(),
            'enable_safeguards': self.chk_safeguards.isChecked(),
            'seed': self.spin_seed.value(),
        }


class PSOPlanner(BasePlanner):
    """PSO - Particle Swarm Optimization for path planning.
    
    Each particle represents a complete path. Particles move through the search
    space influenced by their best known position and the swarm's best position.
    """
    
    name = "PSO"
    description = "Particle Swarm Optimization (Kennedy & Eberhart 1995) over waypoint paths"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_particles: int = 30, num_points: int = 20, max_iters: int = 500,
                 w: float = 1.0, c1: float = 2.0, c2: float = 2.0, vmax: float = 20.0,
                 enable_safeguards: bool = False, seed: int = 42):
        super().__init__(occ, start, goal)

        self.num_particles = num_particles
        self.num_points = num_points
        self.max_iters = max_iters
        # Exact Kennedy & Eberhart (1995, sec. 3.6) defaults: full momentum (no
        # inertia weight, w = 1.0), acceleration constants c1 = c2 = 2.0, Vmax clamp.
        self.w_inertia = w  # Inertia weight (1.0 = the 1995 form, full momentum)
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.vmax = float(vmax)  # velocity clamp |v| <= Vmax
        # Off-by-default non-1995 robustness heuristics (adaptive inertia/social
        # gain, diversity injection, random immigrants, swarm restart).
        self.enable_safeguards = bool(enable_safeguards)
        self.rng = np.random.default_rng(seed)
        
        # Initialize particles (each is a path with num_points waypoints)
        # Shape: (num_particles, num_points, 2)
        self.particles = np.zeros((num_particles, num_points, 2))
        self.velocities = np.zeros((num_particles, num_points, 2))
        
        start_arr = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)
        
        for i in range(num_particles):
            # Initialize with straight line + noise
            for j in range(self.num_points):
                t = j / (self.num_points - 1)
                base = start_arr + t * (goal_arr - start_arr)
                noise = self.rng.normal(0, 30, 2) if 0 < j < self.num_points - 1 else 0
                self.particles[i, j] = base + noise
            # Fix start and goal
            self.particles[i, 0] = start_arr
            self.particles[i, -1] = goal_arr
        
        # Initialize velocities
        self.velocities = self.rng.normal(0, 5, (num_particles, num_points, 2))
        self.velocities[:, 0] = 0  # Don't move start
        self.velocities[:, -1] = 0  # Don't move goal
        
        # Personal best for each particle
        self.p_best = self.particles.copy()
        self.p_best_cost = np.full(num_particles, float('inf'))
        self.p_best_valid = np.zeros(num_particles, dtype=bool)
        
        # Global best
        self.g_best = self.particles[0].copy()
        self.g_best_cost = float('inf')
        
        # Distance field for obstacle cost - MUST be initialized before _evaluate_path
        self.dist_field = make_distance_field(self.occ)

        # Collision/clearance tuning
        self.collision_penalty = 1e7
        self.clearance_distance = 18.0
        self.clearance_weight = 30.0
        self.display_smoothing_passes = 1
        self.g_best_is_valid = False
        self.early_exit_on_collision = True
        self.social_gain_when_invalid = 0.05

        # Diversification settings to escape repeated invalid corridors.
        self.mutation_std = 24.0
        self.mutation_fraction = 0.50
        self.random_immigrant_fraction = 0.20
        self.stagnation_window = 14
        self.no_improve_iters = 0
        self.valid_particle_count = 0
        self.zero_valid_iters = 0
        self.zero_valid_restart_window = 180
        self.restart_fraction = 0.55

        # Adaptive inertia: broad exploration early, stable convergence later.
        self.w_max = 0.90
        self.w_min = 0.35
        
        # Evaluate initial positions
        for i in range(num_particles):
            cost, is_valid = self._evaluate_path(self.particles[i])
            self.p_best_cost[i] = cost
            self.p_best_valid[i] = is_valid
            self._maybe_update_global_best(self.particles[i], cost, is_valid)
            if np.array_equal(self.g_best, self.particles[i]):
                self.best_particle_idx = i

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return ((angle + np.pi) % (2.0 * np.pi)) - np.pi

    def _segment_collision_samples(self, p1: np.ndarray, p2: np.ndarray) -> int:
        """Adaptive samples based on segment length."""
        seg_len = float(np.linalg.norm(p2 - p1))
        return int(np.clip(np.ceil(seg_len * 0.75), 8, 80))

    def _segment_grid_points(self, p1: np.ndarray, p2: np.ndarray) -> List[Point]:
        """Rasterize a segment and return all visited grid points for robust collision checks."""
        x0 = int(np.clip(round(p1[0]), 0, self.w - 1))
        y0 = int(np.clip(round(p1[1]), 0, self.h - 1))
        x1 = int(np.clip(round(p2[0]), 0, self.w - 1))
        y1 = int(np.clip(round(p2[1]), 0, self.h - 1))

        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy)))
        if steps == 0:
            return [(x0, y0)]

        pts: List[Point] = []
        for i in range(steps + 1):
            t = i / steps
            x = int(np.clip(round(x0 + t * dx), 0, self.w - 1))
            y = int(np.clip(round(y0 + t * dy), 0, self.h - 1))
            if not pts or pts[-1] != (x, y):
                pts.append((x, y))
        return pts

    def _segment_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        for x, y in self._segment_grid_points(p1, p2):
            if self.occ[y, x]:
                return False
        return True

    def _segment_cost_and_collision(self, p1: np.ndarray, p2: np.ndarray, samples: int) -> Tuple[float, bool]:
        """Compute segment clearance/collision cost in a single sampling pass."""
        cost = 0.0
        for x, y in self._segment_grid_points(p1, p2):

            if self.occ[y, x]:
                return self.collision_penalty, True

            d = self.dist_field[y, x]
            if d < self.clearance_distance:
                delta = self.clearance_distance - d
                cost += self.clearance_weight * delta * delta
        return cost, False

    def _maybe_update_global_best(self, candidate: np.ndarray, cost: float, is_valid: bool) -> None:
        """Prefer valid global best paths; allow invalid only as fallback until a valid one exists."""
        if is_valid:
            if (not self.g_best_is_valid) or (cost < self.g_best_cost):
                self.g_best = candidate.copy()
                self.g_best_cost = cost
                self.g_best_is_valid = True
            return

        if (not self.g_best_is_valid) and (cost < self.g_best_cost):
            self.g_best = candidate.copy()
            self.g_best_cost = cost

    def _point_obstacle_penalty(self, x: int, y: int) -> float:
        """Penalty at a single grid point based on obstacle proximity."""
        if self.occ[y, x] > 0:
            return 1000.0

        dist = self.dist_field[y, x]
        if dist < 10:
            return float((10 - dist) * 5)
        return 0.0

    def _segment_obstacle_cost(self, p1: np.ndarray, p2: np.ndarray, samples: int = 12) -> float:
        """Accumulate obstacle penalty along a segment to catch crossings between waypoints."""
        cost = 0.0
        for k in range(samples + 1):
            t = k / samples
            pt = (1.0 - t) * p1 + t * p2
            x = int(np.clip(pt[0], 0, self.w - 1))
            y = int(np.clip(pt[1], 0, self.h - 1))
            cost += self._point_obstacle_penalty(x, y)
        return cost
    
    def _evaluate_path(self, path: np.ndarray) -> Tuple[float, bool]:
        """Evaluate path cost (lower is better) and validity."""
        # Path length cost
        length_cost = 0.0
        for i in range(len(path) - 1):
            length_cost += np.linalg.norm(path[i+1] - path[i])
        
        # Obstacle/clearance cost and segment collision accounting
        obstacle_cost = 0.0
        collision_segments = 0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            samples = self._segment_collision_samples(p1, p2)
            seg_cost, seg_collision = self._segment_cost_and_collision(p1, p2, samples)
            obstacle_cost += seg_cost
            if seg_collision:
                collision_segments += 1
                if self.early_exit_on_collision:
                    remaining = (len(path) - 2) - i
                    obstacle_cost += remaining * self.collision_penalty
                    break
        
        # Smoothness cost
        smooth_cost = 0.0
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
                continue
            raw_diff = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle_diff = self._wrap_angle(raw_diff)
            smooth_cost += abs(angle_diff) * 10

        if collision_segments > 0:
            obstacle_cost += 0.0

        total_cost = length_cost + obstacle_cost + smooth_cost
        return total_cost, collision_segments == 0

    def _mutate_particle(self, idx: int, std: float) -> None:
        """Apply strong random perturbation to internal waypoints to explore new corridors."""
        if self.num_points <= 2:
            return

        self.particles[idx, 1:-1] += self.rng.normal(0.0, std, (self.num_points - 2, 2))
        self.particles[idx, :, 0] = np.clip(self.particles[idx, :, 0], 0, self.w - 1)
        self.particles[idx, :, 1] = np.clip(self.particles[idx, :, 1], 0, self.h - 1)
        self.particles[idx, 0] = np.array(self.start, dtype=float)
        self.particles[idx, -1] = np.array(self.goal, dtype=float)

        self.velocities[idx] = 0.0

    def _randomize_particle(self, idx: int, base_std: float = 42.0) -> None:
        """Reinitialize a particle as random line+noise immigrant to escape swarm collapse."""
        start_arr = np.array(self.start, dtype=float)
        goal_arr = np.array(self.goal, dtype=float)
        for j in range(self.num_points):
            t = j / (self.num_points - 1)
            base = start_arr + t * (goal_arr - start_arr)
            noise = self.rng.normal(0.0, base_std, 2) if 0 < j < self.num_points - 1 else 0.0
            self.particles[idx, j] = base + noise

        self.particles[idx, :, 0] = np.clip(self.particles[idx, :, 0], 0, self.w - 1)
        self.particles[idx, :, 1] = np.clip(self.particles[idx, :, 1], 0, self.h - 1)
        self.particles[idx, 0] = start_arr
        self.particles[idx, -1] = goal_arr
        self.velocities[idx] = self.rng.normal(0.0, 8.0, (self.num_points, 2))
        self.velocities[idx, 0] = 0.0
        self.velocities[idx, -1] = 0.0

    def _inject_diversity(self) -> None:
        """Mutate worst particles when the swarm stagnates or has no valid global best."""
        if self.num_particles <= 2:
            return

        valid_ratio = self.valid_particle_count / max(1, self.num_particles)

        # Adaptive mutation: stronger without valid particles, weaker once valid paths exist.
        if valid_ratio <= 0.0:
            mutation_fraction = min(0.70, self.mutation_fraction + 0.15)
            std = self.mutation_std * 1.7
        elif valid_ratio < 0.2:
            mutation_fraction = self.mutation_fraction
            std = self.mutation_std * 1.2
        else:
            mutation_fraction = max(0.20, self.mutation_fraction - 0.20)
            std = max(10.0, self.mutation_std * 0.6)

        mutate_count = max(1, int(self.num_particles * mutation_fraction))
        worst_indices = np.argsort(self.p_best_cost)[-mutate_count:]

        for idx in worst_indices:
            if idx == getattr(self, 'best_particle_idx', -1):
                continue
            self._mutate_particle(int(idx), std)

        if not self.g_best_is_valid:
            immigrant_count = max(1, int(self.num_particles * self.random_immigrant_fraction))
            immigrants = np.argsort(self.p_best_cost)[-immigrant_count:]
            for idx in immigrants:
                if idx == getattr(self, 'best_particle_idx', -1):
                    continue
                self._randomize_particle(int(idx))

    def _restart_swarm_worst(self) -> None:
        """Hard restart of worst particles after long zero-valid stagnation."""
        if self.num_particles <= 2:
            return

        restart_count = max(1, int(self.num_particles * self.restart_fraction))
        worst_indices = np.argsort(self.p_best_cost)[-restart_count:]

        for idx in worst_indices:
            if idx == getattr(self, 'best_particle_idx', -1):
                continue
            j = int(idx)
            self._randomize_particle(j, base_std=50.0)
            self.p_best[j] = self.particles[j].copy()
            self.p_best_cost[j] = float('inf')
            self.p_best_valid[j] = False

        self.no_improve_iters = 0

    def _smooth_path_for_display(self, path: np.ndarray) -> np.ndarray:
        """Optional light smoothing while preserving collision-free segments."""
        smoothed = path.copy()
        n = len(smoothed)
        if n <= 2:
            return smoothed

        for _ in range(self.display_smoothing_passes):
            for i in range(1, n - 1):
                candidate = 0.25 * smoothed[i - 1] + 0.5 * smoothed[i] + 0.25 * smoothed[i + 1]
                candidate[0] = np.clip(candidate[0], 0, self.w - 1)
                candidate[1] = np.clip(candidate[1], 0, self.h - 1)

                old_point = smoothed[i].copy()
                smoothed[i] = candidate
                if (not self._segment_collision_free(smoothed[i - 1], smoothed[i]) or
                        not self._segment_collision_free(smoothed[i], smoothed[i + 1])):
                    smoothed[i] = old_point

        smoothed[0] = np.array(self.start, dtype=float)
        smoothed[-1] = np.array(self.goal, dtype=float)
        return smoothed
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if self.iteration >= self.max_iters:
            self._check_best_path()
            self.done = True
            return StepResult(done=True, found_path=self.found_path)

        self.iteration += 1
        prev_best_cost = self.g_best_cost
        prev_best_valid = self.g_best_is_valid
        valid_count = 0

        if self.enable_safeguards:
            # Non-1995: adaptive inertia (Shi & Eberhart 1998) and a social gain
            # that stays low until valid paths exist.
            progress = min(1.0, self.iteration / max(1, self.max_iters))
            inertia = self.w_max - (self.w_max - self.w_min) * progress
            valid_ratio = self.valid_particle_count / max(1, self.num_particles)
            if valid_ratio <= 0.0:
                social_gain_dynamic = self.social_gain_when_invalid
            elif valid_ratio < 0.25:
                social_gain_dynamic = self.social_gain_when_invalid + (valid_ratio / 0.25) * (0.60 - self.social_gain_when_invalid)
            else:
                social_gain_dynamic = 1.0
        else:
            # Exact Kennedy & Eberhart (1995): constant momentum w (= 1.0) and the
            # full social term -- v <- w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x).
            inertia = self.w_inertia
            social_gain_dynamic = 1.0
        
        # Update velocities and positions for each particle
        for i in range(self.num_particles):
            r1 = self.rng.random((self.num_points, 2))
            r2 = self.rng.random((self.num_points, 2))
            
            # Velocity update
            cognitive = self.c1 * r1 * (self.p_best[i] - self.particles[i])
            social_gain = 1.0 if self.g_best_is_valid else social_gain_dynamic
            social = self.c2 * social_gain * r2 * (self.g_best - self.particles[i])
            self.velocities[i] = inertia * self.velocities[i] + cognitive + social

            # Clamp velocity to +/- Vmax (Kennedy & Eberhart 1995)
            self.velocities[i] = np.clip(self.velocities[i], -self.vmax, self.vmax)
            self.velocities[i, 0] = 0  # Don't move start
            self.velocities[i, -1] = 0  # Don't move goal
            
            # Position update
            self.particles[i] += self.velocities[i]
            
            # Clamp to bounds
            self.particles[i, :, 0] = np.clip(self.particles[i, :, 0], 0, self.w - 1)
            self.particles[i, :, 1] = np.clip(self.particles[i, :, 1], 0, self.h - 1)
            
            # Evaluate
            cost, is_valid = self._evaluate_path(self.particles[i])
            if is_valid:
                valid_count += 1
            
            # Update personal best
            if is_valid:
                better = (not self.p_best_valid[i]) or (cost < self.p_best_cost[i])
            else:
                better = (not self.p_best_valid[i]) and (cost < self.p_best_cost[i])

            if better:
                self.p_best_cost[i] = cost
                self.p_best[i] = self.particles[i].copy()
                self.p_best_valid[i] = is_valid

            old_best_cost = self.g_best_cost
            self._maybe_update_global_best(self.particles[i], cost, is_valid)
            if self.g_best_cost != old_best_cost and np.array_equal(self.g_best, self.particles[i]):
                self.best_particle_idx = i

        improved = (self.g_best_is_valid and not prev_best_valid) or (self.g_best_cost < prev_best_cost)
        if improved:
            self.no_improve_iters = 0
        else:
            self.no_improve_iters += 1

        self.valid_particle_count = valid_count

        if valid_count == 0:
            self.zero_valid_iters += 1
        else:
            self.zero_valid_iters = 0

        if self.enable_safeguards:
            if (not self.g_best_is_valid and self.iteration % 4 == 0) or (self.no_improve_iters >= self.stagnation_window):
                self._inject_diversity()
                if self.no_improve_iters >= self.stagnation_window:
                    self.no_improve_iters = 0

            if self.zero_valid_iters >= self.zero_valid_restart_window:
                self._restart_swarm_worst()
                self.zero_valid_iters = 0
        
        # Check if best path is collision-free
        self.found_path = self.g_best_is_valid
        if self.iteration % 10 == 0:
            self._check_best_path()

        if not self.g_best_is_valid:
            # While searching, visualize exploratory particle motion so progress remains visible.
            vis_particle_idx = self.iteration % self.num_particles
            vis_seg_idx = self.iteration % (self.num_points - 1)

            p1_arr = self.particles[vis_particle_idx, vis_seg_idx]
            p2_arr = self.particles[vis_particle_idx, vis_seg_idx + 1]
            p1 = (int(p1_arr[0]), int(p1_arr[1]))
            p2 = (int(p2_arr[0]), int(p2_arr[1]))

            if self._segment_collision_free(p1_arr, p2_arr):
                return StepResult(edge=(p1, p2))

            mid = (int(round((p1[0] + p2[0]) * 0.5)), int(round((p1[1] + p2[1]) * 0.5)))
            return StepResult(rejected_point=mid)
        
        # Visualization: show best particle's path segment
        idx = self.iteration % (self.num_points - 1)
        p1 = (int(self.g_best[idx, 0]), int(self.g_best[idx, 1]))
        p2 = (int(self.g_best[idx + 1, 0]), int(self.g_best[idx + 1, 1]))

        if not self._segment_collision_free(self.g_best[idx], self.g_best[idx + 1]):
            return StepResult()
        
        return StepResult(edge=(p1, p2))
    
    def _check_best_path(self):
        """Check if best path is collision-free."""
        if not self.g_best_is_valid:
            self.found_path = False
            return

        path = self.g_best
        for i in range(len(path) - 1):
            if not self._segment_collision_free(path[i], path[i + 1]):
                self.found_path = False
                return
        self.found_path = True
    
    def extract_path(self) -> List[Tuple[int, int]]:
        path = self.g_best
        if self.done and self.found_path:
            path = self._smooth_path_for_display(path)
        return [(int(p[0]), int(p[1])) for p in path]
    
    def get_status(self) -> str:
        status = "FOUND" if self.found_path else "searching"
        return (
            f"PSO: iter {self.iteration}, best_cost: {self.g_best_cost:.0f}, "
            f"valid_particles: {self.valid_particle_count}/{self.num_particles}, {status}"
        )
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return PSOParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'PSOPlanner':
        params = params_widget.get_params()
        return PSOPlanner(occ, start, goal, **params)
