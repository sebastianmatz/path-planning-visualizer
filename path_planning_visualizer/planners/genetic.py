from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..geometry import (
    line_collision_free,
    make_distance_field,
)
from .base import BasePlanner, StepResult


class GeneticPlanner(BasePlanner):
    """Genetic Algorithm for path planning.
    
    Uses selection, crossover, and mutation to evolve a population of paths.
    """
    
    name = "Genetic"
    description = "Genetic Algorithm"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 pop_size: int = 50, num_points: int = 20, max_iters: int = 500,
                 mutation_rate: float = 0.1, seed: int = 42):
        super().__init__(occ, start, goal)
        
        self.pop_size = pop_size
        self.num_points = num_points
        self.max_iters = max_iters
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        self.max_mutation_rate = 0.45
        self.rng = np.random.default_rng(seed)
        
        # Initialize population
        self.population = np.zeros((pop_size, num_points, 2))
        start_arr = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)
        
        for i in range(pop_size):
            for j in range(num_points):
                t = j / (num_points - 1)
                base = start_arr + t * (goal_arr - start_arr)
                noise = self.rng.normal(0, 40, 2) if 0 < j < num_points - 1 else 0
                self.population[i, j] = base + noise
            self.population[i, 0] = start_arr
            self.population[i, -1] = goal_arr

        # Distance field must be ready before fitness evaluation.
        self.dist_field = make_distance_field(self.occ)

        # Collision/clearance penalties for segment-aware fitness.
        self.collision_penalty = 3000.0
        self.clearance_distance = 14.0
        self.clearance_weight = 20.0
        self.valid_individuals = 0
        self.no_valid_generations = 0
        self.random_immigrant_fraction = 0.20
        self.hard_restart_window = 18
        self.hard_restart_fraction = 0.60
        
        # Fitness scores
        self.fitness = np.zeros(pop_size)
        self._evaluate_population()
        
        # Best individual
        best_idx = np.argmax(self.fitness)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def _segment_collision_samples(self, p1: np.ndarray, p2: np.ndarray) -> int:
        seg_len = float(np.linalg.norm(p2 - p1))
        return int(np.clip(np.ceil(seg_len * 0.75), 8, 80))

    def _segment_cost_and_collision(self, p1: np.ndarray, p2: np.ndarray, samples: int) -> Tuple[float, bool]:
        penalty = 0.0
        for k in range(samples + 1):
            t = k / samples
            x = int(np.clip(round(p1[0] + t * (p2[0] - p1[0])), 0, self.w - 1))
            y = int(np.clip(round(p1[1] + t * (p2[1] - p1[1])), 0, self.h - 1))

            if self.occ[y, x]:
                return self.collision_penalty, True

            d = self.dist_field[y, x]
            if d < self.clearance_distance:
                delta = self.clearance_distance - d
                penalty += self.clearance_weight * delta * delta
        return penalty, False

    def _is_path_valid(self, path: np.ndarray) -> bool:
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            samples = self._segment_collision_samples(p1, p2)
            p1i = (int(p1[0]), int(p1[1]))
            p2i = (int(p2[0]), int(p2[1]))
            if not line_collision_free(p1i, p2i, self.occ, samples=samples):
                return False
        return True

    def _repair_individual(self, individual: np.ndarray, max_passes: int = 2, tries_per_segment: int = 8) -> np.ndarray:
        """Local collision repair: nudges offending waypoints sideways to open free segments."""
        repaired = individual.copy()
        n = len(repaired)
        if n <= 2:
            return repaired

        for _ in range(max_passes):
            changed = False
            for i in range(n - 1):
                p1 = repaired[i]
                p2 = repaired[i + 1]
                if self._is_segment_free(p1, p2):
                    continue

                # Move a non-fixed endpoint of the offending segment.
                if 0 < i + 1 < n - 1:
                    move_idx = i + 1
                    anchor = p1
                elif 0 < i < n - 1:
                    move_idx = i
                    anchor = p2
                else:
                    continue

                base = repaired[move_idx].copy()
                direction = base - anchor
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    direction = np.array([1.0, 0.0])
                    norm = 1.0
                direction = direction / norm
                perp = np.array([-direction[1], direction[0]])

                found = False
                for _try in range(tries_per_segment):
                    step = 8.0 + 3.0 * _try
                    sign = -1.0 if (_try % 2) else 1.0
                    jitter = self.rng.normal(0.0, 5.0, 2)
                    candidate = base + sign * perp * step + jitter
                    candidate[0] = np.clip(candidate[0], 0, self.w - 1)
                    candidate[1] = np.clip(candidate[1], 0, self.h - 1)
                    repaired[move_idx] = candidate

                    left_ok = True
                    right_ok = True
                    if move_idx > 0:
                        left_ok = self._is_segment_free(repaired[move_idx - 1], repaired[move_idx])
                    if move_idx < n - 1:
                        right_ok = self._is_segment_free(repaired[move_idx], repaired[move_idx + 1])

                    if left_ok and right_ok:
                        found = True
                        changed = True
                        break

                if not found:
                    repaired[move_idx] = base

            if not changed:
                break

        repaired[0] = np.array(self.start, dtype=float)
        repaired[-1] = np.array(self.goal, dtype=float)
        return repaired

    def _is_segment_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        samples = self._segment_collision_samples(p1, p2)
        p1i = (int(p1[0]), int(p1[1]))
        p2i = (int(p2[0]), int(p2[1]))
        return line_collision_free(p1i, p2i, self.occ, samples=samples)
    
    def _evaluate_individual(self, path: np.ndarray) -> float:
        """Evaluate fitness (higher is better)."""
        # Path length cost (shorter is better)
        length = 0.0
        obstacle_penalty = 0.0
        collision_segments = 0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            length += np.linalg.norm(p2 - p1)
            samples = self._segment_collision_samples(p1, p2)
            seg_penalty, seg_collision = self._segment_cost_and_collision(p1, p2, samples)
            obstacle_penalty += seg_penalty
            if seg_collision:
                collision_segments += 1
                # Early abort on invalid segment to speed up evaluation.
                break

        # Waypoint proximity penalty
        for point in path:
            x, y = int(np.clip(point[0], 0, self.w - 1)), int(np.clip(point[1], 0, self.h - 1))
            if self.occ[y, x] > 0:
                obstacle_penalty += self.collision_penalty
            else:
                dist = self.dist_field[y, x]
                if dist < 5:
                    obstacle_penalty += (5 - dist) * 10

        if collision_segments > 0:
            obstacle_penalty += collision_segments * self.collision_penalty
        
        # Smoothness (less turning is better)
        smooth_penalty = 0.0
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0.1 and n2 > 0.1:
                cos_angle = np.dot(v1, v2) / (n1 * n2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                smooth_penalty += angle * 5

        total_cost = length + obstacle_penalty + smooth_penalty

        # Strongly prefer fully valid paths, but preserve gradient among invalid ones.
        secondary = 5000.0 / (1.0 + total_cost)

        # Lexicographic fitness shaping:
        # 1) Any fully valid path dominates any invalid path.
        # 2) Invalid paths still keep useful ranking among themselves.
        if collision_segments == 0:
            return 1_000_000.0 + secondary

        invalid_factor = 1.0 + 5.0 * collision_segments
        return secondary / invalid_factor
    
    def _evaluate_population(self):
        """Evaluate fitness of entire population."""
        valid_count = 0
        for i in range(self.pop_size):
            self.fitness[i] = self._evaluate_individual(self.population[i])
            if self._is_path_valid(self.population[i]):
                valid_count += 1
        self.valid_individuals = valid_count
    
    def _select_parents(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tournament selection."""
        def tournament():
            candidates = self.rng.choice(self.pop_size, size=3, replace=False)
            winner = candidates[np.argmax(self.fitness[candidates])]
            return self.population[winner]
        
        return tournament(), tournament()
    
    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Single-point crossover."""
        child = np.zeros_like(p1)
        crossover_point = self.rng.integers(1, self.num_points - 1)
        child[:crossover_point] = p1[:crossover_point]
        child[crossover_point:] = p2[crossover_point:]
        # Ensure start and goal are correct
        child[0] = p1[0]
        child[-1] = p1[-1]
        return child
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        mutated = individual.copy()
        for i in range(1, self.num_points - 1):
            if self.rng.random() < self.mutation_rate:
                std = 22 if self.valid_individuals == 0 else 12
                mutated[i] += self.rng.normal(0, std, 2)
                mutated[i, 0] = np.clip(mutated[i, 0], 0, self.w - 1)
                mutated[i, 1] = np.clip(mutated[i, 1], 0, self.h - 1)
        return mutated

    def _random_individual(self) -> np.ndarray:
        start_arr = np.array(self.start, dtype=float)
        goal_arr = np.array(self.goal, dtype=float)
        ind = np.zeros((self.num_points, 2), dtype=float)
        for j in range(self.num_points):
            t = j / (self.num_points - 1)
            base = start_arr + t * (goal_arr - start_arr)
            noise = self.rng.normal(0, 45, 2) if 0 < j < self.num_points - 1 else 0
            ind[j] = base + noise
        ind[:, 0] = np.clip(ind[:, 0], 0, self.w - 1)
        ind[:, 1] = np.clip(ind[:, 1], 0, self.h - 1)
        ind[0] = start_arr
        ind[-1] = goal_arr
        return ind
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        if self.iteration >= self.max_iters:
            self._check_best_path()
            self.done = True
            return StepResult(done=True, found_path=self.found_path)
        
        # Create new population
        new_population = np.zeros_like(self.population)
        
        # Elitism: keep best individual
        best_idx = np.argmax(self.fitness)
        new_population[0] = self.population[best_idx].copy()
        
        # Generate rest through selection, crossover, mutation
        for i in range(1, self.pop_size):
            p1, p2 = self._select_parents()
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            # Try local repair to turn near-miss children into valid paths.
            if self.valid_individuals == 0 or self.rng.random() < 0.35:
                child = self._repair_individual(child)
            new_population[i] = child
        
        self.population = new_population

        # If no valid individuals for several generations, inject random immigrants.
        if self.valid_individuals == 0:
            self.no_valid_generations += 1
        else:
            self.no_valid_generations = 0

        if self.no_valid_generations >= 10:
            immigrant_count = max(1, int(self.pop_size * self.random_immigrant_fraction))
            for i in range(self.pop_size - immigrant_count, self.pop_size):
                new_population[i] = self._repair_individual(self._random_individual())
            self.population = new_population
            self.no_valid_generations = 5

        # Hard restart if validity is stuck at zero for too long.
        if self.no_valid_generations >= self.hard_restart_window:
            restart_count = max(1, int(self.pop_size * self.hard_restart_fraction))
            for i in range(self.pop_size - restart_count, self.pop_size):
                new_population[i] = self._repair_individual(self._random_individual())
            # Keep elite and reset stagnation counter.
            new_population[0] = self.best_individual.copy()
            self.population = new_population
            self.no_valid_generations = 0

        # Adaptive mutation schedule.
        if self.valid_individuals == 0:
            self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate + 0.02)
        else:
            self.mutation_rate = max(self.base_mutation_rate, self.mutation_rate - 0.01)

        self._evaluate_population()
        
        # Update best
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_individual = self.population[best_idx].copy()
        
        # Check if best path is valid
        self._check_best_path()
        
        # Visualization
        idx = self.iteration % (self.num_points - 1)
        p1 = (int(self.best_individual[idx, 0]), int(self.best_individual[idx, 1]))
        p2 = (int(self.best_individual[idx + 1, 0]), int(self.best_individual[idx + 1, 1]))
        
        return StepResult(edge=(p1, p2))
    
    def _check_best_path(self):
        """Check if best path is collision-free."""
        path = self.best_individual
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            samples = self._segment_collision_samples(p1, p2)
            p1i = (int(p1[0]), int(p1[1]))
            p2i = (int(p2[0]), int(p2[1]))
            if not line_collision_free(p1i, p2i, self.occ, samples=samples):
                self.found_path = False
                return
        self.found_path = True
    
    def extract_path(self) -> List[Tuple[int, int]]:
        return [(int(p[0]), int(p[1])) for p in self.best_individual]
    
    def get_status(self) -> str:
        status = "FOUND" if self.found_path else "evolving"
        return (
            f"GA: gen {self.iteration}, fitness: {self.best_fitness:.2f}, "
            f"valid: {self.valid_individuals}/{self.pop_size}, mut: {self.mutation_rate:.2f}, {status}"
        )
    
    