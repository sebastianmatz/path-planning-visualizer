from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..geometry import (
    line_collision_free,
    smooth_display_path,
)
from ..types import Edge, Point
from ._rgg import rgg_radius
from .base import BasePlanner, StepResult


class BITStarPlanner(BasePlanner):
    """BIT* - Batch Informed Trees.
    
    BIT* uses batches of samples and processes edges in order of potential
    solution quality. It combines the benefits of RRT* (anytime, asymptotic
    optimality) with graph-based search (ordered edge processing).
    """
    
    name = "BIT*"
    description = "Batch-informed tree search with ordered vertex/edge queues and rewiring in a 2D occupancy-grid adaptation"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 batch_size: int = 200, max_iters: int = 10000, rewire_radius: Optional[float] = None,
                 step_size: float = 26.0, cap_edges_to_step: bool = False, seed: int = 42):
        super().__init__(occ, start, goal)

        self.batch_size = int(max(1, batch_size))
        self.max_iters = int(max(1, max_iters))
        self.step_size = float(max(4.0, step_size))
        # Paper-faithful default: consider every edge within the RGG radius r.
        # When True, cap local connections at step_size (visualization only).
        self.cap_edges_to_step = bool(cap_edges_to_step)
        self.rng = np.random.default_rng(seed)

        self.goal_state = np.array(goal, dtype=float)
        self.start_state = np.array(start, dtype=float)
        self.free_space_volume = float(np.count_nonzero(~self.occ))

        # Tree vertices
        self.V: List[np.ndarray] = [self.start_state.copy()]
        self.parent: Dict[int, Optional[int]] = {0: None}
        self.children: Dict[int, Set[int]] = {0: set()}
        self.g_cost: Dict[int, float] = {0: 0.0}

        # Visualization edges kept in sync incrementally (child_idx -> edge), so
        # the GUI no longer rebuilds the whole tree from parent pointers per step.
        self._edge_by_child: Dict[int, Edge] = {}
        self._tree_edges_cache: Optional[List[Edge]] = None

        # Unconnected samples, including a single goal sample.
        self.X_samples: List[Optional[np.ndarray]] = [self.goal_state.copy()]
        self.sample_point_keys: Set[Point] = {self.goal}

        # Ordered search queues
        self.Q_V: List[Tuple[float, int, int]] = []
        self.Q_E: List[Tuple[float, float, int, int, int, int]] = []
        self._queue_counter = 0
        self.vertex_expanded_batch: Dict[int, int] = {}

        # Solution tracking
        self.best_cost = float('inf')
        self.goal_idx_in_V: Optional[int] = None
        
        # Informed set parameters
        self.c_min = np.linalg.norm(np.array(goal) - np.array(start))
        self.x_center = (np.array(start) + np.array(goal)) / 2
        self.C = self._rotation_to_world()

        # Batch / RGG tracking
        self.batch_count = 0
        self.user_radius = rewire_radius
        self.r = self._compute_radius(rewire_radius)
        self.edge_collision_checks = 0
        # Snapshot of |V| at batch start: only vertices new in the batch enqueue
        # vertex-vertex (rewiring) edges (Alg. 2 line 4).
        self.v_old_count = len(self.V)
        self._new_batch()
    
    def _rotation_to_world(self) -> np.ndarray:
        """Compute rotation matrix from ellipse frame to world frame."""
        diff = np.array(self.goal) - np.array(self.start)
        angle = np.arctan2(diff[1], diff[0])
        return np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    
    def _sample_uniform(self) -> np.ndarray:
        """Sample uniformly in configuration space."""
        x = int(self.rng.integers(0, self.w))
        y = int(self.rng.integers(0, self.h))
        return np.array([x, y], dtype=float)
    
    def _sample_ellipse(self) -> np.ndarray:
        """Sample uniformly in the informed set (ellipse)."""
        if self.best_cost >= float('inf'):
            return self._sample_uniform()
        
        c_best = self.best_cost
        # Semi-axes of the ellipse
        r1 = c_best / 2.0
        r2_sq = c_best**2 - self.c_min**2
        r2 = 1.0 if r2_sq <= 0 else np.sqrt(r2_sq) / 2.0
        
        # Sample in unit disk
        theta = self.rng.uniform(0, 2 * np.pi)
        rho = np.sqrt(self.rng.uniform(0, 1))
        
        # Scale to ellipse
        x_ell = np.array([r1 * rho * np.cos(theta), r2 * rho * np.sin(theta)])
        
        # Rotate and translate
        x_world = self.C @ x_ell + self.x_center
        
        # Clamp to bounds and discretize to the grid used by the environment.
        x_world[0] = np.clip(np.round(x_world[0]), 0, self.w - 1)
        x_world[1] = np.clip(np.round(x_world[1]), 0, self.h - 1)
        return x_world.astype(float)
    
    def _g_hat(self, v_idx: int) -> float:
        """Estimated cost from start to vertex v."""
        return self.g_cost.get(v_idx, float('inf'))
    
    def _h_hat(self, x: np.ndarray) -> float:
        """Estimated cost from x to goal."""
        return np.linalg.norm(x - self.goal_state)
    
    def _f_hat(self, v_idx: int, x: np.ndarray) -> float:
        """Estimated total cost through edge (v, x)."""
        v = self.V[v_idx]
        g_v = self._g_hat(v_idx)
        c_vx = np.linalg.norm(v - x)  # Edge cost
        h_x = self._h_hat(x)
        return g_v + c_vx + h_x

    def _point_key(self, x: np.ndarray) -> Point:
        return (int(round(float(x[0]))), int(round(float(x[1]))))

    def _compute_radius(self, radius: Optional[float]) -> float:
        if radius is not None and radius > 0:
            return float(radius)
        # BIT*/RRG use the 2*(1+1/d)^(1/d) constant over q = |V| + |X_samples|.
        q = len(self.V) + self.num_unconnected_samples()
        return rgg_radius(q, self.free_space_volume, plus_one=True)

    def num_unconnected_samples(self) -> int:
        return sum(1 for x in self.X_samples if x is not None)

    def _sample_free(self) -> Optional[np.ndarray]:
        for _ in range(500):
            x_new = self._sample_ellipse()
            ix, iy = self._point_key(x_new)
            if self.occ[iy, ix]:
                continue
            if (ix, iy) == self.start or (ix, iy) in self.sample_point_keys:
                continue
            self.sample_point_keys.add((ix, iy))
            return np.array([ix, iy], dtype=float)
        return None

    def _vertex_key(self, v_idx: int) -> float:
        return self._g_hat(v_idx) + self._h_hat(self.V[v_idx])

    def _edge_key(self, parent_idx: int, target: np.ndarray) -> float:
        return self._f_hat(parent_idx, target)

    def _push_vertex(self, v_idx: int) -> None:
        key = self._vertex_key(v_idx)
        self._queue_counter += 1
        heapq.heappush(self.Q_V, (key, self._queue_counter, v_idx))

    def _push_edge(self, key: float, parent_idx: int, target_kind: int, target_idx: int) -> None:
        self._queue_counter += 1
        # Tie-break by the source vertex cost-to-come g_T(v) (Alg. 1 line 12),
        # then insertion order.
        heapq.heappush(
            self.Q_E,
            (key, self._g_hat(parent_idx), self._queue_counter, parent_idx, target_kind, target_idx),
        )

    def _iter_active_samples(self):
        for idx, sample in enumerate(self.X_samples):
            if sample is not None:
                yield idx, sample

    def _edge_radius(self) -> float:
        """Neighbourhood radius for candidate edges.

        Paper-faithful BIT* uses the full RGG radius ``self.r``; the optional
        visualization cap limits connections to the steering ``step_size``.
        """
        if self.cap_edges_to_step:
            return min(self.r, self.step_size)
        return self.r

    def _near_samples(self, v_idx: int) -> List[Tuple[int, np.ndarray, float]]:
        v = self.V[v_idx]
        near: List[Tuple[int, np.ndarray, float]] = []
        edge_radius = self._edge_radius()
        for idx, sample in self._iter_active_samples():
            d = float(np.linalg.norm(v - sample))
            if d <= edge_radius:
                near.append((idx, sample, d))
        return near

    def _near_vertices(self, v_idx: int) -> List[Tuple[int, float]]:
        v = self.V[v_idx]
        near: List[Tuple[int, float]] = []
        edge_radius = self._edge_radius()
        for idx, other in enumerate(self.V):
            if idx == v_idx:
                continue
            d = float(np.linalg.norm(v - other))
            if d <= edge_radius:
                near.append((idx, d))
        return near

    def _is_ancestor(self, ancestor_idx: int, v_idx: int) -> bool:
        current = self.parent.get(v_idx)
        while current is not None:
            if current == ancestor_idx:
                return True
            current = self.parent.get(current)
        return False

    def _set_parent(self, child_idx: int, parent_idx: Optional[int]) -> None:
        old_parent = self.parent.get(child_idx)
        if old_parent is not None:
            self.children.setdefault(old_parent, set()).discard(child_idx)
        self.parent[child_idx] = parent_idx
        if parent_idx is not None:
            self.children.setdefault(parent_idx, set()).add(child_idx)
        self.children.setdefault(child_idx, set())
        # Mirror the topology change into the visualization-edge map.
        if parent_idx is None:
            self._edge_by_child.pop(child_idx, None)
        else:
            self._edge_by_child[child_idx] = (
                self._point_key(self.V[parent_idx]),
                self._point_key(self.V[child_idx]),
            )
        self._tree_edges_cache = None

    def _propagate_cost_delta(self, root_idx: int, delta: float) -> None:
        stack = [root_idx]
        while stack:
            current = stack.pop()
            self.g_cost[current] = self.g_cost.get(current, float('inf')) + delta
            self._push_vertex(current)
            stack.extend(self.children.get(current, ()))

    def _is_collision_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        self.edge_collision_checks += 1
        return line_collision_free(self._point_key(a), self._point_key(b), self.occ)

    def _sample_lower_bound(self, x: np.ndarray) -> float:
        return np.linalg.norm(x - self.start_state) + self._h_hat(x)

    def _prune(self) -> None:
        if not np.isfinite(self.best_cost):
            return

        # Drop samples that can provably not improve the incumbent.
        for idx, sample in enumerate(self.X_samples):
            if sample is None:
                continue
            if self._sample_lower_bound(sample) >= self.best_cost:
                self.X_samples[idx] = None

        # Rebuild ordered queues against the incumbent.
        new_qv: List[Tuple[float, int, int]] = []
        for _, _, v_idx in self.Q_V:
            if self._vertex_key(v_idx) < self.best_cost:
                self._queue_counter += 1
                heapq.heappush(new_qv, (self._vertex_key(v_idx), self._queue_counter, v_idx))
        self.Q_V = new_qv

        new_qe: List[Tuple[float, float, int, int, int, int]] = []
        for _, _, _, parent_idx, target_kind, target_idx in self.Q_E:
            if target_kind == 0:
                if target_idx >= len(self.X_samples) or self.X_samples[target_idx] is None:
                    continue
                key = self._edge_key(parent_idx, self.X_samples[target_idx])
            else:
                if target_idx >= len(self.V):
                    continue
                key = self._edge_key(parent_idx, self.V[target_idx])
            if key < self.best_cost:
                self._queue_counter += 1
                heapq.heappush(
                    new_qe,
                    (key, self._g_hat(parent_idx), self._queue_counter, parent_idx, target_kind, target_idx),
                )
        self.Q_E = new_qe
    
    def _new_batch(self):
        """Add a new batch of samples (Alg. 1 lines 4-9)."""
        self.batch_count += 1

        # Alg. 1 line 5: prune samples/edges that cannot improve the incumbent.
        self._prune()

        # Alg. 1 line 7: V_old <- V. Only vertices added during this batch will
        # enqueue vertex-vertex (rewiring) edges in ExpandVertex (Alg. 2 line 4).
        self.v_old_count = len(self.V)

        for _ in range(self.batch_size):
            x_new = self._sample_free()
            if x_new is not None:
                self.X_samples.append(x_new)

        self.r = self._compute_radius(self.user_radius)
        self.Q_V = []
        self.Q_E = []
        self.vertex_expanded_batch = {}
        for v_idx in range(len(self.V)):
            if self._vertex_key(v_idx) < self.best_cost:
                self._push_vertex(v_idx)
    
    def _expand_vertex(self, v_idx: int):
        """Expand vertex by adding edges to nearby samples."""
        if self.vertex_expanded_batch.get(v_idx) == self.batch_count:
            return
        self.vertex_expanded_batch[v_idx] = self.batch_count

        g_v = self._g_hat(v_idx)

        if self._vertex_key(v_idx) >= self.best_cost:
            return

        for x_idx, x, dist_vx in self._near_samples(v_idx):
            f_est = g_v + dist_vx + self._h_hat(x)
            if f_est < self.best_cost:
                self._push_edge(f_est, v_idx, 0, x_idx)

        # Alg. 2 line 4: only vertices added in this batch enqueue vertex-vertex
        # (rewiring) edges; old-old pairs were already considered in earlier batches.
        if v_idx >= self.v_old_count:
            for w_idx, dist_vw in self._near_vertices(v_idx):
                if self._is_ancestor(w_idx, v_idx):
                    continue
                g_new = g_v + dist_vw
                if g_new + self._h_hat(self.V[w_idx]) >= self.best_cost:
                    continue
                if g_new + 1e-9 >= self.g_cost.get(w_idx, float('inf')):
                    continue
                self._push_edge(g_new + self._h_hat(self.V[w_idx]), v_idx, 1, w_idx)
    
    def _is_goal(self, x: np.ndarray) -> bool:
        """Check if x is the goal."""
        return np.linalg.norm(x - np.array(self.goal)) < 1.0
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=self.found_path)
        
        while True:
            while True:
                best_vertex = self._best_vertex_cost()
                best_edge = self._best_edge_cost()
                if not self.Q_V or best_vertex > best_edge:
                    break
                _, _, v_idx = heapq.heappop(self.Q_V)
                if v_idx >= len(self.V):
                    continue
                if self._vertex_key(v_idx) >= self.best_cost:
                    continue
                if self.vertex_expanded_batch.get(v_idx) == self.batch_count:
                    continue
                self._expand_vertex(v_idx)

            if not self.Q_E:
                if self.num_unconnected_samples() == 0:
                    self.done = True
                    return StepResult(done=True, found_path=self.found_path)
                self._new_batch()
                return StepResult()

            _, _, _, parent_idx, target_kind, target_idx = heapq.heappop(self.Q_E)
            if parent_idx >= len(self.V):
                continue

            parent_state = self.V[parent_idx]
            if target_kind == 0:
                if target_idx >= len(self.X_samples):
                    continue
                x = self.X_samples[target_idx]
                if x is None:
                    continue

                edge_cost = float(np.linalg.norm(parent_state - x))
                g_new = self._g_hat(parent_idx) + edge_cost
                f_new = g_new + self._h_hat(x)
                if f_new >= self.best_cost:
                    continue

                p1 = self._point_key(parent_state)
                p2 = self._point_key(x)
                if not self._is_collision_free(parent_state, x):
                    return StepResult(rejected_point=p2)

                new_v_idx = len(self.V)
                self.V.append(x.copy())
                self.g_cost[new_v_idx] = g_new
                self._set_parent(new_v_idx, parent_idx)
                self.X_samples[target_idx] = None
                self._push_vertex(new_v_idx)

                path_improved = False
                if self._is_goal(x) and g_new < self.best_cost:
                    self.best_cost = g_new
                    self.goal_idx_in_V = new_v_idx
                    self.found_path = True
                    path_improved = True
                    self._prune()

                return StepResult(edge=(p1, p2), path_improved=path_improved)

            if target_idx >= len(self.V):
                continue
            if self._is_ancestor(target_idx, parent_idx) or target_idx == parent_idx:
                continue

            child_state = self.V[target_idx]
            edge_cost = float(np.linalg.norm(parent_state - child_state))
            new_cost = self._g_hat(parent_idx) + edge_cost
            current_cost = self.g_cost.get(target_idx, float('inf'))
            if new_cost + 1e-9 >= current_cost:
                continue
            if new_cost + self._h_hat(child_state) >= self.best_cost:
                continue

            p1 = self._point_key(parent_state)
            p2 = self._point_key(child_state)
            if not self._is_collision_free(parent_state, child_state):
                return StepResult(rejected_point=p2)

            delta = new_cost - current_cost
            self._set_parent(target_idx, parent_idx)
            self._propagate_cost_delta(target_idx, delta)

            path_improved = False
            goal_updated = (
                self.goal_idx_in_V is not None
                and (self.goal_idx_in_V == target_idx or self._is_ancestor(target_idx, self.goal_idx_in_V))
            )
            if goal_updated and self.goal_idx_in_V is not None:
                goal_cost = self.g_cost.get(self.goal_idx_in_V, float('inf'))
                if goal_cost < self.best_cost:
                    self.best_cost = goal_cost
                    self.found_path = True
                    path_improved = True
                    self._prune()

            if self.goal_idx_in_V == target_idx and new_cost < self.best_cost:
                self.best_cost = new_cost
                self.found_path = True
                path_improved = True
                self._prune()

            return StepResult(edge=(p1, p2), path_improved=path_improved)
    
    def _best_vertex_cost(self) -> float:
        """Get the best potential cost through any vertex in Q_V."""
        while self.Q_V:
            key, _, v_idx = self.Q_V[0]
            if v_idx >= len(self.V):
                heapq.heappop(self.Q_V)
                continue
            current_key = self._vertex_key(v_idx)
            if abs(current_key - key) > 1e-9 or self.vertex_expanded_batch.get(v_idx) == self.batch_count:
                heapq.heappop(self.Q_V)
                continue
            return current_key
        return float('inf')
    
    def _best_edge_cost(self) -> float:
        """Get the best potential cost in edge queue."""
        while self.Q_E:
            key, _, _, parent_idx, target_kind, target_idx = self.Q_E[0]
            if parent_idx >= len(self.V):
                heapq.heappop(self.Q_E)
                continue

            if target_kind == 0:
                if target_idx >= len(self.X_samples) or self.X_samples[target_idx] is None:
                    heapq.heappop(self.Q_E)
                    continue
                current_key = self._edge_key(parent_idx, self.X_samples[target_idx])
            else:
                if target_idx >= len(self.V):
                    heapq.heappop(self.Q_E)
                    continue
                current_key = self._edge_key(parent_idx, self.V[target_idx])

            if abs(current_key - key) > 1e-9:
                heapq.heappop(self.Q_E)
                continue
            return current_key
        return float('inf')
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if self.goal_idx_in_V is None:
            return []
        
        path = []
        current = self.goal_idx_in_V
        while current is not None:
            pos = self.V[current]
            path.append((int(pos[0]), int(pos[1])))
            current = self.parent.get(current)
        path.reverse()
        return path

    def extract_display_path(self) -> List[Tuple[float, float]]:
        path = self.extract_path()
        if len(path) < 3:
            return [(float(p[0]), float(p[1])) for p in path]
        spacing = max(2.0, min(4.0, self.step_size / 4.0))
        return smooth_display_path(path, self.occ, spacing=spacing, iterations=1)

    def extract_tree_edges(self) -> List[Edge]:
        # Served from the incrementally maintained edge map; only rebuilt when the
        # tree changed since the last call (the cache is invalidated in _set_parent).
        if self._tree_edges_cache is None:
            self._tree_edges_cache = list(self._edge_by_child.values())
        return self._tree_edges_cache
    
    def get_status(self) -> str:
        cost_str = f"{self.best_cost:.1f}" if self.best_cost < float('inf') else "inf"
        return (
            f"BIT*: batch {self.batch_count}, V={len(self.V)}, X={self.num_unconnected_samples()}, "
            f"Qv={len(self.Q_V)}, Qe={len(self.Q_E)}, cc={self.edge_collision_checks}, cost: {cost_str}"
        )
    
    