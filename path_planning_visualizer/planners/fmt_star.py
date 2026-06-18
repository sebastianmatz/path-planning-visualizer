from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..geometry import (
    line_collision_free,
)
from ..types import Edge, Point
from ._rgg import rgg_radius
from .base import BasePlanner, StepResult


class FMTStarPlanner(BasePlanner):
    """FMT* - Fast Marching Tree.

    Faithful 2D-point-robot adaptation of Janson, Schmerling, Clark & Pavone (2015,
    IJRR 34(7); cite ``JSC15``, sec. 3.1, Algorithm 1). FMT* samples a batch of free
    states up front, then grows a single tree outward as a *lazy* dynamic-programming
    wavefront: it repeatedly expands the lowest-cost node on the frontier, connecting
    each newly reached node to its locally optimal already-reached neighbour and
    collision-checking only that one candidate edge.

    Paper correspondence (Algorithm 1):

    1. ``x_init`` starts in ``V_open``, every other sample in ``V_unvisited``; tree
       rooted at ``x_init`` -- ``__init__``.
    2. ``z`` = lowest-cost node in ``V_open`` -- the ``open_heap`` pop in ``step_once``.
    3-6. for each ``V_unvisited`` neighbour ``x`` of ``z``, choose the parent
       ``y_min = argmin_{y in V_open cap N(x)} cost(y) + ||y - x||`` and, *only if*
       ``(y_min, x)`` is collision-free, add the edge -- ``step_once``.
    7-8. connected ``x`` move ``V_unvisited -> V_open``; ``z`` moves
       ``V_open -> V_closed``.
    9. terminate when ``V_open`` empties (failure) or the lowest-cost node *popped*
       from ``V_open`` is the goal (success) -- ``step_once``.

    The single-best-edge collision check -- with **no fallback** to other neighbours
    when ``y_min`` is blocked -- is FMT*'s defining "lazy" step.

    Radius ``r_n = gamma (log n / n)^(1/d)`` with the paper's
    ``gamma > 2 (1/d)^(1/d) (mu(X_free)/zeta_d)^(1/d)``; see ``rgg_radius``.

    Adaptations (stated for fidelity):

    - 2D holonomic point robot: neighbours lie in the Euclidean ``r_n``-disk,
      ``Cost(y, x) = ||y - x||``, edges are raster line checks on the grid.
    - ``n`` in ``r_n`` is the requested sample count; the goal region is the single
      goal sample (a ``xi``-regular goal region reduced to a point).

    See ``literature/fidelity/fmt_star.md`` and ``tests/test_fmt_star_fidelity.py``.
    """

    name = "FMT*"
    description = "Fast Marching Tree with uniform free-space sampling and lazy wavefront expansion"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 num_samples: int = 400, radius: Optional[float] = None, seed: int = 42):
        super().__init__(occ, start, goal)
        
        self.num_samples = int(max(50, num_samples))
        self.rng = np.random.default_rng(seed)
        self.samples: List[np.ndarray] = [np.array(start, dtype=float)]
        self._sample_points_uniform()
        self.samples.append(np.array(goal, dtype=float))
        self.goal_idx = len(self.samples) - 1

        self.num_nodes = len(self.samples)
        self.free_space_volume = float(np.count_nonzero(~self.occ))
        self.radius = self._compute_connection_radius(radius)

        self.V_open: Set[int] = {0}
        self.V_closed: Set[int] = set()
        self.V_unvisited: Set[int] = set(range(1, self.num_nodes))

        self.parent: Dict[int, int] = {0: -1}
        self.cost: Dict[int, float] = {0: 0.0}
        self.open_heap: List[Tuple[float, int]] = [(0.0, 0)]

        self.neighbors: Dict[int, List[Tuple[int, float]]] = self._precompute_neighbors()
        self.collision_cache: Dict[Tuple[int, int], bool] = {}
        self.collision_checks = 0

    def _sample_points_uniform(self) -> None:
        """Sample free states uniformly from the occupancy grid."""
        attempts = 0
        seen: Set[Point] = {self.start, self.goal}
        while len(self.samples) < self.num_samples + 1 and attempts < self.num_samples * 50:
            attempts += 1
            x = int(self.rng.integers(0, self.w))
            y = int(self.rng.integers(0, self.h))
            point = (x, y)
            if point in seen or self.occ[y, x]:
                continue
            seen.add(point)
            self.samples.append(np.array(point, dtype=float))

    def _compute_connection_radius(self, radius: Optional[float]) -> float:
        if radius is not None and radius > 0:
            return float(radius)
        # FMT* uses the 2*(1/d)^(1/d) constant (Janson et al. 2015).
        return rgg_radius(self.num_samples, self.free_space_volume, plus_one=False)

    def _precompute_neighbors(self) -> Dict[int, List[Tuple[int, float]]]:
        """Precompute each sample's neighbours within the connection radius ``r_n``.

        The ``r_n``-disk adjacency (Euclidean distance ``<= self.radius``) is fixed once
        and reused every iteration; lists are sorted by distance. This is the disk graph
        over which FMT*'s wavefront runs.
        """
        coords = np.array(self.samples, dtype=np.float64)
        neighborhoods: Dict[int, List[Tuple[int, float]]] = {}
        for idx in range(self.num_nodes):
            deltas = coords - coords[idx]
            dists = np.linalg.norm(deltas, axis=1)
            mask = (dists > 0.0) & (dists <= self.radius)
            nbrs = [(int(j), float(dists[j])) for j in np.where(mask)[0]]
            nbrs.sort(key=lambda item: item[1])
            neighborhoods[idx] = nbrs
        return neighborhoods

    def _neighbors_in_set(self, idx: int, candidate_set: Set[int]) -> List[Tuple[int, float]]:
        return [(other_idx, dist) for other_idx, dist in self.neighbors[idx] if other_idx in candidate_set]

    def _collision_free_indices(self, a_idx: int, b_idx: int) -> bool:
        """Cached collision check for the edge between two samples (Alg. 1 line 6).

        These are the *only* collision checks FMT* performs -- one per locally optimal
        candidate edge -- so the result is cached by unordered index pair to avoid
        re-rasterizing an edge that comes up again.
        """
        key = (a_idx, b_idx) if a_idx <= b_idx else (b_idx, a_idx)
        cached = self.collision_cache.get(key)
        if cached is not None:
            return cached

        a = self.samples[a_idx]
        b = self.samples[b_idx]
        free = line_collision_free((int(a[0]), int(a[1])), (int(b[0]), int(b[1])), self.occ)
        self.collision_cache[key] = free
        self.collision_checks += 1
        return free
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        while self.open_heap and self.open_heap[0][1] not in self.V_open:
            heapq.heappop(self.open_heap)

        if not self.open_heap:
            # Alg. 1 line 9.1: V_open empty -> report failure.
            self.done = True
            return StepResult(done=True, found_path=False)

        _, z = heapq.heappop(self.open_heap)

        # Alg. 1 line 9.2: terminate when the lowest-cost node in V_open is the
        # goal (the goal is connected like any other sample; its cost is final
        # once it enters V_open, so popping it returns the unique optimal path).
        if z == self.goal_idx:
            self.found_path = True
            self.done = True
            return StepResult(done=True, found_path=True)

        # Alg. 1 lines 3-6: for each unvisited neighbour x of z, connect it to the
        # locally optimal one-step parent y_min in V_open, then lazily check that one edge.
        z_neighbors = self._neighbors_in_set(z, self.V_unvisited)

        edges: List[Edge] = []
        H_new: List[int] = []

        for x, _ in z_neighbors:
            x_open_neighbors = self._neighbors_in_set(x, self.V_open)
            if not x_open_neighbors:
                continue

            # y_min = argmin over V_open neighbours of cost-to-arrive + edge length
            # (Alg. 1 lines 4-5). Costs of V_open nodes are already final (no rewiring).
            y_min = None
            c_min = float('inf')
            for y, dist_yx in x_open_neighbors:
                potential_cost = self.cost[y] + dist_yx
                if potential_cost < c_min:
                    c_min = potential_cost
                    y_min = y

            if y_min is None:
                continue

            # Lazy step (Alg. 1 line 6): check *only* the best edge. If it collides, skip
            # x entirely (no fallback to a worse neighbour) -- it may be reached later.
            if not self._collision_free_indices(y_min, x):
                continue

            self.parent[x] = y_min
            self.cost[x] = c_min
            H_new.append(x)

            p1 = self.samples[y_min]
            p2 = self.samples[x]
            edges.append(((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))))

        for x in H_new:
            self.V_unvisited.discard(x)
            self.V_open.add(x)
            heapq.heappush(self.open_heap, (self.cost[x], x))

        # In FMT*, z stays in the open wavefront while its neighbors are
        # processed, then moves to closed after the batch update. The goal is
        # connected like any other sample; termination happens when it is later
        # popped as the lowest-cost node in V_open (handled at the top).
        self.V_open.remove(z)
        self.V_closed.add(z)

        return StepResult(edge=edges[0] if len(edges) == 1 else None, edges=edges if edges else None)
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if self.goal_idx not in self.parent:
            return []
        
        path: List[Tuple[int, int]] = []
        current = self.goal_idx
        while current != -1:
            pos = self.samples[current]
            path.append((int(pos[0]), int(pos[1])))
            current = self.parent.get(current, -1)
        path.reverse()
        return path
    
    def get_status(self) -> str:
        status = "FOUND" if self.found_path else "searching"
        return (
            f"FMT*: open {len(self.V_open)}, closed {len(self.V_closed)}, "
            f"unvisited {len(self.V_unvisited)}, cc {self.collision_checks}, {status}"
        )
    
    