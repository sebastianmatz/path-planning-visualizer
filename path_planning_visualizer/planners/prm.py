from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..geometry import (
    dist,
    line_collision_free,
)
from ..types import Edge, Point
from .base import BasePlanner, StepResult


class PRMPlanner(BasePlanner):
    """sPRM - simplified Probabilistic Roadmap (Karaman & Frazzoli 2011).

    Two phases:
    1. Learning phase: build a roadmap of collision-free random samples, each
       connected to its neighbours within ``max_edge_dist`` (capped at
       ``k_neighbors``) via a straight-line local planner.
    2. Query phase: connect start/goal to the roadmap and search the graph.

    ``remove_cycles=True`` reproduces the original Kavraki et al. (1996)
    construction step, which skips edges between nodes already in the same
    connected component (a cycle-free forest). The default ``False`` keeps cycles,
    i.e. the asymptotically-optimal sPRM (see ``ClassicPRMPlanner`` for the forest).
    """

    name = "sPRM"
    description = "Simplified probabilistic roadmap (asymptotically optimal; keeps cycles)"

    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                  num_samples: int = 500, k_neighbors: int = 15, max_edge_dist: int = 100,
                  seed: int = 42, remove_cycles: bool = False):
        super().__init__(occ, start, goal)

        self.num_samples = int(max(1, num_samples))
        self.k_neighbors = int(max(1, k_neighbors))
        self.max_edge_dist = float(max(1.0, max_edge_dist))
        self.remove_cycles = bool(remove_cycles)
        # Union-find over node indices, used only when remove_cycles is set to
        # build the original Kavraki (1996) cycle-free forest.
        self._uf_parent: List[int] = []
        self.rng = np.random.default_rng(seed)

        # Roadmap is learned independently of the query.
        self.nodes: List[Point] = []
        self.edges: Dict[int, List[Tuple[int, float]]] = {}
        self.roadmap_size = 0
        self.sample_pool: List[Point] = self._build_sample_pool()

        # Query nodes are added only after roadmap learning.
        self.start_idx: Optional[int] = None
        self.goal_idx: Optional[int] = None

        # Phase tracking
        self.phase = "sampling"  # sampling -> connecting -> query_start -> query_goal -> searching -> done
        self.sample_idx = 0
        self.connect_idx = 0
        self.search_open: List[Tuple[float, float, int]] = []  # (f_cost, g_cost, node)
        self.search_closed: Set[int] = set()
        self.search_parent: Dict[int, int] = {}
        self.search_g: Dict[int, float] = {}
        self.current_path: List[int] = []

    def _build_sample_pool(self) -> List[Point]:
        """Pre-sample unique free roadmap nodes, excluding query configurations.

        Operates on the ``np.where`` coordinate arrays directly instead of
        materializing a Python tuple per free pixel, while drawing the exact same
        samples for a given seed (the row-major free-pixel order is preserved).
        """
        free_y, free_x = np.where(~self.occ)
        xs = free_x.astype(np.int64, copy=False)
        ys = free_y.astype(np.int64, copy=False)

        # Exclude the start/goal configurations from the roadmap pool.
        keep = np.ones(xs.shape[0], dtype=bool)
        for px, py in (self.start, self.goal):
            keep &= ~((xs == px) & (ys == py))
        xs = xs[keep]
        ys = ys[keep]

        n_free = xs.shape[0]
        if n_free == 0:
            return []

        sample_count = min(self.num_samples, n_free)
        chosen = self.rng.choice(n_free, size=sample_count, replace=False)
        return [(int(xs[i]), int(ys[i])) for i in chosen]

    def _add_node(self, point: Point) -> int:
        node_idx = len(self.nodes)
        self.nodes.append(point)
        self.edges[node_idx] = []
        self._uf_parent.append(node_idx)  # make-set (used only when remove_cycles)
        return node_idx

    def _edge_exists(self, a: int, b: int) -> bool:
        return any(neighbor == b for neighbor, _ in self.edges[a])

    def _uf_find(self, i: int) -> int:
        while self._uf_parent[i] != i:
            self._uf_parent[i] = self._uf_parent[self._uf_parent[i]]  # path compression
            i = self._uf_parent[i]
        return i

    def _uf_union(self, a: int, b: int) -> None:
        ra, rb = self._uf_find(a), self._uf_find(b)
        if ra != rb:
            self._uf_parent[rb] = ra

    def _candidate_neighbors(self, point: Point, candidate_indices: List[int]) -> List[Tuple[float, int]]:
        distances: List[Tuple[float, int]] = []
        for idx in candidate_indices:
            other = self.nodes[idx]
            d = dist(point, other)
            if d <= self.max_edge_dist:
                distances.append((d, idx))
        distances.sort(key=lambda item: item[0])
        return distances[:self.k_neighbors]

    def _connect_node(self, node_idx: int, candidate_indices: List[int]) -> List[Edge]:
        node = self.nodes[node_idx]
        added_edges: List[Edge] = []
        for edge_dist, neighbor_idx in self._candidate_neighbors(node, candidate_indices):
            if neighbor_idx == node_idx or self._edge_exists(node_idx, neighbor_idx):
                continue
            # Kavraki (1996) construction step: skip nodes already in the same
            # connected component (yields a cycle-free forest). sPRM keeps them.
            if self.remove_cycles and self._uf_find(node_idx) == self._uf_find(neighbor_idx):
                continue
            if not line_collision_free(node, self.nodes[neighbor_idx], self.occ):
                continue
            self.edges[node_idx].append((neighbor_idx, edge_dist))
            self.edges[neighbor_idx].append((node_idx, edge_dist))
            if self.remove_cycles:
                self._uf_union(node_idx, neighbor_idx)
            added_edges.append((node, self.nodes[neighbor_idx]))
        return added_edges
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        # If already found path, just return done
        if self.found_path:
            self.done = True
            return StepResult(done=True, found_path=True)
        
        self.iteration += 1
        
        if self.phase == "sampling":
            return self._sampling_step()
        elif self.phase == "connecting":
            return self._connecting_step()
        elif self.phase == "query_start":
            return self._query_connect_step(self.start)
        elif self.phase == "query_goal":
            return self._query_connect_step(self.goal)
        elif self.phase == "searching":
            return self._searching_step()
        
        return StepResult(done=True)
    
    def _sampling_step(self) -> StepResult:
        """Sample random points in free space."""
        if self.sample_idx >= len(self.sample_pool):
            self.roadmap_size = len(self.nodes)
            self.phase = "connecting"
            return StepResult()

        x, y = self.sample_pool[self.sample_idx]
        self._add_node((x, y))
        self.sample_idx += 1
        return StepResult(edge=((x - 2, y - 2), (x + 2, y + 2)))  # Small marker
    
    def _connecting_step(self) -> StepResult:
        """Connect roadmap nodes to nearby roadmap neighbors."""
        if self.connect_idx >= self.roadmap_size:
            self.phase = "query_start"
            return StepResult()

        candidate_indices = [idx for idx in range(self.roadmap_size) if idx != self.connect_idx]
        edges = self._connect_node(self.connect_idx, candidate_indices)
        self.connect_idx += 1
        return StepResult(edge=edges[0] if len(edges) == 1 else None, edges=edges if len(edges) > 1 else None)

    def _query_connect_step(self, query_point: Point) -> StepResult:
        """Attach a query configuration to the learned roadmap."""
        if self.roadmap_size == 0:
            self.phase = "done"
            self.done = True
            return StepResult(done=True, found_path=False)

        node_idx = self._add_node(query_point)
        # Query nodes connect to the previously built roadmap and, once available,
        # to earlier query nodes as well (e.g. direct start-goal visibility).
        edges = self._connect_node(node_idx, list(range(node_idx)))

        if query_point == self.start:
            self.start_idx = node_idx
            self.phase = "query_goal"
        else:
            self.goal_idx = node_idx
            if not edges or self.start_idx is None:
                self.phase = "done"
                self.done = True
                return StepResult(done=True, found_path=False)
            self.phase = "searching"
            self.search_open = []
            self.search_closed = set()
            self.search_parent = {}
            self.search_g = {self.start_idx: 0.0}
            heapq.heappush(self.search_open, (self._heuristic(self.start_idx), 0.0, self.start_idx))

        return StepResult(edge=edges[0] if len(edges) == 1 else None, edges=edges if len(edges) > 1 else None)
    
    def _searching_step(self) -> StepResult:
        """A* search on the roadmap."""
        if self.start_idx is None or self.goal_idx is None:
            self.phase = "done"
            self.done = True
            return StepResult(done=True, found_path=False)

        if not self.search_open:
            self.phase = "done"
            self.done = True
            return StepResult(done=True, found_path=False)
        
        _, current_g, current = heapq.heappop(self.search_open)

        if current == self.goal_idx:
            self.current_path = [current]
            while current in self.search_parent:
                current = self.search_parent[current]
                self.current_path.append(current)
            self.current_path.reverse()
            self.found_path = True
            self.phase = "done"
            self.done = True
            return StepResult(done=True, found_path=True)
        
        if current in self.search_closed:
            return StepResult()
        
        self.search_closed.add(current)
        
        edge = None
        for neighbor, cost in self.edges[current]:
            if neighbor in self.search_closed:
                continue

            tentative_g = current_g + cost
            if tentative_g < self.search_g.get(neighbor, float('inf')):
                self.search_parent[neighbor] = current
                self.search_g[neighbor] = tentative_g
                h_cost = self._heuristic(neighbor)
                heapq.heappush(self.search_open, (tentative_g + h_cost, tentative_g, neighbor))
                edge = (self.nodes[current], self.nodes[neighbor])
        
        return StepResult(edge=edge)
    
    def _heuristic(self, node_idx: int) -> float:
        """Euclidean distance to goal."""
        node = self.nodes[node_idx]
        return np.sqrt((node[0] - self.goal[0])**2 + (node[1] - self.goal[1])**2)
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if not self.current_path:
            return []
        return [self.nodes[i] for i in self.current_path]
    
    def get_status(self) -> str:
        total_edges = sum(len(e) for e in self.edges.values()) // 2
        return (
            f"PRM: {self.phase}, roadmap nodes: {self.roadmap_size}, "
            f"total nodes: {len(self.nodes)}, edges: {total_edges}"
        )
    
    
class ClassicPRMPlanner(PRMPlanner):
    """PRM - the original Kavraki et al. (1996) construction step.

    Identical to ``sPRM`` (``PRMPlanner``) except that it skips connections
    between nodes already in the same connected component, producing the paper's
    cycle-free forest roadmap. This is *not* asymptotically optimal and tends to
    return longer query paths than ``sPRM`` (a difference the original paper notes
    and addresses with smoothing).
    """

    name = "PRM"
    description = "Original Kavraki et al. (1996) construction: a cycle-free forest roadmap"

    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 **kwargs) -> None:
        kwargs.pop('remove_cycles', None)
        super().__init__(occ, start, goal, remove_cycles=True, **kwargs)

