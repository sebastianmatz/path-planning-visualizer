from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..geometry import (
    dist,
    line_collision_free,
)
from ..types import Edge, Point
from ._spatial import GridIndex
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
        # Uniform-grid index over roadmap nodes, built once learning finishes, so the
        # connecting phase finds each node's ``N_c`` candidates (neighbours within
        # ``max_edge_dist``) in O(neighbours) instead of scanning all nodes (O(n^2)).
        # Query nodes are not indexed (only 2 of them, and re-query strips/re-adds them).
        self._grid: Optional[GridIndex] = None
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
        # Snapshot of the roadmap's union-find forest taken once learning finishes,
        # so a re-query (drag start/goal) can restore it after stripping query nodes.
        self._roadmap_uf_snapshot: List[int] = []

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

    def _candidate_neighbors_grid(self, node_idx: int) -> List[Tuple[float, int]]:
        """Same ``N_c`` candidate set as ``_candidate_neighbors`` for a roadmap node,
        but found via the grid index instead of scanning every node.

        Roadmap-node coordinates are integer pixels and ``max_edge_dist`` is an
        integer-valued float, so ``GridIndex.within`` (``dx^2+dy^2 <= r^2``) selects
        exactly the nodes with ``dist <= max_edge_dist``. ``within`` returns ascending
        indices, so the stable distance-sort breaks ties by lowest index — identical to
        scanning the ascending ``candidate_indices``. ``max_index`` reproduces the
        incremental (Kavraki forest) restriction to earlier nodes; ``None`` is sPRM's
        all-pairs set.
        """
        point = self.nodes[node_idx]
        max_index = node_idx if self.remove_cycles else None
        distances: List[Tuple[float, int]] = []
        for idx in self._grid.within(point[0], point[1], self.max_edge_dist):
            if idx == node_idx:
                continue
            if max_index is not None and idx >= max_index:
                continue
            distances.append((dist(point, self.nodes[idx]), idx))
        distances.sort(key=lambda item: item[0])
        return distances[:self.k_neighbors]

    def _connect_node(self, node_idx: int, neighbors: List[Tuple[float, int]]) -> List[Edge]:
        node = self.nodes[node_idx]
        added_edges: List[Edge] = []
        for edge_dist, neighbor_idx in neighbors:
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
            # Index the finished roadmap so the connecting phase is O(n), not O(n^2).
            self._grid = GridIndex(self.max_edge_dist)
            for i in range(self.roadmap_size):
                self._grid.add(self.nodes[i][0], self.nodes[i][1])
            self.phase = "connecting"
            return StepResult()

        x, y = self.sample_pool[self.sample_idx]
        self._add_node((x, y))
        self.sample_idx += 1
        return StepResult(node_marker=(x, y))  # roadmap milestone, drawn as a dot
    
    def _connecting_step(self) -> StepResult:
        """Connect a roadmap node to nearby neighbours.

        PRM (Kavraki et al. 1996) builds the roadmap *incrementally*: each node is
        connected only to nodes already added (earlier indices), skipping pairs already
        in the same connected component -> a cycle-free forest grown organically. sPRM
        (Karaman & Frazzoli 2011) is *batch*: connect each node to all neighbours within
        range, keeping cycles. Restricting the forest variant to earlier nodes both
        matches Kavraki's construction and avoids the first-processed node grabbing all
        k neighbours at once (the mega-hub artifact of an all-pairs pass).
        """
        if self.connect_idx >= self.roadmap_size:
            self._roadmap_uf_snapshot = list(self._uf_parent)  # forest state for re-queries
            self.phase = "query_start"
            return StepResult()

        edges = self._connect_node(self.connect_idx, self._candidate_neighbors_grid(self.connect_idx))
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
        # to earlier query nodes as well (e.g. direct start-goal visibility). Only two
        # query nodes exist (and re-query strips/re-adds them), so a full scan is cheap
        # and avoids indexing nodes the grid would never be able to remove.
        edges = self._connect_node(node_idx, self._candidate_neighbors(query_point, list(range(node_idx))))

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

    def can_requery(self) -> bool:
        """True once the roadmap is learned, so new start/goal can reuse it."""
        return self.roadmap_size > 0 and bool(self._roadmap_uf_snapshot)

    def graph_edges(self) -> List[Edge]:
        """Current roadmap + query-connection edges as deduplicated point pairs."""
        return self._edges_as_points(include_query=True)

    def roadmap_edges(self) -> List[Edge]:
        """The learned roadmap edges only — excludes the transient start/goal query
        connections, so the displayed roadmap stays fixed across re-queries (only the
        path changes) rather than appearing to sprout new edges at each drop point."""
        return self._edges_as_points(include_query=False)

    def _edges_as_points(self, include_query: bool) -> List[Edge]:
        seen: Set[Tuple[int, int]] = set()
        out: List[Edge] = []
        for i, neighbours in self.edges.items():
            if not include_query and i >= self.roadmap_size:
                continue
            for j, _ in neighbours:
                if not include_query and j >= self.roadmap_size:
                    continue
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    continue
                seen.add(key)
                out.append((self.nodes[i], self.nodes[j]))
        return out

    def requery(self, new_start: Point, new_goal: Point) -> bool:
        """Re-solve for new start/goal on the already-learned roadmap (no re-sampling).

        Reuses ``nodes[:roadmap_size]`` and their edges; only the start/goal attachment
        and the graph search are redone — this is what makes a roadmap method's
        learned structure worth keeping. Returns True if a path was found.
        """
        if not self.can_requery():
            return False

        # Drop the previous query nodes (always trailing, index >= roadmap_size) and
        # any roadmap back-edges pointing at them; restore the forest's union-find.
        del self.nodes[self.roadmap_size:]
        for i in list(self.edges.keys()):
            if i >= self.roadmap_size:
                del self.edges[i]
            else:
                self.edges[i] = [(n, d) for (n, d) in self.edges[i] if n < self.roadmap_size]
        self._uf_parent = list(self._roadmap_uf_snapshot)

        self.start = new_start
        self.goal = new_goal
        self.current_path = []
        self.found_path = False
        self.start_idx = self._add_node(new_start)
        self._connect_node(self.start_idx, self._candidate_neighbors(new_start, list(range(self.roadmap_size))))
        self.goal_idx = self._add_node(new_goal)
        self._connect_node(self.goal_idx, self._candidate_neighbors(new_goal, list(range(self.goal_idx))))  # roadmap + start

        # A* over the roadmap graph (synchronous; fast relative to learning).
        open_heap: List[Tuple[float, float, int]] = [(self._heuristic(self.start_idx), 0.0, self.start_idx)]
        g_score: Dict[int, float] = {self.start_idx: 0.0}
        parent: Dict[int, int] = {}
        closed: Set[int] = set()
        while open_heap:
            _, g_cost, current = heapq.heappop(open_heap)
            if current == self.goal_idx:
                path = [current]
                while current in parent:
                    current = parent[current]
                    path.append(current)
                path.reverse()
                self.current_path = path
                self.found_path = True
                return True
            if current in closed:
                continue
            closed.add(current)
            for neighbor, cost in self.edges[current]:
                if neighbor in closed:
                    continue
                tentative = g_cost + cost
                if tentative < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative
                    parent[neighbor] = current
                    heapq.heappush(open_heap, (tentative + self._heuristic(neighbor), tentative, neighbor))
        return False

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

