from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from ..types import Edge
from .base import BasePlanner, StepResult


class AStarPlanner(BasePlanner):
    """A* - heuristic shortest-path search on an occupancy-grid graph.

    Faithful adaptation of Hart, Nilsson & Raphael (1968, IEEE Trans. SSC 4(2);
    cite ``HNR68``), Algorithm A* (sec. II-A). A* is Dijkstra guided by a heuristic:
    it expands the open node with the smallest ``f = g + h`` (estimated *total* path
    cost) rather than the smallest cost-so-far, so the search is pulled toward the goal.

    Paper correspondence (Algorithm A*, sec. II-A):

    - **Evaluation** ``f(n) = g(n) + h(n)`` (Eq. 2): ``g`` = best cost-to-come found so
      far, ``h`` = estimated cost-to-go -- ``g_score`` / ``_heuristic`` / ``f_score``.
    - **Step 2** (select the open node of smallest ``f``) = the ``heappop`` -- ``step_once``.
    - **Step 3** (terminate when a goal node is selected) -- ``step_once``.
    - **Step 4** (close ``n``, expand it, relax each successor with ``f = g + h``) --
      ``step_once``.
    - **No reopening:** a closed node is never revisited. This is correct *because* the
      heuristic is **consistent** (Eq. 5, the triangle inequality
      ``h(m) <= c(m, n) + h(n)``): under consistency every node is closed at its optimal
      cost, so the paper's reopening provision (Lemma 2) is vacuous here.
    - **Admissibility (Theorem 1):** ``h(n) <=`` true cost-to-go guarantees the returned
      path is optimal -- so A* yields the same optimal cost as Dijkstra on the same graph
      (``tests/test_planners.py::test_astar_matches_dijkstra_length``).

    Adaptations (stated for fidelity) -- the same induced-grid model as ``DijkstraPlanner``:

    - **Induced coarse grid** (``grid_size`` downsample; a supercell is an obstacle iff it
      contains any obstacle pixel; clicked start/goal cells forced free). Optimal *on this
      induced grid*, not the continuous pixel plane.
    - 8- or 4-connected with Euclidean / Manhattan branch weights; no corner-cutting
      through an occupied orthogonal neighbour; the path is stitched to the real
      start/goal pixels.

    See ``literature/fidelity/astar.md`` and ``tests/test_astar_fidelity.py``.
    """

    name = "A*"
    description = "Grid-based heuristic search"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 grid_size: int = 5, allow_diagonal: bool = True):
        super().__init__(occ, start, goal)
        
        self.grid_size = grid_size
        self.allow_diagonal = allow_diagonal
        
        # Convert to grid coordinates
        self.grid_w = int(np.ceil(self.w / grid_size))
        self.grid_h = int(np.ceil(self.h / grid_size))
        self.start_grid = (
            min(start[0] // grid_size, self.grid_w - 1),
            min(start[1] // grid_size, self.grid_h - 1),
        )
        self.goal_grid = (
            min(goal[0] // grid_size, self.grid_w - 1),
            min(goal[1] // grid_size, self.grid_h - 1),
        )
        
        # Pre-compute grid occupancy
        self.grid_occ = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                # Check if any cell in this grid block is occupied
                y1, y2 = gy * grid_size, min((gy + 1) * grid_size, self.h)
                x1, x2 = gx * grid_size, min((gx + 1) * grid_size, self.w)
                if np.any(self.occ[y1:y2, x1:x2] > 0):
                    self.grid_occ[gy, gx] = True

        # The clicked start/goal pixels are known free, so never let the
        # conservative supercell occupancy make their cells unreachable.
        self.grid_occ[self.start_grid[1], self.start_grid[0]] = False
        self.grid_occ[self.goal_grid[1], self.goal_grid[0]] = False

        # A* data structures
        import heapq
        self.open_set: List[Tuple[float, Tuple[int, int]]] = []
        self.came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.g_score: Dict[Tuple[int, int], float] = {self.start_grid: 0}
        self.f_score: Dict[Tuple[int, int], float] = {self.start_grid: self._heuristic(self.start_grid)}
        heapq.heappush(self.open_set, (self.f_score[self.start_grid], self.start_grid))
        self.closed_set: Set[Tuple[int, int]] = set()
        self.path_grid: List[Tuple[int, int]] = []
        
    def _heuristic(self, pos: Tuple[int, int]) -> float:
        """Cost-to-go estimate ``h`` (HNR68 Eq. 2): Euclidean (8-conn) or Manhattan (4-conn).

        For its connectivity each form is both *admissible* (never overestimates the true
        grid cost, Theorem 1) and *consistent* (satisfies the triangle inequality, Eq. 5).
        That is exactly what makes the no-reopening policy in ``step_once`` correct and the
        returned path optimal.
        """
        dx = abs(pos[0] - self.goal_grid[0])
        dy = abs(pos[1] - self.goal_grid[1])
        if self.allow_diagonal:
            return self.grid_size * np.sqrt(dx*dx + dy*dy)
        return self.grid_size * (dx + dy)

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """The graph branches out of ``pos``: (neighbour, branch length) pairs.

        4- or 8-connected on the induced grid, with Euclidean branch lengths scaled by
        ``grid_size``. A diagonal is dropped if either orthogonal cell it passes between is
        occupied -- forbidding corner-cutting through an obstacle. This is the same graph
        as ``DijkstraPlanner``; A* differs only in the ``f = g + h`` selection order.
        """
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if self.allow_diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                if self.grid_occ[ny, nx]:
                    continue

                # Block diagonal corner cutting through occupied orthogonal neighbors.
                if dx != 0 and dy != 0:
                    side_a = (pos[0] + dx, pos[1])
                    side_b = (pos[0], pos[1] + dy)
                    if self.grid_occ[side_a[1], side_a[0]] or self.grid_occ[side_b[1], side_b[0]]:
                        continue

                cost = self.grid_size * np.sqrt(dx*dx + dy*dy)
                neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def step_once(self) -> StepResult:
        import heapq
        
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        
        if not self.open_set:
            self.done = True
            return StepResult(done=True, found_path=False)
        
        # Step 2: select the open node with the smallest f = g + h.
        _, current = heapq.heappop(self.open_set)

        # Step 3: a goal node was selected -> its f is optimal; reconstruct the path.
        if current == self.goal_grid:
            self.path_grid = [current]
            while current in self.came_from:
                current = self.came_from[current]
                self.path_grid.append(current)
            self.path_grid.reverse()
            self.found_path = True
            self.done = True
            return StepResult(done=True, found_path=True)

        # Lazy deletion: a node already closed was reached again via a stale (larger-f)
        # heap entry; skip it.
        if current in self.closed_set:
            return StepResult()

        self.closed_set.add(current)  # Step 4: close n

        # Step 4 (cont.): expand n and relax each successor. Closed successors are not
        # reopened -- valid because the heuristic is consistent (Eq. 5 / Lemma 2).
        edges: List[Edge] = []
        for neighbor, cost in self._get_neighbors(current):
            if neighbor in self.closed_set:
                continue

            tentative_g = self.g_score.get(current, float('inf')) + cost

            # Relax: if this path to `neighbor` is cheaper, record it and recompute f = g + h.
            if tentative_g < self.g_score.get(neighbor, float('inf')):
                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g
                self.f_score[neighbor] = tentative_g + self._heuristic(neighbor)
                heapq.heappush(self.open_set, (self.f_score[neighbor], neighbor))

                # Every neighbor relaxed this expansion is a "new bit" of the search,
                # highlighted as the active frontier (the tree itself is drawn from
                # came_from; these are for the fading highlight only).
                p1 = (current[0] * self.grid_size + self.grid_size // 2,
                      current[1] * self.grid_size + self.grid_size // 2)
                p2 = (neighbor[0] * self.grid_size + self.grid_size // 2,
                      neighbor[1] * self.grid_size + self.grid_size // 2)
                edges.append((p1, p2))

        return StepResult(edges=edges or None)
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if not self.path_grid:
            return []
        centers = [(gx * self.grid_size + self.grid_size // 2,
                    gy * self.grid_size + self.grid_size // 2) for gx, gy in self.path_grid]
        # Stitch the actual clicked start/goal pixels onto the cell-center
        # polyline so the path connects to the markers and metrics use the
        # real endpoints rather than the quantized cell centers.
        stitched = [self.start] + centers + [self.goal]
        deduped: List[Tuple[int, int]] = []
        for p in stitched:
            if not deduped or deduped[-1] != p:
                deduped.append(p)
        return deduped

    def extract_tree_edges(self) -> List[Edge]:
        """Current search tree (``came_from``) as parent->node edges in pixel coords.

        A* relaxes nodes that are still in the open set, so a node's parent can change
        before it is expanded. Redrawing the tree from ``came_from`` each step (rather
        than accumulating one relaxation edge per pop) keeps the displayed search tree
        faithful: it shows the actual current best-known tree with no superseded edges.
        """
        gs, half = self.grid_size, self.grid_size // 2

        def center(c: Tuple[int, int]) -> Tuple[int, int]:
            return (c[0] * gs + half, c[1] * gs + half)

        return [(center(par), center(node)) for node, par in self.came_from.items()]

    def get_status(self) -> str:
        return f"A*: explored {len(self.closed_set)}, open {len(self.open_set)}"
    
    