from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from ..types import Edge
from .base import BasePlanner, StepResult


class DijkstraPlanner(BasePlanner):
    """Dijkstra's algorithm - shortest path on an occupancy-grid graph.

    Faithful adaptation of Dijkstra (1959, Numerische Mathematik 1; cite ``Dij59``),
    *Problem 2* (the minimum-length path between two nodes P and Q).

    Paper correspondence (Problem 2):

    - **Set A** ("nodes whose minimal path from P is known") = ``closed_set``, finalized
      in order of increasing distance from the start.
    - **Frontier set B** ("connected to A, one tentative branch each") = the heap plus
      the ``dist`` map (tentative cost-to-come per node).
    - **Step 2** ("move the B node of minimum distance from P into A") = the
      ``heappop`` of the lowest-distance entry -- ``step_once``.
    - **Step 1** (relaxation: "for each branch from the new A node, replace the tentative
      branch iff it is shorter") = the neighbour loop ``new = current_dist + cost``;
      update ``came_from`` / ``dist`` and push when shorter -- ``step_once``.
    - **Termination:** the search stops when Q (the goal cell) is moved to A (popped).
    - Stale heap entries (a node already in A) are skipped on pop -- the modern
      lazy-deletion equivalent of the paper's "one branch per B node, replaced when a
      shorter one is found".

    Adaptations (stated for fidelity):

    - **Induced coarse grid:** search runs on a ``grid_size``-downsampled grid (a
      supercell is an obstacle if it contains *any* obstacle pixel; the clicked
      start/goal cells are forced free). The result is optimal *on this induced grid*,
      not on the continuous pixel plane.
    - 8- or 4-connected graph with Euclidean branch lengths
      ``grid_size * sqrt(dx^2 + dy^2)``; corner-cutting through an occupied orthogonal
      neighbour is forbidden. Dijkstra is exact on this graph.
    - The returned path stitches the actual clicked start/goal pixels onto the
      cell-center polyline.

    See ``literature/fidelity/dijkstra.md`` and ``tests/test_dijkstra_fidelity.py``.
    """

    name = "Dijkstra"
    description = "Grid-based uniform-cost search"
    
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
                y1, y2 = gy * grid_size, min((gy + 1) * grid_size, self.h)
                x1, x2 = gx * grid_size, min((gx + 1) * grid_size, self.w)
                if np.any(self.occ[y1:y2, x1:x2] > 0):
                    self.grid_occ[gy, gx] = True

        # The clicked start/goal pixels are known free, so never let the
        # conservative supercell occupancy make their cells unreachable.
        self.grid_occ[self.start_grid[1], self.start_grid[0]] = False
        self.grid_occ[self.goal_grid[1], self.goal_grid[0]] = False

        # Dijkstra data structures
        import heapq
        self.open_set: List[Tuple[float, Tuple[int, int]]] = []
        self.came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.dist: Dict[Tuple[int, int], float] = {self.start_grid: 0}
        heapq.heappush(self.open_set, (0, self.start_grid))
        self.closed_set: Set[Tuple[int, int]] = set()
        self.path_grid: List[Tuple[int, int]] = []
        
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """The graph branches out of ``pos``: (neighbour, branch length) pairs.

        4- or 8-connected on the induced grid. Branch lengths are Euclidean and scaled
        by ``grid_size`` (``grid_size * sqrt(dx^2 + dy^2)``, so a diagonal costs sqrt(2)x
        a straight step). A diagonal is dropped if *either* orthogonal cell it passes
        between is occupied -- forbidding corner-cutting through an obstacle, which would
        otherwise let the path squeeze through a diagonal gap a point robot cannot fit.
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
        
        # Step 2: extract the frontier (B) node of minimum distance from P. It is now
        # finalized -- its minimal path is known -- so it joins set A below.
        current_dist, current = heapq.heappop(self.open_set)

        # Termination: the goal Q has just been moved to A, so its distance is final;
        # reconstruct the path by walking came_from back to the start.
        if current == self.goal_grid:
            self.path_grid = [current]
            while current in self.came_from:
                current = self.came_from[current]
                self.path_grid.append(current)
            self.path_grid.reverse()
            self.found_path = True
            self.done = True
            return StepResult(done=True, found_path=True)

        # Lazy deletion: a node already in A was reached again by a stale (longer) heap
        # entry; skip it. (The paper keeps a single branch per B node; the heap instead
        # keeps duplicates and discards the obsolete ones here.)
        if current in self.closed_set:
            return StepResult()

        self.closed_set.add(current)  # move `current` into set A

        # Step 1: relax every branch out of the node just added to A. If a branch gives a
        # neighbour a shorter path from P, replace its tentative branch (came_from/dist).
        edges: List[Edge] = []
        for neighbor, cost in self._get_neighbors(current):
            if neighbor in self.closed_set:
                continue

            new_dist = current_dist + cost

            if new_dist < self.dist.get(neighbor, float('inf')):
                self.came_from[neighbor] = current
                self.dist[neighbor] = new_dist
                heapq.heappush(self.open_set, (new_dist, neighbor))

                # Highlight every neighbor relaxed this expansion as the active
                # frontier (the tree is drawn from came_from; these fade).
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

        Redrawn whole from ``came_from`` each step (rather than accumulating one
        relaxation edge per pop), so the displayed search tree faithfully reflects the
        current best-known parent of every reached node.
        """
        gs, half = self.grid_size, self.grid_size // 2

        def center(c: Tuple[int, int]) -> Tuple[int, int]:
            return (c[0] * gs + half, c[1] * gs + half)

        return [(center(par), center(node)) for node, par in self.came_from.items()]


    def get_status(self) -> str:
        return f"Dijkstra: explored {len(self.closed_set)}, open {len(self.open_set)}"
    
    