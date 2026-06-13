from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)

from .base import BasePlanner, StepResult


class DijkstraParamsWidget(QWidget):
    """Parameters widget for Dijkstra planner."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_grid_size = QSpinBox()
        self.spin_grid_size.setRange(1, 20)
        self.spin_grid_size.setValue(5)
        self.spin_grid_size.setToolTip("Grid cell size")
        
        self.check_diagonal = QCheckBox()
        self.check_diagonal.setChecked(True)
        self.check_diagonal.setToolTip("Allow diagonal movement")
        
        layout.addRow("Grid size:", self.spin_grid_size)
        layout.addRow("Diagonal:", self.check_diagonal)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'grid_size': self.spin_grid_size.value(),
            'allow_diagonal': self.check_diagonal.isChecked(),
        }


class DijkstraPlanner(BasePlanner):
    """Dijkstra's Algorithm - Grid-based uniform-cost pathfinding."""
    
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
        
        current_dist, current = heapq.heappop(self.open_set)
        
        if current == self.goal_grid:
            self.path_grid = [current]
            while current in self.came_from:
                current = self.came_from[current]
                self.path_grid.append(current)
            self.path_grid.reverse()
            self.found_path = True
            self.done = True
            return StepResult(done=True, found_path=True)
        
        if current in self.closed_set:
            return StepResult()
        
        self.closed_set.add(current)
        
        edge = None
        for neighbor, cost in self._get_neighbors(current):
            if neighbor in self.closed_set:
                continue
            
            new_dist = current_dist + cost
            
            if new_dist < self.dist.get(neighbor, float('inf')):
                self.came_from[neighbor] = current
                self.dist[neighbor] = new_dist
                heapq.heappush(self.open_set, (new_dist, neighbor))
                
                p1 = (current[0] * self.grid_size + self.grid_size // 2,
                      current[1] * self.grid_size + self.grid_size // 2)
                p2 = (neighbor[0] * self.grid_size + self.grid_size // 2,
                      neighbor[1] * self.grid_size + self.grid_size // 2)
                edge = (p1, p2)
        
        return StepResult(edge=edge)
    
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
    
    def get_status(self) -> str:
        return f"Dijkstra: explored {len(self.closed_set)}, open {len(self.open_set)}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return DijkstraParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'DijkstraPlanner':
        params = params_widget.get_params()
        return DijkstraPlanner(occ, start, goal, **params)
