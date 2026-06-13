from __future__ import annotations

from typing import List, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)

from ..geometry import (
    clamp_point,
    dist,
    line_collision_free,
    steer,
)
from .base import BasePlanner, StepResult
from ._spatial import GridIndex
from ._rgg import rgg_radius


class RRTStarParamsWidget(QWidget):
    """Widget for RRT* parameter configuration."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 200)
        self.spin_step.setValue(18)
        self.spin_step.setToolTip("Distance for each tree expansion step")
        
        self.spin_goal_rate = QDoubleSpinBox()
        self.spin_goal_rate.setRange(0.0, 1.0)
        self.spin_goal_rate.setSingleStep(0.01)
        self.spin_goal_rate.setValue(0.10)
        self.spin_goal_rate.setToolTip("Probability of sampling the goal directly")
        
        self.spin_goal_tol = QSpinBox()
        self.spin_goal_tol.setRange(1, 200)
        self.spin_goal_tol.setValue(20)
        self.spin_goal_tol.setToolTip("Distance threshold to consider goal reached")
        
        self.spin_search_radius = QSpinBox()
        self.spin_search_radius.setRange(0, 500)
        self.spin_search_radius.setValue(0)
        self.spin_search_radius.setToolTip(
            "0 = auto: shrinking RGG radius min(gamma*(log n/n)^(1/2), step), "
            "asymptotically optimal (Karaman & Frazzoli 2011). "
            ">0 = fixed radius (legacy)."
        )
        
        self.spin_col = QSpinBox()
        self.spin_col.setRange(10, 500)
        self.spin_col.setValue(80)
        self.spin_col.setToolTip("Number of samples for collision checking along edges")
        
        self.spin_maxit = QSpinBox()
        self.spin_maxit.setRange(100, 200000)
        self.spin_maxit.setValue(25000)
        self.spin_maxit.setToolTip("Maximum number of iterations")
        
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000_000)
        self.spin_seed.setValue(1)
        self.spin_seed.setToolTip("Random seed for reproducibility")
        
        layout.addRow("Step size:", self.spin_step)
        layout.addRow("Goal sample rate:", self.spin_goal_rate)
        layout.addRow("Goal tolerance:", self.spin_goal_tol)
        layout.addRow("Search radius (0=auto):", self.spin_search_radius)
        layout.addRow("Collision samples:", self.spin_col)
        layout.addRow("Max iterations:", self.spin_maxit)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'step_size': self.spin_step.value(),
            'goal_sample_rate': self.spin_goal_rate.value(),
            'goal_tolerance': self.spin_goal_tol.value(),
            'search_radius': self.spin_search_radius.value(),
            'collision_samples': self.spin_col.value(),
            'max_iters': self.spin_maxit.value(),
            'seed': self.spin_seed.value(),
        }


class RRTStarPlanner(BasePlanner):
    """RRT* - Rewiring-based RRT variant with incremental path improvement."""
    
    name = "RRT*"
    description = "Rewiring-based sampling planner with incremental path improvement"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 step_size: int = 18, goal_sample_rate: float = 0.10, goal_tolerance: int = 20,
                 search_radius: int = 0, collision_samples: int = 80,
                 max_iters: int = 25000, seed: int = 1):
        super().__init__(occ, start, goal)

        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
        self.search_radius = float(search_radius)
        # search_radius <= 0 -> use the shrinking RGG radius (paper-faithful);
        # otherwise use the fixed radius the user supplied (legacy behaviour).
        self.adaptive_radius = self.search_radius <= 0.0
        self.free_space_volume = float(np.count_nonzero(~self.occ))
        self.collision_samples = collision_samples
        self.max_iters = max_iters

        self.rng = np.random.default_rng(seed)
        self.nodes = [start]
        self.parent = [-1]
        self.children: List[set] = [set()]  # children[i] = nodes whose parent is i
        self.cost = [0.0]  # Cost from start to each node
        self.goal_idx = None
        self.best_goal_cost = float('inf')
        self._index = GridIndex(max(1.0, self.step_size))
        self._index.add(start[0], start[1])

    def _nearest(self, p: Tuple[int, int]) -> int:
        """Find index of nearest node to point p."""
        return self._index.nearest(p[0], p[1])

    def _near(self, p: Tuple[int, int], radius: float) -> List[int]:
        """Find all nodes within radius of point p."""
        return self._index.within(p[0], p[1], radius)

    def _add_node(self, point: Tuple[int, int], parent: int, cost: float) -> int:
        """Append a node, register it in the spatial index and child adjacency."""
        idx = len(self.nodes)
        self.nodes.append(point)
        self.parent.append(-1)
        self.children.append(set())
        self.cost.append(cost)
        self._set_parent(idx, parent)
        self._index.add(point[0], point[1])
        return idx

    def _set_parent(self, child: int, parent: int) -> None:
        """Reparent ``child``, keeping the children adjacency consistent."""
        old = self.parent[child]
        if old != -1:
            self.children[old].discard(child)
        self.parent[child] = parent
        if parent != -1:
            self.children[parent].add(child)

    def _connection_radius(self) -> float:
        """Shrinking RGG ball radius r_n = min(gamma*(log n/n)^(1/d), step_size).

        Karaman & Frazzoli (2011): with this radius RRT* is asymptotically
        optimal; the cap at the steering range eta = step_size matches OMPL.
        """
        return rgg_radius(len(self.nodes), self.free_space_volume,
                          plus_one=True, eta=self.step_size)
    
    def _sample(self) -> Tuple[int, int]:
        """Sample a random point, biased towards goal."""
        if self.rng.random() < self.goal_sample_rate:
            return self.goal
        for _ in range(100):
            p = (int(self.rng.integers(0, self.w)), int(self.rng.integers(0, self.h)))
            if self.is_free(p):
                return p
        return (int(self.rng.integers(0, self.w)), int(self.rng.integers(0, self.h)))
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=self.found_path)
        
        self.iteration += 1
        q_rand = self._sample()
        i_near = self._nearest(q_rand)
        q_near = self.nodes[i_near]
        q_new = clamp_point(steer(q_near, q_rand, self.step_size), self.w, self.h)
        
        # Check if new point is valid
        if not self.is_free(q_new):
            return StepResult(rejected_point=q_new)
        if not line_collision_free(q_near, q_new, self.occ, samples=self.collision_samples):
            return StepResult(rejected_point=q_new)
        
        # Find nearby nodes for potential better parent / rewiring
        radius = self._connection_radius() if self.adaptive_radius else self.search_radius
        near_indices = self._near(q_new, radius)
        
        # Choose best parent (lowest cost)
        best_parent = i_near
        best_cost = self.cost[i_near] + dist(q_near, q_new)
        
        for i in near_indices:
            potential_cost = self.cost[i] + dist(self.nodes[i], q_new)
            if potential_cost < best_cost:
                if line_collision_free(self.nodes[i], q_new, self.occ, samples=self.collision_samples):
                    best_parent = i
                    best_cost = potential_cost
        
        # Add new node with best parent
        new_idx = self._add_node(q_new, best_parent, best_cost)

        edge = (self.nodes[best_parent], q_new)
        
        # Rewire: check if nearby nodes can be improved through new node
        # NEVER rewire the goal node (goal_idx) - its parent is managed separately
        # Also prevent cycles by not rewiring ancestors of the new node
        ancestors = set()
        p = best_parent
        while p != -1:
            ancestors.add(p)
            p = self.parent[p]
        
        for i in near_indices:
            if i == best_parent:
                continue
            if self.goal_idx is not None and i == self.goal_idx:
                continue  # Don't rewire goal node
            if i in ancestors:
                continue  # Don't rewire ancestors (would create cycle)
            potential_cost = best_cost + dist(q_new, self.nodes[i])
            if potential_cost < self.cost[i]:
                if line_collision_free(q_new, self.nodes[i], self.occ, samples=self.collision_samples):
                    self._set_parent(i, new_idx)
                    self._update_costs(i, potential_cost)
        
        # Track if path was improved this step
        path_improved = False
        
        # Check if goal reached
        goal_dist = dist(q_new, self.goal)
        if goal_dist <= self.goal_tolerance:
            if line_collision_free(q_new, self.goal, self.occ, samples=self.collision_samples):
                goal_cost = best_cost + goal_dist
                if goal_cost < self.best_goal_cost:
                    # Check: new_idx must not have goal_idx as ancestor (would create cycle)
                    creates_cycle = False
                    if self.goal_idx is not None:
                        p = new_idx
                        while p != -1:
                            if p == self.goal_idx:
                                creates_cycle = True
                                break
                            p = self.parent[p]
                    
                    if not creates_cycle:
                        # Found better path to goal
                        if self.goal_idx is None:
                            # First time reaching goal - add goal node
                            self.goal_idx = self._add_node(self.goal, new_idx, goal_cost)
                        else:
                            # Update existing goal connection
                            self._set_parent(self.goal_idx, new_idx)
                            self.cost[self.goal_idx] = goal_cost
                        self.best_goal_cost = goal_cost
                        self.found_path = True
                        path_improved = True
        
        return StepResult(edge=edge, path_improved=path_improved)
    
    def _update_costs(self, idx: int, new_cost: float):
        """Propagate a cost change down the subtree rooted at ``idx``.

        Walks the children adjacency (O(affected subtree)) instead of scanning
        every parent pointer. When the goal's cost changes, the incumbent
        ``best_goal_cost`` is resynced so a later reconnection can never accept a
        path that is worse than the goal's current true cost.
        """
        stack = [(idx, new_cost)]
        while stack:
            current_idx, current_cost = stack.pop()
            self.cost[current_idx] = current_cost
            if current_idx == self.goal_idx:
                self.best_goal_cost = current_cost
            for child in self.children[current_idx]:
                child_cost = current_cost + dist(self.nodes[current_idx], self.nodes[child])
                stack.append((child, child_cost))
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if self.goal_idx is None:
            return []
        path = []
        i = self.goal_idx
        visited = set()  # Prevent infinite loop from bad parent pointers
        while i != -1 and i not in visited:
            visited.add(i)
            path.append(self.nodes[i])
            i = self.parent[i]
        path.reverse()
        return path
    
    def get_status(self) -> str:
        cost_str = f", path cost: {self.best_goal_cost:.1f}" if self.found_path else ""
        if self.adaptive_radius:
            radius_str = f", r {self._connection_radius():.1f} (auto)"
        else:
            radius_str = f", r {self.search_radius:.0f} (fixed)"
        return f"RRT*: iter {self.iteration}/{self.max_iters}, nodes {len(self.nodes)}{radius_str}{cost_str}"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return RRTStarParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'RRTStarPlanner':
        params = params_widget.get_params()
        return RRTStarPlanner(occ, start, goal, **params)
