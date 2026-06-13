from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from PyQt6.QtWidgets import (
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


class RRTConnectParamsWidget(QWidget):
    """Widget for RRT-Connect parameter configuration."""
    
    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 200)
        self.spin_step.setValue(18)
        self.spin_step.setToolTip("Distance for each tree expansion step")
        
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
        layout.addRow("Collision samples:", self.spin_col)
        layout.addRow("Max iterations:", self.spin_maxit)
        layout.addRow("Seed:", self.spin_seed)
        
        self.setLayout(layout)
    
    def get_params(self) -> dict:
        return {
            'step_size': self.spin_step.value(),
            'collision_samples': self.spin_col.value(),
            'max_iters': self.spin_maxit.value(),
            'seed': self.spin_seed.value(),
        }


class RRTConnectPlanner(BasePlanner):
    """RRT-Connect: Bidirectional RRT that grows trees from start and goal."""
    
    name = "RRT-Connect"
    description = "Bidirectional RRT - grows two trees and connects them"
    
    def __init__(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 step_size: int = 18, collision_samples: int = 80, 
                 max_iters: int = 25000, seed: int = 1):
        super().__init__(occ, start, goal)
        
        self.step_size = step_size
        self.collision_samples = collision_samples
        self.max_iters = max_iters
        
        self.rng = np.random.default_rng(seed)
        
        # Tree from start (tree_a)
        self.nodes_a = [start]
        self.parent_a = [-1]
        
        # Tree from goal (tree_b)
        self.nodes_b = [goal]
        self.parent_b = [-1]
        
        # Connection point indices (when trees connect)
        self.connect_idx_a = None
        self.connect_idx_b = None
        
        # Track which tree is currently extending (swap each iteration)
        self.swap_trees = False

        # One spatial index per tree (indices stay in lockstep with nodes_a/nodes_b).
        cell = max(1.0, float(self.step_size))
        self._index_a = GridIndex(cell)
        self._index_a.add(start[0], start[1])
        self._index_b = GridIndex(cell)
        self._index_b.add(goal[0], goal[1])

    def _sample(self) -> Tuple[int, int]:
        """Sample a random point in free space."""
        for _ in range(100):
            p = (int(self.rng.integers(0, self.w)), int(self.rng.integers(0, self.h)))
            if self.is_free(p):
                return p
        return (int(self.rng.integers(0, self.w)), int(self.rng.integers(0, self.h)))
    
    def _extend(self, nodes: List[Tuple[int, int]], parent: List[int], index: GridIndex,
                target: Tuple[int, int]) -> Tuple[str, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Extend tree towards target.
        Returns: (status, new_point, rejected_point)
        status: 'reached' if target reached, 'advanced' if extended, 'trapped' if blocked
        """
        i_near = index.nearest(target[0], target[1])
        q_near = nodes[i_near]
        q_new = clamp_point(steer(q_near, target, self.step_size), self.w, self.h)

        if not self.is_free(q_new):
            return ('trapped', None, q_new)
        if not line_collision_free(q_near, q_new, self.occ, samples=self.collision_samples):
            return ('trapped', None, q_new)

        nodes.append(q_new)
        parent.append(i_near)
        index.add(q_new[0], q_new[1])

        if dist(q_new, target) < self.step_size:
            return ('reached', q_new, None)
        return ('advanced', q_new, None)

    def _connect(self, nodes: List[Tuple[int, int]], parent: List[int], index: GridIndex,
                 target: Tuple[int, int]) -> Tuple[str, List[Tuple[Tuple[int,int], Tuple[int,int]]], Optional[Tuple[int, int]]]:
        """
        Try to connect tree to target point by repeatedly extending.
        Returns: (status, edges_added, last_rejected)
        """
        edges = []

        while True:
            i_near = index.nearest(target[0], target[1])
            q_near = nodes[i_near]

            # Final step: land exactly on the target so the two trees share a
            # vertex. The residual segment is collision-checked, so the joined
            # path has no unchecked gap (see RRT-Connect "CONNECT" semantics).
            if dist(q_near, target) <= self.step_size:
                if not line_collision_free(q_near, target, self.occ, samples=self.collision_samples):
                    return ('trapped', edges, target)
                nodes.append(target)
                parent.append(i_near)
                index.add(target[0], target[1])
                edges.append((q_near, target))
                return ('reached', edges, None)

            q_new = clamp_point(steer(q_near, target, self.step_size), self.w, self.h)

            if not self.is_free(q_new):
                return ('trapped', edges, q_new)
            if not line_collision_free(q_near, q_new, self.occ, samples=self.collision_samples):
                return ('trapped', edges, q_new)

            nodes.append(q_new)
            parent.append(i_near)
            index.add(q_new[0], q_new[1])
            edges.append((q_near, q_new))
    
    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)
        
        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=False)
        
        self.iteration += 1
        
        # Alternate which tree extends and which connects
        if self.swap_trees:
            nodes_extend, parent_extend, index_extend = self.nodes_b, self.parent_b, self._index_b
            nodes_connect, parent_connect, index_connect = self.nodes_a, self.parent_a, self._index_a
        else:
            nodes_extend, parent_extend, index_extend = self.nodes_a, self.parent_a, self._index_a
            nodes_connect, parent_connect, index_connect = self.nodes_b, self.parent_b, self._index_b

        # Sample random point and extend first tree
        q_rand = self._sample()
        status, q_new, rejected = self._extend(nodes_extend, parent_extend, index_extend, q_rand)

        if status == 'trapped':
            self.swap_trees = not self.swap_trees
            return StepResult(rejected_point=rejected)

        # Try to connect second tree to the new point
        connect_status, connect_edges, connect_rejected = self._connect(
            nodes_connect, parent_connect, index_connect, q_new
        )
        
        # Determine the edge to visualize (from extend step)
        i_new = len(nodes_extend) - 1
        i_parent = parent_extend[i_new]
        edge = (nodes_extend[i_parent], q_new)
        
        if connect_status == 'reached':
            # Trees connected!
            if self.swap_trees:
                self.connect_idx_b = len(self.nodes_b) - 1
                self.connect_idx_a = len(self.nodes_a) - 1
            else:
                self.connect_idx_a = len(self.nodes_a) - 1
                self.connect_idx_b = len(self.nodes_b) - 1
            
            self.done = True
            self.found_path = True
            return StepResult(edge=edge, done=True, found_path=True)
        
        self.swap_trees = not self.swap_trees
        return StepResult(edge=edge, rejected_point=connect_rejected)
    
    def extract_path(self) -> List[Tuple[int, int]]:
        if self.connect_idx_a is None or self.connect_idx_b is None:
            return []
        
        # Path from start to connection point
        path_a = []
        i = self.connect_idx_a
        while i != -1:
            path_a.append(self.nodes_a[i])
            i = self.parent_a[i]
        path_a.reverse()
        
        # Path from connection point to goal
        path_b = []
        i = self.connect_idx_b
        while i != -1:
            path_b.append(self.nodes_b[i])
            i = self.parent_b[i]
        
        # Combine paths (skip duplicate connection point)
        return path_a + path_b[1:] if len(path_b) > 1 else path_a + path_b
    
    def get_status(self) -> str:
        total_nodes = len(self.nodes_a) + len(self.nodes_b)
        return f"RRT-Connect: iter {self.iteration}/{self.max_iters}, nodes {total_nodes} (A:{len(self.nodes_a)}, B:{len(self.nodes_b)})"
    
    @staticmethod
    def get_params_widget() -> QWidget:
        return RRTConnectParamsWidget()
    
    @staticmethod
    def create_from_params(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                          params_widget: QWidget) -> 'RRTConnectPlanner':
        params = params_widget.get_params()
        return RRTConnectPlanner(occ, start, goal, **params)
