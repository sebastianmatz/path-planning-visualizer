from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..geometry import (
    clamp_point,
    dist,
    line_collision_free,
    steer,
)
from ._spatial import GridIndex
from .base import BasePlanner, StepResult


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
        """RANDOM_CONFIG: sample uniformly over the whole space C.

        An occupied configuration is fine -- it is only a steering target; the new
        vertex produced by EXTEND is collision-checked.
        """
        return (int(self.rng.integers(0, self.w)), int(self.rng.integers(0, self.h)))

    def _extend(self, nodes: List[Tuple[int, int]], parent: List[int], index: GridIndex,
                target: Tuple[int, int]) -> Tuple[str, Optional[int], Optional[Tuple[Tuple[int, int], Tuple[int, int]]], Optional[Tuple[int, int]]]:
        """EXTEND (paper Fig. 2): one eps-step from the nearest vertex toward target.

        Returns ``(status, new_idx, edge, rejected)`` where ``status`` is
        ``'reached'`` (q_new == target), ``'advanced'`` (a step was taken), or
        ``'trapped'`` (blocked); ``new_idx`` is the index of the resulting vertex.
        """
        i_near = index.nearest(target[0], target[1])
        q_near = nodes[i_near]

        # NEW_CONFIG: step eps toward target, landing exactly on it when within range.
        reached_target = dist(q_near, target) <= self.step_size
        q_new = target if reached_target else clamp_point(steer(q_near, target, self.step_size), self.w, self.h)

        if q_new == q_near:
            # No motion possible; if we are already at the target the tree reaches it.
            if reached_target:
                return ('reached', i_near, None, None)
            return ('trapped', None, None, q_new)

        if not self.is_free(q_new):
            return ('trapped', None, None, q_new)
        if not line_collision_free(q_near, q_new, self.occ, samples=self.collision_samples):
            return ('trapped', None, None, q_new)

        nodes.append(q_new)
        parent.append(i_near)
        index.add(q_new[0], q_new[1])
        new_idx = len(nodes) - 1
        edge = (q_near, q_new)

        if q_new == target:  # NEW_CONFIG reached q exactly
            return ('reached', new_idx, edge, None)
        return ('advanced', new_idx, edge, None)

    def _connect(self, nodes: List[Tuple[int, int]], parent: List[int], index: GridIndex,
                 target: Tuple[int, int]) -> Tuple[str, Optional[int], List[Tuple[Tuple[int, int], Tuple[int, int]]], Optional[Tuple[int, int]]]:
        """CONNECT (paper Fig. 5): repeat EXTEND until the result is not Advanced.

        Returns ``(status, connect_idx, edges, rejected)`` where ``connect_idx`` is
        the index of the vertex at the connection target when ``status='reached'``.
        """
        edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        last_idx: Optional[int] = None
        while True:
            status, new_idx, edge, rejected = self._extend(nodes, parent, index, target)
            if edge is not None:
                edges.append(edge)
            if new_idx is not None:
                last_idx = new_idx
            if status != 'advanced':
                return (status, last_idx, edges, rejected)

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=False)

        self.iteration += 1

        # Alternate which tree extends (one step) and which connects (greedy).
        if self.swap_trees:
            nodes_extend, parent_extend, index_extend = self.nodes_b, self.parent_b, self._index_b
            nodes_connect, parent_connect, index_connect = self.nodes_a, self.parent_a, self._index_a
        else:
            nodes_extend, parent_extend, index_extend = self.nodes_a, self.parent_a, self._index_a
            nodes_connect, parent_connect, index_connect = self.nodes_b, self.parent_b, self._index_b

        # EXTEND(T_a, q_rand): one step of the first tree toward a random config.
        q_rand = self._sample()
        status, extend_idx, extend_edge, rejected = self._extend(
            nodes_extend, parent_extend, index_extend, q_rand
        )

        if status == 'trapped':
            self.swap_trees = not self.swap_trees
            return StepResult(rejected_point=rejected)

        # CONNECT(T_b, q_new): greedily grow the other tree toward the new vertex.
        q_new = nodes_extend[extend_idx]
        connect_status, connect_idx, connect_edges, connect_rejected = self._connect(
            nodes_connect, parent_connect, index_connect, q_new
        )

        edges = ([extend_edge] if extend_edge is not None else []) + connect_edges

        if connect_status == 'reached':
            # Trees meet at the shared, collision-checked vertex q_new.
            if self.swap_trees:  # extend tree is T_b, connect tree is T_a
                self.connect_idx_b = extend_idx
                self.connect_idx_a = connect_idx
            else:                # extend tree is T_a, connect tree is T_b
                self.connect_idx_a = extend_idx
                self.connect_idx_b = connect_idx
            self.done = True
            self.found_path = True
            if len(edges) > 1:
                return StepResult(edges=edges, done=True, found_path=True)
            return StepResult(edge=edges[0] if edges else None, done=True, found_path=True)

        self.swap_trees = not self.swap_trees
        if len(edges) > 1:
            return StepResult(edges=edges, rejected_point=connect_rejected)
        return StepResult(edge=edges[0] if edges else None, rejected_point=connect_rejected)
    
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
    
    