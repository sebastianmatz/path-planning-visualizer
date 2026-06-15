from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..geometry import (
    clamp_point,
    compute_path_length,
    dist,
    line_collision_free,
    linf_dist,
    segment_points,
)
from ..types import Edge, Point
from .base import BasePlanner, StepResult


@dataclass
class SBLSegmentState:
    """Lazy collision-check state for an SBL segment."""

    kappa: int = 0
    safe: bool = False


@dataclass
class SBLNode:
    """Milestone node used by SBL."""

    point: Point
    tree_id: int
    parent: Optional[int] = None
    children: Set[int] = field(default_factory=set)


class SBLPlanner(BasePlanner):
    """SBL - Single-query bi-directional planner with lazy collision checking."""

    name = "SBL"
    description = "Bi-directional lazy roadmap planner with deferred collision checking and a lightweight post-optimizer"

    def __init__(
        self,
        occ: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_iters: int = 12000,
        rho: int = 45,
        lazy_resolution: int = 4,
        max_candidates: int = 6,
        grid_cells: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__(occ, start, goal)

        self.max_iters = max_iters
        self.rho = float(rho)
        self.lazy_resolution = float(lazy_resolution)
        self.max_candidates = max_candidates
        self.grid_cells = grid_cells
        self.rng = np.random.default_rng(seed)

        self.nodes: List[SBLNode] = [
            SBLNode(point=start, tree_id=0, parent=None),
            SBLNode(point=goal, tree_id=1, parent=None),
        ]
        self.tree_nodes: Dict[int, Set[int]] = {0: {0}, 1: {1}}
        self.cell_maps: Dict[int, Dict[Tuple[int, int], Set[int]]] = {0: {}, 1: {}}
        self.edge_states: Dict[frozenset[int], SBLSegmentState] = {}
        self._add_to_cell_map(0)
        self._add_to_cell_map(1)

        self.raw_solution_path: List[Point] = []
        self.solution_path: List[Point] = []
        self.lazy_checks = 0
        self.transfer_count = 0
        self.bridge_attempts = 0
        self.last_collision_point: Optional[Point] = None

    def _edge_key(self, a: int, b: int) -> frozenset[int]:
        return frozenset((a, b))

    def _cell_for_point(self, p: Point) -> Tuple[int, int]:
        cx = min(int(p[0] * self.grid_cells / max(1, self.w)), self.grid_cells - 1)
        cy = min(int(p[1] * self.grid_cells / max(1, self.h)), self.grid_cells - 1)
        return (cx, cy)

    def _add_to_cell_map(self, node_id: int) -> None:
        node = self.nodes[node_id]
        cell = self._cell_for_point(node.point)
        self.cell_maps[node.tree_id].setdefault(cell, set()).add(node_id)

    def _remove_from_cell_map(self, node_id: int, tree_id: Optional[int] = None) -> None:
        actual_tree = self.nodes[node_id].tree_id if tree_id is None else tree_id
        cell = self._cell_for_point(self.nodes[node_id].point)
        ids = self.cell_maps[actual_tree].get(cell)
        if ids is None:
            return
        ids.discard(node_id)
        if not ids:
            del self.cell_maps[actual_tree][cell]

    def _move_node_between_trees(self, node_id: int, new_tree: int) -> None:
        old_tree = self.nodes[node_id].tree_id
        if old_tree == new_tree:
            return
        self._remove_from_cell_map(node_id, tree_id=old_tree)
        self.tree_nodes[old_tree].discard(node_id)
        self.nodes[node_id].tree_id = new_tree
        self.tree_nodes[new_tree].add(node_id)
        self._add_to_cell_map(node_id)

    def _pick_node_by_density(self, tree_id: int) -> int:
        non_empty_cells = [cell for cell, ids in self.cell_maps[tree_id].items() if ids]
        if not non_empty_cells:
            return 0 if tree_id == 0 else 1
        cell = non_empty_cells[int(self.rng.integers(0, len(non_empty_cells)))]
        ids = list(self.cell_maps[tree_id][cell])
        return ids[int(self.rng.integers(0, len(ids)))]

    def _random_tree_node(self, tree_id: int) -> Optional[int]:
        ids = list(self.tree_nodes[tree_id])
        if not ids:
            return None
        return ids[int(self.rng.integers(0, len(ids)))]

    def _sample_near(self, center: Point, radius: float) -> Point:
        radius = max(1.0, radius)
        iradius = max(1, int(np.ceil(radius)))
        for _ in range(40):
            dx = int(self.rng.integers(-iradius, iradius + 1))
            dy = int(self.rng.integers(-iradius, iradius + 1))
            p = clamp_point((center[0] + dx, center[1] + dy), self.w, self.h)
            if p != center and linf_dist(center, p) <= radius + 1e-6:
                return p
        return clamp_point((center[0] + iradius, center[1]), self.w, self.h)

    def _add_node(self, parent_id: int, point: Point, tree_id: int) -> int:
        node_id = len(self.nodes)
        self.nodes.append(SBLNode(point=point, tree_id=tree_id, parent=parent_id))
        self.nodes[parent_id].children.add(node_id)
        self.tree_nodes[tree_id].add(node_id)
        self._add_to_cell_map(node_id)
        self.edge_states[self._edge_key(parent_id, node_id)] = SBLSegmentState()
        return node_id

    def _expand_tree(self) -> Tuple[Optional[int], Optional[Edge]]:
        tree_id = int(self.rng.integers(0, 2))
        for _ in range(60):
            parent_id = self._pick_node_by_density(tree_id)
            parent_point = self.nodes[parent_id].point
            for i in range(1, self.max_candidates + 1):
                q = self._sample_near(parent_point, self.rho / i)
                if q == parent_point or not self.is_free(q):
                    continue
                node_id = self._add_node(parent_id, q, tree_id)
                return node_id, (parent_point, q)
        return None, None

    def _chain_root_to_node(self, node_id: int) -> List[int]:
        chain: List[int] = []
        current: Optional[int] = node_id
        while current is not None:
            chain.append(current)
            current = self.nodes[current].parent
        chain.reverse()
        return chain

    def _segment_length(self, edge_key: frozenset[int]) -> float:
        a, b = tuple(edge_key)
        return linf_dist(self.nodes[a].point, self.nodes[b].point)

    def _segment_new_points(self, a: Point, b: Point, level: int) -> List[Point]:
        denom = 2 ** level
        pts: List[Point] = []
        for odd in range(1, denom, 2):
            t = odd / denom
            x = int(round(a[0] + t * (b[0] - a[0])))
            y = int(round(a[1] + t * (b[1] - a[1])))
            p = clamp_point((x, y), self.w, self.h)
            if p != a and p != b and (not pts or pts[-1] != p):
                pts.append(p)
        return pts

    def _test_segment_once(self, edge_key: frozenset[int]) -> Optional[Point]:
        state = self.edge_states.setdefault(edge_key, SBLSegmentState())
        a_id, b_id = tuple(edge_key)
        a = self.nodes[a_id].point
        b = self.nodes[b_id].point
        next_level = state.kappa + 1
        self.lazy_checks += 1

        for p in self._segment_new_points(a, b, next_level):
            if not self.is_free(p):
                return p

        state.kappa = next_level
        if self._segment_length(edge_key) / (2 ** next_level) < self.lazy_resolution:
            if not line_collision_free(a, b, self.occ):
                for p in segment_points(a, b):
                    if not self.is_free(p):
                        return p
                return a
            state.safe = True
        return None

    def _segment_score(self, edge_key: frozenset[int]) -> float:
        state = self.edge_states[edge_key]
        if state.safe:
            return -1.0
        return self._segment_length(edge_key) / (2 ** state.kappa)

    def _build_solution_path(self, start_chain: List[int], goal_chain: List[int]) -> List[Point]:
        path = [self.nodes[node_id].point for node_id in start_chain]
        path.extend(self.nodes[node_id].point for node_id in reversed(goal_chain))

        deduped: List[Point] = []
        for p in path:
            if not deduped or deduped[-1] != p:
                deduped.append(p)
        return deduped

    def _point_at_arclength(self, path: List[Point], arclength: float) -> Tuple[int, Point]:
        """Return a point sampled along a polyline at the given arclength."""
        if len(path) < 2:
            return 0, path[0]

        remaining = max(0.0, arclength)
        for seg_idx in range(len(path) - 1):
            a = path[seg_idx]
            b = path[seg_idx + 1]
            seg_len = dist(a, b)
            if seg_len <= 1e-9:
                continue
            if remaining <= seg_len or seg_idx == len(path) - 2:
                t = float(np.clip(remaining / seg_len, 0.0, 1.0))
                x = int(round(a[0] + t * (b[0] - a[0])))
                y = int(round(a[1] + t * (b[1] - a[1])))
                return seg_idx, clamp_point((x, y), self.w, self.h)
            remaining -= seg_len
        return len(path) - 2, path[-1]

    def _splice_shortcut(
        self,
        path: List[Point],
        first_seg_idx: int,
        first_point: Point,
        second_seg_idx: int,
        second_point: Point,
    ) -> List[Point]:
        """Replace a subpath by a direct shortcut segment."""
        rebuilt: List[Point] = []
        for p in path[: first_seg_idx + 1]:
            if not rebuilt or rebuilt[-1] != p:
                rebuilt.append(p)
        if not rebuilt or rebuilt[-1] != first_point:
            rebuilt.append(first_point)
        if rebuilt[-1] != second_point:
            rebuilt.append(second_point)
        for p in path[second_seg_idx + 1:]:
            if rebuilt[-1] != p:
                rebuilt.append(p)
        return rebuilt

    def _optimize_solution_path(self, path: List[Point]) -> List[Point]:
        """Apply the SBL-style lightweight random path optimizer."""
        if len(path) < 3:
            return list(path)

        optimized = list(path)
        for _ in range(16):
            total_len = compute_path_length(optimized)
            if total_len <= 1e-6:
                break

            s1 = float(self.rng.uniform(0.0, total_len))
            s2 = float(self.rng.uniform(0.0, total_len))
            if s1 > s2:
                s1, s2 = s2, s1
            if s2 - s1 < 4.0:
                continue

            first_seg_idx, q1 = self._point_at_arclength(optimized, s1)
            second_seg_idx, q2 = self._point_at_arclength(optimized, s2)
            if q1 == q2:
                continue
            if not line_collision_free(q1, q2, self.occ):
                continue

            optimized = self._splice_shortcut(optimized, first_seg_idx, q1, second_seg_idx, q2)

        return optimized

    def _candidate_path_segments(
        self,
        start_chain: List[int],
        goal_chain: List[int],
        bridge_key: frozenset[int],
    ) -> List[frozenset[int]]:
        segments: List[frozenset[int]] = []
        for i in range(1, len(start_chain)):
            segments.append(self._edge_key(start_chain[i - 1], start_chain[i]))
        segments.append(bridge_key)
        for i in range(len(goal_chain) - 1, 0, -1):
            segments.append(self._edge_key(goal_chain[i], goal_chain[i - 1]))
        return segments

    def _test_candidate_path(
        self,
        start_chain: List[int],
        goal_chain: List[int],
        bridge_key: frozenset[int],
    ) -> Tuple[Optional[frozenset[int]], Optional[Point]]:
        path_segments = self._candidate_path_segments(start_chain, goal_chain, bridge_key)
        for seg in path_segments:
            self.edge_states.setdefault(seg, SBLSegmentState())

        unresolved = [seg for seg in path_segments if not self.edge_states[seg].safe]
        while unresolved:
            seg = max(unresolved, key=self._segment_score)
            collision_point = self._test_segment_once(seg)
            if collision_point is not None:
                return seg, collision_point
            unresolved = [key for key in path_segments if not self.edge_states[key].safe]
        return None, None

    def _find_edge_index(self, chain: List[int], edge_key: frozenset[int]) -> Optional[int]:
        for i in range(1, len(chain)):
            if self._edge_key(chain[i - 1], chain[i]) == edge_key:
                return i
        return None

    def _collect_subtree(self, root_id: int) -> Set[int]:
        stack = [root_id]
        subtree: Set[int] = set()
        while stack:
            node_id = stack.pop()
            if node_id in subtree:
                continue
            subtree.add(node_id)
            stack.extend(self.nodes[node_id].children)
        return subtree

    def _transfer_subtree_after_collision(
        self,
        chain: List[int],
        edge_index: int,
        opposite_bridge: int,
        new_tree: int,
        bridge_key: frozenset[int],
    ) -> None:
        parent = chain[edge_index - 1]
        child = chain[edge_index]
        collision_key = self._edge_key(parent, child)

        self.nodes[parent].children.discard(child)
        if self.nodes[child].parent == parent:
            self.nodes[child].parent = None
        self.edge_states.pop(collision_key, None)

        chain_sub = chain[edge_index:]  # child -> ... -> bridge endpoint on this side
        subtree_nodes = self._collect_subtree(child)

        for node_id in subtree_nodes:
            self._move_node_between_trees(node_id, new_tree)

        for i in range(len(chain_sub) - 1):
            old_parent = chain_sub[i]
            old_child = chain_sub[i + 1]
            self.nodes[old_parent].children.discard(old_child)

        rev_chain = list(reversed(chain_sub))  # bridge endpoint -> ... -> child
        bridge_side = rev_chain[0]
        self.nodes[bridge_side].parent = opposite_bridge
        self.nodes[opposite_bridge].children.add(bridge_side)
        self.edge_states.setdefault(bridge_key, SBLSegmentState())

        for i in range(1, len(rev_chain)):
            node_id = rev_chain[i]
            new_parent = rev_chain[i - 1]
            self.nodes[node_id].parent = new_parent
            self.nodes[new_parent].children.add(node_id)

        self.transfer_count += 1

    def _handle_path_collision(
        self,
        collision_key: frozenset[int],
        start_chain: List[int],
        goal_chain: List[int],
        bridge_key: frozenset[int],
    ) -> None:
        self.raw_solution_path = []
        self.solution_path = []
        if collision_key == bridge_key:
            self.edge_states.pop(bridge_key, None)
            return

        start_idx = self._find_edge_index(start_chain, collision_key)
        if start_idx is not None:
            self._transfer_subtree_after_collision(
                start_chain,
                start_idx,
                opposite_bridge=goal_chain[-1],
                new_tree=1,
                bridge_key=bridge_key,
            )
            return

        goal_idx = self._find_edge_index(goal_chain, collision_key)
        if goal_idx is not None:
            self._transfer_subtree_after_collision(
                goal_chain,
                goal_idx,
                opposite_bridge=start_chain[-1],
                new_tree=0,
                bridge_key=bridge_key,
            )
            return

        a_id, b_id = tuple(collision_key)
        if self.nodes[a_id].parent == b_id:
            self.nodes[b_id].children.discard(a_id)
            self.nodes[a_id].parent = None
        elif self.nodes[b_id].parent == a_id:
            self.nodes[a_id].children.discard(b_id)
            self.nodes[b_id].parent = None
        self.edge_states.pop(collision_key, None)

    def _attempt_connection(self, new_node_id: int) -> Optional[StepResult]:
        this_tree = self.nodes[new_node_id].tree_id
        other_tree = 1 - this_tree
        new_point = self.nodes[new_node_id].point
        same_cell = self._cell_for_point(new_point)

        candidate_ids: List[int] = []
        same_cell_ids = list(self.cell_maps[other_tree].get(same_cell, set()))
        if same_cell_ids:
            best_same_cell = min(same_cell_ids, key=lambda node_id: linf_dist(new_point, self.nodes[node_id].point))
            candidate_ids.append(best_same_cell)

        random_other = self._random_tree_node(other_tree)
        if random_other is not None and random_other not in candidate_ids:
            candidate_ids.append(random_other)

        attempted_edges: List[Edge] = []
        last_rejected_point: Optional[Point] = None
        for other_id in candidate_ids:
            other_point = self.nodes[other_id].point
            if linf_dist(new_point, other_point) >= self.rho:
                continue

            self.bridge_attempts += 1
            if this_tree == 0:
                start_bridge, goal_bridge = new_node_id, other_id
            else:
                start_bridge, goal_bridge = other_id, new_node_id

            start_chain = self._chain_root_to_node(start_bridge)
            goal_chain = self._chain_root_to_node(goal_bridge)
            bridge_key = self._edge_key(start_bridge, goal_bridge)
            self.edge_states.setdefault(bridge_key, SBLSegmentState())

            collision_key, collision_point = self._test_candidate_path(start_chain, goal_chain, bridge_key)
            bridge_edge = (self.nodes[start_bridge].point, self.nodes[goal_bridge].point)

            if collision_key is None:
                self.raw_solution_path = self._build_solution_path(start_chain, goal_chain)
                self.solution_path = self._optimize_solution_path(self.raw_solution_path)
                self.found_path = True
                self.done = True
                return StepResult(edge=bridge_edge, done=True, found_path=True)

            attempted_edges.append(bridge_edge)
            last_rejected_point = collision_point
            self.last_collision_point = collision_point
            self._handle_path_collision(collision_key, start_chain, goal_chain, bridge_key)
            if collision_key == bridge_key:
                continue
            return StepResult(edge=bridge_edge, rejected_point=collision_point)

        if attempted_edges:
            if len(attempted_edges) == 1:
                return StepResult(edge=attempted_edges[0], rejected_point=last_rejected_point)
            return StepResult(edges=attempted_edges, rejected_point=last_rejected_point)
        return None

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=False)

        self.iteration += 1

        new_node_id, expansion_edge = self._expand_tree()
        if new_node_id is None:
            return StepResult()

        connect_result = self._attempt_connection(new_node_id)
        edges: List[Edge] = [expansion_edge] if expansion_edge is not None else []

        if connect_result is not None:
            if connect_result.edge is not None:
                edges.append(connect_result.edge)
            if connect_result.edges:
                edges.extend(connect_result.edges)
            if len(edges) > 1:
                connect_result.edges = edges
                connect_result.edge = None
            elif len(edges) == 1 and connect_result.edge is None:
                connect_result.edge = edges[0]
            return connect_result

        if len(edges) == 1:
            return StepResult(edge=edges[0])
        if edges:
            return StepResult(edges=edges)
        return StepResult()

    def extract_path(self) -> List[Tuple[int, int]]:
        return list(self.solution_path)

    def get_status(self) -> str:
        if self.found_path:
            return (
                f"SBL: iter {self.iteration}/{self.max_iters}, "
                f"T0={len(self.tree_nodes[0])}, T1={len(self.tree_nodes[1])}, "
                f"checks {self.lazy_checks}, FOUND"
            )
        return (
            f"SBL: iter {self.iteration}/{self.max_iters}, "
            f"T0={len(self.tree_nodes[0])}, T1={len(self.tree_nodes[1])}, "
            f"checks {self.lazy_checks}, transfers {self.transfer_count}"
        )

