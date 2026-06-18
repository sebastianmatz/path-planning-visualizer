from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..geometry import (
    clamp_point,
    dist,
    line_collision_free,
    segment_points,
    steer,
)
from ..types import Edge, Point
from .base import BasePlanner, StepResult


@dataclass
class KPIECEMotion:
    """A single motion (tree edge) in the KPIECE exploration tree.

    A "motion" here is a straight, Range-bounded extension from ``start`` to ``end``
    -- the geometric 2D-point-robot stand-in for the paper's forward-propagated,
    control-driven motion (see the class docstring's adaptation notes). ``cell_coord``
    is the projection cell the motion is filed under (no motion crosses a cell
    boundary; see ``_split_segment_by_cells``).
    """

    start: Point
    end: Point
    parent: Optional[int]
    cell_coord: Tuple[int, int]


@dataclass
class KPIECECell:
    """One cell of the projection grid, carrying the SK09 importance bookkeeping.

    The field names map directly onto the importance equation
    ``Importance(p) = log(I) * score / (S * N * C)`` (SK09 p. 6):

    - ``creation_iteration`` -> I: the iteration the cell was instantiated.
    - ``score``              -> the exploration-progress weight (init 1, shrunk by the
      progress penalty when the cell stops making progress).
    - ``selections``         -> S: how many times the cell has been selected (init 1).
    - ``neighbor_count``     -> N: instantiated same-level axis-neighbours.
    - ``coverage``           -> C: sum of motion "durations" (here lengths) in the cell.
    - ``border``             -> exterior flag: True iff ``neighbor_count < 2n = 4``
      (SK09 p. 4, the "< 2n" interior/exterior rule for n = 2).
    """

    coord: Tuple[int, int]
    motion_indices: List[int] = field(default_factory=list)
    coverage: float = 0.0
    selections: int = 1
    score: float = 1.0
    creation_iteration: int = 1
    importance: float = 0.0
    border: bool = True
    neighbor_count: int = 0


class KPIECEPlanner(BasePlanner):
    """KPIECE - Kinodynamic Planning by Interior-Exterior Cell Exploration.

    Faithful single-level *geometric* adaptation of Sucan & Kavraki (2009, WAFR 2008;
    cite ``SK09``) to a 2D holonomic point robot. The defining idea is a *projection
    grid* over the workspace whose cells carry an **importance** score; each iteration
    the planner selects the most important cell (biased toward the *exterior* of the
    explored region, where growth is still possible), grows a motion from it, and
    penalizes cells that stop making progress -- so exploration keeps pushing into
    unexplored space instead of churning in already-covered regions.

    Paper correspondence (Section 3, Algorithms 1-2; equations p. 6):

    - Importance ``log(I) * score / (S * N * C)`` -- ``_compute_importance``.
    - Exterior (border) cell iff ``< 2n = 4`` instantiated axis-neighbours
      (n = 2) -- ``KPIECECell.border`` / ``_recompute_cells``.
    - Cell selection: deterministic highest-importance pick with a fixed exterior
      bias of 70-80% -- ``_select_cell`` (Alg. 1 line 5).
    - Motion picked from the cell by a half-normal law (recent motions preferred),
      then a state uniformly along it -- ``_select_motion_index`` /
      ``_select_state_on_motion`` (Alg. 1 lines 6-7).
    - AddMotion splits an extension at cell boundaries so no motion straddles two
      cells -- ``_split_segment_by_cells`` / ``_add_motion_chain`` (Alg. 2 line 20).
    - Progress penalty ``P = alpha + beta * (deltaC / simulated_time)``; if ``P < 1``
      the selected cell's ``score`` is multiplied by ``P`` -- ``_apply_progress_penalty``
      (Alg. 1 lines 16-17).

    Adaptations (stated for fidelity):

    - **Single-level grid (k = 1)** instead of the paper's multilevel grids
      ``L_1..L_k``; coverage is the sum of motion lengths in that one level.
    - **Geometric, not kinodynamic:** there is no forward-propagation ``f`` / physics.
      A "motion" is a straight Range-bounded extension whose collision-free prefix is
      kept (``min_valid_path_fraction``); "simulated time" in the progress term is the
      traveled *distance* -- the 2D point-robot analogue. Controls ``u in U`` (Alg. 1
      line 8) do not apply.
    - A **score-underflow reset** uniformly lifts all scores when the best drops below
      machine epsilon, keeping the importance ordering numerically stable on long runs.
    - An optional small ``goal_bias`` (off in the paper) aids single-query 2D use.

    See ``literature/fidelity/kpiece.md`` and ``tests/test_kpiece_fidelity.py``.
    """

    name = "KPIECE"
    description = "Projection-grid planner with state sampling along motions, border-cell preference, and progress-based score penalties"

    def __init__(
        self,
        occ: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        range: float = 24.0,
        goal_bias: float = 0.02,
        goal_tolerance: int = 24,
        border_fraction: float = 0.80,
        progress_alpha: float = 0.10,
        progress_beta: float = 0.90,
        min_valid_path_fraction: float = 0.20,
        cell_size: int = 28,
        max_iters: int = 25000,
        seed: int = 1,
    ) -> None:
        super().__init__(occ, start, goal)

        self.max_distance = float(max(1.0, range))
        self.goal_bias = float(np.clip(goal_bias, 0.0, 1.0))
        self.goal_tolerance = float(max(1, goal_tolerance))
        self.border_fraction = float(np.clip(border_fraction, 0.0, 1.0))
        self.progress_alpha = float(max(1e-6, progress_alpha))
        self.progress_beta = float(max(0.0, progress_beta))
        self.min_valid_path_fraction = float(np.clip(min_valid_path_fraction, 1e-6, 1.0))
        self.cell_size = int(max(2, cell_size))
        self.max_iters = int(max_iters)
        self.rng = np.random.default_rng(seed)

        self.motions: List[KPIECEMotion] = []
        self.cells: Dict[Tuple[int, int], KPIECECell] = {}
        self.point_set: Set[Point] = set()
        self.goal_idx: Optional[int] = None
        self.closest_goal_idx: Optional[int] = None
        self.closest_goal_dist = float('inf')
        self.border_cell_count = 0

        start_idx, _ = self._add_motion(start, start, parent=None)
        self.closest_goal_idx = start_idx
        self.closest_goal_dist = dist(start, goal)

        if start == goal:
            self.goal_idx = start_idx
            self.found_path = True
            self.done = True

    def _project(self, point: Point) -> Tuple[int, int]:
        """Projection from a workspace pixel to its grid-cell coordinate (the map ``p``).

        KPIECE explores a low-dimensional *projection* of the state space; for a 2D
        point robot the natural projection is the identity-into-a-coarse-grid, i.e.
        integer-divide the pixel by ``cell_size``.
        """
        return (int(point[0] // self.cell_size), int(point[1] // self.cell_size))

    @staticmethod
    def _neighbor_coords(coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        """The four axis (non-diagonal) neighbours used for the ``< 2n`` exterior test."""
        cx, cy = coord
        return [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]

    def _compute_importance(self, cell: KPIECECell) -> None:
        """Importance ``log(I) * score / (S * N * C)`` (SK09 p. 6).

        Each factor is clamped -- ``log(max(2, I))`` and ``max(1, .)`` on every
        denominator term -- so the importance stays finite and strictly positive even
        for a brand-new cell with no neighbours and unit coverage (where N, C, S = 1).
        """
        creation_term = float(np.log(max(2, cell.creation_iteration)))   # log(I)
        neighbor_term = float(max(1, cell.neighbor_count))               # N
        coverage_term = max(1.0, cell.coverage)                          # C
        selection_term = float(max(1, cell.selections))                  # S
        cell.importance = creation_term * cell.score / (selection_term * neighbor_term * coverage_term)

    def _recompute_cells(self, coords: Set[Tuple[int, int]]) -> None:
        """Refresh neighbour count, exterior/border flag, and importance for ``coords``.

        A cell is *exterior* (border) when it has fewer than ``2n = 4`` instantiated
        axis-neighbours (SK09 p. 4, the "< 2n" rule for n = 2). ``border_cell_count``
        is kept in sync on every flip so ``_select_cell`` can bias toward the exterior.
        """
        for coord in coords:
            cell = self.cells.get(coord)
            if cell is None:
                continue
            prev_border = cell.border
            cell.neighbor_count = sum(1 for ncoord in self._neighbor_coords(coord) if ncoord in self.cells)
            cell.border = cell.neighbor_count < 4
            if cell.border != prev_border:
                self.border_cell_count += 1 if cell.border else -1
            self._compute_importance(cell)

    def _add_motion(
        self, start: Point, end: Point, parent: Optional[int], coord: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, float]:
        coord = self._project(end) if coord is None else coord
        motion_idx = len(self.motions)
        self.motions.append(KPIECEMotion(start=start, end=end, parent=parent, cell_coord=coord))
        self.point_set.add(end)

        cell = self.cells.get(coord)
        affected_coords: Set[Tuple[int, int]] = {coord}
        if cell is None:
            # New cell: stamp its creation iteration I and refresh the border status of
            # itself and its neighbours (a fresh neighbour can flip an adjacent cell from
            # exterior to interior).
            cell = KPIECECell(coord=coord, creation_iteration=max(1, self.iteration))
            self.cells[coord] = cell
            self.border_cell_count += 1
            affected_coords.update(self._neighbor_coords(coord))

        # Coverage C is the sum of motion "durations" in the cell. The root seed counts
        # as one unit; every real motion contributes its length (the geometric analogue
        # of the paper's motion duration).
        coverage_delta = 1.0 if parent is None else max(1e-6, dist(start, end))
        cell.coverage += coverage_delta
        cell.motion_indices.append(motion_idx)

        self._recompute_cells(affected_coords)

        goal_dist = dist(end, self.goal)
        if goal_dist < self.closest_goal_dist:
            self.closest_goal_dist = goal_dist
            self.closest_goal_idx = motion_idx

        return motion_idx, coverage_delta

    def _select_cell(self) -> KPIECECell:
        """Select the cell to expand from (Alg. 1 line 5): top importance, exterior-biased.

        First flip a biased coin: with probability ``border_fraction`` (the paper's
        fixed 70-80% exterior bias) restrict the candidate pool to exterior cells,
        otherwise to interior cells. Then pick *deterministically* the highest-importance
        cell in the pool (ties broken by older creation, fewer selections, more motions).

        Score-underflow reset: importance is proportional to ``score``, which the
        progress penalty repeatedly shrinks; if the chosen cell's score has decayed
        below machine epsilon, lift every cell's score uniformly and re-rank. This
        preserves the *relative* importance ordering while keeping the arithmetic away
        from denormals on long runs (an adaptation, not in the paper).
        """
        all_cells = [cell for cell in self.cells.values() if cell.motion_indices]
        border_cells = [cell for cell in all_cells if cell.border]
        interior_cells = [cell for cell in all_cells if not cell.border]

        if border_cells and (not interior_cells or self.rng.random() < self.border_fraction):
            pool = border_cells
        else:
            pool = interior_cells if interior_cells else all_cells

        chosen = max(
            pool,
            key=lambda cell: (
                cell.importance,
                cell.creation_iteration,
                -cell.selections,
                len(cell.motion_indices),
            ),
        )
        if chosen.score < np.finfo(np.float64).eps:
            for cell in self.cells.values():
                cell.score += 1.0
                self._compute_importance(cell)
            chosen = max(
                pool,
                key=lambda cell: (
                    cell.importance,
                    cell.creation_iteration,
                    -cell.selections,
                    len(cell.motion_indices),
                ),
            )
        # Selecting the cell costs it one "selection" S, lowering its future
        # importance so attention rotates to other cells.
        chosen.selections += 1
        self._compute_importance(chosen)
        return chosen

    def _select_motion_index(self, cell: KPIECECell) -> int:
        """Pick a motion in the cell by a half-normal law over recency (Alg. 1 line 6).

        ``motion_indices`` is append-ordered, so index ``-1`` is the most recent motion.
        A folded-Gaussian (half-normal) offset from the newest end biases selection
        toward recently added motions -- the paper's preference for growing from the
        latest frontier -- while still occasionally reaching back to older motions. The
        spread ``sigma`` grows with the number of motions so larger cells stay reachable.
        """
        if len(cell.motion_indices) == 1:
            return cell.motion_indices[0]

        sigma = max(1.0, len(cell.motion_indices) / 3.0)
        offset = min(int(abs(self.rng.normal(0.0, sigma))), len(cell.motion_indices) - 1)
        return cell.motion_indices[-1 - offset]

    def _select_state_on_motion(self, motion_idx: int) -> Point:
        """Pick a state uniformly along the chosen motion (Alg. 1 line 7).

        Geometric adaptation: the paper samples a state along a forward-propagated
        trajectory; for a straight 2D motion this is a uniform interpolation
        ``start + t*(end - start)``, ``t ~ U[0, 1]``.
        """
        motion = self.motions[motion_idx]
        if motion.start == motion.end:
            return motion.end
        t = float(self.rng.random())
        x = int(round(motion.start[0] + t * (motion.end[0] - motion.start[0])))
        y = int(round(motion.start[1] + t * (motion.end[1] - motion.start[1])))
        return clamp_point((x, y), self.w, self.h)

    def _sample_target(self, from_pt: Point) -> Point:
        """Sample an extension target within the ``Range`` disk around ``from_pt``.

        With probability ``goal_bias`` aim straight at the goal (a single-query 2D aid,
        off in the paper); otherwise sample uniformly in the disk of radius
        ``max_distance`` (``Range``) and clamp the result back inside it, so no single
        motion is longer than ``Range``.
        """
        if self.rng.random() < self.goal_bias:
            target = self.goal
        else:
            angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
            radius = float(self.max_distance * np.sqrt(self.rng.random()))
            target = (
                int(round(from_pt[0] + radius * np.cos(angle))),
                int(round(from_pt[1] + radius * np.sin(angle))),
            )

        target = clamp_point(target, self.w, self.h)
        if dist(from_pt, target) > self.max_distance:
            target = clamp_point(steer(from_pt, target, self.max_distance), self.w, self.h)
        return target

    def _split_segment_by_cells(
        self, start: Point, end: Point
    ) -> List[Tuple[Point, Point, Tuple[int, int]]]:
        """Split an extension at projection-cell boundaries (AddMotion, Alg. 2 line 20).

        The paper requires that a stored motion lie within a single cell so that
        coverage and cell membership are well defined. We walk the rasterized segment
        and cut it wherever it crosses into a new cell, returning ``(seg_start,
        seg_end, cell)`` pieces -- each piece becomes one ``KPIECEMotion`` filed under
        its own cell.
        """
        points = segment_points(start, end)
        if len(points) < 2:
            return []

        segments: List[Tuple[Point, Point, Tuple[int, int]]] = []
        current_start = points[0]
        current_cell = self._project(points[0])
        for point in points[1:]:
            point_cell = self._project(point)
            if point_cell != current_cell:
                if current_start != point:
                    segments.append((current_start, point, current_cell))
                current_start = point
                current_cell = point_cell
        if current_start != points[-1]:
            segments.append((current_start, points[-1], current_cell))
        return segments

    def _partial_extension(self, from_pt: Point, target: Point) -> Tuple[Optional[Point], float]:
        """Geometric stand-in for the paper's control propagation toward ``target``.

        The paper propagates a control until it collides or the duration elapses, then
        keeps the valid prefix. Here we walk the straight segment and keep its longest
        collision-free prefix. A *partial* extension (one that stops before ``target``)
        is only accepted if it covered at least ``min_valid_path_fraction`` of the way;
        otherwise the motion is too stunted to be worth storing and we reject it.

        Returns ``(last_valid_point, distance_traveled)``, or ``(None, .)`` on rejection.
        """
        if target == from_pt:
            return None, 0.0

        total_dist = dist(from_pt, target)
        if total_dist <= 1e-6:
            return None, 0.0

        last_valid = from_pt
        for point in segment_points(from_pt, target):
            if point == from_pt:
                continue
            if not self.is_free(point):
                break
            last_valid = point

        if last_valid == from_pt:
            return None, 0.0

        moved_dist = dist(from_pt, last_valid)
        valid_fraction = moved_dist / total_dist
        if last_valid != target and valid_fraction < self.min_valid_path_fraction:
            return None, moved_dist
        return last_valid, moved_dist

    def _apply_progress_penalty(self, cell: KPIECECell, coverage_delta: float, simulated_distance: float) -> None:
        """Progress penalty ``P = alpha + beta * (deltaC / simulated_time)`` (Alg. 1 lines 16-17).

        ``deltaC`` is how much coverage this expansion added and ``simulated_time`` is
        the traveled distance (the 2D point-robot analogue of the paper's simulation
        time). Only a *lack* of progress is penalized: if ``P < 1`` the cell's ``score``
        is multiplied by ``P`` (lowering its future importance); if ``P >= 1`` the score
        is left untouched -- the paper deliberately does not reward over and over a cell
        that happens to make a lot of progress.
        """
        if simulated_distance <= 1e-6:
            progress = self.progress_alpha
        else:
            progress = self.progress_alpha + self.progress_beta * (coverage_delta / simulated_distance)
        if progress < 1.0:
            cell.score *= progress
            self._compute_importance(cell)

    def _add_motion_chain(
        self, parent_idx: int, start: Point, end: Point
    ) -> Tuple[Optional[int], List[Edge], float]:
        """Add the boundary-split extension as a chain of single-cell motions (Alg. 2).

        Each piece returned by ``_split_segment_by_cells`` becomes one motion, chained
        parent->child, so the stored tree never has a motion crossing a cell boundary.
        Returns the last motion's index, the per-piece edges (for drawing), and the
        total coverage added.
        """
        segments = self._split_segment_by_cells(start, end)
        if not segments:
            return None, [], 0.0

        total_coverage_delta = 0.0
        edges: List[Edge] = []
        current_parent = parent_idx
        last_idx: Optional[int] = None
        for seg_start, seg_end, seg_cell in segments:
            if seg_start == seg_end:
                continue
            motion_idx, coverage_delta = self._add_motion(seg_start, seg_end, current_parent, coord=seg_cell)
            total_coverage_delta += coverage_delta
            edges.append((seg_start, seg_end))
            current_parent = motion_idx
            last_idx = motion_idx
        return last_idx, edges, total_coverage_delta

    def _connect_goal(self, from_idx: int) -> Optional[Tuple[int, List[Edge]]]:
        """Single-query goal connection: link a new motion straight to the goal if close.

        Single-query 2D termination (the WAFR paper plans toward a goal *region*, not a
        clicked point): if the new motion's endpoint is within ``goal_tolerance`` of the
        goal and has a collision-free straight line to it, attach the goal and finish.
        """
        from_pt = self.motions[from_idx].end
        if dist(from_pt, self.goal) > self.goal_tolerance:
            return None
        if not line_collision_free(from_pt, self.goal, self.occ):
            return None

        if from_pt == self.goal:
            self.goal_idx = from_idx
            self.done = True
            self.found_path = True
            return (from_idx, [])

        goal_idx, goal_edges, _ = self._add_motion_chain(from_idx, from_pt, self.goal)
        if goal_idx is None:
            return None

        self.goal_idx = goal_idx
        self.done = True
        self.found_path = True
        return (goal_idx, goal_edges)

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if self.iteration >= self.max_iters:
            self.done = True
            return StepResult(done=True, found_path=False)

        # One KPIECE iteration (Alg. 1): select cell -> select motion -> select a state
        # on it -> sample a Range-bounded target -> extend -> penalize progress.
        self.iteration += 1
        cell = self._select_cell()                       # Alg. 1 line 5
        motion_idx = self._select_motion_index(cell)     # Alg. 1 line 6
        source = self._select_state_on_motion(motion_idx)  # Alg. 1 line 7
        target = self._sample_target(source)
        attempted_distance = max(dist(source, target), 1e-6)
        new_point, traveled = self._partial_extension(source, target)

        if new_point is None or new_point in self.point_set:
            # Failed / duplicate extension: zero coverage gained, so the progress ratio
            # is small and the cell's score is penalized for not advancing the frontier.
            self._apply_progress_penalty(cell, coverage_delta=0.0, simulated_distance=attempted_distance)
            return StepResult(rejected_point=target)

        new_idx, edges, coverage_delta = self._add_motion_chain(motion_idx, source, new_point)
        self._apply_progress_penalty(cell, coverage_delta=coverage_delta, simulated_distance=max(traveled, attempted_distance))
        if new_idx is None:
            return StepResult(rejected_point=target)

        goal_result = self._connect_goal(new_idx)
        if goal_result is not None:
            _, goal_edges = goal_result
            if goal_edges:
                return StepResult(edges=edges + goal_edges, done=True, found_path=True)
            return StepResult(edges=edges, done=True, found_path=True)

        if len(edges) == 1:
            return StepResult(edge=edges[0])
        return StepResult(edges=edges)

    def extract_path(self) -> List[Point]:
        if self.goal_idx is None:
            return []

        chain: List[Tuple[int, Point]] = []
        current: Optional[int] = self.goal_idx
        target = self.motions[self.goal_idx].end
        while current is not None:
            chain.append((current, target))
            motion = self.motions[current]
            target = motion.start
            current = motion.parent

        chain.reverse()
        root_motion = self.motions[chain[0][0]]
        path: List[Point] = [root_motion.start]
        for _, target_state in chain:
            if path[-1] != target_state:
                path.append(target_state)
        return path

    def get_status(self) -> str:
        status = "FOUND" if self.found_path else f"closest {self.closest_goal_dist:.1f}px"
        return (
            f"KPIECE: iter {self.iteration}/{self.max_iters}, motions {len(self.motions)}, "
            f"cells {len(self.cells)} (border {self.border_cell_count}), {status}"
        )

