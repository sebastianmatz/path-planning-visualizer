from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

from ..geometry import (
    integrate_holonomic_state,
    line_collision_free,
    round_point,
    select_holonomic_input,
)
from ..types import FloatPoint, OccupancyGrid, Point
from ._spatial import GridIndex
from .base import BasePlanner, StepResult


class RRTPlanner(BasePlanner):
    """RRT adapted to a 2D holonomic occupancy-grid world.

    The implementation follows the structure of LaValle's
    ``GENERATE_RRT(x_init, K, Delta t)``:
    - ``RANDOM_STATE`` samples uniformly from the bounded workspace
    - ``NEAREST_NEIGHBOR`` uses Euclidean distance in the continuous state
    - ``SELECT_INPUT`` and ``NEW_STATE`` are specialized to ``x_dot = u`` with
      ``||u|| <= 1`` over a fixed ``Delta t``
    - an optional OMPL-style ``goal_bias`` can replace a random sample with the
      exact goal state
    - the UI's single-query path-planning interpretation terminates when a
      vertex enters a goal region around the clicked goal point
    """

    name = "RRT"
    description = "Rapidly-exploring Random Tree implementation after LaValle (1998) with optional OMPL-style goal bias"

    def __init__(
        self,
        occ: OccupancyGrid,
        start: Point,
        goal: Point,
        delta_t: float = 18.0,
        goal_region_radius: float = 20.0,
        goal_bias: float = 0.05,
        collision_samples: int = 80,
        max_vertices: int = 25000,
        seed: int = 1
    ) -> None:
        super().__init__(occ, start, goal)

        self.delta_t = float(max(1e-6, delta_t))
        self.goal_region_radius = float(max(1.0, goal_region_radius))
        self.goal_bias = float(np.clip(goal_bias, 0.0, 1.0))
        self.collision_samples = int(collision_samples)
        self.max_vertices = int(max(2, max_vertices))

        self.rng = np.random.default_rng(seed)
        self.states: List[FloatPoint] = [(float(start[0]), float(start[1]))]
        self.nodes: List[Point] = [start]
        self.node_set: Set[Point] = {start}
        self.parent: List[int] = [-1]
        self.controls: List[Optional[FloatPoint]] = [None]
        self.goal_idx: Optional[int] = None
        self._index = GridIndex(max(1.0, self.delta_t))
        self._index.add(self.states[0][0], self.states[0][1])
        if self._goal_reached(self.states[0]):
            self.goal_idx = 0
            self.found_path = True
            self.done = True

    def _sample_state(self) -> FloatPoint:
        """Sample uniformly from the bounded workspace, as in RANDOM_STATE."""
        if self.rng.random() < self.goal_bias:
            return (float(self.goal[0]), float(self.goal[1]))
        return (
            float(self.rng.uniform(0.0, max(0.0, self.w - 1))),
            float(self.rng.uniform(0.0, max(0.0, self.h - 1))),
        )

    def _nearest(self, state: FloatPoint) -> int:
        """Find the nearest existing continuous state to ``state``."""
        return self._index.nearest(state[0], state[1])

    def _goal_reached(self, state: FloatPoint) -> bool:
        goal_state = (float(self.goal[0]), float(self.goal[1]))
        if float(np.hypot(state[0] - goal_state[0], state[1] - goal_state[1])) > self.goal_region_radius:
            return False
        # Being within the goal region is not enough: the vertex must also have a
        # collision-free straight line to the actual goal, otherwise a vertex on
        # the far side of a thin wall would be accepted and the reported path
        # would appear to pass through the obstacle.
        return line_collision_free(round_point(state), self.goal, self.occ, samples=self.collision_samples)

    def _attempt_extension(
        self,
        sample_state: FloatPoint,
    ) -> Tuple[Optional[Point], Optional[Point], Optional[int], Optional[FloatPoint], Optional[FloatPoint]]:
        """Attempt one paper-style EXTEND step toward ``sample_state``."""
        i_near = self._nearest(sample_state)
        x_near = self.states[i_near]
        u = select_holonomic_input(x_near, sample_state, self.delta_t)
        if abs(u[0]) <= 1e-12 and abs(u[1]) <= 1e-12:
            return None, round_point(sample_state), None, None, None

        x_new = integrate_holonomic_state(x_near, u, self.delta_t)
        q_near = self.nodes[i_near]
        q_new = round_point(x_new)

        # Do not insert null motions caused by discretizing the continuous state.
        if q_new == q_near:
            return None, q_new, None, None, None
        if q_new in self.node_set:
            return None, q_new, None, None, None
        if not self.is_free(q_new):
            return None, q_new, None, None, None
        if not line_collision_free(q_near, q_new, self.occ, samples=self.collision_samples):
            return None, q_new, None, None, None
        return q_near, q_new, i_near, u, x_new

    def step_once(self) -> StepResult:
        if self.done:
            return StepResult(done=True, found_path=self.found_path)

        if len(self.nodes) >= self.max_vertices:
            self.done = True
            return StepResult(done=True, found_path=False)

        self.iteration += 1
        sample_state = self._sample_state()
        q_near, q_new, i_near, u, x_new = self._attempt_extension(sample_state)
        if q_near is None or q_new is None or i_near is None or u is None or x_new is None:
            return StepResult(rejected_point=q_new if q_new is not None else round_point(sample_state))

        self.nodes.append(q_new)
        self.node_set.add(q_new)
        self.parent.append(i_near)
        self.states.append(x_new)
        self.controls.append(u)
        self._index.add(x_new[0], x_new[1])

        if self._goal_reached(x_new):
            self.goal_idx = len(self.nodes) - 1
            self.done = True
            self.found_path = True
            return StepResult(edge=(q_near, q_new), done=True, found_path=True)

        if len(self.nodes) >= self.max_vertices:
            self.done = True
            return StepResult(edge=(q_near, q_new), done=True, found_path=False)

        return StepResult(edge=(q_near, q_new))

    def extract_path(self) -> List[Point]:
        if self.goal_idx is None:
            return []
        path: List[Point] = []
        i = self.goal_idx
        while i != -1:
            path.append(self.nodes[i])
            i = self.parent[i]
        path.reverse()
        # The accepting vertex was verified to have a collision-free line to the
        # goal (see _goal_reached), so finish the path at the actual goal.
        if path and path[-1] != self.goal:
            path.append(self.goal)
        return path

    def extract_display_path(self) -> List[Tuple[float, float]]:
        """The genuine RRT path: the polyline through the tree vertices, unsmoothed.

        RRT paths are jagged by nature (they follow the random tree); smoothing the
        displayed path would misrepresent the algorithm's actual output — and visually
        blur the difference from the optimizing variants (RRT*/BIT*).
        """
        return [(float(p[0]), float(p[1])) for p in self.extract_path()]

    def get_status(self) -> str:
        return (
            f"RRT: iter {self.iteration}, vertices {len(self.nodes)}/{self.max_vertices}, "
            f"Delta t {self.delta_t:.1f}, goal radius {self.goal_region_radius:.1f}, "
            f"goal bias {self.goal_bias:.2f}"
        )

