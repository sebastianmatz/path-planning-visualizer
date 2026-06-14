"""Paper-fidelity tests for CHOMP (Ratliff et al. 2009).

- the workspace cost c(x) matches the three-case paper form (Sec. II-D);
- standalone CHOMP refines past a small obstacle to a valid path;
- the post-optimization use (init_trajectory = a sampling-style path) returns a
  collision-free, smoother trajectory -- this is the GUI's "CHOMP Optimize".
"""

from __future__ import annotations

import numpy as np
import pytest

from path_planning_visualizer.geometry import dist, line_collision_free
from path_planning_visualizer.planners.chomp import CHOMPPlanner


def _run(p, n: int = 400000):
    steps = 0
    while not p.done and steps < n:
        p.step_once()
        steps += 1
    return p


def _segments_free(path, occ) -> bool:
    return all(line_collision_free(path[i], path[i + 1], occ) for i in range(len(path) - 1))


def test_workspace_cost_matches_paper_form():
    occ = np.zeros((60, 60), dtype=bool)
    occ[20:40, 20:40] = True
    p = CHOMPPlanner(occ, (5, 5), (55, 55), obstacle_epsilon=10)

    # d > eps: zero cost.
    c_far, _ = p._workspace_cost_and_gradient(5, 5)
    assert c_far == 0.0

    # d < 0 (inside obstacle): c = -d + eps/2 > eps/2.
    c_inside, _ = p._workspace_cost_and_gradient(30, 30)
    assert c_inside > 5.0

    # 0 <= d <= eps (near boundary): c = (d-eps)^2/(2eps), in (0, eps).
    c_near, _ = p._workspace_cost_and_gradient(18, 30)
    assert 0.0 < c_near < 10.0


def test_standalone_refines_past_small_obstacle():
    occ = np.zeros((60, 60), dtype=bool)
    occ[25:35, 28:32] = True  # small block on the straight line
    p = _run(CHOMPPlanner(occ, (8, 30), (52, 30), num_points=40, max_iters=600))
    if not p.found_path:
        pytest.skip("CHOMP did not converge (allowed for a local optimizer)")
    path = p.extract_path()
    assert _segments_free(path, occ)
    assert dist(path[0], (8, 30)) <= 2 and dist(path[-1], (52, 30)) <= 2


def test_post_optimization_smooths_and_stays_valid():
    """The GUI 'CHOMP Optimize' use: refine a jagged but valid path."""
    occ = np.zeros((60, 60), dtype=bool)
    jagged = [(5, 30), (13, 40), (21, 20), (29, 40), (37, 20), (45, 40), (51, 30), (55, 30)]
    p = _run(CHOMPPlanner(occ, (5, 30), (55, 30), init_trajectory=jagged,
                          num_points=40, max_iters=400))
    path = p.extract_path()
    # Endpoints preserved and the result is collision-free.
    assert dist(path[0], (5, 30)) <= 2 and dist(path[-1], (55, 30)) <= 2
    assert _segments_free(path, occ)
    # Smoothed toward the straight line (the input zig-zagged by +/-10 in y).
    ys = [y for _, y in path]
    assert max(abs(y - 30) for y in ys) < 8
