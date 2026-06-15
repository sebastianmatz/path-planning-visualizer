"""Behavioural tests for the path planners.

These tests assert the properties a planner *must* uphold regardless of its
internal strategy:

* Completeness-ish: the resolution-complete / graph-search planners must find a
  path on a simple solvable map.
* Soundness: whenever a planner reports ``found_path``, the path it returns must
  be collision-free and actually connect start to goal.
* Optimality (A*/Dijkstra): both must return the same cost on the same induced
  grid, since both are optimal there.

No Qt objects are constructed; planners are built directly from keyword args, so
the suite runs headless.
"""

from __future__ import annotations

import numpy as np
import pytest

import path_planning_visualizer as ppv

# ---------------------------------------------------------------------------
# Maps and fixtures
# ---------------------------------------------------------------------------

START = (8, 8)
GOAL = (52, 8)
GOAL_TOLERANCE = 26.0  # generous; some planners stop within a goal region


def wall_map() -> np.ndarray:
    """60x60 map with a thick vertical wall and an open gap at the bottom."""
    occ = np.zeros((60, 60), dtype=bool)
    occ[0:45, 30] = True
    occ[0:45, 31] = True  # 2px thick so coarse grids still see it
    return occ


@pytest.fixture()
def occ() -> np.ndarray:
    return wall_map()


# Per-planner parameters tuned to keep the suite fast but reliable. Budgets are
# deliberately small: the 60x60 map solves in well under these limits, and the
# experimental optimizers only need a short run (they may legitimately skip).
TUNED = {
    "RRT": dict(delta_t=12.0, goal_region_radius=14.0, goal_bias=0.1, max_vertices=4000, seed=1),
    "RRT-Connect": dict(step_size=10, max_iters=4000, seed=1),
    "BiTRRT": dict(range=12.0, max_iters=4000, seed=1),
    "KPIECE": dict(range=12.0, goal_tolerance=14, cell_size=8, max_iters=4000, seed=1),
    "RRT*": dict(step_size=10, goal_tolerance=14, search_radius=24, max_iters=2500, seed=1),
    "PRM": dict(num_samples=400, k_neighbors=15, max_edge_dist=40, seed=1),
    "sPRM": dict(num_samples=400, k_neighbors=15, max_edge_dist=40, seed=1),
    "SBL": dict(max_iters=5000, rho=20, grid_cells=8, seed=1),
    "FMT*": dict(num_samples=500, seed=1),
    "BIT*": dict(batch_size=200, max_iters=5000, step_size=18.0, seed=1),
    "A*": dict(grid_size=5, allow_diagonal=True),
    "Dijkstra": dict(grid_size=5, allow_diagonal=True),
    "APF": dict(max_iters=2000, seed=1),
    "CHOMP": dict(num_points=40, max_iters=400),
    "STOMP": dict(num_points=40, max_iters=250, seed=1),
    "TrajOpt": dict(num_points=40, max_iters=250),
    "ITOMP": dict(num_points=30, max_iters=400, seed=1),
    "GPMP": dict(num_points=25, max_iters=400),
    "PSO": dict(num_particles=30, num_points=20, max_iters=150, seed=1),
    "Genetic": dict(pop_size=40, num_points=20, max_iters=150, seed=1),
}

# Planners that should always find a path on the simple wall map.
COMPLETE = ["RRT", "RRT-Connect", "BiTRRT", "KPIECE", "RRT*", "PRM", "sPRM", "SBL", "FMT*", "BIT*", "A*", "Dijkstra"]

ALL = list(ppv.AVAILABLE_PLANNERS)

# APF only guarantees its waypoints are free, not full-segment validity.
STRICT = [name for name in ALL if name != "APF"]


def build(name: str, occ: np.ndarray):
    return ppv.AVAILABLE_PLANNERS[name](occ, START, GOAL, **TUNED[name])


def run(planner, max_steps: int = 400_000):
    steps = 0
    while not planner.done and steps < max_steps:
        planner.step_once()
        steps += 1
    return planner


def segments_collision_free(path, occ) -> bool:
    return all(ppv.line_collision_free(path[i], path[i + 1], occ) for i in range(len(path) - 1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", COMPLETE)
def test_complete_planners_find_path(name, occ):
    planner = run(build(name, occ))
    assert planner.found_path, f"{name} failed to find a path on a solvable map"


@pytest.mark.parametrize("name", STRICT)
def test_found_paths_are_sound(name, occ):
    """Any reported path must be collision-free and connect start to goal."""
    planner = run(build(name, occ))
    if not planner.found_path:
        pytest.skip(f"{name} did not converge to a path (allowed for this planner)")

    path = planner.extract_path()
    assert len(path) >= 2, f"{name} reported success with a degenerate path"
    assert segments_collision_free(path, occ), f"{name} returned a path that clips an obstacle"
    assert ppv.dist(path[0], START) <= 2.0, f"{name} path does not start at start"
    assert ppv.dist(path[-1], GOAL) <= GOAL_TOLERANCE, f"{name} path does not reach goal"


def test_apf_waypoints_are_free(occ):
    """APF cannot guarantee full-segment validity, but its waypoints must be free."""
    planner = run(build("APF", occ))
    if not planner.found_path:
        pytest.skip("APF got stuck in a local minimum (allowed)")
    for x, y in planner.extract_path():
        assert not occ[y, x], "APF visited an occupied pixel"


def test_astar_matches_dijkstra_length(occ):
    """A* and Dijkstra must agree on cost: both are optimal on the same grid."""
    a = run(build("A*", occ))
    d = run(build("Dijkstra", occ))
    assert a.found_path and d.found_path
    la = ppv.compute_path_length(a.extract_path())
    ld = ppv.compute_path_length(d.extract_path())
    assert abs(la - ld) <= 5.0, f"A*={la:.2f} vs Dijkstra={ld:.2f} diverge on the same grid"


@pytest.mark.parametrize("name", ["A*", "Dijkstra"])
def test_grid_path_touches_real_endpoints(name, occ):
    """Regression: stitched path must touch the clicked start/goal exactly."""
    planner = run(build(name, occ))
    path = planner.extract_path()
    assert path[0] == START
    assert path[-1] == GOAL


@pytest.mark.parametrize("name", ["A*", "Dijkstra"])
def test_goal_in_occupied_supercell_still_solvable(name):
    """Regression: a free goal whose coarse supercell contains an obstacle pixel
    must remain reachable (the supercell-occupancy bug used to report No Path)."""
    occ = np.zeros((40, 40), dtype=bool)
    occ[10, 20] = True  # single obstacle pixel sharing the goal's 8x8 supercell
    start = (2, 2)
    goal = (21, 10)  # free pixel; same 8-cell as the obstacle
    planner = ppv.AVAILABLE_PLANNERS[name](occ, start, goal, grid_size=8, allow_diagonal=True)
    while not planner.done:
        planner.step_once()
    assert planner.found_path
    path = planner.extract_path()
    assert path[0] == start and path[-1] == goal


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_rrt_connect_join_is_collision_free(occ, seed):
    """Regression for the unchecked connection gap: every found RRT-Connect path
    must be fully collision-free, including the segment joining the two trees."""
    planner = ppv.RRTConnectPlanner(occ, START, GOAL, step_size=12, max_iters=8000, seed=seed)
    run(planner)
    if not planner.found_path:
        pytest.skip("RRT-Connect did not connect for this seed")
    path = planner.extract_path()
    assert segments_collision_free(path, occ)
    assert path[0] == START and path[-1] == GOAL


@pytest.mark.parametrize("name", ["RRT", "RRT-Connect", "BiTRRT", "RRT*"])
def test_seeded_runs_are_deterministic(name, occ):
    """Same seed must produce the same path (guards reproducibility and the
    spatial-index nearest-neighbor, which is exact-equivalent to the old scan)."""
    p1 = run(build(name, occ))
    p2 = run(build(name, occ))
    assert p1.found_path == p2.found_path
    assert p1.extract_path() == p2.extract_path()
