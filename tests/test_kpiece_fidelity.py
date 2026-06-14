"""Paper-fidelity tests for KPIECE (Şucan & Kavraki 2009).

- the cell importance equals log(I)*score / (S*N*C) (p. 6);
- the progress penalty multiplies score by P iff P < 1, else leaves it (Alg 1 l.16-17);
- a cell with 2n=4 axis-neighbours is interior, with fewer it is exterior (p. 4).
"""

from __future__ import annotations

import numpy as np

from path_planning_visualizer.planners.kpiece import KPIECECell, KPIECEPlanner


def _planner(**kw) -> KPIECEPlanner:
    occ = np.zeros((120, 120), dtype=bool)
    return KPIECEPlanner(occ, (10, 10), (110, 110), **kw)


def test_importance_formula():
    p = _planner()
    cell = KPIECECell(coord=(0, 0))
    cell.creation_iteration = 10
    cell.score = 2.0
    cell.selections = 3
    cell.neighbor_count = 2
    cell.coverage = 5.0
    p._compute_importance(cell)
    expected = np.log(10) * 2.0 / (3 * 2 * 5.0)
    assert abs(cell.importance - expected) < 1e-9


def test_progress_penalty_only_below_one():
    p = _planner(progress_alpha=0.1, progress_beta=0.9)
    cell = KPIECECell(coord=(0, 0))

    # P = 0.1 + 0.9*(0/10) = 0.1 < 1  ->  score *= 0.1
    cell.score = 1.0
    p._apply_progress_penalty(cell, coverage_delta=0.0, simulated_distance=10.0)
    assert abs(cell.score - 0.1) < 1e-9

    # P = 0.1 + 0.9*(10/1) = 9.1 >= 1  ->  unchanged
    cell.score = 1.0
    p._apply_progress_penalty(cell, coverage_delta=10.0, simulated_distance=1.0)
    assert abs(cell.score - 1.0) < 1e-9


def test_interior_exterior_uses_2n_rule():
    p = _planner()
    center = (5, 5)
    neighbours = [(4, 5), (6, 5), (5, 4), (5, 6)]
    p.cells = {c: KPIECECell(coord=c) for c in [center, *neighbours]}

    p._recompute_cells({center})
    assert p.cells[center].border is False  # 4 = 2n neighbours -> interior

    del p.cells[(4, 5)]
    p._recompute_cells({center})
    assert p.cells[center].border is True   # 3 < 2n neighbours -> exterior
