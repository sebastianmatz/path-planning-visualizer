"""Lock in the GUI/Qt decoupling: importing the package and running a planner
must not pull in PyQt6.

Runs in a subprocess so the check is unaffected by other tests in the session
having already imported PyQt6.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_package_and_planner_do_not_import_pyqt6():
    code = textwrap.dedent(
        """
        import sys
        import numpy as np
        import path_planning_visualizer as ppv

        assert "PyQt6" not in sys.modules, "importing the package loaded PyQt6"

        occ = np.zeros((30, 30), dtype=bool)
        occ[5:25, 15] = True
        p = ppv.AVAILABLE_PLANNERS["RRT"](occ, (2, 2), (27, 27), max_vertices=3000, seed=1)
        while not p.done:
            p.step_once()

        assert "PyQt6" not in sys.modules, "running a planner loaded PyQt6"
        print("OK", p.found_path)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK")


def test_benchmark_imports_without_pyqt6():
    code = textwrap.dedent(
        """
        import sys
        import path_planning_visualizer.benchmark as bm
        assert "PyQt6" not in sys.modules, "importing the benchmark loaded PyQt6"
        bm.benchmark(["A*"], ["open"], seeds=1, max_steps=50000)
        assert "PyQt6" not in sys.modules
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
