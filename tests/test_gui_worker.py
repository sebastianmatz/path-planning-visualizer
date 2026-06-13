"""Tests for the off-thread planner builder (PlannerBuilder).

Drives a real QThread and spins the Qt event loop, so it constructs a
QApplication. Headless via QT_QPA_PLATFORM=offscreen (set in CI).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication

import path_planning_visualizer as ppv
from path_planning_visualizer.gui.worker import PlannerBuilder


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _spin_until(app, predicate, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)
    return predicate()


def test_planner_builder_emits_result_offthread(qapp):
    occ = np.zeros((50, 50), dtype=bool)
    occ[10:40, 25] = True
    captured = {}

    def build():
        return ppv.RRTPlanner(occ, (5, 25), (45, 25), seed=1)

    builder = PlannerBuilder(build)
    builder.result_ready.connect(lambda pl: captured.__setitem__("planner", pl))
    builder.build_failed.connect(lambda msg: captured.__setitem__("error", msg))
    builder.start()

    assert _spin_until(qapp, lambda: "planner" in captured or "error" in captured)
    builder.wait(2000)
    assert "error" not in captured, captured.get("error")
    assert isinstance(captured["planner"], ppv.RRTPlanner)


def test_planner_builder_reports_failure(qapp):
    captured = {}

    def build():
        raise ValueError("boom")

    builder = PlannerBuilder(build)
    builder.result_ready.connect(lambda pl: captured.__setitem__("planner", pl))
    builder.build_failed.connect(lambda msg: captured.__setitem__("error", msg))
    builder.start()

    assert _spin_until(qapp, lambda: "planner" in captured or "error" in captured)
    builder.wait(2000)
    assert captured.get("error") == "boom"
