"""Background worker for off-the-GUI-thread planner construction.

Some planners do heavy work in ``__init__`` (e.g. ``PRM`` materializes free
pixels, ``FMT*`` precomputes neighbours, ``BIT*`` samples a batch). On large
maps that can block the UI for seconds. ``PlannerBuilder`` runs a build callable
on a worker thread and reports the result back to the GUI thread via signals.
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QThread, pyqtSignal


class PlannerBuilder(QThread):
    """Runs a no-arg build callable off the GUI thread.

    Emits ``result_ready(planner)`` on success or ``build_failed(message)`` on
    error. The built planner is a plain Python object (no Qt), so handing it back
    across the thread boundary is safe.
    """

    result_ready = pyqtSignal(object)
    build_failed = pyqtSignal(str)

    def __init__(self, build_fn: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self._build_fn = build_fn

    def run(self) -> None:  # executed on the worker thread
        try:
            planner = self._build_fn()
        except Exception as exc:  # surface the error on the GUI thread
            self.build_failed.emit(str(exc))
            return
        self.result_ready.emit(planner)
