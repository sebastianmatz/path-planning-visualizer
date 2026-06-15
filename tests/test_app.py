"""Tests for the top-level GUI exception hook (app.py)."""

from __future__ import annotations

import sys

import pytest
from PyQt6.QtWidgets import QApplication

import path_planning_visualizer.app as app_mod


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_install_excepthook_sets_hook():
    original = sys.excepthook
    try:
        app_mod.install_excepthook()
        assert sys.excepthook is app_mod._excepthook
    finally:
        sys.excepthook = original


def test_excepthook_keyboardinterrupt_skips_dialog(monkeypatch):
    shown = []
    monkeypatch.setattr(app_mod, "_show_error_dialog", lambda *a: shown.append(1))
    app_mod._excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert shown == []  # Ctrl+C should not pop a dialog


def test_excepthook_reports_real_errors(monkeypatch):
    shown = []
    monkeypatch.setattr(app_mod, "_show_error_dialog", lambda *a: shown.append(1))
    app_mod._excepthook(ValueError, ValueError("boom"), None)
    assert shown == [1]


def test_show_error_dialog_builds_and_execs(qapp, monkeypatch):
    calls = []
    # Never actually block on a modal dialog in a headless test.
    monkeypatch.setattr(app_mod.QMessageBox, "exec", lambda self: calls.append(1) or 0)
    app_mod._show_error_dialog(ValueError, ValueError("boom"), None)
    assert calls == [1]
