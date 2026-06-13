from __future__ import annotations

import sys
import traceback

from PyQt6.QtWidgets import QApplication, QMessageBox

from .gui.main_window import MainWindow


def _show_error_dialog(exc_type, exc_value, exc_tb) -> None:
    """Show an uncaught exception in a modal dialog (no-op if there is no GUI)."""
    app = QApplication.instance()
    if app is None:
        return
    try:
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Unexpected error")
        box.setText(
            "An unexpected error occurred.\n"
            "The application may be in an unstable state; consider restarting it."
        )
        box.setInformativeText(str(exc_value) or exc_type.__name__)
        box.setDetailedText(details)
        box.exec()
    except Exception:
        # The error handler must never become the thing that crashes the app.
        pass


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    """Route uncaught exceptions to stderr and a GUI dialog instead of a silent crash."""
    traceback.print_exception(exc_type, exc_value, exc_tb)
    if issubclass(exc_type, KeyboardInterrupt):
        return  # let Ctrl+C behave normally; no dialog
    _show_error_dialog(exc_type, exc_value, exc_tb)


def install_excepthook() -> None:
    """Install the GUI exception hook.

    Installing a custom ``sys.excepthook`` also stops PyQt from calling
    ``abort()`` on an unhandled exception raised inside a Qt slot, so the window
    survives a stray error long enough to report it.
    """
    sys.excepthook = _excepthook


def main() -> None:
    """Main entry point for the Path Planning Visualizer application.

    Creates and shows the main window, then runs the Qt event loop.
    """
    app = QApplication(sys.argv)
    install_excepthook()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
