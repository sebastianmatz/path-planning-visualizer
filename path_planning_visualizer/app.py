from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from .gui.main_window import MainWindow


def main() -> None:
    """Main entry point for the Path Planning Visualizer application.
    
    Creates and shows the main window, then runs the Qt event loop.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
