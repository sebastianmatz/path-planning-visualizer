"""A small dialog to pick one of the bundled example maps.

Kept separate from ``main_window`` so the planners and headless benchmark still
import without PyQt6. Maps are resolved through ``resources`` so this works from a
frozen ``.exe`` (the assets are unpacked under ``sys._MEIPASS``) as well as source.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..resources import asset_path, list_maps

_THUMB = 160  # thumbnail edge in px


class MapPickerDialog(QDialog):
    """Grid of thumbnail previews of the bundled ``maze*.png`` example maps."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Change Map")
        self.selected_path: Optional[str] = None

        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(QSize(_THUMB, _THUMB))
        self.list.setGridSize(QSize(_THUMB + 28, _THUMB + 46))
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setMovement(QListWidget.Movement.Static)
        self.list.setSpacing(10)
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(True)

        for name in list_maps():
            path = asset_path(name)
            pixmap = QPixmap(path)
            if pixmap.isNull():
                continue
            thumb = pixmap.scaled(
                _THUMB, _THUMB,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label = name[:-4] if name.lower().endswith(".png") else name
            item = QListWidgetItem(QIcon(thumb), label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            item.setToolTip(f"{label}  ({pixmap.width()}×{pixmap.height()})")
            self.list.addItem(item)

        if self.list.count():
            self.list.setCurrentRow(0)
        self.list.itemDoubleClicked.connect(self._accept_item)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Open).setText("Load")
        buttons.accepted.connect(self._accept_current)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.list)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.resize(580, 480)

    def _accept_item(self, item: QListWidgetItem) -> None:
        self.selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _accept_current(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            self._accept_item(item)

    @staticmethod
    def choose(parent: Optional[QWidget] = None) -> Optional[str]:
        """Show the dialog; return the chosen map path, or ``None`` if cancelled/empty."""
        dialog = MapPickerDialog(parent)
        if not dialog.list.count():
            return None
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_path
        return None
