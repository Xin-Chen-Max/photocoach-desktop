"""Passive workspace containers; visibility never changes editing state."""
import re
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton, QSizePolicy


class ToolSection(QWidget):
    def __init__(self, title, expanded=False):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.toggle = QToolButton()
        self.toggle.setObjectName("sectionToggle")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.body = QWidget()
        self.content = QVBoxLayout(self.body)
        self.content.setContentsMargins(0, 0, 0, 6)
        self.content.setSpacing(6)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded=True):
        self.toggle.setChecked(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)


class ParameterLabel(QWidget):
    """Read-only left label/right value; compatible with controller setText calls."""
    def __init__(self, text=""):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.name = QLabel()
        self.value = QLabel()
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value.setMinimumWidth(64)
        row.addWidget(self.name, 1)
        row.addWidget(self.value)
        self.setText(text)

    def setText(self, text):
        self._text = text
        match = re.match(r"^(.*?)\s+([+−-]?\d.*)$", text)
        self.name.setText(match[1] if match else text)
        self.value.setText(match[2] if match else "")
        self.setAccessibleName(text)

    def text(self):
        return self._text
