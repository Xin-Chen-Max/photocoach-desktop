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
        self.content.setContentsMargins(0, 2, 0, 6)
        self.content.setSpacing(4)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded=True):
        self.toggle.setChecked(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)


_NEUTRAL_VALUES = {"0", "+0", "-0", "0.00", "+0.00", "-0.00", "+0.00 EV", "0.00 EV", "+0°", "0°", "1.00"}


class ParameterLabel(QWidget):
    """Read-only left label/right value; compatible with controller setText calls."""
    def __init__(self, text=""):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 0)
        self.name = QLabel()
        self.value = QLabel()
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value.setMinimumWidth(64)
        row.addWidget(self.name, 1)
        row.addWidget(self.value)
        self.setText(text)

    def setText(self, text):
        from .i18n import translate_parameter_name
        self._text = text
        match = re.match(r"^(.*?)\s+([+−-]?\d.*)$", text)
        name_text = match[1] if match else text
        val_text = match[2] if match else ""
        self.name.setText(translate_parameter_name(name_text))
        self.value.setText(val_text)
        is_neutral = (val_text in _NEUTRAL_VALUES) or (name_text == "Saturation" and val_text == "1.00") or (name_text == "Feather" and val_text == "50%")
        self.value.setStyleSheet("color:#9a9a9a;" if is_neutral else "color:#9fc6ff;font-weight:600;")
        self.setAccessibleName(text)

    def refresh_language(self):
        if hasattr(self, "_text"):
            self.setText(self._text)

    def text(self):
        return self._text
