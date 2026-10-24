"""Compact Color Mixer; selection and expansion are window-local view choices."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel, QSlider, QPushButton, QToolButton

from .color_mixer import ColorMixer, COLOR_NAMES, AXES
from .edit_controls import EditSlider
from .workspace_widgets import ParameterLabel


class ColorMixerEditor(QWidget):
    gestureStarted = Signal()
    gestureEnded = Signal()
    valuesChanged = Signal(object)
    resetRequested = Signal(object)

    def __init__(self):
        super().__init__()
        self.values = ColorMixer()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = QToolButton()
        self.toggle.setText("Color Mixer")
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        layout.addWidget(self.toggle)
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 4, 0, 4)
        self.color = QComboBox()
        self.color.addItems(COLOR_NAMES)
        body.addWidget(self.color)
        self.sliders, self.labels = [], []
        for axis, limit in zip(AXES, (30, 100, 100)):
            label = ParameterLabel()
            slider = EditSlider(Qt.Orientation.Horizontal)
            slider.setRange(-limit, limit)
            slider.setAccessibleName(f"Color Mixer {axis.title()}")
            slider.sliderPressed.connect(self.gestureStarted)
            slider.sliderReleased.connect(self.gestureEnded)
            slider.valueChanged.connect(lambda value, index=len(self.sliders): self._change(index, value))
            self.labels.append(label)
            self.sliders.append(slider)
            body.addWidget(label)
            body.addWidget(slider)
        self.reset_color = QPushButton("Reset selected color")
        self.reset_all = QPushButton("Reset Color Mixer")
        # Separate rows remain readable in the existing narrow edit panel.
        body.addWidget(self.reset_color)
        body.addWidget(self.reset_all)
        self.reset_color.clicked.connect(self._reset_color)
        self.reset_all.clicked.connect(lambda: self.resetRequested.emit(ColorMixer()))
        note = QLabel("Adjusts matching colors across the whole photo. Lightness is HSL lightness, not exposure or a local mask.")
        self.toggle.setToolTip(note.text())
        layout.addWidget(self.body)
        self.body.hide()
        self.toggle.toggled.connect(self._expand)
        self.color.currentIndexChanged.connect(self._select)
        self.set_values(self.values)

    def _expand(self, expanded):
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)

    def _select(self):
        self.gestureEnded.emit()
        self.set_values(self.values)

    def _change(self, axis, value):
        self.values = self.values.with_value(self.color.currentIndex(), axis, value)
        self.set_values(self.values)
        self.valuesChanged.emit(self.values)

    def _reset_color(self):
        bands = list(self.values.bands)
        bands[self.color.currentIndex()] = (0., 0., 0.)
        self.resetRequested.emit(ColorMixer(bands))

    def set_values(self, values):
        self.values = values
        band = values.bands[self.color.currentIndex()]
        for axis, slider, label, value in zip(AXES, self.sliders, self.labels, band):
            slider.blockSignals(True)
            slider.setValue(round(value))
            slider.blockSignals(False)
            unit = "°" if axis == "hue" else ""
            label.setText(f"{axis.title()}  {value:+g}{unit}")

    def locate(self, color=None):
        if color in tuple(name.lower() for name in COLOR_NAMES):
            self.color.setCurrentIndex(tuple(name.lower() for name in COLOR_NAMES).index(color))
        self.toggle.setChecked(True)
