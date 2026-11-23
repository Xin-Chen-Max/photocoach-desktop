"""Detail controls UI; all image changes are delegated to the state manager."""
from __future__ import annotations

from dataclasses import replace
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .detail import CAMERA_RAW_DETAIL, DetailControls
from .edit_controls import EditSlider
from .workspace_widgets import ParameterLabel

DETAIL_SLIDERS = (
    ("sharpening", "Sharpening", "Luminance unsharp mask amount (0 to 100). Enhances edge contrast without adding color fringes. Inspect at 100%."),
    ("masking", "Edge masking", "Restricts sharpening to stronger edges (0 to 100). Increase to protect smooth sky, water, or skin from grain amplification."),
    ("luminance_nr", "Luminance NR", "Conservative edge-preserving luma smoothing (0 to 100). Reduces fine sensor grain while retaining structural edges."),
    ("chroma_nr", "Color NR", "Opponent chroma noise reduction (0 to 100). Suppresses Bayer color mottling and shadow color speckles without softening detail."),
)


class DetailEditor(QWidget):
    gestureStarted = Signal()
    gestureEnded = Signal()
    valuesChanged = Signal(object)
    resetRequested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.values = DetailControls()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.note = QLabel("Inspect at 100% · Luminance USM and color NR")
        self.note.setProperty("secondary", True)
        self.note.setWordWrap(True)
        self.note.setToolTip(
            "Detail controls operate on encoded luminance and opponent chroma with 20-pixel strip halo padding so 100% previews and exports match."
        )
        layout.addWidget(self.note)
        self.sliders: dict[str, EditSlider] = {}
        self.labels: dict[str, ParameterLabel] = {}
        for name, title, tooltip in DETAIL_SLIDERS:
            label = ParameterLabel()
            slider = EditSlider(Qt.Orientation.Horizontal)
            slider.setAccessibleName(title)
            slider.setRange(0, 100)
            slider.sliderPressed.connect(self.gestureStarted)
            slider.sliderReleased.connect(self.gestureEnded)
            slider.valueChanged.connect(lambda value, name=name: self._change(name, value))
            slider.setToolTip(tooltip)
            self.sliders[name] = slider
            self.labels[name] = label
            layout.addWidget(label)
            layout.addWidget(slider)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.preset = QPushButton("Camera Raw preset")
        self.preset.setProperty("compact", True)
        self.preset.setToolTip(
            "Apply calibrated baseline detail (Sharpening 40, Edge masking 10, Color NR 25). One Undo step."
        )
        self.preset.clicked.connect(lambda: self.resetRequested.emit(CAMERA_RAW_DETAIL))
        self.reset = QPushButton("Reset detail")
        self.reset.setProperty("compact", True)
        self.reset.setToolTip("Reset all detail and noise reduction sliders to 0. One Undo step.")
        self.reset.clicked.connect(lambda: self.resetRequested.emit(DetailControls()))
        actions.addWidget(self.preset, 1)
        actions.addWidget(self.reset, 1)
        layout.addLayout(actions)
        self.set_values(self.values)

    def _change(self, name: str, value: int) -> None:
        self.set_values(replace(self.values, **{name: float(value)}))
        self.valuesChanged.emit(self.values)

    def set_values(self, values: DetailControls) -> None:
        self.values = values
        titles = {name: title for name, title, _ in DETAIL_SLIDERS}
        for name, slider in self.sliders.items():
            value = getattr(values, name)
            blocked = slider.blockSignals(True)
            slider.setValue(round(value))
            slider.blockSignals(blocked)
            self.labels[name].setText(f"{titles[name]}  {round(value):d}")
