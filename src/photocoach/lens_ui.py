"""Manual lens controls; all image changes are delegated to the state manager."""
from dataclasses import replace
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from .edit_controls import EditSlider
from .workspace_widgets import ParameterLabel
from .lens import LensCorrections


class LensEditor(QWidget):
    gestureStarted = Signal()
    gestureEnded = Signal()
    valuesChanged = Signal(object)
    resetRequested = Signal(object)

    def __init__(self):
        super().__init__()
        self.values = LensCorrections()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.note = QLabel('Manual correction')
        self.note.setProperty("secondary", True)
        self.note.setWordWrap(True)
        self.note.setToolTip('No lens profile or automatic calibration. Distortion trims the edges with a centered safe scale; crop output dimensions stay unchanged.')
        layout.addWidget(self.note)
        self.sliders, self.labels = {}, {}
        for name, title, limit in (('distortion', 'Distortion', 100), ('vignetting_ev', 'Vignetting', 200)):
            label = ParameterLabel()
            slider = EditSlider(Qt.Orientation.Horizontal)
            slider.setAccessibleName(title)
            slider.setRange(-limit, limit)
            slider.sliderPressed.connect(self.gestureStarted)
            slider.sliderReleased.connect(self.gestureEnded)
            slider.valueChanged.connect(lambda value, name=name: self._change(name, value))
            slider.setToolTip('Negative: barrel correction. Positive: pincushion correction. Resets the photo to Fit.' if name == 'distortion' else
                             'Corner exposure in linear light. Positive brightens corners; negative darkens. Preserves your viewing position.')
            self.sliders[name], self.labels[name] = slider, label
            layout.addWidget(label)
            layout.addWidget(slider)
        self.reset = QPushButton('Reset lens corrections')
        self.reset.setProperty("compact", True)
        self.reset.clicked.connect(lambda: self.resetRequested.emit(LensCorrections()))
        layout.addWidget(self.reset)
        self.set_values(self.values)

    def _change(self, name, value):
        self.set_values(replace(self.values, **{name: value / 100 if name == 'vignetting_ev' else value}))
        self.valuesChanged.emit(self.values)

    def set_values(self, values):
        self.values = values
        for name, slider in self.sliders.items():
            value = getattr(values, name)
            blocked = slider.blockSignals(True)
            slider.setValue(round(value * 100 if name == 'vignetting_ev' else value))
            slider.blockSignals(blocked)
            title = 'Distortion' if name == 'distortion' else 'Vignetting'
            formatted = f'{value:+g}' if name == 'distortion' else f'{value:+.2f} EV'
            self.labels[name].setText(f'{title}  {formatted}')
