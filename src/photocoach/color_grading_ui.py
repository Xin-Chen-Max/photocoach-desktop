"""Compact 3-way Color Grading UI; zone selection and expansion are window-local view choices."""
from __future__ import annotations

from dataclasses import replace
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .color_grading import ColorGrading, ColorWheel, WHEEL_FIELDS, WHEEL_NAMES, _hue_to_rgb
from .edit_controls import EditSlider
from .workspace_widgets import ParameterLabel


ZONE_TITLES_EN: dict[str, str] = {
    "shadows": "Shadows",
    "midtones": "Midtones",
    "highlights": "Highlights",
}
ZONE_TITLES_ZH: dict[str, str] = {
    "shadows": "阴影",
    "midtones": "中间调",
    "highlights": "高光",
}

_NEUTRAL_SWATCH_STYLE = "background:#383838; border-radius:2px;"


class ColorGradingEditor(QWidget):
    gestureStarted = Signal()
    gestureEnded = Signal()
    valuesChanged = Signal(object)
    resetRequested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.values = ColorGrading()
        self.current_zone = "shadows"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.toggle = QToolButton()
        self.toggle.setObjectName("subsectionToggle")
        self.toggle.setText("Color Grading")
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setToolTip(
            "3-way display-space Color Grading (Shadows, Midtones, Highlights) applied before Color Mixer and the RGB curve. "
            "Hue/Saturation tints preserve display luma before SDR clipping; Luminance shifts encoded SDR values. "
            "Blend controls zone overlap; Balance shifts the midpoint."
        )
        layout.addWidget(self.toggle)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 4, 0, 4)
        body.setSpacing(4)

        zone_row = QHBoxLayout()
        zone_row.setContentsMargins(0, 0, 0, 0)
        zone_row.setSpacing(4)
        self._zone_group = QButtonGroup(self)
        self._zone_group.setExclusive(True)
        self.zone_buttons: dict[str, QPushButton] = {}
        self.zone_swatches: dict[str, QFrame] = {}

        for zone in WHEEL_NAMES:
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(2)
            btn = QPushButton(ZONE_TITLES_EN[zone])
            btn.setCheckable(True)
            btn.setProperty("compact", True)
            btn.setAccessibleName(f"Color Grading {ZONE_TITLES_EN[zone]}")
            btn.toggled.connect(lambda checked, z=zone: checked and self.select_zone(z))
            self._zone_group.addButton(btn)
            self.zone_buttons[zone] = btn

            swatch = QFrame()
            swatch.setFixedHeight(4)
            swatch.setStyleSheet(_NEUTRAL_SWATCH_STYLE)
            self.zone_swatches[zone] = swatch

            col.addWidget(btn)
            col.addWidget(swatch)
            zone_row.addLayout(col, 1)

        blocked = self.zone_buttons[self.current_zone].blockSignals(True)
        self.zone_buttons[self.current_zone].setChecked(True)
        self.zone_buttons[self.current_zone].blockSignals(blocked)
        body.addLayout(zone_row)

        self.sliders: dict[str, EditSlider] = {}
        self.labels: dict[str, ParameterLabel] = {}

        zone_specs = (
            ("hue", 0, 359, "Hue angle (0° to 359°) for the active zone. Visible when Saturation is above 0%."),
            ("saturation", 0, 100, "Tint strength (0% to 100%) for the active zone. 0% leaves hue neutral."),
            ("luminance", -100, 100, "Encoded display luminance shift (-100 to +100) for the active zone, not linear RAW exposure."),
        )
        for key, low, high, tooltip in zone_specs:
            self._add_slider(body, key, low, high, tooltip)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color:#383838;")
        body.addWidget(divider)

        shared_specs = (
            ("blend", 0, 100, "Overlap width between Shadows, Midtones, and Highlights (0% to 100%, neutral 50%)."),
            ("balance", -100, 100, "Shifts zone weighting toward Highlights (+) or Shadows (-); 0 is neutral."),
        )
        for key, low, high, tooltip in shared_specs:
            self._add_slider(body, key, low, high, tooltip)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 2, 0, 0)
        actions.setSpacing(6)
        self.reset_zone = QPushButton("Reset zone")
        self.reset_zone.setProperty("compact", True)
        self.reset_zone.setToolTip("Reset the selected zone's Hue, Saturation, and Luminance to neutral. One Undo step.")
        self.reset_zone.clicked.connect(self._reset_zone)

        self.reset_all = QPushButton("Reset Color Grading")
        self.reset_all.setProperty("compact", True)
        self.reset_all.setToolTip("Reset all three zones, Blend, and Balance to neutral. One Undo step.")
        self.reset_all.clicked.connect(lambda: self.resetRequested.emit(ColorGrading()))

        actions.addWidget(self.reset_zone, 1)
        actions.addWidget(self.reset_all, 1)
        body.addLayout(actions)

        layout.addWidget(self.body)
        self.body.hide()
        self.toggle.toggled.connect(self._expand)
        self.set_values(self.values)

    def _add_slider(self, parent_layout: QVBoxLayout, key: str, low: int, high: int, tooltip: str) -> None:
        label = ParameterLabel()
        slider = EditSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setAccessibleName(f"Color Grading {key.title()}")
        slider.setToolTip(tooltip)
        slider.sliderPressed.connect(self.gestureStarted)
        slider.sliderReleased.connect(self.gestureEnded)
        slider.valueChanged.connect(lambda value, name=key: self._change(name, value))
        self.labels[key] = label
        self.sliders[key] = slider
        parent_layout.addWidget(label)
        parent_layout.addWidget(slider)

    def _expand(self, expanded: bool) -> None:
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)

    def select_zone(self, zone: str, *, emit_gesture_end: bool = True) -> None:
        if zone not in WHEEL_NAMES:
            return
        if emit_gesture_end:
            self.gestureEnded.emit()
        self.current_zone = zone
        btn = self.zone_buttons[zone]
        blocked = btn.blockSignals(True)
        btn.setChecked(True)
        btn.blockSignals(blocked)
        self.set_values(self.values)

    def _change(self, field: str, raw_value: int) -> None:
        if field == "hue":
            wheel = replace(getattr(self.values, self.current_zone), hue=float(raw_value % 360))
            updated = self.values.with_wheel(self.current_zone, wheel)
        elif field == "saturation":
            wheel = replace(getattr(self.values, self.current_zone), saturation=raw_value / 100.0)
            updated = self.values.with_wheel(self.current_zone, wheel)
        elif field == "luminance":
            wheel = replace(getattr(self.values, self.current_zone), luminance=raw_value / 100.0)
            updated = self.values.with_wheel(self.current_zone, wheel)
        elif field == "blend":
            updated = replace(self.values, blend=raw_value / 100.0)
        elif field == "balance":
            updated = replace(self.values, balance=raw_value / 100.0)
        else:
            return
        self.set_values(updated)
        self.valuesChanged.emit(self.values)

    def _reset_zone(self) -> None:
        self.resetRequested.emit(self.values.with_wheel(self.current_zone, ColorWheel()))

    def _update_swatches(self, values: ColorGrading) -> None:
        for zone in WHEEL_NAMES:
            wheel = getattr(values, zone)
            swatch = self.zone_swatches[zone]
            if wheel.is_identity:
                swatch.setStyleSheet(_NEUTRAL_SWATCH_STYLE)
                continue
            base = max(0.15, min(0.85, 0.45 + 0.25 * wheel.luminance))
            if wheel.saturation > 0.0:
                cr, cg, cb = _hue_to_rgb(wheel.hue)
                mix = 0.75 * wheel.saturation
                r_f = (1.0 - mix) * base + mix * cr
                g_f = (1.0 - mix) * base + mix * cg
                b_f = (1.0 - mix) * base + mix * cb
            else:
                r_f = g_f = b_f = base
            r = max(0, min(255, round(r_f * 255)))
            g = max(0, min(255, round(g_f * 255)))
            b = max(0, min(255, round(b_f * 255)))
            swatch.setStyleSheet(f"background:rgb({r},{g},{b}); border-radius:2px;")

    def set_values(self, values: ColorGrading) -> None:
        self.values = values
        wheel = getattr(values, self.current_zone)
        hue_i = min(359, max(0, round(wheel.hue)))
        sat_i = round(wheel.saturation * 100)
        lum_i = round(wheel.luminance * 100)
        blend_i = round(values.blend * 100)
        bal_i = round(values.balance * 100)

        mapped = {
            "hue": (hue_i, f"Hue  {hue_i:d}°"),
            "saturation": (sat_i, f"Saturation  {sat_i:d}%"),
            "luminance": (lum_i, f"Luminance  {lum_i:+d}"),
            "blend": (blend_i, f"Blend  {blend_i:d}%"),
            "balance": (bal_i, f"Balance  {bal_i:+d}"),
        }
        for key, (val_i, text) in mapped.items():
            slider = self.sliders[key]
            blocked = slider.blockSignals(True)
            slider.setValue(val_i)
            slider.blockSignals(blocked)
            self.labels[key].setText(text)
        self._update_swatches(values)

    def locate(self, target: str | None = None) -> tuple[QWidget, QWidget]:
        key = (target or "").strip().lower()
        parts = [p for p in key.split(".") if p]
        zone = parts[0] if parts and parts[0] in WHEEL_NAMES else None
        if zone is not None:
            self.select_zone(zone, emit_gesture_end=False)
        self.toggle.setChecked(True)
        if key in ("blend", "balance"):
            return self.sliders[key], self.labels[key]
        if len(parts) == 2 and zone is not None and parts[1] in WHEEL_FIELDS:
            field = parts[1]
            return self.sliders[field], self.labels[field]
        if zone is not None:
            return self.zone_buttons[zone], self.zone_buttons[zone]
        return self.zone_buttons[self.current_zone], self.toggle
