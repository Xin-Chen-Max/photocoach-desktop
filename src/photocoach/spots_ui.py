"""Spot removal darkroom UI controls and direct-manipulation canvas overlay."""
from __future__ import annotations

from dataclasses import replace
import math
from uuid import uuid4

from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .edit_controls import EditSlider
from .gradients import source_point
from .i18n import get_language, tr
from .spots import (
    DEFAULT_SPOT_FEATHER,
    DEFAULT_SPOT_OPACITY,
    DEFAULT_SPOT_RADIUS,
    MAX_SPOT_RADIUS,
    MAX_SPOTS,
    MIN_SPOT_RADIUS,
    SpotRemoval,
    suggest_spot_source,
    validate_spots,
)
from .workspace_widgets import ParameterLabel


class SpotEditor(QWidget):
    """Collapsible Spot Removal controls inside the Edit sidebar."""

    selectionChanged = Signal()
    changed = Signal(object)
    gestureStarted = Signal()
    gestureEnded = Signal()
    editModeChanged = Signal(bool)
    toolChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.values: tuple[SpotRemoval, ...] = ()
        self._cap_warning = False
        self._tool_radius = DEFAULT_SPOT_RADIUS
        self._tool_feather = DEFAULT_SPOT_FEATHER
        self._tool_opacity = DEFAULT_SPOT_OPACITY
        self._tool_mode = "heal"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        self.list = QComboBox()
        self.list.setAccessibleName("Selected spot")
        self.count_label = QLabel(f"0 / {MAX_SPOTS}")
        self.count_label.setProperty("secondary", True)
        top_row.addWidget(self.list, 1)
        top_row.addWidget(self.count_label)
        layout.addLayout(top_row)

        switches_row = QHBoxLayout()
        self.edit_on_photo = QCheckBox("Edit on photo")
        self.edit_on_photo.setToolTip(
            "Click dust spots or distractions on the photo to heal or clone. Esc finishes."
        )
        self.enabled = QCheckBox("Enabled")
        self.enabled.setToolTip("Temporarily toggle the selected spot on or off.")
        switches_row.addWidget(self.edit_on_photo)
        switches_row.addWidget(self.enabled)
        layout.addLayout(switches_row)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.mode_heal = QPushButton("Heal")
        self.mode_clone = QPushButton("Clone")
        for btn in (self.mode_heal, self.mode_clone):
            btn.setCheckable(True)
            btn.setProperty("compact", True)
        self.mode_heal.setChecked(True)
        self.mode_heal.setAccessibleName("Heal spot mode")
        self.mode_clone.setAccessibleName("Clone spot mode")
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.mode_heal)
        self.mode_group.addButton(self.mode_clone)
        mode_row.addWidget(self.mode_heal)
        mode_row.addWidget(self.mode_clone)
        layout.addLayout(mode_row)

        self.sliders: dict[str, EditSlider] = {}
        self.labels: dict[str, ParameterLabel] = {}
        specs = (
            ("radius", 3, 150, round(DEFAULT_SPOT_RADIUS * 1000), 2, 10,
             "Spot size relative to the shorter image edge (0.3% to 15.0%)."),
            ("feather", 0, 100, round(DEFAULT_SPOT_FEATHER * 100), 1, 10,
             "Soft transition inside the spot edge: 0% is hard, 100% fades from center."),
            ("opacity", 0, 100, round(DEFAULT_SPOT_OPACITY * 100), 1, 10,
             "Blend strength of the selected spot (0% to 100%)."),
        )
        for key, low, high, default_val, step, page, tip in specs:
            label = ParameterLabel()
            slider = EditSlider(Qt.Orientation.Horizontal)
            slider.setRange(low, high)
            slider.setSingleStep(step)
            slider.setPageStep(page)
            slider.setValue(default_val)
            slider.setToolTip(tip)
            slider.sliderPressed.connect(self._on_slider_pressed)
            slider.sliderReleased.connect(self._on_slider_released)
            slider.valueChanged.connect(lambda val, k=key: self._on_slider_changed(k, val))
            self.labels[key] = label
            self.sliders[key] = slider
            layout.addWidget(label)
            layout.addWidget(slider)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)
        self.delete_button = QPushButton("Delete spot")
        self.reset_button = QPushButton("Reset spots")
        self.delete_button.setProperty("compact", True)
        self.reset_button.setProperty("compact", True)
        actions_row.addWidget(self.delete_button)
        actions_row.addWidget(self.reset_button)
        layout.addLayout(actions_row)

        self.hint = QLabel()
        self.hint.setProperty("secondary", True)
        self.hint.setWordWrap(True)
        self.hint.setMinimumHeight(28)
        layout.addWidget(self.hint)

        self.setToolTip(
            f"Up to {MAX_SPOTS} non-destructive circular heal or clone spots anchored to original image coordinates."
        )

        self.list.currentIndexChanged.connect(self._select)
        self.edit_on_photo.toggled.connect(self._edit_toggled)
        self.enabled.toggled.connect(self._enabled_toggled)
        self.mode_group.buttonToggled.connect(self._mode_toggled)
        self.delete_button.clicked.connect(self._delete)
        self.reset_button.clicked.connect(self._reset_all)
        self.set_values(())

    @property
    def selected(self) -> SpotRemoval | None:
        sid = self.list.currentData()
        if not sid:
            return None
        return next((s for s in self.values if s.id == sid), None)

    @property
    def tool_radius(self) -> float:
        return self._tool_radius

    @property
    def tool_feather(self) -> float:
        return self._tool_feather

    @property
    def tool_opacity(self) -> float:
        return self._tool_opacity

    @property
    def tool_mode(self) -> str:
        return self._tool_mode

    def _edit_toggled(self, checked: bool) -> None:
        self._cap_warning = False
        self._refresh_hint()
        self.editModeChanged.emit(bool(checked))

    def _select(self) -> None:
        self._cap_warning = False
        self.gestureEnded.emit()
        self.set_values(self.values, self.list.currentData())
        self.selectionChanged.emit()

    def _on_slider_pressed(self) -> None:
        if self.selected is not None:
            self.gestureStarted.emit()

    def _on_slider_released(self) -> None:
        if self.selected is not None:
            self.gestureEnded.emit()

    def _on_slider_changed(self, key: str, raw_val: int) -> None:
        self._cap_warning = False
        if key == "radius":
            val = max(MIN_SPOT_RADIUS, min(MAX_SPOT_RADIUS, raw_val / 1000.0))
            self._tool_radius = val
        elif key == "feather":
            val = raw_val / 100.0
            self._tool_feather = val
        else:
            val = raw_val / 100.0
            self._tool_opacity = val
        self._update_label_texts(
            radius_tenths=self.sliders["radius"].value(),
            feather_pct=self.sliders["feather"].value(),
            opacity_pct=self.sliders["opacity"].value(),
        )
        current = self.selected
        if current is None:
            self.toolChanged.emit()
            return
        updated = replace(current, **{key: val})
        new_values = validate_spots(tuple(updated if s.id == current.id else s for s in self.values))
        self.changed.emit(new_values)

    def _enabled_toggled(self, checked: bool) -> None:
        self._cap_warning = False
        current = self.selected
        if current is None or current.enabled == bool(checked):
            return
        updated = replace(current, enabled=bool(checked))
        self.changed.emit(validate_spots(tuple(updated if s.id == current.id else s for s in self.values)))

    def _mode_toggled(self, button, checked: bool) -> None:
        if not checked:
            return
        self._cap_warning = False
        mode = "clone" if button is self.mode_clone else "heal"
        self._tool_mode = mode
        current = self.selected
        if current is None:
            self.toolChanged.emit()
            return
        if current.mode != mode:
            updated = replace(current, mode=mode)
            self.changed.emit(validate_spots(tuple(updated if s.id == current.id else s for s in self.values)))

    def _delete(self) -> None:
        self._cap_warning = False
        current = self.selected
        if current is not None:
            self.changed.emit(tuple(s for s in self.values if s.id != current.id))

    def _reset_all(self) -> None:
        self._cap_warning = False
        if self.values:
            self.changed.emit(())

    def show_cap_warning(self) -> None:
        self._cap_warning = True
        self.hint.setText(tr(
            f"Maximum {MAX_SPOTS} spots reached. Delete or reset spots to add more.",
            f"最多支持 {MAX_SPOTS} 个污点。请删除部分污点或重置后再添加。",
        ))

    def _update_label_texts(self, *, radius_tenths: int, feather_pct: int, opacity_pct: int) -> None:
        self.labels["radius"].setText(f"Spot size  {radius_tenths / 10:.1f}%")
        self.labels["feather"].setText(f"Feather  {feather_pct}%")
        self.labels["opacity"].setText(f"Opacity  {opacity_pct}%")

    def _refresh_hint(self) -> None:
        if self._cap_warning:
            return
        if self.edit_on_photo.isChecked():
            if self.selected is not None:
                self.hint.setText(tr(
                    "Click photo to add a spot; drag target, source, or edge handle. Del deletes; Esc finishes.",
                    "点击画面添加污点；拖动目标圈、采样源圈或边缘手柄调整。Del 删除，Esc 退出。",
                ))
            else:
                self.hint.setText(tr(
                    "Click on any dust spot or blemish in the photo to heal or clone it. Space + drag pans.",
                    "点击画面中的灰尘或瑕疵即可修复或仿制；空格 + 拖动平移画布。",
                ))
        else:
            self.hint.setText(tr(
                "Enable Edit on photo, then click dust spots or distractions on the canvas.",
                "勾选“在画面上编辑”，然后点击画布上的灰尘或杂物即可修补。",
            ))

    def set_values(self, values: tuple[SpotRemoval, ...], selected_id: str | None = None) -> None:
        prev_idx = self.list.currentIndex()
        prev_id = selected_id if selected_id is not None else self.list.currentData()
        self.values = tuple(values)
        self.count_label.setText(f"{len(self.values)} / {MAX_SPOTS}")

        self.list.blockSignals(True)
        self.list.clear()
        if not self.values:
            self.list.addItem(tr("No spots placed", "尚未添加污点"), None)
            self.list.setCurrentIndex(0)
            self.list.setEnabled(False)
        else:
            for idx, spot in enumerate(self.values, start=1):
                mode_text = tr("Heal", "修复") if spot.mode == "heal" else tr("Clone", "仿制")
                off_text = tr(" · off", " · 禁用") if not spot.enabled else ""
                label = f"#{idx} · {mode_text} · {spot.radius * 100:.1f}%{off_text}"
                self.list.addItem(label, spot.id)
                self.list.setItemData(self.list.count() - 1, label, Qt.ItemDataRole.ToolTipRole)
            self.list.setEnabled(True)
            index = self.list.findData(prev_id) if prev_id else -1
            if index < 0:
                index = max(0, min(prev_idx, len(self.values) - 1)) if prev_idx >= 0 else 0
            self.list.setCurrentIndex(index)
        self.list.blockSignals(False)

        current = self.selected
        self.delete_button.setEnabled(current is not None)
        self.reset_button.setEnabled(bool(self.values))

        self.enabled.blockSignals(True)
        self.enabled.setEnabled(current is not None)
        self.enabled.setChecked(bool(current and current.enabled))
        self.enabled.blockSignals(False)

        mode = current.mode if current is not None else self._tool_mode
        self.mode_heal.blockSignals(True)
        self.mode_clone.blockSignals(True)
        self.mode_heal.setChecked(mode == "heal")
        self.mode_clone.setChecked(mode == "clone")
        self.mode_heal.blockSignals(False)
        self.mode_clone.blockSignals(False)

        r_val = round((current.radius if current is not None else self._tool_radius) * 1000)
        f_val = round((current.feather if current is not None else self._tool_feather) * 100)
        o_val = round((current.opacity if current is not None else self._tool_opacity) * 100)
        for key, slider, val in (
            ("radius", self.sliders["radius"], r_val),
            ("feather", self.sliders["feather"], f_val),
            ("opacity", self.sliders["opacity"], o_val),
        ):
            slider.blockSignals(True)
            slider.setValue(val)
            slider.blockSignals(False)
        self._update_label_texts(radius_tenths=r_val, feather_pct=f_val, opacity_pct=o_val)
        self._refresh_hint()

    def set_language(self, lang: str) -> None:
        zh = str(lang).lower().startswith("zh")
        self.edit_on_photo.setText("在画面上编辑" if zh else "Edit on photo")
        self.enabled.setText("启用" if zh else "Enabled")
        self.mode_heal.setText("修复 Heal" if zh else "Heal")
        self.mode_clone.setText("仿制 Clone" if zh else "Clone")
        self.delete_button.setText("删除污点" if zh else "Delete spot")
        self.reset_button.setText("重置污点" if zh else "Reset spots")
        self.sliders["radius"].setToolTip(
            "污点半径相对于图像短边的比例（0.3% 至 15.0%）。"
            if zh
            else "Spot size relative to the shorter image edge (0.3% to 15.0%)."
        )
        self.sliders["feather"].setToolTip(
            "污点边缘羽化过渡：0% 为硬边，100% 从中心渐变羽化。"
            if zh
            else "Soft transition inside the spot edge: 0% is hard, 100% fades from center."
        )
        self.sliders["opacity"].setToolTip(
            "当前选中污点的混合不透明度（0% 至 100%）。"
            if zh
            else "Blend strength of the selected spot (0% to 100%)."
        )
        current = self.selected
        self.set_values(self.values, current.id if current is not None else None)


class SpotInteraction(QObject):
    """Direct manipulation of circular heal/clone spots on ImageView."""

    def __init__(self, view, controller: SpotController) -> None:
        super().__init__(view)
        self.view = view
        self.controller = controller
        self.drag: tuple | None = None
        self.started = False
        self.hover_pos: QPointF | None = None
        view.spot_layer = self
        view.installEventFilter(self)

    def is_active(self) -> bool:
        editor = self.controller.editor
        return bool(editor.isEnabled() and editor.edit_on_photo.isChecked() and all(self.view.source_size))

    def normalized(self, position: QPointF) -> tuple[float, float]:
        view = self.view
        p = source_point(view.source_at(position), view.source_size, view.composition, distortion=view.distortion)
        return tuple(max(0.0, min(1.0, v / size)) for v, size in zip(p, view.source_size))

    def canvas_point(self, point: tuple[float, float]):
        view = self.view
        return source_point(
            tuple(v * s for v, s in zip(point, view.source_size)),
            view.source_size,
            view.composition,
            inverse=True,
            distortion=view.distortion,
        )

    def screen_point(self, point: tuple[float, float]) -> QPointF | None:
        mapped = self.canvas_point(point)
        if mapped is None:
            return None
        x, y = mapped
        view = self.view
        scale = view.zoom / view.devicePixelRatioF()
        return QPointF(
            view.width() / 2 + (x - view.center[0]) * scale,
            view.height() / 2 + (y - view.center[1]) * scale,
        )

    def resize_handle_norm(self, spot: SpotRemoval) -> tuple[float, float]:
        w, h = self.view.source_size
        if w < 1 or h < 1:
            return spot.target
        r_px = spot.radius * min(w, h)
        dx = r_px / w
        tx, ty = spot.target
        hx = tx + dx if tx + dx <= 0.99 else max(0.0, tx - dx)
        return (hx, ty)

    def eventFilter(self, watched, event) -> bool:
        try:
            return self._handle_event(watched, event)
        except RuntimeError:
            return False

    def _handle_event(self, watched, event) -> bool:
        view = self.view
        editor = self.controller.editor
        kind = event.type()

        if kind == QEvent.Type.Leave:
            self.hover_pos = None
            view.update()
        if kind in (QEvent.Type.Hide, QEvent.Type.FocusOut):
            self.hover_pos = None
            self.finish()
        if kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            if self.drag is not None or self.is_active():
                self.finish(cancel=True)
                self.hover_pos = None
                editor.edit_on_photo.setChecked(False)
                return True
        if self.drag is not None and (
            kind == QEvent.Type.Wheel
            or (kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space)
        ):
            self.finish()
        if not self.is_active():
            return False
        if kind == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.drag is None and editor.selected is not None:
                editor._delete()
                return True
        if kind == QEvent.Type.MouseMove and self.drag is None:
            self.hover_pos = event.position()
            view.update()
            return False
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton and not view._space:
            composed = view.source_at(event.position())
            if not (0 <= composed[0] <= view.full_size[0] and 0 <= composed[1] <= view.full_size[1]):
                return False
            scale = view.zoom / view.devicePixelRatioF()
            ref = float(min(view.source_size))
            pos = event.position()
            selected = editor.selected

            if selected is not None:
                r_scr = selected.radius * ref * scale
                hit_r = max(12.0, min(48.0, r_scr + 4.0))
                handle_scr = self.screen_point(self.resize_handle_norm(selected))
                tgt_scr = self.screen_point(selected.target)
                src_scr = self.screen_point(selected.source)
                if handle_scr is not None:
                    d_handle = math.hypot(pos.x() - handle_scr.x(), pos.y() - handle_scr.y())
                    d_tgt = (
                        math.hypot(pos.x() - tgt_scr.x(), pos.y() - tgt_scr.y())
                        if tgt_scr is not None
                        else math.inf
                    )
                    if d_handle <= 10.0 and (r_scr >= 10.0 or d_handle + 2.0 < d_tgt):
                        self.drag = ("resize", selected.id, pos, selected)
                        self.started = False
                        self.controller.selected_id = selected.id
                        view.setFocus()
                        return True
                if src_scr is not None and math.hypot(pos.x() - src_scr.x(), pos.y() - src_scr.y()) <= hit_r:
                    self.drag = ("source", selected.id, pos, selected)
                    self.started = False
                    self.controller.selected_id = selected.id
                    view.setFocus()
                    return True
                if tgt_scr is not None and math.hypot(pos.x() - tgt_scr.x(), pos.y() - tgt_scr.y()) <= hit_r:
                    self.drag = ("target", selected.id, pos, selected)
                    self.started = False
                    self.controller.selected_id = selected.id
                    view.setFocus()
                    return True

            for other in reversed(editor.values):
                if selected is not None and other.id == selected.id:
                    continue
                other_scr = self.screen_point(other.target)
                other_hit = max(12.0, min(48.0, other.radius * ref * scale + 4.0))
                if other_scr is not None and math.hypot(pos.x() - other_scr.x(), pos.y() - other_scr.y()) <= other_hit:
                    self.controller.selected_id = other.id
                    self.drag = ("target", other.id, pos, other)
                    self.started = False
                    editor.set_values(editor.values, other.id)
                    view.setFocus()
                    view.update()
                    return True

            if len(editor.values) >= MAX_SPOTS:
                editor.show_cap_warning()
                return True

            target = self.normalized(pos)
            radius = editor.tool_radius
            source = suggest_spot_source(target, radius, view.source_size)
            spot = SpotRemoval(
                id=uuid4().hex[:12],
                target=target,
                source=source,
                radius=radius,
                feather=editor.tool_feather,
                mode=editor.tool_mode,
                opacity=editor.tool_opacity,
                enabled=True,
            )
            self.controller.selected_id = spot.id
            self.drag = ("new", spot.id, pos, spot)
            self.controller.window._begin_parameter_gesture("spots")
            self.started = True
            self.controller.change((*editor.values, spot))
            view.setFocus()
            view.update()
            return True

        if kind == QEvent.Type.MouseMove and self.drag is not None:
            mode, spot_id, origin_pos, base_spot = self.drag
            self.hover_pos = event.position()
            current = next((s for s in editor.values if s.id == spot_id), None)
            if current is None:
                return True
            p = self.normalized(event.position())
            if mode == "new":
                if p == current.target:
                    return True
                updated = replace(
                    current,
                    target=p,
                    source=suggest_spot_source(p, current.radius, view.source_size),
                )
            elif mode == "target":
                if p == current.target:
                    return True
                updated = replace(current, target=p)
            elif mode == "source":
                if p == current.source:
                    return True
                updated = replace(current, source=p)
            else:
                w, h = view.source_size
                ref = float(min(w, h))
                dist_px = math.hypot((p[0] - base_spot.target[0]) * w, (p[1] - base_spot.target[1]) * h)
                new_radius = max(MIN_SPOT_RADIUS, min(MAX_SPOT_RADIUS, dist_px / ref))
                if abs(new_radius - current.radius) < 1e-6:
                    return True
                updated = replace(current, radius=new_radius)
            if not self.started:
                self.controller.window._begin_parameter_gesture("spots")
                self.started = True
            self.controller.selected_id = spot_id
            self.controller.change(tuple(updated if s.id == spot_id else s for s in editor.values))
            return True

        if kind == QEvent.Type.MouseButtonRelease and self.drag is not None:
            self.finish()
            return True
        return False

    def finish(self, cancel: bool = False) -> None:
        if self.drag is None:
            return
        self.drag = None
        if self.started:
            self.started = False
            self.controller.end_geometry(cancel=cancel)
        self.view.update()

    def _draw_circle(self, painter, center_norm: tuple[float, float], radius_px: float,
                     pen: QPen, *, brush=Qt.BrushStyle.NoBrush) -> None:
        view = self.view
        painter.setPen(pen)
        painter.setBrush(brush)
        if view.distortion:
            w, h = view.source_size
            pts = []
            for k in range(33):
                th = 2.0 * math.pi * k / 32.0
                pt = self.canvas_point((
                    center_norm[0] + (radius_px * math.cos(th)) / w,
                    center_norm[1] + (radius_px * math.sin(th)) / h,
                ))
                if pt is not None:
                    pts.append(QPointF(*pt))
            if len(pts) >= 2:
                path = QPainterPath(pts[0])
                for pt in pts[1:]:
                    path.lineTo(pt)
                painter.drawPath(path)
            return
        center = self.canvas_point(center_norm)
        if center is not None:
            painter.drawEllipse(QPointF(*center), radius_px, radius_px)

    def paint(self, painter) -> None:
        if not self.is_active():
            return
        view = self.view
        editor = self.controller.editor
        scale = view.zoom / view.devicePixelRatioF()
        ref = float(min(view.source_size))
        selected = editor.selected

        painter.save()
        unselected_pen = QPen(QColor(159, 198, 255, 130), 1.0)
        unselected_pen.setCosmetic(True)
        for spot in editor.values:
            if selected is not None and spot.id == selected.id:
                continue
            self._draw_circle(painter, spot.target, spot.radius * ref, unselected_pen)

        if selected is not None:
            r_px = selected.radius * ref
            ring_color = QColor("#79cfd3") if selected.mode == "heal" else QColor("#f0b876")
            if not selected.enabled:
                ring_color.setAlpha(135)
            tgt_canvas = self.canvas_point(selected.target)
            src_canvas = self.canvas_point(selected.source)

            if tgt_canvas is not None and src_canvas is not None:
                link_pen = QPen(QColor(232, 232, 232, 175), 1.1)
                link_pen.setCosmetic(True)
                painter.setPen(link_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(QPointF(*src_canvas), QPointF(*tgt_canvas))
                dx = tgt_canvas[0] - src_canvas[0]
                dy = tgt_canvas[1] - src_canvas[1]
                dist = math.hypot(dx, dy)
                if dist > r_px + 4.0 / scale:
                    ux, uy = dx / dist, dy / dist
                    tip = QPointF(tgt_canvas[0] - ux * r_px, tgt_canvas[1] - uy * r_px)
                    wing = 6.0 / scale
                    px, py = -uy, ux
                    p1 = QPointF(tip.x() - ux * wing + px * wing * 0.55, tip.y() - uy * wing + py * wing * 0.55)
                    p2 = QPointF(tip.x() - ux * wing - px * wing * 0.55, tip.y() - uy * wing - py * wing * 0.55)
                    painter.drawLine(tip, p1)
                    painter.drawLine(tip, p2)

            src_pen = QPen(QColor(232, 232, 232, 210), 1.25, Qt.PenStyle.DashLine)
            src_pen.setCosmetic(True)
            self._draw_circle(painter, selected.source, r_px, src_pen)
            if src_canvas is not None:
                painter.setPen(src_pen)
                painter.setBrush(QColor("#e8e8e8"))
                painter.drawEllipse(QPointF(*src_canvas), 2.5 / scale, 2.5 / scale)

            tgt_pen = QPen(ring_color, 1.5)
            tgt_pen.setCosmetic(True)
            self._draw_circle(painter, selected.target, r_px, tgt_pen)
            inner_factor = 1.0 - selected.feather
            if 0.02 < inner_factor < 0.98:
                feather_pen = QPen(
                    QColor(ring_color.red(), ring_color.green(), ring_color.blue(), 150),
                    1.0,
                    Qt.PenStyle.DotLine,
                )
                feather_pen.setCosmetic(True)
                self._draw_circle(painter, selected.target, r_px * inner_factor, feather_pen)

            handle_canvas = self.canvas_point(self.resize_handle_norm(selected))
            if handle_canvas is not None:
                painter.setPen(tgt_pen)
                painter.setBrush(QColor("#202020"))
                painter.drawEllipse(QPointF(*handle_canvas), 4.5 / scale, 4.5 / scale)

        if self.hover_pos is not None and self.drag is None and not view._space:
            composed = view.source_at(self.hover_pos)
            if 0 <= composed[0] <= view.full_size[0] and 0 <= composed[1] <= view.full_size[1]:
                hover_color = QColor("#79cfd3") if editor.tool_mode == "heal" else QColor("#f0b876")
                hover_color.setAlpha(155)
                hover_pen = QPen(hover_color, 1.1)
                hover_pen.setCosmetic(True)
                hover_norm = self.normalized(self.hover_pos)
                self._draw_circle(painter, hover_norm, editor.tool_radius * ref, hover_pen)
                painter.setPen(hover_pen)
                painter.setBrush(hover_color)
                painter.drawEllipse(QPointF(*composed), 2.0 / scale, 2.0 / scale)

        painter.restore()


class SpotController:
    """Connects SpotEditor controls, SpotInteraction overlay, and MainWindow state."""

    def __init__(self, window, editor: SpotEditor) -> None:
        self.window = window
        self.editor = editor
        self.selected_id: str | None = None
        self.layer = SpotInteraction(window.preview, self)
        editor.changed.connect(self.change)
        editor.gestureStarted.connect(lambda: window._begin_parameter_gesture("spots"))
        editor.gestureEnded.connect(window._finish_gesture)
        editor.selectionChanged.connect(self.selection_changed)
        editor.editModeChanged.connect(self.edit_mode_changed)
        editor.toolChanged.connect(self._tool_changed)

    def _tool_changed(self) -> None:
        self.window.preview.update()

    def sync(self) -> None:
        state = self.window.store.current
        spots = state.spots if state else ()
        if self.selected_id and not any(s.id == self.selected_id for s in spots):
            self.selected_id = None
        self.editor.set_values(spots, self.selected_id)
        if self.editor.selected is not None:
            self.selected_id = self.editor.selected.id
        self.window.preview.update()

    def change(self, values: tuple[SpotRemoval, ...]) -> None:
        self.window.change_parameter("spots", values)
        self.sync()

    def selection_changed(self) -> None:
        self.layer.finish()
        current = self.editor.selected
        self.selected_id = current.id if current is not None else None
        self.window.preview.update()

    def edit_mode_changed(self, enabled: bool) -> None:
        if enabled:
            self.window.pick_wb.setChecked(False)
            self.window.local_gradients.edit_on_photo.setChecked(False)
            self.window.preview.setFocus()
            self.window.preview.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.layer.finish()
            self.layer.hover_pos = None
            self.window.preview.setCursor(
                Qt.CursorShape.CrossCursor
                if self.window.preview.picking_neutral
                else Qt.CursorShape.OpenHandCursor
                if self.window.preview._space
                else Qt.CursorShape.ArrowCursor
            )
        self.window._refresh_view_notice()
        self.window.preview.update()

    def end_geometry(self, cancel: bool = False) -> None:
        window = self.window
        window.store.end_drag(cancel=cancel)
        self.sync()
        window._after_edit()
        window._finalize_preview()

    def locate(self) -> None:
        self.window.spots_section.set_expanded(True)
        self.window.tabs.setCurrentIndex(0)

        def reveal() -> None:
            if not self.window._closing:
                self.window.edit_scroll.ensureWidgetVisible(self.editor.reset_button)
                self.window.edit_scroll.ensureWidgetVisible(self.editor.edit_on_photo)
                self.editor.edit_on_photo.setFocus()

        QTimer.singleShot(0, self.window, reveal)


__all__ = ["SpotEditor", "SpotInteraction", "SpotController", "get_language"]
