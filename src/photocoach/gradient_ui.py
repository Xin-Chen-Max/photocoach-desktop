"""Local adjustment widgets and state-manager wiring; no image algorithms."""
from dataclasses import replace
from uuid import uuid4
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                               QPushButton, QLineEdit, QCheckBox, QButtonGroup)
from .edit_controls import EditSlider
from .workspace_widgets import ParameterLabel
from .gradients import (LinearGradient, RadialGradient, BrushMask, BrushStroke, OverlaySpec,
                        DEFAULT_BRUSH_RADIUS, DEFAULT_BRUSH_FEATHER, DEFAULT_LUMA_FEATHER, MAX_STROKES_PER_MASK)
from .i18n import tr


class GradientEditor(QWidget):
    selectionChanged = Signal()
    changed = Signal(object)
    gestureStarted = Signal()
    gestureEnded = Signal()
    drawRequested = Signal(bool)
    overlayChanged = Signal()
    brushToolChanged = Signal()

    def __init__(self):
        super().__init__()
        self.values = ()
        self._drawing_brush = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QComboBox()
        self.list.setAccessibleName('Selected local mask')
        self.list.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.list.setMinimumContentsLength(10)
        layout.addWidget(self.list)
        self.new_shape = QComboBox()
        self.new_shape.addItems(('Linear', 'Radial', 'Brush'))
        self.new_shape.setAccessibleName('New mask shape')
        self.new_shape.setToolTip('Shape for the next new mask. Existing masks are unchanged.')
        self.new_mask_label = QLabel('New mask')
        creation = QHBoxLayout()
        creation.addWidget(self.new_mask_label)
        creation.addWidget(self.new_shape, 1)
        layout.addLayout(creation)
        row = QHBoxLayout()
        self.add = QPushButton('Add mask')
        self.done = QPushButton('Done')
        self.add.setProperty('compact', True)
        self.done.setProperty('compact', True)
        row.addWidget(self.add)
        row.addWidget(self.done)
        layout.addLayout(row)

        self.brush_panel = QWidget()
        brush_layout = QVBoxLayout(self.brush_panel)
        brush_layout.setContentsMargins(0, 2, 0, 2)
        brush_mode_row = QHBoxLayout()
        self.brush_paint = QPushButton('Paint')
        self.brush_erase = QPushButton('Erase')
        for btn in (self.brush_paint, self.brush_erase):
            btn.setCheckable(True)
            btn.setProperty('compact', True)
        self.brush_paint.setChecked(True)
        self.brush_paint.setAccessibleName('Brush paint mode')
        self.brush_erase.setAccessibleName('Brush erase mode')
        self.brush_mode_group = QButtonGroup(self)
        self.brush_mode_group.setExclusive(True)
        self.brush_mode_group.addButton(self.brush_paint)
        self.brush_mode_group.addButton(self.brush_erase)
        brush_mode_row.addWidget(self.brush_paint)
        brush_mode_row.addWidget(self.brush_erase)
        brush_layout.addLayout(brush_mode_row)
        self.brush_size_label = ParameterLabel()
        self.brush_size = EditSlider(Qt.Orientation.Horizontal)
        self.brush_size.setRange(5, 350)
        self.brush_size.setSingleStep(5)
        self.brush_size.setPageStep(20)
        self.brush_size.setValue(round(DEFAULT_BRUSH_RADIUS * 1000))
        self.brush_size.setAccessibleName('Brush size')
        self.brush_size.setToolTip('Brush diameter relative to the shorter image edge (0.5% to 35%).')
        brush_layout.addWidget(self.brush_size_label)
        brush_layout.addWidget(self.brush_size)
        self.brush_feather_label = ParameterLabel()
        self.brush_feather = EditSlider(Qt.Orientation.Horizontal)
        self.brush_feather.setRange(0, 100)
        self.brush_feather.setValue(round(DEFAULT_BRUSH_FEATHER * 100))
        self.brush_feather.setAccessibleName('Brush feather')
        self.brush_feather.setToolTip('Soft transition inside the brush edge: 0% is hard, 100% fades from center.')
        brush_layout.addWidget(self.brush_feather_label)
        brush_layout.addWidget(self.brush_feather)
        self.brush_panel.setVisible(False)
        layout.addWidget(self.brush_panel)

        self.controls = QWidget()
        body = QVBoxLayout(self.controls)
        body.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit()
        self.name.setMaxLength(40)
        self.name.setPlaceholderText('Gradient name')
        self.name.setAccessibleName('Gradient name')
        body.addWidget(self.name)
        switches = QHBoxLayout()
        self.enabled = QCheckBox('Enabled')
        self.inverted = QCheckBox('Invert')
        switches.addWidget(self.enabled)
        switches.addWidget(self.inverted)
        body.addLayout(switches)
        self.sliders = {}
        self.labels = {}
        for name, title, limit in (('exposure_ev', 'Exposure', 300), ('highlights', 'Highlights', 100), ('shadows', 'Shadows', 100), ('warmth', 'Warmth', 100)):
            label = ParameterLabel()
            slider = EditSlider(Qt.Orientation.Horizontal)
            slider.setRange(-limit, limit)
            slider.setAccessibleName(f'Local gradient {title}')
            slider.sliderPressed.connect(self.gestureStarted)
            slider.sliderReleased.connect(self.gestureEnded)
            slider.valueChanged.connect(lambda value, key=name: self._change(**{key: value / 100}))
            self.sliders[name] = slider
            self.labels[name] = label
            body.addWidget(label)
            body.addWidget(slider)
        self.feather_label = ParameterLabel()
        self.feather = EditSlider(Qt.Orientation.Horizontal)
        self.feather.setRange(0, 100)
        self.feather.setAccessibleName('Radial feather')
        self.feather.setToolTip('Soft transition inside the ellipse: 0% is a hard edge, 100% fades from the center.')
        self.feather.sliderPressed.connect(self.gestureStarted)
        self.feather.sliderReleased.connect(self.gestureEnded)
        self.feather.valueChanged.connect(lambda value: self._change(feather=value / 100))
        body.addWidget(self.feather_label)
        body.addWidget(self.feather)
        self.angle_label = ParameterLabel()
        self.angle = EditSlider(Qt.Orientation.Horizontal)
        self.angle.setRange(-90, 90)
        self.angle.setAccessibleName('Radial angle')
        self.angle.setToolTip('Tilt angle of the radial light beam (-90° to +90°).')
        self.angle.sliderPressed.connect(self.gestureStarted)
        self.angle.sliderReleased.connect(self.gestureEnded)
        self.angle.valueChanged.connect(lambda value: self._change(angle=float(value)))
        body.addWidget(self.angle_label)
        body.addWidget(self.angle)
        self.edge_aware = QCheckBox('Edge-aware')
        self.edge_aware.setToolTip('Snap mask softly to image luminance edges to prevent sky halos.')
        body.addWidget(self.edge_aware)
        self.luma_min_label = ParameterLabel()
        self.luma_min = EditSlider(Qt.Orientation.Horizontal)
        self.luma_min.setRange(0, 100)
        self.luma_min.setAccessibleName('Mask luma minimum')
        self.luma_min.setToolTip('Minimum perceptual luminance targeted by this mask (0% = shadows, 100% = whites).')
        self.luma_min.sliderPressed.connect(self.gestureStarted)
        self.luma_min.sliderReleased.connect(self.gestureEnded)
        self.luma_min.valueChanged.connect(self._on_luma_min_changed)
        body.addWidget(self.luma_min_label)
        body.addWidget(self.luma_min)
        self.luma_max_label = ParameterLabel()
        self.luma_max = EditSlider(Qt.Orientation.Horizontal)
        self.luma_max.setRange(0, 100)
        self.luma_max.setValue(100)
        self.luma_max.setAccessibleName('Mask luma maximum')
        self.luma_max.setToolTip('Maximum perceptual luminance targeted by this mask (lower to exclude bright skies and avoid halos).')
        self.luma_max.sliderPressed.connect(self.gestureStarted)
        self.luma_max.sliderReleased.connect(self.gestureEnded)
        self.luma_max.valueChanged.connect(self._on_luma_max_changed)
        body.addWidget(self.luma_max_label)
        body.addWidget(self.luma_max)
        self.luma_feather_label = ParameterLabel()
        self.luma_feather = EditSlider(Qt.Orientation.Horizontal)
        self.luma_feather.setRange(0, 100)
        self.luma_feather.setValue(round(DEFAULT_LUMA_FEATHER * 100))
        self.luma_feather.setAccessibleName('Mask luma feather')
        self.luma_feather.setToolTip('Soft falloff width around the luminance range boundaries.')
        self.luma_feather.sliderPressed.connect(self.gestureStarted)
        self.luma_feather.sliderReleased.connect(self.gestureEnded)
        self.luma_feather.valueChanged.connect(lambda value: self._change(luma_feather=value / 100))
        body.addWidget(self.luma_feather_label)
        body.addWidget(self.luma_feather)
        self.redraw = QPushButton('Redraw mask')
        self.reset = QPushButton('Reset adjustments')
        self.delete = QPushButton('Delete mask')
        self.redraw.setProperty('compact', True)
        self.delete.setProperty('compact', True)
        self.reset.setProperty('compact', True)
        mask_actions = QHBoxLayout()
        mask_actions.addWidget(self.redraw)
        mask_actions.addWidget(self.delete)
        body.addLayout(mask_actions)
        body.addWidget(self.reset)
        self.edit_on_photo = QCheckBox('Edit on photo')
        self.show_mask = QCheckBox('Show selected mask')
        body.addWidget(self.edit_on_photo)
        body.addWidget(self.show_mask)
        layout.addWidget(self.controls)
        self.hint = QLabel('Add a gradient, then drag from the fully affected side toward the unaffected side.')
        self.hint.setProperty('secondary', True)
        self.hint.setWordWrap(True)
        self.hint.setMinimumHeight(28)
        layout.addWidget(self.hint)
        self.setToolTip('Up to four manual linear, radial, or brush masks, attached to original image coordinates. Exposure adds in stops; overlapping Highlights/Shadows combine with global amounts and are limited to −1…+1. No RAW recovery or automatic subject detection.')
        self.add.clicked.connect(lambda: self.drawRequested.emit(False))
        self.redraw.clicked.connect(lambda: self.drawRequested.emit(True))
        self.done.clicked.connect(self._done)
        self.name.editingFinished.connect(self._rename)
        self.enabled.toggled.connect(lambda v: self._change(enabled=v))
        self.inverted.toggled.connect(lambda v: self._change(inverted=v))
        self.edge_aware.toggled.connect(lambda v: self._change(edge_aware=v))
        self.reset.clicked.connect(lambda: self._change(
            exposure_ev=0., highlights=0., shadows=0., warmth=0.,
            luma_min=0., luma_max=1., luma_feather=DEFAULT_LUMA_FEATHER, edge_aware=False,
        ))
        self.delete.clicked.connect(self._delete)
        self.list.currentIndexChanged.connect(self._select)
        self.show_mask.toggled.connect(self.overlayChanged)
        self.edit_on_photo.toggled.connect(self.overlayChanged)
        self.brush_size.valueChanged.connect(self._brush_tool_updated)
        self.brush_feather.valueChanged.connect(self._brush_tool_updated)
        self.brush_mode_group.buttonToggled.connect(lambda *_: self._brush_tool_updated())
        self.set_values(())

    @property
    def selected(self):
        return next((g for g in self.values if g.id == self.list.currentData()), None)

    @property
    def brush_radius(self) -> float:
        return self.brush_size.value() / 1000.0

    @property
    def brush_feather_value(self) -> float:
        return self.brush_feather.value() / 100.0

    @property
    def is_erasing(self) -> bool:
        return bool(self.brush_erase.isChecked())

    def set_drawing_brush(self, active: bool):
        self._drawing_brush = bool(active)
        self.brush_panel.setVisible(isinstance(self.selected, BrushMask) or self._drawing_brush)
        self.done.setEnabled(self.selected is not None or self.edit_on_photo.isChecked() or self._drawing_brush)

    def _brush_tool_updated(self, *_):
        self.brush_size_label.setText(f'Brush size  {self.brush_size.value() / 10:.1f}%')
        self.brush_feather_label.setText(f'Brush feather  {self.brush_feather.value()}%')
        self.brushToolChanged.emit()

    def _select(self):
        self.gestureEnded.emit()
        self.set_values(self.values)
        self.selectionChanged.emit()

    def _rename(self):
        try:
            self._change(name=self.name.text())
        except ValueError as error:
            self.hint.setText(str(error))
            self.set_values(self.values)

    def _on_luma_min_changed(self, value: int) -> None:
        current = self.selected
        new_min = value / 100
        if current and new_min > current.luma_max:
            self._change(luma_min=new_min, luma_max=new_min)
        else:
            self._change(luma_min=new_min)

    def _on_luma_max_changed(self, value: int) -> None:
        current = self.selected
        new_max = value / 100
        if current and new_max < current.luma_min:
            self._change(luma_min=new_max, luma_max=new_max)
        else:
            self._change(luma_max=new_max)

    def _change(self, **fields):
        current = self.selected
        if current is None:
            return
        updated = replace(current, **fields)
        from .gradients import validate_gradients
        values = tuple(updated if g.id == current.id else g for g in self.values)
        validate_gradients(values)
        self.changed.emit(values)

    def _delete(self):
        if self.selected:
            self.changed.emit(tuple(g for g in self.values if g.id != self.selected.id))

    def _done(self):
        self._drawing_brush = False
        self.show_mask.setChecked(False)
        self.edit_on_photo.setChecked(False)

    def set_values(self, values, selected_id=None):
        selected_id = selected_id or self.list.currentData()
        self.values = values
        self.list.blockSignals(True)
        self.list.clear()
        for g in values:
            shape = 'Brush' if isinstance(g, BrushMask) else ('Radial' if isinstance(g, RadialGradient) else 'Linear')
            label = f'{g.name} · {shape}' + (' · off' if not g.enabled else '')
            self.list.addItem(label, g.id)
            self.list.setItemData(self.list.count() - 1, label, Qt.ItemDataRole.ToolTipRole)
        index = self.list.findData(selected_id)
        self.list.setCurrentIndex(max(0, index))
        self.list.blockSignals(False)
        self.add.setEnabled(len(values) < 4)
        current = self.selected
        radial = isinstance(current, RadialGradient)
        brush = isinstance(current, BrushMask)
        self.brush_panel.setVisible(brush or self._drawing_brush)
        self.brush_size_label.setText(f'Brush size  {self.brush_size.value() / 10:.1f}%')
        self.brush_feather_label.setText(f'Brush feather  {self.brush_feather.value()}%')
        self.feather_label.setVisible(radial)
        self.feather.setVisible(radial)
        self.feather.blockSignals(True)
        self.feather.setValue(round(current.feather * 100) if radial else 50)
        self.feather.blockSignals(False)
        self.feather_label.setText(f'Feather  {self.feather.value()}%')
        self.angle_label.setVisible(radial)
        self.angle.setVisible(radial)
        self.angle.blockSignals(True)
        self.angle.setValue(round(getattr(current, 'angle', 0.)) if radial else 0)
        self.angle.blockSignals(False)
        self.angle_label.setText(f'Angle  {self.angle.value():+d}°')
        self.controls.setEnabled(current is not None)
        self.done.setEnabled(current is not None or self.edit_on_photo.isChecked() or self._drawing_brush)
        for widget, value in ((self.name, current.name if current else ''),
                              (self.enabled, bool(current and current.enabled)),
                              (self.inverted, bool(current and current.inverted)),
                              (self.edge_aware, bool(current and current.edge_aware))):
            widget.blockSignals(True)
            widget.setText(value) if widget is self.name else widget.setChecked(value)
            widget.blockSignals(False)
        for slider, val in (
            (self.luma_min, round(current.luma_min * 100) if current else 0),
            (self.luma_max, round(current.luma_max * 100) if current else 100),
            (self.luma_feather, round(current.luma_feather * 100) if current else round(DEFAULT_LUMA_FEATHER * 100)),
        ):
            slider.blockSignals(True)
            slider.setValue(val)
            slider.blockSignals(False)
        self.luma_min_label.setText(f'Luma min  {self.luma_min.value()}%')
        self.luma_max_label.setText(f'Luma max  {self.luma_max.value()}%')
        self.luma_feather_label.setText(f'Range feather  {self.luma_feather.value()}%')
        for key, slider in self.sliders.items():
            value = getattr(current, key) if current else 0.
            slider.blockSignals(True)
            slider.setValue(round(value * 100))
            slider.blockSignals(False)
            title = 'Exposure' if key == 'exposure_ev' else key.title()
            self.labels[key].setText(f'{title}  {value:+.2f}' + (' EV' if key == 'exposure_ev' else ''))


class GradientController:
    def __init__(self, window, editor):
        from .gradient_view import GradientInteraction
        self.window, self.editor = window, editor
        self.drawing_id = None
        self.drawing_kind = 'linear'
        self.redraw_replace = False
        self.layer = GradientInteraction(window.preview, self)
        editor.changed.connect(self.change)
        editor.gestureStarted.connect(lambda: window._begin_parameter_gesture('local_gradients'))
        editor.gestureEnded.connect(window._finish_gesture)
        editor.selectionChanged.connect(self.selection_changed)
        editor.overlayChanged.connect(self.overlay_changed)
        editor.drawRequested.connect(self.draw)
        editor.brushToolChanged.connect(self._brush_tool_changed)
        window.clipping.toggled.connect(self.clipping_changed)

    @property
    def drawing_radial(self) -> bool:
        return self.drawing_kind == 'radial'

    @drawing_radial.setter
    def drawing_radial(self, value: bool) -> None:
        self.drawing_kind = 'radial' if value else 'linear'

    def _brush_tool_changed(self):
        self.window.preview.update()

    def sync(self):
        state = self.window.store.current
        self.editor.set_values(state.local_gradients if state else (), self.drawing_id)
        self.window.coach_text.set_edit_values(state.values if state else None)
        current = self.editor.selected
        if current and not self.layer.new_drawing:
            size = self.window.preview.full_size
            if isinstance(current, BrushMask):
                points = [self.layer.canvas_point(p) for s in current.strokes for p in s.points]
                outside = not any(p is not None and 0 <= p[0] <= size[0] and 0 <= p[1] <= size[1] for p in points)
                self.editor.hint.setText(
                    tr('Brush strokes are outside the crop. Redraw mask to paint inside; its adjustments are retained.',
                       '画笔笔画位于当前裁剪范围外。可点击“重绘蒙版”在画面内重新涂抹，调整参数会保留。')
                    if outside else
                    tr(f'Brush mask ({len(current.strokes)}/{MAX_STROKES_PER_MASK} strokes). Drag on photo to paint or erase. Space + drag pans.',
                       f'画笔蒙版（{len(current.strokes)}/{MAX_STROKES_PER_MASK} 笔）。在画面拖动可涂抹或擦除；空格 + 拖动平移。')
                )
            else:
                points = [self.layer.canvas_point(p) for p in (current.start, current.end)]
                outside = not any(p is not None and 0 <= p[0] <= size[0] and 0 <= p[1] <= size[1] for p in points)
                if outside:
                    self.editor.hint.setText(tr(
                        'Handles are outside the crop. Redraw mask to reposition it; its adjustments are retained.',
                        '手柄位于当前裁剪范围外。可点击“重绘蒙版”重新定位，调整参数会保留。'))
                elif isinstance(current, RadialGradient):
                    self.editor.hint.setText(tr(
                        'Drag either corner to resize the ellipse; drag the center to move. Space + drag pans.',
                        '拖动任意角落可缩放椭圆，拖动中心可移动。空格 + 拖动平移。'))
                else:
                    self.editor.hint.setText(tr(
                        'Drag either end to change the transition; drag the center to move. Space + drag pans.',
                        '拖动任意端点可改变渐变过渡，拖动中心可移动。空格 + 拖动平移。'))
        elif current is None and not self.layer.new_drawing:
            self.editor.hint.setText(tr(
                'Add a gradient, then drag from the fully affected side toward the unaffected side.',
                '添加蒙版后，在画面上拖动或涂抹以创建局部调整范围。'))
        self.window.preview.update()

    def change(self, values):
        self.window.change_parameter('local_gradients', values)
        self.sync()

    def draw(self, redraw=False):
        if not self.window.store.current or self.window._exporting:
            return
        self.layer.finish()
        if not redraw and len(self.editor.values) >= 4:
            return
        self.window.pick_wb.setChecked(False)
        if hasattr(self.window, 'spots_editor'):
            self.window.spots_editor.edit_on_photo.setChecked(False)
        selected = self.editor.selected
        self.drawing_id = selected.id if redraw and selected else uuid4().hex
        if redraw and selected is not None:
            self.drawing_kind = 'brush' if isinstance(selected, BrushMask) else ('radial' if isinstance(selected, RadialGradient) else 'linear')
        else:
            idx = self.editor.new_shape.currentIndex()
            self.drawing_kind = 'brush' if idx == 2 else ('radial' if idx == 1 else 'linear')
        self.redraw_replace = bool(redraw and self.drawing_kind == 'brush')
        self.layer.new_drawing = True
        self.editor.set_drawing_brush(self.drawing_kind == 'brush')
        self.editor.edit_on_photo.setChecked(True)
        self.editor.show_mask.setChecked(True)
        self.editor.done.setEnabled(True)
        if self.drawing_kind == 'brush':
            self.editor.hint.setText(tr(
                'Paint on the photo to create brush strokes. Switch Paint/Erase anytime; Esc cancels.',
                '在画面上涂抹以创建画笔蒙版，可随时切换涂抹/擦除；按 Esc 取消。'))
        elif self.drawing_kind == 'radial':
            self.editor.hint.setText(tr(
                'Drag opposite corners of an ellipse. Inside is affected; Invert targets outside. Esc cancels.',
                '拖出椭圆的对角边界。默认影响内部，勾选“反向”可影响外部；按 Esc 取消。'))
        else:
            self.editor.hint.setText(tr(
                'Drag on the photo: start = full effect, end = no effect. Esc cancels. Space + drag pans.',
                '在画面拖动：起点为全效，终点为无影响。按 Esc 取消；空格 + 拖动平移。'))
        self.window.preview.setFocus()
        self.window.preview.setCursor(Qt.CursorShape.CrossCursor)

    def update_gradient(self, start, end):
        current = next((g for g in self.editor.values if g.id == self.drawing_id), None)
        if current is None:
            names = {g.name.casefold() for g in self.editor.values}
            prefix = 'Radial' if self.drawing_radial else 'Gradient'
            name = next(f'{prefix} {i}' for i in range(1, 6) if f'{prefix} {i}'.casefold() not in names)
            kind = RadialGradient if self.drawing_radial else LinearGradient
            updated = kind(self.drawing_id, name, start, end)
            values = (*self.editor.values, updated)
        else:
            updated = replace(current, start=start, end=end)
            values = tuple(updated if g.id == current.id else g for g in self.editor.values)
        self.change(values)

    def update_brush_stroke(self, stroke: BrushStroke, *, base_strokes: tuple[BrushStroke, ...] = (), replace_all: bool = False):
        current = next((g for g in self.editor.values if g.id == self.drawing_id), None)
        if current is None:
            names = {g.name.casefold() for g in self.editor.values}
            name = next(f'Brush {i}' for i in range(1, 6) if f'Brush {i}'.casefold() not in names)
            updated = BrushMask(self.drawing_id, name, strokes=(stroke,))
            values = (*self.editor.values, updated)
        else:
            new_strokes = (stroke,) if replace_all else (*base_strokes, stroke)
            updated = replace(current, strokes=new_strokes)
            values = tuple(updated if g.id == current.id else g for g in self.editor.values)
        self.change(values)

    def overlay_spec(self):
        if self.window.clipping.isChecked():
            return OverlaySpec('clipping')
        if self.editor.show_mask.isChecked() and self.editor.selected:
            if not self.layer.new_drawing or self.drawing_id == self.editor.selected.id:
                return OverlaySpec('mask', self.editor.selected)
        return None

    def clipping_changed(self, enabled):
        if enabled:
            self.editor.show_mask.blockSignals(True)
            self.editor.show_mask.setChecked(False)
            self.editor.show_mask.blockSignals(False)
        self.overlay_changed()

    def selection_changed(self):
        self.layer.finish()
        self.drawing_id = None
        self.redraw_replace = False
        self.editor.set_drawing_brush(False)
        self.overlay_changed()

    def overlay_changed(self):
        if self.editor.show_mask.isChecked() and self.window.clipping.isChecked():
            self.window.clipping.blockSignals(True)
            self.window.clipping.setChecked(False)
            self.window.clipping.blockSignals(False)
        if self.editor.edit_on_photo.isChecked():
            if hasattr(self.window, 'spots_editor'):
                self.window.spots_editor.edit_on_photo.setChecked(False)
        else:
            self.layer.finish()
            self.layer.new_drawing = False
            self.layer.hover_pos = None
            self.drawing_id = None
            self.redraw_replace = False
            self.editor.set_drawing_brush(False)
            self.window.preview.setCursor(Qt.CursorShape.CrossCursor if self.window.preview.picking_neutral else
                                          Qt.CursorShape.OpenHandCursor if self.window.preview._space else Qt.CursorShape.ArrowCursor)
        self.window.preview.overlay_generation += 1
        self.window.preview.clear_overlay()
        self.window.preview.update()
        self.show_notice()
        if self.window.store.current and not self.window._exporting:
            self.window.request_preview(interactive=self.window.store.gesture_active)

    def show_notice(self):
        self.window.view_notice.setText('SDR: red highs / blue blacks' if self.window.clipping.isChecked() else
                                        'Selected mask overlay' if self.editor.show_mask.isChecked() and self.editor.selected else '')

    def end_geometry(self, cancel=False):
        window = self.window
        window.store.end_drag(cancel=cancel)
        self.redraw_replace = False
        self.editor.set_drawing_brush(False)
        self.sync()
        self.drawing_id = None
        window._after_edit()
        window._finalize_preview()

    def locate(self, gradient_id=None):
        self.window.local_section.set_expanded(True)
        index = self.editor.list.findData(gradient_id)
        if index >= 0:
            self.editor.list.setCurrentIndex(index)
        self.window.tabs.setCurrentIndex(0)

        def reveal():
            if not self.window._closing:
                self.window.edit_scroll.ensureWidgetVisible(self.editor.reset)
                self.window.edit_scroll.ensureWidgetVisible(self.editor.list)
                self.editor.list.setFocus()

        QTimer.singleShot(0, self.window, reveal)
