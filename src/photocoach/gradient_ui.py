"""Local adjustment widgets and state-manager wiring; no image algorithms."""
from dataclasses import replace
from uuid import uuid4
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,
                              QPushButton,QLineEdit,QCheckBox)
from .edit_controls import EditSlider
from .workspace_widgets import ParameterLabel
from .gradients import LinearGradient,RadialGradient,OverlaySpec


class GradientEditor(QWidget):
    selectionChanged=Signal()
    changed=Signal(object)
    gestureStarted=Signal()
    gestureEnded=Signal()
    drawRequested=Signal(bool)
    overlayChanged=Signal()

    def __init__(self):
        super().__init__()
        self.values=()
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.list=QComboBox();self.list.setAccessibleName('Selected local mask')
        layout.addWidget(self.list)
        self.new_shape=QComboBox();self.new_shape.addItems(('Linear','Radial'))
        self.new_shape.setAccessibleName('New mask shape')
        self.new_shape.setToolTip('Shape for the next new mask. Existing masks are unchanged.')
        creation=QHBoxLayout();creation.addWidget(QLabel('New mask'));creation.addWidget(self.new_shape,1)
        layout.addLayout(creation)
        row=QHBoxLayout()
        self.add=QPushButton('Add mask');self.done=QPushButton('Done')
        row.addWidget(self.add);row.addWidget(self.done);layout.addLayout(row)
        self.controls=QWidget();body=QVBoxLayout(self.controls);body.setContentsMargins(0,0,0,0)
        self.name=QLineEdit();self.name.setMaxLength(40);self.name.setPlaceholderText('Gradient name')
        self.name.setAccessibleName('Gradient name')
        body.addWidget(self.name)
        switches=QHBoxLayout()
        self.enabled=QCheckBox('Enabled');self.inverted=QCheckBox('Invert')
        switches.addWidget(self.enabled);switches.addWidget(self.inverted);body.addLayout(switches)
        self.sliders={};self.labels={}
        for name,title,limit in (('exposure_ev','Exposure',300),('highlights','Highlights',100),('shadows','Shadows',100)):
            label=ParameterLabel();slider=EditSlider(Qt.Orientation.Horizontal)
            slider.setRange(-limit,limit);slider.setAccessibleName(f'Local gradient {title}')
            slider.sliderPressed.connect(self.gestureStarted);slider.sliderReleased.connect(self.gestureEnded)
            slider.valueChanged.connect(lambda value,key=name:self._change(**{key:value/100}))
            self.sliders[name]=slider;self.labels[name]=label
            body.addWidget(label);body.addWidget(slider)
        self.feather_label=ParameterLabel()
        self.feather=EditSlider(Qt.Orientation.Horizontal)
        self.feather.setRange(0,100);self.feather.setAccessibleName('Radial feather')
        self.feather.setToolTip('Soft transition inside the ellipse: 0% is a hard edge, 100% fades from the center.')
        self.feather.sliderPressed.connect(self.gestureStarted)
        self.feather.sliderReleased.connect(self.gestureEnded)
        self.feather.valueChanged.connect(lambda value:self._change(feather=value/100))
        body.addWidget(self.feather_label);body.addWidget(self.feather)
        self.redraw=QPushButton('Redraw mask')
        self.reset=QPushButton('Reset adjustments');self.delete=QPushButton('Delete mask')
        for button in (self.redraw,self.reset,self.delete):body.addWidget(button)
        self.edit_on_photo=QCheckBox('Edit on photo');self.show_mask=QCheckBox('Show selected mask')
        body.addWidget(self.edit_on_photo);body.addWidget(self.show_mask)
        layout.addWidget(self.controls)
        self.hint=QLabel('Add a gradient, then drag from the fully affected side toward the unaffected side.')
        self.hint.setWordWrap(True);self.hint.setMinimumHeight(55);layout.addWidget(self.hint)
        self.setToolTip('Up to four manual linear or radial masks, attached to original image coordinates. Exposure adds in stops; overlapping Highlights/Shadows combine with global amounts and are limited to −1…+1. No RAW recovery or automatic subject detection.')
        self.add.clicked.connect(lambda:self.drawRequested.emit(False))
        self.redraw.clicked.connect(lambda:self.drawRequested.emit(True))
        self.done.clicked.connect(self._done)
        self.name.editingFinished.connect(self._rename)
        self.enabled.toggled.connect(lambda v:self._change(enabled=v))
        self.inverted.toggled.connect(lambda v:self._change(inverted=v))
        self.reset.clicked.connect(lambda:self._change(exposure_ev=0.,highlights=0.,shadows=0.))
        self.delete.clicked.connect(self._delete)
        self.list.currentIndexChanged.connect(self._select)
        self.show_mask.toggled.connect(self.overlayChanged)
        self.edit_on_photo.toggled.connect(self.overlayChanged)
        self.set_values(())

    @property
    def selected(self):
        return next((g for g in self.values if g.id==self.list.currentData()),None)

    def _select(self):
        self.gestureEnded.emit()
        self.set_values(self.values)
        self.selectionChanged.emit()

    def _rename(self):
        try:self._change(name=self.name.text())
        except ValueError as error:
            self.hint.setText(str(error));self.set_values(self.values)

    def _change(self,**fields):
        current=self.selected
        if current is None:return
        updated=replace(current,**fields)
        from .gradients import validate_gradients
        values=tuple(updated if g.id==current.id else g for g in self.values)
        validate_gradients(values)
        self.changed.emit(values)

    def _delete(self):
        if self.selected:self.changed.emit(tuple(g for g in self.values if g.id!=self.selected.id))

    def _done(self):
        self.show_mask.setChecked(False)
        self.edit_on_photo.setChecked(False)

    def set_values(self,values,selected_id=None):
        selected_id=selected_id or self.list.currentData()
        self.values=values
        self.list.blockSignals(True);self.list.clear()
        for g in values:
            shape='Radial' if isinstance(g,RadialGradient) else 'Linear'
            label=f'{g.name} · {shape}'+(' · off' if not g.enabled else '')
            self.list.addItem(label,g.id)
            self.list.setItemData(self.list.count()-1,label,Qt.ItemDataRole.ToolTipRole)
        index=self.list.findData(selected_id);self.list.setCurrentIndex(max(0,index))
        self.list.blockSignals(False)
        self.add.setEnabled(len(values)<4)
        current=self.selected
        radial=isinstance(current,RadialGradient)
        self.feather_label.setVisible(radial);self.feather.setVisible(radial)
        self.feather.blockSignals(True)
        self.feather.setValue(round(current.feather*100) if radial else 50)
        self.feather.blockSignals(False)
        self.feather_label.setText(f'Feather  {self.feather.value()}%')
        self.controls.setEnabled(current is not None)
        self.done.setEnabled(current is not None or self.edit_on_photo.isChecked())
        for widget,value in ((self.name,current.name if current else ''),
                             (self.enabled,bool(current and current.enabled)),(self.inverted,bool(current and current.inverted))):
            widget.blockSignals(True)
            widget.setText(value) if widget is self.name else widget.setChecked(value)
            widget.blockSignals(False)
        for key,slider in self.sliders.items():
            value=getattr(current,key) if current else 0.
            slider.blockSignals(True);slider.setValue(round(value*100));slider.blockSignals(False)
            title='Exposure' if key=='exposure_ev' else key.title()
            self.labels[key].setText(f'{title}  {value:+.2f}'+(' EV' if key=='exposure_ev' else ''))


class GradientController:
    def __init__(self,window,editor):
        from .gradient_view import GradientInteraction
        self.window,self.editor=window,editor
        self.drawing_id=None
        self.drawing_radial=False
        self.layer=GradientInteraction(window.preview,self)
        editor.changed.connect(self.change)
        editor.gestureStarted.connect(lambda:window._begin_parameter_gesture('local_gradients'))
        editor.gestureEnded.connect(window._finish_gesture)
        editor.selectionChanged.connect(self.selection_changed)
        editor.overlayChanged.connect(self.overlay_changed)
        editor.drawRequested.connect(self.draw)
        window.clipping.toggled.connect(self.clipping_changed)

    def sync(self):
        state=self.window.store.current
        self.editor.set_values(state.local_gradients if state else (),self.drawing_id)
        self.window.coach_text.set_gradients(state.local_gradients if state else ())
        current=self.editor.selected
        if current and not self.layer.new_drawing:
            size=self.window.preview.full_size
            points=[self.layer.canvas_point(p) for p in (current.start,current.end)]
            outside=not any(0<=p[0]<=size[0] and 0<=p[1]<=size[1] for p in points)
            self.editor.hint.setText('Handles are outside the crop. Redraw mask to reposition it; its adjustments are retained.' if outside else
                'Drag either corner to resize the ellipse; drag the center to move. Space + drag pans.' if isinstance(current,RadialGradient) else
                'Drag either end to change the transition; drag the center to move. Space + drag pans.')
        elif current is None and not self.layer.new_drawing:
            self.editor.hint.setText('Add a gradient, then drag from the fully affected side toward the unaffected side.')
        self.window.preview.update()

    def change(self,values):
        self.window.change_parameter('local_gradients',values)
        self.sync()

    def draw(self,redraw=False):
        if not self.window.store.current or self.window._exporting:return
        self.layer.finish()
        if not redraw and len(self.editor.values)>=4:return
        self.window.pick_wb.setChecked(False)
        self.drawing_id=self.editor.selected.id if redraw and self.editor.selected else uuid4().hex
        self.drawing_radial=isinstance(self.editor.selected,RadialGradient) if redraw else self.editor.new_shape.currentText()=='Radial'
        self.layer.new_drawing=True
        self.editor.edit_on_photo.setChecked(True)
        self.editor.show_mask.setChecked(True)
        self.editor.done.setEnabled(True)
        self.editor.hint.setText('Drag opposite corners of an ellipse. Inside is affected; Invert targets outside. Esc cancels.' if self.drawing_radial else
                                'Drag on the photo: start = full effect, end = no effect. Esc cancels. Space + drag pans.')
        self.window.preview.setFocus();self.window.preview.setCursor(Qt.CursorShape.CrossCursor)

    def update_gradient(self,start,end):
        current=next((g for g in self.editor.values if g.id==self.drawing_id),None)
        if current is None:
            names={g.name.casefold() for g in self.editor.values}
            prefix='Radial' if self.drawing_radial else 'Gradient'
            name=next(f'{prefix} {i}' for i in range(1,6) if f'{prefix} {i}'.casefold() not in names)
            kind=RadialGradient if self.drawing_radial else LinearGradient
            updated=kind(self.drawing_id,name,start,end)
            values=(*self.editor.values,updated)
        else:
            updated=replace(current,start=start,end=end)
            values=tuple(updated if g.id==current.id else g for g in self.editor.values)
        self.change(values)

    def overlay_spec(self):
        if self.window.clipping.isChecked():return OverlaySpec('clipping')
        if self.editor.show_mask.isChecked() and self.editor.selected:
            if not self.layer.new_drawing or self.drawing_id==self.editor.selected.id:
                return OverlaySpec('mask',self.editor.selected)
        return None

    def clipping_changed(self,enabled):
        if enabled:
            self.editor.show_mask.blockSignals(True);self.editor.show_mask.setChecked(False);self.editor.show_mask.blockSignals(False)
        self.overlay_changed()

    def selection_changed(self):
        self.layer.finish()
        self.drawing_id=None
        self.overlay_changed()

    def overlay_changed(self):
        if self.editor.show_mask.isChecked() and self.window.clipping.isChecked():
            self.window.clipping.blockSignals(True);self.window.clipping.setChecked(False);self.window.clipping.blockSignals(False)
        if not self.editor.edit_on_photo.isChecked():
            self.layer.finish();self.layer.new_drawing=False;self.drawing_id=None
            self.window.preview.setCursor(Qt.CursorShape.CrossCursor if self.window.preview.picking_neutral else
                Qt.CursorShape.OpenHandCursor if self.window.preview._space else Qt.CursorShape.ArrowCursor)
        self.window.preview.overlay_generation+=1
        self.window.preview.clear_overlay()
        self.window.preview.update()
        self.show_notice()
        if self.window.store.current and not self.window._exporting:
            self.window.request_preview(interactive=self.window.store.gesture_active)

    def show_notice(self):
        self.window.view_notice.setText('SDR: red highs / blue blacks' if self.window.clipping.isChecked() else
            'Selected mask overlay' if self.editor.show_mask.isChecked() and self.editor.selected else '')

    def end_geometry(self,cancel=False):
        window=self.window
        window.store.end_drag(cancel=cancel)
        self.sync()
        self.drawing_id=None
        self.editor.hint.setText('Drag either corner to resize the ellipse; drag the center to move. Space + drag pans.' if isinstance(self.editor.selected,RadialGradient) else
                                'Drag either end to change the transition; drag the center to move. Space + drag pans.')
        window._after_edit()
        window._finalize_preview()

    def locate(self,gradient_id=None):
        self.window.local_section.set_expanded(True)
        index=self.editor.list.findData(gradient_id)
        if index>=0:self.editor.list.setCurrentIndex(index)
        self.window.tabs.setCurrentIndex(0)
        def reveal():
            if not self.window._closing:
                self.window.edit_scroll.ensureWidgetVisible(self.editor.list)
                self.editor.list.setFocus()
        QTimer.singleShot(0,self.window,reveal)
