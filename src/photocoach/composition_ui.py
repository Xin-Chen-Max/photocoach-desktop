"""Local crop drafts; only Apply commits to the editor's state manager."""
from dataclasses import replace
from PySide6.QtCore import Qt, Signal, QObject, QRunnable, QThreadPool, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QComboBox, QDoubleSpinBox, QPushButton, QDialogButtonBox)
from .composition import Composition, canvas_size, output_size
from .adjustments import render_photo
from .imaging import quantize_srgb
from .preview_view import display_image


class CropCanvas(QWidget):
    cropChanged = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(280, 180)
        self.image = None
        self.crop = (0., 0., 1., 1.)
        self.aspect = None  # normalized width / height
        self._drag = None
        self.setMouseTracking(True)

    def image_rect(self):
        if self.image is None:
            return QRectF()
        scale = min((self.width()-24)/self.image.width(), (self.height()-24)/self.image.height())
        w, h = self.image.width()*scale, self.image.height()*scale
        return QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)

    def set_crop(self, crop):
        self.crop = tuple(crop)
        self.update()
        self.cropChanged.emit(self.crop)

    def crop_rect(self):
        box = self.image_rect()
        l, t, r, b = self.crop
        return QRectF(box.left()+l*box.width(), box.top()+t*box.height(),
                      (r-l)*box.width(), (b-t)*box.height())

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#202020"))
        if self.image is None:
            return
        box, crop = self.image_rect(), self.crop_rect()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(box, self.image)
        shade = QColor(0, 0, 0, 150)
        for rect in (QRectF(box.left(), box.top(), box.width(), crop.top()-box.top()),
                     QRectF(box.left(), crop.bottom(), box.width(), box.bottom()-crop.bottom()),
                     QRectF(box.left(), crop.top(), crop.left()-box.left(), crop.height()),
                     QRectF(crop.right(), crop.top(), box.right()-crop.right(), crop.height())):
            p.fillRect(rect, shade)
        p.setPen(QPen(QColor("#79afff"), 1))
        p.drawRect(crop)
        for fraction in (1/3, 2/3):
            p.drawLine(QPointF(crop.left()+crop.width()*fraction, crop.top()),
                       QPointF(crop.left()+crop.width()*fraction, crop.bottom()))
            p.drawLine(QPointF(crop.left(), crop.top()+crop.height()*fraction),
                       QPointF(crop.right(), crop.top()+crop.height()*fraction))
        for point in (crop.topLeft(), crop.topRight(), crop.bottomLeft(), crop.bottomRight()):
            p.fillRect(QRectF(point.x()-4, point.y()-4, 8, 8), QColor("#eeeeee"))

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.image is None:
            return
        rect = self.crop_rect()
        points = (rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight())
        corner = next((i for i, p in enumerate(points) if (p-event.position()).manhattanLength() <= 18), None)
        if corner is not None or rect.contains(event.position()):
            self._drag = (event.position(), self.crop, corner)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        point, original, corner = self._drag
        box = self.image_rect()
        dx, dy = (event.position().x()-point.x())/box.width(), (event.position().y()-point.y())/box.height()
        l, t, r, b = original
        if corner is None:
            dx, dy = max(-l, min(1-r, dx)), max(-t, min(1-b, dy))
            crop = (l+dx, t+dy, r+dx, b+dy)
        else:
            left, top = corner in (0, 2), corner in (0, 1)
            anchor_x, anchor_y = (r if left else l), (b if top else t)
            w = max(.02, min(anchor_x if left else 1-anchor_x, r-l + (-dx if left else dx)))
            h = max(.02, min(anchor_y if top else 1-anchor_y, b-t + (-dy if top else dy)))
            if self.aspect:
                max_w = min(anchor_x if left else 1-anchor_x,
                            (anchor_y if top else 1-anchor_y)*self.aspect)
                w = min(max_w, max(.02, .02*self.aspect, min(w, h*self.aspect)))
                h = w/self.aspect
            crop = (anchor_x-w if left else anchor_x, anchor_y-h if top else anchor_y,
                    anchor_x if left else anchor_x+w, anchor_y if top else anchor_y+h)
        # Round tiny binary errors at 2% / canvas boundaries.
        crop = tuple(round(max(0., min(1., v)), 12) for v in crop)
        try:
            Composition(crop)
        except ValueError:
            return
        self.set_crop(crop)

    def mouseReleaseEvent(self, event):
        self._drag = None


class _Signals(QObject):
    ready = Signal(object, object)


class _Render(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = _Signals()

    def run(self):
        try:
            self.signals.ready.emit(self.function(), None)
        except Exception as error:
            self.signals.ready.emit(None, str(error))


class CompositionDialog(QDialog):
    def __init__(self, parent, linear, source_size, values):
        super().__init__(parent)
        self.setWindowTitle("Crop and straighten")
        self.resize(900, 680)
        self.linear, self.source_size, self.values = linear, source_size, values
        self._worker = None
        self._generation = 0
        self._closing_result = None
        self._rendered_angle = None
        layout = QVBoxLayout(self)
        instruction = QLabel("Drag a corner to resize; drag inside to move. Apply makes one undoable edit.")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        self.canvas = CropCanvas()
        self.canvas.crop = values.composition.crop
        layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.ratio = QComboBox()
        for name, ratio in (("Free", None), ("Original", source_size[0]/source_size[1]),
                            ("3:2", 3/2), ("4:3", 4/3), ("1:1", 1.), ("16:9", 16/9),
                            ("2:3", 2/3), ("3:4", 3/4), ("9:16", 9/16)):
            self.ratio.addItem(name, ratio)
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-15, 15)
        self.angle.setDecimals(2)
        self.angle.setSingleStep(.1)
        self.angle.setSuffix("°")
        self.angle.setValue(values.composition.angle)
        self.angle.setKeyboardTracking(False)
        self.angle.setToolTip("Positive rotates clockwise. Empty corners are trimmed automatically.")
        for widget in (QLabel("Aspect"), self.ratio, QLabel("Straighten"), self.angle):
            row.addWidget(widget)
        layout.addLayout(row)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        note = QLabel("Reduced preview · Straightening trims edges and resamples pixels. Originals stay untouched.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.clicked.connect(lambda: self.done(QDialog.DialogCode.Accepted))
        buttons.rejected.connect(self.reject)
        self.reset_button = QPushButton("Reset crop and angle")
        buttons.addButton(self.reset_button, QDialogButtonBox.ButtonRole.ResetRole)
        self.reset_button.clicked.connect(self.reset)
        layout.addWidget(buttons)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self._start_render)
        self.angle.valueChanged.connect(self._schedule)
        self.ratio.currentIndexChanged.connect(self._aspect_changed)
        self.canvas.cropChanged.connect(self._update_info)
        self._schedule()

    @property
    def composition(self):
        return Composition(self.canvas.crop, self.angle.value())

    def _aspect_changed(self):
        ratio = self.ratio.currentData()
        cw, ch = canvas_size(self.source_size, self.angle.value())
        self.canvas.aspect = None if ratio is None else ratio*ch/cw
        if self.canvas.aspect:
            l, t, r, b = self.canvas.crop
            cx, cy = (l+r)/2, (t+b)/2
            w = max(.020001, .020001*self.canvas.aspect, min(r-l, (b-t)*self.canvas.aspect))
            w = min(1., self.canvas.aspect, w)
            h = w/self.canvas.aspect
            cx, cy = max(w/2, min(1-w/2, cx)), max(h/2, min(1-h/2, cy))
            self.canvas.set_crop((cx-w/2, cy-h/2, cx+w/2, cy+h/2))

    def _update_info(self):
        w, h = output_size(self.source_size, self.composition)
        self.info.setText(f"Output: {w} × {h} pixels" if self._rendered_angle == self.angle.value()
                          else "Updating composition preview…")

    def reset(self):
        self.ratio.setCurrentIndex(0)
        self.canvas.set_crop((0., 0., 1., 1.))
        self.angle.setValue(0.)
        self._schedule()

    def _schedule(self):
        self._generation += 1
        self.apply_button.setEnabled(False)
        self.canvas.setEnabled(False)
        self._aspect_changed()
        self._update_info()
        self.timer.start()

    def _start_render(self):
        if self._worker is not None or self._closing_result is not None:
            return
        self.timer.stop()
        generation = self._generation
        angle = self.angle.value()
        values = replace(self.values, composition=Composition(angle=angle))
        worker = _Render(lambda: quantize_srgb(render_photo(self.linear, values,
                                  source_size=self.source_size, max_edge=1000)))
        self._worker = worker
        worker.signals.ready.connect(lambda pixels, error: self._ready(generation, angle, pixels, error))
        QThreadPool.globalInstance().start(worker)

    def _ready(self, generation, angle, pixels, error):
        self._worker = None
        if self._closing_result is not None:
            super().done(self._closing_result)
            return
        if generation != self._generation:
            self._start_render()
            return
        if error:
            self.info.setText("Preview failed: " + error)
            return
        self.canvas.image = display_image(pixels)
        self.canvas.setEnabled(True)
        self.canvas.update()
        self._rendered_angle = angle
        self.apply_button.setEnabled(True)
        self._update_info()

    def done(self, result):
        if result == QDialog.DialogCode.Accepted and not self.apply_button.isEnabled():
            return
        self.timer.stop()
        if self._worker is not None:
            self._closing_result = result
            self.setEnabled(False)
            return
        super().done(result)

    def closeEvent(self, event):
        event.ignore()
        self.reject()
