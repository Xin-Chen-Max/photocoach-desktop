"""A source-coordinate photo canvas. All raster processing stays in workers."""
from __future__ import annotations

import math
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QEvent
from PySide6.QtGui import QImage, QColorSpace, QPainter, QColor
from PySide6.QtWidgets import QWidget, QApplication, QGridLayout

from .preview import DetailRequest, visible_region
from .composition import Composition, output_size


def display_image(pixels) -> QImage:
    pixels = np.ascontiguousarray(pixels)
    height, width, _ = pixels.shape
    image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGB888).copy()
    image.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    return image


class PhotoToolbar(QWidget):
    """Keep navigation/comparison visible; move quality to a second row if narrow."""
    def __init__(self, navigation, comparison, status):
        super().__init__()
        self.parts = (navigation, comparison, status)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(14)
        self.two_rows = None
        self._arrange()

    def _arrange(self):
        required=sum(part.minimumSizeHint().width() for part in self.parts)+3*self.grid.horizontalSpacing()
        narrow = self.width() < max(740,required)
        if narrow == self.two_rows:
            return
        self.two_rows = narrow
        for part in self.parts:
            self.grid.removeWidget(part)
        self.grid.addWidget(self.parts[0], 0, 0, Qt.AlignmentFlag.AlignLeft)
        self.grid.addWidget(self.parts[1], 0, 1, Qt.AlignmentFlag.AlignLeft)
        self.grid.setColumnStretch(2, 1)
        if narrow:
            self.grid.addWidget(self.parts[2], 1, 0, 1, 3, Qt.AlignmentFlag.AlignLeft)
        else:
            self.grid.addWidget(self.parts[2], 0, 3, Qt.AlignmentFlag.AlignRight)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrange()


class ImageView(QWidget):
    viewChanged = Signal()
    statusChanged = Signal()
    neutralPicked = Signal(object)
    pickerCancelled = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 260)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMouseTracking(True)
        self.full_size = (0, 0)
        self.source_size = (0, 0)
        self.composition = Composition()
        self.center = (0.0, 0.0)
        self.zoom = 1.0
        self.fit_mode = True
        self.view_generation = 0
        self.quality = "Quick preview"
        self.quick_current = False
        self._quick = QImage()
        self._detail = QImage()
        self._detail_temporary = False
        self._detail_rect = (0, 0, 0, 0)
        self._message = "Open a Sony .ARW to begin"
        self._space = False
        self._drag = None
        self.picking_neutral = False
        self._dpr = self.devicePixelRatioF()
        self.gradient_layer = None
        self.overlay_generation = 0
        self._overlay = QImage()
        self._overlay_rect = (0,0,0,0)
        # Observe Space without consuming it in text fields. This also permits
        # Space+click immediately after adjusting a slider, without a focus click.
        self._observing_keys = False

    @property
    def zoom_text(self):
        return ("Fit · " if self.fit_mode else "") + f"{self.zoom * 100:.0f}%"

    @property
    def has_quick(self):
        return not self._quick.isNull()

    def set_quality(self, text):
        self.quality = text
        self.statusChanged.emit()

    def setText(self, text):
        self._message = text
        self.update()

    def clear_photo(self):
        self.set_neutral_picker(False)
        self.set_quality("Opening photo…")
        # Retain the previous image and geometry until opening actually succeeds.
        self.setText("Reading RAW…")
        self.view_generation += 1

    def configure_photo(self, size):
        self.clear_overlay()
        self.full_size = tuple(size)
        self.source_size = tuple(size)
        self.composition = Composition()
        self._quick = QImage()
        self._detail = QImage()
        self.quick_current = False
        self._message = ""
        self.fit_mode = True
        self._changed()

    def set_composition(self, composition):
        if composition == self.composition or not all(self.source_size):
            return
        self.composition = composition
        self.clear_overlay()
        self.full_size = output_size(self.source_size, composition)
        self._quick = QImage()
        self._detail = QImage()
        self.quick_current = False
        self._message = "Updating composition…"
        self.fit_mode = True
        self._changed()

    def display(self, pixels):
        self.clear_overlay()
        self._quick = display_image(pixels)
        self._detail = QImage()
        self._message = ""
        if not all(self.full_size):
            self.configure_photo((pixels.shape[1], pixels.shape[0]))
            self._quick = display_image(pixels)
        self.quick_current = True
        self.update()

    def display_detail(self, pixels, request):
        if request.generation != self.view_generation or request.full_size != self.full_size:
            return False
        self._detail = display_image(pixels)
        self._detail_temporary = False
        self.clear_overlay()
        self._detail_rect = request.rect
        self.set_quality("Full detail")
        self.update()
        return True

    def display_adjusting(self, pixels, request):
        """Replace only the visible ROI; retain source geometry and old background."""
        if request.generation != self.view_generation or request.full_size != self.full_size:
            return False
        self._detail = display_image(pixels)
        self._detail_temporary = True
        self.clear_overlay()
        self._detail_rect = request.rect
        self.set_quality("Adjusting")
        self.update()
        return True

    def invalidate_edit(self, quality="Refining"):
        self.clear_overlay()
        # Keep the previous frame while updating; display() replaces it if a
        # new quick frame arrives before full detail. The quality label is explicit.
        self.set_quality(quality)
        self.quick_current = False
        self.update()

    def fit(self):
        self.fit_mode = True
        self._changed()

    def actual_size(self, position=None):
        if not all(self.full_size):
            return
        if position is not None:
            self.center = self.source_at(position)
        self.fit_mode = False
        self.zoom = 1.0
        self._changed()

    def source_at(self, position):
        dpr = self.devicePixelRatioF()
        return (self.center[0] + (position.x() - self.width() / 2) * dpr / self.zoom,
                self.center[1] + (position.y() - self.height() / 2) * dpr / self.zoom)

    def zoom_at(self, factor, position):
        if not all(self.full_size):
            return
        before = self.source_at(position)
        self.fit_mode = False
        self.zoom = min(8.0, max(0.01, self.zoom * factor))
        after = self.source_at(position)
        self.center = (self.center[0] + before[0] - after[0], self.center[1] + before[1] - after[1])
        self._changed()

    def pan(self, dx, dy):
        if not all(self.full_size) or self.fit_mode:
            return
        scale = self.devicePixelRatioF() / self.zoom
        self.center = (self.center[0] - dx * scale, self.center[1] - dy * scale)
        self._changed()

    def _changed(self):
        if not all(self.full_size):
            return
        dpr = self.devicePixelRatioF()
        self._dpr = dpr
        width, height = self.full_size
        if self.fit_mode:
            self.zoom = min(self.width() * dpr / width, self.height() * dpr / height)
            self.center = (width / 2, height / 2)
        else:
            center = []
            for length, viewport, value in zip(self.full_size, (self.width(), self.height()), self.center):
                half = viewport * dpr / self.zoom / 2
                value = length / 2 if half >= length / 2 else max(half, min(length - half, value))
                # Snap the image origin, not its center, to physical pixels at 1:1.
                if abs(self.zoom - 1.0) < 1e-9:
                    value = viewport * dpr / 2 - round(viewport * dpr / 2 - value)
                center.append(value)
            self.center = tuple(center)
        self.view_generation += 1
        self.set_quality("Quick preview")
        self.viewChanged.emit()
        self.update()

    def needs_detail(self):
        if self._message or self._quick.isNull() or not all(self.full_size):
            return False
        return (not self.fit_mode and self.zoom >= 1) or self.zoom > min(
            self._quick.width() / self.full_size[0], self._quick.height() / self.full_size[1]) * 1.01

    def detail_request(self):
        rect = visible_region(self.full_size, (self.width(), self.height()), self.center,
                              self.zoom, self.devicePixelRatioF())
        return DetailRequest(self.view_generation, self.full_size, rect,
                             max(1, math.ceil(1 / self.zoom)), self.source_size)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202020"))
        if self._message:
            painter.setPen(QColor("#ababab"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return
        if self._quick.isNull():
            return
        scale = self.zoom / self.devicePixelRatioF()
        painter.translate(self.width() / 2 - self.center[0] * scale,
                          self.height() / 2 - self.center[1] * scale)
        painter.scale(scale, scale)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(QRectF(0, 0, *self.full_size), self._quick)
        if not self._detail.isNull():
            left, top, right, bottom = self._detail_rect
            # Nearest at >=100% preserves native pixels. DPR is applied once by Qt.
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                                  self._detail_temporary or self.zoom < 1)
            painter.drawImage(QRectF(left, top, right - left, bottom - top), self._detail)
        painter.setClipRect(QRectF(0,0,*self.full_size),Qt.ClipOperation.IntersectClip)
        if not self._overlay.isNull():
            l,t,r,b=self._overlay_rect
            painter.drawImage(QRectF(l,t,r-l,b-t),self._overlay)
        if self.gradient_layer is not None:
            self.gradient_layer.paint(painter)

    def clear_overlay(self):
        self._overlay=QImage()
        self.update()

    def accept_overlay(self,frame):
        self.clear_overlay()
        if frame is not None:
            pixels,rect=frame
            h,w,_=pixels.shape
            self._overlay=QImage(pixels.data,w,h,pixels.strides[0],QImage.Format.Format_RGBA8888).copy()
            self._overlay_rect=rect
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._changed()

    def event(self, event):
        if event.type() == QEvent.Type.DevicePixelRatioChange and hasattr(self, "full_size"):
            self._changed()
        return super().event(event)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            self._space = True
        elif (event.type() == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_Space
              and not event.isAutoRepeat()):
            self._stop_pan()
        elif event.type() == QEvent.Type.WindowDeactivate and watched is self.window():
            self._stop_pan()
        return False

    def showEvent(self, event):
        if not self._observing_keys:
            QApplication.instance().installEventFilter(self)
            self._observing_keys = True
        super().showEvent(event)

    def hideEvent(self, event):
        if self._observing_keys:
            QApplication.instance().removeEventFilter(self)
            self._observing_keys = False
        self._stop_pan()
        super().hideEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y() or event.pixelDelta().y()
            self.zoom_at(1.25 ** (delta / 120), event.position())
            event.accept()
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        if self.picking_neutral:
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.fit_mode:
                self.actual_size(event.position())
            else:
                self.fit()
            event.accept()

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._space and event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif self.picking_neutral and event.button() == Qt.MouseButton.LeftButton:
            point = self.source_at(event.position())
            if 0 <= point[0] < self.full_size[0] and 0 <= point[1] < self.full_size[1]:
                self.set_neutral_picker(False)
                self.neutralPicked.emit(point)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            delta = event.position() - self._drag
            self._drag = event.position()
            self.pan(delta.x(), delta.y())
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag = None
        self.setCursor(Qt.CursorShape.OpenHandCursor if self._space else
                       Qt.CursorShape.CrossCursor if self.picking_neutral else Qt.CursorShape.ArrowCursor)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.picking_neutral:
            self.set_neutral_picker(False)
            self.pickerCancelled.emit()
            event.accept()
        elif event.key() == Qt.Key.Key_Space:
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._stop_pan()
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def _stop_pan(self):
        self._space = False
        self._drag = None
        self.setCursor(Qt.CursorShape.CrossCursor if self.picking_neutral else Qt.CursorShape.ArrowCursor)

    def set_neutral_picker(self, enabled):
        self.picking_neutral = bool(enabled)
        self._drag = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def focusOutEvent(self, event):
        self._stop_pan()
        super().focusOutEvent(event)
