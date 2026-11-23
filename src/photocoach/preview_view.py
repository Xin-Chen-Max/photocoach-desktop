"""A source-coordinate photo canvas. All raster processing stays in workers."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QEvent
from PySide6.QtGui import QImage, QColorSpace, QPainter, QColor
from PySide6.QtWidgets import QWidget, QApplication, QGridLayout

from .preview import DetailRequest, visible_region
from .composition import Composition, output_size


MAX_DETAIL_LAYERS = 6
MAX_DETAIL_LAYER_BYTES = 48 * 1024 * 1024


@dataclass(frozen=True)
class _DetailLayer:
    rect: tuple[int, int, int, int]
    image: QImage
    temporary: bool
    step: int
    epoch: int
    is_backdrop: bool = False


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
        self.distortion = 0.
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
        self._detail_layers: list[_DetailLayer] = []
        self._edit_epoch = 0
        self._layer_epoch = 0
        self._detail_layers_stale = False
        self._message = "Open a Sony .ARW or Nikon .NEF to begin"
        self._space = False
        self._drag = None
        self.picking_neutral = False
        self._dpr = self.devicePixelRatioF()
        self.gradient_layer = None
        self.spot_layer = None
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

    def _is_full_rect(self, rect) -> bool:
        return bool(all(self.full_size)) and tuple(rect) == (0, 0, self.full_size[0], self.full_size[1])

    def has_current_full_frame(self) -> bool:
        if self._detail_layers_stale or not all(self.full_size):
            return False
        full_rect = (0, 0, self.full_size[0], self.full_size[1])
        return any(l.rect == full_rect and l.epoch == self._layer_epoch for l in self._detail_layers)

    def _prune_detail_layers(self, newest: _DetailLayer) -> None:
        while len(self._detail_layers) > MAX_DETAIL_LAYERS or sum(l.image.sizeInBytes() for l in self._detail_layers) > MAX_DETAIL_LAYER_BYTES:
            candidates = [i for i, l in enumerate(self._detail_layers) if l is not newest]
            if not candidates:
                break
            non_full = [i for i in candidates if not self._is_full_rect(self._detail_layers[i].rect)]
            if non_full:
                # Drop oldest non-full layer first, prioritizing older epochs then smallest/oldest tiles
                victim = min(non_full, key=lambda i: (self._detail_layers[i].epoch, i))
            else:
                victim = min(
                    candidates,
                    key=lambda i: (
                        self._detail_layers[i].epoch,
                        0 if self._detail_layers[i].is_backdrop else 1,
                        i,
                    ),
                )
            self._detail_layers.pop(victim)

    def _store_detail_layer(
        self,
        rect,
        image: QImage,
        *,
        temporary: bool,
        step: int,
        is_backdrop: bool = False,
    ) -> None:
        rect = tuple(rect)
        eff_step = max(1, int(step))
        if not is_backdrop and self._detail_layers_stale:
            self._layer_epoch = self._edit_epoch
            self._detail_layers_stale = False
            # Keep at most the single best previous full-frame layer as a temporary fallback
            full_candidates = [l for l in self._detail_layers if self._is_full_rect(l.rect)]
            if full_candidates:
                best_full = max(
                    full_candidates,
                    key=lambda l: (l.epoch, 0 if l.is_backdrop else 1, 0 if l.temporary else 1, -l.step),
                )
                self._detail_layers = [best_full]
            else:
                self._detail_layers.clear()

        if self._is_full_rect(rect) and not is_backdrop:
            # Current formal/adjusting full frame supersedes older fallbacks and backdrops
            self._detail_layers = [
                l for l in self._detail_layers
                if l.epoch == self._layer_epoch and not l.is_backdrop and l.rect != rect
            ]

        l0, t0, r0, b0 = rect
        filtered: list[_DetailLayer] = []
        for old in self._detail_layers:
            if old.rect == rect:
                replace_same = False
                if self._layer_epoch > old.epoch:
                    replace_same = True
                elif not is_backdrop and old.is_backdrop:
                    replace_same = True
                elif is_backdrop == old.is_backdrop and (not temporary or old.temporary):
                    replace_same = True
                elif eff_step <= old.step and (not temporary or old.temporary):
                    replace_same = True
                if replace_same:
                    continue
            if not is_backdrop and old.epoch == self._layer_epoch:
                ol, ot, or_, ob = old.rect
                if ol >= l0 and ot >= t0 and or_ <= r0 and ob <= b0 and old.step >= eff_step:
                    if not temporary or old.temporary:
                        continue
            filtered.append(old)
        self._detail_layers = filtered

        newest = _DetailLayer(rect, image, temporary, eff_step, self._layer_epoch, is_backdrop)
        self._detail_layers.append(newest)
        self._prune_detail_layers(newest)
        self._detail_layers.sort(
            key=lambda l: (
                l.epoch,
                -(l.rect[2] - l.rect[0]) * (l.rect[3] - l.rect[1]),
                -l.step,
                0 if l.is_backdrop else 1,
                0 if l.temporary else 1,
            )
        )

    def configure_photo(self, size):
        self.clear_overlay()
        self.full_size = tuple(size)
        self.source_size = tuple(size)
        self.composition = Composition()
        self.distortion = 0.
        self._quick = QImage()
        self._detail = QImage()
        self._detail_layers.clear()
        self._detail_layers_stale = False
        self._edit_epoch += 1
        self._layer_epoch = self._edit_epoch
        self.quick_current = False
        self._message = ""
        self.fit_mode = True
        self._changed()

    def set_composition(self, composition, *, distortion=0.):
        if (composition == self.composition and distortion == self.distortion) or not all(self.source_size):
            return
        composition_changed = composition != self.composition
        self.composition = composition
        self.distortion = distortion
        self.clear_overlay()
        self.full_size = output_size(self.source_size, composition)
        self._edit_epoch += 1
        if composition_changed:
            self._quick = QImage()
            self._detail = QImage()
            self._detail_layers.clear()
            self._detail_layers_stale = False
            self._layer_epoch = self._edit_epoch
            self._message = "Updating composition…"
        else:
            self._detail_layers_stale = True
        # Distortion keeps output dimensions: retain the prior raster until its
        # replacement arrives. Its quality is marked Adjusting/Refining.
        self.quick_current = False
        self.fit_mode = True
        self._changed()

    def display(self, pixels):
        self.clear_overlay()
        self._quick = display_image(pixels)
        self._detail = QImage()
        self._detail_layers.clear()
        self._detail_layers_stale = False
        self._edit_epoch += 1
        self._layer_epoch = self._edit_epoch
        self._message = ""
        if not all(self.full_size):
            self.configure_photo((pixels.shape[1], pixels.shape[0]))
            self._quick = display_image(pixels)
        self.quick_current = True
        self.update()

    def display_detail(self, pixels, request):
        if request.generation != self.view_generation or request.full_size != self.full_size:
            return False
        image = display_image(pixels)
        self._detail = image
        self._message = ""
        self._detail_temporary = False
        self.clear_overlay()
        self._detail_rect = request.rect
        self._store_detail_layer(request.rect, image, temporary=False, step=request.step, is_backdrop=False)
        self.set_quality("Full detail")
        self.update()
        return True

    def display_adjusting(self, pixels, request):
        """Replace only the visible ROI; retain source geometry and old background."""
        if request.generation != self.view_generation or request.full_size != self.full_size:
            return False
        image = display_image(pixels)
        self._detail = image
        self._message = ""
        self._detail_temporary = True
        self.clear_overlay()
        self._detail_rect = request.rect
        rw = max(1, request.rect[2] - request.rect[0])
        rh = max(1, request.rect[3] - request.rect[1])
        eff_step = max(request.step, math.ceil(max(rw, rh) / max(1, pixels.shape[1], pixels.shape[0])))
        self._store_detail_layer(request.rect, image, temporary=True, step=eff_step, is_backdrop=False)
        self.set_quality("Adjusting")
        self.update()
        return True

    def display_backdrop(self, pixels, full_size) -> bool:
        """Store a same-revision full-frame backdrop layer without altering active detail status."""
        if tuple(full_size) != self.full_size or not all(self.full_size) or self._detail_layers_stale:
            return False
        full_rect = (0, 0, self.full_size[0], self.full_size[1])
        if any(l.rect == full_rect and l.epoch == self._layer_epoch and not l.is_backdrop for l in self._detail_layers):
            return False
        image = display_image(pixels)
        step = max(1, math.ceil(max(self.full_size) / max(1, pixels.shape[1], pixels.shape[0])))
        self._detail_layers = [
            l for l in self._detail_layers
            if l.epoch == self._layer_epoch and not l.is_backdrop
        ]
        self._store_detail_layer(full_rect, image, temporary=False, step=step, is_backdrop=True)
        self.update()
        return True

    def invalidate_edit(self, quality="Refining"):
        self.clear_overlay()
        # Keep the previous frame while updating; display() replaces it if a
        # new quick frame arrives before full detail. The quality label is explicit.
        self._edit_epoch += 1
        self._detail_layers_stale = True
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
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        w, h = self.width(), self.height()
        if self._message:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            cw, ch = min(380, max(220, w - 40)), (104 if self._message == "Open a Sony .ARW or Nikon .NEF to begin" else 54)
            card = QRectF((w - cw) / 2, (h - ch) / 2, cw, ch)
            painter.setPen(QColor("#383838"))
            painter.setBrush(QColor("#252525"))
            painter.drawRoundedRect(card, 6, 6)
            if self._message == "Open a Sony .ARW or Nikon .NEF to begin":
                font = painter.font()
                font.setPixelSize(14)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QColor("#e4e4e4"))
                painter.drawText(QRectF(card.left() + 16, card.top() + 18, cw - 32, 24),
                                 Qt.AlignmentFlag.AlignCenter, self._message)
                font.setPixelSize(12)
                font.setBold(False)
                painter.setFont(font)
                painter.setPen(QColor("#9e9e9e"))
                painter.drawText(QRectF(card.left() + 16, card.top() + 46, cw - 32, 42),
                                 Qt.AlignmentFlag.AlignCenter,
                                 "Click Open (Ctrl+O) or Open project…\nOriginals remain strictly read-only.")
            else:
                painter.setPen(QColor("#cccccc"))
                painter.drawText(card, Qt.AlignmentFlag.AlignCenter, self._message)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QColor("#303030"))
            painter.drawRect(0, 0, w - 1, h - 1)
            return
        if self._quick.isNull() and self._detail.isNull() and not self._detail_layers:
            return
        scale = self.zoom / self.devicePixelRatioF()
        painter.translate(self.width() / 2 - self.center[0] * scale,
                          self.height() / 2 - self.center[1] * scale)
        painter.scale(scale, scale)
        if not self._quick.isNull():
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(QRectF(0, 0, *self.full_size), self._quick)
        if self._detail_layers:
            desired_step = max(1, math.ceil(1.0 / max(self.zoom, 1e-6)))
            full_cover = next(
                (
                    l for l in reversed(self._detail_layers)
                    if self._is_full_rect(l.rect)
                    and l.epoch == self._layer_epoch
                    and not l.is_backdrop
                    and not l.temporary
                    and l.step <= desired_step * 1.05
                ),
                None,
            )
            layers = (full_cover,) if full_cover is not None else self._detail_layers
            for layer in layers:
                left, top, right, bottom = layer.rect
                # Nearest at >=100% preserves native pixels. DPR is applied once by Qt.
                painter.setRenderHint(
                    QPainter.RenderHint.SmoothPixmapTransform,
                    layer.temporary or self.zoom < 1 or layer.step > 1,
                )
                painter.drawImage(QRectF(left, top, right - left, bottom - top), layer.image)
        elif not self._detail.isNull():
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
        if self.spot_layer is not None:
            self.spot_layer.paint(painter)
        painter.resetTransform()
        painter.setClipping(False)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor("#303030"))
        painter.drawRect(0, 0, w - 1, h - 1)

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
        try:
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
                self._space = True
            elif (event.type() == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_Space
                  and not event.isAutoRepeat()):
                self._stop_pan()
            elif event.type() == QEvent.Type.WindowDeactivate and watched is self.window():
                self._stop_pan()
        except RuntimeError:
            return False
        return False

    def showEvent(self, event):
        if not self._observing_keys:
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
                self._observing_keys = True
        super().showEvent(event)

    def hideEvent(self, event):
        if self._observing_keys:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._observing_keys = False
        self._stop_pan()
        super().hideEvent(event)

    def closeEvent(self, event):
        if self._observing_keys:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._observing_keys = False
        self._stop_pan()
        super().closeEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y() or event.pixelDelta().y()
            self.zoom_at(1.25 ** (delta / 120), event.position())
            event.accept()
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        if self.picking_neutral or (self.spot_layer is not None and self.spot_layer.is_active()):
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

    def _active_tool_cursor(self):
        if self._space:
            return Qt.CursorShape.OpenHandCursor
        if self.picking_neutral or (self.spot_layer is not None and self.spot_layer.is_active()):
            return Qt.CursorShape.CrossCursor
        return Qt.CursorShape.ArrowCursor

    def mouseReleaseEvent(self, event):
        self._drag = None
        self.setCursor(self._active_tool_cursor())

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
        self.setCursor(self._active_tool_cursor())

    def set_neutral_picker(self, enabled):
        self.picking_neutral = bool(enabled)
        self._drag = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def focusOutEvent(self, event):
        self._stop_pan()
        super().focusOutEvent(event)
