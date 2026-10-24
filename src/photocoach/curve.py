"""A shared RGB curve with explicit drag transactions and linear interpolation."""
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QLabel, QWidget

from .state import IDENTITY_CURVE, validate_curve


class CurveEditor(QWidget):
    gestureStarted = Signal()
    pointsChanged = Signal(object)
    gestureEnded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = IDENTITY_CURVE
        self.selected = -1
        self._dragging = False
        self._start_points = self.points
        self._hover_position = None
        self.setMouseTracking(True)
        self.readout = QLabel(self)
        self.readout.setWordWrap(True)
        self.readout.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.readout.setStyleSheet("background: #222222; color: #e4e4e4; padding: 3px; font-size: 12px;")
        self.readout.move(6, 6)
        self.readout.hide()
        self.setMinimumSize(240, 155)
        self.setMaximumHeight(210)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("Input / Output use 0–1 values. Hover near a point for its exact values; elsewhere shows cursor coordinates. Double-click to add a point. Drag to move it. Delete removes an inner point. Escape cancels a drag.")

    def set_points(self, points):
        self.points = validate_curve(points)
        if self.selected >= len(self.points):
            self.selected = -1
        self._update_readout()
        self.update()

    def _update_readout(self):
        if self._hover_position is None:
            self.readout.hide()
            return
        index = self.selected if self._dragging else self._nearest(self._hover_position)
        x, y = self.points[index] if index >= 0 else self._value(self._hover_position)
        kind = "Point" if index >= 0 else "Cursor"
        self.readout.setText(f"{kind} · Input {x:.3f} / Output {y:.3f}")
        self.readout.ensurePolished()
        self.readout.setMaximumWidth(max(1, self.width() - 12))
        self.readout.adjustSize()
        self.readout.show()

    def leaveEvent(self, event):
        self._hover_position = None
        self._update_readout()
        super().leaveEvent(event)

    def resizeEvent(self, event):
        self._update_readout()
        super().resizeEvent(event)

    def _screen(self, point):
        return QPointF(12 + point[0] * (self.width() - 24), 12 + (1 - point[1]) * (self.height() - 24))

    def _value(self, point):
        return (max(0, min(1, (point.x() - 12) / max(1, self.width() - 24))),
                max(0, min(1, 1 - (point.y() - 12) / max(1, self.height() - 24))))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#222222"))
        painter.setPen(QPen(QColor("#444444"), 1))
        for i in range(5):
            v = i / 4
            painter.drawLine(self._screen((v, 0)), self._screen((v, 1)))
            painter.drawLine(self._screen((0, v)), self._screen((1, v)))
        painter.setPen(QPen(QColor("#777777"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(self._screen((0, 0)), self._screen((1, 1)))
        painter.setPen(QPen(QColor("#79afff"), 2))
        painter.drawPolyline(QPolygonF([self._screen(p) for p in self.points]))
        for index, point in enumerate(self.points):
            painter.setBrush(QColor("#ffffff" if index == self.selected else "#79afff"))
            painter.drawEllipse(self._screen(point), 4, 4)

    def _nearest(self, position):
        distances = [((self._screen(p).x() - position.x()) ** 2 +
                      (self._screen(p).y() - position.y()) ** 2, i) for i, p in enumerate(self.points)]
        distance, index = min(distances)
        return index if distance <= 144 else -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.selected = self._nearest(event.position())
            if self.selected >= 0:
                self._start_points = self.points
                self._dragging = True
                self.gestureStarted.emit()
            self.update()

    def mouseMoveEvent(self, event):
        self._hover_position = event.position()
        if not self._dragging:
            self._update_readout()
            return
        x, y = self._value(event.position())
        i = self.selected
        if i == 0:
            x = 0.0
        elif i == len(self.points) - 1:
            x = 1.0
        else:
            lower, upper = self.points[i - 1][0], self.points[i + 1][0]
            gap = min(.005, (upper - lower) / 3)
            x = max(lower + gap, min(upper - gap, x))
        points = list(self.points)
        points[i] = (x, y)
        self.set_points(points)
        self.pointsChanged.emit(self.points)

    def finish(self):
        if self._dragging:
            self._dragging = False
            self.gestureEnded.emit()

    def mouseReleaseEvent(self, event):
        self.finish()

    def focusOutEvent(self, event):
        self.finish()
        super().focusOutEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.finish()
        if event.button() != Qt.MouseButton.LeftButton or len(self.points) >= 16:
            return
        x, y = self._value(event.position())
        if min(abs(x - point[0]) for point in self.points) < .005:
            return
        self.gestureStarted.emit()
        points = sorted((*self.points, (x, y)))
        self.selected = points.index((x, y))
        self.set_points(points)
        self.pointsChanged.emit(self.points)
        self.gestureEnded.emit()

    def remove_selected(self):
        self.finish()
        if 0 < self.selected < len(self.points) - 1:
            self.gestureStarted.emit()
            points = list(self.points)
            del points[self.selected]
            self.selected = -1
            self.set_points(points)
            self.pointsChanged.emit(self.points)
            self.gestureEnded.emit()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_selected()
        elif event.key() == Qt.Key.Key_Escape and self._dragging:
            self.set_points(self._start_points)
            self.pointsChanged.emit(self.points)
            self.finish()
        else:
            super().keyPressEvent(event)
