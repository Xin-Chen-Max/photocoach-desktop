"""Sampled final-SDR histogram; not a sensor histogram or recovery estimate."""
import math
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPolygonF
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from .adjustments import render_edit
from .imaging import quantize_srgb
from .composition import Composition, output_size, sample_composition
from .render_timing import timed
from .gradients import RenderGeometry


def sampled_histogram(linear, values, *, timer=None, source_size=None):
    """Worker-only whole-image grid sampling, <=512 pixels on the longest edge.

    Sample before the pointwise edit pipeline, without interpolation. Percentages
    refer to final 8-bit display levels, not exact exported or sensor clipping.
    """
    with timed(timer, "histogram"):
        return _sampled_histogram(linear, values, source_size=source_size)


def _sampled_histogram(linear, values, *, source_size=None):
    source_size=source_size or (linear.shape[1],linear.shape[0])
    size = output_size((linear.shape[1], linear.shape[0]), values.composition)
    step = max(1, math.ceil(max(size) / 512))
    sampled = (linear[::step, ::step] if values.composition == Composition() else
               sample_composition(linear, values.composition,
                   target_size=(math.ceil(size[0]/step), math.ceil(size[1]/step))))
    if any(g.active for g in values.local_gradients) or any(s.active for s in values.spots) or values.lens_corrections.distortion or values.lens_corrections.vignetting_ev:
        sampled=sample_composition(linear,values.composition,source_size=source_size,
                                   target_size=(sampled.shape[1],sampled.shape[0]),distortion=values.lens_corrections.distortion)
    pixels = quantize_srgb(render_edit(sampled, values,
        geometry=RenderGeometry.full(source_size,values.composition,(sampled.shape[1],sampled.shape[0]),values.lens_corrections.distortion),
        source_linear=linear))
    count = pixels.shape[0] * pixels.shape[1]
    return {"rgb": [np.bincount(pixels[...,channel].ravel(), minlength=256).tolist() for channel in range(3)],
            "samples": count, "stride": step,
            "black_percent": float(np.count_nonzero(np.all(pixels == 0, axis=2)) * 100 / count),
            "white_percent": float(np.count_nonzero(np.any(pixels == 255, axis=2)) * 100 / count)}


class HistogramPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.data = None
        self.setMinimumHeight(84)
        self.setMaximumHeight(100)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1d1d1d"))
        w, h = self.width(), self.height()
        painter.setPen(QColor("#2c2c2c"))
        for quarter in (1, 2, 3):
            gx = round(3 + (w - 6) * quarter / 4)
            painter.drawLine(gx, 1, gx, h - 2)
        if self.data is not None:
            counts = np.log1p(np.asarray(self.data["rgb"]))
            scale = max(1.0, float(counts.max()))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            base_y = h - 3
            for channel, color in zip(counts, ("#ee8b81", "#8dcfa0", "#86b9ef")):
                pts = [QPointF(3 + i*(w-6)/255, base_y - v/scale*(h-6)) for i, v in enumerate(channel)]
                fill = QColor(color)
                fill.setAlpha(34)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawPolygon(QPolygonF([QPointF(3, base_y), *pts, QPointF(w - 3, base_y)]))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QColor(color))
                painter.drawPolyline(QPolygonF(pts))
            for x, hit in ((4, self.data["black_percent"] > 0), (w - 11, self.data["white_percent"] > 0)):
                painter.fillRect(x, 4, 7, 7, QColor("#f2bf80" if hit else "#48545a"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QColor("#3b3b3b"))
        painter.drawRect(0, 0, w - 1, h - 1)


class HistogramView(QWidget):
    EMPTY_LIMITS = "Black floor —\nHighlight ceiling —\nWhole photo · sampled SDR"

    def __init__(self, compact=False):
        super().__init__()
        self.compact = compact
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(4)
        self.title = QLabel("Histogram · No photo")
        self.title.setStyleSheet("color:#c8c8c8;font-size:12px;font-weight:600;")
        self.plot = HistogramPlot()
        if compact:
            self.plot.setMinimumHeight(64)
            self.plot.setMaximumHeight(64)
        # Always reserve three explicit lines. Clearing/wrapping this label used
        # to move every slider underneath it while the pointer was dragging.
        self.limits = QLabel(self._limits_text())
        self.limits.setProperty("secondary", True)
        for widget in (self.title, self.plot, self.limits):
            layout.addWidget(widget)
        self.setToolTip("Whole-photo RGB histogram with a logarithmic count axis. Sampled final 8-bit SDR output, independent of zoom. Black floor: all channels at 0. Highlight ceiling: at least one channel at 255. These may be intentional; they do not prove RAW clipping or recoverability. Small regions may be missed by sampling.")

    def clear(self, message="Updating…"):
        self.title.setText("Histogram · " + message)
        self.plot.data = None
        self.plot.update()
        self.limits.setText(self._limits_text())

    def mark_updating(self):
        # Preserve the previous plot until an accepted render arrives; only
        # photo switches and failures clear it. The old statistics are labeled.
        self.title.setText("Histogram · Updating… (previous)" if self.plot.data is not None
                           else "Histogram · Updating…")

    def display(self, data, *, baseline=False, snapshot_name: str | None = None):
        if data is None:
            self.clear("Unavailable")
            return
        if baseline:
            label = "Baseline"
        elif snapshot_name:
            short = snapshot_name if len(snapshot_name) <= 22 else snapshot_name[:21] + "…"
            label = f"A/B ({short})"
        else:
            label = "Edited"
        self.title.setText("Histogram · " + label)
        self.plot.data = data
        self.plot.update()
        self.limits.setText(self._limits_text(data))

    def _limits_text(self, data=None):
        black = f"{data['black_percent']:.2f}%" if data else "—"
        white = f"{data['white_percent']:.2f}%" if data else "—"
        if self.compact:
            return f"Black {black}   ·   Highlight {white}\nWhole photo · sampled SDR"
        return f"Black floor {black}\nHighlight ceiling {white}\nWhole photo · sampled SDR"
