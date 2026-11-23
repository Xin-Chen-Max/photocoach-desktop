"""Direct manipulation of gradients and brush masks in the original photo coordinate system."""
import math
from PySide6.QtCore import QObject, QEvent, Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QPen, QPainterPath, QTransform
from .gradients import (source_point, RadialGradient, BrushMask, BrushStroke,
                        MAX_POINTS_PER_STROKE, MAX_STROKES_PER_MASK)
from .i18n import tr


def _decimate_points(points: list[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    indices = [round(i * step) for i in range(limit - 1)] + [len(points) - 1]
    return [points[idx] for idx in indices]


class GradientInteraction(QObject):
    def __init__(self, view, controller):
        super().__init__(view)
        self.view, self.controller = view, controller
        self.drag = None
        self.new_drawing = False
        self.started = False
        self.hover_pos: QPointF | None = None
        view.gradient_layer = self
        view.installEventFilter(self)

    def _is_brush_active(self) -> bool:
        if self.new_drawing:
            return self.controller.drawing_kind == 'brush'
        return isinstance(self.controller.editor.selected, BrushMask)

    def normalized(self, position):
        view = self.view
        p = source_point(view.source_at(position), view.source_size, view.composition, distortion=view.distortion)
        return tuple(max(0., min(1., v / size)) for v, size in zip(p, view.source_size))

    def canvas_point(self, point):
        view = self.view
        return source_point(tuple(v * s for v, s in zip(point, view.source_size)),
                            view.source_size, view.composition, inverse=True, distortion=view.distortion)

    def screen_point(self, point):
        mapped = self.canvas_point(point)
        if mapped is None:
            return None
        x, y = mapped
        view = self.view
        scale = view.zoom / view.devicePixelRatioF()
        return QPointF(view.width() / 2 + (x - view.center[0]) * scale,
                       view.height() / 2 + (y - view.center[1]) * scale)

    def eventFilter(self, watched, event):
        try:
            return self._handle_event(watched, event)
        except RuntimeError:
            return False

    def _handle_event(self, watched, event):
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
            if (self.drag is not None or self.new_drawing
                    or (editor.edit_on_photo.isChecked() and isinstance(editor.selected, BrushMask))):
                self.finish(cancel=True)
                self.new_drawing = False
                self.hover_pos = None
                editor.edit_on_photo.setChecked(False)
                return True
        if self.drag is not None and (kind == QEvent.Type.Wheel or
                                      (kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space)):
            self.finish()
        if not editor.isEnabled() or not editor.edit_on_photo.isChecked() or not all(view.source_size):
            return False
        if kind == QEvent.Type.MouseMove and self.drag is None:
            if self._is_brush_active():
                self.hover_pos = event.position()
                view.update()
            return False
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton and not view._space:
            composed = view.source_at(event.position())
            if not (0 <= composed[0] <= view.full_size[0] and 0 <= composed[1] <= view.full_size[1]):
                return False
            current = editor.selected
            if self._is_brush_active():
                if not self.new_drawing:
                    if not isinstance(current, BrushMask):
                        return False
                    if len(current.strokes) >= MAX_STROKES_PER_MASK:
                        editor.hint.setText(tr(
                            f'Maximum {MAX_STROKES_PER_MASK} strokes per brush mask reached. Delete or redraw mask, or add another mask.',
                            f'单个画笔蒙版最多支持 {MAX_STROKES_PER_MASK} 笔。请删除、重绘或新建其他蒙版。',
                        ))
                        return True
                    base_strokes = current.strokes
                    replace_all = False
                    self.controller.drawing_id = current.id
                else:
                    base_strokes = ()
                    replace_all = bool(self.controller.redraw_replace)
                p0 = self.normalized(event.position())
                radius = editor.brush_radius
                feather = editor.brush_feather_value
                erase = editor.is_erasing
                self.drag = ('brush', [p0], radius, feather, erase, base_strokes, replace_all, bool(self.new_drawing))
                self.hover_pos = event.position()
                self.controller.window._begin_parameter_gesture('local_gradients')
                self.started = True
                self.controller.update_brush_stroke(
                    BrushStroke((p0,), radius, feather, erase=erase),
                    base_strokes=base_strokes,
                    replace_all=replace_all,
                )
                view.setFocus()
                view.update()
                return True
            mode = 'new' if self.new_drawing else None
            if mode is None and current:
                a, b = self.screen_point(current.start), self.screen_point(current.end)
                center = self.screen_point(tuple((s + e) / 2 for s, e in zip(current.start, current.end)))
                for key, p in (('start', a), ('end', b), ('move', center)):
                    if p is not None and (p - event.position()).manhattanLength() < 18:
                        mode = key
                        break
            if mode:
                self.drag = (mode, event.position(), self.normalized(event.position()), current)
                self.started = False
                if mode != 'new':
                    self.controller.drawing_id = current.id
                view.setFocus()
                return True
        if kind == QEvent.Type.MouseMove and self.drag is not None:
            if self.drag[0] == 'brush':
                _, points, radius, feather, erase, base_strokes, replace_all, _ = self.drag
                self.hover_pos = event.position()
                p = self.normalized(event.position())
                w, h = view.source_size
                min_step = max(1.5, radius * min(w, h) * 0.15)
                if math.hypot((p[0] - points[-1][0]) * w, (p[1] - points[-1][1]) * h) >= min_step:
                    points.append(p)
                    if len(points) > MAX_POINTS_PER_STROKE:
                        points[:] = _decimate_points(points, MAX_POINTS_PER_STROKE)
                    self.controller.update_brush_stroke(
                        BrushStroke(tuple(points), radius, feather, erase=erase),
                        base_strokes=base_strokes,
                        replace_all=replace_all,
                    )
                view.update()
                return True
            mode, origin, point, current = self.drag
            if mode == 'new' and (origin - event.position()).manhattanLength() < 8:
                return True
            end = self.normalized(event.position())
            if mode == 'new':
                start = point
            elif mode == 'start':
                start, end = end, current.end
            elif mode == 'end':
                start = current.start
            else:
                dx, dy = end[0] - point[0], end[1] - point[1]
                dx = max(-min(current.start[0], current.end[0]), min(1 - max(current.start[0], current.end[0]), dx))
                dy = max(-min(current.start[1], current.end[1]), min(1 - max(current.start[1], current.end[1]), dy))
                start = (current.start[0] + dx, current.start[1] + dy)
                end = (current.end[0] + dx, current.end[1] + dy)
            if math.dist(start, end) < 1e-5:
                return True
            radial = self.controller.drawing_radial if mode == 'new' else isinstance(current, RadialGradient)
            if radial and any(abs(a - b) < 1e-5 for a, b in zip(start, end)):
                return True
            if not self.started:
                self.controller.window._begin_parameter_gesture('local_gradients')
                self.started = True
            self.controller.update_gradient(start, end)
            return True
        if kind == QEvent.Type.MouseButtonRelease and self.drag is not None:
            self.finish()
            return True
        return False

    def finish(self, cancel=False):
        if self.drag is None:
            return
        was_new = self.drag[7] if self.drag[0] == 'brush' else self.drag[0] == 'new'
        self.drag = None
        changed = self.started
        if was_new and (changed or cancel):
            self.new_drawing = False
        if changed:
            self.started = False
            self.controller.end_geometry(cancel)
        self.view.update()

    def paint(self, painter):
        view = self.view
        editor = self.controller.editor
        g = editor.selected
        if not editor.isEnabled() or not editor.edit_on_photo.isChecked():
            return
        if self._is_brush_active():
            self._paint_brush(painter)
            return
        if g is None or isinstance(g, BrushMask):
            return
        if self.new_drawing and self.controller.drawing_id != g.id:
            return
        a, b = self.canvas_point(g.start), self.canvas_point(g.end)
        scale = view.zoom / view.devicePixelRatioF()
        painter.save()
        pen = QPen(QColor('#79cfd3'), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        if view.distortion:
            self._paint_distorted(painter, g, pen)
            painter.restore()
            return
        if isinstance(g, RadialGradient):
            origin = self.canvas_point((0, 0))
            x = self.canvas_point((1, 0))
            y = self.canvas_point((0, 1))
            transform = QTransform(x[0] - origin[0], x[1] - origin[1], y[0] - origin[0], y[1] - origin[1], *origin)
            cx, cy = ((s + e) / 2 for s, e in zip(g.start, g.end))
            rx, ry = (abs(e - s) / 2 for s, e in zip(g.start, g.end))
            guide = QPen(QColor(121, 207, 211, 100), 1, Qt.PenStyle.DotLine)
            guide.setCosmetic(True)
            painter.setPen(guide)
            rot = QTransform().translate(cx, cy).rotate(getattr(g, 'angle', 0.)).translate(-cx, -cy)
            bounds = QPainterPath()
            bounds.addRect(QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry))
            painter.drawPath(transform.map(rot.map(bounds)))
            painter.setPen(pen)
            for factor in (1., 1. - g.feather):
                if factor <= 0:
                    continue
                path = QPainterPath()
                path.addEllipse(QRectF(cx - rx * factor, cy - ry * factor, 2 * rx * factor, 2 * ry * factor))
                painter.drawPath(transform.map(rot.map(path)))
        else:
            painter.drawLine(QPointF(*a), QPointF(*b))
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            for point in (a, b):
                tx, ty = -dy / length * max(view.full_size) * 2, dx / length * max(view.full_size) * 2
                painter.drawLine(QPointF(point[0] - tx, point[1] - ty), QPointF(point[0] + tx, point[1] + ty))
        for point in (a, b, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)):
            painter.setBrush(QColor('#202020'))
            painter.drawEllipse(QPointF(*point), 5 / scale, 5 / scale)
        painter.restore()

    def _paint_brush(self, painter):
        view = self.view
        editor = self.controller.editor
        if not all(view.source_size):
            return
        scale = view.zoom / view.devicePixelRatioF()
        painter.save()
        if self.drag is not None and self.drag[0] == 'brush':
            points = self.drag[1]
            erasing = self.drag[4]
            color = QColor('#f09a76') if erasing else QColor('#79cfd3')
            stroke_pen = QPen(color, 1.5)
            stroke_pen.setCosmetic(True)
            painter.setPen(stroke_pen)
            mapped = [self.canvas_point(p) for p in points]
            valid = [QPointF(*pt) for pt in mapped if pt is not None]
            if len(valid) >= 2:
                path = QPainterPath(valid[0])
                for pt in valid[1:]:
                    path.lineTo(pt)
                painter.drawPath(path)
        if self.hover_pos is not None and not view._space:
            composed = view.source_at(self.hover_pos)
            if 0 <= composed[0] <= view.full_size[0] and 0 <= composed[1] <= view.full_size[1]:
                erasing = editor.is_erasing
                ring_color = QColor('#f09a76') if erasing else QColor('#79cfd3')
                outer_pen = QPen(ring_color, 1.25)
                outer_pen.setCosmetic(True)
                inner_pen = QPen(QColor(ring_color.red(), ring_color.green(), ring_color.blue(), 150),
                                 1, Qt.PenStyle.DotLine)
                inner_pen.setCosmetic(True)
                r_px = editor.brush_radius * min(view.source_size)
                feather = editor.brush_feather_value
                if view.distortion:
                    src_center = source_point(composed, view.source_size, view.composition, distortion=view.distortion)
                    w, h = view.source_size
                    for factor, pen in ((1.0, outer_pen), (1.0 - feather, inner_pen)):
                        if factor <= 0.02 or (factor < 1.0 and factor > 0.98):
                            continue
                        painter.setPen(pen)
                        rad = r_px * factor
                        ring_pts = []
                        for k in range(33):
                            th = 2.0 * math.pi * k / 32.0
                            pt = self.canvas_point(((src_center[0] + rad * math.cos(th)) / w,
                                                    (src_center[1] + rad * math.sin(th)) / h))
                            if pt is not None:
                                ring_pts.append(QPointF(*pt))
                        if len(ring_pts) >= 2:
                            path = QPainterPath(ring_pts[0])
                            for pt in ring_pts[1:]:
                                path.lineTo(pt)
                            painter.drawPath(path)
                else:
                    painter.setPen(outer_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(QPointF(*composed), r_px, r_px)
                    inner_factor = 1.0 - feather
                    if 0.02 < inner_factor < 0.98:
                        painter.setPen(inner_pen)
                        painter.drawEllipse(QPointF(*composed), r_px * inner_factor, r_px * inner_factor)
                painter.setPen(outer_pen)
                painter.setBrush(ring_color)
                painter.drawEllipse(QPointF(*composed), 2.5 / scale, 2.5 / scale)
        painter.restore()

    def _paint_distorted(self, painter, g, pen):
        """Sample source guides adaptively; never stretch an affine ellipse."""
        view = self.view
        cx, cy = ((s + e) / 2 for s, e in zip(g.start, g.end))

        def draw(point_at):
            for segment in adaptive_segments(point_at, self.canvas_point, view.zoom):
                if len(segment) < 2:
                    continue
                path = QPainterPath(QPointF(*segment[0]))
                for point in segment[1:]:
                    path.lineTo(QPointF(*point))
                painter.drawPath(path)

        if isinstance(g, RadialGradient):
            rx, ry = (abs(e - s) / 2 for s, e in zip(g.start, g.end))
            ang = math.radians(getattr(g, 'angle', 0.))
            ca, sa = math.cos(ang), math.sin(ang)
            for factor in (1., 1. - g.feather):
                if factor <= 0:
                    continue
                draw(lambda t, f=factor: (cx + (rx * f * math.cos(t * 2 * math.pi) * ca - ry * f * math.sin(t * 2 * math.pi) * sa),
                                          cy + (rx * f * math.cos(t * 2 * math.pi) * sa + ry * f * math.sin(t * 2 * math.pi) * ca)))
        else:
            draw(lambda t: tuple(a + (b - a) * t for a, b in zip(g.start, g.end)))
            w, h = view.source_size
            dx, dy = (g.end[0] - g.start[0]) * w, (g.end[1] - g.start[1]) * h
            length = math.hypot(dx, dy)
            # Guides are perpendicular in original pixels, then warped together
            # with the source. Clip in source first, viewport in ImageView.paint.
            for point in (g.start, g.end):
                vx, vy = -dy / length / w, dx / length / h
                lo, hi = -max(w, h) * 2, max(w, h) * 2
                for origin, direction in zip(point, (vx, vy)):
                    if direction:
                        a, b = sorted((-origin / direction, (1 - origin) / direction))
                        lo, hi = max(lo, a), min(hi, b)
                if lo < hi:
                    draw(lambda t, point=point, lo=lo, hi=hi: (point[0] + vx * (lo + (hi - lo) * t),
                                                               point[1] + vy * (lo + (hi - lo) * t)))
        scale = view.zoom / view.devicePixelRatioF()
        for source in (g.start, g.end, (cx, cy)):
            point = self.canvas_point(source)
            if point is not None:
                painter.setBrush(QColor('#202020'))
                painter.drawEllipse(QPointF(*point), 5 / scale, 5 / scale)


def adaptive_segments(point_at, mapper, zoom, *, max_vertices=4096, max_depth=10):
    """Curved source guide -> visible polylines with <=.5 physical-pixel error.

    None breaks a segment at the corrected footprint; no false connecting line
    crosses invisible points. Bounded subdivision also limits painting work.
    """
    vertices = []

    def append(point):
        if len(vertices) < max_vertices:
            vertices.append(point)

    def visit(t0, a, t1, b, depth):
        if len(vertices) >= max_vertices:
            return
        tm = (t0 + t1) / 2
        mid = mapper(point_at(tm))
        if a is not None and b is not None and mid is not None:
            error = math.dist(mid, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)) * zoom
            split = error > .5
        else:
            split = not (a is None and b is None and mid is None)
        if split and depth < max_depth:
            visit(t0, a, tm, mid, depth + 1)
            visit(tm, mid, t1, b, depth + 1)
        else:
            # At the cap omit an unresolved span rather than drawing a bad chord.
            if split:
                append(None)
            append(b)

    count = 32
    first = mapper(point_at(0.))
    append(first)
    for i in range(count):
        end = mapper(point_at((i + 1) / count))
        visit(i / count, first, (i + 1) / count, end, 0)
        first = end
    segments = []
    current = []
    for point in vertices:
        if point is None:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(point)
    if current:
        segments.append(current)
    return segments
