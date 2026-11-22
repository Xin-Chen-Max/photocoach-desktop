"""Direct manipulation of gradients in the original photo coordinate system."""
import math
from PySide6.QtCore import QObject,QEvent,Qt,QPointF,QRectF
from PySide6.QtGui import QColor,QPen,QPainterPath,QTransform
from .gradients import source_point,RadialGradient


class GradientInteraction(QObject):
    def __init__(self,view,controller):
        super().__init__(view)
        self.view,self.controller=view,controller
        self.drag=None;self.new_drawing=False;self.started=False
        view.gradient_layer=self
        view.installEventFilter(self)

    def normalized(self,position):
        view=self.view
        p=source_point(view.source_at(position),view.source_size,view.composition,distortion=view.distortion)
        return tuple(max(0.,min(1.,v/size)) for v,size in zip(p,view.source_size))

    def canvas_point(self,point):
        view=self.view
        return source_point(tuple(v*s for v,s in zip(point,view.source_size)),view.source_size,view.composition,inverse=True,distortion=view.distortion)

    def screen_point(self,point):
        mapped=self.canvas_point(point)
        if mapped is None:return None
        x,y=mapped;view=self.view
        scale=view.zoom/view.devicePixelRatioF()
        return QPointF(view.width()/2+(x-view.center[0])*scale,view.height()/2+(y-view.center[1])*scale)

    def eventFilter(self,watched,event):
        try:
            return self._handle_event(watched,event)
        except RuntimeError:
            return False

    def _handle_event(self,watched,event):
        view=self.view;editor=self.controller.editor
        kind=event.type()
        if kind in (QEvent.Type.Hide,QEvent.Type.FocusOut):self.finish()
        if kind==QEvent.Type.KeyPress and event.key()==Qt.Key.Key_Escape:
            if self.drag is not None or self.new_drawing:
                self.finish(cancel=True);self.new_drawing=False
                editor.edit_on_photo.setChecked(False);return True
        if self.drag is not None and (kind==QEvent.Type.Wheel or
                (kind==QEvent.Type.KeyPress and event.key()==Qt.Key.Key_Space)):
            self.finish()
        if not editor.isEnabled() or not editor.edit_on_photo.isChecked() or not all(view.source_size):return False
        if kind==QEvent.Type.MouseButtonPress and event.button()==Qt.MouseButton.LeftButton and not view._space:
            composed=view.source_at(event.position())
            if not (0<=composed[0]<=view.full_size[0] and 0<=composed[1]<=view.full_size[1]):return False
            current=editor.selected
            mode='new' if self.new_drawing else None
            if mode is None and current:
                a,b=self.screen_point(current.start),self.screen_point(current.end)
                center=self.screen_point(tuple((s+e)/2 for s,e in zip(current.start,current.end)))
                for key,p in (('start',a),('end',b),('move',center)):
                    if p is not None and (p-event.position()).manhattanLength()<18:mode=key;break
            if mode:
                self.drag=(mode,event.position(),self.normalized(event.position()),current)
                self.started=False
                if mode!='new':self.controller.drawing_id=current.id
                view.setFocus();return True
        if kind==QEvent.Type.MouseMove and self.drag is not None:
            mode,origin,point,current=self.drag
            if mode=='new' and (origin-event.position()).manhattanLength()<8:return True
            end=self.normalized(event.position())
            if mode=='new':start=point
            elif mode=='start':start,end=end,current.end
            elif mode=='end':start=current.start
            else:
                dx,dy=end[0]-point[0],end[1]-point[1]
                dx=max(-min(current.start[0],current.end[0]),min(1-max(current.start[0],current.end[0]),dx))
                dy=max(-min(current.start[1],current.end[1]),min(1-max(current.start[1],current.end[1]),dy))
                start=(current.start[0]+dx,current.start[1]+dy);end=(current.end[0]+dx,current.end[1]+dy)
            if math.dist(start,end)<1e-5:return True
            radial=self.controller.drawing_radial if mode=='new' else isinstance(current,RadialGradient)
            if radial and any(abs(a-b)<1e-5 for a,b in zip(start,end)):return True
            if not self.started:
                self.controller.window._begin_parameter_gesture('local_gradients');self.started=True
            self.controller.update_gradient(start,end)
            return True
        if kind==QEvent.Type.MouseButtonRelease and self.drag is not None:
            self.finish();return True
        return False

    def finish(self,cancel=False):
        if self.drag is None:return
        was_new=self.drag[0]=='new';self.drag=None
        changed=self.started
        if self.started:
            self.started=False;self.controller.end_geometry(cancel)
        if was_new and (changed or cancel):self.new_drawing=False
        self.view.update()

    def paint(self,painter):
        view=self.view;editor=self.controller.editor;g=editor.selected
        if not editor.isEnabled() or not editor.edit_on_photo.isChecked() or g is None:return
        if self.new_drawing and self.controller.drawing_id!=g.id:return
        a,b=self.canvas_point(g.start),self.canvas_point(g.end)
        scale=view.zoom/view.devicePixelRatioF()
        painter.save()
        pen=QPen(QColor('#79cfd3'),1);pen.setCosmetic(True);painter.setPen(pen)
        if view.distortion:
            self._paint_distorted(painter,g,pen)
            painter.restore()
            return
        if isinstance(g,RadialGradient):
            origin=self.canvas_point((0,0));x=self.canvas_point((1,0));y=self.canvas_point((0,1))
            transform=QTransform(x[0]-origin[0],x[1]-origin[1],y[0]-origin[0],y[1]-origin[1],*origin)
            cx,cy=((s+e)/2 for s,e in zip(g.start,g.end))
            rx,ry=(abs(e-s)/2 for s,e in zip(g.start,g.end))
            guide=QPen(QColor(121,207,211,100),1,Qt.PenStyle.DotLine);guide.setCosmetic(True)
            painter.setPen(guide)
            rot=QTransform().translate(cx,cy).rotate(getattr(g,'angle',0.)).translate(-cx,-cy)
            bounds=QPainterPath();bounds.addRect(QRectF(cx-rx,cy-ry,2*rx,2*ry))
            painter.drawPath(transform.map(rot.map(bounds)));painter.setPen(pen)
            ang=math.radians(getattr(g,'angle',0.));ca,sa=math.cos(ang),math.sin(ang)
            for factor in (1.,1.-g.feather):
                if factor<=0:continue
                path=QPainterPath();path.addEllipse(QRectF(cx-rx*factor,cy-ry*factor,2*rx*factor,2*ry*factor))
                painter.drawPath(transform.map(rot.map(path)))
        else:
            painter.drawLine(QPointF(*a),QPointF(*b))
            dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
            for point in (a,b):
                tx,ty=-dy/length*max(view.full_size)*2,dx/length*max(view.full_size)*2
                painter.drawLine(QPointF(point[0]-tx,point[1]-ty),QPointF(point[0]+tx,point[1]+ty))
        for point in (a,b,((a[0]+b[0])/2,(a[1]+b[1])/2)):
            painter.setBrush(QColor('#202020'));painter.drawEllipse(QPointF(*point),5/scale,5/scale)
        painter.restore()

    def _paint_distorted(self,painter,g,pen):
        """Sample source guides adaptively; never stretch an affine ellipse."""
        view=self.view
        cx,cy=((s+e)/2 for s,e in zip(g.start,g.end))
        def draw(point_at):
            for segment in adaptive_segments(point_at,self.canvas_point,view.zoom):
                if len(segment)<2:continue
                path=QPainterPath(QPointF(*segment[0]))
                for point in segment[1:]:path.lineTo(QPointF(*point))
                painter.drawPath(path)
        if isinstance(g,RadialGradient):
            rx,ry=(abs(e-s)/2 for s,e in zip(g.start,g.end))
            ang=math.radians(getattr(g,'angle',0.));ca,sa=math.cos(ang),math.sin(ang)
            for factor in (1.,1.-g.feather):
                if factor<=0:continue
                draw(lambda t,f=factor: (cx+(rx*f*math.cos(t*2*math.pi)*ca-ry*f*math.sin(t*2*math.pi)*sa),cy+(rx*f*math.cos(t*2*math.pi)*sa+ry*f*math.sin(t*2*math.pi)*ca)))
        else:
            draw(lambda t: tuple(a+(b-a)*t for a,b in zip(g.start,g.end)))
            w,h=view.source_size
            dx,dy=(g.end[0]-g.start[0])*w,(g.end[1]-g.start[1])*h
            length=math.hypot(dx,dy)
            # Guides are perpendicular in original pixels, then warped together
            # with the source. Clip in source first, viewport in ImageView.paint.
            for point in (g.start,g.end):
                vx,vy=-dy/length/w,dx/length/h
                lo,hi=-max(w,h)*2,max(w,h)*2
                for origin,direction in zip(point,(vx,vy)):
                    if direction:
                        a,b=sorted((-origin/direction,(1-origin)/direction))
                        lo,hi=max(lo,a),min(hi,b)
                if lo<hi:
                    draw(lambda t,point=point,lo=lo,hi=hi: (point[0]+vx*(lo+(hi-lo)*t),point[1]+vy*(lo+(hi-lo)*t)))
        scale=view.zoom/view.devicePixelRatioF()
        for source in (g.start,g.end,(cx,cy)):
            point=self.canvas_point(source)
            if point is not None:
                painter.setBrush(QColor('#202020'))
                painter.drawEllipse(QPointF(*point),5/scale,5/scale)


def adaptive_segments(point_at, mapper, zoom, *, max_vertices=4096, max_depth=10):
    """Curved source guide -> visible polylines with <=.5 physical-pixel error.

    None breaks a segment at the corrected footprint; no false connecting line
    crosses invisible points. Bounded subdivision also limits painting work.
    """
    vertices=[]
    def append(point):
        if len(vertices)<max_vertices:vertices.append(point)
    def visit(t0,a,t1,b,depth):
        if len(vertices)>=max_vertices:return
        tm=(t0+t1)/2;mid=mapper(point_at(tm))
        if a is not None and b is not None and mid is not None:
            error=math.dist(mid,((a[0]+b[0])/2,(a[1]+b[1])/2))*zoom
            split=error>.5
        else:
            split=not (a is None and b is None and mid is None)
        if split and depth<max_depth:
            visit(t0,a,tm,mid,depth+1)
            visit(tm,mid,t1,b,depth+1)
        else:
            # At the cap omit an unresolved span rather than drawing a bad chord.
            if split:append(None)
            append(b)
    count=32
    first=mapper(point_at(0.));append(first)
    for i in range(count):
        end=mapper(point_at((i+1)/count))
        visit(i/count,first,(i+1)/count,end,0)
        first=end
    segments=[];current=[]
    for point in vertices:
        if point is None:
            if current:segments.append(current)
            current=[]
        else:current.append(point)
    if current:segments.append(current)
    return segments
