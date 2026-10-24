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
        p=source_point(view.source_at(position),view.source_size,view.composition)
        return tuple(max(0.,min(1.,v/size)) for v,size in zip(p,view.source_size))

    def canvas_point(self,point):
        view=self.view
        return source_point(tuple(v*s for v,s in zip(point,view.source_size)),view.source_size,view.composition,inverse=True)

    def screen_point(self,point):
        x,y=self.canvas_point(point);view=self.view
        scale=view.zoom/view.devicePixelRatioF()
        return QPointF(view.width()/2+(x-view.center[0])*scale,view.height()/2+(y-view.center[1])*scale)

    def eventFilter(self,watched,event):
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
                center=(a+b)/2
                for key,p in (('start',a),('end',b),('move',center)):
                    if (p-event.position()).manhattanLength()<18:mode=key;break
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
        if isinstance(g,RadialGradient):
            origin=self.canvas_point((0,0));x=self.canvas_point((1,0));y=self.canvas_point((0,1))
            transform=QTransform(x[0]-origin[0],x[1]-origin[1],y[0]-origin[0],y[1]-origin[1],*origin)
            cx,cy=((s+e)/2 for s,e in zip(g.start,g.end))
            rx,ry=(abs(e-s)/2 for s,e in zip(g.start,g.end))
            guide=QPen(QColor(121,207,211,100),1,Qt.PenStyle.DotLine);guide.setCosmetic(True)
            painter.setPen(guide)
            bounds=QPainterPath();bounds.addRect(QRectF(cx-rx,cy-ry,2*rx,2*ry))
            painter.drawPath(transform.map(bounds));painter.setPen(pen)
            for factor in (1.,1.-g.feather):
                if factor<=0:continue
                path=QPainterPath();path.addEllipse(QRectF(cx-rx*factor,cy-ry*factor,2*rx*factor,2*ry*factor))
                painter.drawPath(transform.map(path))
        else:
            painter.drawLine(QPointF(*a),QPointF(*b))
            dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
            for point in (a,b):
                tx,ty=-dy/length*max(view.full_size)*2,dx/length*max(view.full_size)*2
                painter.drawLine(QPointF(point[0]-tx,point[1]-ty),QPointF(point[0]+tx,point[1]+ty))
        for point in (a,b,((a[0]+b[0])/2,(a[1]+b[1])/2)):
            painter.setBrush(QColor('#202020'));painter.drawEllipse(QPointF(*point),5/scale,5/scale)
        painter.restore()
