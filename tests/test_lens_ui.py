"""Lens interaction, geometry navigation and unchanged coach/privacy behavior."""
from dataclasses import replace
import math
import numpy as np
import pytest
from PySide6.QtCore import QPointF
from photocoach.lens import LensCorrections
from photocoach.gradients import RadialGradient, source_point
from photocoach.gradient_view import adaptive_segments
from photocoach.coach_view import related_control
from test_progressive_preview import view, wait


def test_collapsed_controls_navigation_no_edits_or_network(view):
    w,app=view
    before=w.store.current;events=list(w.store.events)
    assert not w.lens_section.toggle.isChecked() and w.lens_section.body.isHidden()
    w._locate_control('lens_corrections.vignetting_ev');app.processEvents()
    assert w.store.current==before and w.store.events==events
    assert related_control('Increase Vignetting slightly')=='lens_corrections.vignetting_ev'
    assert related_control('Adjust distortion')=='lens_corrections.distortion'
    assert w.api_budget.attempts==0


def test_vignette_drag_preserves_view_baseline_returns_edited_one_undo(view):
    w,app=view
    w.preview.actual_size();w.preview.pan(10,5)
    pose=(w.preview.center,w.preview.zoom)
    w.compare.setChecked(True)
    slider=w.lens_editor.sliders['vignetting_ev']
    slider.sliderPressed.emit()
    for value in range(10,101,10):slider.setValue(value)
    slider.sliderReleased.emit()
    wait(app,lambda:w._active is None and not w._pending)
    assert not w.store.current.compare_baseline
    assert w.store.current.lens_corrections==LensCorrections(0,1)
    assert len(w.store.undo_stack)==1
    assert (w.preview.center,w.preview.zoom)==pose
    w.undo();wait(app,lambda:w._active is None)
    assert w.store.current.lens_corrections==LensCorrections()
    assert (w.preview.center,w.preview.zoom)==pose
    w.redo();wait(app,lambda:w._active is None)
    assert w.store.current.lens_corrections==LensCorrections(0,1)
    assert (w.preview.center,w.preview.zoom)==pose


def test_distortion_reset_fit_history_reset_and_restore_controls(view):
    w,app=view
    w.preview.actual_size()
    slider=w.lens_editor.sliders['distortion']
    slider.sliderPressed.emit()
    for value in (10,20,30,40,50):slider.setValue(value)
    slider.sliderReleased.emit()
    wait(app,lambda:w._active is None)
    assert w.preview.fit_mode and w.preview.distortion==50
    assert len(w.store.undo_stack)==1
    w.preview.actual_size();w.undo();wait(app,lambda:w._active is None)
    assert w.preview.fit_mode and w.preview.distortion==0 and slider.value()==0
    w.redo();wait(app,lambda:w._active is None)
    assert slider.value()==50
    w.lens_editor.reset.click();wait(app,lambda:w._active is None)
    assert len(w.store.undo_stack)==2 and w.store.current.lens_corrections==LensCorrections()
    w.undo();wait(app,lambda:w._active is None)
    assert w.preview.distortion==50 and slider.value()==50


def test_distorted_mask_center_uses_original_midpoint_and_hidden_handles(view):
    w,app=view
    w.change_parameter('lens_corrections',LensCorrections(-100),sync=True)
    wait(app,lambda:w._active is None)
    layer=w.local_controller.layer
    g=RadialGradient('r','Rock',(.1,.2),(.6,.8))
    center=tuple((a+b)/2 for a,b in zip(g.start,g.end))
    expected=source_point((center[0]*300,center[1]*200),(300,200),w.preview.composition,inverse=True,distortion=-100)
    assert layer.canvas_point(center)==pytest.approx(expected)
    a,b=layer.canvas_point(g.start),layer.canvas_point(g.end)
    assert math.dist(expected,((a[0]+b[0])/2,(a[1]+b[1])/2))>.1
    assert layer.canvas_point((0,0)) is None and layer.screen_point((0,0)) is None
    w.change_parameter('local_gradients',(g,),sync=True)
    wait(app,lambda:w._active is None)
    w.local_gradients.edit_on_photo.setChecked(True)
    # Painting a curved contour including an invisible endpoint must be safe.
    w.preview.grab()
    point=layer.screen_point(center)
    assert layer.normalized(point)==pytest.approx(center,abs=1e-10)


def test_adaptive_outline_error_budget_and_no_bridge_over_hidden_arc():
    circle=lambda t:(100*math.cos(2*math.pi*t),100*math.sin(2*math.pi*t))
    segments=adaptive_segments(circle,lambda p:p,2)
    assert len(segments)==1
    for a,b in zip(segments[0],segments[0][1:]):
        midpoint=((a[0]+b[0])/2,(a[1]+b[1])/2)
        assert (100-math.hypot(*midpoint))*2<=.5+1e-8
    hidden=adaptive_segments(circle,lambda p:None if p[0]>50 else p,2)
    assert all(p[0]<=50 for segment in hidden for p in segment)
    assert sum(map(len,adaptive_segments(circle,lambda p:p,100000,max_vertices=40)))<=40


@pytest.mark.parametrize('quality',['detail','adjusting'])
def test_composition_result_is_painted_without_a_quick_image(view,quality):
    from photocoach.composition import Composition
    from PySide6.QtGui import QColor
    w,app=view
    w.preview.resize(400,300)
    w.preview.set_composition(Composition(angle=5))
    assert w.preview._quick.isNull()
    request=w.preview.detail_request()
    pixels=np.full((200,300,3),(210,75,40),np.uint8)
    getattr(w.preview,'display_'+quality)(pixels,request)
    assert not w.preview._message
    captured=w.preview.grab().toImage()
    center=captured.pixelColor(captured.width()//2,captured.height()//2)
    assert center.red()==210 and center.green()==75


def test_distortion_keeps_last_visible_raster_while_refining(view):
    w,app=view
    before=w.preview._quick.cacheKey()
    w.preview.set_composition(w.preview.composition,distortion=30)
    assert w.preview._quick.cacheKey()==before and not w.preview._message
    w.preview.invalidate_edit('Refining')
    assert w.preview._quick.cacheKey()==before
