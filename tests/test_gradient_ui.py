"""Mask selection/overlays are passive; geometry and tones use grouped history."""
import os
import json
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from PySide6.QtCore import Qt,QPoint,QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from photocoach.app import MainWindow
from photocoach.settings import SettingsStore
from photocoach.gradients import LinearGradient,RadialGradient,OverlaySpec,overlay_pixels,RenderGeometry
from photocoach.composition import Composition
from photocoach.coach import CoachFeedback
from photocoach.coach_view import related_control


@pytest.fixture
def window(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    w=MainWindow(settings_store=SettingsStore(tmp_path/'unused.json'))
    monkeypatch.setattr(w,'request_preview',lambda **kwargs:None)
    monkeypatch.setattr(w,'request_detail',lambda:None)
    monkeypatch.setattr(w.network_pool,'start',lambda *a:pytest.fail('No network allowed'))
    w.store.activate(Path('synthetic.ARW'),'photo-a')
    w.preview.configure_photo((1200,800))
    w.preview.display(np.full((400,600,3),80,np.uint8))
    w._set_controls(True);w._sync_controls();w.resize(1280,800);w.show()
    app.processEvents()
    yield w
    w._close_approved=True;w.close();w.deleteLater();app.processEvents()


def draw(window,start=QPoint(250,180),end=QPoint(500,310)):
    view=window.preview
    window.local_gradients.add.click()
    QTest.mousePress(view,Qt.MouseButton.LeftButton,pos=start)
    QTest.mouseMove(view,end)
    QTest.mouseRelease(view,Qt.MouseButton.LeftButton,pos=end)


def test_drawing_grouped_tones_baseline_cancel_and_undo(window):
    w=window;w.change_compare(True)
    draw(w)
    assert not w.store.current.compare_baseline
    assert len(w.store.current.local_gradients)==1 and len(w.store.undo_stack)==1
    g=w.store.current.local_gradients[0]
    assert w.local_gradients.selected.id==g.id
    center,zoom=w.preview.center,w.preview.zoom
    slider=w.local_gradients.sliders['exposure_ev']
    slider.sliderPressed.emit()
    slider.setValue(40);slider.setValue(65);slider.setValue(80)
    slider.sliderReleased.emit()
    assert len(w.store.undo_stack)==2 and w.store.current.local_gradients[0].exposure_ev==.8
    assert (w.preview.center,w.preview.zoom)==(center,zoom)
    w.undo();assert w.store.current.local_gradients[0].exposure_ev==0
    w.redo();assert w.store.current.local_gradients[0].exposure_ev==.8
    w.local_gradients.add.click()
    QTest.mousePress(w.preview,Qt.MouseButton.LeftButton,pos=QPoint(270,200))
    QTest.mouseMove(w.preview,QPoint(400,330))
    QTest.keyClick(w.preview,Qt.Key.Key_Escape)
    assert len(w.store.current.local_gradients)==1 and len(w.store.undo_stack)==2
    w.local_gradients.reset.click()
    assert w.store.current.local_gradients[0].exposure_ev==0 and len(w.store.undo_stack)==3
    w.local_gradients.delete.click();assert w.store.current.local_gradients==()
    w.undo();assert len(w.store.current.local_gradients)==1


def test_passive_selection_overlay_limit_names_and_no_click_creation(window):
    w=window;view=w.preview
    w.local_gradients.add.click()
    QTest.mouseClick(view,Qt.MouseButton.LeftButton,pos=QPoint(300,200))
    assert not w.store.current.local_gradients and len(w.store.undo_stack)==0
    assert w.local_controller.layer.new_drawing
    for i in range(4):draw(w,QPoint(230+i*10,180),QPoint(480,300+i*10))
    assert len(w.store.current.local_gradients)==4 and not w.local_gradients.add.isEnabled()
    current,events=w.store.current,list(w.store.events)
    w.local_gradients.list.setCurrentIndex(0)
    w.local_gradients.show_mask.setChecked(True)
    w.clipping.setChecked(True)
    assert not w.local_gradients.show_mask.isChecked()
    w.local_gradients.show_mask.setChecked(True)
    assert not w.clipping.isChecked()
    w.local_gradients.done.click()
    assert not w.local_gradients.edit_on_photo.isChecked()
    assert view.cursor().shape()==Qt.CursorShape.ArrowCursor
    assert w.store.current is current and w.store.events==events
    assert w.coach.calls==w.api_budget.attempts==0
    first=w.local_gradients.selected
    w.local_gradients.name.setText(w.store.current.local_gradients[1].name)
    w.local_gradients.name.editingFinished.emit()
    assert w.local_gradients.selected.name==first.name
    assert 'unique' in w.local_gradients.hint.text()


def test_local_advice_never_routes_to_global_exposure(window):
    w=window;draw(w)
    g=w.store.current.local_gradients[0]
    assert related_control(f'Increase Exposure in local gradient {g.name}.',(g,))==f'local_gradients.{g.id}'
    assert related_control('Increase exposure in the sky mask.',())=='local_gradients'
    state=w.store.current;events=list(w.store.events)
    w.coach_text.show_feedback(CoachFeedback(state.photo_id,state.session_id,state.revision,
        'A mask was added.','Local gain is spatial.',f'Increase Exposure in local gradient {g.name}.',
        'Stop if light looks artificial.',(),answer='Use a small local change.'))
    assert f'Go to {g.name}' in w.coach_text.toPlainText()
    w.local_section.set_expanded(False)
    w.coach_text.anchorClicked.emit(QUrl(f'photocoach:locate/local_gradients.{g.id}'))
    assert w.local_section.toggle.isChecked() and w.local_gradients.selected.id==g.id
    assert w.store.current is state and w.store.events==events and w.api_budget.attempts==0


def test_endpoint_move_and_redraw_retain_tones_and_one_undo_step(window):
    w=window;draw(w)
    w.local_gradients.sliders['exposure_ev'].setValue(35)
    original=w.store.current.local_gradients[0]
    view=w.preview;layer=w.local_controller.layer
    start=layer.screen_point(original.start).toPoint()
    depth=len(w.store.undo_stack)
    QTest.mousePress(view,Qt.MouseButton.LeftButton,pos=start)
    QTest.mouseMove(view,start+QPoint(25,15))
    QTest.mouseMove(view,start+QPoint(40,25))
    QTest.mouseRelease(view,Qt.MouseButton.LeftButton,pos=start+QPoint(40,25))
    moved=w.store.current.local_gradients[0]
    assert moved.start!=original.start and moved.end==original.end
    assert moved.exposure_ev==original.exposure_ev and len(w.store.undo_stack)==depth+1
    w.undo();assert w.store.current.local_gradients[0]==original
    w.redo();assert w.store.current.local_gradients[0]==moved
    center=(layer.screen_point(moved.start)+layer.screen_point(moved.end))/2
    center=center.toPoint()
    QTest.mousePress(view,Qt.MouseButton.LeftButton,pos=center)
    QTest.mouseMove(view,center+QPoint(20,10))
    QTest.mouseRelease(view,Qt.MouseButton.LeftButton,pos=center+QPoint(20,10))
    translated=w.store.current.local_gradients[0]
    assert translated.start!=moved.start and translated.end!=moved.end
    assert len(w.store.undo_stack)==depth+2
    w.local_gradients.redraw.click()
    QTest.mousePress(view,Qt.MouseButton.LeftButton,pos=QPoint(300,180))
    QTest.mouseMove(view,QPoint(550,340))
    QTest.mouseRelease(view,Qt.MouseButton.LeftButton,pos=QPoint(550,340))
    redrawn=w.store.current.local_gradients[0]
    assert len(w.store.current.local_gradients)==1 and redrawn.id==original.id
    assert redrawn.exposure_ev==.35 and redrawn.start!=translated.start
    assert len(w.store.undo_stack)==depth+3


def test_overlay_budget_fails_before_allocating_full_image():
    pixels=np.broadcast_to(np.uint8(0),(3000,5000,3))
    geometry=RenderGeometry.full((5000,3000),Composition(),(5000,3000))
    with pytest.raises(MemoryError,match='budget'):overlay_pixels(pixels,geometry,OverlaySpec('clipping'))


def test_radial_creation_feather_grouping_and_passive_shape_choice(window):
    w=window;before=w.store.current;events=list(w.store.events)
    w.local_gradients.new_shape.setCurrentText('Radial')
    assert w.store.current is before and w.store.events==events
    draw(w)
    g=w.store.current.local_gradients[0]
    assert isinstance(g,RadialGradient) and g.feather==.5
    assert w.local_gradients.feather.isVisibleTo(w.local_gradients.controls)
    depth=len(w.store.undo_stack)
    slider=w.local_gradients.feather
    slider.sliderPressed.emit();slider.setValue(60);slider.setValue(75);slider.sliderReleased.emit()
    assert len(w.store.undo_stack)==depth+1 and w.store.current.local_gradients[0].feather==.75
    w.undo();assert w.store.current.local_gradients[0].feather==.5
    w.redo();assert w.store.current.local_gradients[0].feather==.75
    assert related_control('Adjust Radial 1 Exposure.',w.store.current.local_gradients)==f'local_gradients.{g.id}'
    w.local_gradients.new_shape.setCurrentText('Linear')
    assert isinstance(w.store.current.local_gradients[0],RadialGradient)
    draw(w,QPoint(270,210),QPoint(550,330))
    assert type(w.store.current.local_gradients[-1]) is LinearGradient
    assert w.local_gradients.feather.isHidden()
    w.local_gradients.list.setCurrentIndex(0)
    assert not w.local_gradients.feather.isHidden()
    assert w.api_budget.attempts==0

def test_ai_previews_use_real_mask_parameters_but_never_display_overlays(monkeypatch):
    from photocoach.adjustments import render_edit
    from photocoach.imaging import quantize_srgb
    from photocoach.app import PreviewEngine
    from photocoach.imaging import DecodedPhoto,RawOptions
    from photocoach.state import EditState
    from photocoach.api_coach import request_context
    linear=np.full((12,18,3),.12,np.float32);linear.setflags(write=False)
    photo=DecodedPhoto(linear,{'sha256':'a','half_size':False,'output_size':[18,12]},RawOptions(),False,Path('test.ARW'))
    engine=PreviewEngine();monkeypatch.setattr(engine,'photo',lambda *args:(photo,linear))
    before=EditState('a',Path('test.ARW'),'s')
    current=replace(before,revision=1,local_gradients=(LinearGradient("g1","Gradient 1",(.2,.2),(.8,.8),exposure_ev=1),))
    engine.overlay_spec=OverlaySpec('clipping')
    request=engine.coach_request(current,before,[],"Explain Gradient 1")
    expected=quantize_srgb(render_edit(linear,current.values,
        geometry=RenderGeometry.full((18,12),Composition(),(18,12))))
    assert np.array_equal(request.current_preview,expected)
    context=request_context(request)
    assert context['parameters']['local_gradients'][0]['exposure_ev']==1
    assert 'overlay' not in json.dumps(context).lower()
