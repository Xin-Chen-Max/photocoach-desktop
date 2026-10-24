"""Endpoint tone, vibrance, sensor picker and current-state UI contracts."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import copy
import json
from dataclasses import replace
from pathlib import Path
import threading

import numpy as np
import pytest
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from photocoach.adjustments import render_edit, render_photo
from photocoach.color_mixer import ColorMixer
from photocoach.composition import Composition, output_size, sample_composition
from photocoach.imaging import RawOptions, quantize_srgb
from photocoach.state import EditValues, StateStore
from photocoach.project import snapshot_project, upgrade_project, validate_project, write_project, read_project
from photocoach.safe_io import sha256_file
from photocoach.preview import DetailRequest, render_region
from photocoach.white_balance import balance_coordinates, balance_gains, sensor_position, estimate_neutral_gains
from photocoach.coach import build_request
from photocoach.coach_view import related_control
from photocoach.app import MainWindow, Job


@pytest.mark.parametrize("whites", [-1, 0, 1])
@pytest.mark.parametrize("blacks", [-1, 0, 1])
def test_endpoint_monotonic_gray_ramp(whites, blacks):
    source = np.repeat(np.linspace(0, 1, 2000, dtype=np.float32)[None,:,None], 3, axis=2)
    result = render_edit(source, EditValues(whites=whites, blacks=blacks))
    assert np.isfinite(result).all() and np.diff(result[0,:,0]).min() >= -3e-7
    np.testing.assert_array_equal(result[:,:,0], result[:,:,2])
    assert result.min() >= 0 and result.max() <= 1


@pytest.mark.parametrize("name", ["whites", "blacks", "vibrance"])
@pytest.mark.parametrize("value", [True, "0", 1.01, -1.01, float("nan")])
def test_new_controls_strict(name, value):
    with pytest.raises(ValueError): EditValues(**{name: value})


def test_endpoint_formula_order_and_vibrance_gray_and_color_weighting():
    source = np.random.default_rng(4).random((5, 13, 3), dtype=np.float32)
    encoded = render_edit(source, EditValues())
    expected = np.clip(encoded+.25*.6*encoded**4-.25*.4*(1-encoded)**4, 0, 1)
    np.testing.assert_allclose(render_edit(source, EditValues(whites=.6, blacks=-.4)), expected, atol=1e-7)
    gray = np.full((10, 12, 3), .18, np.float32)
    for value in (-1, 1):
        result = render_edit(gray, EditValues(vibrance=value))
        np.testing.assert_allclose(result, render_edit(gray, EditValues()), atol=2e-7)
    muted = np.array([[[.3,.25,.25],[.3,.01,.01]]], np.float32)
    normal = render_edit(muted, EditValues())
    vibrant = render_edit(muted, EditValues(vibrance=1))
    relative_change = (vibrant.max(2)-vibrant.min(2))/(normal.max(2)-normal.min(2))
    assert relative_change[0,0] > relative_change[0,1]


@pytest.mark.parametrize("step", [1, 3])
@pytest.mark.parametrize("angle", [0, 3.25])
def test_parallel_strips_exact_serial_pixels(step, angle):
    source = np.random.default_rng(3).random((1300, 1900, 3), dtype=np.float32)
    source.setflags(write=False)
    values = EditValues(whites=.2, blacks=-.1, vibrance=.3, shadows=.4,
        composition=Composition(angle=angle),
        color_mixer=ColorMixer().with_value(5, 1, -25), rgb_curve=((0,0),(.4,.5),(1,1)))
    w,h = output_size((1900,1300), values.composition)
    request = DetailRequest(1, (w,h), (0,0,w,h), step)
    np.testing.assert_array_equal(render_region(source, values, request, workers=1),
                                  render_region(source, values, request, workers=4))
    if step == 1:
        np.testing.assert_array_equal(render_region(source, values, request), quantize_srgb(render_photo(source, values)))


def test_gain_coordinates_round_trip_without_clipping():
    for gains in ((1,1,1),(.25,4,4),(4,.25,4),(2,1,.7),(.25,.25,4)):
        result = balance_gains(*balance_coordinates(gains))
        np.testing.assert_allclose(np.array(result)/result[1], np.array(gains)/gains[1], atol=1e-12)
        assert min(result) >= .25 and max(result) <= 4
    assert balance_gains(.5, 0)[0] > balance_gains(.5, 0)[2]
    assert balance_gains(0, .5)[0] > balance_gains(0, .5)[1]
    with pytest.raises(ValueError): balance_gains(2, 4)


@pytest.mark.parametrize("flip", range(8))
@pytest.mark.parametrize("angle", [0, 5.5])
def test_picker_coordinates_follow_actual_composition_and_orientation(flip, angle):
    yy,xx = np.mgrid[:100,:150]
    image = np.stack((xx,yy,xx*0), axis=2).astype(np.float32)
    oriented = image[::-1] if flip & 2 else image
    if flip & 1: oriented = oriented[:,::-1]
    if flip & 4: oriented = oriented.transpose(1,0,2)
    comp = Composition((.1,.1,.9,.9), angle)
    sampled = sample_composition(oriented, comp)
    for x,y in ((10,11),(sampled.shape[1]//2,sampled.shape[0]//2)):
        sx,sy = sensor_position((150,100),flip,comp,(x,y))
        np.testing.assert_allclose((sx,sy), sampled[y,x,:2], atol=.501)


def neutral_patch():
    colors = np.tile(np.array([[0,1],[3,2]]), (16,16))
    data = np.array([1064,2064,1564,2064], np.uint16)[colors]
    return data, colors


def test_neutral_sensor_estimation_rejects_dark_clip_and_texture():
    data,colors = neutral_patch()
    gains = estimate_neutral_gains(data,colors,[64]*4,16000,[2,1,1.5,1])
    np.testing.assert_allclose(gains, (1,1,8/9), rtol=1e-10)
    for bad in (np.full_like(data,64), np.full_like(data,15900), data.copy()):
        if bad is not data and np.array_equal(bad,data): bad[:16] = 70
        with pytest.raises(ValueError): estimate_neutral_gains(bad,colors,[64]*4,16000,[2,1,1.5,1])


def make_store(tmp_path):
    source=tmp_path/'test.ARW';source.write_bytes(b'read only fixture')
    store=StateStore();store.activate(source,sha256_file(source))
    return store


def test_ten_new_edits_restore_history_and_ai_snapshot(tmp_path):
    store=make_store(tmp_path);before=store.current
    for i in range(10):
        name=('whites','blacks','vibrance')[i%3]
        store.begin_drag(name);store.set_parameter(name,(i+1)/20);store.end_drag()
    values=store.current.values
    assert len(store.undo_stack)==10
    store.undo();store.redo();assert store.current.values==values
    target=tmp_path/'edit.photocoach';write_project(target,snapshot_project(store))
    restored=read_project(target)
    assert restored.current.values==values and restored.undo_stack==store.undo_stack
    pixels=np.zeros((4,4,3),np.uint8)
    request=build_request(store.current,before,store.events,pixels,pixels)
    for name in ('whites','blacks','vibrance'):
        assert request.parameters[name]==getattr(values,name)
        assert related_control(f'Adjust {name} slightly')==name
    assert related_control('Reduce vibrance slightly')=='vibrance'
    original=snapshot_project(store)
    for name in ('whites','blacks','vibrance'):
        damaged=copy.deepcopy(original)
        del damaged['state']['parameters'][name]
        with pytest.raises(ValueError):validate_project(damaged)
    # Build a genuine v4 audit with neutral new controls, preserving nonneutral HSL.
    oldstore=make_store(tmp_path)
    oldstore.set_parameter('color_mixer',ColorMixer().with_value(2,2,20));oldstore.reset();oldstore.undo()
    old=snapshot_project(oldstore)
    def walk(v):
        if isinstance(v,dict):
            if 'white_balance' in v:
                v.pop('local_gradients')
                for name in ('whites','blacks','vibrance'): v.pop(name)
            for item in v.values(): walk(item)
        elif isinstance(v,list):
            for item in v:walk(item)
    walk(old);old['schema_version']=4;old['pipeline_version']='phase1-sdr-srgb-v4'
    unchanged=copy.deepcopy(old)
    migrated=upgrade_project(old);validate_project(migrated)
    assert old==unchanged and migrated['state']['parameters']==oldstore.current.values.to_dict()
    old['source']['byte_length']=oldstore.current.source_path.stat().st_size
    legacy=tmp_path/'legacy-v4.photocoach';legacy.write_text(json.dumps(old),encoding='utf-8')
    legacy_hash=sha256_file(legacy)
    restored=read_project(legacy)
    assert restored.current.values==oldstore.current.values
    assert restored.undo_stack==oldstore.undo_stack and restored.redo_stack==oldstore.redo_stack
    assert restored.load_warnings and sha256_file(legacy)==legacy_hash
    store.reset();assert store.current.values==EditValues()
    store.undo();assert store.current.values==values


@pytest.fixture
def editor(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    w=MainWindow();w.store=make_store(tmp_path)
    w._set_controls(True);w._sync_controls();w.preview.configure_photo((150,100))
    monkeypatch.setattr(w,'request_preview',lambda **kw:None)
    monkeypatch.setattr(w.network_pool,'start',lambda *_:pytest.fail('No network'))
    yield w,app
    for timer in (w._render_timer,w._settle_timer,w._detail_timer):timer.stop()
    w._close_approved=True;w.close()


def test_ui_new_drags_baseline_view_and_wb_drafts(editor):
    w,app=editor
    w.preview.actual_size();view=(w.preview.zoom,w.preview.center)
    for name in ('whites','blacks','vibrance'):
        w.change_compare(True);old=len(w.store.undo_stack)
        slider=w.adjustment_sliders[name]
        slider.setSliderDown(True)
        for value in (5,10,15):slider.setValue(value)
        slider.setSliderDown(False)
        assert len(w.store.undo_stack)==old+1 and not w.store.current.compare_baseline
        assert (w.preview.zoom,w.preview.center)==view
        w.undo();assert getattr(w.store.current,name)==0
        w.redo();assert getattr(w.store.current,name)==.15
    w.change_compare(True);state=w.store.current
    for slider,value in zip(w.balance_sliders,(20,-10)): slider.setValue(value)
    assert w.store.current==state  # drafts don't alter state or comparison
    w.change_wb()
    assert not w.store.current.compare_baseline and w.store.current.raw_options.wb_mode=='custom'
    assert (w.preview.zoom,w.preview.center)==view
    assert w.api_budget.attempts==0
    w.undo();assert w.store.current.raw_options==RawOptions()


def test_picker_cancel_outside_click_success_and_stale(editor,monkeypatch):
    w,app=editor;w.show();app.processEvents()
    w.pick_wb.setChecked(True)
    QTest.keyClick(w.preview,Qt.Key.Key_Escape)
    assert not w.pick_wb.isChecked() and not w.preview.picking_neutral
    # Outside-photo pixel is ignored, without an operation or worker.
    jobs=[];monkeypatch.setattr(w,'_queue',jobs.append)
    old=w.store.current
    w.pick_wb.setChecked(True)
    QTest.mouseClick(w.preview,Qt.MouseButton.LeftButton,pos=QPoint(1,1))
    assert not jobs and w.store.current==old and w.preview.picking_neutral
    QTest.keyClick(w.preview,Qt.Key.Key_Escape)
    w._sample_wb((50,30));assert len(jobs)==1
    w._active=(None,jobs.pop());w._done(RawOptions('custom',(1.12345678,1,.87654321)),None)
    assert w.store.current.raw_options.wb_mode=='custom' and len(w.store.undo_stack)==1
    sampled=w.store.current
    w.change_wb()
    assert w.store.current==sampled and len(w.store.undo_stack)==1
    w._sample_wb((50,30));job=jobs.pop()
    w.change_parameter('whites',.1)
    latest=w.store.current
    w._active=(None,job);w._done(RawOptions('custom',(2,1,1)),None)
    assert w.store.current==latest and not w._wb_sampling
    w._sample_wb((50,30));w._active=(None,jobs.pop())
    w._done(None,ValueError('The sample is too dark.'))
    assert w.store.current==latest and 'too dark' in w.status.text()


def test_picker_result_from_previous_photo_session_is_discarded(editor,monkeypatch):
    w,app=editor;jobs=[];monkeypatch.setattr(w,'_queue',jobs.append)
    w._sample_wb((50,30));job=jobs.pop()
    w.store.activate(w.store.current.source_path,w.store.current.photo_id)
    latest=w.store.current
    w._active=(None,job);w._done(RawOptions('custom',(2,1,1)),None)
    assert w.store.current==latest and not w.store.undo_stack and not w._wb_sampling


def test_picker_cannot_edit_while_output_state_is_frozen(editor,monkeypatch):
    w,app=editor;jobs=[];monkeypatch.setattr(w,'_queue',jobs.append)
    w._sample_wb((50,30));job=jobs.pop();captured=w.store.current
    w._exporting=True;w._set_controls(False)
    w._active=(None,job);w._done(RawOptions('custom',(2,1,1)),None)
    assert w.store.current==captured and not w.store.undo_stack and not w.pick_wb.isEnabled()
    w._exporting=False;w._set_controls(True)


def test_scroll_over_edit_controls_never_changes_parameters(editor):
    w,app=editor;w.show();app.processEvents()
    w.advanced_wb.setChecked(True)
    w.wb_mode.setCurrentIndex(1)
    w.color_mixer.toggle.setChecked(True)
    controls=(w.exposure_slider,*w.balance_sliders,*w.gains,
              *w.adjustment_sliders.values(),*w.color_mixer.sliders)
    for control in controls:
        w.edit_scroll.ensureWidgetVisible(control);app.processEvents()
        control.setFocus()
        old=control.value();state=w.store.current
        event=QWheelEvent(QPointF(control.rect().center()),
            QPointF(control.mapToGlobal(control.rect().center())),QPoint(),QPoint(0,-120),
            Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        QApplication.sendEvent(control,event)
        assert control.value()==old and w.store.current==state
        assert not event.isAccepted()
    # Keyboard remains an intentional editing mechanism.
    slider=w.adjustment_sliders['vibrance'];slider.setFocus()
    QTest.keyClick(slider,Qt.Key.Key_Right)
    assert w.store.current.vibrance==.01
