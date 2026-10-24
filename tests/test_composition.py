"""Composition geometry, persistence, asynchronous drafts and editor contracts."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import copy
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from photocoach.composition import Composition, canvas_size, crop_bounds, output_size, sample_composition
from photocoach.adjustments import render_edit, render_photo
from photocoach.preview import DetailRequest, render_region
from photocoach.state import StateStore, EditValues
from photocoach.project import snapshot_project, write_project, read_project, validate_project, upgrade_project
from photocoach.safe_io import sha256_file
from photocoach.imaging import quantize_srgb, DecodedPhoto, RawOptions
from photocoach.composition_ui import CompositionDialog
from photocoach.settings import SettingsStore
from photocoach import app as desktop


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def spin(application, predicate):
    deadline = time.monotonic()+8
    while not predicate():
        application.processEvents()
        assert time.monotonic() < deadline
        time.sleep(.002)
    application.processEvents()


@pytest.mark.parametrize("crop,angle", [((0,0,1,1),True), ((0,0,1,1),float("nan")),
    ((0,0,1,1),16), ((0,0,1,1),-16), ((0,0,1,1),"2"), ((0,0,.01,1),0),
    ((0,0,1,0),0), ((1,0,0,1),0), ((-1,0,1,1),0), ((0,0,True,1),0), ((0,0,1),0)])
def test_invalid_composition_is_rejected(crop, angle):
    with pytest.raises(ValueError): Composition(crop,angle)


def test_identity_and_native_crop_are_exact_read_only_slices():
    source = np.random.default_rng(1).random((100,150,3),dtype=np.float32)
    source.setflags(write=False)
    values = EditValues(exposure_ev=.6, shadows=.4, highlights=-.5)
    np.testing.assert_array_equal(render_photo(source, values), render_edit(source, values))
    comp = Composition((.2,.1,.8,.9))
    np.testing.assert_array_equal(sample_composition(source,comp), source[10:90,30:120])
    np.testing.assert_array_equal(render_photo(source,replace(values,composition=comp)),
                                  render_edit(source[10:90,30:120],values))
    with pytest.raises(ValueError):
        sample_composition(np.zeros((10,15,3),np.uint8),comp)


@pytest.mark.parametrize("angle", [-15,-4.25,0,5.5,15])
@pytest.mark.parametrize("size", [(153,100),(100,153)])
def test_native_roi_matches_full_export_for_each_geometry(angle,size):
    source = np.random.default_rng(2).random((size[1],size[0],3),dtype=np.float32)
    original = source.copy()
    source.setflags(write=False)
    values = EditValues(exposure_ev=.4,shadows=.5,highlights=-.6,contrast=.2,
                       composition=Composition((.1,.15,.93,.95),angle))
    rendered = render_photo(source,values)
    w,h = output_size(size,values.composition)
    assert rendered.shape == (h,w,3) and np.isfinite(rendered).all()
    rect=(3,5,w-4,h-6)
    roi=render_region(source,values,DetailRequest(0,(w,h),rect,1,size))
    np.testing.assert_array_equal(roi,quantize_srgb(rendered)[5:h-6,3:w-4])
    np.testing.assert_array_equal(source,original)


def test_clockwise_inverse_mapping_and_empty_corner_trimming():
    yy,xx=np.mgrid[:200,:300]
    source=np.stack((xx,yy,np.ones_like(xx)),axis=2).astype(np.float32)
    comp=Composition(angle=10)
    w,h=output_size((300,200),comp)
    assert w<300 and h<200
    image=sample_composition(source,comp)
    # Clockwise rotates a vertical source line right at the top: inverse source
    # x increases with output y, and inverse source y decreases with output x.
    assert image[-1,0,0]>image[0,0,0] and image[0,-1,1]<image[0,0,1]
    assert image[:,:,0].min()>0 and image[:,:,0].max()<299
    assert image[:,:,1].min()>0 and image[:,:,1].max()<199
    np.testing.assert_allclose(image[:,:,2],1,atol=1e-7)


def test_small_crop_preview_does_not_pretend_to_have_native_resolution():
    source=np.full((400,600,3),.18,np.float32)
    values=EditValues(composition=Composition((.4,.4,.6,.6)))
    result=render_photo(source,values,source_size=(6000,4000),max_edge=1600)
    assert result.shape==(80,120,3)  # 1/10 source density, not an upscaled 1200px crop


def make_store(tmp_path):
    source=tmp_path/"synthetic.ARW"
    source.write_bytes(b"test source - not a real RAW")
    store=StateStore()
    store.activate(source,sha256_file(source))
    return store


def test_composition_history_reset_roundtrip_and_strict_v2_upgrade(tmp_path):
    store=make_store(tmp_path)
    store.set_exposure(.4)
    old=snapshot_project(store)
    # Build the exact previous-v2 shape, including its history snapshots.
    def strip(value):
        if isinstance(value,dict):
            if "white_balance" in value:
                value.pop("composition")
                value.pop("color_mixer")
                value.pop("local_gradients")
                for name in ("whites", "blacks", "vibrance"): value.pop(name)
            for item in value.values(): strip(item)
        elif isinstance(value,list):
            for item in value: strip(item)
    strip(old)
    old["schema_version"]=2
    old["pipeline_version"]="phase1-sdr-srgb-v2"
    before=copy.deepcopy(old)
    upgraded=upgrade_project(old)
    validate_project(upgraded)
    assert old==before and upgraded["schema_version"]==7
    bad=copy.deepcopy(old)
    bad["state"]["parameters"]["composition"]={}
    with pytest.raises(ValueError): validate_project(bad)
    store.set_parameter("composition",Composition((.1,.2,.9,.95),3.5))
    committed=store.current.values
    assert len(store.undo_stack)==2
    store.undo()
    assert store.current.composition==Composition()
    store.redo()
    assert store.current.values==committed
    target=tmp_path/"edit.photocoach"
    write_project(target,snapshot_project(store))
    restored=read_project(target)
    assert restored.current.values==committed and restored.undo_stack==store.undo_stack
    restored.reset()
    assert restored.current.composition==Composition()
    assert sha256_file(store.current.source_path)==store.current.photo_id


def test_preview_baseline_export_geometry_and_ai_context(monkeypatch):
    engine=desktop.PreviewEngine()
    store=StateStore()
    store.activate(Path("synthetic.ARW"),"photo")
    before=store.current
    linear=np.random.default_rng(3).random((100,150,3),dtype=np.float32)
    photo=DecodedPhoto(linear,{"sha256":"photo","half_size":False,"output_size":[150,100],
                              "visible_size":[150,100]},RawOptions(),False,store.current.source_path)
    monkeypatch.setattr(engine,"photo",lambda *a:(photo,linear))
    monkeypatch.setattr(engine.cache,"get",lambda *a,**kw:photo)
    comp=Composition((.1,.1,.9,.9),4)
    store.set_parameter("composition",comp)
    store.set_exposure(.4)
    edited=engine.render(store.current)[0]
    store.set_compare(True)
    baseline=engine.render(store.current)[0]
    assert edited.shape==baseline.shape and not np.array_equal(edited,baseline)
    request=engine.coach_request(store.current,before,store.events,"How is this crop?")
    assert request.parameters["composition"]==comp.to_dict()
    assert request.before_parameters["composition"]==Composition().to_dict()
    assert request.before_preview.shape!=request.current_preview.shape
    np.testing.assert_array_equal(request.current_preview,edited)


def test_dialog_drafts_aspect_drag_angle_and_cancel(application):
    values=EditValues()
    dialog=CompositionDialog(None,np.full((200,300,3),.2,np.float32),(6000,4000),values)
    dialog.show()
    spin(application,lambda:dialog.apply_button.isEnabled())
    box=dialog.canvas.crop_rect()
    QTest.mousePress(dialog.canvas,Qt.MouseButton.LeftButton,pos=box.topLeft().toPoint())
    QTest.mouseMove(dialog.canvas,(box.topLeft()+QPoint(45,30)).toPoint())
    QTest.mouseRelease(dialog.canvas,Qt.MouseButton.LeftButton)
    assert dialog.composition.crop[0]>0 and dialog.composition.crop[1]>0
    dialog.ratio.setCurrentText("1:1")
    w,h=output_size((6000,4000),dialog.composition)
    assert abs(w-h)<=2
    dialog.angle.setValue(4.25)
    assert not dialog.apply_button.isEnabled()
    spin(application,lambda:dialog.apply_button.isEnabled())
    assert dialog.composition.angle==4.25
    dialog.reject()
    assert dialog.result()==QDialog.DialogCode.Rejected and values==EditValues()
    dialog.deleteLater()


def test_dialog_only_latest_angle_and_safe_close(application,monkeypatch):
    import threading
    from photocoach import composition_ui as ui
    started,release=threading.Event(),threading.Event()
    real=ui.render_photo
    angles=[]
    def render(linear,values,**kwargs):
        angles.append(values.composition.angle)
        if len(angles)==1:
            started.set()
            assert release.wait(5)
        return real(linear,values,**kwargs)
    monkeypatch.setattr(ui,"render_photo",render)
    dialog=CompositionDialog(None,np.ones((20,30,3),np.float32),(300,200),EditValues())
    dialog.show()
    try:
        spin(application,started.is_set)
        dialog.angle.setValue(1)
        dialog.angle.setValue(5)
    finally:
        release.set()
    spin(application,lambda:dialog.apply_button.isEnabled())
    assert dialog._rendered_angle==5 and angles==[0,5]
    dialog.reject()
    dialog.deleteLater()


@pytest.mark.parametrize("accept",[False,True])
def test_apply_commits_once_cancel_is_inert_and_no_cloud(application,tmp_path,monkeypatch,accept):
    window=desktop.MainWindow(settings_store=SettingsStore(tmp_path/"settings.json"))
    window.store=make_store(tmp_path)
    window._set_controls(True)
    window.preview.configure_photo((6000,4000))
    window.preview.display(np.zeros((100,150,3),np.uint8))
    window.store.set_compare(True)
    start=window.store.current
    comp=Composition((.1,.2,.9,.95),2)
    class FakeDialog:
        composition=comp
        def __init__(self,*args): pass
        def exec(self): return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected
        def deleteLater(self): pass
    monkeypatch.setattr(desktop,"CompositionDialog",FakeDialog)
    monkeypatch.setattr(window,"request_preview",lambda:None)
    window._edit_composition((np.zeros((20,30,3),np.float32),(6000,4000)),start)
    if accept:
        assert window.store.current.composition==comp and len(window.store.undo_stack)==1
        assert not window.store.current.compare_baseline
        assert window.preview.full_size==output_size((6000,4000),comp)
        window.undo()
        assert window.store.current.composition==Composition()
        window.redo()
        assert window.store.current.composition==comp
    else:
        assert window.store.current==start and not window.store.undo_stack
    assert window.api_budget.attempts==window.coach.calls==0
    window._render_timer.stop()
    window._stop_detail()
    window._close_approved=True
    window.close()
    window.deleteLater()


def test_preparation_failure_recovers_controls_and_stale_dialog_does_not_open(application,tmp_path,monkeypatch):
    window=desktop.MainWindow(settings_store=SettingsStore(tmp_path/"settings.json"))
    window.store=make_store(tmp_path)
    window._set_controls(True)
    snapshot=window.store.current
    def fail(state): raise RuntimeError("Synthetic decode failure")
    monkeypatch.setattr(window.engine,"composition_input",fail)
    monkeypatch.setattr(desktop,"CompositionDialog",lambda *a:pytest.fail("No stale/failed dialog"))
    window.open_composition()
    assert window._exporting and not window.composition_button.isEnabled()
    spin(application,lambda:window._active is None)
    assert not window._exporting and window.composition_button.isEnabled()
    assert "Synthetic decode failure" in window.output_status.text()
    window.store.set_exposure(.2)
    window._edit_composition(None,snapshot)
    window._close_approved=True
    window.close()
    window.deleteLater()


def test_cancel_waits_safely_for_running_draft(application,monkeypatch):
    import threading
    from photocoach import composition_ui as ui
    started,release=threading.Event(),threading.Event()
    def render(*args,**kwargs):
        started.set()
        assert release.wait(5)
        return np.zeros((20,30,3),np.float32)
    monkeypatch.setattr(ui,"render_photo",render)
    dialog=CompositionDialog(None,np.zeros((20,30,3),np.float32),(300,200),EditValues())
    dialog.show()
    try:
        spin(application,started.is_set)
        dialog.reject()
        assert dialog.isVisible() and not dialog.isEnabled()
    finally:
        release.set()
    spin(application,lambda:not dialog.isVisible())
    assert dialog.result()==QDialog.DialogCode.Rejected and dialog._worker is None
    dialog.deleteLater()
