"""Local corrections share original-image coordinates across all image paths."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import threading
from concurrent.futures import CancelledError
import numpy as np
import pytest
from photocoach.gradients import LinearGradient, RenderGeometry, gradient_weight, source_point, tone_fields, OverlaySpec, overlay_pixels
from photocoach.adjustments import render_edit,render_photo
from photocoach.composition import Composition,output_size
from photocoach.imaging import quantize_srgb
from photocoach.preview import DetailRequest,render_region
from photocoach.state import EditValues,StateStore
from photocoach.project import snapshot_project,upgrade_project,validate_project,write_project,read_project
from photocoach.safe_io import sha256_file
from photocoach.coach import build_request


def gradient(**changes):
    return replace(LinearGradient('g1','Gradient 1',(.2,.2),(.8,.8)),**changes)


@pytest.mark.parametrize('changes',[{'start':(float('nan'),.2)},{'end':(.2,.2)},
    {'exposure_ev':4},{'highlights':True},{'enabled':1},{'name':''},{'id':'bad:id'}])
def test_invalid_gradient(changes):
    with pytest.raises(ValueError): gradient(**changes)


def test_collection_and_state_roundtrip():
    g=gradient(exposure_ev=.5)
    values=EditValues(local_gradients=(g,))
    assert EditValues.from_dict(values.to_dict())==values
    for items in ((g,g),tuple(replace(g,id=f'g{i}') for i in range(5)),[g]):
        with pytest.raises(ValueError): EditValues(local_gradients=items)
    with pytest.raises(ValueError): EditValues(local_gradients=(g,replace(g,id='g2',name='GRADIENT 1')))


def test_weight_continuity_inversion_and_source_transform():
    geometry=RenderGeometry.full((1000,500),Composition(),(1000,500))
    g=gradient(start=(.2,.5),end=(.8,.5))
    w=gradient_weight(g,geometry)
    assert w[0,0]==1 and w[-1,-1]==0
    assert np.all(np.diff(w[250])<=0) and np.max(np.abs(np.diff(w[250])))<.003
    assert np.allclose(w+gradient_weight(replace(g,inverted=True),geometry),1)
    composition=Composition((.1,.2,.9,.95),7)
    for p in ((0,0),(100,200),(400,300)):
        assert np.allclose(source_point(source_point(p,(1000,500),composition),(1000,500),composition,inverse=True),p)


@pytest.mark.parametrize('mode',['empty','neutral','disabled'])
def test_neutral_bypass_exact_pixels(mode):
    linear=np.random.default_rng(8).random((131,197,3),dtype=np.float32)
    linear.setflags(write=False); before=linear.copy()
    items=() if mode=='empty' else (gradient(exposure_ev=2,enabled=False) if mode=='disabled' else gradient(),)
    values=EditValues(contrast=.13,shadows=.2)
    a=render_edit(linear,values); b=render_edit(linear,replace(values,local_gradients=items))
    assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
    assert np.array_equal(linear,before)


@pytest.mark.parametrize('angle',[0,3.25,-8])
def test_native_roi_and_export_exact_for_overlapping_gradients(angle):
    linear=np.random.default_rng(6).random((271,419,3),dtype=np.float32)
    original=linear.copy();linear.setflags(write=False)
    gradients=tuple(gradient(id=f'g{i}',name=f'Gradient {i}',exposure_ev=(-1)**i*.6,
        shadows=.4,highlights=-.3,inverted=bool(i%2)) for i in range(4))
    values=EditValues(exposure_ev=.2,contrast=.1,composition=Composition((.05,.08,.97,.96),angle),local_gradients=gradients)
    full=render_photo(linear,values); size=(full.shape[1],full.shape[0]); rect=(9,13,size[0]-7,size[1]-8)
    request=DetailRequest(0,size,rect,1)
    a=render_region(linear,values,request,workers=1);b=render_region(linear,values,request,workers=4)
    assert np.array_equal(a,b)
    assert np.array_equal(a,quantize_srgb(full)[rect[1]:rect[3],rect[0]:rect[2]])
    assert np.isfinite(full).all() and np.array_equal(linear,original)
    geometry=RenderGeometry((419,271),values.composition,rect,(rect[2]-rect[0],rect[3]-rect[1]))
    from photocoach.composition import sample_composition
    roi=render_edit(sample_composition(linear,values.composition,rect=rect),values,geometry=geometry)
    assert np.array_equal(quantize_srgb(roi,bits=16),quantize_srgb(full,bits=16)[rect[1]:rect[3],rect[0]:rect[2]])


def test_extremes_overlap_clamp_geometry_required_and_cancellation():
    items=tuple(gradient(id=f'g{i}',name=f'Gradient {i}',exposure_ev=3,highlights=1,shadows=-1) for i in range(4))
    values=EditValues(local_gradients=items,highlights=1,shadows=-1)
    linear=np.ones((133,103,3),np.float32)*.2
    geometry=RenderGeometry.full((103,133),Composition(),(103,133))
    ev,hl,sh=tone_fields(items,geometry,1,-1)
    assert ev.max()<=12 and hl.max()==1 and sh.min()==-1
    with pytest.raises(ValueError,match='geometry'): render_edit(linear,values)
    out=render_edit(linear,values,geometry=geometry)
    assert np.isfinite(out).all()
    cancel=threading.Event();cancel.set()
    with pytest.raises(CancelledError): render_edit(linear,values,geometry=geometry,cancel=cancel)


def test_history_ten_edits_reset_restore_and_ai_snapshot(tmp_path):
    path=tmp_path/'test.ARW';path.write_bytes(b'read only test original')
    store=StateStore();before=store.activate(path,sha256_file(path))
    g=gradient()
    for i in range(10):
        store.begin_drag('local_gradients')
        for k in range(3): store.set_parameter('local_gradients',(replace(g,exposure_ev=(i+1)*.1+k*.01),))
        store.end_drag()
    final=store.current.values
    assert len(store.undo_stack)==10
    store.undo();store.redo();assert store.current.values==final
    store.begin_drag('local_gradients');store.set_parameter('local_gradients',());store.end_drag(cancel=True)
    assert store.current.values==final and len(store.undo_stack)==10
    target=tmp_path/'session.photocoach';write_project(target,snapshot_project(store))
    restored=read_project(target)
    assert restored.current.values==final and restored.undo_stack==store.undo_stack
    pixels=np.zeros((8,9,3),np.uint8)
    request=build_request(store.current,before,store.events,pixels,pixels)
    assert request.parameters['local_gradients']==final.to_dict()['local_gradients']
    store.reset();assert not store.current.local_gradients
    store.undo();assert store.current.values==final


def test_v5_migration_all_history_and_strict_current_schema(tmp_path):
    store=StateStore();path=tmp_path/'test.ARW';path.write_bytes(b'original')
    store.activate(path,sha256_file(path));store.set_exposure(.2);store.reset();store.undo()
    data=snapshot_project(store)
    def strip(value):
        if isinstance(value,dict):
            if 'white_balance' in value: value.pop('local_gradients')
            for item in value.values():strip(item)
        elif isinstance(value,list):
            for item in value:strip(item)
    strip(data);data['schema_version']=5;data['pipeline_version']='phase1-sdr-srgb-v5'
    old=copy.deepcopy(data);up=upgrade_project(data)
    assert old==data and up['schema_version']==7
    undo,redo=validate_project(up)
    assert undo==store.undo_stack and redo==store.redo_stack
    broken=copy.deepcopy(up);broken['state']['parameters'].pop('local_gradients')
    with pytest.raises(ValueError):validate_project(broken)


def test_clipping_is_display_only_and_mask_overlay_uses_weights():
    pixels=np.array([[[255,0,1],[0,0,0],[1,2,3]]],np.uint8)
    geometry=RenderGeometry.full((3,1),Composition(),(3,1))
    original=pixels.copy()
    overlay=overlay_pixels(pixels,geometry,OverlaySpec('clipping'))
    assert overlay[0,:,3].tolist()==[170,170,0]
    assert np.array_equal(pixels,original)
    g=gradient(start=(0,.5),end=(1,.5))
    overlay=overlay_pixels(pixels,geometry,OverlaySpec('mask',g))
    assert np.array_equal(overlay[...,3],np.rint(gradient_weight(g,geometry)*90).astype(np.uint8))
