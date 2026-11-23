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
            value.pop('compare_snapshot_id', None)
            value.pop('snapshots', None)
            if 'white_balance' in value:
                value.pop("base_rendering")
                value.pop("color_grading")
                value.pop("spots")
                value.pop('local_gradients')
                value.pop('lens_corrections')
            for item in value.values():strip(item)
        elif isinstance(value,list):
            for item in value:strip(item)
    strip(data);data['schema_version']=5;data['pipeline_version']='phase1-sdr-srgb-v5'
    old=copy.deepcopy(data);up=upgrade_project(data)
    assert old==data and up['schema_version']==14
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


def test_shadow_aware_spotlight_preserves_16bit_roi_export_equivalence():
    from photocoach.gradients import RadialGradient
    linear = np.random.default_rng(19).uniform(0.01, 0.25, size=(272, 408, 3)).astype(np.float32)
    # Dark center subject patch inside midtone sand
    linear[90:180, 120:280] = 0.022
    linear.setflags(write=False)
    spotlight = RadialGradient(
        "spot-1", "Overhead Key Light", (0.20, 0.18), (0.80, 0.72),
        exposure_ev=0.44, highlights=0.20, shadows=0.38, warmth=0.36, feather=0.80,
    )
    values = EditValues(local_gradients=(spotlight,))
    full = render_photo(linear, values)
    size = (full.shape[1], full.shape[0])
    rect = (15, 21, size[0] - 11, size[1] - 17)
    roi = render_region(linear, values, DetailRequest(0, size, rect, 1), workers=2)
    assert np.array_equal(roi, quantize_srgb(full)[rect[1]:rect[3], rect[0]:rect[2]])

    # Shadow-aware spotlight lifts dark foreground subjects cleanly in sRGB (>25% gain on dark fur)
    base_full = quantize_srgb(render_photo(linear, EditValues()))
    spot_full = quantize_srgb(full)
    dark_before = float(base_full[135, 200].mean())
    dark_after = float(spot_full[135, 200].mean())
    assert dark_after > dark_before * 1.25


def test_luma_range_and_edge_aware_isolation_and_roi_parity():
    from photocoach.detail import DetailControls
    from photocoach.gradients import BrushMask, BrushStroke, RadialGradient, gradient_from_dict
    from photocoach.lens import LensCorrections
    from photocoach.spots import SpotRemoval

    for bad in ({"luma_min": 0.6, "luma_max": 0.4}, {"luma_min": -0.1}, {"luma_feather": 1.2}, {"edge_aware": 1}):
        with pytest.raises(ValueError):
            gradient(**bad)

    # Neutral defaults omit v14 keys; non-default range/edge fields round-trip cleanly across all 3 mask kinds.
    plain_d = gradient().to_dict()
    assert not ({"luma_min", "luma_max", "luma_feather", "edge_aware"} & set(plain_d))
    for item in (
        LinearGradient("lg", "L", (0.1, 0.1), (0.9, 0.9), luma_min=0.05, luma_max=0.4, luma_feather=0.1, edge_aware=True),
        RadialGradient("rg", "R", (0.2, 0.2), (0.8, 0.8), luma_max=0.35, edge_aware=True),
        BrushMask("bg", "B", strokes=(BrushStroke(((0.3, 0.3), (0.5, 0.5))),), luma_min=0.1, edge_aware=True),
    ):
        assert gradient_from_dict(item.to_dict()) == item

    # Dusk silhouette: dark subject [0.012] backed by brighter smooth twilight sky [0.08, 0.10, 0.14].
    rng = np.random.default_rng(11)
    lin = rng.uniform(0.02, 0.25, size=(304, 440, 3)).astype(np.float32)
    lin[:190, :] = np.array([0.08, 0.10, 0.14], dtype=np.float32)
    lin[70:180, 170:270] = np.array([0.012, 0.011, 0.010], dtype=np.float32)
    lin.setflags(write=False)

    base = quantize_srgb(render_photo(lin, EditValues()))
    plain_radial = RadialGradient("p1", "Plain", (0.30, 0.15), (0.70, 0.65), exposure_ev=0.5, warmth=0.2, feather=0.85)
    plain_out = quantize_srgb(render_photo(lin, EditValues(local_gradients=(plain_radial,))))
    sky_ring = np.zeros((304, 440), dtype=bool)
    sky_ring[60:150, 145:166] = True
    subj = np.zeros((304, 440), dtype=bool)
    subj[90:150, 190:250] = True
    # Old surf_gate inversion is gone: unmasked radial no longer boosts sky more than dark subject.
    assert (plain_out[subj].mean() - base[subj].mean()) >= (plain_out[sky_ring].mean() - base[sky_ring].mean())

    isolated = RadialGradient(
        "iso", "Isolated", (0.30, 0.15), (0.70, 0.65),
        exposure_ev=0.45, shadows=0.30, warmth=0.14, feather=0.85,
        luma_min=0.0, luma_max=0.22, luma_feather=0.08, edge_aware=True,
    )
    iso_vals = EditValues(local_gradients=(isolated,))
    iso_out = quantize_srgb(render_photo(lin, iso_vals))
    assert abs(float(iso_out[sky_ring].mean() - base[sky_ring].mean())) < 2.0
    assert float(iso_out[subj].mean()) > float(base[subj].mean()) * 1.20

    # Overlay reflects luminance isolation on sky vs dark subject.
    geom = RenderGeometry.full((440, 304), Composition(), (440, 304))
    ov = overlay_pixels(base, geom, OverlaySpec("mask", isolated))
    assert int(ov[sky_ring, 3].mean()) <= 5 and int(ov[subj, 3].mean()) >= 40

    # 8-bit and 16-bit 0-LSB ROI/strip parity with edge_aware + luma_range + composition + distortion + detail + spots.
    complex_vals = EditValues(
        local_gradients=(isolated,),
        composition=Composition((0.04, 0.04, 0.94, 0.92), -2.0),
        lens_corrections=LensCorrections(distortion=10.0, vignetting_ev=-0.25),
        detail=DetailControls(sharpening=35.0, masking=10.0, luminance_nr=15.0, chroma_nr=20.0),
        spots=(SpotRemoval("s1", (0.25, 0.75), (0.35, 0.75)),),
    )
    full_f = render_photo(lin, complex_vals)
    out_sz = (full_f.shape[1], full_f.shape[0])
    rect = (19, 27, out_sz[0] - 17, out_sz[1] - 23)
    roi_u8 = render_region(lin, complex_vals, DetailRequest(0, out_sz, rect, 1), workers=2)
    assert np.array_equal(roi_u8, quantize_srgb(full_f)[rect[1]:rect[3], rect[0]:rect[2]])


def test_schema13_to_14_migration_and_legacy_mask_rejection(tmp_path):
    from photocoach.spots import SpotRemoval

    src = tmp_path / "photo.ARW"
    src.write_bytes(b"readonly")
    digest = sha256_file(src)
    store = StateStore()
    store.activate(src, digest, base_rendering="natural_v1")
    store.set_parameter("spots", (SpotRemoval("s1", (0.3, 0.4), (0.5, 0.4)),))
    store.set_parameter("local_gradients", (gradient(),))
    doc = snapshot_project(store)
    assert doc["schema_version"] == 14

    v13_doc = copy.deepcopy(doc)
    v13_doc["schema_version"] = 13
    v13_doc["pipeline_version"] = "phase1-sdr-srgb-v13"
    v13_doc["source"]["byte_length"] = src.stat().st_size
    upgraded = upgrade_project(v13_doc)
    assert upgraded["schema_version"] == 14
    validate_project(upgraded)

    v14_mask = gradient(luma_max=0.35, edge_aware=True).to_dict()
    bad_v13 = copy.deepcopy(v13_doc)
    bad_v13["state"]["parameters"]["local_gradients"] = [v14_mask]
    with pytest.raises(ValueError):
        upgrade_project(bad_v13)
