"""Radial masks share exact source geometry, state history and formal output."""
from dataclasses import replace
import copy
import numpy as np
import pytest
from photocoach.gradients import LinearGradient,RadialGradient,RenderGeometry,gradient_weight,gradient_from_dict
from photocoach.composition import Composition,sample_composition
from photocoach.adjustments import render_edit,render_photo
from photocoach.imaging import quantize_srgb
from photocoach.preview import DetailRequest,render_region
from photocoach.state import EditValues,StateStore
from photocoach.project import snapshot_project,upgrade_project,validate_project,write_project,read_project
from photocoach.safe_io import sha256_file


def radial(**fields):
    return replace(RadialGradient('r1','Radial 1',(.1,.2),(.9,.8)),**fields)


@pytest.mark.parametrize('fields',[{'end':(.1,.9)},{'end':(.8,.2)},
    {'feather':-.1},{'feather':1.1},{'feather':True},{'feather':float('nan')}])
def test_radial_rejects_degenerate_or_invalid_fields(fields):
    with pytest.raises(ValueError):radial(**fields)


def test_radial_weight_center_boundary_feather_inversion_and_corner_order():
    geometry=RenderGeometry.full((1001,501),Composition(),(1001,501))
    g=radial();weight=gradient_weight(g,geometry)
    assert weight[250,500]==1 and weight[0,0]==0
    assert np.allclose(weight,weight[::-1,::-1])
    assert np.array_equal(weight,gradient_weight(replace(g,start=g.end,end=g.start),geometry))
    assert np.allclose(weight+gradient_weight(replace(g,inverted=True),geometry),1)
    hard=gradient_weight(replace(g,feather=0),geometry)
    assert set(np.unique(hard))=={0.,1.}
    soft=gradient_weight(replace(g,feather=1),geometry)
    assert np.all(soft<=weight+1e-6) and np.max(abs(np.diff(soft[250])))<.01
    assert np.all((weight>=0)&(weight<=1))


def test_mixed_collection_roundtrip_neutral_identity_and_cap():
    linear=LinearGradient('l1','Gradient 1',(0.,0.),(1.,1.))
    values=EditValues(contrast=.13,shadows=.3,local_gradients=(linear,radial()))
    assert EditValues.from_dict(values.to_dict())==values
    assert gradient_from_dict(radial().to_dict())==radial()
    pixels=np.random.default_rng(4).random((139,183,3),dtype=np.float32)
    assert np.array_equal(render_edit(pixels,values),render_edit(pixels,replace(values,local_gradients=())))
    disabled=replace(values,local_gradients=(radial(exposure_ev=3,enabled=False),))
    assert np.array_equal(render_edit(pixels,disabled),render_edit(pixels,replace(values,local_gradients=())))
    with pytest.raises(ValueError):EditValues(local_gradients=tuple(radial(id=f'r{i}',name=f'Radial {i}') for i in range(5)))


@pytest.mark.parametrize('angle',[0,3.25,-8])
@pytest.mark.parametrize('feather',[0,.5,1])
def test_mixed_masks_rotated_native_roi_matches_full_8_and_16_bit(angle,feather):
    pixels=np.random.default_rng(6).random((271,419,3),dtype=np.float32)
    original=pixels.copy();pixels.setflags(write=False)
    masks=(radial(feather=feather,exposure_ev=.8,shadows=.4),
           LinearGradient('l1','Gradient 1',(.1,.2),(.8,.9),highlights=-.3),
           radial(id='r2',name='Radial 2',inverted=True,exposure_ev=-.4))
    values=EditValues(composition=Composition((.05,.08,.97,.96),angle),local_gradients=masks)
    full=render_photo(pixels,values);size=(full.shape[1],full.shape[0]);rect=(9,13,size[0]-7,size[1]-8)
    request=DetailRequest(0,size,rect,1)
    actual=render_region(pixels,values,request,workers=4)
    assert np.array_equal(actual,quantize_srgb(full)[rect[1]:rect[3],rect[0]:rect[2]])
    geometry=RenderGeometry((419,271),values.composition,rect,(rect[2]-rect[0],rect[3]-rect[1]))
    roi=render_edit(sample_composition(pixels,values.composition,rect=rect),values,geometry=geometry)
    assert np.array_equal(quantize_srgb(roi,bits=16),quantize_srgb(full,bits=16)[rect[1]:rect[3],rect[0]:rect[2]])
    assert np.array_equal(original,pixels) and np.isfinite(full).all()


def test_v6_non_neutral_linear_history_migrates_unchanged(tmp_path):
    source=tmp_path/'original.ARW';source.write_bytes(b'read only fixture')
    store=StateStore();store.activate(source,sha256_file(source))
    g=LinearGradient('l','Existing',(.2,.3),(.8,.7),exposure_ev=.4)
    store.set_parameter('local_gradients',(g,));store.set_exposure(.2);store.undo()
    data=snapshot_project(store);data['schema_version']=6;data['pipeline_version']='phase1-sdr-srgb-v6'
    old=copy.deepcopy(data);up=upgrade_project(data)
    assert data==old and up['schema_version']==7
    undo,redo=validate_project(up)
    assert undo==store.undo_stack and redo==store.redo_stack
    bad=copy.deepcopy(data);bad['state']['parameters']['local_gradients']=[radial().to_dict()]
    with pytest.raises(ValueError):upgrade_project(bad)


def test_radial_grouped_history_restore_and_reset(tmp_path):
    source=tmp_path/'original.ARW';source.write_bytes(b'read only fixture')
    store=StateStore();store.activate(source,sha256_file(source))
    for i in range(10):
        store.begin_drag('local_gradients')
        for feather in (.2,.3,.4):store.set_parameter('local_gradients',(radial(feather=feather,exposure_ev=.1*i),))
        store.end_drag()
    final=store.current.values
    assert len(store.undo_stack)==10
    store.undo();store.redo();assert store.current.values==final
    target=tmp_path/'radial.photocoach';write_project(target,snapshot_project(store))
    restored=read_project(target)
    assert restored.current.values==final and restored.undo_stack==store.undo_stack
    store.reset();assert not store.current.local_gradients
    store.undo();assert store.current.values==final
