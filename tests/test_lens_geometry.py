"""Production geometry compared to an independent scalar lens oracle."""
from dataclasses import replace
from concurrent.futures import CancelledError
import threading
import numpy as np
import pytest
from photocoach.lens import LensCorrections, map_to_source, map_from_source, apply_vignetting
from photocoach.composition import Composition, output_size, sample_composition
from photocoach.gradients import RenderGeometry, source_point, LinearGradient, RadialGradient, gradient_weight
from photocoach.adjustments import render_photo, render_edit
from photocoach.preview import DetailRequest, render_region
from photocoach.imaging import quantize_srgb
from photocoach.state import EditValues
from photocoach.detail_cache import DetailGeometryCache
from photocoach.white_balance import sensor_position
from reference_lens_contract import canvas_to_source, source_to_canvas, vignette_gain, sample_reference


@pytest.mark.parametrize('size',[(101,73),(73,101),(1,5)])
@pytest.mark.parametrize('distortion',[-100,-1,0,1,100])
def test_mapping_matches_oracle_and_visible_inverse(size,distortion):
    w,h=size
    x,y=np.meshgrid(np.linspace(0,w,19),np.linspace(0,h,17))
    sx,sy=map_to_source(x,y,size,distortion)
    assert (sx>=-1e-10).all() and (sx<=w+1e-10).all()
    assert (sy>=-1e-10).all() and (sy<=h+1e-10).all()
    for row,col in ((0,0),(8,9),(16,18),(2,14)):
        p=x[row,col],y[row,col]
        expected=canvas_to_source(p,size,distortion)
        assert (sx[row,col],sy[row,col])==pytest.approx(expected,abs=1e-12)
        assert map_from_source(expected,size,distortion)==pytest.approx(p,abs=1e-10)
    assert map_from_source((w,h),size,distortion)==source_to_canvas((w,h),size,distortion)


@pytest.mark.parametrize('distortion',[-100,0,100])
@pytest.mark.parametrize('angle',[-15,0,15])
@pytest.mark.parametrize('reduced',[False,True])
def test_single_sampling_matches_scalar_ramp_and_original_radius(distortion,angle,reduced):
    source_size=(103,157)
    h,w=(53,35) if reduced else (157,103)
    y,x=np.mgrid[:h,:w]
    linear=np.stack((x+1,y+2,x+y+3),-1).astype(np.float32)/256
    linear.setflags(write=False);before=linear.copy()
    comp=Composition((.1,.05,.91,.97),angle)
    target=(29,41)
    geo=RenderGeometry.full(source_size,comp,target,distortion)
    actual=sample_composition(linear,comp,source_size=source_size,target_size=target,distortion=distortion)
    gained=apply_vignetting(actual,geo,2)
    gx,gy=geo.source_grid()
    ow,oh=output_size(source_size,comp)
    for row,col in ((0,0),(12,17),(40,28)):
        point=((col+.5)*ow/target[0],(row+.5)*oh/target[1])
        affine=source_point(point,source_size,comp)
        expected=canvas_to_source(affine,source_size,distortion)
        assert (gx[row,col],gy[row,col])==pytest.approx(expected,abs=1e-10)
        assert source_point(expected,source_size,comp,inverse=True,distortion=distortion)==pytest.approx(point,abs=1e-9)
        np.testing.assert_allclose(actual[row,col],sample_reference(linear,expected,source_size=source_size),rtol=3e-7)
        np.testing.assert_allclose(gained[row,col],sample_reference(linear,expected,2,source_size=source_size),rtol=3e-7)
    assert np.array_equal(linear,before)
    assert np.isfinite(gained).all()


@pytest.mark.parametrize('distortion',[-100,0,100])
@pytest.mark.parametrize('angle',[0,7.25])
@pytest.mark.parametrize('ev',[-2,0,2])
def test_native_export_and_strip_seams_exact_with_mixed_masks(distortion,angle,ev):
    linear=np.random.default_rng(93).random((277,411,3),dtype=np.float32)*.6
    original=linear.copy();linear.setflags(write=False)
    comp=Composition((.07,.12,.97,.98),angle)
    masks=(LinearGradient('l','Slope',(.2,.2),(.8,.7),exposure_ev=.3),
           RadialGradient('r','Rock',(.3,.2),(.6,.8),exposure_ev=-.2))
    values=EditValues(composition=comp,lens_corrections=LensCorrections(distortion,ev),
                      local_gradients=masks,highlights=-.2,saturation=1.1)
    full=render_photo(linear,values);size=(full.shape[1],full.shape[0])
    rect=(7,11,size[0]-5,size[1]-3)
    request=DetailRequest(0,size,rect)
    preview=render_region(linear,values,request,workers=4)
    np.testing.assert_array_equal(preview,quantize_srgb(full)[11:rect[3],7:rect[2]])
    geo=RenderGeometry((411,277),comp,rect,(rect[2]-7,rect[3]-11),distortion)
    patch=sample_composition(linear,comp,rect=rect,distortion=distortion)
    roi=render_edit(patch,values,geometry=geo)
    np.testing.assert_array_equal(quantize_srgb(roi,bits=16),quantize_srgb(full,bits=16)[11:rect[3],7:rect[2]])
    np.testing.assert_array_equal(linear,original)
    # Source-coordinate mask weights agree with the scalar source mapping.
    scalar_geo=RenderGeometry.full((411,277),Composition(),(1,1))
    for mask in masks:
        weight=gradient_weight(mask,geo)
        grid=geo.source_grid()
        expected=gradient_weight(mask,scalar_geo,(grid[0][12:13,8:9],grid[1][12:13,8:9]))
        assert weight[12,8]==expected[0,0]


def test_neutral_keeps_exact_old_slice_and_pixels():
    image=np.random.default_rng(5).random((139,231,3),dtype=np.float32)
    comp=Composition((.1,.1,.9,.9))
    out=sample_composition(image,comp,distortion=0)
    assert np.shares_memory(out,image)
    np.testing.assert_array_equal(render_photo(image,EditValues()),render_edit(image,EditValues()))


def test_vignette_anchor_symmetry_linear_hdr_and_neutral_identity():
    image=np.full((3,3,3),2.,np.float32)
    geo=RenderGeometry.full((6,8),Composition(),(3,3))
    assert apply_vignetting(image,geo,0) is image
    out=apply_vignetting(image,geo,2)
    assert out[1,1,0]==2 and out[0,0,0]>2
    np.testing.assert_array_equal(out,out[::-1,::-1])
    gx,gy=geo.source_grid()
    assert out[0,0,0]==pytest.approx(2*vignette_gain((gx[0,0],gy[0,0]),(6,8),2),rel=1e-6)


def test_geometry_cache_reuses_only_matching_distortion():
    image=np.random.default_rng(4).random((160,220,3),dtype=np.float32)
    cache=DetailGeometryCache();comp=Composition();rect=(10,10,180,140)
    def get(d):return cache.region(image,'photo',comp,rect,max_bytes=8*1024**2,reserve=lambda _:None,distortion=d)
    assert get(0) is None
    first=get(45);again=get(45)
    assert cache.builds==1 and cache.hits==1
    np.testing.assert_array_equal(first,again)
    np.testing.assert_array_equal(first,sample_composition(image,comp,rect=rect,distortion=45))
    other=get(-45)
    assert cache.builds==2 and not np.array_equal(first,other)
    assert cache.retained_bytes<=8*1024**2


@pytest.mark.parametrize('flip',range(8))
def test_wb_picker_maps_distortion_before_sensor_orientation(flip):
    size=(103,157);oriented=size[::-1] if flip&4 else size
    comp=Composition((.1,.1,.9,.9),5)
    p=(13,21)
    sx,sy=source_point((p[0]+.5,p[1]+.5),oriented,comp,distortion=-100)
    sx-=.5;sy-=.5
    col,row=(sy,sx) if flip&4 else (sx,sy)
    if flip&2:row=size[1]-1-row
    if flip&1:col=size[0]-1-col
    assert sensor_position(size,flip,comp,p,distortion=-100)==(round(col),round(row))


def test_lens_cancel_bounded_before_processing():
    cancel=threading.Event();cancel.set()
    image=np.zeros((139,231,3),np.float32)
    values=EditValues(lens_corrections=LensCorrections(100,2))
    with pytest.raises(CancelledError):render_photo(image,values,cancel=cancel)
    with pytest.raises(CancelledError):render_region(image,values,DetailRequest(0,(231,139),(0,0,231,139)),cancel=cancel)
