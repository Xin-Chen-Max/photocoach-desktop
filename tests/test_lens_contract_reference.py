"""Offline preparation only: analytic fixtures for the proposed lens contract.

These do not exercise a production lens control, v8 project, export or desktop.
Use them as independent expected values when implementing after P0 acceptance.
"""
import math

import numpy as np
import pytest

from photocoach.composition import Composition, output_size
from photocoach.gradients import RenderGeometry, source_point
from reference_lens_contract import (canvas_to_source, source_to_canvas,
                                     vignette_gain, sample_reference)


@pytest.mark.parametrize('distortion', [-100, -50, 0, 50, 100])
@pytest.mark.parametrize('size', [(6000, 4000), (4000, 6000), (101, 73), (1, 5)])
def test_reference_preserves_center_and_has_no_empty_edges(size, distortion):
    w, h = size
    assert canvas_to_source((w/2, h/2), size, distortion) == (w/2, h/2)
    for x in np.linspace(0, w, 13):
        for y in np.linspace(0, h, 13):
            sx, sy = canvas_to_source((x, y), size, distortion)
            assert math.isfinite(sx) and math.isfinite(sy)
            assert -1e-10 <= sx <= w+1e-10 and -1e-10 <= sy <= h+1e-10


def test_reference_has_hand_calculated_coordinate_anchors():
    # A 6x8 rectangle has half-diagonal 5. Corner radius is exactly one.
    assert canvas_to_source((6, 8), (6, 8), 0) == (6, 8)
    assert canvas_to_source((6, 8), (6, 8), -100) == pytest.approx((5.4, 7.2))
    # Positive k=1/5, scale=5/6: factor=5/6*(1+5/36)=205/216.
    assert canvas_to_source((6, 8), (6, 8), 100) == pytest.approx((3+205/72, 4+205/54))
    assert canvas_to_source((4.5, 6), (6, 8), -100) == pytest.approx((4.425, 5.9))


@pytest.mark.parametrize('distortion', [-100, -1, 0, 1, 100])
@pytest.mark.parametrize('size', [(6000,4000), (4000,6000), (101,73)])
def test_reference_handles_roundtrip_inside_and_hide_outside(size, distortion):
    w, h = size
    for x, y in ((.5,.5), (.01,.1), (.9,.95), (0,1), (1,1)):
        point = (w*x, h*y)
        source = canvas_to_source(point, size, distortion)
        assert source_to_canvas(source, size, distortion) == pytest.approx(point, abs=2e-9)
    if distortion:
        # Correction shrinks the visible source footprint; don't pin hidden handles to an edge.
        assert source_to_canvas((w,h), size, distortion) is None


@pytest.mark.parametrize('distortion', [-100, 0, 100])
@pytest.mark.parametrize('angle', [-15, 0, 15])
def test_reference_composes_crop_rotation_strips_and_original_mask_coordinates(distortion, angle):
    size=(101,73)
    composition=Composition((.1,.05,.9,.95),angle)
    target=output_size(size,composition)
    geometry=RenderGeometry.full(size,composition,target)
    gx, gy=np.broadcast_arrays(*geometry.source_grid())
    strip=geometry.rows(3,8)
    sx, sy=np.broadcast_arrays(*strip.source_grid())
    for y, x in ((0,0),(2,target[0]//2),(4,target[0]-1)):
        canvas=(x+.5,y+3+.5)
        composed=source_point(canvas,size,composition)
        expected=canvas_to_source(composed,size,distortion)
        assert canvas_to_source((gx[y+3,x],gy[y+3,x]),size,distortion) == pytest.approx(expected,abs=2e-12)
        assert canvas_to_source((sx[y,x],sy[y,x]),size,distortion) == pytest.approx(expected,abs=2e-12)
        if distortion == 0:
            assert source_to_canvas(expected,size,0) == composed


@pytest.mark.parametrize('ev', [-2, -.25, 0, .25, 2])
def test_reference_vignette_center_corner_symmetry_and_bounds(ev):
    size=(6,8)
    assert vignette_gain((3,4),size,ev) == 1
    assert vignette_gain((6,8),size,ev) == pytest.approx(2**ev)
    assert vignette_gain((4.5,6),size,ev) == pytest.approx(2**(ev/16))
    for point in ((1,2),(2,7),(.5,.5)):
        gain=vignette_gain(point,size,ev)
        assert gain == pytest.approx(vignette_gain((6-point[0],8-point[1]),size,ev))
        assert .25 <= gain <= 4


def test_reference_samples_linear_ramp_once_without_mutating_source():
    y,x=np.mgrid[:8,:6]
    image=np.stack((x+1,y+2,x+y+3),axis=-1).astype(np.float32)/32
    before=image.copy();image.setflags(write=False)
    # Bilinear interpolation exactly preserves affine fields at an interior location.
    point=(2.25,3.75)
    expected=(2.75/32,5.25/32,8/32)
    assert sample_reference(image,point) == pytest.approx(expected)
    gain=vignette_gain(point,(6,8),2)
    assert sample_reference(image,point,2) == pytest.approx(tuple(v*gain for v in expected))
    np.testing.assert_array_equal(image,before)


@pytest.mark.parametrize('distortion,minimum_error', [(-100,50), (100,38)])
def test_reference_mask_center_cannot_average_warped_endpoints(distortion,minimum_error):
    size=(6000,4000)
    start,end=(1200,1200),(4200,2600)
    original_center=tuple((a+b)/2 for a,b in zip(start,end))
    center=source_to_canvas(original_center,size,distortion)
    a,b=source_to_canvas(start,size,distortion),source_to_canvas(end,size,distortion)
    shortcut=tuple((x+y)/2 for x,y in zip(a,b))
    assert math.dist(center,shortcut)>minimum_error
    assert canvas_to_source(center,size,distortion)==pytest.approx(original_center,abs=2e-9)


@pytest.mark.parametrize('source_size,decoded_size', [((101,73),(51,37)), ((73,101),(37,51))])
@pytest.mark.parametrize('ev', [0,2])
def test_reference_reduced_odd_decode_uses_original_gain_radius(source_size,decoded_size,ev):
    sw,sh=source_size;w,h=decoded_size
    y,x=np.mgrid[:h,:w]
    sx,sy=(x+.5)*sw/w,(y+.5)*sh/h
    image=np.stack((sx/(2*sw),sy/(2*sh),(sx+sy)/(2*(sw+sh))),axis=-1).astype(np.float32)
    image.setflags(write=False)
    point=(sw*.23,sh*.61)
    expected=(point[0]/(2*sw),point[1]/(2*sh),sum(point)/(2*(sw+sh)))
    gain=vignette_gain(point,source_size,ev)
    assert sample_reference(image,point,ev,source_size=source_size)==pytest.approx(tuple(v*gain for v in expected))
