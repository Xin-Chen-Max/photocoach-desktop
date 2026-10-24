"""Shared floating-point editing pipeline for previews and full-resolution export."""
import numpy as np
from concurrent.futures import CancelledError

from .imaging import render_srgb
from .state import EditValues, IDENTITY_CURVE
from .composition import output_size, sample_composition, Composition
from .color_mixer import apply_color_mixer
from .render_timing import timed
from .scratch import Scratch
from .gradients import RenderGeometry, tone_fields


def render_photo(linear, values, *, source_size=None, max_edge=None, cancel=None, timer=None,
                 scratch: Scratch | None = None):
    """Shared composition then tonal pipeline; use strips for full exports."""
    if scratch is None:
        scratch = Scratch()
    source_size = source_size or (linear.shape[1], linear.shape[0])
    size = output_size(source_size, values.composition)
    if max_edge is not None:
        # Do not disguise a small cropped preview as native detail by upscaling
        # reduced source data. Its real pixel density drives detail requests.
        factor = min(1., max_edge/max(size), linear.shape[1]/source_size[0],
                     linear.shape[0]/source_size[1])
        target = (max(1, round(size[0]*factor)), max(1, round(size[1]*factor)))
        with timed(timer, "geometry"):
            sampled = sample_composition(linear, values.composition, source_size=source_size,
                                         target_size=target, scratch=scratch)
        return render_edit(sampled, values, cancel=cancel, timer=timer, scratch=scratch,
                           geometry=RenderGeometry.full(source_size,values.composition,target))
    if values.composition == Composition():
        return render_edit(linear, values, cancel=cancel, timer=timer, scratch=scratch,
                           geometry=RenderGeometry.full(source_size,values.composition,(linear.shape[1],linear.shape[0])))
    result = np.empty((size[1], size[0], 3), np.float32)
    for top in range(0, size[1], 128):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        bottom = min(top+128, size[1])
        with timed(timer, "geometry"):
            sampled = sample_composition(linear, values.composition,
                                         source_size=source_size, rect=(0, top, size[0], bottom),
                                         scratch=scratch)
        result[top:bottom] = render_edit(sampled, values, cancel=cancel, timer=timer, scratch=scratch,
            geometry=RenderGeometry(tuple(source_size),values.composition,(0,top,size[0],bottom),(size[0],bottom-top)))
    return result


def render_edit(linear_rgb: np.ndarray, values: EditValues, *, cancel=None, timer=None,
                scratch: Scratch | None = None, geometry: RenderGeometry | None = None) -> np.ndarray:
    """Exposure -> linear Highlights/Shadows -> SDR clip/OETF -> Whites/Blacks -> contrast -> saturation -> Vibrance -> HSL -> RGB curve.

    Contrast pivots encoded sRGB at 0.5 with slope 2**contrast. Saturation
    interpolates encoded RGB around weighted gray (0.2126,0.7152,0.0722).
    This gray is a display-space luma proxy, not physical luminance.
    Clip after contrast and saturation, then apply a piecewise-linear curve.
    """
    if not isinstance(values, EditValues):
        raise ValueError("Rendering requires validated editing values.")
    if scratch is None:
        scratch = Scratch()
    with timed(timer, "tones"):
        if any(g.active for g in values.local_gradients):
            if geometry is None or geometry.target_size!=(linear_rgb.shape[1],linear_rgb.shape[0]):
                raise ValueError("Active gradients require an explicit matching source sampling geometry.")
            result=np.empty(linear_rgb.shape,np.float32)
            for top in range(0,linear_rgb.shape[0],128):
                if cancel is not None and cancel.is_set():
                    raise CancelledError()
                bottom=min(top+128,linear_rgb.shape[0])
                fields=tone_fields(values.local_gradients,geometry.rows(top,bottom),values.highlights,values.shadows)
                result[top:bottom]=render_srgb(linear_rgb[top:bottom],values.exposure_ev,
                    highlights=values.highlights,shadows=values.shadows,cancel=cancel,scratch=scratch,local_fields=fields)
        else:
            result = render_srgb(linear_rgb, values.exposure_ev, highlights=values.highlights,
                                 shadows=values.shadows, cancel=cancel, scratch=scratch)
    x, y = np.asarray(values.rgb_curve, dtype=np.float32).T
    for start in range(0, result.shape[0], 128):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        block = result[start:start + 128]
        with timed(timer, "tones"):
            _tone_block(block, values, scratch=scratch)
        with timed(timer, "hsl"):
            apply_color_mixer(block, values.color_mixer, scratch=scratch)
        if values.rgb_curve != IDENTITY_CURVE:
            with timed(timer, "curve"):
                block[:] = np.interp(block, x, y)
    return result


def _tone_block(block, values, *, scratch: Scratch | None = None):
    """Whites/Blacks -> contrast -> saturation -> Vibrance, in place on one row block."""
    if not (values.whites or values.blacks or values.contrast or values.saturation != 1 or values.vibrance):
        return
    if scratch is None:
        scratch = Scratch()
    if values.whites or values.blacks:
        # Smooth monotone shared-channel endpoint shaping in encoded sRGB.
        shift = scratch.get("comp_s0", block.shape, np.float32)
        tmp = scratch.get("comp_s1", block.shape, np.float32)
        shift.fill(0)
        if values.whites:
            np.power(block, 4, out=tmp)
            tmp *= np.float32(.25 * values.whites)
            shift += tmp
        if values.blacks:
            np.subtract(np.float32(1), block, out=tmp)
            np.power(tmp, 4, out=tmp)
            tmp *= np.float32(.25 * values.blacks)
            shift += tmp
        block += shift
        np.clip(block, 0, 1, out=block)
    if values.contrast:
        block -= np.float32(.5)
        block *= np.float32(2 ** values.contrast)
        block += np.float32(.5)
        np.clip(block, 0, 1, out=block)
    shape = block.shape[:2]
    if values.saturation != 1 or values.vibrance:
        red = scratch.get("hsl_r", shape, np.float32)
        green = scratch.get("hsl_g", shape, np.float32)
        blue = scratch.get("hsl_b", shape, np.float32)
        np.copyto(red, block[..., 0])
        np.copyto(green, block[..., 1])
        np.copyto(blue, block[..., 2])
        channels = (red, green, blue)
        gray2 = scratch.get("hsl_b0", shape, np.float32)
        tmp2 = scratch.get("hsl_b1", shape, np.float32)
        if values.saturation != 1:
            sat_factor = np.float32(values.saturation)
            np.multiply(red, np.float32(.2126), out=gray2)
            np.multiply(green, np.float32(.7152), out=tmp2)
            gray2 += tmp2
            np.multiply(blue, np.float32(.0722), out=tmp2)
            gray2 += tmp2
            for channel, plane in enumerate(channels):
                plane -= gray2
                plane *= sat_factor
                plane += gray2
                np.clip(plane, 0, 1, out=plane if values.vibrance else block[..., channel])
        if values.vibrance:
            maximum = scratch.get("hsl_max", shape, np.float32)
            minimum = scratch.get("hsl_min", shape, np.float32)
            sat = scratch.get("hsl_sat", shape, np.float32)
            pos = scratch.get("hsl_col", shape, np.bool_)
            np.maximum(red, green, out=maximum)
            np.maximum(maximum, blue, out=maximum)
            np.minimum(red, green, out=minimum)
            np.minimum(minimum, blue, out=minimum)
            sat.fill(0)
            np.subtract(maximum, minimum, out=tmp2)
            np.greater(maximum, 0, out=pos)
            np.divide(tmp2, maximum, out=sat, where=pos)
            np.subtract(np.float32(1), sat, out=sat)
            np.square(sat, out=sat)
            sat *= np.float32(values.vibrance)
            sat += np.float32(1)
            np.multiply(red, np.float32(.2126), out=gray2)
            np.multiply(green, np.float32(.7152), out=tmp2)
            gray2 += tmp2
            np.multiply(blue, np.float32(.0722), out=tmp2)
            gray2 += tmp2
            for channel, plane in enumerate(channels):
                plane -= gray2
                plane *= sat
                plane += gray2
                np.clip(plane, 0, 1, out=block[..., channel])
