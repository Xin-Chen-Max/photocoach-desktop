"""Phase 1 shared float pipeline, used by preview and full export."""
import numpy as np
from concurrent.futures import CancelledError

from .imaging import render_srgb
from .lens import apply_vignetting
from .detail import DETAIL_HALO_PX, apply_detail
from .state import EditValues, IDENTITY_CURVE
from .composition import output_size, sample_composition, Composition
from .color_mixer import apply_color_mixer
from .color_grading import apply_color_grading
from .render_timing import timed
from .scratch import Scratch
from .gradients import (MASK_EDGE_HALO_PX, RenderGeometry, local_mask_fields,
                        perceptual_luma_guide)
from .spots import apply_spots, prepare_spots


def required_halo(values: EditValues) -> int:
    """Minimum spatial pixel halo needed for 0-LSB strip/ROI invariance."""
    mask_halo = MASK_EDGE_HALO_PX if any(g.active and g.edge_aware for g in values.local_gradients) else 0
    detail_halo = DETAIL_HALO_PX if values.detail.active else 0
    return max(detail_halo, mask_halo)


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
                                         target_size=target, scratch=scratch, distortion=values.lens_corrections.distortion)
        return render_edit(sampled, values, cancel=cancel, timer=timer, scratch=scratch,
                           geometry=RenderGeometry.full(source_size,values.composition,target,values.lens_corrections.distortion),
                           source_linear=linear)
    if values.composition == Composition() and not values.lens_corrections.distortion:
        return render_edit(linear, values, cancel=cancel, timer=timer, scratch=scratch,
                           geometry=RenderGeometry.full(source_size,values.composition,(linear.shape[1],linear.shape[0])),
                           source_linear=linear)
    result = np.empty((size[1], size[0], 3), np.float32)
    halo = required_halo(values)
    for top in range(0, size[1], 128):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        bottom = min(top+128, size[1])
        p_top = max(0, top - halo)
        p_bottom = min(size[1], bottom + halo)
        with timed(timer, "geometry"):
            sampled = sample_composition(linear, values.composition,
                                         source_size=source_size, rect=(0, p_top, size[0], p_bottom),
                                         scratch=scratch, distortion=values.lens_corrections.distortion)
        rendered = render_edit(sampled, values, cancel=cancel, timer=timer, scratch=scratch,
            geometry=RenderGeometry(tuple(source_size),values.composition,(0,p_top,size[0],p_bottom),(size[0],p_bottom-p_top),values.lens_corrections.distortion),
            source_linear=linear)
        result[top:bottom] = rendered[top - p_top : top - p_top + (bottom - top)] if halo else rendered
    return result


def render_edit(linear_rgb: np.ndarray, values: EditValues, *, cancel=None, timer=None,
                scratch: Scratch | None = None, geometry: RenderGeometry | None = None,
                source_linear: np.ndarray | None = None) -> np.ndarray:
    """Spot removal -> Exposure -> linear Highlights/Shadows -> base rendering -> SDR clip/OETF -> Whites/Blacks -> contrast -> saturation -> Vibrance -> Color Grading -> HSL -> RGB curve -> Detail.

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
        active_masks = any(g.active for g in values.local_gradients)
        active_spots = any(s.active for s in values.spots)
        if active_spots or active_masks or values.lens_corrections.vignetting_ev:
            if geometry is None or geometry.target_size!=(linear_rgb.shape[1],linear_rgb.shape[0]):
                raise ValueError("Spatial adjustments require an explicit matching source sampling geometry.")
            if active_spots:
                if source_linear is not None:
                    src_lin = source_linear
                elif (values.composition == Composition() and not values.lens_corrections.distortion
                      and geometry.rect == (0, 0, *geometry.source_size)
                      and geometry.target_size == (linear_rgb.shape[1], linear_rgb.shape[0])):
                    src_lin = linear_rgb
                else:
                    raise ValueError("Active spot removals on cropped or transformed regions require source_linear.")
                prepared_spots = prepare_spots(values.spots, src_lin, geometry.source_size)
            else:
                src_lin = None
                prepared_spots = ()
            needs_edge_halo = active_masks and any(g.active and g.edge_aware for g in values.local_gradients)
            needs_guide = active_masks and any(g.active and g.uses_image_guide for g in values.local_gradients)
            result=np.empty(linear_rgb.shape,np.float32)
            for top in range(0,linear_rgb.shape[0],128):
                if cancel is not None and cancel.is_set():
                    raise CancelledError()
                bottom=min(top+128,linear_rgb.shape[0])
                if needs_edge_halo:
                    p_top = max(0, top - MASK_EDGE_HALO_PX)
                    p_bottom = min(linear_rgb.shape[0], bottom + MASK_EDGE_HALO_PX)
                    p_rows = geometry.rows(p_top, p_bottom)
                    p_grid = p_rows.source_grid()
                    p_strip = linear_rgb[p_top:p_bottom]
                    if active_spots:
                        with timed(timer, "spots"):
                            p_strip = apply_spots(p_strip, values.spots, p_rows, src_lin, grid=p_grid, prepared=prepared_spots, cancel=cancel)
                    p_sampled = apply_vignetting(p_strip, p_rows, values.lens_corrections.vignetting_ev)
                    guide_l = perceptual_luma_guide(p_sampled, values.exposure_ev)
                    p_fields, p_w_field = local_mask_fields(
                        values.local_gradients, p_rows, values.highlights, values.shadows,
                        grid=p_grid, guide_l=guide_l,
                    )
                    sl = slice(top - p_top, top - p_top + (bottom - top))
                    sampled = p_sampled[sl]
                    fields = tuple(f[sl] for f in p_fields)
                    w_field = p_w_field[sl] if p_w_field is not None else None
                else:
                    rows=geometry.rows(top,bottom)
                    grid=rows.source_grid() if (active_spots or active_masks) else None
                    strip_in=linear_rgb[top:bottom]
                    if active_spots:
                        with timed(timer, "spots"):
                            strip_in=apply_spots(strip_in,values.spots,rows,src_lin,grid=grid,prepared=prepared_spots,cancel=cancel)
                    sampled=apply_vignetting(strip_in,rows,values.lens_corrections.vignetting_ev)
                    if active_masks:
                        guide_l = perceptual_luma_guide(sampled, values.exposure_ev) if needs_guide else None
                        fields, w_field = local_mask_fields(
                            values.local_gradients, rows, values.highlights, values.shadows,
                            grid=grid, guide_l=guide_l,
                        )
                    else:
                        fields = None
                        w_field = None
                result[top:bottom]=render_srgb(sampled,values.exposure_ev,
                    highlights=values.highlights,shadows=values.shadows,cancel=cancel,scratch=scratch,local_fields=fields,local_warmth=w_field,
                    base_rendering=values.base_rendering)
        else:
            result = render_srgb(linear_rgb, values.exposure_ev, highlights=values.highlights,
                                 shadows=values.shadows, cancel=cancel, scratch=scratch,
                                 base_rendering=values.base_rendering)
    x, y = np.asarray(values.rgb_curve, dtype=np.float32).T
    for start in range(0, result.shape[0], 128):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        block = result[start:start + 128]
        with timed(timer, "tones"):
            _tone_block(block, values, scratch=scratch)
        if not values.color_grading.is_identity:
            with timed(timer, "color_grading"):
                apply_color_grading(block, values.color_grading, scratch=scratch, in_place=True)
        with timed(timer, "hsl"):
            apply_color_mixer(block, values.color_mixer, scratch=scratch)
        if values.rgb_curve != IDENTITY_CURVE:
            with timed(timer, "curve"):
                block[:] = np.interp(block, x, y)
    if values.detail.active:
        with timed(timer, "detail"):
            apply_detail(result, values.detail, cancel=cancel, scratch=scratch)
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
