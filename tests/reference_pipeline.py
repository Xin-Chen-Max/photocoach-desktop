"""Frozen verbatim 0.7.0 reference implementations for exact-pixel optimization tests.

Copied unchanged from photocoach 0.7.0 (commit 9dff8b8) color_mixer.apply_color_mixer
and composition.sample_composition. Do not "improve" these: optimized production
code must stay bit-identical to them.
"""
import math

import numpy as np

from photocoach.color_mixer import CENTERS
from photocoach.composition import canvas_size, crop_bounds, output_size


def reference_apply_color_mixer(rgb, mixer):
    if mixer.neutral:
        return
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    chroma = maximum - minimum
    lightness = (maximum + minimum) * np.float32(.5)
    colored = chroma > 0
    saturation = np.zeros_like(chroma)
    # At almost-white/black pixels float32 can round L to an endpoint while
    # chroma is still nonzero. HSL S is bounded; avoid an endpoint division by 0.
    denominator = np.maximum(1 - np.abs(2 * lightness - 1), chroma)
    np.divide(chroma, denominator, out=saturation, where=colored)
    hue = np.zeros_like(chroma)
    difference = np.empty_like(chroma)
    for channel, offset in ((0, 0), (1, 2), (2, 4)):
        np.subtract(rgb[..., (channel+1) % 3], rgb[..., (channel+2) % 3], out=difference)
        np.divide(difference, chroma, out=difference, where=colored)
        mask = colored & (maximum == rgb[..., channel])
        hue[mask] = difference[mask] + offset
    hue = (hue / np.float32(6)) % 1
    index = np.minimum(np.searchsorted(CENTERS, hue, side="right") - 1, 7)
    blend = (hue - CENTERS[index]) / (CENTERS[index+1] - CENTERS[index])
    blend *= blend * (3 - 2 * blend)
    bands = np.array((*mixer.bands, mixer.bands[0]), np.float32)
    protection = np.clip(saturation / np.float32(.1), 0, 1)
    protection *= protection * (3 - 2 * protection)
    for axis in range(3):
        if not np.any(bands[:, axis]):
            continue
        amount = (bands[index, axis] * (1 - blend) + bands[index+1, axis] * blend) * protection
        if axis == 0:
            hue = (hue + amount / np.float32(360)) % 1
        else:
            amount /= np.float32(100)
            target = saturation if axis == 1 else lightness
            target += np.where(amount >= 0, amount * (1-target), amount * target)
            np.clip(target, 0, 1, out=target)
    amplitude = saturation * np.minimum(lightness, 1-lightness)
    for channel, offset in enumerate((0, 8, 4)):
        k = (offset + hue * 12) % 12
        rgb[..., channel] = lightness - amplitude * np.maximum(-1, np.minimum(np.minimum(k-3, 9-k), 1))
    np.clip(rgb, 0, 1, out=rgb)


def reference_sample_composition(linear, composition, *, source_size=None, rect=None, target_size=None):
    linear = np.asarray(linear)
    if (linear.ndim != 3 or linear.shape[2] != 3 or min(linear.shape[:2]) < 1 or
            not np.issubdtype(linear.dtype, np.floating)):
        raise ValueError("Composition requires a nonempty floating-point RGB array.")
    size = source_size or (linear.shape[1], linear.shape[0])
    ow, oh = output_size(size, composition)
    l, t, r, b = rect or (0, 0, ow, oh)
    if not (0 <= l < r <= ow and 0 <= t < b <= oh):
        raise ValueError("Invalid composition output region.")
    tw, th = target_size or (r-l, b-t)
    if tw < 1 or th < 1:
        raise ValueError("Invalid composition sample size.")
    cl, ct, _, _ = crop_bounds(size, composition)
    if composition.angle == 0 and size == (linear.shape[1], linear.shape[0]) and (tw, th) == (r-l, b-t):
        return linear[ct+t:ct+b, cl+l:cl+r]
    cw, ch = canvas_size(size, composition.angle)
    x = cl + l + (np.arange(tw, dtype=np.float64)+.5)*(r-l)/tw - cw/2
    y = ct + t + (np.arange(th, dtype=np.float64)+.5)*(b-t)/th - ch/2
    a = math.radians(composition.angle)
    c, s = math.cos(a), math.sin(a)
    sx = (c*x[None,:] + s*y[:,None] + size[0]/2)*linear.shape[1]/size[0] - .5
    sy = (-s*x[None,:] + c*y[:,None] + size[1]/2)*linear.shape[0]/size[1] - .5
    sx = np.clip(sx, 0, linear.shape[1]-1)
    sy = np.clip(sy, 0, linear.shape[0]-1)
    x0, y0 = np.floor(sx).astype(np.intp), np.floor(sy).astype(np.intp)
    x1, y1 = np.minimum(x0+1, linear.shape[1]-1), np.minimum(y0+1, linear.shape[0]-1)
    fx, fy = (sx-x0).astype(np.float32)[...,None], (sy-y0).astype(np.float32)[...,None]
    upper = linear[y0,x0]*(1-fx) + linear[y0,x1]*fx
    lower = linear[y1,x0]*(1-fx) + linear[y1,x1]*fx
    return upper*(1-fy) + lower*fy


def reference_render_srgb(linear_rgb, exposure_ev=0.0, *, highlights=0.0, shadows=0.0):
    source = np.asarray(linear_rgb)
    result = np.empty(source.shape, dtype=np.float32)
    multiplier = np.float32(2.0 ** float(exposure_ev))
    for start in range(0, source.shape[0], 128):
        block = source[start:start + 128]
        out = result[start:start + 128]
        np.multiply(block, multiplier, out=out)
        if highlights or shadows:
            luminance = np.maximum(out[...,0] * .2126 + out[...,1] * .7152 + out[...,2] * .0722, 0)
            square = luminance * luminance
            shift = np.float32(shadows) * np.float32(.18**2) / (square + np.float32(.18**2))
            shift += np.float32(highlights) * square / (square + np.float32(.5**2))
            out *= np.exp2(shift)[..., None]
        np.clip(out, 0.0, 1.0, out=out)
        low = out <= np.float32(0.0031308)
        out[low] *= np.float32(12.92)
        out[~low] = np.float32(1.055) * np.power(out[~low], np.float32(1.0 / 2.4)) - np.float32(0.055)
        np.clip(out, 0.0, 1.0, out=out)
    return result


def reference_render_edit(linear_rgb, values):
    from photocoach.state import IDENTITY_CURVE
    result = reference_render_srgb(
        linear_rgb, values.exposure_ev, highlights=values.highlights, shadows=values.shadows
    )
    x, y = np.asarray(values.rgb_curve, dtype=np.float32).T
    for start in range(0, result.shape[0], 128):
        block = result[start:start + 128]
        if values.whites or values.blacks:
            shift = np.zeros_like(block)
            if values.whites:
                shift += np.float32(.25 * values.whites) * block**4
            if values.blacks:
                shift += np.float32(.25 * values.blacks) * (1-block)**4
            block += shift
            np.clip(block, 0, 1, out=block)
        if values.contrast:
            block -= np.float32(.5)
            block *= np.float32(2 ** values.contrast)
            block += np.float32(.5)
            np.clip(block, 0, 1, out=block)
        if values.saturation != 1:
            gray = (block[..., 0] * .2126 + block[..., 1] * .7152 + block[..., 2] * .0722)[..., None]
            block -= gray
            block *= np.float32(values.saturation)
            block += gray
            np.clip(block, 0, 1, out=block)
        if values.vibrance:
            maximum = np.maximum(np.maximum(block[...,0], block[...,1]), block[...,2])
            minimum = np.minimum(np.minimum(block[...,0], block[...,1]), block[...,2])
            sat = np.zeros_like(maximum)
            np.divide(maximum-minimum, maximum, out=sat, where=maximum > 0)
            factor = 1 + np.float32(values.vibrance) * (1-sat)**2
            gray = (block[...,0]*.2126 + block[...,1]*.7152 + block[...,2]*.0722)[...,None]
            block -= gray
            block *= factor[...,None]
            block += gray
            np.clip(block, 0, 1, out=block)
        reference_apply_color_mixer(block, values.color_mixer)
        if values.rgb_curve != IDENTITY_CURVE:
            block[:] = np.interp(block, x, y)
    return result
