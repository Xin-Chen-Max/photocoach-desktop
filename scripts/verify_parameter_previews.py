"""Offline parameter matrix: settled pixels, exports, history and spatial previews.

Synthetic colors exercise bands absent from a particular photograph. A native
RAW region exercises the same cases on actual sensor data. Full-image native UI
acceptance is separate. Source photos are read-only and no service is contacted.
"""
from __future__ import annotations

import argparse
import colorsys
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import tifffile

from photocoach.adjustments import render_photo
from photocoach.app import PreviewEngine
from photocoach.color_mixer import ColorMixer, COLOR_KEYS, AXES
from photocoach.composition import Composition, output_size
from photocoach.detail import CAMERA_RAW_DETAIL, DetailControls
from photocoach.gradients import LinearGradient, RadialGradient
from photocoach.imaging import DecodedPhoto, RawOptions, decode_raw, export_image, quantize_srgb
from photocoach.lens import LensCorrections
from photocoach.preview import DetailRequest, LINEAR_CACHE_BYTES
from photocoach.safe_io import sha256_file
from photocoach.state import EditValues, StateStore
from photocoach.white_balance import balance_gains


def parameter_cases():
    base = EditValues(base_rendering='natural_v1')
    cases = []
    def add(name, after, before=base):
        cases.append((name, before, after))
    for name in ('exposure_ev', 'contrast', 'highlights', 'shadows', 'whites', 'blacks', 'vibrance'):
        for value in (-.5, .5):
            add(f'{name}_{value:+g}', replace(base, **{name: value}))
    for value in (.5, 1.5):
        add(f'saturation_{value:g}', replace(base, saturation=value))
    for value in (.3, .7):
        add(f'curve_{value:g}', replace(base, rgb_curve=((0., 0.), (.5, value), (1., 1.))))
    for color, key in enumerate(COLOR_KEYS):
        for axis, label in enumerate(AXES):
            limit = 12. if axis == 0 else 40.
            for value in (-limit, limit):
                add(f'hsl_{key}_{label}_{value:+g}', replace(base,
                    color_mixer=ColorMixer().with_value(color, axis, value)))
    for name in ('sharpening', 'luminance_nr', 'chroma_nr'):
        for value in (30., 80.):
            add(f'detail_{name}_{value:g}', replace(base, detail=DetailControls(**{name: value})))
    anchor = replace(base, detail=DetailControls(sharpening=80))
    for value in (30., 90.):
        add(f'detail_masking_{value:g}', replace(anchor, detail=replace(anchor.detail, masking=value)), anchor)
    add('camera_raw_detail_preset', replace(base, detail=CAMERA_RAW_DETAIL))
    for name in ('distortion', 'vignetting_ev'):
        for sign in (-1, 1):
            value = sign * (25 if name == 'distortion' else .7)
            add(f'lens_{name}_{value:+g}', replace(base, lens_corrections=LensCorrections(**{name: value})))
    add('crop', replace(base, composition=Composition((.1, .1, .9, .9))))
    for angle in (-3.25, 3.25):
        add(f'straighten_{angle:+g}', replace(base, composition=Composition(angle=angle)))
    for mode in ('neutral_v1', 'legacy_v1'):
        add(mode, replace(base, base_rendering=mode))
    for cls, kind in ((LinearGradient, 'linear'), (RadialGradient, 'radial')):
        mask = cls('audit', 'Audit mask', (.15, .15), (.8, .75))
        for name in ('exposure_ev', 'highlights', 'shadows', 'warmth'):
            for value in (-.5, .5):
                add(f'{kind}_{name}_{value:+g}', replace(base, local_gradients=(replace(mask, **{name: value}),)))
        active = replace(mask, exposure_ev=.8)
        masked = replace(base, local_gradients=(active,))
        for name, value in (('inverted', True), ('enabled', False), ('start', (.35, .3)), ('end', (.9, .9))):
            add(f'{kind}_{name}', replace(base, local_gradients=(replace(active, **{name: value}),)), masked)
        if kind == 'radial':
            for name, value in (('feather', .1), ('feather', 1.), ('angle', -45.), ('angle', 45.)):
                add(f'{kind}_{name}_{value:g}', replace(base, local_gradients=(replace(active, **{name: value}),)), masked)
    combined = replace(base, exposure_ev=.35, highlights=-.3, shadows=.2, contrast=.15,
        detail=CAMERA_RAW_DETAIL, composition=Composition((.05, .05, .95, .95), angle=3.25),
        lens_corrections=LensCorrections(-20, .3),
        local_gradients=(RadialGradient('combined', 'Combined', (.2, .1), (.8, .9), exposure_ev=.5, warmth=.1, angle=30),))
    add('combined_spatial_tonal', combined)
    return cases


def synthetic_image():
    row = np.array([colorsys.hsv_to_rgb(x / 504, .7, .65) for x in range(504)], np.float32)
    image = row[None, ...] * np.linspace(.05, 1.2, 336, dtype=np.float32)[:, None, None]
    yy, xx = np.mgrid[:336, :504]
    image = image + .008 * np.sin(xx * .33)[..., None] + np.random.default_rng(77).normal(0, .004, image.shape)
    return np.clip(image, .0001, .95).astype(np.float32)


def run_matrix(linear, source, digest, output, *, require_effect):
    output.mkdir()
    linear.setflags(write=False)
    h, w = linear.shape[:2]
    def virtual_decode(path, options, *, half_size):
        pixels = linear[::2, ::2].copy() if half_size else linear
        pixels.setflags(write=False)
        return DecodedPhoto(pixels, {'sha256': digest, 'photo_id': digest,
            'visible_size': [w, h], 'output_size': [pixels.shape[1], pixels.shape[0]],
            'orientation_flip': 0, 'half_size': half_size}, options, half_size, path)
    rows, tiles = [], []
    with patch('photocoach.imaging.decode_raw', virtual_decode):
        engine = PreviewEngine()
        engine.load(source)
        store = StateStore()
        store.activate(source, digest, base_rendering='natural_v1')
        for name, before, after in parameter_cases():
            # Activate a new test session, then apply the complete edit as one transaction.
            store.activate(source, digest, base_rendering='natural_v1')
            store.apply_changes(before.fields().items())
            start = store.current
            store.apply_changes(after.fields().items())
            state = store.current
            assert state.values == after
            size = output_size((w, h), after.composition)
            request = DetailRequest(1, size, (0, 0, *size), source_size=(w, h))
            expected = render_photo(linear, after)
            rendered, elapsed, _ = engine.detail(state, request)
            assert np.array_equal(rendered, quantize_srgb(expected)), name + ': formal/export mismatch'
            # Also read an actual 16-bit TIFF, rather than just testing the encoder in memory.
            target = output / (name + '.tif')
            export_image(target, expected, source_path=source)
            saved = tifffile.imread(target)
            error = np.abs(np.rint(saved.astype(np.float64) / 257).astype(np.int16) - rendered.astype(np.int16))
            assert error.max() <= 1, name + ': TIFF quantization mismatch'
            decoded = engine.cache.decode_count
            drag_request = replace(request, step=max(4, int(np.ceil(max(size) / 160))),
                                   adjusting_edge=160, native_detail=True)
            if after.detail.active or before.detail.active:
                drag = engine.adjusting(state, drag_request)[0]
                assert np.array_equal(drag, engine.detail(state, drag_request)[0]), name + ': detail drag scale'
            assert engine.cache.decode_count == decoded
            assert engine.retained_bytes <= LINEAR_CACHE_BYTES
            previous = quantize_srgb(render_photo(linear, before))
            changed = previous.shape != rendered.shape or not np.array_equal(previous, rendered)
            if require_effect:
                assert changed, name + ': no effect on synthetic colors/texture'
            store.undo()
            assert store.current.values == start.values, name + ': undo'
            store.redo()
            assert store.current.values == after, name + ': redo'
            row = dict(parameter=name, effect_visible_in_pixels=changed, native_export_max_8bit_error=int(error.max()),
                       formal_ms=round(elapsed * 1000, 2), retained_bytes=engine.retained_bytes)
            rows.append(row)
            if name in ('detail_sharpening_80', 'detail_luminance_nr_80', 'detail_chroma_nr_80', 'camera_raw_detail_preset', 'combined_spatial_tonal'):
                for label, pixels in (('before', previous), ('after', rendered)):
                    Image.fromarray(pixels).save(output / (name + '-' + label + '.png'))
                    thumb = Image.fromarray(pixels).resize((378, 252))
                    tile = Image.new('RGB', (390, 282), '#202020')
                    tile.paste(thumb, (6, 24)); ImageDraw.Draw(tile).text((6, 5), name + ' ' + label, fill='white')
                    tiles.append(tile)
    sheet = Image.new('RGB', (780, 282 * ((len(tiles) + 1) // 2)), '#202020')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (390 * (i % 2), 282 * (i // 2)))
    sheet.save(output / 'detail-comparison.jpg')
    (output / 'report.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    digest = sha256_file(args.source)
    fixture = args.output / 'synthetic.input'
    fixture.write_bytes(b'generated offline input')
    synthetic = run_matrix(synthetic_image(), fixture, hashlib.sha256(fixture.read_bytes()).hexdigest(),
                           args.output / 'synthetic', require_effect=True)
    photo = decode_raw(args.source, half_size=False)
    h, w = photo.linear_rgb.shape[:2]
    # A 756 x 504 native region: this is not claimed as a full-frame visual audit.
    left, top = min(w - 756, w // 3), min(h - 504, h // 3)
    native = photo.linear_rgb[top:top + 504, left:left + 756].copy()
    real = run_matrix(native, args.source, digest, args.output / 'raw-region', require_effect=False)
    wb = []
    for label, gains in [('warmth', balance_gains(.25, 0)), ('tint', balance_gains(0, .25)),
                         ('red_gain', (1.25, 1., 1.)), ('green_gain', (1., 1.25, 1.)), ('blue_gain', (1., 1., 1.25))]:
        options = RawOptions('custom', gains)
        start = time.perf_counter()
        changed = decode_raw(args.source, options, half_size=False)
        assert changed.options == options and not np.array_equal(changed.linear_rgb, photo.linear_rgb)
        wb.append(dict(parameter=label, decode_seconds=round(time.perf_counter()-start, 3)))
        del changed
    assert sha256_file(args.source) == digest
    report = dict(synthetic_cases=len(synthetic), raw_region_cases=len(real), white_balance=wb,
                  raw_region=[left, top, left+native.shape[1], top+native.shape[0]], source_sha256=digest,
                  original_unchanged=True, provider_calls=0)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
