"""Five warmed repeats for faithful detail previews; one read-only RAW."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
from photocoach.app import PreviewEngine
from photocoach.detail import DetailControls
from photocoach.preview import DetailRequest, oriented_size, LINEAR_CACHE_BYTES
from photocoach.safe_io import sha256_file
from photocoach.state import EditState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    engine = PreviewEngine()
    start = time.perf_counter()
    photo, _ = engine.load(args.source)
    cold_import = time.perf_counter()-start
    size = oriented_size(photo.metadata)
    state = EditState(photo.metadata['sha256'], args.source, 'benchmark', base_rendering='natural_v1')
    sharp = replace(state, detail=DetailControls(sharpening=80))
    w, h = size
    native_rect = (w//3, h//3, min(w, w//3+960), min(h, h//3+640))
    fit = DetailRequest(1, size, (0, 0, *size), step=max(1, int(np.ceil(max(size)/960))), source_size=size, native_detail=True)
    start = time.perf_counter()
    engine.detail(state, fit)
    first_native = time.perf_counter()-start
    decoded = engine.cache.decode_count
    rows = []
    for label, request, edited in [('fit-detail', fit, sharp),
        ('100-detail', DetailRequest(1, size, native_rect, source_size=size, native_detail=True), sharp),
        ('fit-tonal-fast', replace(fit, native_detail=False), replace(state, exposure_ev=.5))]:
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = engine.adjusting(edited, request)[0]
            times.append((time.perf_counter()-start)*1000)
        row = dict(view=label, repeats_ms=[round(v, 2) for v in times],
                   median_ms=round(statistics.median(times),2), max_ms=round(max(times),2),
                   retained_bytes=engine.retained_bytes)
        if label.endswith('detail'):
            actual = engine.detail(edited, request)[0]
            assert np.array_equal(result, actual)
            neutral = engine.adjusting(state, request)[0]
            row.update(max_formal_error=0, effect_mae_8bit=float(np.abs(result.astype(float)-neutral).mean()))
            Image.fromarray(result).save(args.output/(label+'-sharp.png'))
            Image.fromarray(neutral).save(args.output/(label+'-neutral.png'))
        rows.append(row)
        assert engine.retained_bytes <= LINEAR_CACHE_BYTES
    assert engine.cache.decode_count == decoded
    assert sha256_file(args.source) == state.photo_id
    peak = None
    if __import__('sys').platform == 'win32':
        import ctypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ('peak', 'working', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb):
            peak = counters.peak
    report = dict(rows=rows, cold_import_seconds=round(cold_import,3),
                  first_native_including_decode_seconds=round(first_native,3), decode_count=decoded,
                  repeated_render_decodes=0, peak_working_set_bytes=peak, source_unchanged=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
