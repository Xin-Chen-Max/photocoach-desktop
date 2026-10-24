"""Offline worker-only preview benchmark, with exact pixels and source checks.

This measures PreviewEngine.detail synchronously, excluding GUI timers, painting,
input latency and monitor scaling. --dpr2 doubles the requested native ROI; it
does not verify a physical high-DPI desktop. No settings, keys or APIs are used.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import time

import numpy as np

from photocoach import __version__
from photocoach.app import PreviewEngine
from photocoach.cli import _make_output_directory
from photocoach.composition import Composition, output_size
from photocoach.imaging import dependency_versions
from photocoach.preview import DetailRequest, oriented_size
from photocoach.safe_io import sha256_file
from photocoach.state import EditState


def fingerprint(pixels, metadata):
    """Hash contiguous output directly without retaining or copying image data."""
    assert pixels.dtype == np.uint8 and pixels.flags.c_contiguous
    histogram = json.dumps(metadata.get("display_histogram"), sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return {"shape": list(pixels.shape), "pixel_sha256": hashlib.sha256(memoryview(pixels)).hexdigest(),
            "pixel_sum": int(pixels.sum(dtype=np.uint64)),
            "histogram_sha256": hashlib.sha256(histogram).hexdigest()}


def timings(values):
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "values": values}


def cache_bytes(engine):
    small = getattr(engine, "_small", None)
    raw = engine.cache.retained_bytes
    return {"decoded": raw, "small_preview": 0 if small is None else small.nbytes,
            "engine_retained": getattr(engine, "retained_bytes", raw + (0 if small is None else small.nbytes))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to a local Sony ARW file")
    parser.add_argument("--output", type=Path, required=True, help="New directory; its parent must exist")
    parser.add_argument("--dpr2", action="store_true", help="1828 x 1372 native ROI instead of 914 x 686")
    parser.add_argument("--compare", type=Path, help="Earlier report.json; require identical output hashes")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = _make_output_directory(args.output)
    digest = sha256_file(source)
    engine = PreviewEngine()
    started = time.perf_counter()
    photo, pixels = engine.load(source)
    source_size = oriented_size(photo.metadata)
    load_ms = (time.perf_counter() - started) * 1000
    del photo, pixels
    composition = Composition((.08, .12, .92, .94), 3.25)
    state = EditState(digest, source, "offline-worker-benchmark", highlights=-.75,
                      shadows=.65, composition=composition)
    full_size = output_size(source_size, composition)
    factor = 2 if args.dpr2 else 1
    width, height = 914 * factor, 686 * factor
    assert full_size[0] > width + 14 and full_size[1] > height + 10
    left, top = (full_size[0] - width) // 2, (full_size[1] - height) // 2
    request = DetailRequest(0, full_size, (left, top, left + width, top + height),
                            source_size=source_size)
    started = time.perf_counter()
    pixels, _, metadata = engine.detail(state, request)
    first_ms = (time.perf_counter() - started) * 1000
    initial = fingerprint(pixels, metadata)
    del pixels, metadata
    decodes = engine.cache.decode_count
    edits, pans, frames = [], [], []
    for index in range(20):
        state = replace(state, revision=index + 1, contrast=.02 + index * .01)
        started = time.perf_counter()
        pixels, _, metadata = engine.detail(state, request)
        edits.append((time.perf_counter() - started) * 1000)
        frames.append({"kind": "edit", "index": index, "rect": list(request.rect),
                       **fingerprint(pixels, metadata)})
        del pixels, metadata
    for index in range(20):
        dx, dy = ((-7, -5) if index % 2 == 0 else (0, 0))
        request = replace(request, generation=index + 1,
                          rect=(left + dx, top + dy, left + width + dx, top + height + dy))
        started = time.perf_counter()
        pixels, _, metadata = engine.detail(state, request)
        pans.append((time.perf_counter() - started) * 1000)
        frames.append({"kind": "pan", "index": index, "rect": list(request.rect),
                       **fingerprint(pixels, metadata)})
        del pixels, metadata
    assert engine.cache.decode_count == decodes, "Warm editing or panning unexpectedly decoded RAW"
    # Extra profiled edit and same-ROI repeat are excluded from the timings above.
    profiler = cProfile.Profile()
    profiler.enable()
    state = replace(state, revision=21, contrast=.22)
    pixels, _, metadata = engine.detail(state, request)
    del pixels, metadata
    pixels, _, metadata = engine.detail(state, replace(request, generation=21))
    del pixels, metadata
    profiler.disable()
    stats = pstats.Stats(profiler)
    profile = [{"function": f"{Path(name[0]).name}:{name[1]}:{name[2]}",
                "calls": value[1], "self_ms": value[2] * 1000, "cumulative_ms": value[3] * 1000}
               for name, value in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:24]]
    unchanged = sha256_file(source) == digest
    assert unchanged, "Source contents changed"
    report = {"app_version": __version__, "sample": source.name, "source_sha256": digest,
              "original_unchanged": unchanged, "dependencies": dependency_versions(),
              "platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
              "measurement": "Synchronous worker only; excludes GUI debounce, paint and input latency",
              "dpr": factor, "source_size": source_size, "composition_size": full_size,
              "native_roi_size": [width, height], "initial_parameters": replace(state, contrast=0).values.to_dict(),
              "load_ms": load_ms, "first_detail_ms": first_ms,
              "warm_edits_ms": timings(edits), "warm_pan_ms": timings(pans),
              "warm_extra_decodes": engine.cache.decode_count - decodes,
              "total_decode_count": engine.cache.decode_count, "cache_bytes": cache_bytes(engine),
              "initial_output": initial, "frames": frames, "profile_extra_edit_and_repeat": profile,
              "api_attempts": 0}
    if args.compare:
        earlier = json.loads(args.compare.read_text(encoding="utf-8"))
        assert earlier["source_sha256"] == digest and earlier["dpr"] == factor
        report["exact_comparison"] = (earlier["initial_output"] == initial and earlier["frames"] == frames)
        report["comparison_source"] = args.compare.name
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {k: report[k] for k in ("sample", "dpr", "load_ms", "first_detail_ms", "warm_extra_decodes",
                                     "cache_bytes", "original_unchanged")}
    summary.update({key: {k: v for k, v in report[key].items() if k != "values"}
                    for key in ("warm_edits_ms", "warm_pan_ms")})
    if args.compare:
        summary["exact_comparison"] = report["exact_comparison"]
    print(json.dumps(summary, indent=2))
    assert report.get("exact_comparison", True), "Before/after pixels or histogram changed"


if __name__ == "__main__":
    main()
