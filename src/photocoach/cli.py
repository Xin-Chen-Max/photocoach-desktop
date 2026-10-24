"""Local RAW verification, processing diagnostics, and sample export.

Examples::

    python -m photocoach.cli --versions
    python -m photocoach.cli photo.ARW --output new-check --exposure 0.5 --tiff

The output directory must be new and its parent must already exist. Technical
checks do not certify visual color quality, correct orientation, or camera
compatibility beyond the one actual source file supplied by the caller.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

from . import __version__
from .imaging import (
    PIPELINE_VERSION,
    RawOptions,
    decode_raw,
    dependency_versions,
    export_image,
    render_srgb,
    srgb_icc_profile,
)
from .safe_io import atomic_write, sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _make_output_directory(path: Path) -> Path:
    """Create exactly one new directory, with no exist_ok / parent creation."""
    requested = path.expanduser()
    if requested.name in ("", ".", "..") or ":" in requested.name or requested.name.endswith((" ", ".")):
        raise ValueError("Invalid output directory name.")
    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError("The output directory's parent does not exist.")
    target = parent / requested.name
    if os.path.lexists(target):
        raise FileExistsError("The output directory already exists. Choose a new directory name.")
    target.mkdir(parents=False, exist_ok=False)
    return target


def _write_report(path: Path, report: dict[str, Any], source: Path) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_write(
        path,
        lambda temporary: temporary.write_text(payload, encoding="utf-8"),
        forbidden_paths=(source,),
    )


def _image_check(image: np.ndarray) -> dict[str, Any]:
    valid_shape = image.ndim == 3 and image.shape[2] == 3 and min(image.shape[:2]) > 0
    finite = bool(np.isfinite(image).all())
    result = {
        "shape_height_width_channels": list(image.shape),
        "dtype": str(image.dtype),
        "rgb_shape_valid": bool(valid_shape),
        "all_finite": finite,
        "minimum": float(image.min()) if finite and image.size else None,
        "maximum": float(image.max()) if finite and image.size else None,
    }
    if not valid_shape or not finite or image.dtype != np.float32:
        raise ValueError("The linear image must contain finite float32 RGB data.")
    return result


def _orientation_check(metadata: dict[str, Any]) -> dict[str, Any]:
    """Only compare reported dimensions; a scene must confirm actual rotation."""
    visible = list(metadata["visible_size"])
    actual = list(metadata["output_size"])
    flip = int(metadata["orientation_flip"])
    expected = visible[::-1] if flip in (5, 6) else visible
    known = flip in (0, 3, 5, 6)
    return {
        "raw_flip": flip,
        "visible_size_width_height": visible,
        "full_output_size_width_height": actual,
        "expected_full_size_from_orientation": expected if known else None,
        "dimensions_match_orientation": actual == expected if known else None,
        "visual_rotation_check": "pending_manual",
        "limitation": "Matching dimensions do not prove correct scene orientation. Check landscape and portrait images manually.",
    }


def _inspect_export(path: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    """Reopen completed outputs and verify their encoding and embedded profile."""
    from PIL import Image, ImageCms

    if path.suffix.lower() in (".jpg", ".jpeg"):
        with Image.open(path) as image:
            image.load()
            dimensions = image.size
            mode = image.mode
            icc = image.info.get("icc_profile", b"")
            pixels = np.asarray(image)
            bits = 8 if pixels.dtype == np.uint8 else None
            exif_absent = not bool(image.getexif())
            dtype = str(pixels.dtype)
    else:
        import tifffile

        with tifffile.TiffFile(path) as tiff:
            page = tiff.pages[0]
            pixels = page.asarray()
            dimensions = (page.imagewidth, page.imagelength)
            mode = "RGB" if pixels.ndim == 3 and pixels.shape[2] == 3 else "unexpected"
            tag = page.tags.get(34675)
            icc = b"" if tag is None else bytes(tag.value)
            bits = 16 if pixels.dtype == np.uint16 else None
            exif_absent = "ExifTag" not in page.tags and "GPSTag" not in page.tags
            dtype = str(pixels.dtype)

    expected_bits = 8 if path.suffix.lower() in (".jpg", ".jpeg") else 16
    profile_matches = icc == srgb_icc_profile()
    profile_name = ImageCms.getProfileName(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip() if icc else None
    valid = dimensions == expected_size and mode == "RGB" and bits == expected_bits and profile_matches and exif_absent
    result = {
        "file": path.name,
        "byte_length": path.stat().st_size,
        "size_width_height": list(dimensions),
        "mode": mode,
        "dtype": dtype,
        "bits_per_channel": bits,
        "icc_bytes": len(icc),
        "icc_profile_name": profile_name,
        "icc_matches_srgb_profile": profile_matches,
        "source_exif_absent": exif_absent,
        "roundtrip_check_passed": valid,
    }
    if not valid:
        raise ValueError("The exported image's dimensions, bit depth, or ICC profile do not match the expected values.")
    return result


def _difference(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Report numerical differences without setting a fake aesthetic threshold."""
    if first.shape != second.shape:
        return {
            "comparable": False,
            "first_shape": list(first.shape),
            "second_shape": list(second.shape),
            "reason": "Output dimensions differ. No implicit second resize or pixel comparison was performed.",
        }
    difference = np.abs(first - second)
    return {
        "comparable": True,
        "mean_absolute_error_srgb_0_to_1": float(difference.mean(dtype=np.float64)),
        "max_absolute_error_srgb_0_to_1": float(difference.max()),
        "rmse_srgb_0_to_1": float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))),
        "interpretation": "Descriptive measurements only. Half-size and full-size reconstruction differ; these are not color or visual-quality acceptance thresholds.",
    }


def verify(source: Path, output: Path, *, exposure: float, tiff: bool, wb_checks: bool) -> int:
    """Run one file's probe synchronously (CLI only); return an exit status."""
    # Delayed import keeps --versions independent of the preview/UI module.
    from .preview import reduce_linear

    source = source.expanduser().resolve(strict=True)
    if source.suffix.lower() != ".arw" or not source.is_file():
        raise ValueError("Select an existing Sony .ARW file.")
    if not math.isfinite(exposure) or not -5 <= exposure <= 5:
        raise ValueError("Exposure must be a finite value between -5 and +5 EV.")
    report: dict[str, Any] = {
        "report_format": "photocoach-phase0-verification",
        "report_schema_version": 1,
        "started_utc": _utc_now(),
        "application_version": __version__,
        "pipeline_version": PIPELINE_VERSION,
        "dependency_versions": dependency_versions(),
        "source_name": source.name,
        "source_byte_length": source.stat().st_size,
        "parameters": {"exposure_ev": exposure, "white_balance": "camera", "preview_max_edge": 1600},
        "decoder_parameters": {
            "demosaic_algorithm": "AHD (full-size)",
            "half_size_preview": True,
            "half_size_export": False,
            "four_color_rgb": False,
            "use_camera_wb": False,
            "use_auto_wb": False,
            "user_wb": "validated as-shot RGBG multipliers normalized by G1; actual values in metadata",
            "output_color": "sRGB",
            "output_bps": 16,
            "gamma": [1.0, 1.0],
            "no_auto_bright": True,
            "no_auto_scale": False,
            "adjust_maximum_thr": 0.0,
            "bright": 1.0,
            "highlight_mode": "Clip",
            "exp_shift": 1.0,
            "exp_preserve_highlights": 0.0,
            "user_flip": "RAW orientation (no second rotation)",
        },
        "timings_seconds": {},
        "exports": [],
        "manual_checks": {
            "orientation_horizontal_and_portrait": "pending_manual",
            "camera_white_balance_and_color": "pending_manual",
            "highlights_and_clipping": "pending_manual",
            "display_icc_and_sdr_configuration": "pending_manual",
            "interactive_performance": "pending_gui_manual",
        },
        "stage0_gate": "NOT_PASSED_REQUIRES_USER_SAMPLE_REVIEW_AND_MANUAL_ACCEPTANCE",
        "notes": [
            "This run checks only the supplied file, not every A7C II RAW mode.",
            "Camera WB and custom unity gains use explicit user_wb. Automatic WB and automatic brightening are disabled.",
            "Clipping may occur before LibRaw's 16-bit output. This report cannot establish RAW highlight recoverability.",
            "Half-size previews cannot establish final detail, sharpness, or noise quality. Exports use an independent full-size decode.",
            "Embedding and reading an ICC profile does not establish correct monitor color management.",
            "CLI timings do not replace manual GUI responsiveness checks.",
        ],
    }
    destination = _make_output_directory(output)
    start = time.perf_counter()
    current_step = "source_hash_before"
    error: Exception | None = None

    def measured(name: str, action: Callable[[], Any]) -> Any:
        nonlocal current_step
        current_step = name
        before = time.perf_counter()
        try:
            return action()
        finally:
            report["timings_seconds"][name] = round(time.perf_counter() - before, 6)

    try:
        report["source_sha256_before"] = measured("source_hash_before", lambda: sha256_file(source))
        half = measured("decode_camera_half", lambda: decode_raw(source, RawOptions(), half_size=True))
        report["half_metadata"] = half.metadata
        report["half_linear_check"] = _image_check(half.linear_rgb)
        half_linear = measured("reduce_half_linear", lambda: reduce_linear(half.linear_rgb, max_edge=1600))
        baseline = measured("render_preview_baseline", lambda: render_srgb(half_linear, 0.0))
        current = measured("render_preview_current", lambda: render_srgb(half_linear, exposure))
        for filename, pixels in (("preview-baseline.jpg", baseline), ("preview-current.jpg", current)):
            saved = measured(filename, lambda name=filename, data=pixels: export_image(destination / name, data, source_path=source))
            report["exports"].append(_inspect_export(saved, (pixels.shape[1], pixels.shape[0])))

        if wb_checks:
            unity = measured(
                "decode_custom_unity_half",
                lambda: decode_raw(source, RawOptions("custom", (1.0, 1.0, 1.0)), half_size=True),
            )
            unity_equal = bool(np.array_equal(half.linear_rgb, unity.linear_rgb))
            report["custom_unity_wb_check"] = {
                "bit_exact_linear_match_to_camera": unity_equal,
                "camera_effective_multipliers": half.metadata["effective_whitebalance"],
                "custom_effective_multipliers": unity.metadata["effective_whitebalance"],
            }
            if not unity_equal:
                raise ValueError("Camera WB and custom unity gains produced different results.")
            del unity
            custom = measured(
                "decode_custom_red_1_1_half",
                lambda: decode_raw(source, RawOptions("custom", (1.1, 1.0, 1.0)), half_size=True),
            )
            custom_preview = render_srgb(reduce_linear(custom.linear_rgb, max_edge=1600), exposure)
            report["custom_red_1_1_wb_check"] = {
                "relative_rgb_gain": [1.1, 1.0, 1.0],
                "effective_multipliers": custom.metadata["effective_whitebalance"],
                "comparison_to_camera": _difference(current, custom_preview),
                "visual_assessment": "pending_manual",
            }
            saved = measured("preview-custom-red-1.1.jpg", lambda: export_image(destination / "preview-custom-red-1.1.jpg", custom_preview, source_path=source))
            report["exports"].append(_inspect_export(saved, (custom_preview.shape[1], custom_preview.shape[0])))
            del custom, custom_preview
        else:
            report["custom_white_balance_checks"] = "not_requested_use_--wb-checks"
        del half, baseline, half_linear

        full = measured("decode_camera_full", lambda: decode_raw(source, RawOptions(), half_size=False))
        report["full_metadata"] = full.metadata
        report["full_linear_check"] = _image_check(full.linear_rgb)
        report["orientation_dimension_check"] = _orientation_check(full.metadata)
        if report["orientation_dimension_check"]["dimensions_match_orientation"] is False:
            raise ValueError("Full-size output dimensions do not match the RAW visible area and orientation fields.")
        full_reduced = measured("reduce_full_linear", lambda: reduce_linear(full.linear_rgb, max_edge=1600))
        full_preview = measured("render_full_reduced", lambda: render_srgb(full_reduced, exposure))
        report["half_vs_full_reduced"] = _difference(current, full_preview)
        saved = measured("full-reduced-reference.jpg", lambda: export_image(destination / "full-reduced-reference.jpg", full_preview, source_path=source))
        report["exports"].append(_inspect_export(saved, (full_preview.shape[1], full_preview.shape[0])))
        del current, full_reduced, full_preview

        final_pixels = measured("render_full_current", lambda: render_srgb(full.linear_rgb, exposure))
        del full
        for filename in (["current-full.jpg", "current-full-16bit.tiff"] if tiff else ["current-full.jpg"]):
            saved = measured(filename, lambda name=filename: export_image(destination / name, final_pixels, source_path=source))
            report["exports"].append(_inspect_export(saved, (final_pixels.shape[1], final_pixels.shape[0])))
        report["technical_probe_completed"] = True
    except Exception as exc:
        error = exc
        report["technical_probe_completed"] = False
        # Exception strings can contain absolute filenames from dependencies.
        report["error"] = {"step": current_step, "type": type(exc).__name__, "detail": "Technical verification stopped. Local paths from dependency errors are omitted."}
    finally:
        try:
            report["source_sha256_after"] = measured("source_hash_after", lambda: sha256_file(source))
            report["source_unchanged"] = report.get("source_sha256_before") == report["source_sha256_after"]
            if not report["source_unchanged"]:
                report["technical_probe_completed"] = False
                report["integrity_error"] = "The source hashes differ before and after verification. Verification failed."
                error = error or RuntimeError("Source integrity check failed.")
        except Exception as exc:
            report["source_unchanged"] = None
            report["technical_probe_completed"] = False
            report["integrity_error"] = "The final source hash could not be checked."
            error = error or exc
        report["finished_utc"] = _utc_now()
        report["total_elapsed_seconds"] = round(time.perf_counter() - start, 6)

    report_name = "report.json" if error is None else "failure-report.json"
    _write_report(destination / report_name, report, source)
    if error is not None:
        print(f"Technical verification failed: {report.get('error', {}).get('step', 'source_integrity')} ({type(error).__name__}).", file=sys.stderr)
        print(f"Failure report saved: {destination / report_name}", file=sys.stderr)
        return 1
    print(f"Technical checks and sample exports completed: {destination / report_name}")
    print("Orientation, color, highlights, and desktop responsiveness require visual review.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhotoCoach: local RAW verification and sample export, with no network requests.")
    parser.add_argument("source", nargs="?", type=Path, help="A real Sony .ARW source file (read-only)")
    parser.add_argument("--output", type=Path, help="A new output directory; its parent must already exist")
    parser.add_argument("--exposure", type=float, default=0.0, help="Linear exposure in EV, from -5 to +5; default: 0")
    parser.add_argument("--tiff", action="store_true", help="Also export a 16-bit RGB sRGB TIFF")
    parser.add_argument("--wb-checks", action="store_true", help="Check custom unity WB and red gain x1.1; requires two additional half-size decodes")
    parser.add_argument("--versions", action="store_true", help="Print installed versions without opening a photo")
    args = parser.parse_args(argv)
    if args.versions:
        print(json.dumps({"photocoach": __version__, "pipeline": PIPELINE_VERSION, **dependency_versions()}, ensure_ascii=False, indent=2))
        return 0
    if args.source is None or args.output is None:
        parser.error("Verification requires SOURCE and --output NEW_DIRECTORY.")
    try:
        return verify(args.source, args.output, exposure=args.exposure, tiff=args.tiff, wb_checks=args.wb_checks)
    except Exception as exc:
        # No source path or API secret from a dependency error is printed.
        print(f"Could not start or complete verification ({type(exc).__name__}). Check the source file, output directory, and installation.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
