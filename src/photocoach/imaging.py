"""RAW decoding pipeline: sensor WB -> linear sRGB -> exposure -> SDR sRGB.

This is a bounded SDR development experiment, not an unclipped scene-linear
RAW engine. LibRaw's uint16 output may already clip highlights / out-of-gamut
colors before our float32 stage. Negative exposure cannot recover that data.
Half-size Bayer processing differs from full-size AHD demosaicing; it cannot
establish noise, sharpening, or fine-detail quality.

Primary references consulted for this implementation:
https://letmaik.github.io/rawpy/api/rawpy.Params.html
https://letmaik.github.io/rawpy/api/rawpy.RawPy.html
https://github.com/LibRaw/LibRaw/blob/master/src/postprocessing/postprocessing_utils_dcrdefs.cpp
https://www.libraw.org/node/2601
https://pillow.readthedocs.io/en/stable/reference/ImageCms.html
https://github.com/cgohlke/tifffile
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import importlib.metadata
import math
from pathlib import Path
import platform
import threading
from concurrent.futures import CancelledError
from typing import Any, Literal

import numpy as np

from .safe_io import atomic_write, sha256_file, validate_destination
from .scratch import Scratch

PIPELINE_VERSION = "phase0-linear-srgb-v1"


@dataclass(frozen=True)
class RawOptions:
    """WB gains multiply camera sensor coefficients, not display RGB or Kelvin.

    Effective RGGB coefficients are normalized by the first green coefficient.
    LibRaw additionally normalizes them during scale_colors. Consequently a
    common gain applied to all channels is NOT exposure; use exposure_ev.
    """

    wb_mode: Literal["camera", "custom"] = "camera"
    wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.wb_mode not in ("camera", "custom"):
            raise ValueError("White balance mode must be camera or custom.")
        if len(self.wb_gains) != 3:
            raise ValueError("White balance requires three relative RGB multipliers.")
        gains = tuple(float(value) for value in self.wb_gains)
        if any(not math.isfinite(value) or not 0.25 <= value <= 4.0 for value in gains):
            raise ValueError("Relative white balance multipliers must be finite values from 0.25 to 4.0.")
        object.__setattr__(self, "wb_gains", gains)


@dataclass(frozen=True)
class DecodedPhoto:
    """Read-only float32 image; metadata is local only, never an AI payload.

    metadata contains: photo_id/sha256, file_size, pipeline_version, versions,
    raw_size/visible_size/output_size (width,height), orientation_flip,
    optional recommended_crop (unrotated left,top,width,height; not applied),
    camera_whitebalance/effective_whitebalance (sensor R,G,B,G2), black_levels,
    white_level, half_size, color_space and warnings. Optional shooting fields:
    make, model, lens, iso, exposure_seconds, aperture, focal_length_mm.
    No GPS, author, timestamps, or filesystem paths are copied into metadata.
    """

    linear_rgb: np.ndarray
    metadata: dict[str, Any]
    options: RawOptions
    half_size: bool
    source_path: Path


def dependency_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version()}
    for package in ("rawpy", "numpy", "Pillow", "tifffile", "PySide6"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    try:
        import rawpy

        versions["libraw"] = ".".join(str(part) for part in rawpy.libraw_version)
    except ImportError:
        versions["libraw"] = "not installed"
    return versions


def effective_whitebalance(
    camera_wb: list[float] | tuple[float, ...], options: RawOptions,
    *, color_desc: str = "RGBG", num_colors: int = 3,
) -> tuple[float, float, float, float]:
    """Validate as-shot RGBG coefficients and derive reproducible user WB."""
    if color_desc.rstrip("\x00") != "RGBG" or num_colors not in (3, 4):
        raise ValueError("Only RGBG sensor layouts are supported.")
    values = [float(value) for value in camera_wb]
    if len(values) != 4 or any(not math.isfinite(value) or value <= 0 for value in values[:3]):
        raise ValueError("This RAW has no valid camera white balance. Processing stopped to avoid an automatic fallback.")
    if values[3] == 0 and num_colors == 3:
        values[3] = values[1]
    if not math.isfinite(values[3]) or values[3] <= 0:
        raise ValueError("The second green white balance coefficient is invalid.")
    gains = options.wb_gains if options.wb_mode == "custom" else (1.0, 1.0, 1.0)
    multiplied = [values[0] * gains[0], values[1] * gains[1], values[2] * gains[2], values[3] * gains[1]]
    return tuple(value / multiplied[1] for value in multiplied)  # type: ignore[return-value]


def _rational(value: Any) -> float:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _shooting_metadata(path: Path) -> dict[str, Any]:
    """Read only allowlisted TIFF/EXIF headers, never embedded image pixels."""
    import tifffile

    result: dict[str, Any] = {}
    try:
        with tifffile.TiffFile(path) as tiff:
            tags = tiff.pages[0].tags
            for tag, key in (("Make", "make"), ("Model", "model")):
                if tag in tags:
                    result[key] = str(tags[tag].value).strip("\x00 ")
            exif = tags["ExifTag"].value if "ExifTag" in tags else {}
            if isinstance(exif, dict):
                for tag, key in (
                    ("ExposureTime", "exposure_seconds"),
                    ("FNumber", "aperture"),
                    ("FocalLength", "focal_length_mm"),
                    ("ISOSpeedRatings", "iso"),
                    ("PhotographicSensitivity", "iso"),
                ):
                    if tag in exif:
                        try:
                            value = _rational(exif[tag])
                            if math.isfinite(value) and value > 0:
                                result[key] = value
                        except (TypeError, ValueError, ZeroDivisionError):
                            pass
                if "LensModel" in exif:
                    result["lens"] = str(exif["LensModel"]).strip("\x00 ")
    except (OSError, ValueError, KeyError, IndexError):
        result["metadata_warning"] = "Some shooting information could not be read; RAW decoding is unaffected."
    return result


def _fingerprint(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def decode_raw(
    path: str | Path, options: RawOptions = RawOptions(), *, half_size: bool = True
) -> DecodedPhoto:
    """Decode with explicit WB and brightness settings; run on a worker thread.

    no_auto_scale stays False: disabling it also disables sensor white balance.
    adjust_maximum_thr=0 suppresses the image-dependent maximum adjustment.
    Camera WB is supplied explicitly rather than use_camera_wb=True, which can
    silently fall back to auto WB. Explicit user WB is used for both modes so
    LibRaw's color matrix selection follows the same semantics in both modes.
    """
    import rawpy

    source = Path(path).expanduser().resolve(strict=True)
    if source.suffix.lower() != ".arw" or not source.is_file():
        raise ValueError("Choose an existing Sony .ARW file.")
    before = _fingerprint(source)
    digest = sha256_file(source)
    # File-like input guarantees our source handle is opened read-only.
    with source.open("rb") as stream, rawpy.imread(stream) as raw:
        sizes = raw.sizes
        camera_wb = list(raw.camera_whitebalance)
        color_desc = raw.color_desc.decode("ascii", errors="replace")
        wb = effective_whitebalance(camera_wb, options, color_desc=color_desc, num_colors=raw.num_colors)
        metadata: dict[str, Any] = {
            "photo_id": digest,
            "sha256": digest,
            "file_size": before[0],
            "pipeline_version": PIPELINE_VERSION,
            "versions": dependency_versions(),
            "raw_size": [int(sizes.raw_width), int(sizes.raw_height)],
            "visible_size": [int(sizes.width), int(sizes.height)],
            "orientation_flip": int(sizes.flip),
            "camera_whitebalance": camera_wb,
            "effective_whitebalance": list(wb),
            "black_levels": [int(value) for value in raw.black_level_per_channel],
            "white_level": int(raw.white_level),
            "half_size": bool(half_size),
            "color_space": "linear sRGB (D65), bounded LibRaw uint16 -> float32",
            "warnings": [
                "LibRaw may clip highlights or out-of-gamut colors before output. Lowering exposure does not recover clipped RAW highlights.",
                "Half-size previews cannot establish final detail, noise, or sharpness." if half_size else "Full-size AHD reconstruction has different detail from the half-size preview.",
            ],
        }
        crop_fields = ("crop_left_margin", "crop_top_margin", "crop_width", "crop_height")
        if all(hasattr(sizes, name) for name in crop_fields):
            metadata["recommended_crop"] = [int(getattr(sizes, name)) for name in crop_fields]
        decoded = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            half_size=bool(half_size),
            four_color_rgb=False,
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=list(wb),
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            gamma=(1.0, 1.0),
            no_auto_bright=True,
            no_auto_scale=False,
            adjust_maximum_thr=0.0,
            bright=1.0,
            highlight_mode=rawpy.HighlightMode.Clip,
            exp_shift=1.0,
            exp_preserve_highlights=0.0,
            median_filter_passes=0,
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
            # user_flip=None uses RAW orientation; no second EXIF rotation.
        )
    if _fingerprint(source) != before:
        raise RuntimeError("Another program changed the source file during reading. Reopen the photo.")
    if decoded.dtype != np.uint16 or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError("The decoder did not return the expected 16-bit RGB image.")
    linear = decoded.astype(np.float32)
    linear *= np.float32(1.0 / 65535.0)
    linear.setflags(write=False)
    metadata["output_size"] = [int(linear.shape[1]), int(linear.shape[0])]
    metadata.update(_shooting_metadata(source))
    return DecodedPhoto(linear, metadata, options, bool(half_size), source)


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("The image must be a nonempty H x W x 3 RGB array.")
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Processing requires floating-point RGB input.")
    return image


def render_srgb(linear_rgb: np.ndarray, exposure_ev: float = 0.0, *,
                highlights: float = 0.0, shadows: float = 0.0, cancel=None,
                scratch: Scratch | None = None, local_fields=None) -> np.ndarray:
    """Same exposure and display transfer for preview and export, no auto tone.

    Exposure is 2**EV in linear light. Optional luminance-weighted zone gains
    follow exposure, before SDR clipping. With Y=max(0, dot(RGB, sRGB weights)),
    gain=2**(shadows*.18**2/(Y**2+.18**2) + highlights*Y**2/(Y**2+.5**2)).
    Amounts are [-1,1]; both zero is exactly the original pipeline. Positive
    lifts, negative darkens. This pointwise mapping preserves RGB ratios before
    clipping and has monotone luminance (minimum log slope > 1-ln(2)). It cannot
    reconstruct information clipped by LibRaw. SDR clips to [0,1], then applies the
    standard piecewise sRGB transfer. Row blocks keep temporary memory bounded.
    The input (including cached read-only arrays) is never modified.
    """
    source = _validate_rgb(linear_rgb)
    ev = float(exposure_ev)
    if not math.isfinite(ev) or not -5 <= ev <= 5:
        raise ValueError("Exposure must be a finite value from -5 to +5 EV.")
    for amount in (highlights, shadows):
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or not -1 <= amount <= 1:
            raise ValueError("Highlights and Shadows must be finite amounts from -1 to +1.")
    if scratch is None:
        scratch = Scratch()
    result = np.empty(source.shape, dtype=np.float32)
    if local_fields is not None and (len(local_fields)!=3 or any(
            field.shape!=source.shape[:2] or not np.isfinite(field).all() for field in local_fields)):
        raise ValueError("Local tonal fields must match the sample grid.")
    multiplier = np.float32(2.0 ** ev)
    sh_c = np.float32(shadows) * np.float32(.18**2)
    sh_d = np.float32(.18**2)
    hl_c = np.float32(highlights)
    hl_d = np.float32(.5**2)
    for start in range(0, source.shape[0], 128):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        block = source[start:start + 128]
        if not np.isfinite(block).all():
            raise ValueError("The image contains non-finite values.")
        out = result[start:start + 128]
        np.multiply(block, multiplier, out=out)
        if local_fields is not None:
            ev_field,hl_field,sh_field=(field[start:start+128] for field in local_fields)
            out *= np.exp2(ev_field)[...,None]
        if highlights or shadows or local_fields is not None:
            shape2 = out.shape[:2]
            luminance = scratch.get("srgb_lum", shape2, np.float32)
            square = scratch.get("srgb_sq", shape2, np.float32)
            shift = scratch.get("srgb_sh", shape2, np.float32)
            np.multiply(out[..., 0], np.float32(.2126), out=luminance)
            np.multiply(out[..., 1], np.float32(.7152), out=square)
            luminance += square
            np.multiply(out[..., 2], np.float32(.0722), out=square)
            luminance += square
            np.maximum(luminance, 0, out=luminance)
            np.multiply(luminance, luminance, out=square)
            np.add(square, sh_d, out=luminance)
            np.divide(sh_c if local_fields is None else sh_field*sh_d, luminance, out=shift)
            np.add(square, hl_d, out=luminance)
            np.multiply(hl_c if local_fields is None else hl_field, square, out=square)
            np.divide(square, luminance, out=luminance)
            shift += luminance
            np.exp2(shift, out=shift)
            out *= shift[..., None]
        np.clip(out, 0.0, 1.0, out=out)
        low = scratch.get("srgb_low", out.shape, np.bool_)
        buf = scratch.get("comp_s0", out.shape, np.float32)
        np.less_equal(out, np.float32(0.0031308), out=low)
        np.power(out, np.float32(1.0 / 2.4), out=buf)
        buf *= np.float32(1.055)
        buf -= np.float32(0.055)
        out *= np.float32(12.92)
        np.copyto(buf, out, where=low)
        np.clip(buf, 0.0, 1.0, out=out)
    return result


def quantize_srgb(srgb_float: np.ndarray, *, bits: int = 8) -> np.ndarray:
    """Quantize once at a display/export boundary, rejecting invalid samples."""
    source = _validate_rgb(srgb_float)
    if bits not in (8, 16):
        raise ValueError("Output precision must be 8 or 16 bits.")
    result = np.empty(source.shape, dtype=np.uint8 if bits == 8 else np.uint16)
    maximum = np.float32((1 << bits) - 1)
    for start in range(0, source.shape[0], 128):
        block = source[start:start + 128]
        if not np.isfinite(block).all():
            raise ValueError("The image contains non-finite values.")
        result[start:start + 128] = np.rint(np.clip(block, 0.0, 1.0) * maximum).astype(result.dtype)
    return result


@lru_cache(maxsize=1)
def srgb_icc_profile() -> bytes:
    """LittleCMS-generated standard sRGB profile for encoded SDR output."""
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def export_image(
    path: str | Path,
    srgb_float: np.ndarray,
    *,
    source_path: str | Path | None = None,
    jpeg_quality: int = 95,
) -> Path:
    """Export already rendered latest-state pixels, with ICC and no source EXIF.

    The caller must render a captured current state at full resolution. This
    function never substitutes cached preview pixels or loads another state.
    TIFF uses uint16 RGB and the same encoded sRGB transfer as JPEG.
    """
    from PIL import Image
    import tifffile

    forbidden = () if source_path is None else (source_path,)
    destination = validate_destination(path, forbidden_paths=forbidden)
    suffix = destination.suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".tif", ".tiff"):
        raise ValueError("Supported export extensions are .jpg, .jpeg, .tif, and .tiff.")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be from 1 to 100.")
    pixels = quantize_srgb(srgb_float, bits=8 if suffix in (".jpg", ".jpeg") else 16)
    profile = srgb_icc_profile()

    def write(temporary: Path) -> None:
        if suffix in (".jpg", ".jpeg"):
            Image.fromarray(pixels).save(
                temporary, format="JPEG", quality=jpeg_quality, subsampling=0,
                icc_profile=profile,
            )
        else:
            tifffile.imwrite(
                temporary, pixels, photometric="rgb", metadata=None,
                compression=None, iccprofile=profile,
                software=f"PhotoCoach {PIPELINE_VERSION}",
            )

    return atomic_write(destination, write, forbidden_paths=forbidden)


class DecodedCache:
    """Small worker-owned LRU. Exposure never changes the decode cache key.

    A WB, resolution, source stat, or pipeline change invalidates the key.
    WB changes currently redo unpack + LibRaw processing (on release, not each
    slider movement). Keeping an unpacked LibRaw object is a later optimization
    requiring memory/performance measurements on the user's real ARW.
    """

    def __init__(self, max_entries: int = 2, *, max_bytes: int = 512 * 1024 * 1024) -> None:
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("The cache must allow at least one entry.")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.decode_count = 0
        self.hits = 0
        self.high_water_bytes = 0
        self.full_identity: tuple[str, RawOptions] | None = None
        self._entries: OrderedDict[tuple[Any, ...], DecodedPhoto] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def retained_bytes(self) -> int:
        return sum(photo.linear_rgb.nbytes for photo in self._entries.values())

    def _drop_oldest(self) -> None:
        _, photo = self._entries.popitem(last=False)
        if not photo.half_size:
            self.full_identity = None

    def trim_to(self, max_bytes: int) -> None:
        """Release LRU decodes to make room for worker-owned derived pixels."""
        if max_bytes < 0:
            raise ValueError("The cache byte limit cannot be negative.")
        with self._lock:
            while self._entries and self.retained_bytes > max_bytes:
                self._drop_oldest()

    def get(self, path: str | Path, options: RawOptions = RawOptions(), *, half_size: bool = True,
            expected_bytes: int = 0, before_decode=None) -> DecodedPhoto:
        source = Path(path).expanduser().resolve(strict=True)
        key = (source, _fingerprint(source), options, bool(half_size), PIPELINE_VERSION)
        # Serialize raw operations for this cache; call only from a worker.
        with self._lock:
            if key in self._entries:
                self.hits += 1
                self._entries.move_to_end(key)
                return self._entries[key]
            # Let the serial worker release derived arrays before a new RAW
            # allocation. The callback must not call back into this cache.
            if before_decode is not None:
                before_decode()
            if expected_bytes > self.max_bytes:
                raise MemoryError("Full detail exceeds the linear cache budget. Quick preview remains available.")
            if not half_size:
                # Release a previous full reconstruction before allocating another.
                for old_key in list(self._entries):
                    if not self._entries[old_key].half_size:
                        del self._entries[old_key]
                        self.full_identity = None
            while self._entries and self.retained_bytes + expected_bytes > self.max_bytes:
                self._drop_oldest()
            decoded = decode_raw(source, options, half_size=half_size)
            self.decode_count += 1
            if decoded.linear_rgb.nbytes > self.max_bytes:
                raise MemoryError("Decoded image exceeds the linear cache budget.")
            while self._entries and (len(self._entries) >= self.max_entries or
                    self.retained_bytes + decoded.linear_rgb.nbytes > self.max_bytes):
                self._drop_oldest()
            self._entries[key] = decoded
            if not half_size:
                self.full_identity = (decoded.metadata["sha256"], options)
            self.high_water_bytes = max(self.high_water_bytes, self.retained_bytes)
            return decoded

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.full_identity = None
