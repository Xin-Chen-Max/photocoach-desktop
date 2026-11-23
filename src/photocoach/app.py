"""Single-photo editor. Serial image work, coalesced previews and protected user operations."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from urllib.parse import urlsplit
import sys
import time
import threading
from concurrent.futures import CancelledError
import numpy as np
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, QEvent
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox, QSlider, QWidget, QDialog,
)

from . import __version__
from .imaging import DecodedCache, RawOptions, decode_raw, export_image, quantize_srgb, render_srgb
from .models import ProbeState, ProbeStore
from .state import EditValues
from .lens import LensCorrections
from .detail import DetailControls
from .adjustments import render_edit, render_photo
from .composition import Composition, output_size, sample_composition
from .composition_ui import CompositionDialog
from .histogram import sampled_histogram
from .project import snapshot_project, write_project, read_project, SourceMissing
from .coach import MockCoach, CoachFeedback, build_request
from .coach_agent import check_trial
from .agent_memory import (PlaybookRAG, SessionWorkingMemory, StyleMemoryStore,
                           TurnSummary, compact_conversation)
from .agent_runtime import AgentProgress, AgentRunResult, AgentRuntime, RunBudget
from .agent_tools import build_default_tool_registry
from .telemetry import TraceStore, export_otel_spans
from .devmode import dev_mode_enabled
from .auto_coach import AutoCoachSchedule
from .api_coach import ApiCoach, ApiError, ApiResult, TransportProgress, TRANSPORT_STAGES
from .budget import SessionBudget
from .coach_ui import ApiSettingsDialog, SendReviewDialog, TextProbeReviewDialog, budget_text
from .coach_view import format_live_agent_step, usage_summary
from .settings import SettingsStore, SettingsError, resolve_api_key, credential_target
from .connections import ConnectionStore, settings_for, ConnectionBook, infer_provider, connection_from_settings
from .model_catalog import (ModelCatalogStore, classify_model, capability_evidence, CatalogError,
                            DEFAULT_CURATED, KNOWN_VISION_MODELS, is_ui_selectable_model)
from .credentials import WindowsCredentialStore
from .text_probe import run_text_probe, TextProbeResult
from .white_balance import balance_coordinates, balance_gains, sample_white_balance, inspect_sensor_headroom
from .preview import (reduce_linear, oriented_size, DetailRequest, render_region,
                      LINEAR_CACHE_BYTES, SMALL_RESERVE_BYTES)
from .detail_cache import DetailGeometryCache
from .render_timing import StageTimer, timed
from .gradients import RenderGeometry, OverlaySpec, overlay_pixels
from .safe_io import atomic_write, is_supported_raw_suffix, sha256_file, validate_destination


class TaskSignals(QObject):
    done = Signal(object, object)


class Worker(QRunnable):
    def __init__(self, function: Callable) -> None:
        super().__init__()
        self.function = function
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            result = self.function()
        except Exception as error:
            self.signals.done.emit(None, error)
        else:
            self.signals.done.emit(result, None)


@dataclass
class Job:
    kind: str
    generation: int
    function: Callable
    snapshot: ProbeState | None = None
    coach_token: int = 0
    view_request: DetailRequest | None = None
    automatic: bool = False
    cancel: threading.Event | None = None
    # Local diagnostics only: intent creation -> worker start, and per-stage
    # worker totals for image jobs. Never part of a project or coach payload.
    queued_at: float = field(default_factory=time.perf_counter)
    timing: dict | None = None
    overlay: OverlaySpec | None = None
    overlay_generation: int = 0
    coach_request: object | None = None


TIMED_JOB_KINDS = ("preview", "detail", "adjusting")
ADJUSTING_MAX_EDGE = 960


def display_render_spec(state: ProbeState) -> tuple[RawOptions, EditValues]:
    """Return the (RawOptions, EditValues) pair to render on canvas without altering state.values.

    - Baseline comparison keeps current crop/angle and distortion with neutral vignetting.
    - Snapshot A/B comparison keeps the snapshot's tone/color/detail/local masks while
      aligning crop/angle and distortion to the active viewport geometry so pan/zoom stay intact.
    """
    if state.compare_baseline:
        return (
            RawOptions(),
            EditValues(
                base_rendering=state.base_rendering,
                composition=state.composition,
                lens_corrections=LensCorrections(state.lens_corrections.distortion),
            ),
        )
    if state.compare_snapshot is not None:
        snap = state.compare_snapshot.values
        if (
            snap.composition != state.composition
            or snap.lens_corrections.distortion != state.lens_corrections.distortion
        ):
            snap = replace(
                snap,
                composition=state.composition,
                lens_corrections=replace(
                    snap.lens_corrections,
                    distortion=state.lens_corrections.distortion,
                ),
            )
        return snap.raw_options, snap
    return state.raw_options, state.values


class PreviewEngine:
    """Used exclusively by the serial worker; never decode for an EV change."""
    def __init__(self) -> None:
        self.cancel = None
        self.timer = None
        self.overlay_spec = None
        self.cache = DecodedCache(max_entries=3, max_bytes=LINEAR_CACHE_BYTES - SMALL_RESERVE_BYTES)
        self._geometry = DetailGeometryCache()
        self._small_key = None
        self._small = None
        self._histogram_key = None
        self._histogram = None
        self._adjusting_key = None
        self._adjusting_pixels = None

    def clear(self) -> None:
        self._adjusting_key = None
        self._adjusting_pixels = None
        self._geometry.clear()
        self.cache.clear()
        self._small_key = None
        self._small = None
        self._histogram_key = None
        self._histogram = None

    @property
    def retained_bytes(self) -> int:
        """All retained linear buffers; excludes temporary render allocations."""
        return (self.cache.retained_bytes + self._geometry.retained_bytes
                + (0 if self._small is None else self._small.nbytes)
                + (0 if self._adjusting_pixels is None else self._adjusting_pixels.nbytes))

    def _clear_derived(self):
        self._geometry.clear()
        self._adjusting_pixels = None
        self._adjusting_key = None

    def _display_metadata(self, photo, values):
        key = (photo.metadata["sha256"], photo.options, photo.half_size, values)
        if key != self._histogram_key:
            with timed(self.timer, "histogram"):
                options={} if not (any(g.active for g in values.local_gradients) or any(s.active for s in values.spots) or values.lens_corrections != LensCorrections()) else {
                    'source_size':(photo.linear_rgb.shape[1],photo.linear_rgb.shape[0]) if not photo.half_size else oriented_size(photo.metadata)}
                self._histogram = sampled_histogram(photo.linear_rgb, values,**options)
            self._histogram_key = key
        return dict(photo.metadata, display_histogram=self._histogram)

    def photo(self, path: Path, options: RawOptions):
        self._geometry.clear()
        photo = self.cache.get(path, options, half_size=True, before_decode=self._clear_derived)
        if self._adjusting_key and self._adjusting_key[:2] != (photo.metadata["sha256"], options):
            self._clear_derived()
        key = (photo.metadata["sha256"], options)
        if key != self._small_key:
            self._small = reduce_linear(photo.linear_rgb)
            if self._small is photo.linear_rgb:
                self._small = self._small.copy()
                self._small.setflags(write=False)
            self._small_key = key
        self.cache.trim_to(LINEAR_CACHE_BYTES - self._small.nbytes -
                          (0 if self._adjusting_pixels is None else self._adjusting_pixels.nbytes))
        return photo, self._small

    def load(self, path: Path):
        self.clear()
        photo, small = self.photo(path, RawOptions())
        values = EditValues(base_rendering="natural_v1")
        return replace(photo, metadata=self._display_metadata(photo, values)), quantize_srgb(render_edit(small, values))

    def render(self, state: ProbeState):
        options, values = display_render_spec(state)
        if values.detail.active:
            # A half-size decode changes the radius/noise represented by detail
            # controls. Restored projects must use the same native filters too.
            size = self._source_size(state)
            from .composition import output_size
            full = output_size(size, values.composition)
            request = DetailRequest(0, full, (0, 0, *full),
                                    max(1, int(np.ceil(max(full) / 1600))), size)
            return self.detail(state, request)
        photo, small = self.photo(state.source_path, options)
        if photo.metadata["sha256"] != state.photo_id:
            raise RuntimeError("The source file has changed. Reopen the photo.")
        started = time.perf_counter()
        pixels = self._render_small(photo, small, values, cancel=self.cancel, timer=self.timer)
        metadata = self._display_metadata(photo, values)
        if self.overlay_spec is not None:
            self._overlay(metadata,pixels,RenderGeometry.full(oriented_size(photo.metadata),values.composition,(pixels.shape[1],pixels.shape[0]),values.lens_corrections.distortion))
        return pixels, time.perf_counter() - started, metadata

    def backdrop(self, state: ProbeState):
        """Render a viewport-independent full-frame backdrop strictly from already-decoded RAM cache."""
        options, values = display_render_spec(state)
        photo = self.cache.peek_loaded(state.photo_id, options)
        if photo is None:
            return None
        small = (
            self._small
            if self._small is not None and self._small_key == (state.photo_id, options)
            else reduce_linear(photo.linear_rgb)
        )
        if values.detail.active:
            values = replace(values, detail=DetailControls())
        pixels = self._render_small(photo, small, values, cancel=self.cancel, timer=None)
        return pixels, output_size(oriented_size(photo.metadata), values.composition)

    def _source_size(self, state):
        options, _ = display_render_spec(state)
        warm = self.cache.full_identity == (state.photo_id, options)
        photo = self.cache.get(state.source_path, options, half_size=not warm, before_decode=self._clear_derived)
        return oriented_size(photo.metadata)

    def detail(self, state: ProbeState, request: DetailRequest, *, include_metadata=True):
        options, values = display_render_spec(state)
        started = time.perf_counter()
        size = request.source_size or request.full_size
        photo = self.cache.get(state.source_path, options, half_size=False,
                               expected_bytes=size[0] * size[1] * 12,
                               before_decode=self._clear_derived)
        if photo.metadata["sha256"] != state.photo_id:
            raise RuntimeError("The source file has changed. Reopen the photo.")
        small_bytes = 0 if self._small is None else self._small.nbytes
        temporary_bytes = 0 if self._adjusting_pixels is None else self._adjusting_pixels.nbytes
        if values.detail.active:
            region = None
        else:
            with timed(self.timer, "geometry"):
                region = self._geometry.region(
                    photo.linear_rgb, (state.photo_id, photo.options, photo.half_size),
                    values.composition, request.rect, distortion=values.lens_corrections.distortion,
                    max_bytes=LINEAR_CACHE_BYTES - small_bytes - photo.linear_rgb.nbytes - temporary_bytes,
                    reserve=lambda needed: self.cache.trim_to(LINEAR_CACHE_BYTES - small_bytes - needed - temporary_bytes))
        pixels = render_region(photo.linear_rgb, values, request, composed_region=region, cancel=self.cancel,
                               timer=self.timer)
        metadata = self._display_metadata(photo, values) if include_metadata else {}
        if self.overlay_spec is not None:
            self._overlay(metadata,pixels,RenderGeometry((photo.linear_rgb.shape[1],photo.linear_rgb.shape[0]),values.composition,request.rect,(pixels.shape[1],pixels.shape[0]),values.lens_corrections.distortion))
        return pixels, time.perf_counter() - started, metadata

    def adjusting(self, state, request):
        """Fast tonal approximation; native spatial filters before reduction."""
        started = time.perf_counter()
        options, values = display_render_spec(state)
        edge = ADJUSTING_MAX_EDGE if request.adjusting_edge is None else request.adjusting_edge
        if type(edge) is not int or not 160 <= edge <= 1280:
            raise ValueError("Invalid adjustment preview resolution.")
        if values.detail.active or request.native_detail:
            # Sharpening/NR operate in output-source pixels, never the reduced
            # preview grid. Coalescing and per-strip cancellation still apply.
            l, t, r, b = request.rect
            native_request = replace(request, step=max(request.step, int(np.ceil(max(r-l, b-t) / edge))))
            return self.detail(state, native_request, include_metadata=False)
        warm = self.cache.full_identity == (state.photo_id, options)
        photo = self.cache.get(state.source_path, options, half_size=not warm,
                               before_decode=self._clear_derived)
        if photo.metadata["sha256"] != state.photo_id:
            raise RuntimeError("The source file has changed. Reopen the photo.")
        key = (state.photo_id, options, photo.half_size, id(photo.linear_rgb),
               values.composition, request.full_size, request.rect, edge, values.lens_corrections.distortion)
        if self._adjusting_key != key:
            self._adjusting_pixels = None
            self._adjusting_key = None
            l, t, r, b = request.rect
            factor = min(1., edge / max(r-l, b-t))
            width, height = max(1, round((r-l)*factor)), max(1, round((b-t)*factor))
            needed = width * height * 12
            assert needed <= 32 * 1024 * 1024
            small = 0 if self._small is None else self._small.nbytes
            if photo.linear_rgb.nbytes + small + needed > LINEAR_CACHE_BYTES:
                raise MemoryError("Adjustment preview exceeds the linear cache budget. Release the control for the final preview.")
            if photo.linear_rgb.nbytes + small + self._geometry.retained_bytes + needed > LINEAR_CACHE_BYTES:
                self._geometry.clear()
            self.cache.trim_to(LINEAR_CACHE_BYTES - small - self._geometry.retained_bytes - needed)
            pixels = np.empty((height, width, 3), np.float32)
            # Sample directly into a bounded ROI; never form a full-size temporary.
            with timed(self.timer, "geometry"):
                for top in range(0, height, 128):
                    bottom = min(top+128, height)
                    rect = ((l, t+top, r, t+bottom) if factor == 1 else
                            (l, t+(b-t)*top/height, r, t+(b-t)*bottom/height))
                    pixels[top:bottom] = sample_composition(photo.linear_rgb, state.composition,
                        source_size=request.source_size, rect=rect,
                        target_size=(width, bottom-top), distortion=values.lens_corrections.distortion)
            pixels.setflags(write=False)
            self._adjusting_pixels, self._adjusting_key = pixels, key
        geometry=RenderGeometry(tuple(request.source_size or request.full_size),values.composition,
                                request.rect,(self._adjusting_pixels.shape[1],self._adjusting_pixels.shape[0]),values.lens_corrections.distortion)
        rendered = render_edit(self._adjusting_pixels, values, cancel=self.cancel, timer=self.timer,geometry=geometry,
                               source_linear=photo.linear_rgb)
        with timed(self.timer, "quantize"):
            pixels = quantize_srgb(rendered)
        metadata={}
        self._overlay(metadata,pixels,geometry)
        return pixels, time.perf_counter()-started, metadata

    def _overlay(self,metadata,pixels,geometry):
        if self.overlay_spec is not None:
            try:
                metadata["display_overlay"]=(overlay_pixels(pixels,geometry,self.overlay_spec),geometry.rect)
            except MemoryError as error:
                metadata["overlay_error"]=str(error)

    def load_project(self, path: Path, source_override: Path | None = None):
        store = read_project(path, source_override=source_override)
        self.clear()
        pixels, _, metadata = self.render(store.current)
        return store, pixels, metadata

    def coach_request(self, state, before, events, question, *, follow_up=None):
        images = []
        for snapshot in (before, state):
            photo, small = self.photo(snapshot.source_path, snapshot.raw_options)
            if photo.metadata["sha256"] != state.photo_id:
                raise RuntimeError("The source changed. Reopen it before analysis.")
            images.append(self._render_small(photo, small, snapshot.values, 768))
        return build_request(state, before, events, *images, question,
                             source_size=oriented_size(photo.metadata), follow_up=follow_up)

    def render_trial_values(self, state, values: EditValues, *, source_size=None, cancel=None) -> np.ndarray:
        """Render one immutable EditValues candidate at <=768px using the formal Detail path."""
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        from .composition import output_size
        snapshot = replace(state.with_values(values), compare_baseline=False, compare_snapshot=None)
        size = source_size or self._source_size(snapshot)
        full = output_size(size, values.composition)
        detail = DetailRequest(0, full, (0, 0, *full), max(1, int(np.ceil(max(full) / 768))), size)
        pixels = self.detail(snapshot, detail, include_metadata=False)[0]
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        return pixels

    def check_coach_trial(self, state, request, feedback, *, cancel=None):
        """Dry-run one proposal on the serial image worker; never mutate StateStore."""
        if not isinstance(feedback, CoachFeedback):
            raise ValueError("The coach returned invalid feedback.")
        if (state.photo_id, state.session_id, state.revision) != (
                request.photo_id, request.session_id, request.state_version):
            raise ValueError("Trial snapshot no longer matches its request.")
        return check_trial(
            request, feedback,
            lambda values: self.render_trial_values(state, values, source_size=request.source_size, cancel=cancel),
            cancel=cancel,
        )

    def inspect_region_100pct(self, state, x_norm: float, y_norm: float, size_px: int = 512,
                              *, apply_current_edits: bool = True, cancel=None) -> tuple[np.ndarray, dict]:
        """Render a bounded 1:1 native-resolution crop and compute objective sharpness/noise metrics."""
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        if not (0.0 <= float(x_norm) <= 1.0 and 0.0 <= float(y_norm) <= 1.0):
            raise ValueError("Crop coordinates must lie within [0, 1].")
        from .composition import output_size
        from .coach_agent import measure_preview
        snapshot = replace(state, compare_baseline=not apply_current_edits, compare_snapshot=None)
        size = self._source_size(snapshot)
        full = output_size(size, snapshot.composition)
        edge = max(128, min(512, int(size_px)))
        cw, ch = min(full[0], edge), min(full[1], edge)
        cx, cy = round(float(x_norm) * full[0]), round(float(y_norm) * full[1])
        left = max(0, min(full[0] - cw, cx - cw // 2))
        top = max(0, min(full[1] - ch, cy - ch // 2))
        rect = (left, top, left + cw, top + ch)
        detail = DetailRequest(0, full, rect, 1, size, native_detail=True)
        patch = np.array(self.detail(snapshot, detail, include_metadata=False)[0], copy=True)
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        patch.setflags(write=False)
        rgb = patch.astype(np.float32) / 255.0
        luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        gx = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
        gy = np.abs(np.diff(luma, axis=0, prepend=luma[:1, :]))
        edge_mag = gx + gy
        if luma.shape[0] >= 3 and luma.shape[1] >= 3:
            lap = (4.0 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1]
                   - luma[1:-1, :-2] - luma[1:-1, 2:])
            lap_energy = float(np.mean(np.abs(lap)))
        else:
            lap_energy = 0.0
        flat_mask = edge_mag <= np.percentile(edge_mag, 25)
        flat_noise = float(luma[flat_mask].std()) if np.any(flat_mask) else float(luma.std())
        whole_stats = measure_preview(patch)["whole"]
        metrics = {
            "rect_px": list(rect),
            "full_size": list(full),
            "patch_shape": [int(patch.shape[1]), int(patch.shape[0])],
            "edge_energy_mean": round(float(edge_mag.mean()), 5),
            "laplacian_sharpness": round(lap_energy, 5),
            "flat_patch_noise_std": round(flat_noise, 5),
            "mean_luminance": round(whole_stats["mean_luminance"], 5),
            "display_ceiling_fraction": round(whole_stats["display_ceiling_fraction"], 5),
            "display_floor_fraction": round(whole_stats["display_floor_fraction"], 5),
        }
        return patch, metrics

    def inspect_raw_headroom(self, state, x_norm: float = 0.5, y_norm: float = 0.5, patch_px: int = 64) -> dict:
        """Read-only Bayer CFA headroom inspection through white_balance.inspect_sensor_headroom."""
        return inspect_sensor_headroom(state, x_norm=x_norm, y_norm=y_norm, patch_px=patch_px)

    @staticmethod
    def _render_small(photo, small, values, max_edge=1600, *, cancel=None, timer=None):
        if values.composition == Composition() and not values.lens_corrections.distortion:
            with timed(timer, "reduction"):
                reduced = reduce_linear(small, max_edge)
            rendered = render_edit(reduced, values, cancel=cancel, timer=timer,
                geometry=RenderGeometry.full(oriented_size(photo.metadata),values.composition,(reduced.shape[1],reduced.shape[0])),
                source_linear=reduced)
        else:
            rendered = render_photo(small, values, source_size=oriented_size(photo.metadata), max_edge=max_edge,
                                    cancel=cancel, timer=timer)
        with timed(timer, "quantize"):
            return quantize_srgb(rendered)

    def composition_input(self, state):
        photo, small = self.photo(state.source_path, state.raw_options)
        if photo.metadata["sha256"] != state.photo_id:
            raise RuntimeError("The source has changed. Reopen the photo.")
        return reduce_linear(small, 1000).copy(), oriented_size(photo.metadata)


class MainWindow(QMainWindow):
    _agentToolRequested = Signal(object)

    def __init__(self, *, settings_store=None, credentials=None, dev_mode: bool | None = None) -> None:
        super().__init__()
        self._gui_thread_id = threading.get_ident()
        self._agentToolRequested.connect(self._enqueue_agent_tool)
        # The local mock coach is a development/testing tool, never a normal-user mode.
        self.dev_mode = dev_mode_enabled(dev_mode)
        self.setWindowTitle(f"PhotoCoach {__version__} · Single-photo editor")
        self.resize(1320, 900)
        self.store = ProbeStore()
        self.engine = PreviewEngine()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._active: tuple[Worker, Job] | None = None
        self._pending: Job | None = None
        self._pending_detail: Job | None = None
        self._pending_backdrop: Job | None = None
        self._protected_jobs: list[Job] = []
        self._generation = 0
        self._exporting = False
        self._closing = False
        self._metadata = {}
        self._metadata_binding = None
        self._render_diagnostics = {}
        self._interactive_preview = False
        self._wb_sampling = False
        self._wb_draft_error = False
        self._final_frame_binding = None
        self._formal_preview_session = None
        self._gesture_preview_edge = None
        self._native_detail_preview = False
        self._refining_preview = False
        self._coach_unread = False
        self._review_settings = None
        self._review_include_before = True
        self._review_max_turns = 1
        self._project_path: Path | None = None
        self._opening_project: Path | None = None
        self._close_approved = False
        self.coach = MockCoach()
        self.auto_schedule = AutoCoachSchedule()
        self._auto_observed = None
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(100)
        self._auto_timer.timeout.connect(self._poll_auto_coach)
        self.api_settings = None
        self._coach_model_catalogs = {}
        self._session_settings = {}
        self._session_only_ids = set()
        self.api_budget = SessionBudget()
        self.local_budget = SessionBudget(call_limit=100, input_usd_per_million=0, output_usd_per_million=0)
        self._network_budget = None
        self._local_dialog = None
        self.credentials = credentials if credentials is not None else WindowsCredentialStore()
        self.connection_store = None
        self.settings_store = settings_store
        self.model_catalog_store = None
        self.connection_book = None
        startup_notice = ""
        try:
            self.connection_store = ConnectionStore(settings_store.path if settings_store is not None else None)
            self.settings_store = self.connection_store
            self.model_catalog_store = ModelCatalogStore(self.connection_store.path.with_name('model_catalog.json'))
            self.connection_book = self.connection_store.load()
            if self.connection_book is not None and self.connection_book.active_connection is not None:
                active_conn = self.connection_book.active_connection
                self.api_settings = settings_for(active_conn, self.connection_book.budget_settings)
                self.api_budget.configure(**self.connection_book.budget_settings)
                backup_note = "Settings were recovered from backup (settings.json.bak). " if getattr(self.connection_store, "recovered_from_backup", False) else ""
                startup_notice = backup_note + (("Connection restored. Select AI coach when you want feedback. " if self.dev_mode
                                  else "Connection restored. Ask coach when you want feedback. ") + "Nothing is sent until you review and confirm. This is a new local session count; provider charges do not reset.")
        except SettingsError as error:
            startup_notice = str(error) + " Editing remains available; configure a connection when needed."
        from .local_runtime import LocalRuntime, LocalSettingsStore, LocalConfig, default_models_dir
        local_path = (self.connection_store.path.with_name("local_ai.json") if self.connection_store else
                      default_models_dir().parent / "local_ai.json")
        agent_base_dir = local_path.parent
        self.agent_memory = SessionWorkingMemory()
        self.style_memory = StyleMemoryStore(agent_base_dir / "agent_style_memory.json")
        self.playbook = PlaybookRAG()
        self.trace_store = TraceStore(agent_base_dir / "traces")
        self.local_runtime = LocalRuntime(LocalSettingsStore(local_path))
        try:
            self.local_config = self.local_runtime.store.load()
        except ValueError:
            self.local_config = LocalConfig()
            startup_notice += " Local AI settings need to be configured again."
        self.local_budget.configure(call_limit=self.local_config.call_limit)
        if self.api_settings is not None and self.api_settings.config.protocol == "ollama":
            try:
                synced_cfg = replace(
                    self.api_settings.config,
                    endpoint=self.local_config.endpoint,
                    model=self.local_config.model,
                )
                self.api_settings = replace(self.api_settings, config=synced_cfg)
            except Exception:
                pass
        self.network_pool = QThreadPool(self)
        self.network_pool.setMaxThreadCount(1)
        self._network_active = None
        self._network_progress = None
        self._network_timer = QTimer(self)
        self._network_timer.setInterval(250)
        self._network_timer.timeout.connect(self._update_network_progress)
        self._reviewing = False
        self._coach_generation = 0
        self._pending_follow_up = False
        self._expanded_coach_height = 280
        self._coach_cancel_requested = False
        self._highlighted_label = None
        self._highlight_timer = QTimer(self)
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.setInterval(3500)
        self._highlight_timer.timeout.connect(self._clear_control_highlight)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(16)
        self._render_timer.timeout.connect(lambda: self.request_preview(interactive=self._interactive_preview))
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(150)
        self._settle_timer.timeout.connect(self._finalize_preview)
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(120)
        self._detail_timer.timeout.connect(self.request_detail)
        self._notice_timer = QTimer(self)
        self._notice_timer.setSingleShot(True)
        self._notice_timer.setInterval(3000)
        from .i18n import set_global_language
        set_global_language('en')
        self._build_ui()
        self.style_memory_enabled.blockSignals(True)
        self.style_memory_enabled.setChecked(self.style_memory.enabled)
        self.style_memory_enabled.blockSignals(False)
        self.style_memory_enabled.toggled.connect(self.style_memory.set_enabled)
        self.clear_style_memory_button.clicked.connect(self.style_memory.clear)
        self._notice_timer.timeout.connect(self._refresh_view_notice)
        self.coach_collapse.setChecked(True)
        self.coach_text.set_question_draft(self.question.text())
        for slider in (self.exposure_slider, *self.adjustment_sliders.values(), *self.color_mixer.sliders,
                       *self.color_grading.sliders.values(),
                       *self.local_gradients.sliders.values(), self.local_gradients.feather, self.local_gradients.angle,
                       *self.spots_editor.sliders.values(),
                       *self.lens_editor.sliders.values(), *self.detail_editor.sliders.values()):
            slider.installEventFilter(self)
        if not self.dev_mode:
            # Normal launches always use the real AI coach; the mock selector is hidden.
            for selector in (self.coach_mode,):
                selector.blockSignals(True)
                selector.setCurrentIndex(1)
                selector.blockSignals(False)
        self._set_controls(False)
        if startup_notice:
            self.coach_text.setPlainText(startup_notice)
        for keys, callback in (("Ctrl+Z", self.undo), ("Ctrl+Shift+Z", self.redo),
                               ("Ctrl+Y", self.redo), ("Ctrl+S", self.save_project),
                               ("Ctrl+O", self.choose_raw)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(callback)

    def _build_ui(self) -> None:
        from .workspace_ui import build_workspace
        build_workspace(self)

    def _build_coach_panel(self) -> QWidget:
        from .workspace_ui import build_coach
        return build_coach(self)

    def _set_controls(self, enabled: bool) -> None:
        if not enabled:
            self.pick_wb.setChecked(False)
        for widget in (self.exposure_slider, self.base_rendering, self.wb_mode, self.apply_wb, self.compare, self.edited_button,
                       self.fit_button, self.actual_size_button, self.composition_button, self.crop_open_button,
                       self.export_button, self.report_button, self.save_project_button,
                       self.reset_button, self.curve, self.reset_curve_button, self.lens_editor, self.detail_editor, self.color_mixer, self.color_grading, self.local_gradients, self.spots_editor, self.snapshots_panel, self.clipping, self.goal, self.question, self.coach_light_menu_button,
                       *self.adjustment_sliders.values()):
            widget.setEnabled(enabled)
        self.pick_wb.setEnabled(enabled and not self._wb_sampling)
        self.advanced_wb.setEnabled(enabled)
        for slider in self.balance_sliders:
            slider.setEnabled(enabled)
        self.apply_wb.setEnabled(enabled and not self._wb_draft_error)
        for spin in self.gains:
            spin.setEnabled(enabled and self.wb_mode.currentIndex() == 1)
        self.undo_button.setEnabled(enabled and bool(self.store.undo_stack))
        self.redo_button.setEnabled(enabled and bool(self.store.redo_stack))
        self.coach_text.set_edit_values(self.store.current.values if (enabled and self.store.current) else None)
        self._sync_snapshots()
        self._update_crop_summary()
        self._update_coach_status()

    def _wb_form_enabled(self) -> None:
        for spin in self.gains:
            spin.setEnabled(self.wb_mode.isEnabled() and self.wb_mode.currentIndex() == 1)
            self.wb_form.setRowVisible(spin, self.advanced_wb.isChecked())
        if self.wb_mode.currentIndex() == 0:
            self._wb_draft_error = False
            self.apply_wb.setEnabled(self.wb_mode.isEnabled())

    def _wb_mode_changed(self):
        if self.wb_mode.currentIndex() == 0:
            for spin in self.gains:
                spin.blockSignals(True)
                spin.setValue(1.)
                spin.blockSignals(False)
            self._gain_draft_changed()
            self.wb_hint.setText("Camera WB draft · Apply to restore the camera balance.")
        self._wb_form_enabled()

    def _balance_labels(self):
        warmth, tint = (slider.value()/100 for slider in self.balance_sliders)
        self.balance_labels[0].setText(f"Warmth {warmth:+.2f}")
        self.balance_labels[1].setText(f"Tint {tint:+.2f}")

    def _balance_draft_changed(self):
        self._balance_labels()
        self.wb_mode.setCurrentIndex(1)
        try:
            gains = balance_gains(*(slider.value()/100 for slider in self.balance_sliders))
        except ValueError as error:
            self._wb_draft_error = True
            self.apply_wb.setEnabled(False)
            self.wb_hint.setText(str(error))
            return
        for spin, value in zip(self.gains, gains):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._wb_draft_error = False
        self.apply_wb.setEnabled(self.wb_mode.isEnabled())
        self.wb_hint.setText("Draft · Apply to process RAW. Warmer (+) / cooler (−); magenta (+) / green (−). Relative balance, not Kelvin.")

    def _gain_draft_changed(self):
        warmth, tint = balance_coordinates(tuple(spin.value() for spin in self.gains))
        for slider, value in zip(self.balance_sliders, (warmth, tint)):
            slider.blockSignals(True)
            slider.setValue(round(value*100))
            slider.blockSignals(False)
        self._balance_labels()
        self._wb_draft_error = False
        self.apply_wb.setEnabled(self.wb_mode.isEnabled())

    def _pick_wb_toggled(self, enabled):
        if enabled:
            self.local_gradients.edit_on_photo.setChecked(False)
            self.spots_editor.edit_on_photo.setChecked(False)
        self.preview.set_neutral_picker(enabled)
        self.pick_wb.setText("Cancel white balance picker" if enabled else "Pick neutral white balance")
        if enabled:
            self.preview.setFocus()
            self.status.setText("Click an evenly lit neutral gray/white patch. Avoid colored, dark or clipped areas. Esc cancels.")
        else:
            self.status.clear()

    def _sample_wb(self, position):
        self.pick_wb.setChecked(False)
        if self._wb_sampling or not self.store.current or self._exporting:
            return
        self._finish_gesture()
        state = self.store.current
        self._wb_sampling = True
        self.pick_wb.setEnabled(False)
        self.status.setText("Sampling original sensor data… Editing remains available.")
        self._queue(Job("wb_pick", self._generation, lambda: sample_white_balance(state, position), state))

    def choose_raw(self) -> None:
        if not self.open_button.isEnabled() or not self._allow_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open RAW photo (read-only)", "", "RAW photos (*.ARW *.arw *.NEF *.nef);;Sony ARW (*.ARW *.arw);;Nikon NEF (*.NEF *.nef)")
        if path:
            self.open_raw(Path(path))

    def open_raw(self, path: Path) -> None:
        self._formal_preview_session = None
        self.local_controller.layer.finish()
        self.local_gradients.edit_on_photo.setChecked(False)
        self.spots_controller.layer.finish()
        self.spots_editor.edit_on_photo.setChecked(False)
        self._cancel_obsolete_render()
        self._interactive_preview = False
        self._gesture_preview_edge = None
        self._native_detail_preview = False
        self._refining_preview = False
        self._settle_timer.stop()
        self.store.end_drag()
        self._opening_project = None
        self._pending_follow_up = False
        self._invalidate_coach("A new photo is opening.")
        self._generation += 1
        self._render_timer.stop()
        self._stop_detail()
        self._pending = None
        self._pending_backdrop = None
        self._notice_timer.stop()
        self.view_notice.clear()
        self._render_diagnostics = {}
        self.render_details.setText("Waiting for the new photo.")
        self._set_controls(False)
        self.open_button.setEnabled(False)
        self.open_project_button.setEnabled(False)
        self.preview.clear_photo()
        self.histogram.clear("Opening photo…")
        self.output_status.hide()
        self.filename.setText(path.name)
        self.metadata_label.setText("Reading and verifying the source file…")
        self.status.setText("Decoding RAW and reading camera white balance in the background…")
        self._queue(Job("load", self._generation, lambda: self.engine.load(path)))

    def _queue(self, job: Job) -> None:
        if job.kind in ("detail", "preview", "adjusting"):
            job.overlay=self.local_controller.overlay_spec()
            job.overlay_generation=self.preview.overlay_generation
            job.cancel = threading.Event()
            if job.kind in ("preview", "adjusting"):
                self._pending_backdrop = None
            self._cancel_obsolete_render()
        elif job.kind == "backdrop":
            if job.cancel is None:
                job.cancel = threading.Event()
        if self._active is not None:
            if job.kind in ("detail", "preview", "adjusting"):
                # All render intents share one latest slot, not a per-quality backlog.
                self._pending = None
                self._pending_detail = None
            if job.kind == "detail":
                self._pending_detail = job
            elif job.kind in ("preview", "adjusting"):
                self._pending = job
            elif job.kind == "backdrop":
                self._pending_backdrop = job
            else:
                self._protected_jobs.append(job)
            self._update_coach_status()
            return
        def run():
            self.engine.cancel = job.cancel
            self.engine.overlay_spec = job.overlay
            timer = StageTimer() if job.kind in TIMED_JOB_KINDS else None
            self.engine.timer = timer
            started = time.perf_counter()
            try:
                return job.function()
            finally:
                self.engine.cancel = None
                self.engine.overlay_spec = None
                self.engine.timer = None
                if timer is not None:
                    job.timing = {"queue_wait_ms": round((started - job.queued_at) * 1000, 1),
                                  "worker_wall_ms": round((time.perf_counter() - started) * 1000, 1),
                                  "stages_ms": timer.milliseconds()}
        worker = Worker(run)
        worker.signals.done.connect(self._done)
        self._active = (worker, job)
        self.pool.start(worker)
        self._update_coach_status()

    def _done(self, result, error) -> None:
        assert self._active is not None
        _, job = self._active
        self._active = None
        valid = job.generation == self._generation and not isinstance(error, CancelledError)
        if job.kind in TIMED_JOB_KINDS:
            valid=valid and job.overlay_generation==self.preview.overlay_generation
        if job.kind == "wb_pick":
            self._wb_sampling = False
            self.pick_wb.setEnabled(self.exposure_slider.isEnabled() and not self._exporting)
            # A save/export/crop preparation freezes a captured state. A pick
            # finishing meanwhile must not edit behind those disabled controls.
            valid = valid and not self._exporting
        if job.kind in ("preview", "detail", "adjusting", "backdrop", "coach", "coach_check", "api_prepare", "composition_prepare", "wb_pick"):
            valid = valid and self.store.accepts(job.snapshot)
        if job.kind in ("detail", "adjusting"):
            valid = valid and job.view_request.generation == self.preview.view_generation
        if job.kind == "adjusting":
            valid = (valid and self._interactive_preview and
                     self._final_frame_binding != self._frame_binding(job.snapshot, job.view_request))
        if job.kind == "preview":
            valid = valid and not self._refining_preview
        if job.kind == "api_prepare":
            valid = valid and job.coach_token == self._coach_generation and not self.coach.paused
        if job.kind in ("coach", "coach_check"):
            valid = valid and job.coach_token == self._coach_generation and not self.coach.paused
            if job.automatic:
                valid = valid and self.auto_mock.isChecked() and self.coach_mode.currentIndex() == 0
            if error is None and not isinstance(result, CoachFeedback):
                error = ValueError("The coach returned invalid feedback.")
            if error is None:
                valid = valid and (result.photo_id, result.session_id, result.state_version) == (
                    job.snapshot.photo_id, job.snapshot.session_id, job.snapshot.revision)
        if valid and not self._closing:
            if job.kind == "agent_tool":
                pass
            elif error is not None:
                self.status.setText(f"Operation failed: {error}")
                if job.kind in ("load", "project_load", "preview"):
                    self.histogram.clear("Unavailable")
                if job.kind == "detail":
                    self.preview.set_quality("Detail unavailable")
                    self.detail_label.setToolTip(str(error))
                    self.status.setText(f"Detail unavailable: {error} Editing remains available.")
                if job.kind == "adjusting":
                    self.preview.set_quality("Adjustment preview unavailable")
                    self.detail_label.setToolTip(str(error))
                if job.kind in ("export", "diagnostics", "project_save", "composition_prepare"):
                    self.output_status.setText(f"Output failed: {error}")
                    self.output_status.show()
                if job.kind in ("load", "project_load"):
                    self.coach_text.setPlainText("Open failed. You can retry or continue with the previous photo.", kind="error")
                    self.open_button.setEnabled(True)
                    self.open_project_button.setEnabled(True)
                    self._set_controls(self.store.current is not None)
                    if self.store.current:
                        self.filename.setText(self.store.current.source_path.name)
                        state = self.store.current
                        self._pending = self._resume_preview_job(state)
                    else:
                        self.preview.setText("Unable to open this file. Choose another photo or project.")
                        self.preview.set_quality("No photo")
                    self.output_status.setText(f"Open failed: {error}")
                    self.output_status.show()
                    if isinstance(error, SourceMissing):
                        QTimer.singleShot(0, self.locate_project_source)
                elif job.kind in ("coach", "coach_check"):
                    if job.automatic:
                        self.auto_mock.setChecked(False)
                    message = (f"Mock analysis failed: {error}" if job.kind == "coach" else
                               "Coach workflow failed. No edit was applied; analyze again.")
                    self.coach_text.setPlainText(message + "\nEditing remains available.", kind="error")
                elif job.kind == "api_prepare":
                    self.coach_text.setPlainText("Unable to prepare matching previews. Render the current edit and try again.", kind="error")
            elif job.kind == "load":
                photo, pixels = result
                state = self.store.activate(photo.source_path, photo.metadata["photo_id"], base_rendering="natural_v1")
                self.agent_memory.clear()
                self._pending_follow_up = False
                self._project_path = None
                self.snapshots_panel.clear_name_input()
                self.snapshots_panel.set_feedback("")
                for widget in (self.exposure_slider, self.compare, self.wb_mode):
                    widget.blockSignals(True)
                self.exposure_slider.setValue(0)
                self.compare.setChecked(False)
                self.wb_mode.setCurrentIndex(0)
                for spin in self.gains:
                    spin.setValue(1)
                for widget in (self.exposure_slider, self.compare, self.wb_mode):
                    widget.blockSignals(False)
                self.exposure_label.setText("Exposure  +0.00 EV")
                self.store.mark_preview(state)
                self.preview.configure_photo(oriented_size(photo.metadata))
                self.preview.display(pixels)
                self.coach_text.set_preview_pixels(pixels)
                self._show_metadata(photo.metadata)
                self._set_controls(True)
                self.open_button.setEnabled(True)
                self.open_project_button.setEnabled(True)
                self._sync_controls()
                self.coach_text.setPlainText("Photo ready. Set a goal or ask about your edit.")
                self._auto_observed = self._auto_edit_key()
                self.status.clear()
                if self.local_controller.overlay_spec() is not None:self.preview.quick_current=False
                self._view_changed()
            elif job.kind == "preview":
                pixels, elapsed, metadata = result
                presented = time.perf_counter()
                self.store.mark_preview(job.snapshot)
                self.preview.display(pixels)
                self.coach_text.set_preview_pixels(pixels)
                self.preview.accept_overlay(metadata.pop('display_overlay',None))
                self._final_frame_binding = self._frame_binding(job.snapshot)
                self.preview.set_quality("Quick preview")
                self._show_metadata(metadata)
                self._record_render_details(job, elapsed, presentation=time.perf_counter() - presented)
                self.status.clear()
                self._view_changed()
            elif job.kind == "detail":
                pixels, elapsed, metadata = result
                presented = time.perf_counter()
                self.preview.display_detail(pixels, job.view_request)
                self._refining_preview = False
                self.preview.accept_overlay(metadata.pop('display_overlay',None))
                self._final_frame_binding = self._frame_binding(job.snapshot, job.view_request)
                self._formal_preview_session = (job.snapshot.photo_id, job.snapshot.session_id)
                if self.store.preview_revision != job.snapshot.revision:
                    self.store.mark_preview(job.snapshot)
                self._show_metadata(metadata)
                self._record_render_details(job, elapsed, presentation=time.perf_counter() - presented)
                self.status.clear()
                options, _ = display_render_spec(job.snapshot)
                if (
                    not self.preview.has_current_full_frame()
                    and not self._interactive_preview
                    and not self._closing
                    and not self._exporting
                    and self.engine.cache.peek_loaded(job.snapshot.photo_id, options) is not None
                ):
                    snap = job.snapshot
                    self._pending_backdrop = Job(
                        "backdrop",
                        self._generation,
                        lambda s=snap: self.engine.backdrop(s),
                        snap,
                    )
            elif job.kind == "backdrop":
                if result is not None:
                    pixels, full_size = result
                    self.preview.display_backdrop(pixels, full_size)
            elif job.kind == "adjusting":
                pixels, elapsed, metadata = result
                presented = time.perf_counter()
                self.preview.display_adjusting(pixels, job.view_request)
                self.preview.accept_overlay(metadata.pop('display_overlay',None))
                self._record_render_details(job, elapsed, presentation=time.perf_counter() - presented)
            elif job.kind == "project_load":
                self.store, pixels, metadata = result
                self.agent_memory.clear()
                self._pending_follow_up = False
                self._project_path = self._opening_project
                self.snapshots_panel.clear_name_input()
                self.snapshots_panel.set_feedback("")
                self.store.mark_preview(self.store.current)
                self.filename.setText(self.store.current.source_path.name)
                self.preview.configure_photo(oriented_size(metadata))
                self.preview.set_composition(self.store.current.composition, distortion=self.store.current.lens_corrections.distortion)
                self.preview.display(pixels)
                self.coach_text.set_preview_pixels(pixels)
                self._show_metadata(metadata)
                self._sync_controls()
                self.open_button.setEnabled(True)
                self.open_project_button.setEnabled(True)
                self._set_controls(True)
                self._refresh_view_notice()
                self.status.setText("Project restored, including edit history, snapshots, and undo/redo.")
                self.coach_text.setPlainText("Project ready. Ask about the restored edit.")
                self._auto_observed = self._auto_edit_key()
                if self.local_controller.overlay_spec() is not None:self.preview.quick_current=False
                self._view_changed()
                if self.store.load_warnings:
                    self.output_status.setText(" ".join(self.store.load_warnings))
                    self.output_status.show()
            elif job.kind == "wb_pick":
                self._apply_white_balance(result)
                self.wb_hint.setText("Sample applied · One undo step. Assumes a neutral patch; compare with Camera WB if colors look wrong.")
            elif job.kind == "composition_prepare":
                self.status.clear()
                self.output_status.hide()
                QTimer.singleShot(0, lambda: self._edit_composition(result, job.snapshot))
            elif job.kind == "project_save":
                self._project_path = result
                self.store.mark_saved(job.snapshot.revision)
                self.output_status.setText(f"Project saved: {result}")
                self.output_status.show()
                self.status.clear()
            elif job.kind in ("coach", "coach_check"):
                self._record_coach_advice(result, job.snapshot.values, spans=getattr(result, "agent_trace", ()))
                self.coach_text.show_feedback(result)
            elif job.kind == "api_prepare":
                # Finish the serial render queue before entering a modal review.
                QTimer.singleShot(0, lambda result=result, job=job: self.review_api_request(result, job))
            elif job.kind == "export":
                self.output_status.setText(f"Export saved: {result}")
                self.status.clear()
                self.output_status.show()
            elif job.kind == "diagnostics":
                self.output_status.setText(f"Diagnostics saved: {result} (not a reopenable project)")
                self.status.clear()
                self.output_status.show()
            if job.kind in ("export", "diagnostics", "project_save", "composition_prepare"):
                self._exporting = False
                self.open_button.setEnabled(True)
                self.open_project_button.setEnabled(True)
                self._set_controls(self.store.current is not None)
                if self.store.current and self.store.preview_revision != self.store.current.revision:
                    state = self.store.current
                    self._pending = self._resume_preview_job(state)
                else:
                    self._view_changed()
            self._show_version()
            if job.kind in TIMED_JOB_KINDS and error is None and metadata.get('overlay_error'):
                self.view_notice.setText(metadata['overlay_error'])
            elif job.kind in TIMED_JOB_KINDS and error is None and not self._notice_timer.isActive():
                self._refresh_view_notice()
        if job.kind in ("coach", "coach_check", "api_prepare"):
            if not valid:
                self.coach.last_request = None
            self._update_coach_status()
            if self.store.current and not (self._pending or self._pending_detail) and self.store.preview_revision != self.store.current.revision and not self._closing:
                state = self.store.current
                self._pending = self._resume_preview_job(state)
        while not self._closing:
            if self._protected_jobs:
                pending = self._protected_jobs.pop(0)
            elif self._pending:
                pending, self._pending = self._pending, None
            elif self._pending_detail:
                pending, self._pending_detail = self._pending_detail, None
            else:
                pending, self._pending_backdrop = self._pending_backdrop, None
            if pending is None:
                break
            if pending.generation != self._generation:
                continue
            if pending.kind in ("preview", "detail", "adjusting", "backdrop", "coach", "coach_check", "api_prepare", "composition_prepare") and not self.store.accepts(pending.snapshot):
                continue
            if pending.kind in ("coach", "coach_check", "api_prepare") and pending.coach_token != self._coach_generation:
                continue
            if pending.kind in ("detail", "adjusting") and pending.view_request.generation != self.preview.view_generation:
                continue
            if pending.kind == "backdrop" and self.preview.has_current_full_frame():
                continue
            self._queue(pending)
            break
        self._update_coach_status()
        if self._closing:
            QTimer.singleShot(0, self.close)

    def _show_metadata(self, metadata: dict) -> None:
        self._metadata = metadata
        state = self.store.current
        snap_name = state.compare_snapshot.name if (state and state.compare_snapshot is not None) else None
        self.histogram.display(
            metadata.get("display_histogram"),
            baseline=bool(state and state.compare_baseline),
            snapshot_name=snap_name,
        )
        self._metadata_binding = None if state is None else {
            "photo_id": state.photo_id, "session_id": state.session_id,
            "preview_revision": self.store.preview_revision,
            "is_baseline": state.compare_baseline,
            "compare_snapshot_id": state.compare_snapshot_id,
        }
        self.metadata_label.setText(
            f"{metadata.get('make', '—')} {metadata.get('model', '—')}\n"
            f"RAW visible area {metadata['visible_size'][0]} × {metadata['visible_size'][1]}\n"
            f"Orientation code {metadata.get('orientation_flip', '—')}\n"
            f"ISO {metadata.get('iso', '—')}  ·  f/{metadata.get('aperture', '—')}\n"
            f"Shutter {metadata.get('exposure_seconds', '—')} s  ·  {metadata.get('focal_length_mm', '—')} mm\n"
            f"{metadata.get('lens', '')}"
        )

    def _show_version(self) -> None:
        if self.store.current:
            self.version_label.setText(f"State v{self.store.current.revision} / Preview v{self.store.preview_revision}")
        self._refresh_history()

    def _record_render_details(self, job, elapsed, presentation=None):
        snap_name = job.snapshot.compare_snapshot.name if job.snapshot.compare_snapshot is not None else None
        self._render_diagnostics = {"kind": job.kind, "revision": job.snapshot.revision,
            "elapsed_ms": round(elapsed * 1000, 1), "baseline": job.snapshot.compare_baseline,
            "compare_snapshot_id": job.snapshot.compare_snapshot_id,
            "view_generation": self.preview.view_generation, "zoom": self.preview.zoom,
            "dpr": self.preview.devicePixelRatioF()}
        # Detailed timing stays in diagnostics; stage totals are summed worker
        # time across strips and can exceed elapsed wall time.
        if job.timing:
            self._render_diagnostics.update(job.timing)
        if presentation is not None:
            self._render_diagnostics["presentation_ms"] = round(presentation * 1000, 1)
        mode_label = (
            "Baseline"
            if job.snapshot.compare_baseline
            else f"Snapshot ({snap_name})" if snap_name else "Edited"
        )
        self.render_details.setText(
            f"{'Full detail' if job.kind == 'detail' else 'Quick preview'} · v{job.snapshot.revision}\n"
            f"Worker time: {elapsed * 1000:.0f} ms\n"
            f"{mode_label} · "
            f"{self.preview.zoom * 100:.0f}% · DPR {self.preview.devicePixelRatioF():g}")

    def _show_preview_details(self):
        self.preview_details.show()
        self.preview_details.raise_()

    def _sync_comparison_buttons(self):
        baseline = bool(self.store.current and self.store.current.compare_baseline)
        comparing_snapshot = bool(self.store.current and self.store.current.compare_snapshot is not None)
        self.compare.blockSignals(True)
        self.compare.setChecked(baseline)
        self.compare.blockSignals(False)
        self.edited_button.setChecked(not baseline and not comparing_snapshot)

    def _sync_snapshots(self, *, selected_id: str | None = None) -> None:
        if not hasattr(self, "snapshots_panel"):
            return
        state = self.store.current
        self.snapshots_panel.set_snapshots(
            self.store.snapshots if state else (),
            compare_snapshot_id=state.compare_snapshot_id if state else None,
            selected_id=selected_id,
        )

    def _refresh_view_notice(self) -> None:
        state = self.store.current
        if state and state.compare_snapshot is not None:
            from .i18n import get_language
            zh = get_language() == "zh"
            snap = state.compare_snapshot
            geom_aligned = (
                snap.values.composition != state.composition
                or snap.values.lens_corrections.distortion != state.lens_corrections.distortion
            )
            if zh:
                text = f"A/B 对比：{snap.name}" + ("（已对齐当前构图）" if geom_aligned else "")
            else:
                text = f"Comparing A/B: {snap.name}" + (" (aligned crop)" if geom_aligned else "")
            self.view_notice.setText(text)
            return
        self.local_controller.show_notice()
        if not self.view_notice.text() and self.spots_editor.edit_on_photo.isChecked():
            from .i18n import tr
            self.view_notice.setText(tr("Spot removal overlay · display only", "污点修复标记 · 仅用于显示"))

    def _ensure_edited_view(self):
        state = self.store.current
        if state and (state.compare_baseline or state.compare_snapshot is not None):
            # Do this before begin_drag: set_compare / set_compare_snapshot ends an existing transaction.
            if state.compare_baseline:
                self.store.set_compare(False)
            if self.store.current and self.store.current.compare_snapshot is not None:
                self.store.set_compare_snapshot(None)
            self._sync_comparison_buttons()
            self._sync_snapshots()
            from .i18n import tr
            self.view_notice.setText(tr("Showing edited photo", "当前显示编辑效果"))
            self._notice_timer.start()

    def _begin_parameter_gesture(self, name):
        if self.store.current and not self._exporting:
            self._ensure_edited_view()
            self.store.begin_drag(name)
            self._gesture_preview_edge = ADJUSTING_MAX_EDGE
            self._native_detail_preview = name == "detail" or self.store.current.detail.active
            # Even a press without a value change has switched the display mode.
            self._after_edit(interactive=True)

    def change_exposure(self, value: int) -> None:
        if not self.store.current or self._exporting:
            return
        self._ensure_edited_view()
        self.store.set_exposure(value / 100.0)
        self.exposure_label.setText(f"Exposure  {value / 100:+.2f} EV")
        self._after_edit(interactive=True)

    def change_wb(self) -> None:
        if not self.store.current or self._exporting:
            return
        options = RawOptions("custom" if self.wb_mode.currentIndex() else "camera",
                             tuple(spin.value() for spin in self.gains))
        # The picker stores full precision. Reapplying untouched rounded fields
        # must not introduce a tiny extra WB edit or throw away that precision.
        applied = self.store.current.raw_options
        if options.wb_mode == applied.wb_mode and all(
                spin.value() == round(value, spin.decimals()) for spin, value in zip(self.gains, applied.wb_gains)):
            options = applied
        self._apply_white_balance(options)

    def _apply_white_balance(self, options):
        self._ensure_edited_view()
        self.store.set_whitebalance(options)
        self._sync_controls()
        self._after_edit()
        self.status.setText("Applying white balance in the background…")

    def change_compare(self, checked: bool) -> None:
        if self.store.current:
            self._finish_gesture()
            self._interactive_preview = False
            self._gesture_preview_edge = None
            self._native_detail_preview = False
            self._settle_timer.stop()
            self._pending_backdrop = None
            if not checked and self.store.current.compare_snapshot is not None:
                self.store.set_compare_snapshot(None)
            self.store.set_compare(checked)
            self._sync_comparison_buttons()
            self._sync_snapshots()
            self._refresh_view_notice()
            self._invalidate_coach(keep_reference=True)
            self.request_preview()

    def save_named_snapshot(self, name: str) -> None:
        if not self.store.current or self._exporting or not self.snapshots_panel.isEnabled():
            return
        self._finish_gesture()
        self.change_goal()
        try:
            snapshot = self.store.save_snapshot(name)
        except ValueError as error:
            self.snapshots_panel.set_feedback(str(error), error=True)
            self.status.setText(str(error))
            return
        self.snapshots_panel.clear_name_input()
        self.snapshots_panel.set_feedback("")
        self._sync_snapshots(selected_id=snapshot.id)
        self._refresh_history()
        from .i18n import get_language
        zh = get_language() == "zh"
        self.status.setText(
            f"已保存快照：{snapshot.name}" if zh else f"Saved snapshot: {snapshot.name}"
        )

    def delete_named_snapshot(self, snapshot_id: str) -> None:
        if not self.store.current or self._exporting or not self.snapshots_panel.isEnabled():
            return
        self._finish_gesture()
        was_comparing = self.store.current.compare_snapshot_id == snapshot_id
        try:
            removed = self.store.delete_snapshot(snapshot_id)
        except ValueError as error:
            self.snapshots_panel.set_feedback(str(error), error=True)
            return
        self.snapshots_panel.set_feedback("")
        self._sync_comparison_buttons()
        self._sync_snapshots()
        self._refresh_view_notice()
        self._refresh_history()
        if was_comparing:
            self._invalidate_coach(keep_reference=True)
            self.request_preview()
        from .i18n import get_language
        zh = get_language() == "zh"
        self.status.setText(
            f"已删除快照：{removed.name}" if zh else f"Deleted snapshot: {removed.name}"
        )

    def restore_named_snapshot(self, snapshot_id: str) -> None:
        if not self.store.current or self._exporting or not self.snapshots_panel.isEnabled():
            return
        self._finish_gesture()
        was_comparing = (
            self.store.current.compare_baseline
            or self.store.current.compare_snapshot is not None
        )
        try:
            snap = self.store.get_snapshot(snapshot_id)
            changed = self.store.restore_snapshot(snapshot_id)
        except ValueError as error:
            self.snapshots_panel.set_feedback(str(error), error=True)
            return
        self.snapshots_panel.set_feedback("")
        self._sync_controls()
        self._refresh_view_notice()
        if changed or was_comparing:
            self._after_edit()
        from .i18n import get_language
        zh = get_language() == "zh"
        self.status.setText(
            (f"已恢复快照：{snap.name}（Ctrl+Z 可撤销）" if zh else f"Restored snapshot: {snap.name} · Undo (Ctrl+Z) reverts")
            if changed
            else (f"当前已与快照“{snap.name}”一致" if zh else f"Current edits already match snapshot: {snap.name}")
        )

    def change_compare_snapshot(self, snapshot_id: str | None) -> None:
        if not self.store.current or self._exporting or not self.snapshots_panel.isEnabled():
            return
        self._finish_gesture()
        self._interactive_preview = False
        self._gesture_preview_edge = None
        self._native_detail_preview = False
        self._settle_timer.stop()
        self._pending_backdrop = None
        try:
            self.store.set_compare_snapshot(snapshot_id)
        except ValueError as error:
            self.snapshots_panel.set_feedback(str(error), error=True)
            self._sync_snapshots()
            return
        self.snapshots_panel.set_feedback("")
        self._sync_comparison_buttons()
        self._sync_snapshots(selected_id=snapshot_id)
        self._refresh_view_notice()
        self._invalidate_coach(keep_reference=True)
        self.request_preview()
        from .i18n import get_language
        zh = get_language() == "zh"
        if self.store.current.compare_snapshot is not None:
            snap_name = self.store.current.compare_snapshot.name
            self.status.setText(f"A/B 对比：{snap_name}" if zh else f"Comparing A/B: {snap_name}")
        else:
            self.status.setText("已切回编辑效果" if zh else "Returned to edited view")

    def request_preview(self, *, interactive=False) -> None:
        if self.store.current and not self._exporting and not self._closing:
            if not interactive:
                self._interactive_preview = False
                self._settle_timer.stop()
                self._refining_preview |= self._has_formal_preview(self.store.current)
            self._stop_detail()
            self.preview.set_composition(self.store.current.composition, distortion=self.store.current.lens_corrections.distortion)
            self.preview.invalidate_edit("Adjusting" if interactive else "Refining")
            self.histogram.mark_updating()
            state = self.store.current
            self._show_version()
            if interactive:
                if self._gesture_preview_edge is None:
                    self._gesture_preview_edge = ADJUSTING_MAX_EDGE
                request = replace(self.preview.detail_request(), adjusting_edge=self._gesture_preview_edge,
                                  native_detail=self._native_detail_preview)
                self.preview.set_quality("Adjusting")
                self._queue(Job("adjusting", self._generation, lambda: self.engine.adjusting(state, request),
                                state, view_request=request))
                return
            options, _ = display_render_spec(state)
            if (self._refining_preview or (self.preview.needs_detail() and
                    self.engine.cache.full_identity == (state.photo_id, options))):
                # Refine gestures through the formal path even at Fit. A cold
                # full decode is paid once, not replaced with a false detail label.
                request = self.preview.detail_request()
                self._queue(Job("detail", self._generation, lambda: self.engine.detail(state, request),
                                state, view_request=request))
            else:
                self._queue(Job("preview", self._generation, lambda: self.engine.render(state), state))

    def _has_formal_preview(self, state):
        """Keep settled quality for this photo session across passive refreshes."""
        return (state is not None and
                self._formal_preview_session == (state.photo_id, state.session_id))

    def _resume_preview_job(self, state):
        """Resume the intended final quality after output/coach work drops a render."""
        self._refining_preview |= self._has_formal_preview(state)
        if self._refining_preview:
            request = self.preview.detail_request()
            return Job("detail", self._generation, lambda: self.engine.detail(state, request),
                       state, view_request=request)
        return Job("preview", self._generation, lambda: self.engine.render(state), state)

    def _view_status(self) -> None:
        self.zoom_label.setText(self.preview.zoom_text)
        quality = self.preview.quality
        self.detail_label.setText("Loading detail…" if quality == "Loading detail" else quality)
        self.detail_busy.setVisible(quality in ("Loading detail", "Refining", "Opening photo…"))
        self.fit_button.setChecked(self.preview.fit_mode)
        self.actual_size_button.setChecked(not self.preview.fit_mode and abs(self.preview.zoom - 1) < 1e-9)
        self.retry_detail_button.setVisible(self.preview.quality == "Detail unavailable")
        if self.preview.quality != "Detail unavailable":
            self.detail_label.setToolTip("")

    def _stop_detail(self) -> None:
        self._detail_timer.stop()
        self._pending_detail = None

    def _view_changed(self) -> None:
        self._cancel_obsolete_render(view_only=True)
        self._stop_detail()
        if self._interactive_preview:
            self.preview.set_quality("Adjusting")
            self._render_timer.start()
            return
        if (self.store.current and not self._closing and not self._exporting
                and self.exposure_slider.isEnabled()):
            self._refining_preview |= self._has_formal_preview(self.store.current)
            if self.preview.needs_detail() or self._refining_preview:
                self.preview.set_quality("Refining" if self._refining_preview else "Loading detail")
                self._detail_timer.start()
            elif self.preview.has_quick and not self.preview.quick_current:
                self.request_preview()

    def request_detail(self) -> None:
        if (not self.store.current or self._closing or self._exporting or
                not self.exposure_slider.isEnabled() or not (self.preview.needs_detail() or self._refining_preview)):
            return
        # A quick render already queued for the current edit takes precedence.
        state = self.store.current
        options, _ = display_render_spec(state)
        if (not self._refining_preview and self.store.preview_revision != state.revision and
                self.engine.cache.full_identity != (state.photo_id, options)):
            return
        state, request = self.store.current, self.preview.detail_request()
        self.preview.set_quality("Refining" if self._refining_preview else "Loading detail")
        self._queue(Job("detail", self._generation, lambda: self.engine.detail(state, request),
                        state, view_request=request))

    def choose_export(self) -> None:
        if self._exporting or not self.export_button.isEnabled():
            return
        self._finish_gesture()
        state = self.store.current
        if state is None:
            return
        path, selected = QFileDialog.getSaveFileName(self, "Full-resolution export: choose a new filename", "", "JPEG (*.jpg);;16-bit RGB TIFF (*.tif)")
        if not path:
            return
        destination = Path(path)
        if not destination.suffix:
            destination = destination.with_suffix(".tif" if selected.startswith("16") else ".jpg")
        try:
            validate_destination(destination, forbidden_paths=(state.source_path,))
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Output unavailable", str(error))
            return
        self._freeze_output("Exporting full resolution using the current parameters. Baseline / A/B view does not change the export.")

        def export():
            self.engine.clear()
            photo = decode_raw(state.source_path, state.raw_options, half_size=False)
            if photo.metadata["sha256"] != state.photo_id:
                raise RuntimeError("The source file has changed. Reopen the photo.")
            rendered = render_photo(photo.linear_rgb, state.values)
            if sha256_file(state.source_path) != state.photo_id:
                raise RuntimeError("The source file changed during export. Output was not written.")
            return export_image(destination, rendered, source_path=state.source_path)

        self._queue(Job("export", self._generation, export, state))

    def _freeze_output(self, message: str) -> None:
        self._render_timer.stop()
        self._stop_detail()
        self._pending = None
        self._exporting = True
        self._set_controls(False)
        self.open_button.setEnabled(False)
        self.open_project_button.setEnabled(False)
        self.output_status.setText(message)
        self.output_status.show()
        self.status.setText(message)

    def save_diagnostics(self) -> None:
        self._finish_gesture()
        state = self.store.current
        if state is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save local diagnostics (not a project)", "", "JSON (*.json)")
        if not path:
            return
        destination = Path(path)
        if not destination.suffix:
            destination = destination.with_suffix(".json")
        data = {"kind": "phase0_diagnostics_not_a_project", "photo_id": state.photo_id,
                "source_name": state.source_path.name, "revision": state.revision,
                "exposure_ev": state.exposure_ev, "wb_mode": state.raw_options.wb_mode,
                "wb_gains": state.raw_options.wb_gains,
                "parameters": state.values.to_dict(),
                "last_displayed_metadata": self._metadata,
                "last_displayed_metadata_binding": self._metadata_binding,
                "preview_render": self._render_diagnostics,
                "events": list(self.store.events), "api_attempts": self.api_budget.attempts,
                "api_calls": self.api_budget.attempts}
        self._freeze_output("Saving local session diagnostics…")
        self._queue(Job("diagnostics", self._generation, lambda: atomic_write(
            destination, lambda temporary: temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"),
            forbidden_paths=(state.source_path,)), state))

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.FocusOut and isinstance(watched, QSlider) and watched.isSliderDown():
            watched.setSliderDown(False)
            self._finish_gesture()
        return super().eventFilter(watched, event)

    def _finish_gesture(self) -> None:
        self.local_controller.layer.finish()
        self.spots_controller.layer.finish()
        was_dragging = self.store.gesture_active
        self.curve.finish()
        self.store.end_drag()
        self._refresh_history()
        self._sync_coach_presentation()
        self.auto_schedule.released()
        if was_dragging:
            self._finalize_preview()

    def _frame_binding(self, state, request=None):
        return (state.photo_id, state.session_id, state.revision,
                request.generation if request is not None else self.preview.view_generation,self.preview.overlay_generation)

    def _finalize_preview(self):
        if self.store.gesture_active:
            return
        was_interactive = self._interactive_preview
        self._interactive_preview = False
        self._gesture_preview_edge = None
        self._native_detail_preview = False
        self._settle_timer.stop()
        if was_interactive:
            self._refining_preview = True
            self._render_timer.stop()
            self.request_preview()

    def _invalidate_coach(self, message="Edits changed. Analyze again for feedback on the current state.", *, keep_reference=False, clear_auto=True, coach_changed=False) -> None:
        if clear_auto:
            self._cancel_auto_coach()
        self._coach_generation += 1
        if self._network_active is not None:
            self._network_active[2].set()
        if self._active and self._active[1].kind in ("coach_check", "agent_tool") and self._active[1].cancel is not None:
            self._active[1].cancel.set()
        state = self.store.current
        retained = (keep_reference and state is not None and
                    self.coach_text.keep_as_reference(state.photo_id, state.session_id, state.revision, coach_changed=coach_changed))
        if not retained:
            self.coach_text.setPlainText(message)
            self._clear_control_highlight()
        self.coach.last_request = None

    def _after_edit(self, *, interactive=False) -> None:
        self.pick_wb.setChecked(False)
        self._cancel_obsolete_render()
        self._interactive_preview = interactive
        # Discrete edits (Undo/Redo/reset/WB) need the same formal settling
        # path as a released drag, including when the viewer is in Fit.
        self._refining_preview = not interactive
        if interactive:
            if self._gesture_preview_edge is None:
                self._gesture_preview_edge = ADJUSTING_MAX_EDGE
        else:
            self._native_detail_preview = False
        self._settle_timer.stop()
        if interactive:
            self._settle_timer.start()
        self._stop_detail()
        if self.store.current:
            self.preview.set_composition(self.store.current.composition, distortion=self.store.current.lens_corrections.distortion)
            self.local_controller.sync()
            self.spots_controller.sync()
        self.preview.invalidate_edit("Adjusting" if interactive else "Refining")
        self.histogram.mark_updating()
        self._invalidate_coach(keep_reference=True, clear_auto=False)
        self._sync_coach_presentation()
        self._auto_note_edit()
        self._show_version()
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _cancel_obsolete_render(self, *, view_only: bool = False):
        if not view_only:
            self._pending_backdrop = None
        if self._active is not None:
            _, active = self._active
            if active.kind == "backdrop" and view_only:
                return
            if active.cancel is not None and active.kind not in ("coach_check", "agent_tool"):
                active.cancel.set()

    def open_composition(self):
        if not self.store.current or self._exporting or not self.composition_button.isEnabled():
            return
        self._finish_gesture()
        state = self.store.current
        self._freeze_output("Preparing crop preview…")
        self._queue(Job("composition_prepare", self._generation,
                        lambda: self.engine.composition_input(state), state))

    def _edit_composition(self, result, snapshot):
        if self._closing or not self.store.accepts(snapshot):
            return
        dialog = CompositionDialog(self, *result, snapshot.values)
        if dialog.exec() == QDialog.DialogCode.Accepted and self.store.accepts(snapshot):
            if dialog.composition != snapshot.composition:
                self.change_parameter("composition", dialog.composition, sync=True)
        dialog.deleteLater()

    def reset_composition(self):
        state = self.store.current
        if not state or self._exporting or state.composition == Composition():
            return
        self.change_parameter("composition", Composition(), sync=True)

    def _update_crop_summary(self):
        state = self.store.current
        composition = state.composition if state else Composition()
        cropped = composition.crop != Composition().crop
        if cropped:
            left, top, right, bottom = composition.crop
            frame = f"Cropped {round((right - left) * 100)}% × {round((bottom - top) * 100)}%"
        else:
            frame = "Full frame"
        self.crop_summary.setText(f"{frame} · Angle {composition.angle:+.2f}°")
        self.crop_reset_button.setEnabled(bool(state) and self.crop_open_button.isEnabled()
                                          and composition != Composition())

    def _question_changed(self) -> None:
        if self._closing:
            return
        from .i18n import tr
        if not self.question.text().strip():
            self._pending_follow_up = False
        self.coach_text.set_question_draft(self.question.text())
        self._invalidate_coach(tr("Question updated. Analyze when ready.", "提问已更新，准备好后点击分析。"), keep_reference=True)

    def _base_rendering_changed(self, index: int) -> None:
        if index >= 0 and not self._closing:
            self.change_parameter("base_rendering", self.base_rendering.itemData(index), sync=True)

    def change_parameter(self, name, value, *, sync=False) -> None:
        if not self.store.current or self._exporting:
            return
        if name != "goal":
            self._ensure_edited_view()
        if name == "detail" and not sync:
            self._native_detail_preview = True
        self.store.set_parameter(name, value)
        if name in self.adjustment_labels:
            self.adjustment_labels[name].setText(f"{name.title()}  {value:.2f}")
        if sync:
            self._sync_controls()
        self._after_edit(interactive=not sync and name in ("contrast", "saturation", "highlights", "shadows", "whites", "blacks", "vibrance", "rgb_curve", "color_mixer", "color_grading", "local_gradients", "spots", "lens_corrections", "detail"))

    def change_goal(self) -> None:
        if self.store.current and not self._exporting and self.goal.text() != self.store.current.goal:
            self.change_parameter("goal", self.goal.text())

    def _sync_controls(self) -> None:
        state = self.store.current
        if state is None:
            return
        widgets = [self.exposure_slider, self.compare, self.wb_mode, self.base_rendering, self.goal,
                   *self.gains, *self.adjustment_sliders.values()]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.exposure_slider.setValue(round(state.exposure_ev * 100))
            self.exposure_label.setText(f"Exposure  {state.exposure_ev:+.2f} EV")
            self.compare.setChecked(state.compare_baseline)
            self.base_rendering.setCurrentIndex(self.base_rendering.findData(state.base_rendering))
            self.wb_mode.setCurrentIndex(1 if state.raw_options.wb_mode == "custom" else 0)
            for spin, value in zip(self.gains, state.raw_options.wb_gains):
                spin.setValue(value if state.raw_options.wb_mode == "custom" else 1.)
                spin.setToolTip(f"Applied relative gain: {value:.12g}. Draft input applies only on click.")
            self._gain_draft_changed()
            self.wb_hint.setText("Applied WB · Warmth / Tint are relative sensor balance, not Kelvin. Changes apply only on click.")
            for name, slider in self.adjustment_sliders.items():
                value = getattr(state, name)
                slider.setValue(round(value * 100))
                self.adjustment_labels[name].setText(f"{name.title()}  {value:.2f}")
            self.curve.set_points(state.rgb_curve)
            self.color_mixer.set_values(state.color_mixer)
            self.color_grading.set_values(state.color_grading)
            self.lens_editor.set_values(state.lens_corrections)
            self.detail_editor.set_values(state.detail)
            self.local_controller.sync()
            self.spots_controller.sync()
            self.goal.setText(state.goal)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._wb_form_enabled()
        self._sync_comparison_buttons()
        self._sync_snapshots()
        self._update_crop_summary()
        self._refresh_history()

    def _refresh_history(self) -> None:
        if not hasattr(self, "history_list"):
            return
        enabled = self.exposure_slider.isEnabled() and not self._exporting
        self.undo_button.setEnabled(enabled and bool(self.store.undo_stack))
        self.redo_button.setEnabled(enabled and bool(self.store.redo_stack))
        self.coach_text.set_edit_values(self.store.current.values if (enabled and self.store.current) else None)
        self.history_list.clear()
        events = self.store.events[self.store._event_start:]
        meaningful = [event for event in events if event.get("history_entry") or
                      event["type"] in ("undo", "redo", "compare", "compare_snapshot", "snapshot_save", "snapshot_delete", "snapshot_restore", "photo_switch", "project_open")]
        from .coach_view import CONTROL_NAMES
        from .i18n import get_language
        zh = get_language() == "zh"
        type_names = (
            {
                "photo_switch": "打开照片",
                "project_open": "打开工程",
                "compare": "切换基准对比",
                "compare_snapshot": "A/B 快照对比",
                "snapshot_save": "保存快照",
                "snapshot_delete": "删除快照",
                "snapshot_restore": "恢复快照",
                "reset": "重置全部修改",
                "undo": "撤销",
                "redo": "重做",
            }
            if zh
            else {
                "photo_switch": "Opened photo",
                "project_open": "Opened project",
                "compare": "View comparison",
                "compare_snapshot": "A/B snapshot comparison",
                "snapshot_save": "Saved snapshot",
                "snapshot_delete": "Deleted snapshot",
                "snapshot_restore": "Restored snapshot",
                "reset": "Reset all edits",
                "undo": "Undo",
                "redo": "Redo",
            }
        )
        for event in meaningful[-100:]:
            etype = event["type"]
            param = event["parameter"] or ""
            if etype == "parameter":
                detail = CONTROL_NAMES.get(param, param.replace("_", " ").title())
            elif etype in ("snapshot_save", "snapshot_delete") and isinstance(event.get("after") or event.get("before"), dict):
                snap_meta = event.get("after") or event.get("before")
                detail = f"{type_names[etype]}: {snap_meta.get('name', '')}"
            elif etype == "snapshot_restore" and isinstance(event.get("history_entry"), dict):
                entry_label = event["history_entry"].get("label") or ""
                if zh and entry_label.startswith("Restore snapshot: "):
                    detail = f"恢复快照：{entry_label.removeprefix('Restore snapshot: ')}"
                else:
                    detail = entry_label or type_names[etype]
            else:
                detail = type_names.get(etype, etype.replace("_", " ").title())
            self.history_list.addItem(f"{event['sequence']} · v{event['state_version']} · {detail}")
            item = self.history_list.item(self.history_list.count() - 1)
            if item is not None:
                item.setToolTip(f"{event['sequence']} · v{event['state_version']} · {etype} · {param}")
        self.history_list.scrollToBottom()
        if zh:
            self.history_summary.setText(
                f"{len(events)} 条审计事件 · {len(self.store.undo_stack)} 可撤销 / {len(self.store.redo_stack)} 可重做\n"
                + ("有未保存修改" if self.store.dirty else "已保存状态")
            )
        else:
            self.history_summary.setText(
                f"{len(events)} audit events · {len(self.store.undo_stack)} undo / {len(self.store.redo_stack)} redo\n"
                + ("Unsaved changes" if self.store.dirty else "Saved state")
            )
        self.unsaved_label.setVisible(bool(self.store.current and self.store.dirty))

    def undo(self) -> None:
        if self.store.current and self.exposure_slider.isEnabled() and not self._exporting:
            self._finish_gesture()
            if not self.store.undo_stack:
                return
            self._ensure_edited_view()
            self.store.undo()
            self._sync_controls()
            self._after_edit()

    def redo(self) -> None:
        if self.store.current and self.exposure_slider.isEnabled() and not self._exporting:
            self._finish_gesture()
            if not self.store.redo_stack:
                return
            self._ensure_edited_view()
            self.store.redo()
            self._sync_controls()
            self._after_edit()

    def reset_edits(self) -> None:
        if self.store.current and not self._exporting:
            self.local_controller.layer.finish()
            self.spots_controller.layer.finish()
            self.curve.finish()
            self._ensure_edited_view()
            self.store.reset()
            self._sync_controls()
            self._after_edit()

    def _allow_discard(self) -> bool:
        self._finish_gesture()
        self.change_goal()
        if not self.store.dirty:
            return True
        answer = QMessageBox.warning(self, "Unsaved project changes",
            "This photo has unsaved changes. Choose Cancel to save the project first, or Discard to continue.",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Discard

    def choose_project(self) -> None:
        if not self.open_project_button.isEnabled() or not self._allow_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open PhotoCoach project", "", "PhotoCoach project (*.photocoach *.json)")
        if path:
            self.open_project(Path(path))

    def open_project(self, path: Path, source_override: Path | None = None) -> None:
        self._formal_preview_session = None
        self.local_controller.layer.finish()
        self.local_gradients.edit_on_photo.setChecked(False)
        self.spots_controller.layer.finish()
        self.spots_editor.edit_on_photo.setChecked(False)
        self._cancel_obsolete_render()
        self._interactive_preview = False
        self._settle_timer.stop()
        self._finish_gesture()
        self._gesture_preview_edge = None
        self._native_detail_preview = False
        self._refining_preview = False
        self._opening_project = path
        self._pending_follow_up = False
        self._generation += 1
        self._render_timer.stop()
        self._stop_detail()
        self._pending = None
        self._pending_backdrop = None
        self._notice_timer.stop()
        self.view_notice.clear()
        self._invalidate_coach("Opening project. Previous feedback has been cleared.")
        self._render_diagnostics = {}
        self.render_details.setText("Waiting for the new photo.")
        self._set_controls(False)
        self.open_button.setEnabled(False)
        self.open_project_button.setEnabled(False)
        self.preview.clear_photo()
        self.histogram.clear("Opening project…")
        self.output_status.hide()
        self.status.setText("Validating project history and source hash in the background…")
        self._queue(Job("project_load", self._generation, lambda: self.engine.load_project(path, source_override)))

    def locate_project_source(self) -> None:
        if self._opening_project is None or self._closing:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Locate the project's original RAW (hash must match)", "", "RAW photos (*.ARW *.arw *.NEF *.nef);;Sony ARW (*.ARW *.arw);;Nikon NEF (*.NEF *.nef)")
        if path:
            self.open_project(self._opening_project, Path(path))

    def save_project(self) -> None:
        if not self.store.current or self._exporting or not self.save_project_button.isEnabled():
            return
        self._finish_gesture()
        self.change_goal()
        state = self.store.current
        suggested = str(self._project_path) if self._project_path else state.source_path.stem + ".photocoach"
        path, _ = QFileDialog.getSaveFileName(self, "Save PhotoCoach project", suggested,
            "PhotoCoach project (*.photocoach)", "", QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        destination = Path(path)
        if not destination.suffix:
            destination = destination.with_suffix(".photocoach")
        overwrite = destination.exists()
        if overwrite and QMessageBox.question(self, "Replace project?", f"Replace the existing file {destination.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            validate_destination(destination, forbidden_paths=(state.source_path,), allow_existing=overwrite)
            data = snapshot_project(self.store)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot save project", str(error))
            return
        self._freeze_output("Saving the current state and complete edit history…")
        self._queue(Job("project_save", self._generation, lambda: write_project(destination, data, overwrite=overwrite), state))

    def set_coach_paused(self, paused: bool) -> None:
        from .i18n import get_language
        zh = get_language() == "zh"
        self.coach.paused = paused
        self._invalidate_coach("Coach paused." if paused else "Coach ready. Analysis is manual.")
        self.pause_coach.setText(("恢复教练" if paused else "暂停教练") if zh else ("Resume coach" if paused else "Pause coach"))
        self._update_coach_status()

    def configure_api(self, checked=False, *, new_protocol=None) -> None:
        if self._network_active is not None or self._reviewing:
            return
        if not new_protocol and self.api_settings and self.api_settings.config.protocol == "ollama":
            self.open_local_models()
            return
        previous = None if new_protocol else self.api_settings
        dialog = ApiSettingsDialog(self, previous, self.api_budget,
                                   settings_store=self.settings_store, credentials=self.credentials,
                                   connection_book=self.connection_book, session_only_ids=self._session_only_ids)
        if new_protocol:
            dialog.protocol.setCurrentIndex(dialog.protocol.findData(new_protocol))
            dialog.model.clear()
        result = dialog.exec()
        # Metadata is cached by its explicit refresh action, independently of applying settings.
        if self.connection_store:
            self.model_catalog_store = ModelCatalogStore(self.connection_store.path.with_name('model_catalog.json'))
        if result != QDialog.DialogCode.Accepted:
            dialog.deleteLater()
            self._update_coach_status()
            return
        self.api_settings = dialog.result_settings
        if hasattr(dialog, "result_book") and dialog.result_book is not None:
            self.connection_book = dialog.result_book
            self._session_only_ids = set(getattr(dialog, 'result_session_only_ids', self._session_only_ids))
            if self.connection_book.active:
                self._session_settings[self.connection_book.active] = self.api_settings
                if self.api_settings.remember_settings:
                    self._session_only_ids.discard(self.connection_book.active)
                else:
                    self._session_only_ids.add(self.connection_book.active)
        self._remember_coach_models(getattr(dialog, "_catalog", {}))
        self._review_settings = None
        self.api_budget.configure(**self.api_settings.budget_settings)
        notice = dialog.result_notice
        dialog.key.clear()
        dialog.deleteLater()
        saved = "Connection settings saved for next time." if self.api_settings.remember_settings else "Connection settings are in use for this session."
        action = "Select AI coach" if self.dev_mode else "Ask coach"
        self._invalidate_coach((notice or saved) + f" No request was sent. {action} when ready.")
        self._update_coach_status()

    def _coach_budget(self):
        return self.local_budget if self.api_settings and self.api_settings.config.protocol == "ollama" else self.api_budget

    def open_local_models(self, checked=False):
        if self._network_active is not None or self._reviewing or self._closing:
            return
        if self._local_dialog and self._local_dialog.isVisible():
            self._local_dialog.raise_()
            return
        from .local_models_ui import LocalModelsDialog
        dialog = LocalModelsDialog(self, self.local_runtime, self.local_config)
        self._local_dialog = dialog
        dialog.selected.connect(self._use_local_config)
        def _sync_local_on_close():
            self.local_config = dialog.config
            if self.api_settings is not None and self.api_settings.config.protocol == "ollama":
                try:
                    synced_cfg = replace(
                        self.api_settings.config,
                        endpoint=self.local_config.endpoint,
                        model=self.local_config.model,
                    )
                    self.api_settings = replace(self.api_settings, config=synced_cfg)
                except Exception:
                    pass
            self._update_coach_status()
        dialog.finished.connect(_sync_local_on_close)
        dialog.show()

    def _use_local_config(self, config):
        if self._network_active is not None or self._reviewing or self._closing:
            return
        from .api_coach import ApiConfig
        from .settings import ApiSettings
        from .ollama_client import ENDPOINT
        self.local_config = config
        self.local_budget.configure(call_limit=config.call_limit)
        endpoint = getattr(config, "endpoint", ENDPOINT)
        settings = ApiSettings(ApiConfig(endpoint, config.model, "ollama", 300, 0),
                               {"call_limit": config.call_limit}, key_environment="", remember_settings=True)
        book = self.connection_book or ConnectionBook()
        if self.api_settings and book.active:
            self._session_settings[book.active] = self.api_settings
        existing = book.connection_for_provider("ollama")
        connection = connection_from_settings(settings, connection_id=existing.id if existing else None)
        self.connection_book = book.with_connection(connection)
        self.api_settings = settings
        self._session_settings[connection.id] = settings
        self._review_settings = None
        self.coach_mode.setCurrentIndex(1)
        self._invalidate_coach("Local AI selected. Analysis is manual; no photo has been sent.", keep_reference=True, coach_changed=True)
        self._persist_connection_book()
        self._update_coach_status()
        if self._local_dialog:
            self._local_dialog.accept()

    def _persist_connection_book(self):
        if not self.connection_store or not self.connection_book:
            return
        if self.api_settings is None or not self.api_settings.remember_settings:
            return
        book = self.connection_book
        for connection_id in self._session_only_ids:
            book = book.without(connection_id)
        try:
            self.connection_store.save(book)
        except SettingsError as error:
            self.coach_text.setPlainText(str(error) + ' Your changes are available in this window only.', kind="error")

    def _clear_model_prices(self):
        if self.api_settings and self.api_settings.config.protocol == "ollama":
            return
        # A different model/destination cannot inherit another model's token rates.
        self.api_budget.configure(input_usd_per_million=None, output_usd_per_million=None)
        if self.connection_book:
            rates = dict(self.connection_book.budget_settings, input_usd_per_million=None, output_usd_per_million=None)
            self.connection_book = replace(self.connection_book, budget_settings=rates)
            self.api_settings = replace(self.api_settings, budget_settings=rates)

    def _choose_coach_provider(self, index: int) -> None:
        if self._network_active is not None or self._reviewing:
            self._update_coach_status()
            return
        choice = self.coach_provider_picker.itemData(index)
        connection = self.connection_book.get(choice) if self.connection_book else None
        if connection is None or (self.connection_book.active == choice and self.api_settings is not None):
            return
        if self.connection_book.active and self.api_settings:
            self._session_settings[self.connection_book.active] = self.api_settings
        self.connection_book = self.connection_book.with_active(choice)
        saved = self._session_settings.get(choice)
        self.api_settings = (saved if saved and saved.config == connection.config else
                             settings_for(connection, self.connection_book.budget_settings,
                                          remember=choice not in self._session_only_ids))
        self._clear_model_prices()
        self._review_settings = None
        self._invalidate_coach('Connection changed. No request was sent. Review and ask when ready.', keep_reference=True, coach_changed=True)
        self._persist_connection_book()
        self._update_coach_status()

    def _remember_coach_models(self, models=()):
        if self.api_settings is None:
            return {}
        from .model_catalog import DEFAULT_CURATED, KNOWN_VISION_MODELS, is_ui_selectable_model; config = self.api_settings.config
        provider, scope = infer_provider(config), credential_target(config)
        if provider == "ollama":
            verified = tuple(getattr(self.local_config, "verified_models", ()) or ())
            return {mid: mid for mid in (self.local_config.model, *verified) if mid}
        stored = self.model_catalog_store.get_models(provider, scope) if self.model_catalog_store else ()
        catalog = {m.id: m.id for m in stored if is_ui_selectable_model(provider, m.id)}
        for mid in (*DEFAULT_CURATED.get(provider, ()), *KNOWN_VISION_MODELS.get(provider, ())):
            if is_ui_selectable_model(provider, mid):
                catalog[mid] = mid
        extra = self._coach_model_catalogs.setdefault(scope, {})
        for mid in models:
            if is_ui_selectable_model(provider, mid):
                extra[mid] = mid
        catalog.update(extra)
        connection = self.connection_book.active_connection if self.connection_book else None
        if connection and credential_target(connection.config) == scope:
            for mid in (*connection.favorites, *connection.recent):
                if is_ui_selectable_model(provider, mid):
                    catalog[mid] = mid
        if is_ui_selectable_model(provider, config.model) or provider != 'gemini':
            catalog[config.model] = config.model
        return catalog

    def _select_coach_model(self, model):
        if (self.api_settings is None or self.coach_mode.currentIndex() != 1 or
                self._network_active is not None or self._reviewing):
            return
        if model == self.api_settings.config.model:
            return
        try:
            config = replace(self.api_settings.config, model=model)
        except (ApiError, ValueError) as error:
            self.coach_text.setPlainText(str(error), kind="error")
            return
        if self.connection_book and self.connection_book.active:
            self.connection_book = self.connection_book.with_model(self.connection_book.active, model)
        self.api_settings = replace(self.api_settings, config=config)
        if config.protocol == "ollama" and hasattr(self.local_config, "with_active_model"):
            try:
                self.local_config = self.local_config.with_active_model(model)
                self.local_runtime.store.save(self.local_config)
            except Exception:
                pass
        self._clear_model_prices()
        idx = self.coach_model_picker.findData(model)
        if idx >= 0:
            self.coach_model_picker.setCurrentIndex(idx)
        if self.connection_book and self.connection_book.active:
            self._session_settings[self.connection_book.active] = self.api_settings
        self._review_settings = None
        self._invalidate_coach('Model changed. No request was sent. Review and ask when ready.', keep_reference=True, coach_changed=True)
        self._persist_connection_book()
        self._update_coach_status()

    def _choose_coach_model(self, index):
        model = self.coach_model_picker.itemData(index)
        if model is None:
            # This row is an action, never a manual model-name draft.
            self.configure_api()
            self._update_coach_status()
        else:
            self._select_coach_model(model)

    def _toggle_coach_favorite(self):
        if not self.api_settings or not self.connection_book or not self.connection_book.active:
            return
        try:
            self.connection_book = self.connection_book.toggle_favorite(self.connection_book.active, self.api_settings.config.model)
        except SettingsError:
            self.coach_text.setPlainText('Favorite limit reached. Remove another favorite first.')
            return
        self._persist_connection_book()
        self._update_coach_status()

    def change_coach_mode(self) -> None:
        if self.coach_mode.currentIndex() == 1:
            self.auto_mock.setChecked(False)
        self._invalidate_coach("Coach mode changed. Analysis is manual; no request has been sent.")
        self._update_coach_status()

    def _update_coach_status(self) -> None:
        cloud = self.coach_mode.currentIndex() == 1
        provider = infer_provider(self.api_settings.config) if self.api_settings else "gemini"
        picker = self.coach_provider_picker
        picker.blockSignals(True)
        connections = self.connection_book.connections if self.connection_book else ()
        def _conn_label(c):
            host = urlsplit(c.config.endpoint).hostname or ''
            official = (c.label == 'Google Gemini' and host == 'generativelanguage.googleapis.com') or (
                c.label == 'OpenAI' and host == 'api.openai.com')
            shared = sum(1 for other in connections if other.label == c.label) > 1
            return c.label if (official and not shared) else f'{c.label} · {host}' if host else c.label
        choices = [(_conn_label(c), c.id) for c in connections]
        if not choices:
            from .i18n import tr
            choices = [(tr("No saved connections", "尚未保存连接"), None)]
        if [(picker.itemText(i), picker.itemData(i)) for i in range(picker.count())] != choices:
            picker.clear()
            for label, value in choices:
                picker.addItem(label, value)
                connection = self.connection_book.get(value) if self.connection_book else None
                if connection:
                    picker.setItemData(picker.count()-1, connection.config.endpoint, Qt.ItemDataRole.ToolTipRole)
        if self.connection_book and self.connection_book.active:
            picker.setCurrentIndex(max(0, picker.findData(self.connection_book.active)))
        picker.blockSignals(False)
        picker.setEnabled(cloud and bool(connections) and self._network_active is None and not self._reviewing)
        catalog = self._remember_coach_models()
        active_conn = self.connection_book.active_connection if self.connection_book else None
        favorites = set(active_conn.favorites if active_conn else ())
        recent = list(active_conn.recent if active_conn else ())
        from .model_catalog import DEFAULT_CURATED, KNOWN_VISION_MODELS, is_ui_selectable_model; config = self.api_settings.config if self.api_settings else None
        scope = credential_target(config) if config else None
        stored = self.model_catalog_store.get_models(provider, scope) if self.model_catalog_store and config else []
        info = {m.id: m for m in stored}
        def order(mid):
            return (0 if mid in favorites else 1 if mid in recent else 2, recent.index(mid) if mid in recent else 0, mid)
        rows = []
        for mid in sorted(catalog, key=order):
            model = info.get(mid)
            cap = classify_model(provider, mid)
            tag = 'Verified photo analysis' if model and model.verified_utc else (
                'Image analysis supported' if cap == 'supported' else 'Not suitable' if cap == 'unsupported' else 'Needs verification')
            if provider == "ollama":
                from .i18n import tr
                tag = tr("Local model", "本地模型")
            prefix = '[Favorite] ' if mid in favorites else 'Recent · ' if mid in recent else ''
            rows.append((f'{prefix}{mid} · {tag}', mid))
        rows.append(('Choose / load more models…', None))
        self.coach_model_picker.blockSignals(True)
        if [(self.coach_model_picker.itemText(i), self.coach_model_picker.itemData(i)) for i in range(self.coach_model_picker.count())] != rows:
            self.coach_model_picker.clear()
            for label, mid in rows:
                self.coach_model_picker.addItem(label, mid)
                source, checked = capability_evidence(provider, mid) if mid else (None, None)
                tip = label + (f'\nOfficial capability: {source} · Checked {checked}' if source else '')
                if source:
                    tip += '\nImage input is documented; account access and PhotoCoach response compatibility are unverified.'
                if mid and mid in info and info[mid].verified_utc:
                    tip += '\nPhoto analysis succeeded on this destination at ' + info[mid].verified_utc
                self.coach_model_picker.setItemData(self.coach_model_picker.count()-1, tip, Qt.ItemDataRole.ToolTipRole)
        if config:
            self.coach_model_picker.setCurrentIndex(max(0, self.coach_model_picker.findData(config.model)))

        self.coach_model_picker.blockSignals(False)
        self.coach_model_picker.setEnabled(cloud and config is not None and self._network_active is None and not self._reviewing)
        self.coach_favorite.setEnabled(self.coach_model_picker.isEnabled() and active_conn is not None)
        from .i18n import get_language
        zh = get_language() == "zh"
        local = bool(config and config.protocol == "ollama")
        self.local_ai_button.setText("本地模型" if zh else "Local models")
        self.local_ai_button.setEnabled(self._network_active is None and not self._reviewing and not self._closing)
        self.coach_favorite.setText(('已收藏' if config and config.model in favorites else '收藏') if zh else ('Favorited' if config and config.model in favorites else 'Favorite'))
        self.coach_favorite.setChecked(bool(config and config.model in favorites))
        updated = self.model_catalog_store.get_last_updated(provider, scope) if self.model_catalog_store and config else None
        self.coach_catalog_status.setText(('Models updated ' + updated[:19].replace('T', ' ') + ' UTC') if updated else 'Models from documentation · Account access not checked')
        if local:
            self.coach_catalog_status.setText("每次分析前检查本地模型状态" if zh else "Local model readiness checked before each analysis")
        if self.model_catalog_store and self.model_catalog_store.notice:
            self.coach_catalog_status.setText(self.coach_catalog_status.text() + '\n' + self.model_catalog_store.notice)
        self.auto_mock.setEnabled(not cloud)
        self.auto_idle.setEnabled(not cloud)
        self.auto_interval.setEnabled(not cloud)
        self._update_auto_status()
        direct = self.api_settings is not None and self._review_settings is self.api_settings
        self.restore_review_button.setEnabled(direct)
        self.connection_label.setText("Local processing · Manual API coach" if cloud else
                                     "Local processing · Auto mock coach" if self.auto_mock.isChecked() else
                                     "Local processing · Mock coach")
        if self.api_settings:

            config = self.api_settings.config
            provider = {"gemini": "Google Gemini", "responses": "Responses API",
                        "chat_completions": "Compatible API", "ollama": "Local AI · Ollama"}[config.protocol]
            source = {"saved": "Saved key selected", "environment": "Environment key selected",
                      "session": "Session key selected"}[self.api_settings.key_source]
            prefix = "" if cloud else "Saved connection · " if self.api_settings.remember_settings else "Available connection · "
            self.coach_connection.setText(f"{prefix}{provider} · {config.model}\n{source}")
            if local:
                self.coach_connection.setText(f"Local AI · {config.model}\nNo API key · This PC only")
        else:
            self.coach_connection.setText(("No AI connection yet · Set up AI coach" if not self.dev_mode
                                           else "Choose a model in Connection…") if cloud else "Local parameter rules · No model connected")
        self.coach_hint.setText((f"Direct send this session · {'Before + current' if self._review_include_before else 'Current only'} · May incur charges" if direct else "Review previews before sending · Manual only") if cloud else
                               "Local rules only; the demo cannot judge photo quality.")
        if local:
            self.connection_label.setText("本机 AI · 手动分析" if zh else "Local AI · Manual analysis")
            self.coach_hint.setText("照片仅在本机分析 · 手动触发" if zh else "Photo analysis on this PC only · Manual")
        self.mock_limit_row.setVisible(not cloud)
        self.text_probe_button.setVisible(bool(cloud and self.api_settings and self.api_settings.config.protocol == "gemini"))
        self.coach_selector.setToolTip(self.coach_connection.text())
        self.coach_connection_button.setToolTip(self.coach_connection.text())
        self.coach_usage.setText(usage_summary(self._coach_budget(), self.coach.calls, cloud))
        if local:
            self.coach_usage.setText((f"本地请求：{self.local_budget.attempts}/{self.local_budget.call_limit} · 无 API 费用" if zh else
                f"Local requests: {self.local_budget.attempts}/{self.local_budget.call_limit} · No API charges"))
        self.text_probe_button.setEnabled(bool(cloud and self.api_settings and
            self.api_settings.config.protocol == "gemini" and not self._exporting and not self.coach.paused and
            not self._closing and not self._coach_busy() and not self._reviewing))
        if self.api_budget.attempts or cloud:
            self.coach_count.setText((f"Mock analyses: {self.coach.calls}\n" if self.dev_mode else "") + budget_text(self._coach_budget()))
        else:
            self.coach_count.setText(f"Mock analyses: {self.coach.calls} · API attempts: 0 · Cost: $0")
        if local:
            self.coach_count.setText(self.coach_usage.text())
        if self._local_dialog and self._local_dialog.isVisible():
            self._local_dialog.refresh()
        self._sync_coach_presentation()

    def _coach_busy(self):
        return (self._network_active is not None or
                bool(self._active and self._active[1].kind in ("coach", "coach_check", "api_prepare", "agent_tool")) or
                any(job.kind in ("coach", "coach_check", "api_prepare", "agent_tool") for job in self._protected_jobs))

    def _sync_coach_presentation(self):
        if self._closing:
            return
        from .i18n import get_language
        zh = get_language() == "zh"
        def text(en, cn):
            return cn if zh else en
        config = self.api_settings.config if self.api_settings else None
        demo = self.coach_mode.currentIndex() == 0
        local = bool(config and config.protocol == "ollama")
        if demo:
            name = text("Local demo", "本地演示")
        elif config:
            provider = {"gemini": "Gemini", "responses": "OpenAI", "chat_completions": "Compatible", "ollama": "Local AI"}[config.protocol]
            name = config.model + " · " + provider
        else:
            name = text("Choose coach", "选择教练")
        self.coach_selector.setText(self.coach_selector.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, 280))
        busy = self._coach_busy()
        self.coach_selector.setEnabled(not busy and not self._reviewing and not self._closing)
        action = text("Analyze photo", "分析照片")
        enabled = bool(self.store.current and self.exposure_slider.isEnabled() and
                       not self._exporting and not self._closing and not self.coach.paused)
        status = text("Ready", "可以分析")
        if self._reviewing:
            status, enabled = text("Review before sending", "请确认发送内容"), False
        elif busy:
            if self._coach_cancel_requested:
                status, enabled = text("Cancelling analysis…", "正在取消分析…"), False
                action = text("Cancelling…", "正在取消…")
            else:
                status, action, enabled = text("Analyzing…", "正在分析…"), text("Cancel analysis", "取消分析"), True
                if local and self._network_progress is not None:
                    stage, _ = self._network_progress[0].snapshot()
                    status = (text("Preparing local model…", "正在准备本地模型…") if stage == "local_preparing" else
                              text("Loading model / analyzing…", "正在加载模型或分析照片…"))
                if self._active and self._active[1].kind == "coach_check":
                    status = text("Checking proposed edit…", "正在检查试调…")
        elif not demo and config is None:
            status, action = text("Choose a local or cloud coach", "选择本地或云端教练"), text("Set up coach", "设置教练")
            enabled = not self._closing and not self._exporting
        elif local and not self.local_config.verified_digest:
            status, action = text("Local model needs preparation", "请准备本地模型"), text("Prepare local model", "准备本地模型")
            enabled = not self._closing and not self._exporting
        elif self.store.current is None or not self.exposure_slider.isEnabled():
            status = text("Open a photo to analyze", "打开照片后即可分析")
        elif self._exporting:
            status = text("Exporting photo…", "正在导出照片…")
        elif self.coach.paused:
            status = text("Coach paused · resume in Usage", "教练已暂停，可在用量中恢复")
        elif self._coach_cancel_requested:
            status = text("Analysis cancelled", "分析已取消")
        elif self.coach_text._notice_kind == "error":
            status = text("Analysis unavailable · see details below", "分析未完成，请查看下方说明")
        elif self.coach_text._notice_kind == "cancelled":
            status = text("Analysis cancelled", "分析已取消")
        elif self.coach_text._feedback is not None:
            output = self.coach_text
            if output._superseded_coach:
                status = text("Coach changed · analyze again", "教练已切换，请重新分析")
            elif output._superseded_question or output._question_changed():
                status = text("Edit or question changed · analyze again", "编辑或提问已变化，请重新分析")
            elif output._actions and output._values is not None and all(a.is_applied(output._values) for a in output._actions):
                status = text("Applied · Undo to compare", "已应用，可撤销比较")
            elif output.reference_revision is not None:
                status = text("Edit or question changed · analyze again", "编辑或提问已变化，请重新分析")
            elif output._feedback.agent_report is not None:
                report = output._feedback.agent_report
                status = text(*{"review": ("Trial checked · Preview before applying", "试调已检查，请先预览比较"),
                                "blocked": ("Suggestion held · Review trial checks", "建议暂不应用，请查看试调检查"),
                                "no_edit": ("Review complete · No edit proposed", "分析完成，本轮不修改照片")}[report.status])
            else:
                status = text("Reply ready", "建议已就绪")
        self.coach_summary.setText(status)
        self.coach_summary.setToolTip(status)
        self.analyze_button.setText(action)
        self.analyze_button.setEnabled(enabled and not self._closing and not self._exporting and not self.coach.paused)
        self.analyze_button.setToolTip(text("Ctrl+Enter to analyze. Enter adds a line.", "Ctrl+Enter 分析，Enter 换行。"))
        ready_to_send = bool(
            self.store.current and self.exposure_slider.isEnabled()
            and not self._exporting and not self._closing and not self.coach.paused
            and not self._reviewing and not busy
            and (demo or config is not None)
            and not (local and not self.local_config.verified_digest)
        )
        can_follow_up = ready_to_send and self._can_follow_up()
        self.followup_button.setText(text("Did it help?", "效果复盘"))
        self.followup_button.setEnabled(can_follow_up)
        self.followup_button.setToolTip(
            text(
                "Evaluate adjustments since the previous coach advice against your goal.",
                "基于上一轮教练建议与之后的调整，评估这次修改是否有效。",
            )
            if can_follow_up else
            text(
                "Enabled after receiving coach advice and making an edit or Undo on this photo.",
                "需先对当前照片获取一轮教练建议，并继续编辑或撤销后才可复盘。",
            )
        )
        self.coach_questions_button.setEnabled(not busy and not self._reviewing and not self._closing)

    def _can_follow_up(self) -> bool:
        state = self.store.current
        if state is None:
            return False
        return self.agent_memory.has_subsequent_edits(
            state.photo_id,
            state.session_id,
            state.revision,
            self.store.project_events(),
            current_values=state.values,
        )

    def _coach_primary_action(self):
        if not self.analyze_button.isEnabled():
            return
        if self._coach_busy():
            self._coach_cancel_requested = True
            self._protected_jobs = [job for job in self._protected_jobs
                                    if job.kind not in ("coach", "coach_check", "api_prepare", "agent_tool")]
            from .i18n import tr
            self._invalidate_coach(tr("Analysis cancelled.", "分析已取消。"))
            self._sync_coach_presentation()
            return
        if self.coach_mode.currentIndex() == 1 and self.api_settings is None:
            self._show_coach_selection()
            return
        if self.api_settings and self.api_settings.config.protocol == "ollama" and not self.local_config.verified_digest:
            self.open_local_models()
            return
        self._coach_cancel_requested = False
        self.coach_selection.hide()
        self.analyze()

    def _show_coach_selection(self):
        if self._closing:
            return
        self._update_coach_status()
        self.coach_selection.show()
        self.coach_selection.raise_()

    def _configure_cloud_coach(self):
        self.coach_selection.hide()
        if self.api_settings and self.api_settings.config.protocol == "ollama":
            self.configure_api(new_protocol="gemini")
        else:
            self.configure_api()

    def _toggle_coach_goal(self, checked):
        if checked:
            self.coach_collapse.setChecked(False)
        self.goal_panel.setVisible(checked)
        if checked:
            self.goal.setFocus()

    def _use_coach_question(self, text, *, follow_up: bool = False):
        self.coach_collapse.setChecked(False)
        self.question.selectAll()
        self.question.insert(text)
        self._pending_follow_up = bool(follow_up)
        self.question.setFocus()

    def ask_follow_up(self) -> None:
        if (not self._can_follow_up() or self._coach_busy() or self._reviewing
                or self._closing or self._exporting or self.coach.paused):
            return
        self.coach_collapse.setChecked(False)
        state = self.store.current
        prior = self.agent_memory.last_advice_for(state.photo_id, state.session_id) if state else None
        current_q = self.question.text().strip()
        from .i18n import get_language
        zh = get_language() == "zh"
        default_q = (
            "结合上一轮教练建议和我的编辑目标，这次调整有帮助吗？请区分客观画面变化与主观偏好，并说明是否该停下。"
            if zh else
            "Did my adjustment help relative to your previous advice and my editing goal? Separate objective preview changes from preference, and tell me if I should stop."
        )
        if not current_q or (prior is not None and current_q == prior.question.strip()):
            self.question.selectAll()
            self.question.insert(default_q)
        self.analyze(follow_up=True)

    def _show_coach_details(self) -> None:
        self.coach_details.show()
        self.coach_details.raise_()

    def restore_send_confirmation(self):
        self._review_settings = None
        self._update_coach_status()

    def _set_coach_collapsed(self, collapsed: bool) -> None:
        sizes = self.workspace_splitter.sizes()
        total = sum(sizes)
        if collapsed:
            if self.coach_body.isVisible():
                self._expanded_coach_height = sizes[1]
            self.coach_body.hide()
            self.coach_panel.layout().invalidate()
            height = self.coach_panel.minimumSizeHint().height()
            self.coach_panel.setMaximumHeight(height)
            self.workspace_splitter.setSizes([max(300, total - height), height])
        else:
            self._coach_unread = False
            # The final inherited stylesheet/font is available after parenting.
            self.goal.setMinimumHeight(self.goal.sizeHint().height())
            self.question._resize_lines()
            self.coach_panel.setMaximumHeight(16777215)
            self.coach_body.show()
            self.workspace_splitter.setSizes([max(300, total - self._expanded_coach_height), self._expanded_coach_height])
        self._coach_toggle_text()

    def _coach_feedback_changed(self, available):
        self._coach_unread = bool(available and self.coach_collapse.isChecked())
        self._coach_toggle_text()
        if available:
            self._sync_coach_presentation()

    def _coach_toggle_text(self):
        from .i18n import get_language
        if get_language() == "zh":
            self.coach_collapse.setText(("展开教练 · 有新回复" if self._coach_unread else "展开教练")
                                       if self.coach_collapse.isChecked() else "收起")
        else:
            self.coach_collapse.setText(("Open coach · New reply" if self._coach_unread else "Open coach")
                                       if self.coach_collapse.isChecked() else "Collapse")

    def _clear_control_highlight(self) -> None:
        if self._highlighted_label is not None:
            self._highlighted_label.setProperty("coachTarget", False)
            self._highlighted_label.style().unpolish(self._highlighted_label)
            self._highlighted_label.style().polish(self._highlighted_label)
            self._highlighted_label.update()
            self._highlighted_label = None
        self._highlight_timer.stop()

    def _locate_control(self, name: str) -> None:
        if name=='local_gradients' or name.startswith('local_gradients.'):
            self.local_controller.locate(name.partition('.')[2] or None)
            return
        if name == "spots" or name.startswith("spots."):
            self.spots_controller.locate()
            return
        def reveal(widget, *, in_editor=True):
            self._navigation_target = widget
            def after_layout():
                if self._closing or self._navigation_target is not widget:
                    return
                widget.setFocus(Qt.FocusReason.OtherFocusReason)
                if in_editor:
                    panel = self.edit_scroll.widget()
                    vp = self.edit_scroll.viewport()
                    if panel is not None and vp is not None:
                        for parent_widget in (widget.parentWidget(), self.color_grading, self.color_section, panel):
                            if parent_widget is not None and parent_widget.layout() is not None:
                                parent_widget.layout().activate()
                        target_h = panel.heightForWidth(vp.width()) if panel.hasHeightForWidth() else panel.sizeHint().height()
                        panel.resize(vp.width(), max(vp.height(), target_h))
                        if panel.layout() is not None:
                            panel.layout().activate()
                    self.edit_scroll.ensureWidgetVisible(widget, 0, 30)
            # Expanding a group posts a scroll-area resize. Scroll only after
            # that layout request, so a newly visible control is actually reached.
            QTimer.singleShot(0, self, after_layout)
        if name != "composition":
            section = (self.detail_section if name == "detail" or name.startswith("detail.") else self.lens_section if name.startswith("lens_corrections") else self.color_section if name.startswith("color_mixer") or name.startswith("color_grading") or name in
                       ("white_balance", "vibrance", "saturation") else
                       self.curve_section if name == "rgb_curve" else self.light_section)
            section.set_expanded(True)
            section.layout().activate()
            self.edit_scroll.widget().layout().activate()
        if name == "color_mixer" or name.startswith("color_mixer."):
            color = name.partition(".")[2]
            self.color_mixer.locate(color)
            self.tabs.setCurrentIndex(0)
            reveal(self.color_mixer.color)
            return
        if name == "color_grading" or name.startswith("color_grading."):
            sub = name.partition(".")[2]
            widget, label = self.color_grading.locate(sub or None)
            self.color_section.layout().activate()
            self.edit_scroll.widget().layout().activate()
            self._clear_control_highlight()
            self.tabs.setCurrentIndex(0)
            reveal(widget)
            self._highlighted_label = label
            label.setProperty("coachTarget", True)
            label.style().unpolish(label)
            label.style().polish(label)
            label.update()
            self._highlight_timer.start()
            return
        targets = {"exposure_ev": (self.exposure_slider, self.exposure_label),
                   "white_balance": (self.wb_mode, self.wb_form.labelForField(self.wb_mode)),
                   "contrast": (self.adjustment_sliders["contrast"], self.adjustment_labels["contrast"]),
                   "saturation": (self.adjustment_sliders["saturation"], self.adjustment_labels["saturation"]),
                   "rgb_curve": (self.curve, self.curve_label)}
        for parameter in ("highlights", "shadows", "whites", "blacks", "vibrance"):
            targets[parameter] = (self.adjustment_sliders[parameter], self.adjustment_labels[parameter])
        targets["lens_corrections"] = (self.lens_editor.sliders["distortion"], self.lens_editor.labels["distortion"])
        for parameter in ("distortion", "vignetting_ev"):
            targets[f"lens_corrections.{parameter}"] = (self.lens_editor.sliders[parameter], self.lens_editor.labels[parameter])
        targets["detail"] = (self.detail_editor.sliders["sharpening"], self.detail_editor.labels["sharpening"])
        for parameter in ("sharpening", "masking", "luminance_nr", "chroma_nr"):
            targets[f"detail.{parameter}"] = (self.detail_editor.sliders[parameter], self.detail_editor.labels[parameter])
        targets["composition"] = (self.composition_button, self.composition_button)
        if name not in targets:
            return
        widget, label = targets[name]
        self._clear_control_highlight()
        self.tabs.setCurrentIndex(0)
        reveal(widget, in_editor=name != "composition")
        self._highlighted_label = label
        label.setProperty("coachTarget", True)
        label.style().unpolish(label)
        label.style().polish(label)
        label.update()
        self._highlight_timer.start()

    def apply_coach_action(self, action) -> bool:
        if not self.store.current or self._exporting or not self.exposure_slider.isEnabled():
            return False
        self._finish_gesture()
        before_values = self.store.current.values
        changes = getattr(action, "changes", ())
        if len(changes) > 1:
            self._ensure_edited_view()
            self.store.apply_changes(changes, primary_parameter=action.parameter)
            self._sync_controls()
            self._after_edit(interactive=False)
        elif action.parameter == "raw_options":
            self._apply_white_balance(action.new_value)
        else:
            self.change_parameter(action.parameter, action.new_value, sync=True)
        if not action.is_applied(self.store.current.values):
            self.status.setText("The edit does not match all suggested targets.")
            return False
        self.agent_memory.mark_last_applied(
            action.label,
            photo_id=self.store.current.photo_id,
            session_id=self.store.current.session_id,
        )
        try:
            self.style_memory.record_applied_edit(before_values, self.store.current.values, self.store.current.goal)
        except Exception:
            pass
        has_local = action.parameter == "local_gradients" or any(p == "local_gradients" for p, _ in changes)
        if has_local:
            lg_val = self.store.current.local_gradients
            target_id = action.focus_control.partition(".")[2] if action.focus_control.startswith("local_gradients.") else (
                lg_val[-1].id if lg_val else None)
            self.local_gradients.set_values(lg_val, selected_id=target_id)
            self.local_controller.locate(target_id)
        elif action.focus_control:
            self._locate_control(action.focus_control)
        self.status.setText(f"Applied coach suggestion: {action.label} · Use Undo (Ctrl+Z) to compare.")
        return True

    def _apply_coach_suggestion(self, index: int = 0, *, allow_previewed_override: bool = False) -> bool:
        state = self.store.current
        feedback = self.coach_text._feedback
        if state is None or feedback is None or self._exporting or not self.exposure_slider.isEnabled():
            return False
        if ((feedback.photo_id, feedback.session_id, feedback.state_version) !=
                (state.photo_id, state.session_id, state.revision) or self.question.text() != feedback.question
                or self.coach_text._superseded_question or self.coach_text._superseded_coach or self.store.gesture_active or self.coach_text.reference_revision is not None):
            return False
        actions = feedback.actions
        if feedback.agent_plan:
            if not allow_previewed_override:
                if not feedback.agent_report or not feedback.agent_report.can_apply:
                    return False
            else:
                if (feedback.agent_plan.decision != "recommend" or bool(feedback.action_notice)
                        or not feedback.agent_report or feedback.agent_report.trial_preview is None):
                    return False
        if not (0 <= index < len(actions)):
            return False
        return not actions[index].is_applied(state.values) and self.apply_coach_action(actions[index])

    def _apply_lighting_recipe(self, recipe_id: str) -> bool:
        from .coach_actions import make_lighting_action
        state = self.store.current
        if state is None or self._exporting or not self.exposure_slider.isEnabled():
            return False
        try:
            action = make_lighting_action(recipe_id, state.values, preview=self.coach_text._preview_pixels,
                                         source_size=self.preview.source_size)
        except ValueError as error:
            self.status.setText(str(error))
            return False
        return self.apply_coach_action(action)

    def toggle_language(self, checked: bool = False) -> None:
        from .i18n import get_language
        new_lang = "zh" if get_language() == "en" else "en"
        self.set_language(new_lang)

    def set_language(self, lang: str) -> None:
        from .i18n import apply_workspace_language
        apply_workspace_language(self, lang)
        self._refresh_history()
        self._update_coach_status()

    def test_text_only(self) -> None:
        if self._coach_busy():
            return
        if (not self.api_settings or self.api_settings.config.protocol != "gemini" or
                self.coach_mode.currentIndex() != 1 or self.coach.paused or self._closing or
                self._network_active is not None or self._reviewing or self._exporting):
            return
        if self.api_budget.blocked_reason:
            self.coach_text.setPlainText(self.api_budget.blocked_reason, kind="error")
            return
        settings = self.api_settings
        effective_cfg = (replace(settings.config, max_output_tokens=8192)
                         if settings.config.protocol == "gemini" and settings.config.max_output_tokens <= 1500
                         and settings.config.model.startswith(("gemini-3.", "gemini-4"))
                         else settings.config)
        job = Job("text_probe", self._generation, lambda: None, coach_token=self._coach_generation)
        self._reviewing = True
        self._update_coach_status()
        try:
            dialog = TextProbeReviewDialog(self, effective_cfg, self.api_budget)
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            dialog.deleteLater()
        finally:
            self._reviewing = False
            self._update_coach_status()
        if not accepted:
            self.coach_text.setPlainText("Text test cancelled before sending. No API attempt was made.", kind="cancelled")
            return
        if (self._closing or self.coach.paused or self._exporting or self.api_settings is not settings or
                self.coach_mode.currentIndex() != 1 or job.coach_token != self._coach_generation or
                job.generation != self._generation or self._network_active is not None):
            self.coach_text.setPlainText("Settings or session changed during review. No text test was sent.")
            return
        try:
            key = resolve_api_key(settings, self.credentials)
            ApiCoach(effective_cfg, key)  # Validate credentials before reserving an attempt.
            ticket = self.api_budget.reserve()
        except (ApiError, ValueError, RuntimeError) as error:
            self.coach_text.setPlainText(str(error), kind="error")
            return
        cancel, progress = threading.Event(), TransportProgress()
        worker = Worker(lambda: run_text_probe(effective_cfg, key, cancel, progress=progress))
        worker.signals.done.connect(self._network_done)
        self._network_active = (worker, job, cancel, ticket)
        self._network_progress = (progress, settings.config.timeout_seconds)
        self._update_network_progress()
        self._network_timer.start()
        self._update_coach_status()
        self.network_pool.start(worker)

    def _enqueue_agent_tool(self, payload) -> None:
        snapshot, coach_token, fn, box, done_event, cancel_event = payload
        if (self._closing or cancel_event.is_set() or self.coach.paused or
                coach_token != self._coach_generation or not self.store.accepts(snapshot)):
            box.append((None, CancelledError()))
            done_event.set()
            return

        def wrapped():
            try:
                box.append((fn(), None))
            except Exception as exc:
                box.append((None, exc))
            finally:
                done_event.set()

        self._queue(Job("agent_tool", self._generation, wrapped, snapshot, coach_token, cancel=cancel_event))

    def _run_serial_agent_tool(self, snapshot, coach_token: int, cancel_event: threading.Event, fn):
        if cancel_event.is_set() or self._closing or coach_token != self._coach_generation:
            raise CancelledError()
        box: list = []
        done_event = threading.Event()
        payload = (snapshot, coach_token, fn, box, done_event, cancel_event)
        if threading.get_ident() == self._gui_thread_id:
            self._enqueue_agent_tool(payload)
            app = QApplication.instance()
            while not done_event.wait(0.01):
                if app is not None:
                    app.processEvents()
                if cancel_event.is_set() or self._closing or coach_token != self._coach_generation:
                    raise CancelledError()
        else:
            self._agentToolRequested.emit(payload)
            while not done_event.wait(0.025):
                if cancel_event.is_set() or self._closing or coach_token != self._coach_generation:
                    raise CancelledError()
        if not box:
            raise CancelledError()
        res, err = box[0]
        if err is not None:
            raise err
        return res

    def review_api_request(self, request, job) -> None:
        if (self._closing or self._network_active or self._reviewing or not self.api_settings or
                self.coach_mode.currentIndex() != 1 or self.coach.paused or self._exporting or
                job.coach_token != self._coach_generation or not self.store.accepts(job.snapshot)):
            return
        budget = self._coach_budget()
        if budget.blocked_reason:
            self.coach_text.setPlainText(budget.blocked_reason, kind="error")
            return
        settings = self.api_settings
        if settings.config.protocol == "ollama":
            local_out = 0 if settings.config.max_output_tokens in (0, 4096, 8192) else settings.config.max_output_tokens
            effective_cfg = replace(
                settings.config,
                endpoint=self.local_config.endpoint,
                model=self.local_config.model,
                max_output_tokens=local_out,
            )
        elif settings.config.protocol == "gemini" and (
            settings.config.max_output_tokens <= 1500 or settings.config.timeout_seconds <= 60
        ):
            effective_cfg = replace(
                settings.config,
                max_output_tokens=max(settings.config.max_output_tokens, 8192),
                timeout_seconds=max(settings.config.timeout_seconds, 90),
            )
        else:
            effective_cfg = settings.config
        skip_session = False
        max_turns = 1
        retouch_mode = "full_retouch"
        self._reviewing = True
        self._update_coach_status()
        try:
            if self._review_settings is settings:
                accepted = True
                include_before = self._review_include_before
                max_turns = getattr(self, "_review_max_turns", 1)
                retouch_mode = getattr(self, "_review_retouch_mode", "full_retouch")
            else:
                dialog = SendReviewDialog(self, request, effective_cfg, budget)
                accepted = dialog.exec() == QDialog.DialogCode.Accepted
                include_before = dialog.include_before.isChecked()
                skip_session = dialog.skip_session.isChecked()
                max_turns = getattr(dialog, "selected_max_turns", lambda: 1)()
                retouch_mode = getattr(dialog, "selected_retouch_mode", lambda: "full_retouch")()
                dialog.deleteLater()
        finally:
            self._reviewing = False
            self._update_coach_status()
        if not accepted:
            self.coach_text.setPlainText("Request cancelled before sending. No API attempt was made.", kind="cancelled")
            return
        if (self._closing or self.coach.paused or self._exporting or self.api_settings is not settings or
                self.coach_mode.currentIndex() != 1 or job.coach_token != self._coach_generation or
                not self.store.accepts(job.snapshot)):
            self.coach_text.setPlainText("The edit or question changed during review. Analyze again; nothing was sent.")
            return
        if skip_session:
            self._review_settings = settings
            self._review_include_before = include_before
            self._review_max_turns = max_turns
            self._review_retouch_mode = retouch_mode
        while max_turns > 1 and budget.can_authorize_turns(max_turns) is not None:
            max_turns -= 1
        if max_turns <= 1:
            retouch_mode = "coach"
        try:
            # Credentials are read only for a user-confirmed request, never logged or serialized.
            key = resolve_api_key(settings, self.credentials)
            adapter = ApiCoach(effective_cfg, key)
            if effective_cfg.protocol == "ollama":
                if not getattr(self.local_config, "supports_tools", True):
                    adapter.supports_tools = False
                if not getattr(adapter, "supports_tools", False):
                    max_turns = 1
                    retouch_mode = "coach"
            if effective_cfg.protocol == "ollama" and self._local_dialog and self._local_dialog.busy:
                raise ApiError("busy")
            ticket = budget.reserve()
        except (ApiError, ValueError, RuntimeError) as error:
            self.coach_text.setPlainText(str(error), kind="error")
            return
        cancel = threading.Event()
        use_agent_loop = (max_turns > 1 and getattr(adapter, "supports_tools", False) and hasattr(adapter, "step"))
        progress = AgentProgress(max_turns) if use_agent_loop else TransportProgress()
        if effective_cfg.protocol == "ollama":
            progress.advance("local_preparing")
        job.coach_request = request
        local_config = self.local_config
        style_snippet = self.style_memory.prompt_snippet()
        working_snippet = self.agent_memory.prompt_snippet()

        def run_coach_call():
            if not use_agent_loop:
                return adapter.analyze(request, cancel_event=cancel, include_before=include_before, progress=progress)
            runtime = AgentRuntime(
                adapter,
                build_default_tool_registry(),
                run_budget=RunBudget(
                    max_llm_turns=max_turns,
                    max_tool_calls=min(48, max(12, max_turns * 4)),
                    max_trial_renders=min(10, max(3, max_turns - 1)),
                    max_wall_seconds=min(3600.0, max(120.0, float(effective_cfg.timeout_seconds) * max_turns)),
                    retouch_mode=retouch_mode,
                ),
                session_budget=budget,
                playbook=self.playbook,
                style_memory_snippet=style_snippet,
                working_memory_snippet=working_snippet,
                compactor=compact_conversation,
            )
            return runtime.run(
                request,
                state=job.snapshot,
                include_before=include_before,
                initial_ticket=ticket,
                render_trial=lambda values: self._run_serial_agent_tool(
                    job.snapshot, job.coach_token, cancel,
                    lambda: self.engine.render_trial_values(
                        job.snapshot, values, source_size=request.source_size, cancel=cancel),
                ),
                render_crop_100pct=lambda x, y, s, apply_current_edits=True: self._run_serial_agent_tool(
                    job.snapshot, job.coach_token, cancel,
                    lambda: self.engine.inspect_region_100pct(
                        job.snapshot, x, y, s, apply_current_edits=apply_current_edits, cancel=cancel),
                ),
                inspect_cfa=lambda x, y, p: self._run_serial_agent_tool(
                    job.snapshot, job.coach_token, cancel,
                    lambda: self.engine.inspect_raw_headroom(job.snapshot, x, y, p),
                ),
                cancel=cancel,
                progress=progress,
                stale_check=lambda: (
                    self._closing or cancel.is_set() or self.coach.paused
                    or job.generation != self._generation or job.coach_token != self._coach_generation
                ),
            )

        def analyze():
            if effective_cfg.protocol == "ollama":
                try:
                    self.local_runtime.ensure_ready(local_config, cancel)
                except Exception as exc:
                    raise ApiError("configuration", local_detail=str(exc)) from None
                if not self.local_runtime.inference_lock.acquire(blocking=False):
                    raise ApiError("busy")
                try:
                    from .ollama_client import model_supports_tools
                    if use_agent_loop and not model_supports_tools(effective_cfg.model, port=local_config.port):
                        adapter.supports_tools = False
                        return adapter.analyze(request, cancel_event=cancel, include_before=include_before, progress=progress)
                    return run_coach_call()
                finally:
                    self.local_runtime.inference_lock.release()
            return run_coach_call()

        worker = Worker(analyze)
        worker.signals.done.connect(self._network_done)
        self._network_active = (worker, job, cancel, ticket)
        self._network_budget = budget
        self._network_progress = (progress, settings.config.timeout_seconds)
        self._update_network_progress()
        self._network_timer.start()
        self._update_coach_status()
        self.network_pool.start(worker)

    def _update_network_progress(self) -> None:
        if self._network_active is None or self._network_progress is None:
            return
        _, job, cancel, _ = self._network_active
        if (self._closing or cancel.is_set() or self.coach.paused or self.coach_mode.currentIndex() != 1 or
                job.generation != self._generation or job.coach_token != self._coach_generation or
                (job.kind != "text_probe" and not self.store.accepts(job.snapshot))):
            return
        progress, limit = self._network_progress
        stage, elapsed = progress.snapshot()
        context = "Text-only test · No photo or edit data" if job.kind == "text_probe" else f"State v{job.snapshot.revision}"
        from .i18n import tr
        zh = tr("en", "zh") == "zh"
        caption = tr("Text-only test", "纯文本测试") if job.kind == "text_probe" else tr("Analyzing selected previews", "正在分析所选预览")
        local = bool(self.api_settings and self.api_settings.config.protocol == "ollama" and job.kind != "text_probe")
        if local:
            caption = (tr("Preparing local model…", "正在准备本地模型…") if stage == "local_preparing" else
                       tr("Loading local model / analyzing photo…", "正在加载本地模型或分析照片…"))
        loading_notice = tr("First use may take longer while the model loads into memory. You can cancel.",
                            "首次使用可能需要较长时间将模型加载到内存或显存，请稍候；也可取消。")
        if job.kind == "text_probe":
            self.coach_text.setPlainText(caption + f" · {elapsed:.1f} / {limit:g}s")
        elif isinstance(progress, AgentProgress):
            turn, max_turns, step_label, active_detail, spans = progress.live_trace_snapshot()
            if max_turns > 1 and turn > 0:
                caption += (f" · 第 {turn}/{max_turns} 轮" if zh else f" · Turn {turn}/{max_turns}")
            active_step, stage_hint = format_live_agent_step(
                turn, max_turns, step_label, active_detail, stage, zh=zh,
            )
            self.coach_text.show_live_progress(
                caption + f" · {elapsed:.1f} / {limit:g}s",
                spans=spans,
                active_step=active_step,
                stage_hint=stage_hint,
                notice=loading_notice if local else "",
                zh=zh,
            )
        else:
            _, stage_hint = format_live_agent_step(0, 1, "", "", stage, zh=zh)
            self.coach_text.show_live_progress(
                caption + f" · {elapsed:.1f} / {limit:g}s",
                active_step=stage_hint + ("…" if stage_hint and not stage_hint.endswith("…") else ""),
                notice=loading_notice if local else "",
                zh=zh,
            )
        self.coach_hint.setText(loading_notice if local else
                                f"{TRANSPORT_STAGES[stage].capitalize()} · {limit:g}s total connection limit · {context}")
        self._sync_coach_presentation()

    def _record_coach_advice(self, feedback: CoachFeedback, values: EditValues | None, *, spans=()) -> None:
        plan = getattr(feedback, "agent_plan", None)
        actions = tuple(getattr(feedback, "actions", ()) or ())
        self.agent_memory.record(
            feedback.photo_id,
            feedback.session_id,
            TurnSummary(
                state_version=feedback.state_version,
                question=feedback.question,
                decision=plan.decision if plan else "legacy",
                objective=plan.objective if plan else (feedback.answer or feedback.main_change)[:120],
                tools_used=tuple(
                    s.name for s in spans if getattr(s, "node", "") in ("tool_call", "trial_render")
                ),
                action_label=actions[0].label if actions else "",
                applied=bool(actions) and values is not None and all(a.is_applied(values) for a in actions),
                parameters=values.to_dict() if values is not None else None,
                proposed_actions=tuple(a.label for a in actions[:6]),
                next_step=feedback.next_step_or_stop,
                watch_stop=feedback.observe_and_stop_when,
                goal=values.goal if values is not None else "",
            ),
        )

    def _network_done(self, result, error) -> None:
        if self._network_active is None:
            return
        _, job, cancel, ticket = self._network_active
        self._network_active = None
        self._network_timer.stop()
        self._network_progress = None
        active_budget = self._network_budget or self.api_budget
        is_agent_run = isinstance(result, AgentRunResult)
        success = error is None and isinstance(
            result, TextProbeResult if job.kind == "text_probe" else (ApiResult, AgentRunResult)
        )
        if not active_budget.is_settled(ticket):
            in_tok = result.total_input_tokens if is_agent_run else (result.input_tokens if success else None)
            out_tok = result.total_output_tokens if is_agent_run else (result.output_tokens if success else None)
            active_budget.settle(ticket, in_tok if success else None, out_tok if success else None, success=success)
        self._network_budget = None
        valid = (not self._closing and not cancel.is_set() and not self.coach.paused and
                 self.coach_mode.currentIndex() == 1 and job.generation == self._generation and
                 job.coach_token == self._coach_generation and
                 (job.kind == "text_probe" or self.store.accepts(job.snapshot)))
        if valid:
            if success and job.kind == "text_probe":
                self.coach_text.setPlainText(result.text())
            elif success and (result.feedback.photo_id, result.feedback.session_id, result.feedback.state_version) == (
                    job.snapshot.photo_id, job.snapshot.session_id, job.snapshot.revision):
                if is_agent_run:
                    request = job.coach_request
                    try:
                        if request is not None:
                            self.trace_store.record_run(request.request_id, result.to_record(request))
                            export_otel_spans(request.request_id, result.spans, {
                                "turns_used": result.turns_used,
                                "tool_calls_used": result.tool_calls_used,
                            })
                    except Exception:
                        pass
                    if result.feedback.agent_plan is not None and result.feedback.agent_report is None:
                        request = job.coach_request
                        if request is None:
                            self.coach_text.setPlainText("The analysis snapshot is unavailable. Analyze again; no edit was applied.", kind="error")
                        else:
                            from .i18n import tr
                            self.coach_text.setPlainText(tr("Rendering and checking the proposed edit…", "正在生成并检查试调…"))
                            trial_cancel = threading.Event()
                            pending_feedback = replace(result.feedback, agent_trace=result.spans)
                            self._queue(Job("coach_check", self._generation,
                                lambda: self.engine.check_coach_trial(job.snapshot, request, pending_feedback, cancel=trial_cancel),
                                job.snapshot, job.coach_token, cancel=trial_cancel))
                    else:
                        self._record_coach_advice(result.feedback, job.snapshot.values, spans=result.spans)
                        self.coach_text.show_feedback(result.feedback)
                elif result.feedback.agent_plan is not None:
                    request = job.coach_request
                    if request is None:
                        self.coach_text.setPlainText("The analysis snapshot is unavailable. Analyze again; no edit was applied.", kind="error")
                    else:
                        from .i18n import tr
                        self.coach_text.setPlainText(tr("Rendering and checking the proposed edit…", "正在生成并检查试调…"))
                        trial_cancel = threading.Event()
                        self._queue(Job("coach_check", self._generation,
                            lambda: self.engine.check_coach_trial(job.snapshot, request, result.feedback, cancel=trial_cancel),
                            job.snapshot, job.coach_token, cancel=trial_cancel))
                else:
                    self._record_coach_advice(result.feedback, job.snapshot.values)
                    self.coach_text.show_feedback(result.feedback)
                if (result.live_photo_response and self.api_settings and self.api_settings.config.model
                        and self.api_settings.remember_settings):
                    try:
                        if self.model_catalog_store:
                            self.model_catalog_store.mark_model_verified(self.api_settings.config.model,
                                provider=infer_provider(self.api_settings.config), scope=credential_target(self.api_settings.config))
                    except CatalogError:
                        self.coach_catalog_status.setText('Photo reply received; local verification record could not be saved.')
            else:
                # Never display arbitrary provider/transport exception bodies.
                if isinstance(error, ApiError):
                    if self.api_settings and self.api_settings.config.protocol == "ollama":
                        message = "Local analysis failed. Open Local AI to check the engine, model verification and request limit. " + str(error)
                    elif error.code == "authentication":
                        message = "Authentication failed. Check your API key and permissions in Connection settings."
                    elif error.code == "rate_limit":
                        message = "Rate limit reached. Please wait a moment before sending another request."
                    elif error.code == "timeout":
                        if error.transport_stage:
                            message = str(error)
                        else:
                            limit = f" ({error.timeout_seconds:g}s)" if error.timeout_seconds is not None else ""
                            message = f"Request timed out{limit}. Check your internet connection or increase timeout in Connection settings."
                    elif error.http_status == 503 or error.code == "unavailable":
                        status = f"HTTP {error.http_status}: " if error.http_status else ""
                        message = f"{status}Provider service is temporarily unavailable. Your connection settings are unchanged. Try again later."
                    else:
                        message = str(error)
                else:
                    message = "Analysis failed or returned invalid feedback. No retry was made."
                if job.kind == "text_probe":
                    message = "Text-only test failed. " + message + " No photo or edit data was sent."
                self.coach_text.setPlainText(message + "\nEditing remains available.", kind="error")
        self._update_coach_status()
        if self._closing:
            QTimer.singleShot(0, self.close)

    def _auto_binding(self):
        state = self.store.current
        return None if state is None else (state.photo_id, state.session_id, state.revision, self._coach_generation)

    def _auto_edit_key(self):
        state = self.store.current
        return None if state is None else (state.session_id, replace(state.values, goal=""))

    def _cancel_auto_coach(self, *_):
        self.auto_schedule.cancel()
        self._auto_timer.stop()
        if hasattr(self, "auto_status"):
            self._update_auto_status()

    def _auto_mode_changed(self, enabled):
        self._cancel_auto_coach()
        self._auto_observed = self._auto_edit_key()
        # Invalidate a running local result when opt-out changes.
        self._invalidate_coach("Automatic local mock enabled. Make an edit to begin." if enabled else
                               "Automatic local mock off. Manual analysis remains available.", keep_reference=True)
        self._update_coach_status()

    def _auto_timing_changed(self, *_):
        self.auto_schedule.idle_seconds = self.auto_idle.value()
        self.auto_schedule.interval_seconds = self.auto_interval.value()
        self.auto_schedule.released()

    def _auto_note_edit(self):
        key = self._auto_edit_key()
        changed = key != self._auto_observed
        self._auto_observed = key
        if (changed and key is not None and self.auto_mock.isChecked()
                and self.coach_mode.currentIndex() == 0 and not self.coach.paused and not self._closing):
            self.auto_schedule.offer(self._auto_binding())
            self._auto_timer.start()
        self._update_auto_status()

    def _update_auto_status(self):
        if not self.auto_mock.isChecked():
            message = "Auto mock off"
        elif self.coach.paused:
            message = "Auto mock paused"
        elif self.auto_schedule.pending is not None:
            message = "Auto mock waiting for idle / available worker"
        elif self._active and self._active[1].kind == "coach":
            message = "Auto mock analyzing"
        else:
            message = "Auto mock ready · waiting for an image edit"
        self.auto_status.setText(message)

    def _poll_auto_coach(self):
        if (not self.auto_mock.isChecked() or self.coach_mode.currentIndex() != 0
                or self.coach.paused or self._closing or not self.store.current):
            self._cancel_auto_coach()
            return
        blocked = (self.store.gesture_active or self._active is not None or self._pending is not None
                   or self._pending_detail is not None or self._pending_backdrop is not None
                   or bool(self._protected_jobs) or self._exporting
                   or self._network_active is not None or self._reviewing or self._render_timer.isActive()
                   or self._settle_timer.isActive()
                   or not self.analyze_button.isEnabled() or QApplication.activeModalWidget() is not None
                   or self.goal.hasFocus() or self.question.hasFocus())
        if self.auto_schedule.take(self._auto_binding(), blocked=blocked):
            self._auto_timer.stop()
            if self.coach.calls >= self.coach.limit:
                self.auto_mock.setChecked(False)
                self.coach_text.setPlainText("Auto mock stopped: local analysis limit reached. Editing remains available.")
                return
            self.analyze(automatic=True)
        elif self.auto_schedule.pending is None:
            self._auto_timer.stop()
        self._update_auto_status()

    def analyze(self, *, automatic=False, follow_up: bool = False) -> None:
        if self._coach_busy() or self._reviewing or self._closing:
            return
        if automatic and (not self.auto_mock.isChecked() or self.coach_mode.currentIndex() != 0):
            return
        if (not automatic and self.coach_mode.currentIndex() == 1 and self.api_settings is None
                and not self.coach.paused and not self._exporting and not self._closing):
            self._show_coach_selection()
            return
        if (self.coach_mode.currentIndex() == 1 and self.api_settings and
                classify_model(infer_provider(self.api_settings.config), self.api_settings.config.model) == 'unsupported'):
            self.coach_text.setPlainText('This model is not suitable for photo coaching. Choose an image-analysis model. No request was sent.', kind="error")
            return
        if not self.store.current or self.coach.paused or self._exporting or not self.analyze_button.isEnabled():
            return
        want_follow_up = (not automatic) and (follow_up or self._pending_follow_up)
        self._pending_follow_up = False
        self._finish_gesture()
        self.change_goal()
        self._coach_cancel_requested = False
        self.auto_schedule.started()
        self._auto_timer.stop()
        self._render_timer.stop()
        state = self.store.current
        before = self.store.before_last_edit()
        events = self.store.project_events()
        follow_up_ctx = None
        if want_follow_up:
            prior = self.agent_memory.last_advice_for(state.photo_id, state.session_id)
            if prior is not None:
                follow_up_ctx = self.agent_memory.follow_up_context(events, state)
                if prior.parameters is not None:
                    before = replace(
                        state.with_values(EditValues.from_dict(prior.parameters)),
                        revision=prior.state_version,
                        compare_baseline=False,
                        compare_snapshot=None,
                    )
        question = self.question.text()
        token = self._coach_generation
        from .i18n import tr
        self.coach_text.setPlainText(tr("Preparing matching previews…", "正在准备匹配的预览…"))
        if self.coach_mode.currentIndex() == 1:
            if self.api_settings is None:
                if self.dev_mode:
                    self.coach_text.setPlainText("Configure an endpoint and model first. Mock remains available without a key.")
                else:
                    self.coach_text.setPlainText("Set up AI coach to ask questions. No request was sent.")
                self.request_preview()
                if not self.dev_mode:
                    self.configure_api()
                return
            self._queue(Job("api_prepare", self._generation,
                lambda: (self.engine.coach_request(state, before, events, question)
                         if follow_up_ctx is None
                         else self.engine.coach_request(state, before, events, question, follow_up=follow_up_ctx)),
                state, token))
            return
        def analyze_current():
            request = (self.engine.coach_request(state, before, events, question)
                       if follow_up_ctx is None
                       else self.engine.coach_request(state, before, events, question, follow_up=follow_up_ctx))
            return self.engine.check_coach_trial(state, request, self.coach.analyze(request))
        self._queue(Job("coach", self._generation, analyze_current, state, token, automatic=automatic))

    def closeEvent(self, event) -> None:
        self.local_controller.layer.finish()
        self.spots_controller.layer.finish()
        if not self._close_approved:
            if self.isVisible() and not self._allow_discard():
                event.ignore()
                return
            self._close_approved = True
        self.coach_response.reader.hide()
        self.coach_response.trial.hide()
        self.coach_selection.hide()
        if self._local_dialog and self._local_dialog.busy:
            self._closing = True
            self._local_dialog.reject()
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        self._stop_detail()
        self._settle_timer.stop()
        self._cancel_obsolete_render()
        self._cancel_auto_coach()
        if self._active and self._active[1].kind in ("coach_check", "agent_tool") and self._active[1].cancel is not None:
            self._active[1].cancel.set()
        self._notice_timer.stop()
        self._protected_jobs.clear()
        if self._active is not None or self._network_active is not None:
            self._closing = True
            if self._network_active is not None:
                self._network_active[2].set()
            self._pending = None
            self._pending_backdrop = None
            self._render_timer.stop()
            self._set_controls(False)
            self.open_button.setEnabled(False)
            self.open_project_button.setEnabled(False)
            self.status.setText("Waiting for the current background operation to finish before closing…")
            event.ignore()
        else:
            self._closing = True
            self._render_timer.stop()
            self.local_runtime.stop()
            super().closeEvent(event)


def configure_appearance(app: QApplication) -> None:
    """Compatibility entry point shared with visual QA and installed launchers."""
    from .theme import configure_appearance as apply_theme
    apply_theme(app)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhotoCoach offline RAW editor")
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("--dev", action="store_true",
                        help="Developer mode: expose the local mock coach (also PHOTOCOACH_DEV=1)")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    configure_appearance(app)
    window = MainWindow(dev_mode=True if args.dev else None)
    window.show()
    if args.source:
        opener = window.open_raw if is_supported_raw_suffix(args.source) else window.open_project
        QTimer.singleShot(0, lambda: opener(args.source))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
