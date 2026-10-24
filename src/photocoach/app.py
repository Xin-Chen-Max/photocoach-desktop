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
from .adjustments import render_edit, render_photo
from .composition import Composition, sample_composition
from .composition_ui import CompositionDialog
from .histogram import sampled_histogram
from .project import snapshot_project, write_project, read_project, SourceMissing
from .coach import MockCoach, CoachFeedback, build_request
from .devmode import dev_mode_enabled
from .auto_coach import AutoCoachSchedule
from .api_coach import ApiCoach, ApiError, ApiResult, TransportProgress, TRANSPORT_STAGES
from .budget import SessionBudget
from .coach_ui import ApiSettingsDialog, SendReviewDialog, TextProbeReviewDialog, budget_text
from .coach_view import usage_summary
from .settings import SettingsStore, SettingsError, resolve_api_key, credential_target
from .connections import ConnectionStore, settings_for, ConnectionBook, infer_provider, connection_from_settings
from .model_catalog import ModelCatalogStore, classify_model, capability_evidence, CatalogError
from .credentials import WindowsCredentialStore
from .text_probe import run_text_probe, TextProbeResult
from .white_balance import balance_coordinates, balance_gains, sample_white_balance
from .preview import (reduce_linear, oriented_size, DetailRequest, render_region,
                      LINEAR_CACHE_BYTES, SMALL_RESERVE_BYTES)
from .detail_cache import DetailGeometryCache
from .render_timing import StageTimer, timed
from .gradients import RenderGeometry, OverlaySpec, overlay_pixels
from .safe_io import atomic_write, sha256_file, validate_destination


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


TIMED_JOB_KINDS = ("preview", "detail", "adjusting")
ADJUSTING_MAX_EDGE = 960


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
                options={} if not any(g.active for g in values.local_gradients) else {
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
        return replace(photo, metadata=self._display_metadata(photo, EditValues())), quantize_srgb(render_srgb(small))

    def render(self, state: ProbeState):
        options = RawOptions() if state.compare_baseline else state.raw_options
        photo, small = self.photo(state.source_path, options)
        if photo.metadata["sha256"] != state.photo_id:
            raise RuntimeError("The source file has changed. Reopen the photo.")
        values = EditValues(composition=state.composition) if state.compare_baseline else state.values
        started = time.perf_counter()
        pixels = self._render_small(photo, small, values, cancel=self.cancel, timer=self.timer)
        metadata = self._display_metadata(photo, values)
        if self.overlay_spec is not None:
            self._overlay(metadata,pixels,RenderGeometry.full(oriented_size(photo.metadata),values.composition,(pixels.shape[1],pixels.shape[0])))
        return pixels, time.perf_counter() - started, metadata

    def detail(self, state: ProbeState, request: DetailRequest):
        options = RawOptions() if state.compare_baseline else state.raw_options
        started = time.perf_counter()
        size = request.source_size or request.full_size
        photo = self.cache.get(state.source_path, options, half_size=False,
                               expected_bytes=size[0] * size[1] * 12,
                               before_decode=self._clear_derived)
        if photo.metadata["sha256"] != state.photo_id:
            raise RuntimeError("The source file has changed. Reopen the photo.")
        values = EditValues(composition=state.composition) if state.compare_baseline else state.values
        small_bytes = 0 if self._small is None else self._small.nbytes
        temporary_bytes = 0 if self._adjusting_pixels is None else self._adjusting_pixels.nbytes
        with timed(self.timer, "geometry"):
            region = self._geometry.region(
                photo.linear_rgb, (state.photo_id, photo.options, photo.half_size),
                values.composition, request.rect,
                max_bytes=LINEAR_CACHE_BYTES - small_bytes - photo.linear_rgb.nbytes - temporary_bytes,
                reserve=lambda needed: self.cache.trim_to(LINEAR_CACHE_BYTES - small_bytes - needed - temporary_bytes))
        pixels = render_region(photo.linear_rgb, values, request, composed_region=region, cancel=self.cancel,
                               timer=self.timer)
        metadata = self._display_metadata(photo, values)
        if self.overlay_spec is not None:
            self._overlay(metadata,pixels,RenderGeometry((photo.linear_rgb.shape[1],photo.linear_rgb.shape[0]),values.composition,request.rect,(pixels.shape[1],pixels.shape[0])))
        return pixels, time.perf_counter() - started, metadata

    def adjusting(self, state, request):
        """Display-only pre-tone reduction. Never used for exports or coaching."""
        started = time.perf_counter()
        options = RawOptions() if state.compare_baseline else state.raw_options
        warm = self.cache.full_identity == (state.photo_id, options)
        photo = self.cache.get(state.source_path, options, half_size=not warm,
                               before_decode=self._clear_derived)
        if photo.metadata["sha256"] != state.photo_id:
            raise RuntimeError("The source file has changed. Reopen the photo.")
        values = EditValues(composition=state.composition) if state.compare_baseline else state.values
        edge = ADJUSTING_MAX_EDGE if request.adjusting_edge is None else request.adjusting_edge
        if type(edge) is not int or not 160 <= edge <= 1280:
            raise ValueError("Invalid adjustment preview resolution.")
        key = (state.photo_id, options, photo.half_size, id(photo.linear_rgb),
               state.composition, request.full_size, request.rect, edge)
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
                        target_size=(width, bottom-top))
            pixels.setflags(write=False)
            self._adjusting_pixels, self._adjusting_key = pixels, key
        geometry=RenderGeometry(tuple(request.source_size or request.full_size),values.composition,
                                request.rect,(self._adjusting_pixels.shape[1],self._adjusting_pixels.shape[0]))
        rendered = render_edit(self._adjusting_pixels, values, cancel=self.cancel, timer=self.timer,geometry=geometry)
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

    def coach_request(self, state, before, events, question):
        images = []
        for snapshot in (before, state):
            photo, small = self.photo(snapshot.source_path, snapshot.raw_options)
            if photo.metadata["sha256"] != state.photo_id:
                raise RuntimeError("The source changed. Reopen it before analysis.")
            images.append(self._render_small(photo, small, snapshot.values, 768))
        return build_request(state, before, events, *images, question)

    @staticmethod
    def _render_small(photo, small, values, max_edge=1600, *, cancel=None, timer=None):
        if values.composition == Composition():
            with timed(timer, "reduction"):
                reduced = reduce_linear(small, max_edge)
            rendered = render_edit(reduced, values, cancel=cancel, timer=timer,
                geometry=RenderGeometry.full(oriented_size(photo.metadata),values.composition,(reduced.shape[1],reduced.shape[0])))
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
    def __init__(self, *, settings_store=None, credentials=None, dev_mode: bool | None = None) -> None:
        super().__init__()
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
        self._refining_preview = False
        self._coach_unread = False
        self._review_settings = None
        self._review_include_before = True
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
        self.network_pool = QThreadPool(self)
        self.network_pool.setMaxThreadCount(1)
        self._network_active = None
        self._network_progress = None
        self._network_timer = QTimer(self)
        self._network_timer.setInterval(250)
        self._network_timer.timeout.connect(self._update_network_progress)
        self._reviewing = False
        self._coach_generation = 0
        self._expanded_coach_height = 285
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
        self._build_ui()
        self._notice_timer.timeout.connect(self.local_controller.show_notice)
        self.coach_collapse.setChecked(True)
        self.coach_text.set_question_draft(self.question.text())
        for slider in (self.exposure_slider, *self.adjustment_sliders.values(), *self.color_mixer.sliders,*self.local_gradients.sliders.values(),self.local_gradients.feather):
            slider.installEventFilter(self)
        if not self.dev_mode:
            # Normal launches always use the real AI coach; the mock selector is hidden.
            for selector in (self.coach_mode, self.coach_mode_picker):
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
        for widget in (self.exposure_slider, self.wb_mode, self.apply_wb, self.compare, self.edited_button,
                       self.fit_button, self.actual_size_button, self.composition_button,
                       self.export_button, self.report_button, self.save_project_button,
                       self.reset_button, self.curve, self.reset_curve_button, self.color_mixer, self.local_gradients,self.clipping,self.goal, self.question,
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
        if enabled:self.local_gradients.edit_on_photo.setChecked(False)
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
        path, _ = QFileDialog.getOpenFileName(self, "Open Sony RAW (read-only)", "", "Sony RAW (*.ARW *.arw)")
        if path:
            self.open_raw(Path(path))

    def open_raw(self, path: Path) -> None:
        self._formal_preview_session = None
        self.local_controller.layer.finish()
        self.local_gradients.edit_on_photo.setChecked(False)
        self._cancel_obsolete_render()
        self._interactive_preview = False
        self._gesture_preview_edge = None
        self._refining_preview = False
        self._settle_timer.stop()
        self.store.end_drag()
        self._opening_project = None
        self._invalidate_coach("A new photo is opening.")
        self._generation += 1
        self._render_timer.stop()
        self._stop_detail()
        self._pending = None
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
            self._cancel_obsolete_render()
        if self._active is not None:
            if job.kind in ("detail", "preview", "adjusting"):
                # All render intents share one latest slot, not a per-quality backlog.
                self._pending = None
                self._pending_detail = None
            if job.kind == "detail":
                self._pending_detail = job
            elif job.kind in ("preview", "adjusting"):
                self._pending = job
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
        if job.kind in ("preview", "detail", "adjusting", "coach", "api_prepare", "composition_prepare", "wb_pick"):
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
        if job.kind == "coach":
            valid = valid and job.coach_token == self._coach_generation and not self.coach.paused
            if job.automatic:
                valid = valid and self.auto_mock.isChecked() and self.coach_mode.currentIndex() == 0
            if error is None and not isinstance(result, CoachFeedback):
                error = ValueError("The coach returned invalid feedback.")
            if error is None:
                valid = valid and (result.photo_id, result.session_id, result.state_version) == (
                    job.snapshot.photo_id, job.snapshot.session_id, job.snapshot.revision)
        if valid and not self._closing:
            if error is not None:
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
                    self.coach_text.setPlainText("Open failed. You can retry or continue with the previous photo.")
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
                elif job.kind == "coach":
                    if job.automatic:
                        self.auto_mock.setChecked(False)
                    self.coach_text.setPlainText(f"Mock analysis failed: {error}\nEditing remains available.")
                elif job.kind == "api_prepare":
                    self.coach_text.setPlainText("Unable to prepare matching previews. Render the current edit and try again.")
            elif job.kind == "load":
                photo, pixels = result
                state = self.store.activate(photo.source_path, photo.metadata["photo_id"])
                self._project_path = None
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
            elif job.kind == "adjusting":
                pixels, elapsed, metadata = result
                presented = time.perf_counter()
                self.preview.display_adjusting(pixels, job.view_request)
                self.preview.accept_overlay(metadata.pop('display_overlay',None))
                self._record_render_details(job, elapsed, presentation=time.perf_counter() - presented)
            elif job.kind == "project_load":
                self.store, pixels, metadata = result
                self._project_path = self._opening_project
                self.store.mark_preview(self.store.current)
                self.filename.setText(self.store.current.source_path.name)
                self.preview.configure_photo(oriented_size(metadata))
                self.preview.set_composition(self.store.current.composition)
                self.preview.display(pixels)
                self._show_metadata(metadata)
                self._sync_controls()
                self.open_button.setEnabled(True)
                self.open_project_button.setEnabled(True)
                self._set_controls(True)
                self.status.setText("Project restored, including edit history and undo/redo.")
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
            elif job.kind == "coach":
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
                self.local_controller.show_notice()
        if job.kind in ("coach", "api_prepare"):
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
            else:
                pending, self._pending_detail = self._pending_detail, None
            if pending is None:
                break
            if pending.generation != self._generation:
                continue
            if pending.kind in ("preview", "detail", "adjusting", "coach", "api_prepare", "composition_prepare") and not self.store.accepts(pending.snapshot):
                continue
            if pending.kind in ("detail", "adjusting") and pending.view_request.generation != self.preview.view_generation:
                continue
            self._queue(pending)
            break
        self._update_coach_status()
        if self._closing:
            QTimer.singleShot(0, self.close)

    def _show_metadata(self, metadata: dict) -> None:
        self._metadata = metadata
        state = self.store.current
        self.histogram.display(metadata.get("display_histogram"), baseline=bool(state and state.compare_baseline))
        self._metadata_binding = None if state is None else {
            "photo_id": state.photo_id, "session_id": state.session_id,
            "preview_revision": self.store.preview_revision,
            "is_baseline": state.compare_baseline,
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
        self._render_diagnostics = {"kind": job.kind, "revision": job.snapshot.revision,
            "elapsed_ms": round(elapsed * 1000, 1), "baseline": job.snapshot.compare_baseline,
            "view_generation": self.preview.view_generation, "zoom": self.preview.zoom,
            "dpr": self.preview.devicePixelRatioF()}
        # Detailed timing stays in diagnostics; stage totals are summed worker
        # time across strips and can exceed elapsed wall time.
        if job.timing:
            self._render_diagnostics.update(job.timing)
        if presentation is not None:
            self._render_diagnostics["presentation_ms"] = round(presentation * 1000, 1)
        self.render_details.setText(
            f"{'Full detail' if job.kind == 'detail' else 'Quick preview'} · v{job.snapshot.revision}\n"
            f"Worker time: {elapsed * 1000:.0f} ms\n"
            f"{'Baseline' if job.snapshot.compare_baseline else 'Edited'} · "
            f"{self.preview.zoom * 100:.0f}% · DPR {self.preview.devicePixelRatioF():g}")

    def _show_preview_details(self):
        self.preview_details.show()
        self.preview_details.raise_()

    def _sync_comparison_buttons(self):
        baseline = bool(self.store.current and self.store.current.compare_baseline)
        self.compare.blockSignals(True)
        self.compare.setChecked(baseline)
        self.compare.blockSignals(False)
        self.edited_button.setChecked(not baseline)

    def _ensure_edited_view(self):
        if self.store.current and self.store.current.compare_baseline:
            # Do this before begin_drag: set_compare ends an existing transaction.
            self.store.set_compare(False)
            self._sync_comparison_buttons()
            self.view_notice.setText("Showing edited photo")
            self._notice_timer.start()

    def _begin_parameter_gesture(self, name):
        if self.store.current and not self._exporting:
            self._ensure_edited_view()
            self.store.begin_drag(name)
            self._gesture_preview_edge = ADJUSTING_MAX_EDGE
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
            self._interactive_preview = False
            self._gesture_preview_edge = None
            self._settle_timer.stop()
            self.store.set_compare(checked)
            self._sync_comparison_buttons()
            self._invalidate_coach(keep_reference=True)
            self.request_preview()

    def request_preview(self, *, interactive=False) -> None:
        if self.store.current and not self._exporting and not self._closing:
            if not interactive:
                self._interactive_preview = False
                self._settle_timer.stop()
                self._refining_preview |= self._has_formal_preview(self.store.current)
            self._stop_detail()
            self.preview.set_composition(self.store.current.composition)
            self.preview.invalidate_edit("Adjusting" if interactive else "Refining")
            self.histogram.mark_updating()
            state = self.store.current
            self._show_version()
            if interactive:
                if self._gesture_preview_edge is None:
                    self._gesture_preview_edge = ADJUSTING_MAX_EDGE
                request = replace(self.preview.detail_request(), adjusting_edge=self._gesture_preview_edge)
                self.preview.set_quality("Adjusting")
                self._queue(Job("adjusting", self._generation, lambda: self.engine.adjusting(state, request),
                                state, view_request=request))
                return
            options = RawOptions() if state.compare_baseline else state.raw_options
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
        self._cancel_obsolete_render()
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
        options = RawOptions() if state.compare_baseline else state.raw_options
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
        self._freeze_output("Exporting full resolution using the current parameters. Baseline view does not change the export.")

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
        was_dragging = self.store.gesture_active
        self.curve.finish()
        self.store.end_drag()
        self._refresh_history()
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
        self._settle_timer.stop()
        if was_interactive:
            self._refining_preview = True
            self._render_timer.stop()
            self.request_preview()

    def _invalidate_coach(self, message="Edits changed. Analyze again for feedback on the current state.", *, keep_reference=False, clear_auto=True) -> None:
        if clear_auto:
            self._cancel_auto_coach()
        self._coach_generation += 1
        if self._network_active is not None:
            self._network_active[2].set()
        state = self.store.current
        retained = (keep_reference and state is not None and
                    self.coach_text.keep_as_reference(state.photo_id, state.session_id, state.revision))
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
        self._settle_timer.stop()
        if interactive:
            self._settle_timer.start()
        self._stop_detail()
        if self.store.current:
            self.preview.set_composition(self.store.current.composition)
            self.local_controller.sync()
        self.preview.invalidate_edit("Adjusting" if interactive else "Refining")
        self.histogram.mark_updating()
        self._invalidate_coach(keep_reference=True, clear_auto=False)
        self._auto_note_edit()
        self._show_version()
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _cancel_obsolete_render(self):
        if self._active is not None:
            _, active = self._active
            if active.cancel is not None:
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

    def _question_changed(self, text: str) -> None:
        self.coach_text.set_question_draft(text)
        action = "Review and ask coach" if self.dev_mode else "Ask coach"
        self._invalidate_coach(f"Question changed. Click {action} to send it.", keep_reference=True)

    def change_parameter(self, name, value, *, sync=False) -> None:
        if not self.store.current or self._exporting:
            return
        if name != "goal":
            self._ensure_edited_view()
        self.store.set_parameter(name, value)
        if name in self.adjustment_labels:
            self.adjustment_labels[name].setText(f"{name.title()}  {value:.2f}")
        if sync:
            self._sync_controls()
        self._after_edit(interactive=not sync and name in ("contrast", "saturation", "highlights", "shadows", "whites", "blacks", "vibrance", "rgb_curve", "color_mixer", "local_gradients"))

    def change_goal(self) -> None:
        if self.store.current and not self._exporting and self.goal.text() != self.store.current.goal:
            self.change_parameter("goal", self.goal.text())

    def _sync_controls(self) -> None:
        state = self.store.current
        if state is None:
            return
        widgets = [self.exposure_slider, self.compare, self.wb_mode, self.goal,
                   *self.gains, *self.adjustment_sliders.values()]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.exposure_slider.setValue(round(state.exposure_ev * 100))
            self.exposure_label.setText(f"Exposure  {state.exposure_ev:+.2f} EV")
            self.compare.setChecked(state.compare_baseline)
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
            self.local_controller.sync()
            self.goal.setText(state.goal)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._wb_form_enabled()
        self._sync_comparison_buttons()
        self._refresh_history()

    def _refresh_history(self) -> None:
        if not hasattr(self, "history_list"):
            return
        enabled = self.exposure_slider.isEnabled() and not self._exporting
        self.undo_button.setEnabled(enabled and bool(self.store.undo_stack))
        self.redo_button.setEnabled(enabled and bool(self.store.redo_stack))
        self.history_list.clear()
        events = self.store.events[self.store._event_start:]
        meaningful = [event for event in events if event.get("history_entry") or
                      event["type"] in ("undo", "redo", "compare", "photo_switch", "project_open")]
        for event in meaningful[-100:]:
            self.history_list.addItem(f"{event['sequence']} · v{event['state_version']} · {event['type']} · {event['parameter'] or ''}")
        self.history_list.scrollToBottom()
        self.history_summary.setText(f"{len(events)} audit events · {len(self.store.undo_stack)} undo / {len(self.store.redo_stack)} redo\n"
                                     + ("Unsaved changes" if self.store.dirty else "Saved state"))
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
        self._cancel_obsolete_render()
        self._interactive_preview = False
        self._settle_timer.stop()
        self._finish_gesture()
        self._gesture_preview_edge = None
        self._refining_preview = False
        self._opening_project = path
        self._generation += 1
        self._render_timer.stop()
        self._stop_detail()
        self._pending = None
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
        path, _ = QFileDialog.getOpenFileName(self, "Locate the project's original ARW (hash must match)", "", "Sony RAW (*.ARW *.arw)")
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
        self.coach.paused = paused
        self._invalidate_coach("Coach paused." if paused else "Coach ready. Analysis is manual.")
        self.pause_coach.setText("Resume coach" if paused else "Pause coach")
        self._update_coach_status()

    def configure_api(self, checked=False, *, new_protocol=None) -> None:
        if self._network_active is not None or self._reviewing:
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
            self.coach_text.setPlainText(str(error) + ' Your changes are available in this window only.')

    def _clear_model_prices(self):
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
        if isinstance(choice, str) and choice.startswith('new:'):
            protocol = {'gemini': 'gemini', 'openai': 'responses', 'custom': 'chat_completions'}[choice[4:]]
            self.configure_api(new_protocol=protocol)
            self._update_coach_status()
            return
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
        self._invalidate_coach('Connection changed. No request was sent. Review and ask when ready.', keep_reference=True)
        self._persist_connection_book()
        self._update_coach_status()

    def _remember_coach_models(self, models=()):
        if self.api_settings is None:
            return {}
        config = self.api_settings.config
        provider, scope = infer_provider(config), credential_target(config)
        stored = self.model_catalog_store.get_models(provider, scope) if self.model_catalog_store else ()
        catalog = {m.id: m.id for m in stored if m.capability != 'unsupported'}
        extra = self._coach_model_catalogs.setdefault(scope, {})
        for mid in models:
            if classify_model(provider, mid) != 'unsupported':
                extra[mid] = mid
        catalog.update(extra)
        connection = self.connection_book.active_connection if self.connection_book else None
        if connection and credential_target(connection.config) == scope:
            for mid in (*connection.favorites, *connection.recent):
                if classify_model(provider, mid) != 'unsupported':
                    catalog[mid] = mid
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
            self.coach_text.setPlainText(str(error))
            return
        if self.connection_book and self.connection_book.active:
            self.connection_book = self.connection_book.with_model(self.connection_book.active, model)
        self.api_settings = replace(self.api_settings, config=config)
        self._clear_model_prices()
        self.coach_model_picker.setEditText(model)
        if self.connection_book and self.connection_book.active:
            self._session_settings[self.connection_book.active] = self.api_settings
        self._review_settings = None
        self._invalidate_coach('Model changed. No request was sent. Review and ask when ready.', keep_reference=True)
        self._persist_connection_book()
        self._update_coach_status()

    def _choose_typed_model(self):
        self._select_coach_model(self.coach_model_picker.currentText().strip())

    def _choose_coach_model(self, index):
        model = self.coach_model_picker.itemData(index)
        if model is None:
            # This row is an action, never a manual model-name draft.
            if self.api_settings is not None:
                self.coach_model_picker.setEditText(self.api_settings.config.model)
            self.configure_api()
            if self.api_settings is not None:
                self.coach_model_picker.setEditText(self.api_settings.config.model)
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
        self.coach_mode_picker.blockSignals(True)
        self.coach_mode_picker.setCurrentIndex(self.coach_mode.currentIndex())
        self.coach_mode_picker.blockSignals(False)
        provider = infer_provider(self.api_settings.config) if self.api_settings else "gemini"
        picker = self.coach_provider_picker
        picker.blockSignals(True)
        connections = self.connection_book.connections if self.connection_book else ()
        choices = [(c.label + ' · ' + (urlsplit(c.config.endpoint).hostname or ''), c.id) for c in connections]
        choices += [('Add Google Gemini…', 'new:gemini'), ('Add OpenAI…', 'new:openai'), ('Add compatible service…', 'new:custom')]
        if [picker.itemData(i) for i in range(picker.count())] != [value for _, value in choices]:
            picker.clear()
            for label, value in choices:
                picker.addItem(label, value)
                connection = self.connection_book.get(value) if self.connection_book else None
                if connection:
                    picker.setItemData(picker.count()-1, connection.config.endpoint, Qt.ItemDataRole.ToolTipRole)
        if self.connection_book and self.connection_book.active:
            picker.setCurrentIndex(max(0, picker.findData(self.connection_book.active)))
        picker.blockSignals(False)
        picker.setEnabled(cloud and self._network_active is None and not self._reviewing)
        catalog = self._remember_coach_models()
        active_conn = self.connection_book.active_connection if self.connection_book else None
        favorites = set(active_conn.favorites if active_conn else ())
        recent = list(active_conn.recent if active_conn else ())
        config = self.api_settings.config if self.api_settings else None
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
            prefix = '[Favorite] ' if mid in favorites else 'Recent · ' if mid in recent else ''
            rows.append((f'{prefix}{mid} · {tag}', mid))
        rows.append(('Choose / load more models…', None))
        draft = self.coach_model_picker.currentText() if self.coach_model_picker.lineEdit().hasFocus() else None
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
            self.coach_model_picker.setCurrentIndex(self.coach_model_picker.findData(config.model))
            self.coach_model_picker.setEditText(config.model)
        if draft is not None:
            self.coach_model_picker.setEditText(draft)
        self.coach_model_picker.blockSignals(False)
        self.coach_model_picker.setEnabled(cloud and config is not None and self._network_active is None and not self._reviewing)
        self.coach_favorite.setEnabled(self.coach_model_picker.isEnabled() and active_conn is not None)
        self.coach_favorite.setText('Favorited' if config and config.model in favorites else 'Favorite')
        self.coach_favorite.setChecked(bool(config and config.model in favorites))
        updated = self.model_catalog_store.get_last_updated(provider, scope) if self.model_catalog_store and config else None
        self.coach_catalog_status.setText(('Models updated ' + updated[:19].replace('T', ' ') + ' UTC') if updated else 'Models from documentation · Account access not checked')
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
        self.analyze_button.setText((("Send to coach" if direct else "Review and ask coach…" if self.dev_mode else "Ask coach")
                                     if self.api_settings is not None else "Set up AI coach") if cloud else "Try local demo")
        if self.api_settings:
            config = self.api_settings.config
            provider = {"gemini": "Google Gemini", "responses": "Responses API",
                        "chat_completions": "Compatible API"}[config.protocol]
            source = {"saved": "Saved key selected", "environment": "Environment key selected",
                      "session": "Session key selected"}[self.api_settings.key_source]
            prefix = "" if cloud else "Saved connection · " if self.api_settings.remember_settings else "Available connection · "
            self.coach_connection.setText(f"{prefix}{provider} · {config.model}\n{source}")
        else:
            self.coach_connection.setText(("No AI connection yet · Set up AI coach" if not self.dev_mode
                                           else "Choose a model in Connection…") if cloud else "Local parameter rules · No model connected")
        self.coach_hint.setText((f"Direct send this session · {'Before + current' if self._review_include_before else 'Current only'} · May incur charges" if direct else "Review previews before sending · Manual only") if cloud else
                               "Local rules only; the demo cannot judge photo quality.")
        self.mock_limit_row.setVisible(not cloud)
        self.text_probe_button.setVisible(bool(cloud and self.api_settings and self.api_settings.config.protocol == "gemini"))
        self.coach_model_status.setText(
            f"{self.api_settings.config.model} · Manual" if cloud and self.api_settings else
            ("Manual · Set up connection" if self.dev_mode else "Set up AI coach") if cloud else "Local mock · No cloud")
        self.coach_model_status.setToolTip(self.coach_connection.text())
        self.coach_usage.setText(usage_summary(self.api_budget, self.coach.calls, cloud))
        self.analyze_button.setEnabled((self.exposure_slider.isEnabled() or (cloud and self.api_settings is None)) and not self._exporting and
            not self.coach.paused and not self._closing and self._network_active is None and not self._reviewing and
            not any(job.kind in ("coach", "api_prepare") for job in self._protected_jobs) and
            not (self._active and self._active[1].kind in ("coach", "api_prepare")))
        reason = ("Resume coach to ask a question." if self.coach.paused else
                  "Set up a connection. No request is sent by saving settings." if cloud and self.api_settings is None else
                  "Open a photo and wait for its preview." if not self.exposure_slider.isEnabled() else
                  "Wait for the export to finish." if self._exporting else
                  "Wait for the current request or review to finish." if not self.analyze_button.isEnabled() else
                  "Local demo uses rules, not an AI model. Choose AI coach in Mode above." if not cloud else
                  "Review the current photo and send your question to the selected model.")
        self.analyze_button.setToolTip(reason)
        self.text_probe_button.setEnabled(bool(cloud and self.api_settings and
            self.api_settings.config.protocol == "gemini" and not self._exporting and not self.coach.paused and
            not self._closing and self._network_active is None and not self._reviewing))
        if self.api_budget.attempts or cloud:
            self.coach_count.setText((f"Mock analyses: {self.coach.calls}\n" if self.dev_mode else "") + budget_text(self.api_budget))
        else:
            self.coach_count.setText(f"Mock analyses: {self.coach.calls} · API attempts: 0 · Cost: $0")

    def _show_coach_details(self) -> None:
        self.coach_details.show()
        self.coach_details.raise_()

    def restore_send_confirmation(self):
        self._review_settings = None
        self._update_coach_status()

    def _set_coach_collapsed(self, collapsed: bool) -> None:
        sizes = self.workspace_splitter.sizes()
        total = sum(sizes)
        self.coach_selection_row.setVisible(not collapsed)
        for widget in (self.pause_coach, self.coach_menu_button, self.coach_model_status):
            widget.setVisible(not collapsed)
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
            for field in (self.goal, self.question):
                field.setMinimumHeight(field.sizeHint().height())
            self.coach_panel.setMaximumHeight(16777215)
            self.coach_body.show()
            self.workspace_splitter.setSizes([max(300, total - self._expanded_coach_height), self._expanded_coach_height])
        self._coach_toggle_text()

    def _coach_feedback_changed(self, available):
        self._coach_unread = bool(available and self.coach_collapse.isChecked())
        self._coach_toggle_text()

    def _coach_toggle_text(self):
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
        def reveal(widget, *, in_editor=True):
            self._navigation_target = widget
            def after_layout():
                if self._closing or self._navigation_target is not widget:
                    return
                widget.setFocus(Qt.FocusReason.OtherFocusReason)
                if in_editor:
                    self.edit_scroll.ensureWidgetVisible(widget, 0, 30)
            # Expanding a group posts a scroll-area resize. Scroll only after
            # that layout request, so a newly visible control is actually reached.
            QTimer.singleShot(0, self, after_layout)
        if name != "composition":
            section = (self.color_section if name.startswith("color_mixer") or name in
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
        targets = {"exposure_ev": (self.exposure_slider, self.exposure_label),
                   "white_balance": (self.wb_mode, self.wb_form.labelForField(self.wb_mode)),
                   "contrast": (self.adjustment_sliders["contrast"], self.adjustment_labels["contrast"]),
                   "saturation": (self.adjustment_sliders["saturation"], self.adjustment_labels["saturation"]),
                   "rgb_curve": (self.curve, self.curve_label)}
        for parameter in ("highlights", "shadows", "whites", "blacks", "vibrance"):
            targets[parameter] = (self.adjustment_sliders[parameter], self.adjustment_labels[parameter])
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

    def test_text_only(self) -> None:
        if (not self.api_settings or self.api_settings.config.protocol != "gemini" or
                self.coach_mode.currentIndex() != 1 or self.coach.paused or self._closing or
                self._network_active is not None or self._reviewing or self._exporting):
            return
        if self.api_budget.blocked_reason:
            self.coach_text.setPlainText(self.api_budget.blocked_reason)
            return
        settings = self.api_settings
        job = Job("text_probe", self._generation, lambda: None, coach_token=self._coach_generation)
        self._reviewing = True
        self._update_coach_status()
        try:
            dialog = TextProbeReviewDialog(self, settings.config, self.api_budget)
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            dialog.deleteLater()
        finally:
            self._reviewing = False
            self._update_coach_status()
        if not accepted:
            self.coach_text.setPlainText("Text test cancelled before sending. No API attempt was made.")
            return
        if (self._closing or self.coach.paused or self._exporting or self.api_settings is not settings or
                self.coach_mode.currentIndex() != 1 or job.coach_token != self._coach_generation or
                job.generation != self._generation or self._network_active is not None):
            self.coach_text.setPlainText("Settings or session changed during review. No text test was sent.")
            return
        try:
            key = resolve_api_key(settings, self.credentials)
            ApiCoach(settings.config, key)  # Validate credentials before reserving an attempt.
            ticket = self.api_budget.reserve()
        except (ApiError, ValueError, RuntimeError) as error:
            self.coach_text.setPlainText(str(error))
            return
        cancel, progress = threading.Event(), TransportProgress()
        worker = Worker(lambda: run_text_probe(settings.config, key, cancel, progress=progress))
        worker.signals.done.connect(self._network_done)
        self._network_active = (worker, job, cancel, ticket)
        self._network_progress = (progress, settings.config.timeout_seconds)
        self._update_network_progress()
        self._network_timer.start()
        self._update_coach_status()
        self.network_pool.start(worker)

    def review_api_request(self, request, job) -> None:
        if (self._closing or self._network_active or self._reviewing or not self.api_settings or
                self.coach_mode.currentIndex() != 1 or self.coach.paused or self._exporting or
                job.coach_token != self._coach_generation or not self.store.accepts(job.snapshot)):
            return
        if self.api_budget.blocked_reason:
            self.coach_text.setPlainText(self.api_budget.blocked_reason)
            return
        settings = self.api_settings
        skip_session = False
        self._reviewing = True
        self._update_coach_status()
        try:
            if self._review_settings is settings:
                accepted = True
                include_before = self._review_include_before
            else:
                dialog = SendReviewDialog(self, request, settings.config, self.api_budget)
                accepted = dialog.exec() == QDialog.DialogCode.Accepted
                include_before = dialog.include_before.isChecked()
                skip_session = dialog.skip_session.isChecked()
                dialog.deleteLater()
        finally:
            self._reviewing = False
            self._update_coach_status()
        if not accepted:
            self.coach_text.setPlainText("Request cancelled before sending. No API attempt was made.")
            return
        if (self._closing or self.coach.paused or self._exporting or self.api_settings is not settings or
                self.coach_mode.currentIndex() != 1 or job.coach_token != self._coach_generation or
                not self.store.accepts(job.snapshot)):
            self.coach_text.setPlainText("The edit or question changed during review. Analyze again; nothing was sent.")
            return
        if skip_session:
            self._review_settings = settings
            self._review_include_before = include_before
        try:
            # Credentials are read only for a user-confirmed request, never logged or serialized.
            key = resolve_api_key(settings, self.credentials)
            adapter = ApiCoach(settings.config, key)
            ticket = self.api_budget.reserve()
        except (ApiError, ValueError, RuntimeError) as error:
            self.coach_text.setPlainText(str(error))
            return
        cancel = threading.Event()
        progress = TransportProgress()
        worker = Worker(lambda: adapter.analyze(request, cancel_event=cancel, include_before=include_before,
                                               progress=progress))
        worker.signals.done.connect(self._network_done)
        self._network_active = (worker, job, cancel, ticket)
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
        self.coach_text.setPlainText(f"{TRANSPORT_STAGES[stage].capitalize()}…\n"
            f"{elapsed:.1f}s elapsed · {limit:g}s total connection limit · {context}\n"
            "You can continue editing; changed-state results are discarded. No automatic retries.")

    def _network_done(self, result, error) -> None:
        if self._network_active is None:
            return
        _, job, cancel, ticket = self._network_active
        self._network_active = None
        self._network_timer.stop()
        self._network_progress = None
        success = error is None and isinstance(result, TextProbeResult if job.kind == "text_probe" else ApiResult)
        self.api_budget.settle(ticket, result.input_tokens if success else None,
                               result.output_tokens if success else None, success=success)
        valid = (not self._closing and not cancel.is_set() and not self.coach.paused and
                 self.coach_mode.currentIndex() == 1 and job.generation == self._generation and
                 job.coach_token == self._coach_generation and
                 (job.kind == "text_probe" or self.store.accepts(job.snapshot)))
        if valid:
            if success and job.kind == "text_probe":
                self.coach_text.setPlainText(result.text())
            elif success and (result.feedback.photo_id, result.feedback.session_id, result.feedback.state_version) == (
                    job.snapshot.photo_id, job.snapshot.session_id, job.snapshot.revision):
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
                    if error.code == "authentication":
                        message = "Authentication failed. Check your API key and permissions in Connection settings."
                    elif error.code == "rate_limit":
                        message = "Rate limit reached. Please wait a moment before sending another request."
                    elif error.code == "timeout":
                        if error.transport_stage:
                            message = str(error)
                        else:
                            message = f"Request timed out ({error.timeout_seconds:g}s). Check your internet connection or increase timeout in Connection settings."
                    elif error.http_status == 503 or error.code == "unavailable":
                        status = f"HTTP {error.http_status}: " if error.http_status else ""
                        message = f"{status}Provider service is temporarily unavailable. Your connection settings are unchanged. Try again later."
                    else:
                        message = str(error)
                else:
                    message = "Analysis failed or returned invalid feedback. No retry was made."
                if job.kind == "text_probe":
                    message = "Text-only test failed. " + message + " No photo or edit data was sent."
                self.coach_text.setPlainText(message + "\nEditing remains available.")
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
                   or self._pending_detail is not None or bool(self._protected_jobs) or self._exporting
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

    def analyze(self, *, automatic=False) -> None:
        if automatic and (not self.auto_mock.isChecked() or self.coach_mode.currentIndex() != 0):
            return
        if (not automatic and self.coach_mode.currentIndex() == 1 and self.api_settings is None
                and not self.coach.paused and not self._exporting and not self._closing):
            self.configure_api()
            return
        if (self.coach_mode.currentIndex() == 1 and self.api_settings and
                classify_model(infer_provider(self.api_settings.config), self.api_settings.config.model) == 'unsupported'):
            self.coach_text.setPlainText('This model is not suitable for photo coaching. Choose an image-analysis model. No request was sent.')
            return
        if not self.store.current or self.coach.paused or self._exporting or not self.analyze_button.isEnabled():
            return
        self._finish_gesture()
        self.change_goal()
        self.auto_schedule.started()
        self._auto_timer.stop()
        self._render_timer.stop()
        state = self.store.current
        before = self.store.before_last_edit()
        events = self.store.project_events()
        question = self.question.text()
        token = self._coach_generation
        self.coach_text.setPlainText("Preparing current parameters and matching before/after previews…")
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
                lambda: self.engine.coach_request(state, before, events, question), state, token))
            return
        def analyze_current():
            request = self.engine.coach_request(state, before, events, question)
            return self.coach.analyze(request)
        self._queue(Job("coach", self._generation, analyze_current, state, token, automatic=automatic))

    def closeEvent(self, event) -> None:
        self.local_controller.layer.finish()
        if not self._close_approved:
            if self.isVisible() and not self._allow_discard():
                event.ignore()
                return
            self._close_approved = True
        self.coach_response.reader.hide()
        self._stop_detail()
        self._settle_timer.stop()
        self._cancel_obsolete_render()
        self._cancel_auto_coach()
        self._notice_timer.stop()
        self._protected_jobs.clear()
        if self._active is not None or self._network_active is not None:
            self._closing = True
            if self._network_active is not None:
                self._network_active[2].set()
            self._pending = None
            self._render_timer.stop()
            self._set_controls(False)
            self.open_button.setEnabled(False)
            self.open_project_button.setEnabled(False)
            self.status.setText("Waiting for the current background operation to finish before closing…")
            event.ignore()
        else:
            self._closing = True
            self._render_timer.stop()
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
        opener = window.open_raw if args.source.suffix.lower() == ".arw" else window.open_project
        QTimer.singleShot(0, lambda: opener(args.source))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
