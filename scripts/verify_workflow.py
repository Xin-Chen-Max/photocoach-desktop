"""Verify editing, history, project recovery, and export against local ARW files."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PIL import Image
import tifffile
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QFileDialog
from photocoach.app import MainWindow, configure_appearance
from photocoach.cli import _make_output_directory
from photocoach.imaging import decode_raw, dependency_versions, quantize_srgb
from photocoach.adjustments import render_photo
from photocoach.composition import Composition
from photocoach.settings import SettingsStore
from photocoach.safe_io import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [args.source.resolve(strict=True), args.second.resolve(strict=True)]
    output = _make_output_directory(args.output)
    hashes = {p.name: sha256_file(p) for p in sources}
    app = QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    configure_appearance(app)
    window = MainWindow(settings_store=SettingsStore(output / "unused-settings.json"))
    window.show()
    pixels = []
    display = window.preview.display
    def capture(value):
        pixels[:] = [value.copy()]
        display(value)
    window.preview.display = capture
    timings = {}
    def idle(label):
        start = time.perf_counter()
        while (window._active or window._pending or window._pending_detail or window._protected_jobs
               or window._render_timer.isActive() or window._detail_timer.isActive()):
            app.processEvents()
            if time.perf_counter() - start > 90:
                raise TimeoutError(label)
            time.sleep(.005)
        app.processEvents()
        timings[label] = round(time.perf_counter() - start, 3)
        assert window.store.current, window.status.text()
    def screenshot(name, tab):
        window.tabs.setCurrentIndex(tab)
        app.processEvents()
        assert window.grab().save(str(output / name))
    original_dialog = QFileDialog.getSaveFileName
    try:
        window.open_raw(sources[0])
        idle("open_raw")
        assert window.store.current.photo_id == hashes[sources[0].name]
        baseline = pixels[0].copy()
        window.store.begin_drag("exposure_ev")
        for value in (5, 10, 15, 20, 25):
            window.exposure_slider.setValue(value)
            app.processEvents()
        window._finish_gesture()
        window.adjustment_sliders["contrast"].setValue(10)
        window.adjustment_sliders["saturation"].setValue(95)
        window.change_parameter("rgb_curve", ((0, .01), (.5, .52), (1, 1)), sync=True)
        window.wb_mode.setCurrentIndex(1)
        window.gains[0].setValue(1.05)
        window.apply_wb.click()
        window.goal.setText("Keep natural light and believable color.")
        window.change_goal()
        window.adjustment_sliders["highlights"].setValue(-35)
        window.adjustment_sliders["shadows"].setValue(30)
        window.exposure_slider.setValue(35)
        window.change_parameter("composition", Composition((.08,.12,.92,.94),3.25), sync=True)
        idle("ten_edits")
        assert len(window.store.undo_stack) == 10
        final_values = window.store.current.values
        final_pixels = pixels[0].copy()
        for _ in range(10):
            window.undo()
        idle("undo_all")
        np.testing.assert_array_equal(pixels[0], baseline)
        for _ in range(10):
            window.redo()
        idle("redo_all")
        assert window.store.current.values == final_values
        np.testing.assert_array_equal(pixels[0], final_pixels)
        window.compare.setChecked(True)
        idle("baseline")
        assert window.store.current.values == final_values
        assert pixels[0].shape == final_pixels.shape  # Comparison keeps current framing.
        window.compare.setChecked(False)
        idle("edited_again")
        np.testing.assert_array_equal(pixels[0], final_pixels)
        window.undo()
        idle("before_save_with_redo")
        saved_values = window.store.current.values
        saved_pixels = pixels[0].copy()
        project = output / "editable.photocoach"
        QFileDialog.getSaveFileName = lambda *a, **kw: (str(project), "PhotoCoach project (*.photocoach)")
        window.save_project()
        idle("save_project")
        assert project.is_file(), window.output_status.text()
        window.open_project(project)
        idle("reopen_project")
        assert window.store.current.values == saved_values
        assert len(window.store.redo_stack) == 1
        np.testing.assert_array_equal(pixels[0], saved_pixels)
        window.redo()
        idle("redo_after_reopen")
        np.testing.assert_array_equal(pixels[0], final_pixels)
        assert window.preview.fit_mode  # Redo of composition restores Fit.
        screenshot("editor.png", 0)
        window.question.setText("What should I adjust next?")
        window.analyze()
        idle("mock_analysis")
        request = window.coach.last_request
        assert request and request.parameters == window.store.current.values.to_dict()
        assert request.state_version == window.store.current.revision
        assert "MOCK" in window.coach_text.toPlainText()
        assert str(sources[0]) not in repr(request)
        screenshot("coach.png", 0)  # Coach and Edit share the workspace.
        screenshot("history.png", 1)
        window.tabs.setCurrentIndex(0)
        window.preview.actual_size()
        idle("native_detail")
        window.preview.pan(50, -30)
        idle("pan_native_detail")
        center = window.preview.center
        assert window.preview.quality == "Full detail"
        # Export immediately after an edit, before its queued preview can finish.
        window.exposure_slider.setValue(40)
        assert window.preview.center == center and window.preview.zoom == 1
        exported_state = window.store.current
        for suffix, selection in (("tif", "16-bit RGB TIFF (*.tif)"), ("jpg", "JPEG (*.jpg)")):
            destination = output / ("latest-full." + suffix)
            QFileDialog.getSaveFileName = lambda *a, **kw: (str(destination), selection)
            window.choose_export()
            idle("export_" + suffix)
            assert destination.is_file(), window.output_status.text()
            assert window.store.current.values == exported_state.values
        # Output status can change viewport parity. At 1:1 the image origin
        # remains on physical pixels, requiring at most a half-pixel center snap.
        assert window.preview.zoom == 1 and not window.preview.fit_mode
        assert max(abs(a-b) for a,b in zip(center, window.preview.center)) <= .5
        assert window.store.preview_revision == window.store.current.revision
        # Independent full decode + processing checks every 16-bit output value.
        photo = decode_raw(sources[0], exported_state.raw_options, half_size=False)
        expected = quantize_srgb(render_photo(photo.linear_rgb, exported_state.values), bits=16)
        with tifffile.TiffFile(output / "latest-full.tif") as tiff:
            actual = tiff.asarray()
            assert actual.dtype == np.uint16 and 34675 in tiff.pages[0].tags
            np.testing.assert_array_equal(actual, expected)
        from verify_high_res_preview import image_array
        l,t,r,b = window.preview.detail_request().rect
        reference = np.rint(expected[t:b,l:r].astype(np.float32)*(255/65535)).astype(np.uint8)
        difference = int(np.abs(reference.astype(np.int16)-image_array(window.preview._detail).astype(np.int16)).max())
        assert difference <= 1
        with Image.open(output / "latest-full.jpg") as jpeg:
            assert jpeg.size == (expected.shape[1], expected.shape[0])
            assert jpeg.info.get("icc_profile") and not jpeg.getexif()
        window.open_raw(sources[1])
        idle("switch_photo")
        assert window.store.current.photo_id == hashes[sources[1].name]
        assert "MOCK · Local rules" not in window.coach_text.toPlainText()
        assert window.coach.last_request is None
        after = {p.name: sha256_file(p) for p in sources}
        assert after == hashes
        report = {"phase": 1, "dependencies": dependency_versions(), "sources_sha256": hashes,
                  "originals_unchanged": True, "ten_operations": True,
                  "undo_redo_pixels_exact": True, "project_restore_pixels_and_redo_exact": True,
                  "mock_context_matches_state": True, "switch_clears_advice": True,
                  "latest_full_tiff_pixels_exact": True, "exported_parameters": exported_state.values.to_dict(),
                  "native_preview_tiff_max_8bit_difference": difference,
                  "composition_redo_returns_fit": True, "native_edit_keeps_view": True,
                  "api_attempts": window.api_budget.attempts,
                  "export_dimensions": [expected.shape[1], expected.shape[0]], "timings_seconds": timings,
                  "manual_acceptance": "Separate: monitor color, visual preference, real desktop mouse feel."}
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        QFileDialog.getSaveFileName = original_dialog
        if window.store.current:
            window.store.mark_saved(window.store.current.revision)
        window.close()
        while window._active:
            app.processEvents()
            time.sleep(.01)
        app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
