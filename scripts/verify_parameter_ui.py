"""Native Windows real-RAW acceptance for every parameter case and slider wiring."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from photocoach.app import MainWindow, configure_appearance
from photocoach.detail import CAMERA_RAW_DETAIL, DetailControls
from photocoach.safe_io import sha256_file
from photocoach.settings import SettingsStore
from verify_parameter_previews import parameter_cases


def pixels(image):
    return np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width()*3].reshape(image.height(), image.width(), 3).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--screenshots-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = QApplication([])
    assert app.platformName() == 'windows', 'Native fonts required'
    configure_appearance(app)
    class NoCredentials:
        def read(self, *_):
            raise AssertionError('Credentials forbidden')
    window = MainWindow(settings_store=SettingsStore(args.output / 'unused.json'), credentials=NoCredentials())
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.network_pool.start = lambda *_: (_ for _ in ()).throw(AssertionError('No network'))
    window.resize(1280, 800)
    window.show()
    digest = sha256_file(args.source)
    rows = []
    def wait(predicate):
        end = time.monotonic() + 180
        while not predicate():
            app.processEvents()
            if time.monotonic() > end:
                raise TimeoutError(window.status.text())
            time.sleep(.002)
        app.processEvents()
    def idle():
        wait(lambda: not (window._active or window._pending or window._pending_detail
             or window._render_timer.isActive() or window._settle_timer.isActive() or window._detail_timer.isActive()))
    def verify_frame(label):
        assert window.preview.quality == 'Full detail', label
        request = window.preview.detail_request()
        expected = window.engine.detail(window.store.current, request)[0]
        assert np.array_equal(pixels(window.preview._detail), expected), label + ': displayed pixels'
        assert window.store.preview_revision == window.store.current.revision
    try:
        window.open_raw(args.source); idle()
        window.preview.actual_size(); idle()
        for name, before, after in ([] if args.screenshots_only else parameter_cases()):
            window.store.apply_changes(before.fields().items())
            window._sync_controls(); window._after_edit(); idle()
            window.preview.actual_size(); idle()
            count = len(window.store.undo_stack)
            changed = [(field, getattr(after, field)) for field in after.fields() if getattr(after, field) != getattr(before, field)]
            slider = None
            if len(changed) == 1:
                field, value = changed[0]
                if field == 'exposure_ev':
                    slider, target = window.exposure_slider, round(value * 100)
                elif field in window.adjustment_sliders:
                    slider, target = window.adjustment_sliders[field], round(value * 100)
                elif field == 'color_mixer':
                    for color, band in enumerate(value.bands):
                        for axis, amount in enumerate(band):
                            if amount != before.color_mixer.bands[color][axis]:
                                window.color_mixer.color.setCurrentIndex(color)
                                slider, target = window.color_mixer.sliders[axis], round(amount)
                elif field in ('detail', 'lens_corrections'):
                    original = getattr(before, field)
                    attributes = [key for key in value.__dataclass_fields__ if getattr(value, key) != getattr(original, key)]
                    if len(attributes) == 1:
                        key = attributes[0]
                        editor = window.detail_editor if field == 'detail' else window.lens_editor
                        slider, target = editor.sliders[key], round(getattr(value, key) * (100 if key == 'vignetting_ev' else 1))
            if slider is not None:
                slider.setSliderDown(True); slider.setValue(target)
            elif name == 'camera_raw_detail_preset':
                window.detail_editor.preset.click()
            elif len(changed) == 1:
                field, value = changed[0]
                if field in ('base_rendering', 'composition'):
                    window.change_parameter(field, value, sync=True)
                else:
                    window._begin_parameter_gesture(field)
                    window.change_parameter(field, value)
            else:
                window.store.apply_changes(after.fields().items())
                window._sync_controls(); window._after_edit()
            if window.store.gesture_active:
                window._render_timer.stop()
                window.request_preview(interactive=True)
                wait(lambda: window._active is None and window._pending is None and window._pending_detail is None)
                assert window.preview.quality == 'Adjusting', name
                if slider is not None:
                    slider.setSliderDown(False)
                else:
                    window._finish_gesture()
            idle()
            assert window.store.current.values == after, name + ': parameter wiring'
            window.preview.actual_size(); idle()
            verify_frame(name)
            assert len(window.store.undo_stack) == count + 1, name + ': one undo'
            window.undo(); idle()
            assert window.store.current.values == before, name + ': undo values'
            verify_frame(name + ': undo')
            window.redo(); idle()
            assert window.store.current.values == after, name + ': redo values'
            rows.append(dict(parameter=name, displayed_matches_native=True, undo_redo=True, widget=slider is not None))
            if len(rows) % 20 == 0:
                print(f'Checked {len(rows)} native parameter cases', flush=True)
        base = parameter_cases()[0][1]
        def capture(name):
            # Resizing/i18n and value-label changes can move the editor. Scroll
            # after layout settlement so the screenshot proves usable controls.
            app.processEvents()
            window.edit_scroll.ensureWidgetVisible(window.detail_editor, 0, 4)
            app.processEvents()
            viewport = window.edit_scroll.viewport()
            for widget in (window.detail_editor.sliders['sharpening'], window.detail_editor.reset):
                assert viewport.rect().contains(widget.mapTo(viewport, widget.rect().topLeft()))
                assert viewport.rect().contains(widget.mapTo(viewport, widget.rect().bottomRight()))
            assert window.grab().save(str(args.output / name))
        for language in ('en', 'zh'):
            window.set_language(language)
            for width, height in ((1280, 800), (1920, 1080)):
                window.resize(width, height)
                window.detail_section.toggle.setChecked(True)
                window.edit_scroll.ensureWidgetVisible(window.detail_editor)
                for native in (False, True):
                    window.store.apply_changes(base.fields().items())
                    window._sync_controls(); window._after_edit(); idle()
                    (window.preview.actual_size if native else window.preview.fit)(); idle()
                    prefix = f'{language}-{width}-' + ('100' if native else 'fit')
                    capture(prefix + '-off.png')
                    slider = window.detail_editor.sliders['sharpening']
                    slider.setSliderDown(True); slider.setValue(80)
                    window._render_timer.stop(); window.request_preview(interactive=True)
                    wait(lambda: window._active is None and window._pending is None)
                    capture(prefix + '-adjusting.png')
                    slider.setSliderDown(False); idle()
                    verify_frame(prefix)
                    capture(prefix + '-full.png')
        assert sha256_file(args.source) == digest
        report = dict(native_cases=rows, screenshots=24, source_unchanged=True,
                      provider_calls=0, decode_count=window.engine.cache.decode_count)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(dict(native_cases=len(rows), source_unchanged=True, provider_calls=0)))
    finally:
        window._close_approved = True
        window.close(); idle()
        app.processEvents()


if __name__ == '__main__':
    main()
