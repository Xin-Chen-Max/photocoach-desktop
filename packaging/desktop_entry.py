"""Frozen desktop entry with an offline, opt-in bundle verification command."""
from multiprocessing import freeze_support
import sys


def self_check(arguments):
    import json
    from pathlib import Path
    import time
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from photocoach import __version__
    from photocoach.app import MainWindow, configure_appearance
    from photocoach.detail import DetailControls
    from photocoach.imaging import dependency_versions, export_image, quantize_srgb
    from photocoach.adjustments import render_photo
    from photocoach.state import EditValues
    from photocoach.safe_io import sha256_file
    from photocoach.settings import SettingsStore
    import tifffile
    output = Path(arguments[0]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    class NoCredentials:
        def read(self, *_):
            raise AssertionError('Bundle verification must not read credentials')
    app = QApplication([])
    configure_appearance(app)
    window = MainWindow(settings_store=SettingsStore(output / 'isolated-settings.json'), credentials=NoCredentials())
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.network_pool.start = lambda *_: (_ for _ in ()).throw(AssertionError('No network in bundle verification'))
    window.resize(1280, 800)
    window.show()
    def idle():
        deadline = time.monotonic() + 180
        while (window._active or window._pending or window._pending_detail or window._render_timer.isActive()
               or window._detail_timer.isActive() or window._settle_timer.isActive()):
            app.processEvents()
            if time.monotonic() > deadline:
                raise TimeoutError('Bundled RAW rendering did not finish')
            time.sleep(.002)
        app.processEvents()
    report = dict(version=__version__, dependencies=dependency_versions(), frozen=bool(getattr(sys, 'frozen', False)),
                  provider_calls=0, raw_checked=False)
    try:
        app.processEvents()
        assert not window.windowIcon().isNull()
        assert window.grab().save(str(output / 'startup.png'))
        linear = np.repeat(np.linspace(.01, .8, 320, dtype=np.float32)[None, :, None], 240, axis=0)
        linear = np.repeat(linear, 3, axis=2)
        rendered = render_photo(linear, EditValues(detail=DetailControls(sharpening=60)))
        export_image(output / 'export-check.tif', rendered)
        assert np.array_equal(tifffile.imread(output / 'export-check.tif'), quantize_srgb(rendered, bits=16))
        if len(arguments) > 1:
            source = Path(arguments[1]).resolve()
            digest = sha256_file(source)
            window.open_raw(source); idle()
            assert window.store.current is not None and window.store.current.photo_id == digest
            window.preview.actual_size(); idle()
            window.detail_section.toggle.setChecked(True)
            window.change_parameter('detail', DetailControls(sharpening=80), sync=True); idle()
            assert window.preview.quality == 'Full detail'
            assert window.store.current.detail.sharpening == 80
            assert window.grab().save(str(output / 'raw-detail.png'))
            window.undo(); idle()
            assert window.store.current.detail == DetailControls()
            assert sha256_file(source) == digest
            report.update(raw_checked=True, source_unchanged=True, sharpening_undo=True)
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        window._close_approved = True
        window.close()
        idle()
        app.processEvents()
    return 0


if __name__ == '__main__':
    freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == '--self-check':
        raise SystemExit(self_check(sys.argv[2:]))
    from photocoach.app import main
    raise SystemExit(main())
