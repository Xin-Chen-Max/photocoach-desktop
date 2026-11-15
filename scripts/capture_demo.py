"""Capture a JPEG preview fixture in the real UI; this does not decode RAW."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from photocoach.app import MainWindow, configure_appearance
from photocoach.settings import SettingsStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jpeg', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--settings', type=Path, required=True)
    args = parser.parse_args()
    app = QApplication([])
    configure_appearance(app)
    class NoCredentials:
        def read(self, *_):
            raise AssertionError('No credentials in documentation captures')
    window = MainWindow(settings_store=SettingsStore(args.settings), credentials=NoCredentials())
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.network_pool.start = lambda *_: (_ for _ in ()).throw(AssertionError('No network'))
    # Inject an explicit preview fixture, without pretending the JPEG is a RAW.
    window._queue = lambda *_: None
    rgb = np.asarray(Image.open(args.jpeg).convert('RGB')).copy()
    h, w = rgb.shape[:2]
    state = window.store.activate(Path('Glacier-National-Park-preview.jpg'), 'public-jpeg-demo', base_rendering='neutral_v1')
    window.store.mark_saved(state.revision)
    window.store.mark_preview(state)
    window.preview.configure_photo((w, h))
    window.preview.display(rgb)
    window._set_controls(True)
    window._sync_controls()
    window.filename.setText('Glacier National Park - JPEG preview demo')
    samples = rgb[::4, ::4]
    count = samples.shape[0] * samples.shape[1]
    window.histogram.display(dict(rgb=[np.bincount(samples[..., c].ravel(), minlength=256).tolist() for c in range(3)],
        samples=count, black_percent=float(np.all(samples == 0, axis=2).sum() * 100 / count),
        white_percent=float(np.any(samples == 255, axis=2).sum() * 100 / count)))
    window.resize(1600, 1000)
    window.show()
    app.processEvents()
    window.preview.fit()
    for timer in (window._render_timer, window._detail_timer, window._settle_timer):
        timer.stop()
    window.preview.set_quality('JPEG preview demonstration')
    app.processEvents()
    assert window.grab().save(str(args.output))
    window._close_approved = True
    window.close()
    app.processEvents()


if __name__ == '__main__':
    main()
