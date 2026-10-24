"""Controls inside a scrollable editor must not steal its wheel navigation."""
from PySide6.QtWidgets import QSlider, QDoubleSpinBox


class EditSlider(QSlider):
    def wheelEvent(self, event):
        # Drag or use arrow keys to edit. Ignore so Qt bubbles to the scroll area.
        event.ignore()


class DraftGainSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()
