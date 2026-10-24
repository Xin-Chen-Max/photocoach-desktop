"""Passive graphite primitives and scoped application appearance."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QIcon
from photocoach.theme import configure_appearance
from photocoach.workspace_widgets import ToolSection, ParameterLabel


def test_independent_sections_and_readonly_values():
    app = QApplication.instance() or QApplication([])
    light, color = ToolSection("Light", True), ToolSection("Color")
    light.show(); color.show(); app.processEvents()
    assert light.body.isVisible() and not color.body.isVisible()
    color.toggle.click()
    assert color.body.isVisible() and light.body.isVisible()
    label = ParameterLabel("Exposure  +1.25 EV")
    assert label.name.text() == "Exposure" and label.value.text() == "+1.25 EV"
    label.setText("Warmth -0.25")
    assert label.name.text() == "Warmth" and label.value.text() == "-0.25"
    light.close(); color.close()


def test_theme_neutral_palette_and_preserved_icon():
    app = QApplication.instance() or QApplication([])
    previous = (app.style().objectName(), app.styleSheet(), QPalette(app.palette()), QIcon(app.windowIcon()))
    try:
        configure_appearance(app)
        background = app.palette().color(QPalette.ColorRole.Window)
        assert background.red() == background.green() == background.blue()
        assert not app.windowIcon().isNull()
        assert "#3566a5" in app.styleSheet()
    finally:
        app.setStyle(previous[0]); app.setStyleSheet(previous[1])
        app.setPalette(previous[2]); app.setWindowIcon(previous[3])
