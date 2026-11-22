"""Offline Qt layout screenshots. QT_SCALE_FACTOR simulates DPR, not Windows settings."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QScrollArea, QApplication, QDialogButtonBox
from photocoach.app import MainWindow, configure_appearance
from photocoach.settings import SettingsStore
from photocoach.histogram import sampled_histogram
from photocoach.coach import CoachFeedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    configure_appearance(app)
    window = MainWindow(settings_store=SettingsStore(args.output / "unused-settings.json"))
    window.store.activate(Path("Long filename landscape photograph - PhotoCoach UX verification.ARW"), "synthetic")
    window.filename.setText(window.store.current.source_path.name)
    window._set_controls(True)
    window._sync_controls()
    window._show_metadata({"visible_size":[6000,4000]})
    # A synthetic raster keeps layout checks independent of private photos and providers.
    raster = np.zeros((1000,1500,3), np.uint8)
    raster[:,:,0] = np.linspace(20,100,1500).astype(np.uint8)
    raster[:,:,1] = np.linspace(40,140,1000).astype(np.uint8)[:,None]
    raster[:,:,2] = 110
    window.preview.configure_photo((6000,4000))
    window.preview.display(raster)
    window.histogram.display(sampled_histogram(raster.astype(np.float32)/255, window.store.current.values))
    window.goal.setText("Natural light, preserve the hillside texture")
    window.question.setText("具体怎么调整 RGB 曲线？")
    # This check is layout only; full-detail processing has its own real-RAW verifier.
    window._view_changed = lambda: None
    window.preview.viewChanged.disconnect()
    report = []
    window.show()
    for width,height in ((1280,800),(1920,1080)):
        window.resize(width,height)
        for expanded, content in ((False,"empty"),(True,"empty"),(True,"long"),(False,"long")):
            window.coach_collapse.setChecked(not expanded)
            for _ in range(5): app.processEvents()
            panel_height = window.coach_panel.height()
            goal, question = window.goal.text(), window.question.text()
            if content == "empty":
                window.coach_text.setPlainText("Set a goal, then ask about the current edit.")
            else:
                current = window.store.current
                window.coach_text.show_feedback(CoachFeedback(
                    current.photo_id, current.session_id, current.revision,
                    "Synthetic layout example; no model was called.",
                    "A shared curve changes all three channels together. " * 10,
                    "Try a small midtone lift with the RGB curve.",
                    "Stop when the hillside looks natural and bright areas retain separation.",
                    ("Observed: Synthetic layout image.",), source="SIMULATED · Layout only",
                    answer="Use the shared RGB curve to lift midtones gradually. " * 55,
                    question=question))
            for _ in range(5): app.processEvents()
            assert window.coach_panel.height() == panel_height
            assert (window.goal.text(), window.question.text()) == (goal, question)
            if expanded and content == "long":
                assert window.coach_text.verticalScrollBar().maximum() > 0
            if expanded:
                for field in (window.goal, window.question):
                    assert field.height() >= field.sizeHint().height()
                window._show_coach_details()
                scroll = window.coach_details.findChild(QScrollArea)
                scroll.ensureWidgetVisible(window.coach_usage)
                app.processEvents()
                viewport = scroll.viewport()
                assert viewport.rect().contains(window.coach_usage.mapTo(viewport, window.coach_usage.rect().center()))
                scroll.verticalScrollBar().setValue(0)
                window.coach_details.hide()
            assert window.size().width() == width and window.size().height() == height
            for widget in (window.open_button, window.save_project_button, window.export_button,
                           window.more_button, window.fit_button, window.actual_size_button,
                           window.compare, window.edited_button, window.coach_collapse):
                assert widget.isVisible()
                assert window.rect().contains(widget.mapTo(window, widget.rect().topLeft()))
                assert window.rect().contains(widget.mapTo(window, widget.rect().bottomRight()))
            parts = [p.geometry() for p in window.photo_toolbar.parts]
            assert all(window.photo_toolbar.rect().contains(p) for p in parts)
            assert all(not a.intersects(b) for i,a in enumerate(parts) for b in parts[i+1:])
            assert window.preview.width() >= 400 and window.preview.height() >= 260
            for section in (window.light_section, window.color_section, window.curve_section):
                section.set_expanded(True)
            # Settle group geometry before asking QScrollArea to reveal children.
            for _ in range(5): app.processEvents()
            for control in (*window.balance_sliders, window.pick_wb, window.apply_wb,
                            *window.adjustment_sliders.values()):
                window.edit_scroll.ensureWidgetVisible(control)
                app.processEvents()
                viewport = window.edit_scroll.viewport()
                assert viewport.rect().contains(control.mapTo(viewport, control.rect().center())), (
                    control.accessibleName(), control.metaObject().className(),
                    control.mapTo(viewport, control.rect().center()), viewport.rect())
            window.edit_scroll.verticalScrollBar().setValue(0)
            name = f"{width}x{height}-{'expanded' if expanded else 'compact'}-{content}.png"
            assert window.grab().save(str(args.output / name))
            report.append({"logical_window":[width,height], "dpr":window.devicePixelRatioF(),
                           "expanded":expanded,"content":content,"photo_area":[window.preview.width(),window.preview.height()],
                           "toolbar_two_rows":window.photo_toolbar.two_rows,"screenshot":name})
            if expanded:
                response = window.coach_response
                assert response.reader_button.isEnabled() == (content == "long")
                assert response.copy_button.isEnabled() == (content == "long")
                if content == "long":
                    assert response.more_button.isVisible()
                    assert response.rect().contains(response.more_button.mapTo(response, response.more_button.rect().topLeft()))
                    assert response.rect().contains(response.more_button.mapTo(response, response.more_button.rect().bottomRight()))
                if content == "long":
                    window.question.setFocus()
                    cursor = window.question.textCursor()
                    cursor.setPosition(4)
                    window.question.setTextCursor(cursor)
                    response.open_reader()
                    for _ in range(5): app.processEvents()
                    reader = response.reader
                    size = reader.size()
                    assert reader.isVisible() and not reader.isModal()
                    assert response.reader_output.verticalScrollBar().maximum() > 0
                    assert response.reader_output.horizontalScrollBar().maximum() == 0
                    window.coach_text.keep_as_reference(current.photo_id, current.session_id, current.revision+3)
                    for _ in range(5): app.processEvents()
                    assert "Edit or question changed" in response.reader_output.toPlainText()
                    assert reader.size() == size and window.coach_panel.height() == panel_height
                    buttons = reader.findChild(QDialogButtonBox)
                    assert reader.rect().contains(buttons.geometry())
                    image = f"{width}x{height}-coach-reader.png"
                    assert reader.grab().save(str(args.output/image))
                    report.append({"logical_window":[width,height],"dpr":window.devicePixelRatioF(),
                                   "content":"coach-reader","reader_size":[reader.width(),reader.height()],
                                   "stale_warning":True,"screenshot":image})
                    reader.close()
                    assert (window.goal.text(), window.question.text()) == (goal, question)
        window.color_section.set_expanded(True)
        window.color_mixer.toggle.setChecked(True)
        window.color_mixer.color.setCurrentIndex(2)
        for _ in range(5): app.processEvents()
        before_revision = window.store.current.revision
        for control in (window.color_mixer.color, *window.color_mixer.sliders,
                        window.color_mixer.reset_color, window.color_mixer.reset_all):
            window.edit_scroll.ensureWidgetVisible(control)
            app.processEvents()
            viewport = window.edit_scroll.viewport()
            assert viewport.rect().contains(control.mapTo(viewport, control.rect().center()))
            assert control.width() >= control.minimumSizeHint().width()
        window.edit_scroll.ensureWidgetVisible(window.color_mixer.sliders[1])
        app.processEvents()
        name = f"{width}x{height}-color-mixer.png"
        assert window.grab().save(str(args.output / name))
        assert window.store.current.revision == before_revision
        report.append({"logical_window": [width, height], "dpr": window.devicePixelRatioF(),
                       "content": "color-mixer", "screenshot": name})
        window.coach_collapse.setChecked(False)
        for _ in range(5): app.processEvents()
        for control in (window.color_mixer.color, *window.color_mixer.sliders, window.color_mixer.reset_all):
            window.edit_scroll.ensureWidgetVisible(control)
            app.processEvents()
            viewport = window.edit_scroll.viewport()
            assert viewport.rect().contains(control.mapTo(viewport, control.rect().center()))
        assert window.preview.height() >= 260
        window.edit_scroll.ensureWidgetVisible(window.color_mixer.sliders[1])
        app.processEvents()
        name = f"{width}x{height}-color-mixer-coach.png"
        assert window.grab().save(str(args.output / name))
        report.append({"logical_window": [width, height], "dpr": window.devicePixelRatioF(),
                       "content": "color-mixer-with-coach", "screenshot": name})
        window.coach_collapse.setChecked(True)
        window.color_mixer.toggle.setChecked(False)
        for name in ("white-balance", "tone-color", "advanced-wb"):
            window.advanced_wb.setChecked(name == "advanced-wb")
            if name == "tone-color":
                for parameter, value in (("whites",-.1),("blacks",-.08),("vibrance",.2)):
                    window.store.set_parameter(parameter,value)
                window._sync_controls()
                target = window.adjustment_sliders["contrast"]
            else:
                target = window.apply_wb if name == "advanced-wb" else window.balance_sliders[0]
            window.edit_scroll.ensureWidgetVisible(target)
            for _ in range(5): app.processEvents()
            assert window.preview.width() >= 400 and window.preview.height() >= 260
            if name == "advanced-wb":
                for spin in window.gains:
                    window.edit_scroll.ensureWidgetVisible(spin)
                    app.processEvents()
                    assert spin.isVisible()
            screenshot=f"{width}x{height}-{name}.png"
            assert window.grab().save(str(args.output/screenshot))
            report.append({"logical_window":[width,height],"dpr":window.devicePixelRatioF(),
                           "content":name,"screenshot":screenshot})
        window.advanced_wb.setChecked(False)
        window.coach_collapse.setChecked(True)
        window.light_section.set_expanded(True)
        window.color_section.set_expanded(False)
        window.curve_section.set_expanded(False)
        window.edit_scroll.verticalScrollBar().setValue(0)
        for _ in range(5): app.processEvents()
        name = f"{width}x{height}-graphite-default.png"
        assert window.grab().save(str(args.output/name))
        report.append({"logical_window":[width,height],"dpr":window.devicePixelRatioF(),"content":"default-groups","screenshot":name})
        window.color_section.set_expanded(True)
        window.wb_mode.setCurrentIndex(1)
        window.gains[0].setValue(4)
        window.balance_sliders[0].setValue(200)
        window.wb_hint.setText("White balance draft is outside supported sensor gain limits. Reduce Warmth or Tint, then Apply.")
        window.apply_wb.setEnabled(False)
        window.wb_mode.setFocus()
        for _ in range(3): app.processEvents()
        window.edit_scroll.ensureWidgetVisible(window.apply_wb)
        for _ in range(5): app.processEvents()
        viewport = window.edit_scroll.viewport()
        assert viewport.rect().contains(window.wb_hint.mapTo(viewport, window.wb_hint.rect().center()))
        name=f"{width}x{height}-wb-error-focus-disabled.png"
        assert window.grab().save(str(args.output/name))
        report.append({"logical_window":[width,height],"dpr":window.devicePixelRatioF(),"content":"error-focus-disabled","screenshot":name})
    assert window.api_budget.attempts == window.coach.calls == 0
    (args.output / "report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    window._close_approved=True
    window.close()
    print(json.dumps(report))


if __name__ == "__main__": main()
