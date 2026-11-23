"""Tests for Roadmap #4B: Color Grading Darkroom UI, i18n, Coach actions, and Playbook RAG."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import QApplication

from photocoach import api_coach
from photocoach.adjustments import render_edit
from photocoach.agent_memory import PlaybookRAG, StyleMemoryStore
from photocoach.app import MainWindow
from photocoach.coach import CoachFeedback, build_request
from photocoach.coach_actions import resolve_actions
from photocoach.coach_agent import evaluate_trial_candidate, parse_plan
from photocoach.coach_view import CONTROL_NAMES, related_control
from photocoach.color_grading import ColorGrading, ColorWheel, WHEEL_FIELDS, WHEEL_NAMES
from photocoach.imaging import quantize_srgb
from photocoach.safe_io import sha256_file
from photocoach.settings import SettingsStore
from photocoach.state import EditValues, StateStore
from photocoach.theme import configure_appearance


@pytest.fixture
def window(tmp_path: Path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    configure_appearance(app)
    win = MainWindow(settings_store=SettingsStore(tmp_path / "unused-settings.json"))
    monkeypatch.setattr(win, "request_preview", lambda **kw: None)
    monkeypatch.setattr(win, "request_detail", lambda: None)

    source = tmp_path / "synthetic.ARW"
    source.write_bytes(b"readonly synthetic raw fixture")
    state = win.store.activate(source, sha256_file(source))
    win.preview.configure_photo((600, 400))
    win.preview.display(np.full((200, 300, 3), 120, dtype=np.uint8))
    win.store.mark_preview(state)
    win._set_controls(True)
    win._sync_controls()
    win.resize(1280, 800)
    win.show()
    app.processEvents()

    yield win

    win._render_timer.stop()
    win._settle_timer.stop()
    win._detail_timer.stop()
    win._notice_timer.stop()
    win._highlight_timer.stop()
    win._close_approved = True
    win.close()
    win.deleteLater()
    app.processEvents()


def test_color_grading_ui_initial_passive_zone_switch_and_enable_guard(window):
    app = QApplication.instance()
    cg = window.color_grading
    assert not window.color_section.toggle.isChecked()
    assert not cg.toggle.isChecked() and cg.body.isHidden()
    assert cg.current_zone == "shadows"
    assert cg.zone_buttons["shadows"].isChecked()
    assert set(cg.sliders) == {"hue", "saturation", "luminance", "blend", "balance"}
    assert set(cg.zone_buttons) == set(WHEEL_NAMES)
    assert set(cg.zone_swatches) == set(WHEEL_NAMES)

    # Neutral readouts stay muted gray while non-neutral readouts highlight in accent blue.
    assert "#9a9a9a" in cg.labels["hue"].value.styleSheet()
    assert "#9a9a9a" in cg.labels["saturation"].value.styleSheet()
    assert "#9a9a9a" in cg.labels["luminance"].value.styleSheet()
    assert "#9a9a9a" in cg.labels["blend"].value.styleSheet()
    assert "#9a9a9a" in cg.labels["balance"].value.styleSheet()

    before_state = window.store.current
    before_events = list(window.store.events)
    cg.toggle.setChecked(True)
    app.processEvents()
    assert not cg.body.isHidden()
    for zone in ("midtones", "highlights", "shadows"):
        cg.zone_buttons[zone].click()
        app.processEvents()
        assert cg.current_zone == zone

    assert window.store.current is before_state
    assert window.store.events == before_events
    assert not window.store.dirty
    assert window.api_budget.attempts == 0

    window._set_controls(False)
    assert not cg.isEnabled()
    window._set_controls(True)
    assert cg.isEnabled()


def test_color_grading_drag_single_undo_baseline_return_swatches_and_resets(window):
    app = QApplication.instance()
    cg = window.color_grading
    window.color_section.set_expanded(True)
    cg.toggle.setChecked(True)
    app.processEvents()

    window.preview.actual_size()
    window.preview.pan(18, -12)
    pose = (window.preview.center, window.preview.zoom)

    # Dragging from Baseline returns cleanly to Edited without moving zoom/pan, in one Undo step.
    window.change_compare(True)
    assert window.store.current.compare_baseline

    cg.select_zone("shadows", emit_gesture_end=False)
    cg.sliders["hue"].sliderPressed.emit()
    for val in (40, 120, 210):
        cg.sliders["hue"].setValue(val)
    cg.sliders["hue"].sliderReleased.emit()

    cg.sliders["saturation"].sliderPressed.emit()
    for val in (15, 30, 45):
        cg.sliders["saturation"].setValue(val)
    cg.sliders["saturation"].sliderReleased.emit()

    assert not window.store.current.compare_baseline
    assert (window.preview.center, window.preview.zoom) == pose
    assert len(window.store.undo_stack) == 2
    assert window.store.current.color_grading.shadows == ColorWheel(210.0, 0.45, 0.0)
    assert "#9fc6ff" in cg.labels["hue"].value.styleSheet()
    assert "#9fc6ff" in cg.labels["saturation"].value.styleSheet()
    assert "rgb(" in cg.zone_swatches["shadows"].styleSheet()
    assert "#383838" in cg.zone_swatches["highlights"].styleSheet()

    # Edit Highlights and Blend/Balance too.
    cg.zone_buttons["highlights"].click()
    cg.sliders["hue"].sliderPressed.emit()
    cg.sliders["hue"].setValue(35)
    cg.sliders["hue"].sliderReleased.emit()
    cg.sliders["saturation"].sliderPressed.emit()
    cg.sliders["saturation"].setValue(60)
    cg.sliders["saturation"].sliderReleased.emit()
    cg.sliders["luminance"].sliderPressed.emit()
    cg.sliders["luminance"].setValue(-25)
    cg.sliders["luminance"].sliderReleased.emit()
    cg.sliders["blend"].sliderPressed.emit()
    cg.sliders["blend"].setValue(75)
    cg.sliders["blend"].sliderReleased.emit()
    cg.sliders["balance"].sliderPressed.emit()
    cg.sliders["balance"].setValue(20)
    cg.sliders["balance"].sliderReleased.emit()
    assert len(window.store.undo_stack) == 7
    assert "#9fc6ff" in cg.labels["blend"].value.styleSheet()
    assert "#9fc6ff" in cg.labels["balance"].value.styleSheet()

    # Reset zone clears only active zone (highlights), keeping shadows + blend + balance.
    cg.reset_zone.click()
    assert len(window.store.undo_stack) == 8
    assert window.store.current.color_grading.highlights == ColorWheel()
    assert window.store.current.color_grading.shadows == ColorWheel(210.0, 0.45, 0.0)
    assert window.store.current.color_grading.blend == 0.75
    assert window.store.current.color_grading.balance == 0.20

    # Undo restores highlights values & sliders.
    window.undo()
    assert len(window.store.undo_stack) == 7
    assert window.store.current.color_grading.highlights == ColorWheel(35.0, 0.60, -0.25)
    assert cg.sliders["hue"].value() == 35
    assert cg.sliders["saturation"].value() == 60
    assert cg.sliders["luminance"].value() == -25

    # Reset Color Grading restores neutral ColorGrading() in one Undo step; repeated reset is a no-op.
    cg.reset_all.click()
    assert len(window.store.undo_stack) == 8
    assert window.store.current.color_grading == ColorGrading()
    cg.reset_all.click()
    cg.reset_zone.click()
    assert len(window.store.undo_stack) == 8

    # FocusOut eventFilter cleanly ends an active slider drag on all 5 Color Grading sliders.
    for key, slider in cg.sliders.items():
        window._begin_parameter_gesture("color_grading")
        assert window.store.gesture_active, key
        slider.setSliderDown(True)
        window.eventFilter(slider, QFocusEvent(QEvent.Type.FocusOut))
        assert not slider.isSliderDown()
        assert not window.store.gesture_active


def test_color_grading_passive_locate_and_bilingual_i18n(window):
    app = QApplication.instance()
    cg = window.color_grading
    before_state = window.store.current
    before_events = list(window.store.events)

    window.tabs.setCurrentIndex(1)  # Start on History tab.
    for target, expected_zone, expected_focus in (
        ("color_grading", "shadows", cg.zone_buttons["shadows"]),
        ("color_grading.midtones", "midtones", cg.zone_buttons["midtones"]),
        ("color_grading.highlights.saturation", "highlights", cg.sliders["saturation"]),
        ("color_grading.blend", "highlights", cg.sliders["blend"]),
        ("color_grading.balance", "highlights", cg.sliders["balance"]),
    ):
        window._locate_control(target)
        for _ in range(3):
            app.processEvents()
        assert window.tabs.currentIndex() == 0
        assert window.color_section.toggle.isChecked()
        assert cg.toggle.isChecked()
        assert cg.current_zone == expected_zone
        viewport = window.edit_scroll.viewport()
        assert viewport.rect().contains(expected_focus.mapTo(viewport, expected_focus.rect().center()))

    assert window.store.current is before_state
    assert window.store.events == before_events

    # Bilingual EN <-> ZH switching updates subsection title, zone buttons, slider labels, and reset buttons without edits.
    window.set_language("zh")
    assert cg.toggle.text() == "色彩分级 Color Grading"
    assert cg.zone_buttons["shadows"].text() == "阴影"
    assert cg.zone_buttons["midtones"].text() == "中间调"
    assert cg.zone_buttons["highlights"].text() == "高光"
    assert cg.labels["hue"].name.text() == "色相 Hue"
    assert cg.labels["saturation"].name.text() == "饱和度 Saturation"
    assert cg.labels["luminance"].name.text() == "明度 Luminance"
    assert cg.labels["blend"].name.text() == "混合 Blend"
    assert cg.labels["balance"].name.text() == "平衡 Balance"
    assert cg.reset_zone.text() == "重置当前分区"
    assert cg.reset_all.text() == "重置色彩分级"

    window.set_language("en")
    assert cg.toggle.text() == "Color Grading"
    assert cg.zone_buttons["shadows"].text() == "Shadows"
    assert cg.zone_buttons["midtones"].text() == "Midtones"
    assert cg.zone_buttons["highlights"].text() == "Highlights"
    assert cg.labels["luminance"].name.text() == "Luminance"
    assert cg.labels["blend"].name.text() == "Blend"
    assert cg.labels["balance"].name.text() == "Balance"
    assert cg.reset_zone.text() == "Reset zone"
    assert cg.reset_all.text() == "Reset Color Grading"
    assert window.store.current is before_state


def test_color_grading_coach_actions_apply_trial_navigation_and_playbook(window, tmp_path):
    # 1. resolve_actions supports set/add across zones, blend, balance, and circular hue wrap.
    base = EditValues(
        color_grading=ColorGrading(shadows=ColorWheel(350.0, 0.2, 0.0)),
    )
    assert resolve_actions([{"operation": "set", "parameter": "color_grading.blend", "value": 0.5}], base) == ()

    items = [
        {"operation": "add", "parameter": "color_grading.shadows.hue", "value": 25.0},
        {"operation": "set", "parameter": "color_grading.shadows.saturation", "value": 0.45},
        {"operation": "set", "parameter": "color_grading.shadows.luminance", "value": -0.10},
        {"operation": "set", "parameter": "color_grading.midtones.hue", "value": 45.0},
        {"operation": "set", "parameter": "color_grading.midtones.saturation", "value": 0.20},
        {"operation": "set", "parameter": "color_grading.highlights.hue", "value": 35.0},
        {"operation": "set", "parameter": "color_grading.highlights.saturation", "value": 0.50},
        {"operation": "set", "parameter": "color_grading.blend", "value": 0.70},
        {"operation": "add", "parameter": "color_grading.balance", "value": 0.15},
    ]
    action, = resolve_actions(items, base)
    assert action.parameter == "color_grading"
    assert "color_grading.shadows.hue → 15°" in action.label
    assert "color_grading.blend → 0.7" in action.label

    edited = replace(base, **dict(action.changes))
    assert action.is_applied(edited)
    assert edited.color_grading.shadows == ColorWheel(15.0, 0.45, -0.10)
    assert edited.color_grading.midtones == ColorWheel(45.0, 0.20, 0.0)
    assert edited.color_grading.highlights == ColorWheel(35.0, 0.50, 0.0)
    assert edited.color_grading.blend == pytest.approx(0.70)
    assert edited.color_grading.balance == pytest.approx(0.15)

    for invalid in (
        {"operation": "set", "parameter": "color_grading.shadows.hue", "value": 360.0},
        {"operation": "add", "parameter": "color_grading.shadows.hue", "value": 400.0},
        {"operation": "set", "parameter": "color_grading.shadows.saturation", "value": 1.2},
        {"operation": "set", "parameter": "color_grading.blend", "value": -0.1},
        {"operation": "set", "parameter": "color_grading.balance", "value": float("nan")},
        {"operation": "set", "parameter": "color_grading.balance", "value": True},
        {"operation": "set", "parameter": "color_grading.unknown.hue", "value": 20.0},
    ):
        with pytest.raises(ValueError):
            resolve_actions([invalid], base)

    # 2. Apply via MainWindow commits one Undo step and triggers passive navigation.
    fb_action, = resolve_actions(
        [
            {"operation": "set", "parameter": "color_grading.highlights.hue", "value": 38.0},
            {"operation": "set", "parameter": "color_grading.highlights.saturation", "value": 0.40},
        ],
        window.store.current.values,
    )
    assert window.apply_coach_action(fb_action) is True
    assert len(window.store.undo_stack) == 1
    assert window.store.current.color_grading.highlights == ColorWheel(38.0, 0.40, 0.0)
    assert window.color_grading.toggle.isChecked()
    assert window.color_grading.current_zone == "highlights"
    assert window.color_grading.sliders["hue"].value() == 38
    assert window.color_grading.sliders["saturation"].value() == 40
    window.undo()
    assert window.store.current.color_grading == ColorGrading()

    # 3. Trial evaluator allows valid split-tone visual checks and blocks zero-saturation hue-only no-ops.
    store = StateStore()
    src = tmp_path / "trial.ARW"
    src.write_bytes(b"trial fixture")
    st = store.activate(src, sha256_file(src))
    rng = np.random.default_rng(7)
    lin = rng.uniform(0.05, 0.85, (32, 48, 3)).astype(np.float32)
    cur_px = quantize_srgb(render_edit(lin, st.values))
    req = build_request(st, st, [], cur_px, cur_px, question="Add teal shadows and warm highlights in Color Grading")

    plan = parse_plan({
        "version": 1,
        "decision": "recommend",
        "objective": "Apply subtle teal/orange split toning via Color Grading.",
        "findings": [{"region": "whole", "observation": "Neutral tonal distribution.",
                      "interpretation": "Ready for subtle mood tinting.", "category": "preference", "confidence": "medium"}],
        "checks": [{"region": "whole", "metric": "visual", "direction": "visual",
                    "expectation": "Cool shadow and warm highlight tint without neutral gray cast."}],
        "uncertainty": "Subjective mood grading depends on viewer preference.",
    })
    good_actions = resolve_actions([
        {"operation": "set", "parameter": "color_grading.shadows.hue", "value": 205.0},
        {"operation": "set", "parameter": "color_grading.shadows.saturation", "value": 0.35},
        {"operation": "set", "parameter": "color_grading.highlights.hue", "value": 35.0},
        {"operation": "set", "parameter": "color_grading.highlights.saturation", "value": 0.30},
    ], st.values)
    report, delta = evaluate_trial_candidate(
        req, plan, good_actions, lambda v: quantize_srgb(render_edit(lin, v))
    )
    assert report.can_apply and report.status == "review" and not delta["failed_checks"]

    hue_only = resolve_actions([{"operation": "set", "parameter": "color_grading.shadows.hue", "value": 205.0}], st.values)
    bad_report, bad_delta = evaluate_trial_candidate(
        req, plan, hue_only, lambda v: quantize_srgb(render_edit(lin, v))
    )
    assert not bad_report.can_apply and "no_visible_change" in bad_delta["failed_checks"]

    # 4. related_control + CONTROL_NAMES + PlaybookRAG + Instructions + StyleMemoryStore
    assert related_control("In Color Grading, warm the Highlights.") == "color_grading.highlights"
    assert related_control("在色彩分级中降低阴影饱和度。") == "color_grading.shadows"
    assert related_control("In Color Grading, nudge Midtones hue.") == "color_grading.midtones"
    assert related_control("Adjust Color Grading Blend to soften the transition.") == "color_grading.blend"
    assert related_control("Shift Color Grading Balance positively after checking white balance.") == "color_grading.balance"
    assert related_control("Shift Color Grading Balance toward highlights.") == "color_grading"
    assert related_control("Use Color Grading for cool Shadows and warm Highlights.") == "color_grading"
    # Existing non-grading navigation still stays exact.
    assert related_control("Lift Shadows slightly.") == "shadows"
    assert related_control("Check white balance.") == "white_balance"

    assert CONTROL_NAMES["color_grading"] == "Color Grading"
    assert CONTROL_NAMES["color_grading.shadows.hue"] == "Color Grading · Shadows Hue"
    assert CONTROL_NAMES["color_grading.blend"] == "Color Grading · Blend"
    assert CONTROL_NAMES["color_grading.balance"] == "Color Grading · Balance"

    rag = PlaybookRAG()
    assert any(h["id"] == "pb_color_grading" for h in rag.search("color grading split tone teal orange", top_k=3))
    assert any(h["id"] == "pb_color_grading" for h in rag.search("色彩分级 分离色调 青橙 色轮", top_k=3))

    for instr in (api_coach.COACH_INSTRUCTIONS, api_coach.AGENT_INSTRUCTIONS):
        assert "color_grading.shadows.hue" in instr
        assert "3-way Color Grading in encoded sRGB" in instr

    mem = StyleMemoryStore(tmp_path / "style.json")
    mem.set_enabled(True)
    mem.record_applied_edit(EditValues(), edited)
    assert any("Color Grading" in rule for rule in mem.preferences())
