"""Tests for concrete Coach suggestions, 1-click Apply, and Local Mask artificial lighting."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from photocoach.app import MainWindow
from photocoach.coach import CoachFeedback, MockCoach, build_request
from photocoach.coach_actions import (
    LIGHTING_RECIPE_META,
    resolve_actions, analyze_scene_lighting,
    make_lighting_action,
)
from photocoach.detail import CAMERA_RAW_DETAIL
from photocoach.gradients import LinearGradient, RadialGradient
from photocoach.imaging import sha256_file
from photocoach.project import snapshot_project, validate_project
from photocoach.settings import SettingsStore
from photocoach.state import EditValues, StateStore


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _feedback(next_step: str, question: str = "", photo_id: str = "photo-a", session_id: str = "session-a") -> CoachFeedback:
    return CoachFeedback(
        photo_id=photo_id,
        session_id=session_id,
        state_version=1,
        main_change="No changes yet.",
        reason="Test reason.",
        next_step_or_stop=next_step,
        observe_and_stop_when="Stop when balanced.",
        evidence=("Observed: test.",),
        source="AI · test-model",
        answer="Test answer.",
        question=question,
    )


def mask_action():
    return dict(operation="create_mask", kind="radial", name="Subject", start=[.2,.2], end=[.6,.8],
                inverted=False, exposure_ev=.10, highlights=0, shadows=0, warmth=.10, angle=0, feather=.6)


@pytest.mark.parametrize("text", ["保持曝光不变", "Do not change exposure", "Insufficient evidence"])
def test_text_only_feedback_is_read_only(app, text):
    from photocoach.coach_view import CoachResponsePanel
    panel = CoachResponsePanel()
    panel.output.show_feedback(_feedback(text, question="Add a mountain beam?"))
    assert not panel.apply_button.isEnabled()
    assert not panel.output._actions


def test_structured_targets_validate_each_control_and_the_whole_transaction():
    values = EditValues()
    assert resolve_actions([{"operation": "set", "parameter": "exposure_ev", "value": 0}], values) == ()
    spec = [{"operation":"add", "parameter":"exposure_ev", "value":.35},
            {"operation":"set", "parameter":"shadows", "value":.22}]
    action, = resolve_actions(spec, values)
    assert not action.is_applied(replace(values, exposure_ev=.35))
    assert action.is_applied(replace(values, exposure_ev=.35, shadows=.22))
    for invalid in [dict(spec[0], value=100), dict(spec[0], parameter="goal"), dict(spec[0], value=True)]:
        with pytest.raises(ValueError):
            resolve_actions([spec[1], invalid], values)
    extras = [{"operation":"set","parameter":key,"value":v} for key,v in
              [("detail.sharpening",45),("color_mixer.blue.saturation",-20),("lens_corrections.vignetting_ev",.4)]]
    extras.append({"operation":"set_curve","points":[[0,0],[.5,.6],[1,1]]})
    extras.append({"operation":"set_wb","mode":"custom","gains":[1.1,1,1.2]})
    update, = resolve_actions(extras, values)
    edited = replace(values, **dict(update.changes))
    assert (edited.detail.sharpening, edited.color_mixer.bands[5][1], edited.lens_corrections.vignetting_ev) == (45,-20,.4)
    assert edited.rgb_curve[1] == (.5,.6) and edited.raw_options.wb_gains == (1.1,1,1.2)


@pytest.mark.parametrize("recipe_id,title,description", LIGHTING_RECIPE_META)
def test_recipe_preserves_existing_masks(recipe_id,title,description):
    result = make_lighting_action(recipe_id, EditValues())
    assert 1 <= len(result.new_value) <= 2
    full = EditValues(local_gradients=tuple(LinearGradient(f"g{i}",f"Mask {i}",(.1,.2),(.8,.8)) for i in range(4)))
    with pytest.raises(ValueError, match="free mask slots"):
        make_lighting_action(recipe_id, full)
    with pytest.raises(ValueError, match="free mask slots"):
        resolve_actions([mask_action()], full)


def test_preview_mask_preserves_geometry_and_values():
    from photocoach.composition import Composition
    values = EditValues(composition=Composition((.5,0,1,1)))
    act, = resolve_actions([mask_action()], values, (1000,800))
    mask, = act.new_value
    assert mask.start == pytest.approx((.6,.2)) and mask.end == pytest.approx((.8,.8))
    assert mask.exposure_ev == mask.warmth == .10 and len(act.changes) == 1
    assert analyze_scene_lighting(np.full((64,96,3),128,np.uint8)) == {}


def test_fixed_targets_in_window(app,tmp_path,monkeypatch):
    import photocoach.coach_actions as actions
    window = MainWindow(settings_store=SettingsStore(tmp_path/"settings.json"),dev_mode=True)
    monkeypatch.setattr(window,"request_preview",lambda **kw:None)
    monkeypatch.setattr(window,"request_detail",lambda:None)
    path=tmp_path/"fixture.NEF"
    path.write_bytes(b"fixture")
    state=window.store.activate(path,sha256_file(path))
    window.preview.configure_photo((1000,800))
    window._set_controls(True)
    window._sync_controls()
    action, = resolve_actions([{"operation":"add","parameter":"exposure_ev","value":.35}, mask_action()],state.values)
    fb=replace(_feedback("Trial",photo_id=state.photo_id,session_id=state.session_id),
               state_version=state.revision, actions=(action,))
    monkeypatch.setattr(actions,"analyze_scene_lighting",lambda *a,**k:pytest.fail("UI reanalysis"))
    window.coach_text.set_preview_pixels(np.full((1067,1600,3),128,np.uint8))
    window.coach_text.show_feedback(fb)
    window.coach_response.open_reader()
    for lang in ("zh", "en"):
        window.set_language(lang)
        window.coach_text._follow_link(QUrl("photocoach:evidence"))
        assert "0.35 EV" in window.coach_text.toPlainText()
        assert "0.35 EV" in window.coach_response.reader_output.toPlainText()
    assert window.coach_response.reader_output._actions is fb.actions
    window.question.setText("Another question?")
    window.question.clear()
    assert not window.coach_response.apply_button.isEnabled()
    assert not window._apply_coach_suggestion()
    window.coach_text.show_feedback(fb)
    window.coach_response.reader.close()
    assert window._apply_coach_suggestion()
    assert not window._apply_coach_suggestion()
    assert len(window.store.undo_stack)==1 and window.store.current.exposure_ev==.35
    validate_project(snapshot_project(window.store))
    window.undo()
    assert window.store.current.values==state.values and not window._apply_coach_suggestion()
    window.store.activate(path,"different-photo")
    assert not window._apply_coach_suggestion()
    window._close_approved=True
    window.close()


def test_audit_fixes_reader_preview_slider_focus_and_bilingual_status(tmp_path):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QFocusEvent
    from photocoach.connections import ConnectionBook

    app = QApplication.instance() or QApplication([])
    window = MainWindow(settings_store=SettingsStore(tmp_path / "settings.json"), dev_mode=True)
    window.show()
    app.processEvents()

    state = window.store.activate(tmp_path / "sample.ARW", "a" * 64)
    preview = np.zeros((40, 60, 3), dtype=np.uint8)
    preview[:14, :20] = (240, 190, 110)  # Upper-left light source
    preview[22:32, 26:38] = 150
    window.coach_text.set_preview_pixels(preview)
    window._set_controls(True)
    window._sync_controls()

    # 1. CoachResponsePanel._sync_reader forwards _preview_pixels to reader_output
    fb = _feedback(
        "1. Add a Radial local mask 'Light Beam' to cast a light shaft on the center car.",
        question="给中间的车打一道光束",
        photo_id=state.photo_id,
        session_id=state.session_id,
    )
    window.coach_text.show_feedback(fb)
    window.coach_response.open_reader()
    app.processEvents()
    assert window.coach_response.reader_output._preview_pixels is preview
    assert window.coach_response.reader_output._actions == window.coach_text._actions
    window.coach_response.reader.close()

    # 2. FocusOut eventFilter closes gesture on detail/lens/radial-angle sliders
    for slider, param in (
        (window.detail_editor.sliders["sharpening"], "detail"),
        (window.lens_editor.sliders["distortion"], "lens_corrections"),
        (window.local_gradients.angle, "local_gradients"),
    ):
        window._begin_parameter_gesture(param)
        assert window.store.gesture_active
        slider.setSliderDown(True)
        window.eventFilter(slider, QFocusEvent(QEvent.Type.FocusOut))
        assert not slider.isSliderDown()
        assert not window.store.gesture_active

    # 3. Bilingual toggle keeps Coach status buttons translated and resets on new MainWindow
    window.toggle_language()
    assert window.analyze_button.text() in ("分析照片", "设置教练", "准备本地模型")
    assert window.pause_coach.text() == "暂停教练"
    window.coach_question_actions[0].trigger()
    assert "下一步最该调整什么" in window.question.text()
    window.set_coach_paused(True)
    assert window.pause_coach.text() == "恢复教练"
    window.set_coach_paused(False)
    assert window.pause_coach.text() == "暂停教练"

    # Creating a fresh MainWindow resets global language to 'en'
    w2 = MainWindow(settings_store=SettingsStore(tmp_path / "settings2.json"), dev_mode=True)
    assert w2.exposure_label.name.text() == "Exposure"
    assert w2.pause_coach.text() == "Pause coach"
    w2._close_approved = True
    w2.close()
    w2.deleteLater()

    # 4. Gemini default_for_provider has max_output_tokens == 8192 and timeout_seconds == 60
    gem_conn = ConnectionBook().default_for_provider("gemini")
    assert gem_conn.config.max_output_tokens == 8192
    assert gem_conn.config.timeout_seconds == 60

    window._close_approved = True
    window.close()
    window.deleteLater()


def test_surface_selective_rendering():
    from photocoach.adjustments import render_photo
    # Verify surface-selective & aerial-perspective range masking in render_photo at equal geometric weight:
    # Sunlit warm ridge pixels receive strong golden lift while blue sky and deep crevices are protected
    lin = np.zeros((48, 72, 3), dtype=np.float32)
    lin[:16, :] = (0.08, 0.14, 0.28)     # Blue sky (B > R)
    lin[16:35, :] = (0.09, 0.06, 0.035)  # Sunlit warm ridge (R > B)
    lin[35:, :] = (0.004, 0.005, 0.009)  # Deep occluded shadow crevice (low luma, B > R)
    uniform_beam = RadialGradient("g-test", "Test Beam", (0.0, 0.0), (1.0, 1.0), exposure_ev=0.68, warmth=0.56, feather=0.0)
    only_beam = EditValues(local_gradients=(uniform_beam,))
    base_out = render_photo(lin, EditValues())
    beam_out = render_photo(lin, only_beam)
    sky_red_gain = float(beam_out[8, 36, 0] / max(1e-5, base_out[8, 36, 0]))
    ridge_red_gain = float(beam_out[24, 36, 0] / max(1e-5, base_out[24, 36, 0]))
    crevice_red_gain = float(beam_out[40, 36, 0] / max(1e-5, base_out[40, 36, 0]))
    assert ridge_red_gain > sky_red_gain * 1.18
    assert ridge_red_gain > crevice_red_gain * 1.15

