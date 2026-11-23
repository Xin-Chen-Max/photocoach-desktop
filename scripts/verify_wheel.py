"""Build offline from clean inputs and exercise the extracted wheel, outside checkout."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = output/"source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(root/name, source/name)
    shutil.copytree(root/"src"/"photocoach", source/"src"/"photocoach",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    wheels = output/"wheels"
    temporary = output/"temp"
    temporary.mkdir()
    build_environment = dict(os.environ, TEMP=str(temporary), TMP=str(temporary))
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(source), "--no-index",
                    "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheels)],
                    check=True, capture_output=True, text=True, env=build_environment)
    wheel, = wheels.glob("*.whl")
    isolated = output/"isolated"
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        # Match pyproject's existing explicit asset contract.
        import tomllib
        config = tomllib.loads((root/"pyproject.toml").read_text(encoding="utf-8"))
        expected = {f"photocoach/{name}" for name in config["tool"]["setuptools"]["package-data"]["photocoach"]}
        assert expected <= names, expected-names
        assert not any("source.png" in name.lower() for name in names)
        expected_modules = {p.relative_to(root/"src").as_posix() for p in (root/"src/photocoach").rglob("*.py")}
        assert expected_modules == {n for n in names if n.startswith("photocoach/") and n.endswith(".py")}
        archive.extractall(isolated)
    code = r'''
import sys,json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import photocoach
from importlib.metadata import distributions
dist = next(d for d in distributions(path=[sys.argv[1]]) if d.metadata["Name"].lower() == "photocoach")
entry = {e.name: e for e in dist.entry_points if e.group == "console_scripts"}
for name in ("photocoach", "photocoach-verify", "photocoach-mcp", "photocoach-eval"):
    assert callable(entry[name].load())
from photocoach.app import MainWindow, configure_appearance
from photocoach.color_grading import ColorWheel, ColorGrading, apply_color_grading
from photocoach.spots import SpotRemoval, MAX_SPOTS, apply_spots
from photocoach.gradients import RenderGeometry
from photocoach.composition import Composition
from photocoach.snapshots import EditSnapshot, MAX_SNAPSHOTS
from photocoach.project import SCHEMA_VERSION
from photocoach.state import EDIT_PIPELINE_VERSION
from photocoach.settings import SettingsStore
from photocoach.coach import CoachFeedback
from photocoach.coach_agent import PLAN_SCHEMA, measure_preview
from photocoach.agent_runtime import AgentRuntime, RunBudget
from photocoach.agent_tools import build_default_tool_registry
from photocoach.agent_memory import PlaybookRAG
from photocoach.mcp_server import PhotoCoachMCPServer
from photocoach.telemetry import TraceStore
assert len(build_default_tool_registry().specs()) >= 6
assert len(PlaybookRAG().search("sharpening", top_k=2)) >= 1
assert any(h["id"] == "pb_color_grading" for h in PlaybookRAG().search("color grading split tone", top_k=3))
assert any(h["id"] == "pb_follow_up_review" for h in PlaybookRAG().search("follow up did it help", top_k=3))
assert PhotoCoachMCPServer().handle_request({"jsonrpc": "2.0", "id": 1, "method": "ping"})["result"] == {}
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
assert Path(photocoach.__file__).is_relative_to(Path(sys.argv[1]))
assert photocoach.__version__ == "0.12.1"
app=QApplication([])
QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
configure_appearance(app)
w=MainWindow(settings_store=SettingsStore(Path(sys.argv[2])/"unused-settings.json"))
w.show()
app.processEvents()
assert not w.windowIcon().isNull()
assert w.color_mixer.color.count() == 8
assert set(w.color_grading.zone_buttons) == {"shadows", "midtones", "highlights"}
assert set(w.color_grading.sliders) == {"hue", "saturation", "luminance", "blend", "balance"}
assert not w.color_grading.toggle.isChecked() and w.color_grading.body.isHidden()
assert all(name in w.adjustment_sliders for name in ("whites", "blacks", "vibrance"))
assert [w.base_rendering.itemData(i) for i in range(3)] == ["natural_v1", "neutral_v1", "legacy_v1"]
assert w.pick_wb.text() == "Pick neutral white balance"
assert w.clipping.isCheckable() and not w.clipping.isChecked()
assert set(w.local_gradients.sliders)=={'exposure_ev','highlights','shadows','warmth'} and hasattr(w.local_gradients, 'feather') and hasattr(w.local_gradients, 'angle')
assert hasattr(w.local_gradients, 'brush_panel') and w.local_gradients.brush_panel.isHidden()
assert all(hasattr(w.local_gradients, attr) for attr in ('brush_size', 'brush_feather', 'brush_paint', 'brush_erase'))
assert hasattr(w, "spots_section") and hasattr(w, "spots_editor")
assert not w.spots_section.toggle.isChecked() and w.spots_section.body.isHidden()
assert set(w.spots_editor.sliders) == {"radius", "feather", "opacity"}
assert all(hasattr(w.spots_editor, attr) for attr in ("mode_heal", "mode_clone", "edit_on_photo", "delete_button", "reset_button"))
assert set(w.lens_editor.sliders)=={'distortion','vignetting_ev'}
assert set(w.detail_editor.sliders)=={'sharpening','masking','luminance_nr','chroma_nr'}
assert hasattr(w, "lang_button") and w.lang_button.text() == chr(0x4e2d) + chr(0x6587) + " / EN"
assert w.composition_button.text() == "Crop / Straighten"
assert hasattr(w, "crop_section") and w.crop_open_button.text() == "Crop and straighten" + chr(0x2026)
from photocoach.gradients import LinearGradient, RadialGradient, BrushMask, BrushStroke
assert LinearGradient('smoke','Gradient 1',(0.,0.),(1.,1.)).to_dict()['exposure_ev']==0.
assert RadialGradient('radial','Radial 1',(.1,.1),(.9,.9)).to_dict()['kind']=='radial'
assert BrushMask('brush','Brush 1',strokes=(BrushStroke(((0.2,0.3),(0.7,0.6))),)).to_dict()['kind']=='brush'
assert [w.local_gradients.new_shape.itemText(i) for i in range(3)] == ['Linear', 'Radial', 'Brush']
assert w.coach_mode.currentIndex() == 1 and w.api_budget.attempts == 0
assert not w.auto_mock.isChecked() and w.auto_mock.isHidden()
assert w.analyze_button.text() == "Set up coach"
assert hasattr(w, "followup_button") and w.followup_button.text() == "Did it help?" and not w.followup_button.isEnabled()
assert len(w.coach_question_actions) == 4 and w.coach_question_actions[3].text() == "Follow-up: Did it help?"
assert not w.coach_model_picker.isEditable()
assert w.local_ai_button.text() == "Local models"
assert w.coach_selection.isHidden() and w.goal_panel.isHidden()
assert hasattr(w.question, "submitRequested") and w.question.limit == 2000
assert w.coach_response.trial_button.text() == "Preview trial"
assert w.coach_response.trial_button.isHidden() and not w.coach_response.trial.isModal()
assert hasattr(w.coach_response, "trial_compare") and hasattr(w.coach_response, "trial_apply")
assert "findings" in PLAN_SCHEMA["required"]
assert SCHEMA_VERSION == 14 and EDIT_PIPELINE_VERSION == "phase1-sdr-srgb-v14"
assert hasattr(w.local_gradients, "luma_min") and hasattr(w.local_gradients, "edge_aware")
assert MAX_SNAPSHOTS == 24 and MAX_SPOTS == 64
assert hasattr(w, "snapshots_panel") and w.snapshots_panel.count_label.text() == "0 / 24"
spot_sample = SpotRemoval("smoke1", (0.3, 0.4), (0.5, 0.4))
assert SpotRemoval.from_dict(spot_sample.to_dict()) == spot_sample
cg_neutral = ColorGrading()
assert cg_neutral.is_identity and ColorGrading.from_dict(cg_neutral.to_dict()) == cg_neutral
cg_split = ColorGrading(shadows=ColorWheel(210.0, 0.5, 0.0), highlights=ColorWheel(35.0, 0.5, 0.0))
assert not cg_split.is_identity and ColorGrading.from_dict(cg_split.to_dict()) == cg_split
import numpy as np
spot_probe = np.full((8, 8, 3), 0.5, dtype=np.float32)
assert apply_spots(spot_probe, (), RenderGeometry.full((8, 8), Composition(), (8, 8)), spot_probe) is spot_probe
cg_probe = np.full((2, 2, 3), 0.5, dtype=np.float32)
assert apply_color_grading(cg_probe, cg_neutral) is cg_probe
assert measure_preview(np.zeros((9,9,3),np.uint8))["center"]["display_floor_fraction"] == 1.
assert "Local AI" not in [w.coach_menu_button.menu().actions()[i].text() for i in range(2)]
assert w.local_budget.call_limit == 100 and w.local_runtime.process is None
from photocoach.api_coach import ApiConfig, ApiCoach, TransportProgress
loading_progress = TransportProgress()
for stage in ("local_preparing", "local_running"):
    loading_progress.advance(stage)
    assert loading_progress.snapshot()[0] == stage
from photocoach.ollama_client import ENDPOINT, LOCAL_MODEL_CATALOG, MODEL, PullProgressTracker
from photocoach.local_runtime import LocalConfig, LocalHardwareInfo, classify_hardware_fit
assert [m.id for m in LOCAL_MODEL_CATALOG] == ["qwen3.5:4b", "qwen3-vl:8b", "qwen3.5:9b", "gemma4:12b", "gemma4:26b", "qwen3.6:27b"]
assert PullProgressTracker(expected_model_bytes=1000).update({"status": "success"})["percent"] == 100
assert classify_hardware_fit("qwen3.6:27b", LocalHardwareInfo(system_ram_bytes=64*1024**3, vram_bytes=24*1024**3, free_disk_bytes=100), is_installed=True).disk_ok is True
ApiCoach(ApiConfig(ENDPOINT, MODEL, "ollama", 300, 0), "")
ApiCoach(ApiConfig(ENDPOINT, MODEL, "ollama", 300, 4096), "")
assert ApiCoach(ApiConfig("http://127.0.0.1:11434/api/chat", "qwen3-vl:8b", "ollama", 300, 4096), "").supports_tools is False
current=w.store.activate(Path("package-smoke.ARW"),"package-smoke")
w._set_controls(True)
w._sync_controls()
w.snapshots_panel.name_input.setText("Smoke Look")
w.snapshots_panel.save_button.click()
app.processEvents()
assert len(w.store.snapshots) == 1 and w.store.snapshots[0].name == "Smoke Look"
w.snapshots_panel.compare_button.click()
app.processEvents()
assert w.store.current.compare_snapshot == w.store.snapshots[0]
w.edited_button.click()
app.processEvents()
assert w.store.current.compare_snapshot is None and w.api_budget.attempts == 0
w.coach_text.show_feedback(CoachFeedback(current.photo_id,current.session_id,0,
    "No changes.","Neutral parameters.","Use the RGB curve.","Stop if texture looks flat.",(),
    answer="A small midtone lift is a trial.",question="How should I use the curve?"))
assert not w.coach_response.apply_button.isEnabled()
w.coach_response.open_reader()
app.processEvents()
assert w.coach_response.reader.isVisible() and not w.coach_response.reader.isModal()
assert "small midtone lift" in w.coach_response.reader_output.toPlainText()
w.coach_response.reader.close()
assert w.grab().save(str(Path(sys.argv[2])/"isolated-window.png"))
w._close_approved=True
w.close()
print(json.dumps({"version":photocoach.__version__,"isolated_module":photocoach.__file__,
                 "eight_hsl_colors":True,"icon_loaded":True,"coach_reader":True,
                 "local_gradient_controls":True,"lens_controls":True,"detail_controls":True,
                 "bilingual_toggle":True,"display_clipping":True,"color_grading_engine":True,"color_grading_ui":True,"snapshots":True,"brush_masks":True,"spots":True,"coach_followup":True,"api_attempts":0}))
'''
    environment = dict(build_environment, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR="1")
    child = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code, str(isolated), str(output)],
                           cwd=output, env=environment, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
    if child.returncode != 0:
        raise RuntimeError(f"Isolated smoke failed (exit {child.returncode}):\n{child.stderr}\n{child.stdout}")
    result = dict(json.loads(child.stdout), wheel=wheel.name, assets=sorted(expected),
                  original_source_in_build=(source/"src/photocoach/assets/PhotoCoach-source.png").exists(),
                  original_source_excluded=True, dependencies_downloaded=False)
    (output/"report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__": main()
