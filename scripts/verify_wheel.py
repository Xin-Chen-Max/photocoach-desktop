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
    for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
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
for name in ("photocoach", "photocoach-verify"):
    assert callable(entry[name].load())
from photocoach.app import MainWindow, configure_appearance
from photocoach.settings import SettingsStore
from photocoach.coach import CoachFeedback
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
assert Path(photocoach.__file__).is_relative_to(Path(sys.argv[1]))
assert photocoach.__version__ == "0.12.0"
app=QApplication([])
QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
configure_appearance(app)
w=MainWindow(settings_store=SettingsStore(Path(sys.argv[2])/"unused-settings.json"))
w.show()
app.processEvents()
assert not w.windowIcon().isNull()
assert w.color_mixer.color.count() == 8
assert all(name in w.adjustment_sliders for name in ("whites", "blacks", "vibrance"))
assert [w.base_rendering.itemData(i) for i in range(3)] == ["natural_v1", "neutral_v1", "legacy_v1"]
assert w.pick_wb.text() == "Pick neutral white balance"
assert w.clipping.isCheckable() and not w.clipping.isChecked()
assert set(w.local_gradients.sliders)=={'exposure_ev','highlights','shadows','warmth'} and hasattr(w.local_gradients, 'feather') and hasattr(w.local_gradients, 'angle')
assert set(w.lens_editor.sliders)=={'distortion','vignetting_ev'}
assert set(w.detail_editor.sliders)=={'sharpening','masking','luminance_nr','chroma_nr'}
assert hasattr(w, "lang_button") and w.lang_button.text() == chr(0x4e2d) + chr(0x6587) + " / EN"
assert w.composition_button.text() == "Crop / Straighten"
assert hasattr(w, "crop_section") and w.crop_open_button.text() == "Crop and straighten" + chr(0x2026)
from photocoach.gradients import LinearGradient, RadialGradient
assert LinearGradient('smoke','Gradient 1',(0.,0.),(1.,1.)).to_dict()['exposure_ev']==0.
assert RadialGradient('radial','Radial 1',(.1,.1),(.9,.9)).to_dict()['kind']=='radial'
assert w.local_gradients.new_shape.itemText(1)=='Radial'
assert w.coach_mode.currentIndex() == 1 and w.api_budget.attempts == 0
assert not w.auto_mock.isChecked() and w.auto_mock.isHidden()
assert w.analyze_button.text() == "Set up coach"
assert not w.coach_model_picker.isEditable()
assert w.local_ai_button.text() == "Local models"
assert w.coach_selection.isHidden() and w.goal_panel.isHidden()
assert hasattr(w.question, "submitRequested") and w.question.limit == 2000
assert "Local AI" not in [w.coach_menu_button.menu().actions()[i].text() for i in range(2)]
assert w.local_budget.call_limit == 20 and w.local_runtime.process is None
from photocoach.api_coach import ApiConfig, ApiCoach
from photocoach.ollama_client import ENDPOINT, MODEL
ApiCoach(ApiConfig(ENDPOINT, MODEL, "ollama", 300, 4096), "")
current=w.store.activate(Path("package-smoke.ARW"),"package-smoke")
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
                 "bilingual_toggle":True,"display_clipping":True,"api_attempts":0}))
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
