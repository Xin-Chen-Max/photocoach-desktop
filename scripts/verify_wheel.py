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
        assert {"photocoach/color_mixer.py", "photocoach/color_mixer_ui.py", "photocoach/white_balance.py"} <= names
        assert {"photocoach/gradients.py", "photocoach/gradient_ui.py", "photocoach/gradient_view.py"} <= names
        archive.extractall(isolated)
    code = r'''
import sys,json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import photocoach
from photocoach.app import MainWindow, configure_appearance
from photocoach.settings import SettingsStore
from photocoach.coach import CoachFeedback
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
assert Path(photocoach.__file__).is_relative_to(Path(sys.argv[1]))
assert photocoach.__version__ == "0.10.2"
app=QApplication([])
QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
configure_appearance(app)
w=MainWindow(settings_store=SettingsStore(Path(sys.argv[2])/"unused-settings.json"))
w.show()
app.processEvents()
assert not w.windowIcon().isNull()
assert w.color_mixer.color.count() == 8
assert all(name in w.adjustment_sliders for name in ("whites", "blacks", "vibrance"))
assert w.pick_wb.text() == "Pick neutral white balance"
assert w.clipping.isCheckable() and not w.clipping.isChecked()
assert set(w.local_gradients.sliders)=={'exposure_ev','highlights','shadows'}
from photocoach.gradients import LinearGradient, RadialGradient
assert LinearGradient('smoke','Gradient 1',(0.,0.),(1.,1.)).to_dict()['exposure_ev']==0.
assert RadialGradient('radial','Radial 1',(.1,.1),(.9,.9)).to_dict()['kind']=='radial'
assert w.local_gradients.new_shape.itemText(1)=='Radial'
assert w.coach_mode.currentIndex() == 1 and w.api_budget.attempts == 0
assert not w.auto_mock.isChecked() and w.auto_mock.isHidden()
assert w.analyze_button.text() == "Set up AI coach"
assert w.coach_model_picker.isEditable()
current=w.store.activate(Path("package-smoke.ARW"),"package-smoke")
w.coach_text.show_feedback(CoachFeedback(current.photo_id,current.session_id,0,
    "No changes.","Neutral parameters.","Use the RGB curve.","Stop if texture looks flat.",(),
    answer="A small midtone lift is a trial.",question="How should I use the curve?"))
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
                 "local_gradient_controls":True,"display_clipping":True,"api_attempts":0}))
'''
    environment = dict(build_environment, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR="1")
    child = subprocess.run([sys.executable, "-I", "-c", code, str(isolated), str(output)],
                           cwd=output, env=environment, check=True, capture_output=True, text=True)
    result = dict(json.loads(child.stdout), wheel=wheel.name, assets=sorted(expected),
                  original_source_in_build=(source/"src/photocoach/assets/PhotoCoach-source.png").exists(),
                  original_source_excluded=True, dependencies_downloaded=False)
    (output/"report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__": main()
