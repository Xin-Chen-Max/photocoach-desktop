"""Window-wide graphite appearance. No image-processing or persisted preferences."""
from pathlib import Path
from PySide6.QtGui import QColor, QIcon, QPalette

CANVAS = "#202020"
PANEL = "#292929"
TEXT = "#e4e4e4"
MUTED = "#ababab"
ACCENT = "#79afff"


def configure_appearance(app):
    assets = Path(__file__).with_name("assets")
    app.setWindowIcon(QIcon(str(assets / "PhotoCoach.ico")))
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in ((QPalette.ColorRole.Window, PANEL), (QPalette.ColorRole.Base, CANVAS),
                        (QPalette.ColorRole.AlternateBase, "#303030"),
                        (QPalette.ColorRole.Button, "#353535"),
                        (QPalette.ColorRole.WindowText, TEXT), (QPalette.ColorRole.Text, TEXT),
                        (QPalette.ColorRole.ButtonText, TEXT),
                        (QPalette.ColorRole.Highlight, "#315c91"),
                        (QPalette.ColorRole.HighlightedText, "#ffffff"),
                        (QPalette.ColorRole.ToolTipBase, "#373737"),
                        (QPalette.ColorRole.ToolTipText, TEXT)):
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#898989"))
    app.setPalette(palette)
    app.setStyleSheet(STYLE.replace("@assets@", assets.as_posix()))


STYLE = """
QWidget { color:#e4e4e4; font-family:'Segoe UI'; font-size:13px; }
QMainWindow, #appBody { background:#242424; }
QLabel { background:transparent; border:0; }
#photoPanel, #photoCanvas { background:#202020; border:0; }
#editPanel, #coachPanel { background:#292929; border:0; }
#appTitle { font-weight:600; color:#c5c5c5; }
QPushButton, QToolButton { background:#353535; border:1px solid #4a4a4a;
    border-radius:4px; padding:6px 10px; min-height:18px; }
QPushButton:hover, QToolButton:hover { background:#414141; border-color:#777777; }
QPushButton:pressed, QToolButton:pressed { background:#303b4a; }
QPushButton:checked { background:#30445e; border-color:#79afff; color:white; }
QPushButton:focus, QToolButton:focus { border-color:#79afff; }
QPushButton:disabled, QToolButton:disabled { background:#2d2d2d; color:#898989; border-color:#3a3a3a; }
QPushButton#exportAction, QPushButton#askCoach { background:#3566a5; border-color:#5787c6; color:white; font-weight:600; }
QPushButton#exportAction:hover, QPushButton#askCoach:hover { background:#4076b9; }
QPushButton#exportAction:disabled, QPushButton#askCoach:disabled { background:#303944; color:#929aa4; border-color:#424b56; }
QToolButton::menu-button { width:16px; border-left:1px solid #4a4a4a; }
QToolButton[popupMode="1"] { padding-right:24px; }
QToolButton#sectionToggle { text-align:left; background:transparent; border:0;
    border-bottom:1px solid #3d3d3d; border-radius:0; padding:8px 2px; font-size:14px; font-weight:600; }
QToolButton#sectionToggle:focus { border-bottom:1px solid #79afff; }
QToolButton#sectionToggle:hover { background:#303030; }
QMenu { background:#303030; border:1px solid #555555; padding:4px; }
QMenu::item { padding:7px 24px 7px 12px; }
QMenu::item:selected { background:#315c91; color:white; }
QMenu::item:disabled { color:#898989; }
QLineEdit, QTextEdit, QTextBrowser, QComboBox, QDoubleSpinBox, QSpinBox {
    background:#222222; border:1px solid #515151; border-radius:4px; padding:6px;
    selection-background-color:#315c91; }
QLineEdit:focus, QTextEdit:focus, QTextBrowser:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color:#79afff; }
QComboBox::drop-down { border:0; width:24px; }
QComboBox::down-arrow { image:url("@assets@/chevron-down.svg"); width:12px; height:8px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin:border; subcontrol-position:top right; width:18px; border:0; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin:border; subcontrol-position:bottom right; width:18px; border:0; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image:url("@assets@/chevron-up.svg"); width:10px; height:6px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image:url("@assets@/chevron-down.svg"); width:10px; height:6px; }
QCheckBox { spacing:7px; }
QCheckBox::indicator { width:14px; height:14px; border:1px solid #828282; border-radius:3px; background:#222222; }
QCheckBox::indicator:checked { background:#3566a5; image:url("@assets@/check.svg"); }
QCheckBox::indicator:focus { border-color:#79afff; }
QTabWidget::pane { border:0; background:#292929; }
QTabBar::tab { background:#242424; color:#ababab; padding:8px 16px; border-bottom:2px solid transparent; }
QTabBar::tab:selected { color:white; background:#292929; border-bottom-color:#79afff; }
QTabBar::tab:focus { border-color:#79afff; }
QScrollArea { border:0; background:transparent; }
QScrollBar:vertical { background:#292929; width:10px; margin:0; }
QScrollBar::handle:vertical { background:#666666; border-radius:4px; min-height:28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; }
QSlider { min-height:22px; }
QSlider::groove:horizontal { height:3px; background:#555555; border-radius:1px; }
QSlider::sub-page:horizontal { background:#929292; }
QSlider::handle:horizontal { background:#d1d1d1; border:1px solid #eeeeee; width:10px; height:10px; margin:-5px 0; border-radius:5px; }
QSlider::handle:horizontal:hover, QSlider::handle:horizontal:focus { background:#79afff; border-color:#b6d3ff; }
QSlider::groove:horizontal:disabled { background:#3d3d3d; }
QSlider::sub-page:horizontal:disabled { background:#4b4b4b; }
QSlider::handle:horizontal:disabled { background:#666666; border-color:#777777; }
QSlider:disabled { color:#888888; }
QProgressBar { border:0; background:#424242; min-height:5px; }
QProgressBar::chunk { background:#79afff; }
QGroupBox { border:1px solid #464646; border-radius:4px; margin-top:12px; padding:12px 8px 8px; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; }
QWidget[coachTarget="true"] { border:1px solid #79afff; }
QLabel#statusLabel, QLabel[secondary="true"] { color:#ababab; font-size:12px; }
QSplitter::handle { background:#242424; }
QSplitter::handle:hover { background:#59779e; }
QListWidget { background:#292929; border:0; }
QListWidget::item { padding:5px; }
QListWidget::item:selected { background:#315c91; }
QToolTip { background:#373737; color:#eeeeee; border:1px solid #777777; padding:5px; font-size:12px; }
"""
