"""Modeless, bilingual local-model preparation; all operations are explicit."""
from dataclasses import replace
import threading
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit,
    QPushButton, QFileDialog, QSpinBox, QFormLayout, QProgressBar)
from .i18n import tr
from .local_runtime import default_models_dir, find_ollama
from .ollama_client import MODEL, LocalError


class _Signals(QObject):
    done = Signal(object, object)
    progress = Signal(object)


class _Task(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.signals = _Signals()
        self.function = function

    def run(self):
        try:
            result = self.function()
        except Exception as error:
            self.signals.done.emit(None, error)
        else:
            self.signals.done.emit(result, None)


class LocalModelsDialog(QDialog):
    selected = Signal(object)

    def __init__(self, parent, runtime, config):
        super().__init__(parent)
        self.runtime, self.config = runtime, config
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.cancel = None
        self.worker = None
        self._pending_close = None
        self.state = "stopped"
        self.setWindowTitle(tr("Local AI", "本地 AI"))
        self.resize(660, 480)
        layout = QVBoxLayout(self)
        title = QLabel(tr("Prepare a private photo coach", "准备本机照片教练"))
        title.setStyleSheet("font-size:18px; font-weight:600;")
        layout.addWidget(title)
        note = QLabel(tr("Download once using the internet. Photo analysis runs on this PC without an API key. The image test checks recognition and JSON output, not photographic taste.",
            "首次下载需要联网。照片分析在本机运行，无需 API key。图片测试验证识别与 JSON 输出能力，不代表已验证摄影审美。"))
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.models = QComboBox()
        self.models.addItem(tr("Qwen3.6 27B · ~19 GB · 32 GB VRAM recommended", "Qwen3.6 27B · 约 19 GB · 建议 32 GB 显存"), MODEL)
        form.addRow(tr("Model", "模型"), self.models)
        self.folder = QLineEdit(config.model_dir or str(default_models_dir()))
        self.folder.setCursorPosition(0)
        self.folder.setToolTip(self.folder.text())
        self.browse = QPushButton(tr("Choose…", "选择…"))
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(self.browse)
        form.addRow(tr("Model folder", "模型目录"), row)
        self.calls = QSpinBox()
        self.calls.setRange(1, 10000)
        self.calls.setValue(config.call_limit)
        form.addRow(tr("Local requests per session", "每次运行的本地请求上限"), self.calls)
        layout.addLayout(form)
        self.engine = QLabel(tr("Ollama found · starts only when requested" if find_ollama() else "Ollama not installed",
                               "已找到 Ollama · 按需启动" if find_ollama() else "尚未安装 Ollama"))
        self.engine.setWordWrap(True)
        layout.addWidget(self.engine)
        self.status = QLabel(tr("Not started. Detect the engine to check installed models.", "尚未启动。点击检测以检查已安装的模型。"))
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMinimumHeight(20)
        self.progress.setStyleSheet("QProgressBar { border:1px solid #555; text-align:center; }")
        layout.addWidget(self.progress)
        self.detect = QPushButton(tr("Detect / start", "检测 / 启动"))
        self.download = QPushButton(tr("Download and verify", "下载并验证"))
        self.verify = QPushButton(tr("Verify image support", "验证图片能力"))
        row = QHBoxLayout()
        for button in (self.detect, self.download, self.verify):
            row.addWidget(button)
        layout.addLayout(row)
        self.use = QPushButton(tr("Use local coach", "使用本地教练"))
        self.cancel_button = QPushButton(tr("Cancel operation", "取消操作"))
        self.stop = QPushButton(tr("Stop engine", "停止引擎"))
        row = QHBoxLayout()
        for button in (self.use, self.cancel_button, self.stop):
            row.addWidget(button)
        layout.addLayout(row)
        self.install = QPushButton(tr("Install / update Ollama…", "安装 / 更新 Ollama…"))
        self.close_button = QPushButton(tr("Close", "关闭"))
        self.choose_engine = QPushButton(tr("Choose existing engine…", "选择已有引擎…"))
        row = QHBoxLayout()
        row.addWidget(self.install)
        row.addWidget(self.choose_engine)
        row.addStretch()
        row.addWidget(self.close_button)
        layout.addLayout(row)
        layout.addStretch()
        self.browse.clicked.connect(self.choose_folder)
        self.folder.textChanged.connect(self.invalidate)
        self.folder.textChanged.connect(self.folder.setToolTip)
        self.calls.valueChanged.connect(self.invalidate)
        self.detect.clicked.connect(lambda: self.begin("inspect"))
        self.download.clicked.connect(lambda: self.begin("download"))
        self.verify.clicked.connect(lambda: self.begin("verify"))
        self.cancel_button.clicked.connect(lambda: self.cancel.set() if self.cancel else None)
        self.stop.clicked.connect(self.stop_engine)
        self.use.clicked.connect(self.use_model)
        self.close_button.clicked.connect(self.reject)
        self.install.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://ollama.com/download/windows")))
        self.choose_engine.clicked.connect(self.select_engine)
        self.refresh()

    @property
    def busy(self):
        return self.worker is not None

    def invalidate(self):
        self.state = "stopped"
        self.use.setEnabled(False)

    def refresh(self):
        parent = self.parent()
        occupied = bool(parent and (getattr(parent, "_network_active", None) or getattr(parent, "_reviewing", False)))
        for control in (self.folder, self.browse, self.calls, self.models, self.detect, self.download, self.verify, self.install, self.choose_engine):
            control.setEnabled(not self.busy and not occupied)
        self.cancel_button.setEnabled(self.busy)
        self.stop.setEnabled(not self.busy and not occupied and self.runtime.process is not None)
        self.use.setEnabled(not self.busy and not occupied and self.state == "ready")

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr("Choose model folder", "选择模型目录"), self.folder.text())
        if folder:
            self.folder.setText(folder)

    def select_engine(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Choose Ollama", "选择 Ollama"), "", "Ollama (ollama.exe);;All files (*)")
        if path:
            self.runtime.stop()
            self.config = replace(self.config, engine_path=path, verified_digest="", engine_version="")
            self.engine.setText(path)
            self.engine.setToolTip(path)
            self.invalidate()
            self.refresh()

    def begin(self, operation):
        parent = self.parent()
        if self.busy or (parent and (getattr(parent, "_network_active", None) or getattr(parent, "_reviewing", False))):
            return
        try:
            from pathlib import Path
            path = Path(self.folder.text()).expanduser()
            if not path.is_absolute():
                raise ValueError(tr("Choose an absolute folder path.", "请选择绝对目录路径。"))
            changed = not self.config.model_dir or path.resolve() != Path(self.config.model_dir).resolve()
            if changed and self.runtime.process is not None:
                self.runtime.stop()
            self.config = replace(self.config, model_dir=str(path), call_limit=self.calls.value(),
                verified_digest="" if changed else self.config.verified_digest,
                engine_version="" if changed else self.config.engine_version)
            self.runtime.store.save(self.config)
        except (ValueError, OSError, RuntimeError) as error:
            self.status.setText(str(error))
            return
        self.cancel = threading.Event()
        config, cancel = self.config, self.cancel
        task = _Task(lambda: self.runtime.download(config, cancel, task.signals.progress.emit)
                     if operation == "download" else getattr(self.runtime, operation)(config, cancel))
        self.worker = task
        task.signals.progress.connect(self.on_progress)
        task.signals.done.connect(self.finished_operation)
        self.progress.setRange(0, 0)
        self.status.setText(tr("Downloading and testing…" if operation == "download" else "Checking local engine and model…",
                               "正在下载并测试…" if operation == "download" else "正在检查本地引擎和模型…"))
        self.refresh()
        self.pool.start(task)

    def on_progress(self, event):
        if not self.worker or self.cancel.is_set():
            return
        total, completed = event.get("total"), event.get("completed", 0)
        if type(total) is int and type(completed) is int and total > 0:
            percent = max(0, min(100, int(100 * completed / total)))
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
            self.status.setText(tr("Downloading current layer", "正在下载当前分片") +
                f" · {completed/1e9:.2f} / {total/1e9:.2f} GB · {percent}%")
        elif event.get("status") == "success":
            self.progress.setRange(0, 0)
            self.status.setText(tr("Download complete. Testing image support…", "下载完成，正在验证图片能力…"))

    def finished_operation(self, result, error):
        cancelled = self.cancel.is_set()
        self.worker = None
        self.progress.setRange(0, 100)
        if error or cancelled:
            self.state = "failed"
            self.progress.setValue(0)
            messages = {"version": "当前模型需要更新的 Ollama，请安装更新或选择已有的新引擎。", "vision": "图片识别或结构化输出验证未通过。",
                        "connection": "本地引擎不可用，请重新检测。", "timeout": "本地操作超时，没有自动重试。", "busy": "另一项本地模型操作仍在运行。",
                        "response": "本地引擎返回了不完整或无效的数据。", "model": "推荐模型尚未安装。", "http": "本地引擎拒绝了操作，请检查引擎版本和模型。"}
            detail = tr(str(error), messages.get(getattr(error, "code", ""), str(error)))
            self.status.setText(tr("Cancelled. Partial downloads can be resumed.", "已取消，可以继续下载已有的部分。") if cancelled else detail)
        else:
            self.state, self.config = result
            try:
                self.runtime.store.save(self.config)
            except OSError:
                self.state = "failed"
                self.status.setText(tr("Could not save local settings. Check folder permissions.", "无法保存本地设置，请检查目录权限。"))
                self.refresh()
                if self._pending_close is not None:
                    super().done(self._pending_close)
                return
            self.progress.setValue(100 if self.state == "ready" else 0)
            en, zh = {"ready": ("Ready · Image recognition and structured output passed.", "可以使用 · 图片识别和结构化输出测试通过。"),
                      "not_downloaded": ("Model not downloaded. Download and verify when ready.", "尚未下载模型，请点击下载并验证。"),
                      "unverified": ("Model installed. Verify image support before use.", "模型已安装，请验证图片能力后使用。")}[self.state]
            self.status.setText(tr(en, zh))
            self.engine.setText(f"Ollama {self.runtime.version} · " + tr("Local only", "仅本机"))
        self.refresh()
        if self._pending_close is not None:
            super().done(self._pending_close)

    def stop_engine(self):
        self.runtime.stop()
        self.state = "stopped"
        self.status.setText(tr("Engine stopped. Downloaded models are retained.", "引擎已停止，已下载模型保留。"))
        self.refresh()

    def use_model(self):
        if self.state == "ready" and not self.busy:
            self.selected.emit(self.config)

    def done(self, result):
        if self.busy:
            self._pending_close = result
            self.cancel.set()
            self.status.setText(tr("Cancelling…", "正在取消…"))
            return
        super().done(result)
