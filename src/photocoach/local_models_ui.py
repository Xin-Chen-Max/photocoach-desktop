"""Modeless, bilingual local-model manager & guided Ollama setup; all operations are explicit."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .i18n import tr
from .local_runtime import (
    DEFAULT_PORT,
    EXISTING_DEFAULT_PORT,
    LOCAL_MODEL_CATALOG,
    DiscoveredModelRow,
    LocalConfig,
    LocalDiscoveryResult,
    LocalHardwareInfo,
    classify_hardware_fit,
    default_models_dir,
    detect_local_hardware,
    find_ollama,
    get_curated_model,
    is_ollama_version_compatible,
    recommend_best_model,
    required_ollama_version_for,
)
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


_TIER_LABELS_EN = {
    "ultra_compact": "Ultra-Compact",
    "compact": "Compact",
    "balanced": "Balanced",
    "enthusiast": "Enthusiast MoE",
    "workstation": "Workstation",
    "custom": "Custom Vision",
}

_TIER_LABELS_ZH = {
    "ultra_compact": "超轻量",
    "compact": "紧凑级",
    "balanced": "均衡级",
    "enthusiast": "发烧级 MoE",
    "workstation": "工作站旗舰",
    "custom": "自定义视觉",
}


class LocalModelsDialog(QDialog):
    selected = Signal(object)

    def __init__(self, parent, runtime, config: LocalConfig, *, hardware: LocalHardwareInfo | None = None):
        super().__init__(parent)
        self.runtime = runtime
        self.config = config
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.cancel: threading.Event | None = None
        self.worker: _Task | None = None
        self._pending_close: int | None = None
        self._pending_action = None
        self._last_operation = ""
        self.state = "ready" if config.verified_digest else "stopped"
        self._hardware = hardware if hardware is not None else detect_local_hardware(config.model_dir or default_models_dir())
        self._discovered_rows: dict[str, DiscoveredModelRow] = {}
        self._hidden_non_vision_count = 0
        self._syncing_selection = False

        self.setWindowTitle(tr("Local AI", "本地 AI"))
        self.resize(840, 660)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size:17px; font-weight:600;")
        layout.addWidget(self.title_label)

        self.note_label = QLabel()
        self.note_label.setWordWrap(True)
        layout.addWidget(self.note_label)

        self.hardware_label = QLabel()
        self.hardware_label.setWordWrap(True)
        self.hardware_label.setStyleSheet("color:#9fc6ff; font-weight:500;")
        layout.addWidget(self.hardware_label)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(5)
        form.setHorizontalSpacing(10)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        service_row = QHBoxLayout()
        service_row.setSpacing(6)
        self.service_mode = QComboBox()
        self.service_mode.addItem("", "managed")
        self.service_mode.addItem("", "existing")
        idx_mode = max(0, self.service_mode.findData(self.config.service_mode))
        self.service_mode.setCurrentIndex(idx_mode)
        service_row.addWidget(self.service_mode, 1)

        self.endpoint_prefix_label = QLabel("http://127.0.0.1:")
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(self.config.port)
        self.endpoint_suffix_label = QLabel("/api/chat")
        service_row.addWidget(self.endpoint_prefix_label)
        service_row.addWidget(self.port)
        service_row.addWidget(self.endpoint_suffix_label)
        self.service_form_label = QLabel()
        form.addRow(self.service_form_label, service_row)

        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        self.models = QComboBox()
        model_row.addWidget(self.models, 1)
        self.calls_label = QLabel()
        self.calls = QSpinBox()
        self.calls.setRange(1, 10000)
        self.calls.setValue(config.call_limit)
        model_row.addWidget(self.calls_label)
        model_row.addWidget(self.calls)
        self.model_form_label = QLabel()
        form.addRow(self.model_form_label, model_row)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(6)
        self.folder = QLineEdit(config.model_dir or str(default_models_dir()))
        self.folder.setCursorPosition(0)
        self.folder.setToolTip(self.folder.text())
        self.browse = QPushButton()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(self.browse)
        self.folder_form_label = QLabel()
        form.addRow(self.folder_form_label, folder_row)
        layout.addLayout(form)

        self.model_table = QTableWidget(0, 5)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.model_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.verticalHeader().setDefaultSectionSize(26)
        self.model_table.setWordWrap(False)
        self.model_table.setMinimumHeight(188)
        header = self.model_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.model_table, 1)

        self.use_case_label = QLabel()
        self.use_case_label.setWordWrap(True)
        self.use_case_label.setMinimumHeight(20)
        self.use_case_label.setStyleSheet("color:#b8c4d0; padding-top:2px;")
        layout.addWidget(self.use_case_label)

        self.hidden_models_label = QLabel()
        self.hidden_models_label.setWordWrap(True)
        self.hidden_models_label.setStyleSheet("color:#9a9a9a; font-size:11px;")
        self.hidden_models_label.hide()
        layout.addWidget(self.hidden_models_label)

        self.engine = QLabel()
        self.engine.setWordWrap(True)
        layout.addWidget(self.engine)

        self.version_warning_label = QLabel()
        self.version_warning_label.setWordWrap(True)
        self.version_warning_label.setStyleSheet("color:#f1c58c; font-weight:600;")
        self.version_warning_label.hide()
        layout.addWidget(self.version_warning_label)

        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if self.state == "ready" else 0)
        self.progress.setMinimumHeight(18)
        self.progress.setStyleSheet("QProgressBar { border:1px solid #555; text-align:center; }")
        layout.addWidget(self.progress)

        self.detect = QPushButton()
        self.download = QPushButton()
        self.verify = QPushButton()
        self.delete_button = QPushButton()
        action_row1 = QHBoxLayout()
        action_row1.setSpacing(6)
        for button in (self.detect, self.download, self.verify, self.delete_button):
            action_row1.addWidget(button)
        layout.addLayout(action_row1)

        self.use = QPushButton()
        self.cancel_button = QPushButton()
        self.stop = QPushButton()
        action_row2 = QHBoxLayout()
        action_row2.setSpacing(6)
        for button in (self.use, self.cancel_button, self.stop):
            action_row2.addWidget(button)
        layout.addLayout(action_row2)

        self.install = QPushButton()
        self.choose_engine = QPushButton()
        self.close_button = QPushButton()
        footer_row = QHBoxLayout()
        footer_row.setSpacing(6)
        footer_row.addWidget(self.install)
        footer_row.addWidget(self.choose_engine)
        footer_row.addStretch()
        footer_row.addWidget(self.close_button)
        layout.addLayout(footer_row)

        self.browse.clicked.connect(self.choose_folder)
        self.folder.textChanged.connect(self._on_folder_edited)
        self.folder.textChanged.connect(self.folder.setToolTip)
        self.calls.valueChanged.connect(self._on_calls_changed)
        self.service_mode.currentIndexChanged.connect(self._on_service_mode_changed)
        self.port.valueChanged.connect(self._on_port_changed)
        self.models.currentIndexChanged.connect(self._on_combo_model_selected)
        self.model_table.itemSelectionChanged.connect(self._on_table_model_selected)
        self.detect.clicked.connect(lambda: self.begin("inspect"))
        self.download.clicked.connect(lambda: self.begin("download"))
        self.verify.clicked.connect(lambda: self.begin("verify"))
        self.delete_button.clicked.connect(lambda: self.delete_selected_model(confirm=True))
        self.cancel_button.clicked.connect(lambda: self.cancel.set() if self.cancel else None)
        self.stop.clicked.connect(self.stop_engine)
        self.use.clicked.connect(self.use_model)
        self.close_button.clicked.connect(self.reject)
        self.install.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://ollama.com/download/windows")))
        self.choose_engine.clicked.connect(self.select_engine)

        self._apply_static_texts()
        self._rebuild_model_entries()
        self.status.setText(
            tr(
                "Ready · Image recognition and structured output passed."
                if self.state == "ready"
                else "Not started. Detect the engine to check installed models.",
                "可以使用 · 图片识别和结构化输出测试通过。"
                if self.state == "ready"
                else "尚未启动。点击检测以检查已安装的模型。",
            )
        )
        self.refresh()

    @property
    def busy(self) -> bool:
        return self.worker is not None

    def set_language(self, lang: str | None = None) -> None:
        self._apply_static_texts()
        self._rebuild_model_entries()
        self.refresh()

    def _apply_static_texts(self) -> None:
        self.setWindowTitle(tr("Local AI", "本地 AI"))
        self.title_label.setText(tr("Prepare a private photo coach", "准备本机照片教练"))
        self.note_label.setText(
            tr(
                "Download once using the internet. Photo analysis runs on this PC without an API key. "
                "The image test checks recognition and JSON output, not photographic taste.",
                "首次下载需要联网。照片分析在本机运行，无需 API key。图片测试验证识别与 JSON 输出能力，不代表已验证摄影审美。",
            )
        )
        self.service_form_label.setText(tr("Ollama service", "Ollama 服务"))
        self.service_mode.blockSignals(True)
        self.service_mode.setItemText(
            0, tr("Managed by PhotoCoach (isolated folder)", "由 PhotoCoach 托管（独立目录）")
        )
        self.service_mode.setItemText(
            1, tr("Existing local Ollama service (127.0.0.1)", "连接本机已有 Ollama 服务 (127.0.0.1)")
        )
        self.service_mode.blockSignals(False)
        self.model_form_label.setText(tr("Model", "模型"))
        self.calls_label.setText(tr("Session limit", "会话请求上限"))
        self.folder_form_label.setText(tr("Model folder", "模型目录"))
        self.browse.setText(tr("Choose…", "选择…"))
        self.model_table.setHorizontalHeaderLabels(
            [
                tr("Model", "模型"),
                tr("Tier", "级别"),
                tr("Size", "大小"),
                tr("Hardware fit", "硬件匹配"),
                tr("Status & capabilities", "状态与能力"),
            ]
        )
        self.detect.setText(tr("Detect / start", "检测 / 启动"))
        self.download.setText(tr("Download and verify", "下载并验证"))
        self.verify.setText(tr("Verify image support", "验证图片能力"))
        self.delete_button.setText(tr("Delete model", "删除模型"))
        self.use.setText(tr("Use local coach", "使用本地教练"))
        self.cancel_button.setText(tr("Cancel operation", "取消操作"))
        self.stop.setText(tr("Stop engine", "停止引擎"))
        self.install.setText(tr("Install / update Ollama…", "安装 / 更新 Ollama…"))
        self.choose_engine.setText(tr("Choose existing engine…", "选择已有引擎…"))
        self.close_button.setText(tr("Close", "关闭"))
        self._update_engine_label()
        self._update_hardware_banner()

    def _update_engine_label(self) -> None:
        if self.runtime.version:
            mode_tag = (
                tr("Existing loopback service", "本机已有服务")
                if self.config.service_mode == "existing"
                else tr("Local only", "仅本机")
            )
            self.engine.setText(f"Ollama {self.runtime.version} · 127.0.0.1:{self.config.port} · {mode_tag}")
            return
        if self.config.service_mode == "existing":
            self.engine.setText(
                tr(
                    f"Existing Ollama service target: 127.0.0.1:{self.config.port} · Click Detect to connect",
                    f"目标本机 Ollama 服务：127.0.0.1:{self.config.port} · 点击检测以连接",
                )
            )
            return
        if self.config.engine_path:
            self.engine.setText(self.config.engine_path)
            self.engine.setToolTip(self.config.engine_path)
            return
        found = find_ollama()
        self.engine.setText(
            tr(
                "Ollama found · starts only when requested" if found else "Ollama not installed",
                "已找到 Ollama · 按需启动" if found else "尚未安装 Ollama",
            )
        )

    def _installed_model_ids(self) -> set[str]:
        ids = {mid for mid, row in self._discovered_rows.items() if row.installed}
        ids.update(self.config.verified_models.keys())
        if self.config.verified_digest:
            ids.add(self.config.model)
        return ids

    def _update_hardware_banner(self) -> None:
        hw = self._hardware
        rec_id = recommend_best_model(hw, self._installed_model_ids())
        rec_entry = get_curated_model(rec_id)
        rec_name = (rec_entry.title_zh if tr("en", "zh") == "zh" else rec_entry.title_en) if rec_entry else rec_id
        parts_en = []
        parts_zh = []
        if hw.vram_bytes > 0:
            gpu_tag = f" ({hw.gpu_name})" if hw.gpu_name else ""
            parts_en.append(f"Dedicated VRAM: {hw.vram_gb:.1f} GB{gpu_tag}")
            parts_zh.append(f"独立显存：{hw.vram_gb:.1f} GB{gpu_tag}")
        if hw.system_ram_bytes > 0:
            parts_en.append(f"System RAM: {hw.system_ram_gb:.0f} GB")
            parts_zh.append(f"系统内存：{hw.system_ram_gb:.0f} GB")
        if hw.free_disk_bytes > 0:
            parts_en.append(f"Free disk: {hw.free_disk_gb:.1f} GB")
            parts_zh.append(f"可用磁盘：{hw.free_disk_gb:.1f} GB")
        parts_en.append(f"Recommended for this PC: {rec_name} ({rec_id})")
        parts_zh.append(f"本机推荐模型：{rec_name} ({rec_id})")
        self.hardware_label.setText(tr(" · ".join(parts_en), " · ".join(parts_zh)))

    def _fit_badge(self, model_id: str, approx_bytes: int, is_installed: bool, rec_id: str) -> tuple[str, str]:
        curated = get_curated_model(model_id)
        fit = classify_hardware_fit(
            curated if curated is not None else model_id,
            self._hardware,
            is_installed=is_installed,
            installed_size_bytes=approx_bytes,
        )
        is_rec = model_id == rec_id
        if fit.status == "disk_low":
            req_gb = round(fit.required_disk_bytes / 1e9)
            return (
                tr(f"Low disk (<{req_gb} GB)", f"磁盘空间不足 (<{req_gb} GB)"),
                "#f08b8b",
            )
        if fit.status == "fits_vram":
            if is_rec:
                return (tr("★ Recommended · Fits VRAM", "★ 本机推荐 · 显存充足"), "#8fd19e")
            return (tr("Fits in VRAM", "显存可完整加载"), "#8fd19e")
        if fit.status == "cpu_offload":
            if is_rec:
                return (tr("★ Recommended · RAM/CPU split", "★ 本机推荐 · 内存/CPU 混跑"), "#f1c58c")
            return (tr("CPU/RAM offload (slower)", "需借用系统内存（较慢）"), "#f1c58c")
        return (tr("Heavy for this PC", "超出本机建议配置"), "#f08b8b")

    def _effective_rows(self) -> list[DiscoveredModelRow]:
        rows: list[DiscoveredModelRow] = []
        seen: set[str] = set()
        for entry in LOCAL_MODEL_CATALOG:
            seen.add(entry.id)
            if entry.id in self._discovered_rows:
                rows.append(self._discovered_rows[entry.id])
                continue
            v_entry = self.config.verified_models.get(entry.id)
            is_ver = bool(
                (self.config.model == entry.id and self.config.verified_digest)
                or (v_entry and v_entry.get("digest"))
            )
            ver_ok, min_ver = is_ollama_version_compatible(self.runtime.version, entry.id)
            rows.append(
                DiscoveredModelRow(
                    id=entry.id,
                    title_en=entry.title_en,
                    title_zh=entry.title_zh,
                    tier=entry.tier,
                    approx_bytes=entry.approx_download_bytes,
                    installed=is_ver,
                    digest=v_entry["digest"] if v_entry else (self.config.verified_digest if is_ver else ""),
                    has_vision=True,
                    supports_tools=bool(v_entry["supports_tools"]) if v_entry else entry.supports_agent_tools,
                    has_thinking=bool(v_entry["has_thinking"]) if v_entry else entry.has_thinking,
                    verified=is_ver,
                    version_compatible=ver_ok,
                    min_ollama_version=min_ver,
                    use_case_en=entry.use_case_en,
                    use_case_zh=entry.use_case_zh,
                    is_curated=True,
                )
            )
        for mid, d_row in self._discovered_rows.items():
            if mid not in seen and d_row.has_vision:
                rows.append(d_row)
                seen.add(mid)
        if self.config.model not in seen:
            v_entry = self.config.verified_models.get(self.config.model)
            is_ver = bool(self.config.verified_digest or (v_entry and v_entry.get("digest")))
            rows.append(
                DiscoveredModelRow(
                    id=self.config.model,
                    title_en=self.config.model,
                    title_zh=self.config.model,
                    tier="custom",
                    approx_bytes=19_000_000_000,
                    installed=is_ver,
                    digest=self.config.verified_digest,
                    has_vision=True,
                    supports_tools=bool(self.config.supports_tools),
                    has_thinking=bool(self.config.has_thinking),
                    verified=is_ver,
                    version_compatible=True,
                    min_ollama_version="",
                    use_case_en="Custom local vision model.",
                    use_case_zh="自定义本地视觉模型。",
                    is_curated=False,
                )
            )
        return rows

    def _rebuild_model_entries(self) -> None:
        self._syncing_selection = True
        try:
            zh = tr("en", "zh") == "zh"
            rows = self._effective_rows()
            rec_id = recommend_best_model(self._hardware, self._installed_model_ids())
            selected_id = self.config.model

            self.models.blockSignals(True)
            self.models.clear()
            self.model_table.setRowCount(len(rows))

            selected_row_idx = 0
            for idx, row in enumerate(rows):
                if row.id == selected_id:
                    selected_row_idx = idx
                size_gb = row.approx_bytes / 1e9
                fit_text, fit_color = self._fit_badge(row.id, row.approx_bytes, row.installed, rec_id)
                tier_label = (_TIER_LABELS_ZH if zh else _TIER_LABELS_EN).get(row.tier, row.tier)
                tool_badge = tr("Agent v2 tools", "支持工具调用") if row.supports_tools else tr("Single-turn", "单轮分析")
                if not row.version_compatible and row.min_ollama_version:
                    state_text = tr(
                        f"Requires Ollama v{row.min_ollama_version}+",
                        f"需 Ollama v{row.min_ollama_version}+",
                    )
                elif row.verified:
                    state_text = tr("Verified & Ready", "已验证可用")
                elif row.installed:
                    state_text = tr("Installed · Verify image support", "已安装 · 待验证图片能力")
                else:
                    state_text = tr("Not downloaded", "未下载")

                title = row.title_zh if zh else row.title_en
                combo_label = f"{title} ({row.id}) · ~{size_gb:.1f} GB · {fit_text} · {state_text}"
                self.models.addItem(combo_label, row.id)

                cells = (
                    f"{title} ({row.id})" if title != row.id else row.id,
                    tier_label,
                    f"~{size_gb:.1f} GB",
                    fit_text,
                    f"{state_text} · {tool_badge}",
                )
                for col, text_val in enumerate(cells):
                    item = QTableWidgetItem(text_val)
                    item.setData(Qt.ItemDataRole.UserRole, row.id)
                    if col == 3:
                        item.setForeground(QColor(fit_color))
                    self.model_table.setItem(idx, col, item)

            self.models.setCurrentIndex(selected_row_idx)
            self.models.blockSignals(False)
            if rows:
                self.model_table.selectRow(selected_row_idx)
            self._update_selected_model_notes(rows[selected_row_idx] if rows else None)
            if self._hidden_non_vision_count > 0:
                self.hidden_models_label.setText(
                    tr(
                        f"{self._hidden_non_vision_count} non-vision or cloud-proxied model(s) hidden to preserve local image-coaching compatibility.",
                        f"已隐藏 {self._hidden_non_vision_count} 个非视觉或云端代理模型，仅展示本机视觉模型。",
                    )
                )
                self.hidden_models_label.show()
            else:
                self.hidden_models_label.hide()
        finally:
            self._syncing_selection = False

    def _update_selected_model_notes(self, row: DiscoveredModelRow | None) -> None:
        zh = tr("en", "zh") == "zh"
        if row is None:
            self.use_case_label.clear()
            self.version_warning_label.hide()
            return
        use_case = row.use_case_zh if zh else row.use_case_en
        cap_note = (
            tr("Supports multi-step Agent v2 diagnostic tools.", "支持智能体 v2 多步局部查验与沙盒试调工具。")
            if row.supports_tools
            else tr("Uses single-turn structured photo review (max_turns=1).", "使用单轮结构化修图诊断 (max_turns=1)。")
        )
        self.use_case_label.setText(f"{use_case}  {cap_note}".strip())

        ver_ok, min_ver = is_ollama_version_compatible(self.runtime.version, row.id)
        if not ver_ok and min_ver:
            self.version_warning_label.setText(
                tr(
                    f"Ollama v{self.runtime.version} is too old for {row.id} (requires v{min_ver}+). "
                    "Click 'Install / update Ollama…' below to upgrade.",
                    f"当前 Ollama v{self.runtime.version} 版本过低，无法运行 {row.id}（需要 v{min_ver}+）。"
                    "请点击下方“安装 / 更新 Ollama…”进行升级。",
                )
            )
            self.version_warning_label.show()
        else:
            self.version_warning_label.hide()

    def _select_model_id(self, model_id: str) -> None:
        if not model_id or model_id == self.config.model:
            return
        self.config = self.config.with_active_model(model_id)
        d_row = self._discovered_rows.get(model_id)
        if self.config.verified_digest and (
            not self.runtime.version or self.config.engine_version in ("", self.runtime.version)
        ):
            self.state = "ready"
        elif d_row is not None and d_row.installed:
            self.state = "unverified"
        else:
            self.state = "not_downloaded" if self._discovered_rows else "stopped"
        self.progress.setValue(100 if self.state == "ready" else 0)
        self._rebuild_model_entries()
        self.refresh()

    def _on_combo_model_selected(self, index: int) -> None:
        if self._syncing_selection or index < 0:
            return
        model_id = self.models.itemData(index)
        if isinstance(model_id, str):
            self._select_model_id(model_id)

    def _on_table_model_selected(self) -> None:
        if self._syncing_selection:
            return
        items = self.model_table.selectedItems()
        if not items:
            return
        model_id = items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(model_id, str):
            self._select_model_id(model_id)

    def _cancel_and_run_or_defer(self, callback) -> None:
        if self.busy:
            self._pending_action = callback
            if self.cancel is not None:
                self.cancel.set()
            return
        callback()

    def _on_service_mode_changed(self, _index: int = 0) -> None:
        mode = self.service_mode.currentData()
        if mode not in ("managed", "existing") or mode == self.config.service_mode:
            return

        def apply_mode():
            if self.runtime.process is not None and mode == "existing":
                self.runtime.stop()
            new_port = self.port.value()
            if mode == "existing" and new_port == DEFAULT_PORT:
                new_port = EXISTING_DEFAULT_PORT
            elif mode == "managed" and new_port == EXISTING_DEFAULT_PORT:
                new_port = DEFAULT_PORT
            self.port.blockSignals(True)
            self.port.setValue(new_port)
            self.port.blockSignals(False)
            self.config = replace(
                self.config,
                service_mode=mode,
                port=new_port,
                verified_digest="",
                engine_version="",
            )
            self.runtime.version = ""
            self._discovered_rows.clear()
            self._hidden_non_vision_count = 0
            self.invalidate()
            self._update_engine_label()
            self._rebuild_model_entries()
            self.refresh()

        self._cancel_and_run_or_defer(apply_mode)

    def _on_port_changed(self, new_port: int) -> None:
        if new_port == self.config.port:
            return

        def apply_port():
            if self.runtime.process is not None:
                self.runtime.stop()
            self.config = replace(
                self.config,
                port=int(new_port),
                verified_digest="",
                engine_version="",
            )
            self.runtime.version = ""
            self._discovered_rows.clear()
            self._hidden_non_vision_count = 0
            self.invalidate()
            self._update_engine_label()
            self._rebuild_model_entries()
            self.refresh()

        self._cancel_and_run_or_defer(apply_port)

    def _on_folder_edited(self) -> None:
        self.invalidate()
        self._hardware = detect_local_hardware(self.folder.text())
        self._update_hardware_banner()
        self._rebuild_model_entries()

    def _on_calls_changed(self) -> None:
        self.invalidate()

    def invalidate(self) -> None:
        self.state = "stopped"
        self.use.setEnabled(False)

    def refresh(self) -> None:
        parent = self.parent()
        occupied = bool(parent and (getattr(parent, "_network_active", None) or getattr(parent, "_reviewing", False)))
        managed = self.config.service_mode == "managed"
        for control in (
            self.service_mode,
            self.port,
            self.calls,
            self.models,
            self.model_table,
            self.detect,
            self.download,
            self.verify,
            self.install,
        ):
            control.setEnabled(not self.busy and not occupied)
        self.folder.setEnabled(not self.busy and not occupied and managed)
        self.browse.setEnabled(not self.busy and not occupied and managed)
        self.choose_engine.setEnabled(not self.busy and not occupied and managed)
        self.cancel_button.setEnabled(self.busy)

        d_row = self._discovered_rows.get(self.config.model)
        can_delete = bool(
            not self.busy
            and not occupied
            and (
                (d_row is not None and d_row.installed)
                or self.state in ("ready", "unverified")
                or bool(self.config.verified_digest)
            )
        )
        self.delete_button.setEnabled(can_delete)

        self.stop.setEnabled(not self.busy and not occupied and managed and self.runtime.process is not None)
        self.stop.setToolTip(
            tr(
                "Stop the PhotoCoach-managed Ollama process."
                if managed
                else "Managed externally — PhotoCoach does not stop an existing system Ollama service.",
                "停止由 PhotoCoach 启动的本地 Ollama 进程。"
                if managed
                else "外部服务模式 — PhotoCoach 不会停止系统已有的 Ollama 服务。",
            )
        )
        self.use.setEnabled(not self.busy and not occupied and self.state == "ready")

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Choose model folder", "选择模型目录"), self.folder.text())
        if folder:
            self.folder.setText(folder)

    def select_engine(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Choose Ollama", "选择 Ollama"), "", "Ollama (ollama.exe);;All files (*)"
        )
        if path:
            self.runtime.stop()
            self.config = replace(self.config, engine_path=path, verified_digest="", engine_version="")
            self.engine.setText(path)
            self.engine.setToolTip(path)
            self.invalidate()
            self.refresh()

    def delete_selected_model(self, *, confirm: bool = True) -> None:
        parent = self.parent()
        if self.busy or (parent and (getattr(parent, "_network_active", None) or getattr(parent, "_reviewing", False))):
            return
        model_id = self.config.model
        d_row = self._discovered_rows.get(model_id)
        size_gb = (d_row.approx_bytes if d_row is not None else 19_000_000_000) / 1e9
        if confirm:
            if self.config.service_mode == "existing":
                title = tr("Delete shared system model?", "删除系统共享模型？")
                body = tr(
                    f"Delete '{model_id}' (~{size_gb:.1f} GB) from the existing Ollama service on 127.0.0.1:{self.config.port}?\n\n"
                    "Warning: This removes the model from your shared system Ollama installation used by the CLI and other local apps.",
                    f"确定从本机已有 Ollama 服务 (127.0.0.1:{self.config.port}) 中删除“{model_id}”（约 {size_gb:.1f} GB）吗？\n\n"
                    "注意：此操作会从系统共享的 Ollama 模型库中永久移除该模型，其他本地应用或命令行也将无法使用。",
                )
            else:
                title = tr("Delete local model?", "删除本地模型？")
                body = tr(
                    f"Delete '{model_id}' (~{size_gb:.1f} GB) from PhotoCoach's model folder?",
                    f"确定从 PhotoCoach 模型目录中删除“{model_id}”（可释放约 {size_gb:.1f} GB 空间）吗？",
                )
            answer = QMessageBox.question(
                self,
                title,
                body,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.begin("delete")

    def begin(self, operation: str) -> None:
        parent = self.parent()
        if self.busy or (parent and (getattr(parent, "_network_active", None) or getattr(parent, "_reviewing", False))):
            return
        try:
            mode = self.service_mode.currentData() or "managed"
            port = int(self.port.value())
            selected_model = self.models.currentData() or self.config.model
            path_text = self.folder.text().strip()
            if mode == "managed" or path_text:
                path = Path(path_text).expanduser()
                if not path.is_absolute():
                    raise ValueError(tr("Choose an absolute folder path.", "请选择绝对目录路径。"))
                folder_str = str(path)
                changed = (
                    not self.config.model_dir
                    or path.resolve() != Path(self.config.model_dir).resolve()
                    or mode != self.config.service_mode
                    or port != self.config.port
                )
            else:
                folder_str = ""
                changed = mode != self.config.service_mode or port != self.config.port

            if changed and self.runtime.process is not None:
                self.runtime.stop()

            base_cfg = self.config.with_active_model(selected_model)
            self.config = replace(
                base_cfg,
                model_dir=folder_str,
                service_mode=mode,
                port=port,
                call_limit=self.calls.value(),
                verified_digest="" if changed else base_cfg.verified_digest,
                engine_version="" if changed else base_cfg.engine_version,
                verified_models={} if changed else base_cfg.verified_models,
            )
            self.runtime.store.save(self.config)
        except (ValueError, OSError, RuntimeError) as error:
            self.status.setText(str(error))
            return

        self.cancel = threading.Event()
        self._last_operation = operation
        config, cancel = self.config, self.cancel

        def run_op():
            if operation == "download":
                return self.runtime.download(config, cancel, task.signals.progress.emit)
            if operation == "delete":
                return self.runtime.delete_model(config, config.model, cancel)
            res = getattr(self.runtime, operation)(config, cancel)
            if (
                operation == "inspect"
                and "inspect" not in getattr(self.runtime, "__dict__", {})
                and hasattr(self.runtime, "discover_models")
            ):
                try:
                    discovery = self.runtime.discover_models(res[1], cancel)
                    return res[0], res[1], discovery
                except Exception:
                    return res
            return res

        task = _Task(run_op)
        self.worker = task
        task.signals.progress.connect(self.on_progress)
        task.signals.done.connect(self.finished_operation)
        self.progress.setRange(0, 0)
        if operation == "download":
            msg = tr("Downloading and testing…", "正在下载并测试…")
        elif operation == "delete":
            msg = tr("Deleting local model…", "正在删除本地模型…")
        else:
            msg = tr("Checking local engine and model…", "正在检查本地引擎和模型…")
        self.status.setText(msg)
        self.refresh()
        self.pool.start(task)

    def on_progress(self, event: dict[str, Any]) -> None:
        if not self.worker or (self.cancel and self.cancel.is_set()):
            return
        if event.get("status") == "success":
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.status.setText(tr("Download complete. Testing image support…", "下载完成，正在验证图片能力…"))
            return
        total = event.get("total")
        completed = event.get("completed", 0)
        if type(total) is int and type(completed) is int and total > 0:
            percent = event.get("percent")
            if type(percent) is not int:
                percent = max(0, min(100, int(100 * completed / total)))
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
            speed_bps = event.get("speed_bps")
            speed_note = f" · {float(speed_bps) / 1e6:.1f} MB/s" if isinstance(speed_bps, (int, float)) and speed_bps > 0 else ""
            layers = event.get("layers")
            layer_note = (
                (tr(f" ({layers} layers)", f"（{layers} 个分片）"))
                if type(layers) is int and layers > 1
                else ""
            )
            self.status.setText(
                tr("Downloading model layers", "正在下载模型分片")
                + layer_note
                + f" · {completed / 1e9:.2f} / {total / 1e9:.2f} GB · {percent}%{speed_note}"
            )

    def finished_operation(self, result, error) -> None:
        cancelled = bool(self.cancel and self.cancel.is_set())
        self.worker = None
        self.progress.setRange(0, 100)
        if error or cancelled:
            self.state = "failed"
            self.progress.setValue(0)
            code = getattr(error, "code", "")
            min_ver = required_ollama_version_for(self.config.model) or "0.40.0"
            actual_ver = self.runtime.version or tr("current", "当前")
            if code == "version":
                detail = tr(
                    f"Ollama ({actual_ver}) is too old for {self.config.model} (requires v{min_ver}+). "
                    "Click 'Install / update Ollama…' to update, then detect again.",
                    f"当前 Ollama ({actual_ver}) 版本过低，无法运行 {self.config.model}（需 v{min_ver}+）。"
                    "请点击“安装 / 更新 Ollama…”升级后重新检测。",
                )
                self.version_warning_label.setText(detail)
                self.version_warning_label.show()
            else:
                messages = {
                    "vision": "图片识别或结构化输出验证未通过。",
                    "connection": "本地引擎不可用，请重新检测。",
                    "timeout": "本地操作超时，没有自动重试。",
                    "busy": "另一项本地模型操作仍在运行。",
                    "response": "本地引擎返回了不完整或无效的数据。",
                    "model": "所选本地模型尚未安装或不可用。",
                    "http": "本地引擎拒绝了操作，请检查引擎版本和模型。",
                }
                detail = tr(str(error), messages.get(code, str(error)))
            self.status.setText(
                tr("Cancelled. Partial downloads can be resumed.", "已取消，可以继续下载已有的部分。")
                if cancelled
                else detail
            )
        else:
            discovery: LocalDiscoveryResult | None = None
            if isinstance(result, tuple) and len(result) == 3:
                self.state, self.config, discovery = result
            else:
                self.state, self.config = result
            if discovery is not None:
                self._discovered_rows = {row.id: row for row in discovery.models}
                self._hidden_non_vision_count = discovery.hidden_non_vision_count
            elif self._last_operation == "delete":
                prev_row = self._discovered_rows.get(self.config.model)
                if prev_row is not None:
                    self._discovered_rows[self.config.model] = replace(
                        prev_row, installed=False, verified=False, digest=""
                    )
            elif self.state in ("ready", "unverified"):
                prev_row = self._discovered_rows.get(self.config.model)
                if prev_row is not None:
                    self._discovered_rows[self.config.model] = replace(
                        prev_row,
                        installed=True,
                        verified=(self.state == "ready"),
                        digest=self.config.verified_digest or prev_row.digest,
                        supports_tools=bool(self.config.supports_tools),
                        has_thinking=bool(self.config.has_thinking),
                    )
            try:
                self.runtime.store.save(self.config)
            except OSError:
                self.state = "failed"
                self.status.setText(
                    tr("Could not save local settings. Check folder permissions.", "无法保存本地设置，请检查目录权限。")
                )
                self.refresh()
                if self._pending_close is not None:
                    super().done(self._pending_close)
                return
            self.progress.setValue(100 if self.state == "ready" else 0)
            if self._last_operation == "delete":
                self.status.setText(
                    tr(
                        f"Deleted {self.config.model}. Disk space has been freed.",
                        f"已删除 {self.config.model}，磁盘空间已释放。",
                    )
                )
            else:
                en, zh = {
                    "ready": (
                        "Ready · Image recognition and structured output passed.",
                        "可以使用 · 图片识别和结构化输出测试通过。",
                    ),
                    "not_downloaded": (
                        "Model not downloaded. Download and verify when ready.",
                        "尚未下载模型，请点击下载并验证。",
                    ),
                    "unverified": (
                        "Model installed. Verify image support before use.",
                        "模型已安装，请验证图片能力后使用。",
                    ),
                }[self.state]
                self.status.setText(tr(en, zh))
            self._update_engine_label()
            self._hardware = detect_local_hardware(self.config.model_dir or default_models_dir())
            self._update_hardware_banner()
            self._rebuild_model_entries()

        self.refresh()
        if self._pending_action is not None:
            action, self._pending_action = self._pending_action, None
            action()
        if self._pending_close is not None:
            super().done(self._pending_close)

    def stop_engine(self) -> None:
        if self.config.service_mode != "managed":
            return
        self.runtime.stop()
        self.state = "stopped"
        self.status.setText(tr("Engine stopped. Downloaded models are retained.", "引擎已停止，已下载模型保留。"))
        self.refresh()

    def use_model(self) -> None:
        if self.state == "ready" and not self.busy:
            self.selected.emit(self.config)

    def done(self, result: int) -> None:
        if self.busy:
            self._pending_close = result
            if self.cancel is not None:
                self.cancel.set()
        else:
            super().done(result)
