"""Readable coach feedback; provider text is always escaped before display."""
from html import escape
import re

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextOption, QColorSpace, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QHBoxLayout,
                              QLabel, QPushButton, QSizePolicy, QTextBrowser,
                              QVBoxLayout, QWidget, QPlainTextEdit, QMenu, QToolButton)

from .coach import CoachFeedback
from .coach_agent import region_label
from .budget import SessionBudget
from .color_mixer import COLOR_KEYS, COLOR_NAMES


class QuestionInput(QPlainTextEdit):
    """A bounded multi-line draft, submitted explicitly with Ctrl+Enter."""
    submitRequested = Signal()
    limit = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._trimming = False
        self.setTabChangesFocus(True)
        self.setAccessibleName("Question for coach")
        self.textChanged.connect(self._trim)
        self.blockCountChanged.connect(self._resize_lines)
        self._resize_lines()

    def text(self):
        return self.toPlainText()

    def setText(self, text):
        self.setPlainText(text[:self.limit])

    def insert(self, text):
        available = self.limit - len(self.text()) + len(self.textCursor().selectedText())
        self.insertPlainText(text[:max(0, available)])

    def insertFromMimeData(self, source):
        self.insert(source.text())

    def _trim(self):
        if self._trimming or len(self.text()) <= self.limit:
            return
        from PySide6.QtGui import QTextCursor
        self._trimming = True
        cursor = self.textCursor()
        cursor.setPosition(len(self.text()[:self.limit].encode("utf-16-le"))//2)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        self._trimming = False

    def _resize_lines(self, *_):
        lines = min(4, max(2, self.blockCount()))
        self.setFixedHeight(self.fontMetrics().lineSpacing()*lines+18)

    def sizeHint(self):
        size = super().sizeHint()
        size.setHeight(self.minimumHeight())
        return size

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.submitRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


def _text(value: str) -> str:
    return escape(value, quote=True).replace("\n", "<br>")


CONTROL_NAMES = {"exposure_ev": "Exposure", "white_balance": "White balance",
                 "composition": "Crop and straighten",
                 "highlights": "Highlights", "shadows": "Shadows",
                 "whites": "Whites", "blacks": "Blacks", "vibrance": "Vibrance",
                 "contrast": "Contrast", "saturation": "Saturation", "rgb_curve": "RGB curve"}
CONTROL_NAMES.update({'lens_corrections': 'Lens corrections',
                      'lens_corrections.distortion': 'Distortion',
                      'lens_corrections.vignetting_ev': 'Vignetting',
                      'detail': 'Detail',
                      'detail.sharpening': 'Sharpening',
                      'detail.masking': 'Edge masking',
                      'detail.luminance_nr': 'Luminance NR',
                      'detail.chroma_nr': 'Color NR'})
CONTROL_NAMES["color_mixer"] = "Color Mixer"
CONTROL_NAMES['local_gradients']='Local adjustments'
CONTROL_NAMES.update({f"color_mixer.{key}": f"Color Mixer · {name}" for key, name in zip(COLOR_KEYS, COLOR_NAMES)})


_DUP_LABEL = re.compile(
    r"\b(Exposure|White balance|RGB curve)\s*\(\s*(exposure_ev|white_balance|rgb_curve|Exposure|White balance|RGB curve)\s*\)",
    re.I,
)


def _display_text(value: str) -> str:
    # Display labels only; the original reply and request parameter names stay intact.
    def _collapse(match: re.Match[str]) -> str:
        inner = CONTROL_NAMES.get(match[2].lower(), match[2])
        return match[1] if inner.lower() == match[1].lower() else match[0]

    value = _DUP_LABEL.sub(_collapse, value)
    return re.sub(r"\b(exposure_ev|white_balance|rgb_curve)\b",
                  lambda match: CONTROL_NAMES[match[0]], value)


def _format_duration(ms: float) -> str:
    ms = max(0.0, float(ms))
    if ms >= 60_000.0:
        mins = int(ms // 60_000.0)
        secs = (ms % 60_000.0) / 1000.0
        return f"{mins}m {secs:.0f}s"
    if ms >= 1000.0:
        return f"{ms / 1000.0:.1f}s"
    return f"{ms:.0f}ms"


def _turn_number(name: str) -> str:
    match = re.search(r"turn_(\d+)", str(name or ""))
    return match.group(1) if match else ""


def _friendly_tool_name(name: str, zh: bool) -> str:
    labels = {
        "inspect_region_100pct": ("100% crop", "100% 局部查验"),
        "inspect_raw_headroom": ("RAW headroom", "RAW 高光余量"),
        "analyze_color_zones": ("color zones", "色彩分区"),
        "detect_scene_lighting_anchors": ("lighting anchors", "光源锚点"),
        "build_lighting_rig": ("3D lighting rig", "3D 布光配方"),
        "search_editing_playbook": ("playbook search", "知识库检索"),
        "propose_and_preview_trial": ("sandbox trial", "沙盒试调"),
    }
    pair = labels.get(name)
    return (pair[1] if zh else pair[0]) if pair else str(name)


_TRANSPORT_STAGE_LABELS = {
    "connecting": ("Connecting securely (DNS/TCP/TLS)", "正在建立加密连接（DNS/TCP/TLS）"),
    "sending": ("Uploading photo previews & context", "正在上传照片预览与上下文"),
    "waiting": ("Waiting for model response", "正在等待模型响应"),
    "receiving": ("Receiving structured response", "正在接收模型回复"),
    "local_preparing": ("Preparing local engine and model", "正在准备本地推理引擎与模型"),
    "local_running": ("Loading local model & analyzing photo", "正在加载本地模型并分析照片"),
}


def format_live_agent_step(
    turn: int,
    max_turns: int,
    step_label: str,
    active_detail: str,
    stage: str,
    *,
    zh: bool = False,
) -> tuple[str, str]:
    """Return (active_step_line, stage_hint) for the live Coach activity timeline."""
    stage_pair = _TRANSPORT_STAGE_LABELS.get(stage)
    stage_text = (stage_pair[1] if zh else stage_pair[0]) if stage_pair else str(stage)
    if turn <= 0:
        step = "正在准备照片预览与上下文…" if zh else "Preparing photo previews & context…"
        return step, stage_text
    prefix = f"第 {turn}/{max_turns} 轮 · " if zh else f"Turn {turn}/{max_turns} · "
    if step_label == "Reasoning and selecting tools":
        return prefix + ("正在推理并选择工具…" if zh else "Reasoning and selecting tools…"), stage_text
    if step_label == "Finalizing structured guidance":
        return prefix + ("正在生成结构化修图建议…" if zh else "Finalizing structured guidance…"), stage_text
    if step_label in _TRANSPORT_STAGE_LABELS:
        return stage_text + "…", ""
    if step_label:
        tool_label = _friendly_tool_name(step_label, zh)
        detail = active_detail
        if step_label == "propose_and_preview_trial" and active_detail:
            mapped = [CONTROL_NAMES.get(p.strip(), p.strip()) for p in active_detail.split(",") if p.strip()]
            detail = ", ".join(mapped)
        if zh:
            msg = prefix + f"正在执行{tool_label}" + (f"（{detail}）" if detail else "") + "…"
            hint = "正在本地执行诊断或试调渲染工具…"
        else:
            msg = prefix + f"Running {tool_label}" + (f" ({detail})" if detail else "") + "…"
            hint = "Executing local diagnostic tool & checks…"
        return msg, hint
    return prefix + ("处理中…" if zh else "Working…"), stage_text


def format_agent_step_line(span, zh: bool = False, *, include_duration: bool = True) -> str:
    """Format one TraceSpan into a concise, human-readable timeline step line."""
    node = str(getattr(span, "node", "") or "")
    name = str(getattr(span, "name", "") or "")
    status = str(getattr(span, "status", "ok") or "ok")
    duration_ms = float(getattr(span, "duration_ms", 0.0) or 0.0)
    detail = getattr(span, "detail", {}) or {}
    if not isinstance(detail, dict):
        detail = {}

    text = name
    if node == "observe":
        tools_n = detail.get("tools")
        if zh:
            text = "已准备照片预览与编辑上下文" + (f"（{tools_n} 个可用工具）" if tools_n else "")
        else:
            text = "Prepared photo previews & editing context" + (f" ({tools_n} tools)" if tools_n else "")
    elif node == "reason":
        t_num = _turn_number(name)
        prefix = (f"第 {t_num} 轮 · " if t_num else "") if zh else (f"Turn {t_num} · " if t_num else "")
        calls = detail.get("calls")
        if isinstance(calls, (list, tuple)) and calls:
            joined = ", ".join(_friendly_tool_name(str(c), zh) for c in calls)
            text = prefix + (f"规划调用工具：{joined}" if zh else f"Planned tools: {joined}")
        elif name.endswith("_final"):
            text = prefix + ("生成结构化修图建议" if zh else "Synthesized structured coaching guidance")
        elif name.endswith("_prose"):
            text = prefix + ("回复未符合 JSON 结构，请求重试格式" if zh else "Unstructured reply; requesting strict JSON")
        else:
            text = prefix + ("推理分析" if zh else "Reasoning step")
    elif node in ("tool_call", "trial_render"):
        if name == "inspect_region_100pct":
            x = detail.get("x_norm")
            y = detail.get("y_norm")
            coord = f" ({float(x):.2f}, {float(y):.2f})" if x is not None and y is not None else ""
            text = (f"查验 100% 原生局部细节{coord}" if zh
                    else f"Inspected 100% native RAW crop{' at' + coord if coord else ''}")
        elif name == "inspect_raw_headroom":
            x = detail.get("x_norm")
            y = detail.get("y_norm")
            coord = f" ({float(x):.2f}, {float(y):.2f})" if x is not None and y is not None else ""
            text = (f"检测 Bayer CFA 传感器高光余量{coord}" if zh
                    else f"Checked Bayer CFA highlight headroom{' at' + coord if coord else ''}")
        elif name == "analyze_color_zones":
            region = detail.get("region", "whole")
            text = (f"分析 8 色相带与 CIELAB 色彩分布 ({region})" if zh
                    else f"Analyzed 8-band HSL & CIELAB color zones ({region})")
        elif name == "detect_scene_lighting_anchors":
            text = "检测画面光源锚点与前景主体 [1..N]" if zh else "Detected lighting anchors & subject boxes [1..N]"
        elif name == "build_lighting_rig":
            recipe = str(detail.get("recipe", "spotlight") or "spotlight")
            direction = str(detail.get("direction", "auto") or "auto")
            text = (f"生成 3D 布光配方（{recipe}, {direction}）" if zh
                    else f"Built 3D lighting rig ({recipe}, {direction})")
        elif name == "search_editing_playbook":
            query = str(detail.get("query", "") or "")
            if zh:
                text = f"检索修图知识库：{query}" if query else "检索修图知识库"
            else:
                text = f"Searched editing playbook: {query}" if query else "Searched editing playbook"
        elif name == "propose_and_preview_trial":
            raw_params = detail.get("params") or ()
            nice_params = [CONTROL_NAMES.get(str(p), str(p)) for p in raw_params]
            joined = ", ".join(nice_params)
            if zh:
                text = f"渲染沙盒试调预览（{joined}）" if joined else "渲染沙盒试调预览"
            else:
                text = f"Rendered sandbox trial preview ({joined})" if joined else "Rendered sandbox trial preview"
        if status != "ok" and detail.get("error"):
            err = str(detail["error"])
            text += f" [{err}]"
    elif node == "reflect":
        if status == "ok":
            text = "核验试调质量指标 · 通过" if zh else "Verified trial quality checks · Passed"
        else:
            reasons = ", ".join(str(r) for r in (detail.get("reasons") or ())[:2])
            if zh:
                text = f"试调自检发现问题（{reasons}）· 触发参数自修复" if reasons else "试调自检未通过 · 触发参数自修复"
            else:
                text = f"Trial self-check flagged issue ({reasons}) · Queued self-repair" if reasons else "Trial self-check flagged issue · Queued self-repair"
    elif node == "finalize":
        text = "完成建议核验并输出" if zh else "Finalized verified coaching advice"
    elif node == "abort":
        text = f"提前结束当前轮次 ({name})" if zh else f"Stopped early ({name})"

    if include_duration and duration_ms > 0:
        return f"{text} · {_format_duration(duration_ms)}"
    return text


def _span_icon_and_color(status: str) -> tuple[str, str]:
    if status == "ok":
        return "✓", "#8fd19e"
    if status == "blocked":
        return "↻", "#f1c58c"
    return "!", "#f08b8b"


def _render_trace_tree_html(spans, zh: bool, *, active_step_line: str = "", stage_hint: str = "") -> str:
    rows: list[str] = []
    icon_font = "font-family:'Segoe UI Symbol','Segoe UI','Microsoft YaHei';"
    for span in spans or ():
        icon, color = _span_icon_and_color(str(getattr(span, "status", "ok") or "ok"))
        body = format_agent_step_line(span, zh, include_duration=False)
        duration_ms = float(getattr(span, "duration_ms", 0.0) or 0.0)
        dur_html = f' <span style="color:#8a8a8a;">· {_text(_format_duration(duration_ms))}</span>' if duration_ms > 0 else ""
        node = str(getattr(span, "node", "") or "")
        indent = "14px" if node in ("tool_call", "trial_render", "reflect") else "0px"
        rows.append(
            f'<p style="margin:1px 0 2px {indent}; color:#cccccc; font-size:12px;">'
            f'<span style="{icon_font} color:{color}; font-weight:bold;">{_text(icon)}</span> '
            f'{_text(body)}{dur_html}</p>'
        )
    if active_step_line:
        rows.append(
            f'<p style="margin:2px 0 1px 0px; color:#9fc6ff; font-size:12px;">'
            f'<span style="{icon_font} font-weight:bold;">▸</span> {_text(active_step_line)}</p>'
        )
    if stage_hint:
        rows.append(
            f'<p style="margin:1px 0 1px 14px; color:#9a9a9a; font-size:11px;">'
            f'{_text(stage_hint)}</p>'
        )
    if not rows:
        return ""
    return (
        '<table border="0" cellspacing="0" cellpadding="0" style="margin:2px 0 6px 4px;">'
        '<tr>'
        '<td width="2" bgcolor="#435266"></td>'
        '<td style="padding-left:9px;">'
        + "".join(rows)
        + '</td></tr></table>'
    )


def agent_trace_summary(trace, zh: bool = False) -> str:
    """Build a concise one-line Agent v2 badge from TraceSpan records."""
    spans = tuple(trace or ())
    if not spans:
        return ""
    turns = sum(1 for s in spans if getattr(s, "node", "") == "reason")
    raw_tools = [
        str(getattr(s, "name", "") or "tool")
        for s in spans
        if getattr(s, "node", "") in ("tool_call", "trial_render")
    ]
    unique_tools: list[str] = []
    for name in raw_tools:
        friendly = _friendly_tool_name(name, zh)
        if friendly not in unique_tools:
            unique_tools.append(friendly)
    reflect_statuses = [
        str(getattr(s, "status", ""))
        for s in spans
        if getattr(s, "node", "") == "reflect"
    ]
    shown_tools = unique_tools[:4]
    tools_chain = " → ".join(shown_tools)
    if len(unique_tools) > 4:
        tools_chain += f" (+{len(unique_tools) - 4})"
    reflect_label = ""
    if reflect_statuses:
        if reflect_statuses[-1] == "ok" and any(st == "blocked" for st in reflect_statuses[:-1]):
            reflect_label = "试调自检：已自修复" if zh else "Trial self-check: repaired"
        elif reflect_statuses[-1] == "ok":
            reflect_label = "试调自检：通过" if zh else "Trial self-check: passed"
        else:
            reflect_label = "试调自检：需复核" if zh else "Trial self-check: needs review"
    if zh:
        parts = ["智能体 v2"]
        if turns:
            parts.append(f"{turns} 轮")
        if raw_tools:
            parts.append(f"{len(raw_tools)} 次工具：{tools_chain}")
        else:
            parts.append("0 次工具")
        if reflect_label:
            parts.append(reflect_label)
        return " · ".join(parts)
    parts = ["Agent v2"]
    if turns:
        parts.append(f"{turns} turn" if turns == 1 else f"{turns} turns")
    if raw_tools:
        noun = "tool" if len(raw_tools) == 1 else "tools"
        parts.append(f"{len(raw_tools)} {noun}: {tools_chain}")
    else:
        parts.append("0 tools")
    if reflect_label:
        parts.append(reflect_label)
    return " · ".join(parts)


def _body(value: str) -> str:
    return _text(_display_text(value))


def related_control(next_step: str, gradients=()) -> str | None:
    """Offer navigation only for one unambiguous, supported control name."""
    if re.search(
        r'\b(?:gradient|mask|radial|local adjustment|spotlight|key light|sunbeam|window light|backlight glow|sky filter|foreground fill)s?\b|'
        r'渐变|蒙版|局部|径向|椭圆|人造光|聚光|主光|丁达尔|耶稣光|窗户光|侧光|逆光',
        next_step, re.I
    ):
        matches=[g for g in gradients if re.search(r'(?<!\w)'+re.escape(g.name)+r'(?!\w)',next_step,re.I) or
                 re.search(r'(?<!\w)'+re.escape(g.id)+r'(?!\w)',next_step)]
        return f'local_gradients.{matches[0].id}' if len(matches)==1 else 'local_gradients'
    lens_targets = [name for name, pattern in (
        ('distortion', r'\bdistortion\b|畸变'), ('vignetting_ev', r'\bvignett(?:e|ing)(?:_ev)?\b|暗角'))
        if re.search(pattern, next_step, re.I)]
    if lens_targets:
        return 'lens_corrections.' + lens_targets[0] if len(lens_targets) == 1 else 'lens_corrections'
    detail_targets = [name for name, pattern in (
        ('sharpening', r'(?<!\btry )\bsharpen(?:ing)?\b|锐化'),
        ('masking', r'\b(?:edge[ _])?masking\b|边缘蒙版'),
        ('luminance_nr', r'\b(?:luminance[ _](?:nr|noise)|luma[ _]nr)\b|明度降噪'),
        ('chroma_nr', r'\b(?:color[ _](?:nr|noise)|chroma[ _](?:nr|noise))\b|彩噪|彩色降噪'))
        if re.search(pattern, next_step, re.I)]
    if detail_targets:
        return 'detail.' + detail_targets[0] if len(detail_targets) == 1 else 'detail'
    if re.search(r"\b(?:color[ _]mixer|hsl)\b|颜色混合器|分色", next_step, re.I):
        colors = [key for key in COLOR_KEYS if re.search(rf"\b{key}\b", next_step, re.I)]
        chinese_colors = dict(zip(COLOR_KEYS, ("红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "洋红")))
        colors = list(dict.fromkeys(colors + [key for key, text in chinese_colors.items() if text in next_step]))
        return f"color_mixer.{colors[0]}" if len(colors) == 1 else "color_mixer"
    patterns = {"composition": r"\b(?:crop|straighten|composition)\b",
                "exposure_ev": r"\bexposure(?:_ev)?\b",
                "white_balance": r"\b(?:white[ _]balance|wb)\b",
                "contrast": r"\bcontrast\b", "saturation": r"\bsaturation\b",
                "highlights": r"\bhighlights\b", "shadows": r"\bshadows\b",
                "whites": r"\bwhites\b", "blacks": r"\bblacks\b", "vibrance": r"\bvibrance\b",
                "rgb_curve": r"\b(?:rgb[ _])?curve\b"}
    chinese = {"composition": "裁剪", "exposure_ev": "曝光", "white_balance": "白平衡", "contrast": "对比度",
               "highlights": "高光", "shadows": "阴影",
               "whites": "白色", "blacks": "黑色", "vibrance": "自然饱和度",
               "saturation": "饱和度", "rgb_curve": "曲线"}
    found = [name for name, pattern in patterns.items()
             if re.search(pattern, next_step, re.I) or (chinese[name] in next_step and
                 not (name == "saturation" and "自然饱和度" in next_step))]
    return found[0] if len(found) == 1 else None


class ElidedLabel(QLabel):
    """A one-line filename or coach summary with full text in its tooltip."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._full_text = ""
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def setText(self, text):
        self._full_text = text
        self.setToolTip(text)
        self._fit()

    def _fit(self):
        super().setText(self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideRight, max(0, self.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class CoachOutput(QTextBrowser):
    """Question-bound answers, practical steps and optional supporting observations."""
    feedbackAvailable = Signal(bool)
    summaryChanged = Signal(str)
    locateRequested = Signal(str)
    applyRequested = Signal(int)
    contentChanged = Signal()

    def __init__(self, parent=None, *, reading=False):
        super().__init__(parent)
        self._feedback = None
        self._notice_kind = "info"
        self._question_draft = None
        self._superseded_question = False
        self._superseded_coach = False
        self.reference_revision = None
        self.evidence_expanded = False
        self.trace_expanded = False
        self._reading = reading
        self._gradients = ()
        self._values = None
        self._preview_pixels = None
        self._actions = ()
        self._lang = "en"
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self._follow_link)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.document().setDocumentMargin(18 if reading else 6)
        self.setMinimumHeight(110)
        self.setStyleSheet("QTextBrowser { background:#252525; color:#e4e4e4; border:1px solid #3d3d3d; border-radius:4px; }")
        self.setAccessibleName("Photo coach response")

    def setPlainText(self, text, *, kind="info"):
        self._notice_kind = kind
        self._feedback = None
        self._actions = ()
        self.reference_revision = None
        self.evidence_expanded = False
        self.trace_expanded = False
        super().setPlainText(text)
        self.feedbackAvailable.emit(False)
        self.summaryChanged.emit(text.splitlines()[0] if text else "Coach ready")
        self.contentChanged.emit()

    def show_live_progress(
        self,
        headline: str,
        *,
        spans=(),
        active_step: str = "",
        stage_hint: str = "",
        notice: str = "",
        zh: bool = False,
    ) -> None:
        """Render the live step-by-step Coach execution tree while running."""
        self._notice_kind = "info"
        self._feedback = None
        self._actions = ()
        self.reference_revision = None
        self.evidence_expanded = False
        self.trace_expanded = False
        vsb = self.verticalScrollBar()
        was_at_bottom = vsb.value() >= vsb.maximum() - 4
        prev_scroll = vsb.value()
        headline_html = f'<p style="margin:3px 0 4px;"><b>{_text(headline)}</b></p>' if headline else ""
        tree_html = _render_trace_tree_html(spans, zh, active_step_line=active_step, stage_hint=stage_hint)
        notice_html = f'<p style="color:#bcbcbc; margin-top:8px;">{_text(notice)}</p>' if notice else ""
        self.setHtml(
            "<style>body, p { color:#e4e4e4; margin:2px 0 4px; }</style>"
            f"{headline_html}{tree_html}{notice_html}"
        )
        new_vsb = self.verticalScrollBar()
        if was_at_bottom or new_vsb.maximum() <= 0:
            new_vsb.setValue(new_vsb.maximum())
        else:
            new_vsb.setValue(min(prev_scroll, new_vsb.maximum()))
        self.feedbackAvailable.emit(False)
        self.summaryChanged.emit(headline.splitlines()[0] if headline else "Coach ready")
        self.contentChanged.emit()

    def show_feedback(self, feedback: CoachFeedback):
        self._notice_kind = "info"
        self._feedback = feedback
        self._superseded_question = False
        self._superseded_coach = False
        self._actions = feedback.actions
        self.reference_revision = None
        self.evidence_expanded = False
        self.trace_expanded = False
        self._render_feedback()
        self.verticalScrollBar().setValue(0)
        self.feedbackAvailable.emit(True)
        self._summarize()

    def set_question_draft(self, question: str) -> None:
        if self._feedback and question != self._feedback.question:
            self._superseded_question = True
        self._question_draft = question
        if self._feedback is not None:
            position = self.verticalScrollBar().value()
            self._render_feedback()
            self.verticalScrollBar().setValue(position)
            self._summarize()

    def set_edit_values(self, values) -> None:
        gradients = values.local_gradients if values is not None else ()
        if values != self._values or gradients != self._gradients:
            self._values = values
            self._gradients = gradients
            if self._feedback is not None:
                position = self.verticalScrollBar().value()
                self._render_feedback()
                self.verticalScrollBar().setValue(position)
            else:
                self.contentChanged.emit()

    def set_preview_pixels(self, pixels) -> None:
        self._preview_pixels = pixels

    def set_language(self, lang: str) -> None:
        self._lang = lang
        if self._feedback is not None:
            position = self.verticalScrollBar().value()
            self._render_feedback()
            self.verticalScrollBar().setValue(position)

    def _question_changed(self) -> bool:
        return (self._feedback is not None and self._question_draft is not None and
                self._question_draft != self._feedback.question)

    def keep_as_reference(self, photo_id: str, session_id: str, current_revision: int, *, coach_changed=False) -> bool:
        feedback = self._feedback
        if feedback is None or (feedback.photo_id, feedback.session_id) != (photo_id, session_id):
            return False
        self._superseded_coach = self._superseded_coach or coach_changed
        if current_revision == feedback.state_version and not coach_changed:
            return True
        self.reference_revision = current_revision
        position = self.verticalScrollBar().value()
        self._render_feedback()
        self.verticalScrollBar().setValue(position)
        self._summarize()
        return True

    def _summarize(self):
        feedback = self._feedback
        earlier = self.reference_revision is not None or self._superseded_question or self._superseded_coach or self._question_changed()
        prefix = f"Earlier advice · v{feedback.state_version}" if earlier else f"State v{feedback.state_version}"
        summary = " ".join(_display_text(feedback.next_step_or_stop).split())
        self.summaryChanged.emit(f"{prefix} · {summary}")

    def _follow_link(self, link):
        # Provider text is escaped. Only our disclosure and navigation links work.
        target = link.toString()
        if target == "photocoach:evidence" and self._feedback is not None:
            position = self.verticalScrollBar().value()
            self.evidence_expanded = not self.evidence_expanded
            self._render_feedback()
            self.verticalScrollBar().setValue(position)
        elif target == "photocoach:trace" and self._feedback is not None:
            position = self.verticalScrollBar().value()
            self.trace_expanded = not self.trace_expanded
            self._render_feedback()
            self.verticalScrollBar().setValue(position)
        elif self._feedback is not None:
            control = related_control(self._feedback.next_step_or_stop,self._gradients)
            if control and target == f"photocoach:locate/{control}":
                self.locateRequested.emit(control)

    def response_text(self) -> str:
        """Copy the complete structured reply, including hidden observations."""
        feedback = self._feedback
        if feedback is None:
            return self.toPlainText()
        lines = [f"{feedback.source} · State v{feedback.state_version}"]
        if self._superseded_coach:
            lines.append("Coach changed. This reply is reference only.")
        if self.reference_revision is not None:
            lines.append(f"Earlier advice · based on v{feedback.state_version}. "
                         f"Current edit is v{self.reference_revision}; this advice has not been rechecked.")
        if self._question_changed() or self._superseded_question:
            lines.append("Edit or question changed. This reply answers the question shown below.")
        lines.extend([f"Question sent: {feedback.question}" if feedback.question else "Current edit review · No question was sent."])
        for heading, value in (("ANSWER", feedback.answer), ("NEXT STEP", feedback.next_step_or_stop),
                               ("WATCH / STOP", feedback.observe_and_stop_when),
                               ("WHY THIS HELPS", feedback.reason), ("WHAT CHANGED", feedback.main_change)):
            if value:
                lines.extend(["", heading, _display_text(value)])
        lines.extend(_display_text(action.label) for action in self._actions)
        if feedback.action_notice:
            lines.append(feedback.action_notice)
        lines.extend(["", *(_display_text(item) for item in feedback.evidence)])
        if feedback.agent_plan:
            plan = feedback.agent_plan
            lines.extend(["", f"WORKFLOW: {plan.decision} · {plan.objective}"])
            lines.extend(f"{f.region}: {f.observation} · {f.interpretation} ({f.category}, {f.confidence})"
                         for f in plan.findings)
            lines.append("Uncertainty: " + plan.uncertainty)
        if feedback.agent_report:
            lines.extend(feedback.agent_report.lines(self._lang == "zh"))
        return "\n".join(lines)

    def _render_feedback(self):
        feedback = self._feedback
        if feedback is None:
            return
        zh = self._lang == "zh"
        def label(en, cn):
            return cn if zh else en
        warning = ""
        applied = bool(self._actions) and self._values is not None and all(a.is_applied(self._values) for a in self._actions)
        if self._superseded_coach:
            warning = label("Coach changed. Analyze again before applying suggestions.", "教练已切换，请重新分析后再应用建议。")
        elif self._superseded_question or self._question_changed() or (self.reference_revision is not None and not applied):
            warning = label("Edit or question changed. Analyze again before applying suggestions.",
                            "编辑或提问已变化，请重新分析后再应用建议。")
        notice = f'<p style="color:#f1c58c;">{warning}</p>' if warning else ""
        question = (f'<p style="color:#bcbcbc;">{_text(feedback.question)}</p>'
                    if feedback.question and (self._reading or not feedback.agent_plan) else "")
        answer = (f'<a name="answer"></a><p style="font-size:14px;">{_body(feedback.answer)}</p>'
                  if feedback.answer else "")
        targets = ""
        if self._actions:
            targets = '<a name="targets"></a><ul>' + "".join(f'<li>{_body(action.label)}</li>' for action in self._actions) + '</ul>'
        if feedback.action_notice:
            targets += f'<p style="color:#f1c58c;">{_text(feedback.action_notice)}</p>'
        workflow = ""
        if feedback.agent_plan and (self._reading or self.evidence_expanded):
            plan = feedback.agent_plan
            decisions = {"recommend": ("Reversible trial", "可撤销试调"), "stop": ("Stop", "停止调整"),
                         "inspect": ("Inspect first", "先检查照片"), "clarify": ("Clarify the goal", "先明确目标")}
            workflow = (f'<p style="color:#9fc6ff;"><b>{label(*decisions[plan.decision])}</b> · {_body(plan.objective)}</p>'
                        + "".join(f'<p>{_text(region_label(f.region, zh))}: {_body(f.observation)}</p>' for f in plan.findings))
            if feedback.agent_report:
                workflow += "".join(f'<p style="color:#bcbcbc;">{_text(line)}</p>'
                                    for line in feedback.agent_report.lines(zh))
            elif self._actions:
                workflow += f'<p>{label("Trial checks pending. Apply is unavailable.", "等待试调检查，暂不能应用。")}</p>'
        control = related_control(feedback.next_step_or_stop, self._gradients)
        navigation = ""
        if control:
            title = next((g.name for g in self._gradients if control == f"local_gradients.{g.id}"),
                         CONTROL_NAMES.get(control, "Local adjustments"))
            navigation = f'<p><a href="photocoach:locate/{control}">{label("Go to", "定位到")} {_text(title)} →</a></p>'
        details = ""
        if self.evidence_expanded:
            details = (f'<a name="why"></a><p>{_body(feedback.reason)}</p>'
                       f'<p>{_body(feedback.main_change)}</p>'
                       + "".join(f'<p style="color:#bcbcbc;">{_body(item)}</p>' for item in feedback.evidence)
                       + f'<p style="color:#9a9a9a;">{_text(feedback.source)} · v{feedback.state_version}</p>')
            if feedback.agent_plan:
                details += "".join(f'<p>{_text(f.region)} · {_text(f.category)} · {_text(f.confidence)}: '
                                   f'{_body(f.interpretation)}</p>' for f in feedback.agent_plan.findings)
                details += f'<p>{label("Uncertainty", "不确定性")}: {_body(feedback.agent_plan.uncertainty)}</p>'
            if feedback.agent_report:
                details += f'<p>{_text(" → ".join(feedback.agent_report.trace))}</p>'
            if getattr(feedback, "agent_trace", ()):
                trace_items = []
                for span in feedback.agent_trace:
                    s_name = getattr(span, "name", str(span))
                    s_status = getattr(span, "status", "ok")
                    s_ms = getattr(span, "duration_ms", 0.0)
                    badge = f"{s_name} ({s_status}, {s_ms:.0f}ms)"
                    trace_items.append(_text(badge))
                details += f'<p style="color:#9fc6ff;"><b>{label("Agent trace", "智能体轨迹")}:</b> {" → ".join(trace_items)}</p>'
        trace = tuple(getattr(feedback, "agent_trace", ()) or ())
        agent_summary = agent_trace_summary(trace, zh)
        agent_badge = ""
        if agent_summary:
            total_ms = sum(float(getattr(s, "duration_ms", 0.0) or 0.0) for s in trace)
            worked = (f"耗时 {_format_duration(total_ms)}" if zh else f"Worked for {_format_duration(total_ms)}") if total_ms > 0 else ""
            arrow = "▾" if self.trace_expanded else "▸"
            text_part = f"{agent_summary}" + (f" · {worked}" if worked else "")
            agent_badge = (
                '<p style="margin:3px 0 5px;">'
                f'<a href="photocoach:trace"><span style="font-family:\'Segoe UI Symbol\',\'Segoe UI\',\'Microsoft YaHei\';">{arrow}</span> '
                f'<b>{_text(text_part)}</b></a></p>'
            )
            if self.trace_expanded:
                agent_badge += _render_trace_tree_html(trace, zh)
        toggle = label("Hide evidence and reasoning", "收起依据与推断") if self.evidence_expanded else label("Evidence and reasoning", "依据与推断")
        p_margin = "3px 0 6px" if self._reading else "2px 0 3px"
        self.setHtml(f"""
            <style>body, p {{ color:#e4e4e4; margin:{p_margin}; }} a {{ color:#9fc6ff; text-decoration:none; }}
            ul {{ margin-top:2px; margin-bottom:4px; }}</style>
            {notice}{question}{agent_badge}{answer}{workflow}
            <a name="steps"></a><p style="color:#9fc6ff;"><b>{label("Suggestion", "调整建议")}</b></p>
            <p>{_body(feedback.next_step_or_stop)}</p>{targets}{navigation}
            <a name="stop"></a><p style="color:#9fc6ff;"><b>{label("Check after editing", "调整后检查")}</b></p>
            <p>{_body(feedback.observe_and_stop_when)}</p>
            <p><a href="photocoach:evidence">{toggle}</a></p>{details}
        """)
        self.contentChanged.emit()


class CoachResponsePanel(QWidget):
    """Reading, copying, and explicit 1-click applying of coach suggestions."""
    locateRequested = Signal(str)
    applyRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        actions = QHBoxLayout()
        self._lang = "en"
        self.apply_button = QPushButton("Apply suggestion")
        self.apply_button.setProperty("compact", True)
        self.apply_button.setToolTip("Apply the coach's recommended adjustment or local lighting mask in one undoable step.")
        self.apply_button.clicked.connect(lambda: self.applyRequested.emit(0))
        actions.addWidget(self.apply_button)
        self.trial_button = QPushButton("Preview trial")
        self.trial_button.setProperty("compact", True)
        self.trial_button.clicked.connect(self.open_trial)
        actions.addWidget(self.trial_button)
        self.changes_label = ElidedLabel()
        self.changes_label.setAccessibleName("Suggested edit targets")
        actions.addWidget(self.changes_label, 1)
        self.more_button = QToolButton()
        self.more_button.setText("More")
        self.more_button.setProperty("compact", True)
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        more = QMenu(self.more_button)
        self.changes_action = more.addAction("View changes", self.open_changes)
        self.reader_button = more.addAction("Open reader", self.open_reader)
        self.copy_button = more.addAction("Copy response", self.copy_response)
        self.more_button.setMenu(more)
        actions.addWidget(self.more_button)
        layout.addLayout(actions)
        self.output = CoachOutput()
        self.output.locateRequested.connect(self.locateRequested)
        self.output.applyRequested.connect(self.applyRequested)
        layout.addWidget(self.output, 1)

        # Modeless and window-local: updates never move the splitter or steal
        # focus. No reply/history/credential persistence is added here.
        self.reader = QDialog(self)
        self.reader.setWindowTitle("PhotoCoach · Coach reader")
        self.reader.resize(780, 650)
        reader_layout = QVBoxLayout(self.reader)
        reader_navigation = QHBoxLayout()
        reader_navigation.setSpacing(5)
        self.reader_sections = []
        for title, anchor in (("Answer", "answer"), ("Steps", "steps"), ("Stop", "stop"), ("Why", "why")):
            button = QPushButton(title)
            button.setToolTip(f"Jump to {title.lower()} in this response")
            button.clicked.connect(lambda checked=False, target=anchor: self.reader_output.scrollToAnchor(target))
            reader_navigation.addWidget(button)
            self.reader_sections.append(button)
        reader_navigation.addStretch()
        reader_layout.addLayout(reader_navigation)
        self.reader_output = CoachOutput(reading=True)
        self.reader_output.setAccessibleName("Full photo coach response")
        self.reader_output.locateRequested.connect(self._locate_from_reader)
        self.reader_output.applyRequested.connect(self._apply_from_reader)
        reader_layout.addWidget(self.reader_output, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.reader_close = buttons.button(QDialogButtonBox.StandardButton.Close)
        self.reader_apply = buttons.addButton("Apply suggestion", QDialogButtonBox.ButtonRole.ActionRole)
        self.reader_apply.clicked.connect(lambda: self._apply_from_reader(0))
        self.reader_copy = buttons.addButton("Copy response", QDialogButtonBox.ButtonRole.ActionRole)
        self.reader_copy.clicked.connect(self.copy_response)
        buttons.rejected.connect(self.reader.hide)
        reader_layout.addWidget(buttons)
        self.output.contentChanged.connect(self._sync_reader)
        self._trial_feedback = None
        self.trial = QDialog(self)
        self.trial.setWindowTitle("PhotoCoach · Proposed edit")
        trial_layout = QVBoxLayout(self.trial)
        self.trial_notice = QLabel()
        self.trial_notice.setWordWrap(True)
        trial_layout.addWidget(self.trial_notice)
        image_row = QHBoxLayout()
        self.trial_titles, self.trial_images = [], []
        for _ in range(2):
            column = QVBoxLayout()
            title, image = QLabel(), QLabel()
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image.setMinimumSize(100, 80)
            image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
            column.addWidget(title)
            column.addWidget(image, 1)
            image_row.addLayout(column, 1)
            self.trial_titles.append(title)
            self.trial_images.append(image)
        trial_layout.addLayout(image_row, 1)
        self.trial_crop_box = QWidget()
        crop_layout = QHBoxLayout(self.trial_crop_box)
        crop_layout.setContentsMargins(0, 0, 0, 0)
        self.trial_crop_label = QLabel()
        self.trial_crop_label.setWordWrap(True)
        self.trial_crop_image = QLabel()
        self.trial_crop_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trial_crop_image.setFixedSize(128, 128)
        crop_layout.addWidget(self.trial_crop_image)
        crop_layout.addWidget(self.trial_crop_label, 1)
        self.trial_crop_box.hide()
        trial_layout.addWidget(self.trial_crop_box)
        self.trial_report = QTextBrowser()
        self.trial_report.setMaximumHeight(150)
        trial_layout.addWidget(self.trial_report)
        self.trial_close = QPushButton("Close")
        self.trial_close.clicked.connect(self.trial.hide)
        trial_layout.addWidget(self.trial_close)
        self.trial.resize(900, 580)
        self._copied_timer = QTimer(self)
        self._copied_timer.setSingleShot(True)
        self._copied_timer.timeout.connect(self._reset_copy_label)
        self._sync_reader()

    def _sync_reader(self, *, force=False):
        output, reader = self.output, self.reader_output
        available = output._feedback is not None
        for button in (self.reader_button, self.copy_button, self.reader_copy, *self.reader_sections):
            button.setEnabled(available)
        actions = output._actions if available else ()
        self.changes_label.setText(" · ".join(_display_text(action.label) for action in actions))
        self.changes_label.setVisible(bool(actions))
        self.more_button.setVisible(available)
        self.changes_action.setEnabled(bool(actions))
        applied = bool(actions) and output._values is not None and all(a.is_applied(output._values) for a in actions)
        report = output._feedback.agent_report if available else None
        checked = not (available and output._feedback.agent_plan) or bool(report and report.can_apply)
        can_apply = (available and bool(actions) and not applied and
                     checked and output.reference_revision is None and not output._superseded_question and not output._superseded_coach and not output._question_changed())
        trial_available = bool(report and report.trial_preview is not None)
        self.trial_button.setVisible(trial_available)
        self.trial_button.setEnabled(trial_available and output.reference_revision is None and
                                     not output._superseded_question and not output._superseded_coach and not output._question_changed())
        if hasattr(self, "trial") and (not self.trial_button.isEnabled() or self._trial_feedback is not output._feedback):
            self.trial.hide()
        zh = getattr(self, "_lang", "en") == "zh"
        for btn in (self.apply_button, self.reader_apply):
            btn.setVisible(bool(actions))
            btn.setEnabled(can_apply)
            btn.setText(("已应用 ✓" if zh else "Applied ✓") if (available and applied) else ("应用建议" if zh else "Apply suggestion"))
            btn.setToolTip(f"Apply in one undoable step: {actions[0].label}" if actions else
                           "Apply the coach's recommended adjustment or local lighting mask in one undoable step.")
        self._copied_timer.stop()
        self._reset_copy_label()
        # Editing can invalidate a reply many times per gesture. Do not lay
        # out a second long document until the reader is actually being used.
        # Clearing/changing photos still clears any hidden old response.
        if available and not (force or self.reader.isVisible()):
            return
        same_reply = output._feedback is reader._feedback
        scroll = reader.verticalScrollBar().value() if same_reply else 0
        if output._feedback is None:
            reader.setPlainText(output.toPlainText())
        else:
            reader._feedback = output._feedback
            reader._question_draft = output._question_draft
            reader._superseded_question = output._superseded_question
            reader._superseded_coach = output._superseded_coach
            reader.reference_revision = output.reference_revision
            reader._gradients = output._gradients
            reader._values = output._values
            reader._preview_pixels = output._preview_pixels
            reader._actions = output._actions
            if not same_reply:
                reader.evidence_expanded = False
                reader.trace_expanded = False
            reader._render_feedback()
        reader.verticalScrollBar().setValue(scroll)

    def open_trial(self):
        if not self.trial_button.isEnabled():
            return
        feedback = self.output._feedback
        report = feedback.agent_report
        if report is None or report.trial_preview is None:
            return
        zh = self._lang == "zh"
        self._trial_feedback = feedback
        available = self.screen().availableGeometry()
        width, height = min(900, available.width()-40), min(580, available.height()-60)
        self.trial.resize(max(320, width), max(300, height))
        self.trial.setWindowTitle("PhotoCoach · 试调对比" if zh else "PhotoCoach · Proposed edit")
        self.trial_notice.setText(("试调尚未应用。统计不代表审美评分；比较后由你决定是否应用。" if zh else
                                  "This trial has not been applied. Measurements are not aesthetic scores; compare before deciding."))
        self.trial_titles[0].setText(("当前编辑" if zh else "Current edit") + f" · v{feedback.state_version}")
        self.trial_titles[1].setText("试调 · 尚未应用" if zh else "Trial · Not applied")
        for label, pixels in zip(self.trial_images, (report.current_preview, report.trial_preview)):
            h, w = pixels.shape[:2]
            image = QImage(pixels.data, w, h, pixels.strides[0], QImage.Format.Format_RGB888).copy()
            image.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
            label.setPixmap(QPixmap.fromImage(image).scaled(max(100, (width-60)//2), max(80, height-250),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        crops = getattr(feedback, "inspected_crops", ())
        if crops:
            crop_title, crop_pixels = crops[-1]
            ch, cw = crop_pixels.shape[:2]
            c_img = QImage(crop_pixels.data, cw, ch, crop_pixels.strides[0], QImage.Format.Format_RGB888).copy()
            c_img.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
            self.trial_crop_image.setPixmap(QPixmap.fromImage(c_img).scaled(
                128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            prefix = "智能体核验的 100% 原生局部图" if zh else "100% native crop inspected by agent"
            self.trial_crop_label.setText(f"{prefix}: {crop_title}")
            self.trial_crop_box.show()
        else:
            self.trial_crop_box.hide()
        report_lines = list(report.lines(zh))
        summary = agent_trace_summary(getattr(feedback, "agent_trace", ()), zh)
        if summary:
            report_lines.insert(0, summary)
        self.trial_report.setPlainText("\n".join(report_lines))
        self.trial_close.setText("关闭" if zh else "Close")
        self.trial.show()
        self.trial.raise_()

    def open_reader(self):
        if self.output._feedback is None:
            return
        self._sync_reader(force=True)
        # Fit inside the actual available screen, retaining the chosen size
        # between openings. No forced resize when a new response arrives.
        available = self.screen().availableGeometry()
        self.reader.resize(min(self.reader.width(), max(320, available.width()-32)),
                           min(self.reader.height(), max(240, available.height()-64)))
        self.reader.show()
        self.reader.raise_()

    def open_changes(self):
        self.open_reader()
        self.reader_output.scrollToAnchor("targets")

    def copy_response(self):
        if self.output._feedback is not None:
            QApplication.clipboard().setText(self.output.response_text())
            for button in (self.copy_button, self.reader_copy):
                button.setText("Copied")
            self._copied_timer.start(1500)

    def _reset_copy_label(self):
        zh = getattr(self, "_lang", "en") == "zh"
        for button in (self.copy_button, self.reader_copy):
            button.setText("复制回复" if zh else "Copy response")

    def set_language(self, lang: str) -> None:
        self._lang = lang
        zh = lang == "zh"
        for btn, en, cn in zip(self.reader_sections, ("Answer", "Steps", "Stop", "Why"), ("解答", "步骤", "停止条件", "原因")):
            btn.setText(cn if zh else en)
        self.reader_close.setText("关闭" if zh else "Close")
        self.more_button.setText("回复选项" if zh else "Reply options")
        self.changes_action.setText("查看修改清单" if zh else "View changes")
        self.reader_button.setText("阅读窗口" if zh else "Open reader")
        self.trial_button.setText("预览试调" if zh else "Preview trial")
        self.output.set_language(lang)
        self.reader_output.set_language(lang)
        self._sync_reader(force=True)
        if self.trial.isVisible():
            self.open_trial()

    def _locate_from_reader(self, control):
        self.reader.hide()
        self.locateRequested.emit(control)

    def _apply_from_reader(self, index: int = 0):
        self.reader.hide()
        self.applyRequested.emit(index)


def usage_summary(budget: SessionBudget, mock_calls: int, cloud: bool) -> str:
    if not cloud and not budget.attempts:
        return f"Local demo · {mock_calls} analyses · No cloud requests"
    cost = budget.known_cost_usd
    amount = "Cost unknown" if cost is None else f"Est. ${cost:.4f}"
    active = " · Sending" if budget.in_flight else ""
    return f"{budget.attempts}/{budget.call_limit} API attempts this session · {amount}{active}"
