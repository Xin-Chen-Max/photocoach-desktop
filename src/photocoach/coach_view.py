"""Readable coach feedback; provider text is always escaped before display."""
from html import escape
import re

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextOption
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QHBoxLayout,
                              QLabel, QPushButton, QSizePolicy, QTextBrowser,
                              QVBoxLayout, QWidget, QPlainTextEdit, QMenu, QToolButton)

from .coach import CoachFeedback
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


def _display_text(value: str) -> str:
    # Display labels only; the original reply and request parameter names stay intact.
    return re.sub(r"\b(exposure_ev|white_balance|rgb_curve)\b",
                  lambda match: CONTROL_NAMES[match[0]], value)


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
        self.setStyleSheet("QTextBrowser { background:#252525; border:1px solid #3d3d3d; border-radius:4px; }")
        self.setAccessibleName("Photo coach response")

    def setPlainText(self, text, *, kind="info"):
        self._notice_kind = kind
        self._feedback = None
        self._actions = ()
        self.reference_revision = None
        self.evidence_expanded = False
        super().setPlainText(text)
        self.feedbackAvailable.emit(False)
        self.summaryChanged.emit(text.splitlines()[0] if text else "Coach ready")
        self.contentChanged.emit()

    def show_feedback(self, feedback: CoachFeedback):
        self._notice_kind = "info"
        self._feedback = feedback
        self._superseded_question = False
        self._superseded_coach = False
        self._actions = feedback.actions
        self.reference_revision = None
        self.evidence_expanded = False
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
        question = (f'<p style="color:#bcbcbc;">{_text(feedback.question)}</p>' if feedback.question else "")
        answer = (f'<a name="answer"></a><p style="font-size:14px;">{_body(feedback.answer)}</p>'
                  if feedback.answer else "")
        targets = ""
        if self._actions:
            targets = '<a name="targets"></a><ul>' + "".join(f'<li>{_body(action.label)}</li>' for action in self._actions) + '</ul>'
        if feedback.action_notice:
            targets += f'<p style="color:#f1c58c;">{_text(feedback.action_notice)}</p>'
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
        toggle = label("Hide evidence and reasoning", "收起依据与推断") if self.evidence_expanded else label("Evidence and reasoning", "依据与推断")
        self.setHtml(f"""
            <style>p {{ margin:3px 0 6px; }} a {{ color:#9fc6ff; text-decoration:none; }}
            ul {{ margin-top:3px; margin-bottom:6px; }}</style>
            {notice}{question}{answer}
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
        layout.setSpacing(5)
        actions = QHBoxLayout()
        self._lang = "en"
        self.apply_button = QPushButton("Apply suggestion")
        self.apply_button.setProperty("compact", True)
        self.apply_button.setToolTip("Apply the coach's recommended adjustment or local lighting mask in one undoable step.")
        self.apply_button.clicked.connect(lambda: self.applyRequested.emit(0))
        actions.addWidget(self.apply_button)
        self.changes_label = ElidedLabel()
        self.changes_label.setAccessibleName("Suggested edit targets")
        actions.addWidget(self.changes_label, 1)
        self.more_button = QToolButton()
        self.more_button.setText("More")
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
        can_apply = (available and bool(actions) and not applied and
                     output.reference_revision is None and not output._superseded_question and not output._superseded_coach and not output._question_changed())
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
            reader._render_feedback()
        reader.verticalScrollBar().setValue(scroll)

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
        self.output.set_language(lang)
        self.reader_output.set_language(lang)
        self._sync_reader(force=True)

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
