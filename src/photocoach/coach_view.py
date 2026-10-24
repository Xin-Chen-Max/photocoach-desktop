"""Readable coach feedback; provider text is always escaped before display."""
from html import escape
import re

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextOption
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QHBoxLayout,
                              QLabel, QPushButton, QSizePolicy, QTextBrowser,
                              QVBoxLayout, QWidget)

from .coach import CoachFeedback
from .budget import SessionBudget
from .color_mixer import COLOR_KEYS, COLOR_NAMES


def _text(value: str) -> str:
    return escape(value, quote=True).replace("\n", "<br>")


CONTROL_NAMES = {"exposure_ev": "Exposure", "white_balance": "White balance",
                 "composition": "Crop and straighten",
                 "highlights": "Highlights", "shadows": "Shadows",
                 "whites": "Whites", "blacks": "Blacks", "vibrance": "Vibrance",
                 "contrast": "Contrast", "saturation": "Saturation", "rgb_curve": "RGB curve"}
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
    if re.search(r'\b(?:gradient|mask|radial|local adjustment)s?\b',next_step,re.I):
        matches=[g for g in gradients if re.search(r'(?<!\w)'+re.escape(g.name)+r'(?!\w)',next_step,re.I) or
                 re.search(r'(?<!\w)'+re.escape(g.id)+r'(?!\w)',next_step)]
        return f'local_gradients.{matches[0].id}' if len(matches)==1 else 'local_gradients'
    if re.search(r"\b(?:color[ _]mixer|hsl)\b", next_step, re.I):
        colors = [key for key in COLOR_KEYS if re.search(rf"\b{key}\b", next_step, re.I)]
        return f"color_mixer.{colors[0]}" if len(colors) == 1 else "color_mixer"
    patterns = {"composition": r"\b(?:crop|straighten|composition)\b",
                "exposure_ev": r"\bexposure(?:_ev)?\b",
                "white_balance": r"\b(?:white[ _]balance|wb)\b",
                "contrast": r"\bcontrast\b", "saturation": r"\bsaturation\b",
                "highlights": r"\bhighlights\b", "shadows": r"\bshadows\b",
                "whites": r"\bwhites\b", "blacks": r"\bblacks\b", "vibrance": r"\bvibrance\b",
                "rgb_curve": r"\b(?:rgb[ _])?curve\b"}
    found = [name for name, pattern in patterns.items()
             if re.search(pattern, next_step, re.I)]
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
    contentChanged = Signal()

    def __init__(self, parent=None, *, reading=False):
        super().__init__(parent)
        self._feedback = None
        self._question_draft = None
        self.reference_revision = None
        self.evidence_expanded = False
        self._wide = True
        self._reading = reading
        self._gradients = ()
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self._follow_link)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.document().setDocumentMargin(18 if reading else 12)
        self.setMinimumHeight(110)
        self.setStyleSheet("QTextBrowser { background:#252525; border:1px solid #3d3d3d; border-radius:4px; }")
        self.setAccessibleName("Photo coach response")

    def setPlainText(self, text):
        self._feedback = None
        self.reference_revision = None
        self.evidence_expanded = False
        super().setPlainText(text)
        self.feedbackAvailable.emit(False)
        self.summaryChanged.emit(text.splitlines()[0] if text else "Coach ready")
        self.contentChanged.emit()

    def show_feedback(self, feedback: CoachFeedback):
        self._feedback = feedback
        self.reference_revision = None
        self.evidence_expanded = False
        self._render_feedback()
        self.verticalScrollBar().setValue(0)
        self.feedbackAvailable.emit(True)
        self._summarize()

    def set_question_draft(self, question: str) -> None:
        self._question_draft = question
        if self._feedback is not None:
            position = self.verticalScrollBar().value()
            self._render_feedback()
            self.verticalScrollBar().setValue(position)
            self._summarize()

    def set_gradients(self,gradients):
        if gradients!=self._gradients:
            self._gradients=gradients
            position=self.verticalScrollBar().value()
            self._render_feedback()
            self.verticalScrollBar().setValue(position)

    def _question_changed(self) -> bool:
        return (self._feedback is not None and self._question_draft is not None and
                self._question_draft != self._feedback.question)

    def keep_as_reference(self, photo_id: str, session_id: str, current_revision: int) -> bool:
        feedback = self._feedback
        if feedback is None or (feedback.photo_id, feedback.session_id) != (photo_id, session_id):
            return False
        if current_revision == feedback.state_version:
            return True
        self.reference_revision = current_revision
        position = self.verticalScrollBar().value()
        self._render_feedback()
        self.verticalScrollBar().setValue(position)
        self._summarize()
        return True

    def _summarize(self):
        feedback = self._feedback
        earlier = self.reference_revision is not None or self._question_changed()
        prefix = f"Earlier advice · v{feedback.state_version}" if earlier else f"State v{feedback.state_version}"
        summary = " ".join(_display_text(feedback.next_step_or_stop).split())
        self.summaryChanged.emit(f"{prefix} · {summary}")

    def _follow_link(self, link):
        # Model output never becomes a link. Only our disclosure/navigation links work.
        if link.toString() == "photocoach:evidence" and self._feedback is not None:
            position = self.verticalScrollBar().value()
            self.evidence_expanded = not self.evidence_expanded
            self._render_feedback()
            self.verticalScrollBar().setValue(position)
        elif self._feedback is not None:
            control = related_control(self._feedback.next_step_or_stop,self._gradients)
            if control and link.toString() == f"photocoach:locate/{control}":
                self.locateRequested.emit(control)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        wide = self.viewport().width() >= 540
        if wide != self._wide and not self._reading:
            self._wide = wide
            self._render_feedback()

    def response_text(self) -> str:
        """Copy the complete structured reply, including hidden observations."""
        feedback = self._feedback
        if feedback is None:
            return self.toPlainText()
        lines = [f"{feedback.source} · State v{feedback.state_version}"]
        if self.reference_revision is not None:
            lines.append(f"Earlier advice · based on v{feedback.state_version}. "
                         f"Current edit is v{self.reference_revision}; this advice has not been rechecked.")
        if self._question_changed():
            lines.append("Question changed. This reply answers the question shown below.")
        lines.extend([f"Question sent: {feedback.question}" if feedback.question else "Current edit review · No question was sent."])
        for heading, value in (("ANSWER", feedback.answer), ("NEXT STEP", feedback.next_step_or_stop),
                               ("WATCH / STOP", feedback.observe_and_stop_when),
                               ("WHY THIS HELPS", feedback.reason), ("WHAT CHANGED", feedback.main_change)):
            if value:
                lines.extend(["", heading, _display_text(value)])
        lines.extend(["", *(_display_text(item) for item in feedback.evidence)])
        return "\n".join(lines)

    def _render_feedback(self):
        feedback = self._feedback
        if feedback is None:
            return
        details = ""
        if self.evidence_expanded:
            details = (f'<p><b>WHAT CHANGED</b><br>{_body(feedback.main_change)}</p>'
                       + "".join(f'<p style="color:#bcbcbc;">{_body(item)}</p>' for item in feedback.evidence))
        toggle = "Hide change and observations" if self.evidence_expanded else "Show change and observations"
        reference = ""
        if self.reference_revision is not None:
            reference = (f'<p style="color:#f1c58c;"><b>Earlier advice · based on v{feedback.state_version}.</b> '
                         f'Current edit is v{self.reference_revision}; this advice has not been rechecked.</p>')
        if self._question_changed():
            reference += ('<p style="color:#f1c58c;"><b>Question changed.</b> This reply answers the question shown below. '
                          'Click Ask coach to send your new question.</p>')
        question = (f'<p style="color:#bcbcbc;"><b>Question sent:</b> {_text(feedback.question)}</p>'
                    if feedback.question else '<p style="color:#bcbcbc;">Current edit review · No question was sent.</p>')
        answer = (f'<p style="color:#9fc6ff;"><a name="answer"></a><b>ANSWER</b></p>'
                  f'<p style="font-size:15px;color:#eeeeee;line-height:135%;">{_body(feedback.answer)}</p>' if feedback.answer else "")
        control = related_control(feedback.next_step_or_stop,self._gradients)
        title=next((g.name for g in self._gradients if control==f'local_gradients.{g.id}'),CONTROL_NAMES.get(control,'Local adjustments'))
        locate = (f'<p><a href="photocoach:locate/{control}" style="color:#9fc6ff;">Go to {_text(title)}</a></p>'
                  if control else "")
        next_step = (f'<p style="color:#9fc6ff;font-size:12px;"><a name="steps"></a><b>NEXT STEP</b></p>'
                     f'<p style="font-size:14px;color:#eeeeee;">{_body(feedback.next_step_or_stop)}</p>{locate}')
        stop = (f'<p style="color:#9fc6ff;"><a name="stop"></a><b>Watch / stop</b></p>'
                f'<p style="color:#dddddd;">{_body(feedback.observe_and_stop_when)}</p>')
        content = (f'<tr><td width="58%" valign="top">{next_step}</td><td width="42%" valign="top">{stop}</td></tr>'
                   if self._wide and not self._reading else f'<tr><td>{next_step}{stop}</td></tr>')
        self.setHtml(f'''
            <style>p {{ margin-top:6px; margin-bottom:10px; }}
            a {{ text-decoration:none; }}</style>
            <p style="color:#ababab;font-size:12px;">{_text(feedback.source)} · State v{feedback.state_version}</p>
            {reference}
            {question}{answer}
            <table width="100%" cellspacing="0" cellpadding="12" bgcolor="#2c3036">{content}</table>
            <p><a name="why"></a><b>WHY THIS HELPS</b><br>{_body(feedback.reason)}</p>
            <p><a href="photocoach:evidence" style="color:#9fc6ff;">{toggle}</a></p>
            {details}
        ''')
        self.contentChanged.emit()


class CoachResponsePanel(QWidget):
    """Reading and copying are local presentation actions, independent of sending."""
    locateRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        actions = QHBoxLayout()
        title = QLabel("Coaching notes")
        title.setProperty("secondary", True)
        actions.addWidget(title, 1)
        self.reader_button = QPushButton("Open reader")
        self.reader_button.setToolTip("Read the full response in a larger, resizable window. No request is sent.")
        self.reader_button.clicked.connect(self.open_reader)
        self.copy_button = QPushButton("Copy response")
        self.copy_button.setToolTip("Copy the answer, steps, stopping condition and observations, including earlier-advice warnings.")
        self.copy_button.clicked.connect(self.copy_response)
        actions.addWidget(self.reader_button)
        actions.addWidget(self.copy_button)
        layout.addLayout(actions)
        self.output = CoachOutput()
        self.output.locateRequested.connect(self.locateRequested)
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
        reader_layout.addWidget(self.reader_output, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
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
            reader.reference_revision = output.reference_revision
            reader._gradients = output._gradients
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

    def copy_response(self):
        if self.output._feedback is not None:
            QApplication.clipboard().setText(self.output.response_text())
            for button in (self.copy_button, self.reader_copy):
                button.setMinimumWidth(max(button.minimumWidth(), button.sizeHint().width()))
                button.setText("Copied")
            self._copied_timer.start(1500)

    def _reset_copy_label(self):
        for button in (self.copy_button, self.reader_copy):
            button.setText("Copy response")

    def _locate_from_reader(self, control):
        self.reader.hide()
        self.locateRequested.emit(control)


def usage_summary(budget: SessionBudget, mock_calls: int, cloud: bool) -> str:
    if not cloud and not budget.attempts:
        return f"Local demo · {mock_calls} analyses · No cloud requests"
    cost = budget.known_cost_usd
    amount = "Cost unknown" if cost is None else f"Est. ${cost:.4f}"
    active = " · Sending" if budget.in_flight else ""
    return f"{budget.attempts}/{budget.call_limit} API attempts this session · {amount}{active}"
