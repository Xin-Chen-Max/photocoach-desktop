"""Darkroom named snapshots panel: create, select, restore, delete, and A/B compare."""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .snapshots import EditSnapshot, MAX_SNAPSHOTS, MAX_SNAPSHOT_NAME_LENGTH


class SnapshotsPanel(QWidget):
    """Named project edit checkpoints and A/B viewport comparison controls."""

    saveRequested = Signal(str)
    restoreRequested = Signal(str)
    deleteRequested = Signal(str)
    compareToggled = Signal(object)  # snapshot_id: str | None

    def __init__(self) -> None:
        super().__init__()
        self._lang = "en"
        self._snapshots: tuple[EditSnapshot, ...] = ()
        self._compare_snapshot_id: str | None = None
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.header_label = QLabel("Snapshots")
        self.header_label.setStyleSheet("font-weight:600; color:#eeeeee;")
        self.count_label = QLabel(f"0 / {MAX_SNAPSHOTS}")
        self.count_label.setProperty("secondary", True)
        header.addWidget(self.header_label, 1)
        header.addWidget(self.count_label)
        layout.addLayout(header)

        create_row = QHBoxLayout()
        create_row.setContentsMargins(0, 0, 0, 0)
        create_row.setSpacing(6)
        self.name_input = QLineEdit()
        self.name_input.setMaxLength(MAX_SNAPSHOT_NAME_LENGTH)
        self.name_input.setPlaceholderText("Snapshot name (e.g. Warm look)")
        self.name_input.setAccessibleName("Snapshot name")
        self.name_input.returnPressed.connect(self._emit_save)
        self.name_input.textEdited.connect(lambda *_: self.set_feedback(""))
        self.save_button = QPushButton("Save snapshot")
        self.save_button.setProperty("compact", True)
        self.save_button.clicked.connect(self._emit_save)
        create_row.addWidget(self.name_input, 1)
        create_row.addWidget(self.save_button)
        layout.addLayout(create_row)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(92)
        self.list_widget.setMaximumHeight(150)
        self.list_widget.setAccessibleName("Saved snapshots")
        self.list_widget.setStyleSheet(
            "QListWidget { background:#222222; border:1px solid #3c3c3c; border-radius:4px; }"
            "QListWidget::item { padding:5px 7px; border-radius:2px; }"
            "QListWidget::item:selected { background:#315c91; color:#ffffff; }"
        )
        self.list_widget.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.list_widget)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        self.compare_button = QPushButton("Compare A/B")
        self.compare_button.setCheckable(True)
        self.compare_button.setProperty("compact", True)
        self.compare_button.clicked.connect(self._emit_compare_toggle)

        self.restore_button = QPushButton("Restore")
        self.restore_button.setProperty("compact", True)
        self.restore_button.clicked.connect(self._emit_restore)

        self.delete_button = QPushButton("Delete")
        self.delete_button.setProperty("compact", True)
        self.delete_button.clicked.connect(self._emit_delete)

        actions.addWidget(self.compare_button, 1)
        actions.addWidget(self.restore_button, 1)
        actions.addWidget(self.delete_button, 1)
        layout.addLayout(actions)

        self.feedback_label = QLabel()
        self.feedback_label.setProperty("secondary", True)
        self.feedback_label.setWordWrap(True)
        self.feedback_label.hide()
        layout.addWidget(self.feedback_label)

        self.set_language("en")
        self._update_buttons()

    def selected_snapshot_id(self) -> str | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, str) and value else None

    def set_snapshots(
        self,
        snapshots,
        *,
        compare_snapshot_id: str | None = None,
        selected_id: str | None = None,
    ) -> None:
        previous_selected = selected_id if selected_id is not None else self.selected_snapshot_id()
        self._snapshots = tuple(snapshots)
        self._compare_snapshot_id = compare_snapshot_id

        self._syncing = True
        try:
            self.list_widget.clear()
            target_row = -1
            for index, snap in enumerate(self._snapshots):
                is_compared = compare_snapshot_id == snap.id
                label = f"[A/B] {snap.name}" if is_compared else snap.name
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, snap.id)
                stamp = snap.created_utc[:19].replace("T", " ") + " UTC"
                item.setToolTip(f"{snap.name}\n{stamp}")
                self.list_widget.addItem(item)
                if snap.id == previous_selected:
                    target_row = index
            if target_row < 0 and compare_snapshot_id:
                target_row = next(
                    (i for i, s in enumerate(self._snapshots) if s.id == compare_snapshot_id),
                    -1,
                )
            if target_row < 0 and self._snapshots:
                target_row = len(self._snapshots) - 1
            if target_row >= 0:
                self.list_widget.setCurrentRow(target_row)
        finally:
            self._syncing = False

        self.count_label.setText(f"{len(self._snapshots)} / {MAX_SNAPSHOTS}")
        self._update_buttons()

    def set_feedback(self, text: str, *, error: bool = False) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            self.feedback_label.clear()
            self.feedback_label.hide()
            return
        self.feedback_label.setStyleSheet(
            "color:#ef9a9a; font-size:12px;" if error else "color:#79afff; font-size:12px;"
        )
        self.feedback_label.setText(cleaned)
        self.feedback_label.show()

    def clear_name_input(self) -> None:
        self.name_input.clear()

    def set_language(self, lang: str) -> None:
        self._lang = "zh" if str(lang).lower().startswith("zh") else "en"
        zh = self._lang == "zh"
        self.header_label.setText("命名快照" if zh else "Snapshots")
        self.name_input.setPlaceholderText(
            "输入快照名称（如：暖调基础）" if zh else "Snapshot name (e.g. Warm look)"
        )
        self.save_button.setText("保存快照" if zh else "Save snapshot")
        self.save_button.setToolTip(
            "保存当前已应用调色参数的不可变命名快照（不含原片与未应用白平衡草稿）"
            if zh
            else "Save an immutable named checkpoint of current edit parameters (never alters source RAW or calls AI)."
        )
        self.compare_button.setText("A/B 对比" if zh else "Compare A/B")
        self.compare_button.setToolTip(
            "在当前缩放与平移位置对比选中快照与当前编辑效果；继续调色将自动回到编辑视图"
            if zh
            else "Toggle A/B comparison of the selected snapshot at the current zoom and pan. New edits return to Edited."
        )
        self.restore_button.setText("恢复" if zh else "Restore")
        self.restore_button.setToolTip(
            "将选中快照的参数恢复为当前编辑（单步可撤销，不影响导出与撤销栈前序历史）"
            if zh
            else "Restore the selected snapshot's parameters in one Undo step."
        )
        self.delete_button.setText("删除" if zh else "Delete")
        self.delete_button.setToolTip(
            "从当前工程移除选中快照（不改动当前编辑参数）"
            if zh
            else "Delete the selected snapshot from this project without changing current edits."
        )
        self._update_buttons()

    def _update_buttons(self) -> None:
        enabled = self.isEnabled()
        has_selection = self.selected_snapshot_id() is not None
        at_limit = len(self._snapshots) >= MAX_SNAPSHOTS
        self.name_input.setEnabled(enabled)
        self.save_button.setEnabled(enabled and not at_limit)
        self.list_widget.setEnabled(enabled)
        self.compare_button.blockSignals(True)
        self.compare_button.setChecked(bool(self._compare_snapshot_id))
        self.compare_button.blockSignals(False)
        self.compare_button.setEnabled(enabled and (has_selection or bool(self._compare_snapshot_id)))
        self.restore_button.setEnabled(enabled and has_selection)
        self.delete_button.setEnabled(enabled and has_selection)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            self._update_buttons()

    def _selection_changed(self, current, previous) -> None:
        self._update_buttons()
        if self._syncing:
            return
        selected = self.selected_snapshot_id()
        if self._compare_snapshot_id is not None and selected and selected != self._compare_snapshot_id:
            self.compareToggled.emit(selected)

    def _emit_save(self) -> None:
        if not self.isEnabled():
            return
        self.saveRequested.emit(self.name_input.text())

    def _emit_restore(self) -> None:
        snapshot_id = self.selected_snapshot_id()
        if self.isEnabled() and snapshot_id:
            self.restoreRequested.emit(snapshot_id)

    def _emit_delete(self) -> None:
        snapshot_id = self.selected_snapshot_id()
        if self.isEnabled() and snapshot_id:
            self.deleteRequested.emit(snapshot_id)

    def _emit_compare_toggle(self, checked: bool) -> None:
        if not self.isEnabled():
            self._update_buttons()
            return
        if checked:
            snapshot_id = self.selected_snapshot_id()
            if not snapshot_id:
                self._update_buttons()
                return
            self.compareToggled.emit(snapshot_id)
        else:
            self.compareToggled.emit(None)
