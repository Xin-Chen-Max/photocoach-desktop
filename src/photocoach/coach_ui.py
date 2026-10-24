"""Local connection settings and explicit preview disclosure before sending."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
import json
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit

from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QColorSpace, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QScrollArea,
    QSpinBox, QTextEdit, QVBoxLayout, QWidget, QPushButton, QCompleter)

from .api_coach import (ApiConfig, ApiCoach, ApiError, request_context, request_endpoint, COACH_INSTRUCTIONS,
                        GEMINI_ENDPOINT, check_gemini_model, ModelCheckError, ModelCheckResult,
                        GeminiModelsConfig, ProviderModelsConfig, GeminiModelsPage, list_gemini_models)
from .budget import SessionBudget
from .text_probe import build_text_probe_payload
from .settings import (ApiSettings, SettingsStore, SettingsError, credential_target,
                       resolve_api_key, resolve_key_source)
from .credentials import WindowsCredentialStore, CredentialError
from .model_catalog import CatalogError
from .connections import (Connection, ConnectionBook, ConnectionStore, infer_provider,
                          connection_from_settings, settings_for)


class _CheckSignals(QObject):
    done = Signal(object, object)


class _ModelCheckWorker(QRunnable):
    def __init__(self, config, api_key, cancel, page_token=None):
        super().__init__()
        self.config, self.api_key, self.cancel = config, api_key, cancel
        self.page_token = page_token
        self.signals = _CheckSignals()

    def run(self):
        result, error = None, None
        try:
            if hasattr(self.config, "protocol") and self.config.protocol != "gemini":
                from .api_coach import list_openai_models
                result = list_openai_models(self.config.endpoint, self.api_key, self.cancel, protocol=self.config.protocol)
            elif self.page_token is None:
                result = check_gemini_model(self.config, self.api_key, self.cancel)
            else:
                result = list_gemini_models(self.config, self.api_key, self.cancel, self.page_token)
        except ModelCheckError as failure:
            error = failure
        except Exception:
            error = ModelCheckError(ApiError("connection"))
        finally:
            self.api_key = ""
        self.signals.done.emit(result, error)


class ApiSettingsDialog(QDialog):
    def __init__(self, parent, previous: ApiSettings | None, budget: SessionBudget,
                 *, settings_store=None, credentials=None, connection_book=None, session_only_ids=()):
        super().__init__(parent)
        self.credentials = credentials if credentials is not None else WindowsCredentialStore()
        self.settings_store = settings_store
        self._storage_unavailable = False
        self.connection_store = None
        if isinstance(self.settings_store, ConnectionStore):
            self.connection_store = self.settings_store
        elif self.settings_store is not None:
            self.connection_store = ConnectionStore(self.settings_store.path)
        else:
            try:
                self.connection_store = ConnectionStore()
                self.settings_store = SettingsStore(self.connection_store.path)
            except SettingsError:
                self._storage_unavailable = True

        self.connection_book = None
        if self.connection_store is not None:
            try:
                self.connection_book = self.connection_store.load()
            except SettingsError:
                self.connection_book = None
        if connection_book is not None:
            self.connection_book = connection_book
        self._session_only_ids = set(session_only_ids)
        if self.connection_book is None:
            self.connection_book = ConnectionBook()
        if previous is not None:
            existing = next((c for c in self.connection_book.connections
                             if c.credential_target == credential_target(previous.config)), None)
            prev_conn = connection_from_settings(previous, connection_id=existing.id if existing else None)
            if existing:
                prev_conn = replace(prev_conn, favorites=existing.favorites, recent=existing.recent)
            self.connection_book = self.connection_book.with_connection(prev_conn, activate=True)
            if not previous.remember_settings:
                self._session_only_ids.add(prev_conn.id)
        self._delete_target = None
        self._catalog = {}
        self._next_models_page = ""
        self._seen_models_pages = set()
        self.setWindowTitle("AI connection settings")
        available = self.screen().availableGeometry()
        self.resize(min(700, available.width()-40), min(660, available.height()-60))
        layout = QVBoxLayout(self)
        note = QLabel("Save your connection for next time. API keys can be kept in Windows Credential Manager on this PC. Saving or restoring settings sends no network requests.")
        note.setWordWrap(True)
        layout.addWidget(note)
        scroll = QScrollArea()
        self.settings_scroll = scroll
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.settings_form = form
        config = previous.config if previous else SimpleNamespace(**{
            name: definition.default for name, definition in ApiConfig.__dataclass_fields__.items()})
        # Session limits/reservations belong to the live ledger, not a provider preset.
        values = dict(previous.budget_settings) if previous else {name: getattr(budget, name) for name in SessionBudget._FIELDS}
        if previous is None:
            values.update(input_usd_per_million=None, output_usd_per_million=None)
        self.protocol = QComboBox()
        self.protocol.addItem("OpenAI · Responses API", "responses")
        self.protocol.addItem("Compatible provider · Chat Completions", "chat_completions")
        self.protocol.addItem("Google Gemini · native API", "gemini")
        self.protocol.setCurrentIndex(max(0, self.protocol.findData(config.protocol)))
        form.addRow("Provider / API", self.protocol)
        self.endpoint = QLineEdit(config.endpoint)
        self.model = QLineEdit(config.model)
        self.model.setPlaceholderText("Enter the model ID from your provider")
        self.model.setToolTip("Use the exact model ID, not the API key's display name.")
        form.addRow("Model", self.model)
        self.catalog_controls = QWidget()
        catalog_layout = QVBoxLayout(self.catalog_controls)
        catalog_layout.setContentsMargins(0, 0, 0, 0)
        catalog_actions = QHBoxLayout()
        self.load_models_button = QPushButton("Load Gemini models")
        self.more_models_button = QPushButton("Load more")
        self.more_models_button.hide()
        catalog_actions.addWidget(self.load_models_button)
        catalog_actions.addWidget(self.more_models_button)
        catalog_layout.addLayout(catalog_actions)
        self.model_choices = QComboBox()
        self.model_choices.setEditable(True)
        self.model_choices.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_choices.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.model_choices.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self.model_choices.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_choices.setMinimumContentsLength(15)
        self.model_choices.addItem("Choose a listed model…", None)
        self.model_choices.setEnabled(False)
        catalog_layout.addWidget(self.model_choices)
        self.models_status = QLabel("Load model metadata using the selected key. No photos or generation request; 15-second limit. You can also enter an ID above.")
        self.models_status.setTextFormat(Qt.TextFormat.PlainText)
        self.models_status.setWordWrap(True)
        catalog_layout.addWidget(self.models_status)
        form.addRow(self.catalog_controls)
        self.catalog_controls.setVisible(True)
        self.load_models_button.setText("Refresh models")
        self.connection_issue = QLabel()
        self.connection_issue.setWordWrap(True)
        self.connection_issue.setTextFormat(Qt.TextFormat.PlainText)
        self.connection_issue.setStyleSheet("color:#f2bf80;")
        form.addRow(self.connection_issue)
        self.key_source = QComboBox()
        self.key_source.addItem("Enter a key", "session")
        self.key_source.addItem("Use saved key on this PC", "saved")
        self.key_source.addItem("Use environment variable", "environment")
        self.key_source.setCurrentIndex(max(0, self.key_source.findData(previous.key_source if previous else "session")))
        form.addRow("Key source", self.key_source)
        self.key = QLineEdit(previous.api_key if previous and previous.key_source == "session" else "")
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("Paste your API key here")
        form.addRow("API key", self.key)
        self.environment = QLineEdit(previous.key_environment if previous else "PHOTOCOACH_API_KEY")
        form.addRow("Key environment variable", self.environment)
        self.remember_settings = QCheckBox("Remember connection settings on this PC")
        self.remember_settings.setChecked(previous.remember_settings if previous else True)
        self.remember_key = QCheckBox("Remember API key on this PC")
        self.remember_key.setToolTip("Stores this key in Windows Credential Manager for this Windows account and API destination. It is never saved in settings.json or photo projects.")
        form.addRow(self.remember_settings)
        form.addRow(self.remember_key)
        key_actions = QHBoxLayout()
        self.replace_key_button = QPushButton("Replace key")
        self.delete_key_button = QPushButton("Delete saved key")
        key_actions.addWidget(self.replace_key_button)
        key_actions.addWidget(self.delete_key_button)
        form.addRow(key_actions)
        self.storage_status = QLabel()
        self.storage_status.setWordWrap(True)
        self.storage_status.setTextFormat(Qt.TextFormat.PlainText)
        form.addRow(self.storage_status)
        self._check_worker = None
        self._check_cancel = None
        self._metadata_checks = 0
        self.model_check_button = QPushButton("Check key && model · no photos")
        self.model_check_button.clicked.connect(self.check_model)
        self.model_check_status = QLabel("Sends your key to Google's model metadata endpoint only. No photo, editing data or content generation. It does not use an analysis attempt. Maximum wait: 15 seconds.")
        self.model_check_status.setTextFormat(Qt.TextFormat.PlainText)
        self.model_check_status.setWordWrap(True)
        self.model_check_button.setVisible(config.protocol == "gemini")
        self.model_check_status.setVisible(config.protocol == "gemini")
        self.calls = QSpinBox()
        self.calls.setRange(0, 10000)
        self.calls.setValue(values.get("call_limit", budget.call_limit))
        self.calls.setToolTip("Maximum text tests and photo requests in this app session, including failed attempts. Changing settings keeps the used count.")
        form.addRow("Session API attempt limit", self.calls)
        self.connection_details_button = QPushButton("Show connection details")
        self.connection_details_button.setCheckable(True)
        form.addRow(self.connection_details_button)
        self.connection_details = QWidget()
        connection_form = QFormLayout(self.connection_details)
        connection_form.setContentsMargins(0, 0, 0, 0)
        connection_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        connection_form.addRow("HTTPS endpoint / Gemini models base", self.endpoint)
        self.timeout = QSpinBox()
        self.timeout.setRange(5, 120)
        self.timeout.setValue(int(config.timeout_seconds))
        self.timeout.setToolTip("Total connection, upload and response deadline. Increasing it only affects future manually sent requests; it never retries a failed request.")
        connection_form.addRow("Timeout · seconds", self.timeout)
        self.output_tokens = QSpinBox()
        self.output_tokens.setRange(256, 8000)
        self.output_tokens.setValue(config.max_output_tokens)
        connection_form.addRow("Maximum output tokens", self.output_tokens)
        connection_form.addRow(self.model_check_button)
        connection_form.addRow(self.model_check_status)
        form.addRow(self.connection_details)
        self.connection_details.hide()
        self.connection_details_button.toggled.connect(self._toggle_connection_details)
        self.advanced_button = QPushButton("Show advanced cost settings")
        self.advanced_button.setCheckable(True)
        form.addRow(self.advanced_button)
        self.advanced_costs = QWidget()
        costs_form = QFormLayout(self.advanced_costs)
        costs_form.setContentsMargins(0, 0, 0, 0)
        self.amounts = {}
        for key, label in (("budget_usd", "Optional session USD allocation cap"),
                           ("reserve_usd", "Conservative USD reserve per attempt"),
                           ("input_usd_per_million", "Input USD per million tokens"),
                           ("output_usd_per_million", "Output USD per million tokens")):
            value = values.get(key)
            field_widget = QLineEdit("" if value is None else str(value))
            field_widget.setPlaceholderText("Not configured")
            costs_form.addRow(label, field_widget)
            self.amounts[key] = field_widget
        costs_note = QLabel("Blank prices mean unknown cost. A USD cap requires a conservative reserve per attempt and cannot guarantee the provider's bill. API billing is separate from a ChatGPT subscription.")
        costs_note.setWordWrap(True)
        costs_form.addRow(costs_note)
        form.addRow(self.advanced_costs)
        self.advanced_costs.hide()
        self.advanced_button.toggled.connect(self._toggle_advanced)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        warning = QLabel("Limits apply to this app session. Editing settings keeps its attempts and allocations; restarting starts a new local count, not a new provider bill. Failed requests may still be charged. No automatic retries.")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        self.result_settings = None
        self.result_notice = ""
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText("Use settings · no request")
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.protocol.currentIndexChanged.connect(self.select_protocol)
        self.endpoint.textChanged.connect(self._destination_changed)
        self.key_source.currentIndexChanged.connect(self._key_source_changed)
        self.remember_key.toggled.connect(self._remember_key_changed)
        self.remember_settings.toggled.connect(lambda checked: self.remember_key.setChecked(False) if not checked else None)
        self.replace_key_button.clicked.connect(self._replace_key)
        self.delete_key_button.clicked.connect(self._delete_saved_key)
        for field in (self.endpoint, self.model, self.key, self.environment):
            field.textChanged.connect(self._invalidate_model_check)
        self.endpoint.textChanged.connect(self._update_connection_hint)
        self.model.textChanged.connect(self._update_connection_hint)
        for widget in (self.endpoint, self.key, self.environment):
            widget.textChanged.connect(self._invalidate_catalog)
        self.key_source.currentIndexChanged.connect(self._invalidate_catalog)
        self.load_models_button.clicked.connect(lambda: self.load_models())
        self.more_models_button.clicked.connect(lambda: self.load_models(more=True))
        self.model_choices.activated.connect(self._choose_model)
        self._update_connection_hint()
        self._update_key_controls()
        from .model_catalog import ModelCatalogStore
        self.catalog_store = (ModelCatalogStore(self.connection_store.path.with_name('model_catalog.json'))
                              if self.connection_store is not None else None)
        self._restore_catalog()
        if self._storage_unavailable:
            self.remember_settings.setChecked(False)
            self.remember_settings.setEnabled(False)
            self.remember_key.setEnabled(False)
            self.storage_status.setText("Local settings storage is unavailable. You can use a key for this session.")

    def _connection_mismatch(self):
        try:
            host = urlsplit(self.endpoint.text().strip()).hostname
        except ValueError:
            return False  # Full endpoint validation runs when applying settings.
        return host == "api.openai.com" and self.model.text().strip().lower().startswith("gemini-")

    def _update_connection_hint(self):
        mismatch = self._connection_mismatch()
        self.connection_issue.setText("This Gemini model is pointing to OpenAI. Select Google Gemini · native API above, then select your key for that destination." if mismatch else "")
        self.connection_issue.setVisible(mismatch)

    def _toggle_connection_details(self, expanded):
        self.connection_details.setVisible(expanded)
        self.connection_details_button.setText("Hide connection details" if expanded else "Show connection details")

    def _toggle_advanced(self, expanded):
        self.advanced_costs.setVisible(expanded)
        self.advanced_button.setText("Hide advanced cost settings" if expanded else "Show advanced cost settings")

    def _update_key_controls(self):
        source = self.key_source.currentData()
        self.key.setEnabled(source == "session")
        self.environment.setEnabled(source == "environment")
        self.settings_form.setRowVisible(self.key, source == "session")
        self.settings_form.setRowVisible(self.environment, source == "environment")
        self.remember_key.setVisible(source == "session")
        self.remember_key.setEnabled(source == "session" and not self._storage_unavailable)
        if self._delete_target:
            message = "Saved key will be deleted when you apply settings. Cancel keeps it."
        elif source == "saved":
            message = "Saved key selected. Its value is read only when needed and is not displayed here."
        elif source == "environment":
            message = "Only the named environment variable supplies the key. PhotoCoach does not change it."
        elif self.remember_key.isChecked():
            message = "This key will be saved in Windows Credential Manager when you apply settings."
        else:
            message = "This key stays in memory for this session unless you choose Remember API key."
        self.storage_status.setText(message)

    def _key_source_changed(self):
        self.key.clear()
        self.remember_key.setChecked(False)
        self._invalidate_model_check()
        self._update_key_controls()

    def _remember_key_changed(self, checked):
        if checked:
            self.remember_settings.setChecked(True)
        self._update_key_controls()

    def _destination_changed(self):
        self._delete_target = None
        self.delete_key_button.setText("Delete saved key")
        self.key_source.setCurrentIndex(self.key_source.findData("session"))
        self.key.clear()
        self.remember_key.setChecked(False)
        self._invalidate_model_check()
        self._update_key_controls()
        self.storage_status.setText("Destination changed. Enter or explicitly select a key for this address.")

    def _replace_key(self):
        self.key_source.setCurrentIndex(self.key_source.findData("session"))
        self.key.clear()
        self.remember_key.setChecked(not self._storage_unavailable)
        self.key.setFocus()

    def _delete_saved_key(self):
        if self._delete_target:
            self._delete_target = None
            self.delete_key_button.setText("Delete saved key")
        else:
            try:
                config = ApiConfig(self.endpoint.text().strip(), self.model.text().strip(), self.protocol.currentData())
                self._delete_target = credential_target(config)
            except (ApiError, SettingsError):
                self.storage_status.setText("Choose a valid endpoint and model before deleting its saved key.")
                return
            self.key_source.setCurrentIndex(self.key_source.findData("session"))
            self.key.clear()
            self.remember_key.setChecked(False)
            self.delete_key_button.setText("Undo deletion")
        self._invalidate_model_check()
        self._update_key_controls()

    def _invalidate_model_check(self):
        if self._check_cancel is not None:
            self._check_cancel.set()
        if self._metadata_checks:
            self.model_check_status.setText("Settings changed. Check again to verify the current key and model. No request was sent by this change.")

    def select_protocol(self):
        self._destination_changed()
        self._invalidate_catalog()
        protocol = self.protocol.currentData()
        self.catalog_controls.setVisible(True)
        self.load_models_button.setText("Refresh models")
        self.model_check_button.setVisible(protocol == "gemini")
        self.model_check_status.setVisible(protocol == "gemini")
        provider = "gemini" if protocol == "gemini" else ("openai" if protocol == "responses" else "custom")
        saved_conn = self.connection_book.connection_for_provider(provider) if self.connection_book else None
        if saved_conn is not None and saved_conn.config.protocol == protocol:
            self.endpoint.setText(saved_conn.config.endpoint)
            self.model.setText(saved_conn.config.model)
            self.environment.setText(saved_conn.key_environment or ("GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"))
            self.key_source.setCurrentIndex(max(0, self.key_source.findData(saved_conn.key_source)))
            if saved_conn.key_source == "session":
                self.key.clear()
        else:
            if protocol == "gemini":
                self.endpoint.setText(GEMINI_ENDPOINT)
                self.environment.setText("GEMINI_API_KEY")
                self.timeout.setValue(max(60, self.timeout.value()))
                self.model.setPlaceholderText("Exact Gemini model code from Google AI Studio")
            elif self.endpoint.text().rstrip("/") in (GEMINI_ENDPOINT,
                    "https://api.openai.com/v1/responses", "https://api.openai.com/v1/chat/completions"):
                self.endpoint.setText("https://api.openai.com/v1/responses" if protocol == "responses" else "https://api.openai.com/v1/chat/completions")
                self.environment.setText("OPENAI_API_KEY")
                self.model.setPlaceholderText("Enter the model ID from your provider")
        self._restore_catalog()

    def _invalidate_catalog(self):
        if self._check_cancel is not None:
            self._check_cancel.set()
        self._catalog.clear()
        self._next_models_page = ""
        self._seen_models_pages.clear()
        self.model_choices.clear()
        self.model_choices.addItem("Choose a listed model…", None)
        self.model_choices.setEnabled(False)
        self.more_models_button.hide()
        self.models_status.setText("Load model metadata using the selected key. No photos or generation request; 15-second limit. You can also enter an ID above.")

    def _choose_model(self, index):
        model_id = self.model_choices.itemData(index)
        if model_id in self._catalog:
            self.model.setText(model_id)

    def load_models(self, *, more=False):
        if self._check_worker is not None or (more and not self._next_models_page):
            return
        protocol = self.protocol.currentData()
        try:
            if protocol == "gemini":
                config = GeminiModelsConfig(self.endpoint.text().strip())
            else:
                config = ProviderModelsConfig(self.endpoint.text().strip(), protocol)
            key = resolve_key_source(config, self.key_source.currentData(), self.key.text(),
                                     self.environment.text().strip(), self.credentials)
            if not key:
                raise ApiError("configuration")
        except ApiError as error:
            self.models_status.setText(str(ModelCheckError(error)))
            return
        except (CredentialError, SettingsError) as error:
            self.models_status.setText(str(error) + " No request was sent.")
            return
        page_token = self._next_models_page if more else ""
        self._check_cancel = threading.Event()
        worker = _ModelCheckWorker(config, key, self._check_cancel, page_token)
        self._check_worker = worker
        self._metadata_checks += 1
        worker.signals.done.connect(self._models_loaded)
        self._set_check_controls(False)
        self.models_status.setText("Loading model metadata… No photos or generation request. Maximum wait: 15 seconds.")
        QThreadPool.globalInstance().start(worker)

    def _models_loaded(self, result, error):
        worker = self._check_worker
        self._check_worker = None
        self._set_check_controls(True)
        if worker is None or worker.cancel.is_set():
            if self.models_status.text().startswith("Loading model metadata"):
                self.models_status.setText("Model loading cancelled. No generation request was sent.")
            return
        if isinstance(error, ModelCheckError):
            self.models_status.setText(str(error))
            return
        if isinstance(result, tuple):
            rows = [SimpleNamespace(model_id=mid, display_name=mid) for mid in result]
            next_page = ''
        elif isinstance(result, GeminiModelsPage):
            rows, next_page = result.models, result.next_page_token
        else:
            self.models_status.setText('Unexpected model metadata. Keep your model ID or refresh manually.')
            return
        seen = self._seen_models_pages | {worker.page_token or ''}
        if next_page and next_page in seen:
            self.models_status.setText('Unexpected repeated model page. Existing model selection is unchanged.')
            return
        if not worker.page_token:
            self._catalog.clear()
        self._catalog.update((m.model_id, m) for m in rows)
        self._seen_models_pages, self._next_models_page = seen, next_page
        notice = ''
        if self.catalog_store is not None and self.remember_settings.isChecked():
            try:
                config = self._metadata_config()
                self.catalog_store.update_provider_models(infer_provider(config),
                    [{'id': m.model_id, 'display_name': m.display_name} for m in self._catalog.values()],
                    credential_target(config))
            except (ApiError, SettingsError, CatalogError) as failure:
                notice = str(failure)
        self._populate_catalog()
        self.more_models_button.setVisible(bool(next_page))
        self.models_status.setText((f'{len(self._catalog)} models listed. Refreshed just now. ' if self._catalog else 'No matching models on this page. ') +
            ('More available; use Load more. ' if next_page else '') +
            'Listing does not verify image/schema support or quota. Use settings to apply. ' + notice)

    def _metadata_config(self):
        return (GeminiModelsConfig(self.endpoint.text().strip()) if self.protocol.currentData() == 'gemini'
                else ProviderModelsConfig(self.endpoint.text().strip(), self.protocol.currentData()))

    def _populate_catalog(self):
        from .model_catalog import classify_model
        selected = self.model_choices.currentData()
        self.model_choices.clear()
        self.model_choices.addItem('Choose a listed model…', None)
        try:
            provider = infer_provider(self._metadata_config())
        except ApiError:
            provider = 'custom'
        for mid, model in sorted(self._catalog.items()):
            cap = classify_model(provider, mid)
            if cap == 'unsupported':
                continue
            tag = 'Image analysis supported' if cap == 'supported' else 'Needs verification'
            label = f'{mid} · {tag}'
            self.model_choices.addItem(label, mid)
            self.model_choices.setItemData(self.model_choices.count()-1, model.display_name, Qt.ItemDataRole.ToolTipRole)
        self.model_choices.setCurrentIndex(max(0, self.model_choices.findData(selected)))
        self.model_choices.setEnabled(self.model_choices.count() > 1)

    def _restore_catalog(self):
        if not getattr(self, 'catalog_store', None):
            return
        try:
            config = self._metadata_config()
            provider, scope = infer_provider(config), credential_target(config)
            rows = self.catalog_store.get_models(provider, scope)
            self._catalog = {m.id: SimpleNamespace(model_id=m.id, display_name=m.display_name) for m in rows}
            self._populate_catalog()
            stamp = self.catalog_store.get_last_updated(provider, scope)
            self.models_status.setText(('Cached model list · Updated '+stamp[:19] if stamp else
                'Models from provider documentation · Account access not checked.') +
                ' Refresh manually using your selected key. No request was sent.')
        except (ApiError, SettingsError):
            pass

    def check_model(self):
        if self._check_worker is not None:
            return
        try:
            config = ApiConfig(endpoint=self.endpoint.text().strip(), model=self.model.text().strip(),
                               protocol=self.protocol.currentData(), timeout_seconds=15)
            if config.protocol != "gemini":
                raise ApiError("configuration")
            pending = ApiSettings(config, {}, self.key.text(), self.environment.text().strip(),
                                  self.key_source.currentData())
            key = resolve_api_key(pending, self.credentials)
            if not key:
                raise ApiError("configuration")
        except ApiError as error:
            self.model_check_status.setText(str(ModelCheckError(error)))
            return
        except (CredentialError, SettingsError) as error:
            self.model_check_status.setText(str(error) + " No request was sent.")
            return
        self._check_cancel = threading.Event()
        worker = _ModelCheckWorker(config, key, self._check_cancel)
        self._check_worker = worker
        self._metadata_checks += 1
        worker.signals.done.connect(self._model_checked)
        self._set_check_controls(False)
        self.model_check_status.setText(f"Checking model metadata with Google… Maximum wait: 15 seconds. Metadata checks in this dialog: {self._metadata_checks}. No generation request.")
        QThreadPool.globalInstance().start(worker)

    def _set_check_controls(self, enabled):
        for control in (self.protocol, self.endpoint, self.model, self.key, self.environment,
                        self.key_source, self.remember_settings, self.remember_key,
                        self.replace_key_button, self.delete_key_button,
                        self.model_check_button, self.save_button, self.load_models_button,
                        self.more_models_button):
            control.setEnabled(enabled)
        self.model_choices.setEnabled(enabled and bool(self._catalog))
        if enabled:
            self._update_key_controls()
            self.remember_settings.setEnabled(not self._storage_unavailable)

    def _model_checked(self, result, error):
        self._check_worker = None
        self._set_check_controls(True)
        if self._check_cancel is None or self._check_cancel.is_set():
            return
        if isinstance(error, ModelCheckError):
            text = str(error)
        elif isinstance(result, ModelCheckResult):
            text = result.text()
        else:
            text = "Model check failed. No generation request was sent."
        self.model_check_status.setText(text + f" Metadata checks in this dialog: {self._metadata_checks}.")

    def done(self, result):
        if self._check_cancel is not None:
            self._check_cancel.set()
        super().done(result)

    def validate_and_accept(self):
        try:
            if self._connection_mismatch():
                raise ValueError(self.connection_issue.text())
            config = ApiConfig(endpoint=self.endpoint.text().strip(), model=self.model.text().strip(),
                               protocol=self.protocol.currentData(), timeout_seconds=self.timeout.value(),
                               max_output_tokens=self.output_tokens.value())
            settings = {"call_limit": self.calls.value()}
            for key, widget in self.amounts.items():
                settings[key] = Decimal(widget.text().strip()) if widget.text().strip() else None
            SessionBudget(**settings)  # Validate without resetting the real ledger.
            environment = self.environment.text().strip()
            source = self.key_source.currentData()
            if source == "environment" and (not environment or not all(c.isascii() and (c.isalnum() or c == "_") for c in environment)
                                             or environment[0].isdigit()):
                raise ValueError("Enter an environment variable name containing letters, numbers or underscores.")
            entered_key = self.key.text() if source == "session" else ""
            if entered_key or self.remember_key.isChecked():
                ApiCoach(config, entered_key)  # Validation only; no request.
            result = ApiSettings(config, settings, entered_key, environment, source,
                                 self.remember_settings.isChecked())
        except (ValueError, InvalidOperation, ApiError) as error:
            QMessageBox.warning(self, "Check settings", str(error))
            return
        try:
            # Only an accepted, validated settings action can alter local storage.
            if self._delete_target is not None:
                self.credentials.delete(self._delete_target)
                self._delete_target = None
                self.delete_key_button.setText("Delete saved key")
            if self.remember_key.isChecked():
                self.credentials.write(credential_target(config), entered_key)
                result = replace(result, key_source="saved", api_key="")
            provider = infer_provider(config)
            names = {"gemini": "Google Gemini", "openai": "OpenAI", "custom": "Custom service"}
            existing = next((c for c in self.connection_book.connections
                             if c.credential_target == credential_target(config)), None)
            conn = connection_from_settings(result, connection_id=existing.id if existing else None,
                                             label=existing.label if existing else names[provider])
            if existing:
                conn = replace(conn, favorites=existing.favorites, recent=existing.recent)
            if self.connection_book is None:
                self.connection_book = ConnectionBook()
            self.connection_book = self.connection_book.with_connection(conn, activate=True)
            self.connection_book = replace(self.connection_book, budget_settings=settings)
            if result.remember_settings:
                self._session_only_ids.discard(conn.id)
            else:
                self._session_only_ids.add(conn.id)
            persistent_book = self.connection_book
            for session_id in self._session_only_ids:
                persistent_book = persistent_book.without(session_id)

            if self.settings_store is not None:
                if result.remember_settings:
                    self.connection_store.save(persistent_book)
                else:
                    try:
                        if persistent_book.connections:
                            self.connection_store.save(persistent_book)
                        else:
                            self.connection_store.clear()
                    except SettingsError as error:
                        # A read-only settings folder must not prevent session-only editing.
                        self.result_notice = str(error) + " This connection is in use for this session only; the previous saved connection may return on restart."
        except (CredentialError, SettingsError) as error:
            self.storage_status.setText(str(error) + " Your entries are still here. A completed key action may already have taken effect. You can use a session key with both Remember options off, or cancel and keep editing.")
            return
        self.result_book = self.connection_book
        self.result_session_only_ids = frozenset(self._session_only_ids)
        self.result_settings = result
        self.accept()


def budget_text(budget: SessionBudget) -> str:
    cost = budget.known_cost_usd
    known = "Unknown" if cost is None else f"${cost:.6f} (configured rates)"
    allocation = "Unknown" if budget.allocated_usd is None else f"${budget.allocated_usd:.6f}"
    cap = "off" if budget.budget_usd is None else f"${budget.budget_usd:.4f}"
    reserve = "not configured" if budget.reserve_usd is None else f"${budget.reserve_usd:.4f}"
    return (f"API attempts: {budget.attempts}/{budget.call_limit} · In flight: {budget.in_flight} (this app session)\n"
            f"Cost: {known} · Reserved / allocated: {allocation}\n"
            f"USD allocation cap: {cap} · Next attempt reserve: {reserve}\n"
            f"Requests with unknown cost: {budget.unresolved} (including successful requests without pricing).\n"
            "Failed requests may still be charged.")


class TextProbeReviewDialog(QDialog):
    """One disclosed generation attempt, independent of photos and edit state."""
    def __init__(self, parent, config, budget):
        super().__init__(parent)
        self.setWindowTitle("Review one text-only Gemini test")
        self.resize(680, 580)
        layout = QVBoxLayout(self)
        heading = QLabel(f"Send to: {request_endpoint(config)}\nModel: {config.model}\n"
                         f"Timeout: {config.timeout_seconds:g}s · Maximum output tokens: {config.max_output_tokens}")
        heading.setTextFormat(Qt.TextFormat.PlainText)
        heading.setWordWrap(True)
        layout.addWidget(heading)
        description = QLabel("Send the fixed text below to test content generation. This uses one shared API attempt and may incur charges. No photo, goal, question, edit history or JSON feedback schema is sent. Provider thinking defaults are unchanged. Nothing sends until you confirm.")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.payload = QTextEdit()
        self.payload.setReadOnly(True)
        self.payload.setPlainText(json.dumps(build_text_probe_payload(config), indent=2))
        layout.addWidget(self.payload, 1)
        costs = QLabel(budget_text(budget))
        costs.setWordWrap(True)
        layout.addWidget(costs)
        note = QLabel("Success means this text-only request completed. Photo analysis still needs separate verification. Failed or cancelled requests may be charged; no automatic retries.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Send text test · may incur charges")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class SendReviewDialog(QDialog):
    def __init__(self, parent, request, config, budget):
        super().__init__(parent)
        self.setWindowTitle("Ask coach")
        available = self.screen().availableGeometry()
        self.resize(min(600, available.width() - 40), min(490, available.height() - 60))
        outer = QVBoxLayout(self)
        self.body_scroll = QScrollArea()
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.body_scroll.setWidget(content)
        outer.addWidget(self.body_scroll, 1)
        overview = QHBoxLayout()
        text_column = QVBoxLayout()
        overview.addLayout(text_column, 1)
        layout.addLayout(overview)
        for title, value in (("Question", request.question or "Analyze my current edit."),
                             ("Goal", request.goal or "No style goal specified.")):
            text_column.addWidget(QLabel(title))
            text = QTextEdit()
            text.setReadOnly(True)
            text.setPlainText(value)
            text.setFixedHeight(64 if title == "Question" else 44)
            text_column.addWidget(text)
        current = QLabel()
        current.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixels = request.current_preview
        h, w, _ = pixels.shape
        image = QImage(pixels.data, w, h, pixels.strides[0], QImage.Format.Format_RGB888).copy()
        image.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
        current.setPixmap(QPixmap.fromImage(image).scaled(160, 120, Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation))
        overview.addWidget(current)
        self.include_before = QCheckBox("Also send the before preview")
        self.include_before.setChecked(True)
        layout.addWidget(self.include_before)
        summary = QLabel(f"{config.model} · {urlsplit(config.endpoint).hostname}\nMay incur charges. Session attempts: {budget.attempts}.")
        summary.setTextFormat(Qt.TextFormat.PlainText)
        summary.setWordWrap(True)
        outer.addWidget(summary)
        disclosure = QLabel("Sends selected small previews, editing parameters, recent edits, goal and question. No RAW, source path or EXIF/GPS.")
        disclosure.setWordWrap(True)
        layout.addWidget(disclosure)
        self.skip_session = QCheckBox("Don't ask again for this session")
        self.skip_session.setToolTip("Future clicks on Send to coach use this preview selection. Connection/model changes or restarting restore confirmation. Restore it anytime in Usage.")
        layout.addWidget(self.skip_session)
        self.details_button = QPushButton("Details")
        self.details_button.setCheckable(True)
        layout.addWidget(self.details_button)
        self.details = QScrollArea()
        self.details.setWidgetResizable(True)
        self.details.setFixedHeight(190)
        detail_content = QWidget()
        detail_layout = QVBoxLayout(detail_content)
        heading = QLabel(f"Send to: {request_endpoint(config)}\nModel: {config.model} · State v{request.state_version} · Timeout {config.timeout_seconds:g}s\n"
                         "Current edit is sent even when the editor is displaying the baseline.")
        heading.setTextFormat(Qt.TextFormat.PlainText)
        heading.setWordWrap(True)
        detail_layout.addWidget(heading)
        row = QHBoxLayout()
        self.before_label = None
        for name, pixels in ((f"Before · v{request.before_state_version}", request.before_preview),
                             (f"Current · v{request.state_version}", request.current_preview)):
            column = QVBoxLayout()
            column.addWidget(QLabel(name))
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            h, w, _ = pixels.shape
            image = QImage(pixels.data, w, h, pixels.strides[0], QImage.Format.Format_RGB888).copy()
            image.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
            label.setPixmap(QPixmap.fromImage(image).scaled(230, 100, Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation))
            column.addWidget(label)
            row.addLayout(column)
            if self.before_label is None:
                self.before_label = label
        detail_layout.addLayout(row)
        self.context = QTextEdit()
        self.context.setReadOnly(True)
        self.context.setMinimumHeight(180)
        detail_layout.addWidget(self.context)
        self.request = request
        self.include_before.toggled.connect(self.update_context)
        self.update_context()
        disclosure = QLabel("Only selected reduced sRGB previews and the editing information shown above are sent, with photography-coach instructions. No ARW, source path or source EXIF/GPS is attached. Review your own goal/question text for private details. Small previews cannot establish noise, sharpness or RAW highlight recoverability. Provider retention rules still apply; store=false is requested where supported.")
        disclosure.setWordWrap(True)
        detail_layout.addWidget(disclosure)
        costs = QLabel(budget_text(budget))
        costs.setWordWrap(True)
        detail_layout.addWidget(costs)
        self.details.setWidget(detail_content)
        self.details.hide()
        layout.addWidget(self.details)
        self.details_button.toggled.connect(self.details.setVisible)
        self.details_button.toggled.connect(lambda expanded: self.details_button.setText("Hide details" if expanded else "Details"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Send")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def update_context(self):
        include = self.include_before.isChecked()
        self.before_label.setVisible(include)
        self.context.setPlainText(json.dumps(request_context(self.request, include_before=include), indent=2, ensure_ascii=False)
                                  + "\n\nPhotography coach instructions:\n" + COACH_INSTRUCTIONS)
