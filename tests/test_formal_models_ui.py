"""Production coach selection with isolated storage, fake credentials and zero network."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import replace
from pathlib import Path
import threading
import pytest
from PySide6.QtWidgets import QApplication
from photocoach import api_coach as api
from photocoach.app import MainWindow, Job
from photocoach.connections import ConnectionStore, ConnectionBook, connection_from_settings
from photocoach.settings import ApiSettings, credential_target
from photocoach.coach import CoachFeedback


@pytest.fixture
def view(tmp_path, monkeypatch):
    app=QApplication.instance() or QApplication([])
    store=ConnectionStore(tmp_path/'settings.json')
    configs=[api.ApiConfig(api.GEMINI_ENDPOINT,'gemini-3.5-flash-lite','gemini'),
             api.ApiConfig('https://api.openai.com/v1/responses','gpt-4o','responses'),
             api.ApiConfig('https://one.example/v1/chat/completions','local-a','chat_completions'),
             api.ApiConfig('https://two.example/v1/chat/completions','local-b','chat_completions')]
    book=ConnectionBook(budget_settings={'call_limit':5})
    for i,config in enumerate(configs):
        book=book.with_connection(connection_from_settings(ApiSettings(config,{'call_limit':5},key_source='saved'),connection_id=f'c{i}'))
    book=book.with_active('c0');store.save(book)
    class Vault:
        def read(self,*a):pytest.fail('No automatic credential read')
    w=MainWindow(settings_store=store, credentials=Vault(),dev_mode=False)
    monkeypatch.setattr(api,'_https_transport',lambda *a,**k:pytest.fail('No network'))
    monkeypatch.setattr(w.network_pool,'start',lambda *a:pytest.fail('No network task'))
    w.coach_collapse.setChecked(False);w.show();app.processEvents()
    yield w
    w._network_active=None;w._close_approved=True;w.close();w.deleteLater();app.processEvents()


def test_switch_connections_keeps_inputs_and_budget_and_persists_active(view):
    w=view;w.goal.setText('Natural light');w.question.setText('How should I lift midtones?')
    height=w.coach_panel.height();ticket=w.api_budget.reserve();w.api_budget.settle(ticket,success=False)
    for cid in ['c1','c2','c3','c0']:
        w.coach_provider_picker.activated.emit(w.coach_provider_picker.findData(cid))
        assert w.connection_book.active==cid
        assert w.api_budget.attempts==1 and w._network_active is None
        assert w.goal.text()=='Natural light' and w.question.text()=='How should I lift midtones?'
        assert w.coach_panel.height()==height
        assert w.connection_store.load().active==cid
    assert len(w.connection_store.load().connections)==4


def test_favorites_manual_ids_and_recent_survive_restart_without_network(view):
    w=view;w._review_settings=w.api_settings
    w.coach_model_picker.setEditText('gemini-user-future')
    w._choose_typed_model()
    assert w.api_settings.config.model=='gemini-user-future' and w._review_settings is None
    w.coach_favorite.click()
    saved=w.connection_store.load().active_connection
    assert 'gemini-user-future' in saved.favorites and saved.recent[0]=='gemini-user-future'
    w2=MainWindow(settings_store=w.connection_store,credentials=w.credentials,dev_mode=False)
    try:
        assert w2.api_settings.config.model=='gemini-user-future'
        i=w2.coach_model_picker.findData('gemini-user-future')
        assert '[Favorite]' in w2.coach_model_picker.itemText(i)
        assert w2.coach_model_picker.completer().filterMode().name=='MatchContains'
        assert w2.api_budget.attempts==0
    finally:
        w2.close();w2.deleteLater()


def test_model_completion_inserts_id_without_capability_or_favorite_label(view):
    w = view
    w._select_coach_model('gemini-user-future')
    w.coach_favorite.click()
    completer = w.coach_model_picker.completer()
    completer.setCompletionPrefix('user-future')
    assert completer.completionCount() == 1
    index = completer.completionModel().index(0, 0)
    assert '[Favorite]' in index.data()
    assert completer.pathFromIndex(index) == 'gemini-user-future'


def test_clicking_search_completion_applies_only_model_id(view):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    w = view
    w._remember_coach_models(['gemini-user-future'])
    w._update_coach_status()
    picker = w.coach_model_picker
    picker.lineEdit().setFocus()
    picker.lineEdit().selectAll()
    QTest.keyClicks(picker.lineEdit(), 'user-future')
    QApplication.processEvents()
    completer = picker.completer()
    popup = completer.popup()
    index = completer.completionModel().index(0, 0)
    QTest.mouseClick(popup.viewport(), Qt.MouseButton.LeftButton,
                     pos=popup.visualRect(index).center())
    QApplication.processEvents()
    assert picker.currentText() == 'gemini-user-future'
    assert w.api_settings.config.model == 'gemini-user-future'
    assert w.api_budget.attempts == 0


def test_custom_catalogs_are_separate_and_unknown_models_remain_unverified(view):
    w=view
    for cid,mid in [('c2','only-one'),('c3','only-two')]:
        c=w.connection_book.get(cid)
        w.model_catalog_store.update_provider_models('custom',[mid],credential_target(c.config))
    w.coach_provider_picker.activated.emit(w.coach_provider_picker.findData('c2'))
    assert w.coach_model_picker.findData('only-one')>=0
    assert w.coach_model_picker.findData('only-two')==-1
    assert 'Needs verification' in w.coach_model_picker.itemText(w.coach_model_picker.findData('only-one'))
    assert 'Models updated' in w.coach_catalog_status.text()


def test_switches_are_rejected_during_network_request(view):
    w=view;before=w.api_settings
    w._network_active=object();w._update_coach_status()
    w.coach_provider_picker.activated.emit(w.coach_provider_picker.findData('c1'))
    w._select_coach_model('gemini-other')
    assert w.api_settings is before
    assert not w.coach_provider_picker.isEnabled() and not w.coach_model_picker.isEnabled()


def test_simulated_photo_result_does_not_record_live_verification(view):
    w=view;state=w.store.activate(Path('synthetic.ARW'),'photo-a')
    feedback=CoachFeedback(state.photo_id,state.session_id,state.revision,'Change','Why','Step','Stop',())
    job=Job('api_prepare',w._generation,lambda:None,state,w._coach_generation)
    ticket=w.api_budget.reserve();w._network_active=(None,job,threading.Event(),ticket)
    w._network_done(api.ApiResult(feedback,10,10),None)
    models=w.model_catalog_store.get_models('gemini',credential_target(w.api_settings.config))
    assert not any(m.verified_utc for m in models)
    assert w.api_budget.attempts==1

def test_metadata_cache_refresh_is_visible_after_cancel_without_changing_connection(view, monkeypatch):
    from photocoach.coach_ui import ApiSettingsDialog
    from photocoach.model_catalog import ModelCatalogStore
    from PySide6.QtWidgets import QDialog
    w=view;previous=w.api_settings
    scope=credential_target(previous.config)
    def cancel(dialog):
        ModelCatalogStore(w.model_catalog_store.path).update_provider_models('gemini',['gemini-refreshed'],scope)
        return QDialog.DialogCode.Rejected
    monkeypatch.setattr(ApiSettingsDialog,'exec',cancel)
    w.configure_api()
    assert w.api_settings is previous and w.api_budget.attempts==0
    assert w.coach_model_picker.findData('gemini-refreshed')>=0
    assert 'Models updated' in w.coach_catalog_status.text()


@pytest.mark.parametrize('accept', [False, True])
def test_model_settings_action_never_remains_as_model_draft(view, monkeypatch, accept):
    w = view
    previous = w.api_settings
    w.goal.setText('Natural light')
    w.question.setText('Keep this question')
    def configure():
        if accept:
            w.api_settings = replace(previous, config=replace(previous.config, model='gemini-chosen'))
    monkeypatch.setattr(w, 'configure_api', configure)
    picker = w.coach_model_picker
    picker.lineEdit().setFocus()
    QApplication.processEvents()
    index = picker.count() - 1
    picker.setCurrentIndex(index)
    picker.activated.emit(index)
    expected = 'gemini-chosen' if accept else previous.config.model
    assert picker.currentText() == expected
    w._choose_typed_model()
    assert w.api_settings.config.model == expected
    assert w.goal.text() == 'Natural light' and w.question.text() == 'Keep this question'
    assert w.api_budget.attempts == 0


def test_known_incompatible_model_does_not_dispatch_or_consume_budget(view):
    w=view
    w.coach_provider_picker.activated.emit(w.coach_provider_picker.findData('c1'))
    w._select_coach_model('o3-mini')
    w.store.activate(Path('synthetic.ARW'),'photo-a');w._set_controls(True)
    w.analyze()
    assert 'not suitable' in w.coach_text.toPlainText()
    assert w.api_budget.attempts==0 and w._network_active is None

def test_model_switch_keeps_ledger_and_limits_but_drops_other_model_prices(view):
    w=view
    w.api_budget.configure(input_usd_per_million='1',output_usd_per_million='2')
    ticket=w.api_budget.reserve();w.api_budget.settle(ticket,10,20,success=True)
    allocation=w.api_budget.allocated_usd
    w._select_coach_model('gemini-user-future')
    assert w.api_budget.attempts==1 and w.api_budget.call_limit==5
    assert w.api_budget.allocated_usd==allocation
    assert w.api_budget.input_usd_per_million is None and w.api_budget.output_usd_per_million is None
