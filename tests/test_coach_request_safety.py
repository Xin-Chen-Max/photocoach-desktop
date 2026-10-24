"""Regression checks for credential isolation, migration, and request limits."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from decimal import Decimal
import pytest
from PySide6.QtWidgets import QApplication, QDialog
from photocoach.api_coach import GEMINI_ENDPOINT
from photocoach.budget import SessionBudget
from photocoach.coach_ui import ApiSettingsDialog
from photocoach.connections import ConnectionStore


@pytest.fixture(scope='module')
def application():
    return QApplication.instance() or QApplication([])


class NoVault:
    def read(self,*args):pytest.fail('No real or automatic credential reads')
    def write(self,*args):pytest.fail('No credential writes')
    def delete(self,*args):pytest.fail('No credential deletion')


def test_new_provider_setup_preserves_live_budget_limits_and_ledger(application,tmp_path):
    budget=SessionBudget(call_limit=7,budget_usd='2',reserve_usd='.1',
                         input_usd_per_million='1',output_usd_per_million='2')
    budget.settle(budget.reserve(),success=False)
    before=(budget.attempts,budget.allocated_usd,budget.remaining_calls)
    dialog=ApiSettingsDialog(None,None,budget,settings_store=ConnectionStore(tmp_path/'settings.json'),credentials=NoVault())
    dialog.protocol.setCurrentIndex(dialog.protocol.findData('gemini'))
    dialog.model.setText('gemini-offline-fixture')
    dialog.key.setText('fake-session-key')
    dialog.remember_settings.setChecked(False)
    dialog.validate_and_accept()
    try:
        assert dialog.result()==QDialog.DialogCode.Accepted
        result=dialog.result_settings.budget_settings
        assert result['call_limit']==7
        assert result['budget_usd']==Decimal('2')
        assert result['reserve_usd']==Decimal('.1')
        # A new model may not inherit the other model's token prices.
        assert result['input_usd_per_million'] is None
        assert result['output_usd_per_million'] is None
        budget.configure(**result)
        assert (budget.attempts,budget.allocated_usd,budget.remaining_calls)==before
    finally:
        dialog.deleteLater()

def test_remembering_another_connection_does_not_serialize_previous_session_profile(application,tmp_path):
    from photocoach.settings import ApiSettings
    from photocoach.api_coach import ApiConfig
    store=ConnectionStore(tmp_path/'settings.json')
    private=ApiSettings(ApiConfig('https://private-session.example/v1/chat/completions','private-session-model','chat_completions'),{},
                        api_key='fake-private-key',remember_settings=False)
    dialog=ApiSettingsDialog(None,private,SessionBudget(),settings_store=store,credentials=NoVault())
    dialog.protocol.setCurrentIndex(dialog.protocol.findData('responses'))
    dialog.endpoint.setText('https://api.openai.com/v1/responses')
    dialog.model.setText('gpt-offline-fixture')
    dialog.key.setText('fake-other-session-key')
    dialog.remember_settings.setChecked(True)
    dialog.validate_and_accept()
    try:
        assert dialog.result()==QDialog.DialogCode.Accepted
        raw=store.path.read_text(encoding='utf-8')
        assert 'private-session.example' not in raw and 'private-session-model' not in raw
        assert 'fake-private-key' not in raw and 'fake-other-session-key' not in raw
        assert len(store.load().connections)==1
    finally:
        dialog.deleteLater()

def test_session_only_metadata_refresh_does_not_write_a_catalog(application,tmp_path):
    from types import SimpleNamespace
    import threading
    store=ConnectionStore(tmp_path/'settings.json')
    dialog=ApiSettingsDialog(None,None,SessionBudget(),settings_store=store,credentials=NoVault())
    dialog.remember_settings.setChecked(False)
    dialog._check_worker=SimpleNamespace(cancel=threading.Event(),page_token='')
    dialog._models_loaded(('offline-user-model',),None)
    assert dialog.model_choices.findData('offline-user-model')>=0
    assert not (tmp_path/'model_catalog.json').exists()
    dialog.reject();dialog.deleteLater()


def test_runtime_session_connection_survives_other_setup_without_being_saved(application,tmp_path,monkeypatch):
    from photocoach.app import MainWindow
    from photocoach.api_coach import ApiConfig
    from photocoach.settings import ApiSettings
    from photocoach.connections import ConnectionBook,connection_from_settings
    store=ConnectionStore(tmp_path/'settings.json')
    w=MainWindow(settings_store=store,credentials=NoVault(),dev_mode=False)
    private=ApiSettings(ApiConfig('https://private-session.example/v1/chat/completions','private-model','chat_completions'),
                        {},api_key='fake-private-session-key',remember_settings=False)
    conn=connection_from_settings(private,connection_id='private_session')
    w.api_settings=private;w.connection_book=ConnectionBook((conn,),conn.id,{})
    w._session_only_ids={conn.id};w._session_settings[conn.id]=private
    def apply(dialog):
        dialog.model.setText('gpt-offline-model');dialog.key.setText('fake-new-key')
        dialog.remember_settings.setChecked(True);dialog.validate_and_accept()
        return dialog.result()
    monkeypatch.setattr(ApiSettingsDialog,'exec',apply)
    try:
        w.configure_api(new_protocol='responses')
        assert w.connection_book.get(conn.id) is not None
        assert conn.id in w._session_only_ids
        assert 'private-session.example' not in store.path.read_text(encoding='utf-8')
        w._toggle_coach_favorite()
        assert 'private-model' not in store.path.read_text(encoding='utf-8')
        w._choose_coach_provider(w.coach_provider_picker.findData(conn.id))
        assert w.api_settings.api_key=='fake-private-session-key'
        assert not w.api_settings.remember_settings
        assert len(store.load().connections)==1
    finally:
        w._close_approved=True;w.close();w.deleteLater()

def test_unreadable_primary_cannot_be_replaced_even_if_directory_is_writable(tmp_path,monkeypatch):
    from photocoach.connections import ConnectionBook,connection_from_settings
    from photocoach.settings import ApiSettings,SettingsError
    from photocoach.api_coach import ApiConfig
    store=ConnectionStore(tmp_path/'settings.json')
    original=ConnectionBook().with_connection(connection_from_settings(ApiSettings(ApiConfig(model='gpt-offline-first'),{})))
    store.save(original);before=store.path.read_bytes();read=store._read
    def denied(path):
        if path==store.path:raise PermissionError('simulated primary read denial')
        return read(path)
    monkeypatch.setattr(store,'_read',denied)
    with pytest.raises(SettingsError):
        store.save(original.with_model(original.active,'gpt-offline-second'))
    assert store.path.read_bytes()==before
    assert not list(tmp_path.glob('.photocoach-settings-*'))


@pytest.mark.parametrize('damaged', [
    b'{"schema_version":1,"invalid":"keep these bytes"}',
    b'[' * 20000 + b'0' + b']' * 20000,
], ids=['invalid-schema1', 'deep-json'])
def test_invalid_primary_is_preserved_before_explicit_replacement(tmp_path, damaged):
    from photocoach.connections import ConnectionBook,connection_from_settings
    from photocoach.settings import ApiSettings
    from photocoach.api_coach import ApiConfig
    store=ConnectionStore(tmp_path/'settings.json')
    store.path.write_bytes(damaged)
    replacement=ConnectionBook().with_connection(connection_from_settings(ApiSettings(ApiConfig(model='gpt-offline'),{})))
    store.save(replacement)
    assert next(tmp_path.glob('settings.json.damaged-*')).read_bytes()==damaged
    assert store.load().active_connection.config.model=='gpt-offline'
