"""OpenAI/compatible metadata transport: fake HTTPS, no API credentials or billing."""
import json
import threading
import pytest
from photocoach import api_coach as api
from photocoach.settings import credential_target


def test_model_free_openai_listing_uses_get_and_header_only(monkeypatch):
    calls=[]
    class Connection:
        sock=None
        def __init__(self,host,port,timeout):assert host=='api.openai.com' and timeout==15
        def connect(self):pass
        def request(self,method,path,*,body,headers):
            assert method=='GET' and path=='/v1/models' and body is None
            assert headers['Authorization']=='Bearer fake-models-key'
            assert 'fake-models-key' not in path
            calls.append(path)
        def getresponse(self):
            class Response:
                status=200
                def read(self,limit):return b'{"data":[{"id":"gpt-4o"},{"id":"o3-mini"},{"id":"gpt-4o"}]}'
            return Response()
        def close(self):pass
    monkeypatch.setattr(api.http.client,'HTTPSConnection',Connection)
    result=api.list_openai_models('https://api.openai.com/v1/responses','fake-models-key',threading.Event())
    assert result==('gpt-4o','o3-mini') and len(calls)==1
    assert not api._TRANSPORT_LOCK.locked()
    assert credential_target(api.ProviderModelsConfig('https://api.openai.com/v1/responses'))==credential_target(api.ApiConfig(model='gpt-4o'))


@pytest.mark.parametrize('status',[302,401,403,429,503])
def test_listing_errors_are_sanitized_and_not_retried(monkeypatch,status):
    calls=[]
    def transport(*a,**k):calls.append(1);return status,b'fake-key private body'
    monkeypatch.setattr(api,'_https_transport',transport)
    with pytest.raises(api.ModelCheckError) as caught:
        api.list_openai_models('https://api.openai.com/v1/responses','fake-key',threading.Event())
    assert caught.value.http_status==status and 'fake-key' not in str(caught.value)
    assert len(calls)==1


@pytest.mark.parametrize('body',[b'[]',b'{}',b'{"data":{}}',b'{"data":[null]}',b'{"data":[{"id":"fake-key"}]}',b'{"data":[{"id":"unsafe name"}]}'])
def test_invalid_listing_is_rejected(monkeypatch,body):
    monkeypatch.setattr(api,'_https_transport',lambda *a,**k:(200,body))
    with pytest.raises(api.ModelCheckError):
        api.list_openai_models('https://api.openai.com/v1/responses','fake-key',threading.Event())


def test_compatible_metadata_reuses_exact_generation_key_scope(monkeypatch):
    config=api.ProviderModelsConfig('https://provider.example/prefix/v1/chat/completions','chat_completions')
    assert config.models_endpoint=='https://provider.example/prefix/v1/models'
    assert credential_target(config)==credential_target(api.ApiConfig(config.endpoint,'custom','chat_completions'))
    assert not hasattr(config,'model')
    calls=[]
    def timeout(*a,**k):calls.append(1);raise api.ApiError('timeout')
    monkeypatch.setattr(api,'_https_transport',timeout)
    cancel=threading.Event();cancel.set()
    with pytest.raises(api.ModelCheckError):api.list_openai_models(config.endpoint,'fake-key',cancel)
    assert not calls
    with pytest.raises(api.ModelCheckError,match='timed out'):api.list_openai_models(config.endpoint,'fake-key',threading.Event())
    assert len(calls)==1
