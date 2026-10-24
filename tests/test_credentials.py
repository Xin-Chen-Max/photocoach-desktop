"""WinCred tests use a fake DLL; the real roundtrip requires explicit opt-in."""
import ctypes
import os
import uuid

import pytest

from photocoach import credentials as cred


TARGET = "PhotoCoach/api/test-endpoint-fingerprint"
FAKE_KEY = "fake-key-for-offline-tests"


class FakeFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class FakeDLL:
    def __init__(self):
        self.records = {}
        self.calls = []
        self.allocated = []
        self.freed = []
        self.error = 0
        self.failure = None
        self.read_override = None
        self.CredReadW = FakeFunction(self.read)
        self.CredWriteW = FakeFunction(self.write)
        self.CredDeleteW = FakeFunction(self.delete)
        self.CredFree = FakeFunction(self.free)

    def read(self, target, kind, flags, result):
        self.calls.append(("read", target, kind, flags))
        if self.failure == "read":
            self.error = 5
            return 0
        if target not in self.records:
            self.error = cred.ERROR_NOT_FOUND
            return 0
        data = self.records[target]
        blob = (cred._BYTE * len(data))(*data)
        record = cred._CREDENTIALW()
        record.TargetName = target
        record.Type = cred.CRED_TYPE_GENERIC
        record.Persist = cred.CRED_PERSIST_LOCAL_MACHINE
        record.CredentialBlob = ctypes.cast(blob, cred._LPBYTE)
        record.CredentialBlobSize = len(data)
        if self.read_override:
            self.read_override(record)
        self.allocated.append((record, blob))
        ctypes.cast(result, ctypes.POINTER(cred._PCREDENTIALW))[0] = ctypes.pointer(record)
        return 1

    def write(self, pointer, flags):
        record = ctypes.cast(pointer, cred._PCREDENTIALW).contents
        self.calls.append(("write", record.TargetName, record.Type, flags))
        assert record.Persist == cred.CRED_PERSIST_LOCAL_MACHINE
        assert record.Flags == record.AttributeCount == 0
        assert not record.Attributes
        assert record.Comment is None and record.TargetAlias is None
        assert record.UserName == "PhotoCoach"
        if self.failure == "write":
            self.error = 5
            return 0
        self.records[record.TargetName] = ctypes.string_at(
            record.CredentialBlob, record.CredentialBlobSize)
        return 1

    def delete(self, target, kind, flags):
        self.calls.append(("delete", target, kind, flags))
        if self.failure == "delete":
            self.error = 5
            return 0
        if target not in self.records:
            self.error = cred.ERROR_NOT_FOUND
            return 0
        del self.records[target]
        return 1

    def free(self, pointer):
        self.freed.append(ctypes.addressof(ctypes.cast(pointer, cred._PCREDENTIALW).contents))


@pytest.fixture
def fake_store():
    dll = FakeDLL()
    api = cred._WinCredAPI(dll, lambda: dll.error)
    return cred.WindowsCredentialStore(_loader=lambda: api), dll


def test_win32_layout_and_function_signatures_use_fixed_width_types(fake_store):
    _, dll = fake_store
    assert ctypes.sizeof(cred._DWORD) == ctypes.sizeof(cred._BOOL) == 4
    assert ctypes.sizeof(cred._FILETIME) == 8
    is_64_bit = ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(cred._CREDENTIALW) == (80 if is_64_bit else 52)
    assert cred._CREDENTIALW.CredentialBlob.offset == (40 if is_64_bit else 28)
    assert cred._CREDENTIALW.Persist.offset == (48 if is_64_bit else 32)
    assert dll.CredReadW.argtypes == [ctypes.c_wchar_p, cred._DWORD, cred._DWORD,
                                     ctypes.POINTER(cred._PCREDENTIALW)]
    assert dll.CredWriteW.argtypes == [cred._PCREDENTIALW, cred._DWORD]
    assert dll.CredDeleteW.argtypes == [ctypes.c_wchar_p, cred._DWORD, cred._DWORD]
    assert dll.CredFree.argtypes == [ctypes.c_void_p]
    assert dll.CredFree.restype is None


def test_construction_is_lazy_and_loader_only_used_once(fake_store):
    _, dll = fake_store
    calls = []

    def loader():
        calls.append("load")
        return cred._WinCredAPI(dll, lambda: dll.error)

    store = cred.WindowsCredentialStore(_loader=loader)
    assert not calls
    assert store.read(TARGET) is None
    store.delete(TARGET)
    assert calls == ["load"]


def test_exact_target_roundtrip_replace_and_delete_without_enumeration(fake_store):
    store, dll = fake_store
    other = "PhotoCoach/api/other-endpoint"
    dll.records[other] = b"other-key"
    assert store.read(TARGET) is None
    store.write(TARGET, FAKE_KEY)
    assert dll.records[TARGET] == FAKE_KEY.encode("ascii")
    assert store.read(TARGET) == FAKE_KEY
    store.write(TARGET, "replacement-key")
    assert store.read(TARGET) == "replacement-key"
    store.delete(TARGET)
    store.delete(TARGET)
    assert store.read(TARGET) is None
    assert dll.records == {other: b"other-key"}
    assert all(call[1:] == (TARGET, cred.CRED_TYPE_GENERIC, 0) for call in dll.calls)
    assert len(dll.freed) == 2


@pytest.mark.parametrize("target", [None, 123, "", "PhotoCoach/", "OtherApp/key",
    "photocoach/key", "PhotoCoach/a\x00b", "PhotoCoach/a\nb", "PhotoCoach/*",
    "PhotoCoach/a b", "PhotoCoach/\\x", "PhotoCoach/é", "PhotoCoach/" + "a" * 247])
def test_invalid_targets_fail_before_loading_native_api(target):
    def loader():
        pytest.fail("invalid target must not load WinCred")

    store = cred.WindowsCredentialStore(_loader=loader)
    for operation in (lambda: store.read(target), lambda: store.write(target, FAKE_KEY),
                      lambda: store.delete(target)):
        with pytest.raises(cred.CredentialError, match="destination is invalid"):
            operation()


@pytest.mark.parametrize("key", [None, 123, "", " ", "a b", "a\tb", "a\nb",
                                       "a\x00b", "a\x7fb", "aéb", "a" * 4097])
def test_api_invalid_keys_fail_before_loading_native_api(key):
    def loader():
        pytest.fail("invalid key must not load WinCred")

    with pytest.raises(cred.CredentialError) as failure:
        cred.WindowsCredentialStore(_loader=loader).write(TARGET, key)
    assert failure.value.code == "key"


def test_ascii_blob_byte_limit_and_long_key_memory_only_error(fake_store):
    store, dll = fake_store
    key = "x" * cred.MAX_CREDENTIAL_BLOB_BYTES
    store.write(TARGET, key)
    assert len(dll.records[TARGET]) == cred.MAX_CREDENTIAL_BLOB_BYTES
    assert store.read(TARGET) == key
    for size in (cred.MAX_CREDENTIAL_BLOB_BYTES + 1, 4096):
        with pytest.raises(cred.CredentialError) as failure:
            store.write(TARGET, "x" * size)
        assert failure.value.code == "too_long"
        assert "session only" in str(failure.value)


@pytest.mark.parametrize("operation", ["read", "write", "delete"])
def test_native_failures_are_safe_and_not_confused_with_missing(fake_store, operation):
    store, dll = fake_store
    dll.failure = operation
    with pytest.raises(cred.CredentialError) as failure:
        if operation == "write":
            store.write(TARGET, FAKE_KEY)
        else:
            getattr(store, operation)(TARGET)
    assert failure.value.code == operation
    assert FAKE_KEY not in str(failure.value) and TARGET not in str(failure.value)


@pytest.mark.parametrize("operation", ["read", "write", "delete"])
def test_unexpected_native_exceptions_do_not_echo_keys_or_native_text(fake_store, operation):
    store, dll = fake_store

    def fail(*args):
        raise OSError("native-private-path " + FAKE_KEY)

    getattr(dll, "Cred" + operation.title() + "W").function = fail
    with pytest.raises(cred.CredentialError) as failure:
        if operation == "write":
            store.write(TARGET, FAKE_KEY)
        else:
            getattr(store, operation)(TARGET)
    assert failure.value.code == operation
    assert "native-private" not in str(failure.value) and FAKE_KEY not in str(failure.value)
    assert failure.value.__suppress_context__


def test_unavailable_platform_and_loader_error_are_sanitized(monkeypatch):
    monkeypatch.setattr(cred.os, "name", "posix")
    with pytest.raises(cred.CredentialError, match="unavailable"):
        cred.WindowsCredentialStore().read(TARGET)

    def loader():
        raise OSError(FAKE_KEY)

    with pytest.raises(cred.CredentialError) as failure:
        cred.WindowsCredentialStore(_loader=loader).read(TARGET)
    assert failure.value.code == "unavailable"
    assert FAKE_KEY not in str(failure.value)


@pytest.mark.parametrize("data", [b"", b"a b", b"a\x00b", b"a\xffb", b"a\nb"])
def test_malformed_stored_blob_is_rejected_and_always_freed(fake_store, data):
    store, dll = fake_store
    dll.records[TARGET] = data
    with pytest.raises(cred.CredentialError) as failure:
        store.read(TARGET)
    assert failure.value.code == "invalid"
    assert len(dll.freed) == 1
    assert bytes(dll.allocated[0][1]) == bytes(len(data))


@pytest.mark.parametrize("field,value", [("Type", 2), ("CredentialBlobSize", 2561),
                                         ("CredentialBlobSize", 0), ("CredentialBlob", None)])
def test_invalid_native_record_fails_before_unbounded_blob_read(fake_store, field, value):
    store, dll = fake_store
    dll.records[TARGET] = FAKE_KEY.encode("ascii")
    dll.read_override = lambda record: setattr(record, field, value)
    with pytest.raises(cred.CredentialError) as failure:
        store.read(TARGET)
    assert failure.value.code == "invalid"
    assert len(dll.freed) == 1


@pytest.mark.parametrize("fail_write", [False, True])
def test_write_temporary_buffer_is_wiped_on_success_and_failure(fake_store, monkeypatch, fail_write):
    store, dll = fake_store
    if fail_write:
        dll.failure = "write"
    wiped = []
    real_wipe = cred._wipe

    def spy(pointer, size):
        before = ctypes.string_at(pointer, size)
        real_wipe(pointer, size)
        wiped.append((before, ctypes.string_at(pointer, size)))

    monkeypatch.setattr(cred, "_wipe", spy)
    if fail_write:
        with pytest.raises(cred.CredentialError):
            store.write(TARGET, FAKE_KEY)
    else:
        store.write(TARGET, FAKE_KEY)
    assert wiped == [(FAKE_KEY.encode("ascii"), bytes(len(FAKE_KEY)))]


def test_read_native_and_temporary_buffers_are_wiped_before_free(fake_store, monkeypatch):
    store, dll = fake_store
    dll.records[TARGET] = FAKE_KEY.encode("ascii")
    wiped = []
    real_wipe = cred._wipe

    def spy(pointer, size):
        assert not dll.freed
        real_wipe(pointer, size)
        wiped.append(ctypes.string_at(pointer, size))

    monkeypatch.setattr(cred, "_wipe", spy)
    assert store.read(TARGET) == FAKE_KEY
    assert wiped == [bytes(len(FAKE_KEY)), bytes(len(FAKE_KEY))]
    assert len(dll.freed) == 1


@pytest.mark.skipif(os.name != "nt" or os.environ.get("PHOTOCOACH_TEST_WINDOWS_CREDENTIALS") != "1",
                    reason="Opt-in only: PHOTOCOACH_TEST_WINDOWS_CREDENTIALS=1 on Windows")
def test_native_windows_roundtrip_with_unique_fake_credential():
    # Never enumerate/read other targets or use a real provider key. Always clean up.
    target = "PhotoCoach/test/" + uuid.uuid4().hex
    store = cred.WindowsCredentialStore()
    try:
        assert store.read(target) is None
        store.write(target, "fake-PhotoCoach-test-key")
        assert store.read(target) == "fake-PhotoCoach-test-key"
        store.write(target, "replacement-fake-PhotoCoach-test-key")
        assert store.read(target) == "replacement-fake-PhotoCoach-test-key"
        store.delete(target)
        assert store.read(target) is None
    finally:
        store.delete(target)
