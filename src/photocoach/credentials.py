"""PhotoCoach-only Windows Credential Manager access, without third-party packages.

Generic credential blobs are application-defined ASCII bytes, not UTF-16 passwords.
Keys remain Python strings while in use; wiping temporary native buffers is best effort
and cannot erase every copy made by Python, Windows, or an API transport.
"""
from __future__ import annotations

import ctypes
import os
import re
from typing import Callable


CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
ERROR_NOT_FOUND = 1168
MAX_CREDENTIAL_BLOB_BYTES = 5 * 512
MAX_TARGET_CHARACTERS = 256
_TARGET_PATTERN = re.compile(r"PhotoCoach/[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")

# Fixed-width Win32 types also keep fake-DLL tests correct on non-Windows hosts.
_DWORD = ctypes.c_uint32
_BOOL = ctypes.c_int32
_BYTE = ctypes.c_ubyte
_LPBYTE = ctypes.POINTER(_BYTE)


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", _DWORD), ("dwHighDateTime", _DWORD)]


class _CREDENTIAL_ATTRIBUTEW(ctypes.Structure):
    _fields_ = [("Keyword", ctypes.c_wchar_p), ("Flags", _DWORD),
                ("ValueSize", _DWORD), ("Value", _LPBYTE)]


class _CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", _DWORD), ("Type", _DWORD),
        ("TargetName", ctypes.c_wchar_p), ("Comment", ctypes.c_wchar_p),
        ("LastWritten", _FILETIME), ("CredentialBlobSize", _DWORD),
        ("CredentialBlob", _LPBYTE), ("Persist", _DWORD),
        ("AttributeCount", _DWORD),
        ("Attributes", ctypes.POINTER(_CREDENTIAL_ATTRIBUTEW)),
        ("TargetAlias", ctypes.c_wchar_p), ("UserName", ctypes.c_wchar_p),
    ]


_PCREDENTIALW = ctypes.POINTER(_CREDENTIALW)


class CredentialError(RuntimeError):
    """Fixed user-facing messages never include a key, target, or native error text."""
    def __init__(self, code: str):
        messages = {
            "unavailable": "Windows Credential Manager is unavailable. Use a key for this session only.",
            "target": "The saved-key destination is invalid.",
            "key": "Enter a nonempty API key containing printable ASCII characters without spaces.",
            "too_long": "This API key is too long for Windows Credential Manager. Use it for this session only.",
            "read": "The saved API key could not be read from Windows Credential Manager.",
            "write": "The API key could not be saved to Windows Credential Manager.",
            "delete": "The saved API key could not be deleted from Windows Credential Manager.",
            "invalid": "The saved API key is invalid. Replace or delete the saved key.",
        }
        self.code = code if isinstance(code, str) and code in messages else "unavailable"
        super().__init__(messages[self.code])


class _WinCredAPI:
    def __init__(self, dll, last_error: Callable[[], int]):
        self.CredReadW = dll.CredReadW
        self.CredReadW.argtypes = [ctypes.c_wchar_p, _DWORD, _DWORD,
                                   ctypes.POINTER(_PCREDENTIALW)]
        self.CredReadW.restype = _BOOL
        self.CredWriteW = dll.CredWriteW
        self.CredWriteW.argtypes = [_PCREDENTIALW, _DWORD]
        self.CredWriteW.restype = _BOOL
        self.CredDeleteW = dll.CredDeleteW
        self.CredDeleteW.argtypes = [ctypes.c_wchar_p, _DWORD, _DWORD]
        self.CredDeleteW.restype = _BOOL
        self.CredFree = dll.CredFree
        self.CredFree.argtypes = [ctypes.c_void_p]
        self.CredFree.restype = None
        self.last_error = last_error


def _load_api() -> _WinCredAPI:
    if os.name != "nt":
        raise CredentialError("unavailable")
    try:
        # Load the operating-system DLL from System32, never the working directory.
        dll = ctypes.WinDLL("Advapi32.dll", use_last_error=True, winmode=0x00000800)
        return _WinCredAPI(dll, ctypes.get_last_error)
    except Exception:
        raise CredentialError("unavailable") from None


def _validate_target(target: str) -> None:
    if (not isinstance(target, str) or len(target) > MAX_TARGET_CHARACTERS or
            _TARGET_PATTERN.fullmatch(target) is None):
        raise CredentialError("target")


def _validate_key(key: str) -> None:
    # Same character/length constraints as ApiCoach, followed by WinCred's byte cap.
    if (not isinstance(key, str) or not key or len(key) > 4096 or
            any(ord(character) < 33 or ord(character) > 126 for character in key)):
        raise CredentialError("key")
    if len(key) > MAX_CREDENTIAL_BLOB_BYTES:
        raise CredentialError("too_long")


def _wipe(pointer, size: int) -> None:
    """Best-effort cleanup of a known, bounded buffer; never masks the main result."""
    if pointer and 0 < size <= MAX_CREDENTIAL_BLOB_BYTES:
        try:
            ctypes.memset(pointer, 0, size)
        except Exception:
            pass


class WindowsCredentialStore:
    """Exact-target generic credentials for this Windows account and computer only.

    Construction does not access Windows. Read, write, and delete initialize the
    native API lazily. No credentials are enumerated and no keys are cached here.
    ``_loader`` is an offline-test injection point, never read from saved settings.
    """
    def __init__(self, *, _loader: Callable | None = None):
        self._loader = _loader
        self._api = None

    def _get_api(self):
        if self._api is None:
            try:
                self._api = (self._loader or _load_api)()
            except Exception:
                raise CredentialError("unavailable") from None
        return self._api

    def read(self, target: str) -> str | None:
        _validate_target(target)
        api = self._get_api()
        result = _PCREDENTIALW()
        blob = None
        blob_size = 0
        copy = None
        try:
            if not api.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(result)):
                if api.last_error() == ERROR_NOT_FOUND:
                    return None
                raise CredentialError("read")
            if not result:
                raise CredentialError("invalid")
            credential = result.contents
            blob = credential.CredentialBlob
            blob_size = int(credential.CredentialBlobSize)
            if (credential.Type != CRED_TYPE_GENERIC or not blob or
                    not 0 < blob_size <= MAX_CREDENTIAL_BLOB_BYTES):
                raise CredentialError("invalid")
            copy = (_BYTE * blob_size)()
            ctypes.memmove(copy, blob, blob_size)
            if any(value < 33 or value > 126 for value in copy):
                raise CredentialError("invalid")
            return bytes(copy).decode("ascii")
        except CredentialError:
            raise
        except Exception:
            raise CredentialError("read") from None
        finally:
            _wipe(copy, blob_size)
            _wipe(blob, blob_size)
            if result:
                try:
                    api.CredFree(result)
                except Exception:
                    pass

    def write(self, target: str, key: str) -> None:
        _validate_target(target)
        _validate_key(key)
        api = self._get_api()
        blob = (_BYTE * len(key))(*(ord(character) for character in key))
        try:
            credential = _CREDENTIALW()
            credential.Type = CRED_TYPE_GENERIC
            credential.TargetName = target
            credential.CredentialBlobSize = len(key)
            credential.CredentialBlob = ctypes.cast(blob, _LPBYTE)
            credential.Persist = CRED_PERSIST_LOCAL_MACHINE
            credential.UserName = "PhotoCoach"
            if not api.CredWriteW(ctypes.byref(credential), 0):
                raise CredentialError("write")
        except CredentialError:
            raise
        except Exception:
            raise CredentialError("write") from None
        finally:
            _wipe(blob, len(key))

    def delete(self, target: str) -> None:
        _validate_target(target)
        api = self._get_api()
        try:
            if not api.CredDeleteW(target, CRED_TYPE_GENERIC, 0):
                if api.last_error() == ERROR_NOT_FOUND:
                    return
                raise CredentialError("delete")
        except CredentialError:
            raise
        except Exception:
            raise CredentialError("delete") from None
