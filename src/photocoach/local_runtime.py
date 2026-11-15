"""Owned Ollama lifecycle and independent versioned local-model settings."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import ctypes
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from .ollama_client import MODEL, MODEL_BYTES, LocalError, exchange, model_identity, verify_vision


def default_models_dir():
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local/share"))) / "PhotoCoach" / "models"


@dataclass(frozen=True)
class LocalConfig:
    model_dir: str = ""
    model: str = MODEL
    call_limit: int = 20
    verified_digest: str = ""
    engine_version: str = ""
    engine_path: str = ""

    def __post_init__(self):
        if (not isinstance(self.model_dir, str) or (self.model_dir and not Path(self.model_dir).is_absolute())
                or self.model != MODEL or type(self.call_limit) is not int or not 1 <= self.call_limit <= 10000
                or not isinstance(self.verified_digest, str) or not isinstance(self.engine_version, str)
                or not isinstance(self.engine_path, str) or (self.engine_path and not Path(self.engine_path).is_absolute())):
            raise ValueError("Invalid local AI configuration.")


class LocalSettingsStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return LocalConfig()
        try:
            if self.path.stat().st_size > 16384:
                raise ValueError
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            if doc.pop("schema_version") != 1:
                raise ValueError
            return LocalConfig(**doc)
        except (OSError, ValueError, TypeError, KeyError):
            raise ValueError("Local AI settings could not be read. Choose the model folder again.") from None

    def save(self, config):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="local-ai-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"schema_version": 1, **asdict(config)}, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)


def find_ollama():
    binary = shutil.which("ollama")
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"
    return binary or (str(candidate) if candidate.is_file() else None)


class _WindowsJob:
    """Close kills only the process tree assigned to this private job."""
    def __init__(self, process):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64), ("Flags", w.DWORD),
                        ("Minimum", ctypes.c_size_t), ("Maximum", ctypes.c_size_t), ("Active", w.DWORD),
                        ("Affinity", ctypes.c_size_t), ("Priority", w.DWORD), ("Scheduling", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOps", "WriteOps", "OtherOps", "Read", "Write", "Other")]
        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("IO", IO), ("ProcessMemory", ctypes.c_size_t),
                        ("JobMemory", ctypes.c_size_t), ("PeakProcess", ctypes.c_size_t), ("PeakJob", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel, self.handle = kernel, kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (not self.handle or not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info))
                or not kernel.AssignProcessToJobObject(self.handle, int(process._handle))):
            self.close()
            raise RuntimeError("Unable to own the local engine process tree.")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class LocalRuntime:
    def __init__(self, store):
        self.store = store
        self.process = None
        self.job = None
        self.model_dir = None
        self.version = ""
        self._lock_file = None
        self._guard = threading.Lock()
        self.inference_lock = threading.Lock()

    def start(self, config, cancel):
        with self._guard:
            if self.process is not None and self.process.poll() is None:
                if Path(self.model_dir).resolve() != Path(config.model_dir).resolve():
                    raise RuntimeError("Stop Local AI before changing its model folder.")
                return
            if self.process is not None:
                self._stop()
            binary = config.engine_path or find_ollama()
            if not binary:
                raise RuntimeError("Ollama is not installed. Install it from ollama.com/download/windows, then detect again.")
            if not Path(binary).is_file():
                raise RuntimeError("The selected Ollama executable no longer exists. Choose or install it again.")
            folder = Path(config.model_dir)
            if not config.model_dir or not folder.is_absolute():
                raise ValueError("Choose an absolute model folder first.")
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=folder):
                pass
            self.store.path.parent.mkdir(parents=True, exist_ok=True)
            lock = (self.store.path.parent / "local-ai.lock").open("a+b")
            try:
                if os.name == "nt":
                    import msvcrt
                    lock.seek(0)
                    lock.write(b"0")
                    lock.flush()
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 11435))
            except OSError:
                lock.close()
                raise RuntimeError("PhotoCoach's local engine is already in use, or port 11435 is occupied.") from None
            self._lock_file = lock
            env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11435", OLLAMA_MODELS=str(folder), OLLAMA_NO_CLOUD="1",
                       OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_NUM_PARALLEL="1")
            try:
                self.process = subprocess.Popen([binary, "serve"], env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if os.name == "nt":
                    self.job = _WindowsJob(self.process)
                self.model_dir = str(folder)
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if cancel.is_set():
                        raise LocalError("cancelled")
                    if self.process.poll() is not None:
                        raise RuntimeError("Ollama could not start. Install the latest official version and detect again.")
                    try:
                        self.version = exchange("/api/version", cancel=cancel, timeout=1)["version"]
                        if not isinstance(self.version, str):
                            raise ValueError
                        return
                    except LocalError:
                        cancel.wait(.2)
                raise LocalError("timeout")
            except Exception:
                self._stop()
                raise

    def _stop(self):
        if self.job:
            self.job.close()
            self.job = None
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None
        if self._lock_file:
            self._lock_file.close()
            self._lock_file = None
        self.model_dir = None

    def stop(self):
        with self._guard:
            self._stop()

    def inspect(self, config, cancel):
        self.start(config, cancel)
        try:
            digest = model_identity(config.model, cancel=cancel)
        except LocalError as error:
            if error.code == "model":
                return "not_downloaded", replace(config, verified_digest="", engine_version="")
            raise
        ready = config.verified_digest == digest and config.engine_version == self.version
        return ("ready" if ready else "unverified"), (config if ready else replace(config, verified_digest="", engine_version=""))

    def download(self, config, cancel, progress):
        self.start(config, cancel)
        try:
            model_identity(config.model, cancel=cancel)
        except LocalError as error:
            if error.code != "model":
                raise
        else:
            return self.verify(config, cancel)
        blobs = Path(config.model_dir) / "blobs"
        cached = sum(p.stat().st_size for p in blobs.iterdir() if p.is_file() and not p.is_symlink()) if blobs.is_dir() else 0
        required = MODEL_BYTES + 5_000_000_000 - min(cached, MODEL_BYTES)
        if shutil.disk_usage(config.model_dir).free < required:
            raise RuntimeError("Allow 24 GB free space for a fresh download; cached layers reduce the remaining space requirement.")
        exchange("/api/pull", {"model": config.model, "stream": True}, cancel=cancel, timeout=7200, progress=progress)
        return self.verify(config, cancel)

    def verify(self, config, cancel):
        self.start(config, cancel)
        with self.inference_lock:
            digest = model_identity(config.model, cancel=cancel)
            verify_vision(config.model, cancel=cancel)
            if model_identity(config.model, cancel=cancel) != digest:
                raise LocalError("model")
        updated = replace(config, verified_digest=digest, engine_version=self.version)
        self.store.save(updated)
        return "ready", updated

    def ensure_ready(self, config, cancel):
        state, _ = self.inspect(config, cancel)
        if state != "ready":
            raise LocalError("vision")
