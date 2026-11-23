"""Owned or existing loopback Ollama lifecycle, hardware probing, and versioned local-model settings."""
from __future__ import annotations

import copy
import ctypes
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from typing import Any

from .ollama_client import (
    CATALOG_BY_ID,
    CuratedLocalModel,
    DEFAULT_NUM_CTX,
    DEFAULT_PORT,
    EXISTING_DEFAULT_PORT,
    LOCAL_MODEL_CATALOG,
    MODEL,
    MODEL_BYTES,
    LocalError,
    _call_exchange,
    _lookup_observed_capabilities,
    delete_installed_model,
    endpoint_for_port,
    exchange,
    get_curated_model,
    inspect_model_capabilities,
    is_cloud_proxied_model,
    model_download_bytes,
    model_has_thinking,
    model_identity,
    model_supports_tools,
    register_model_capabilities,
    valid_local_model_tag,
    valid_loopback_port,
    verify_vision,
)


def default_models_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local/share"))) / "PhotoCoach" / "models"


def parse_ollama_version(version_str: Any) -> tuple[int, int, int] | None:
    if not isinstance(version_str, str):
        return None
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", version_str.strip())
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def required_ollama_version_for(model: str) -> str:
    curated = get_curated_model(model)
    return curated.min_ollama_version if curated is not None else ""


def is_ollama_version_compatible(actual_version: str, model: str) -> tuple[bool, str]:
    min_ver = required_ollama_version_for(model)
    if not min_ver or not actual_version:
        return True, min_ver
    actual_t = parse_ollama_version(actual_version)
    min_t = parse_ollama_version(min_ver)
    if actual_t is None or min_t is None:
        return True, min_ver
    return actual_t >= min_t, min_ver


@dataclass(frozen=True)
class LocalHardwareInfo:
    system_ram_bytes: int = 0
    vram_bytes: int = 0
    gpu_name: str = ""
    free_disk_bytes: int = 0

    @property
    def system_ram_gb(self) -> float:
        return self.system_ram_bytes / (1024 ** 3) if self.system_ram_bytes > 0 else 0.0

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024 ** 3) if self.vram_bytes > 0 else 0.0

    @property
    def free_disk_gb(self) -> float:
        return self.free_disk_bytes / 1e9 if self.free_disk_bytes > 0 else 0.0


@dataclass(frozen=True)
class HardwareFit:
    model_id: str
    status: str  # "fits_vram" | "cpu_offload" | "heavy" | "disk_low"
    working_bytes: int
    required_disk_bytes: int
    disk_ok: bool
    is_installed: bool = False


@dataclass(frozen=True)
class DiscoveredModelRow:
    id: str
    title_en: str
    title_zh: str
    tier: str
    approx_bytes: int
    installed: bool
    digest: str = ""
    has_vision: bool = True
    supports_tools: bool = False
    has_thinking: bool = False
    verified: bool = False
    version_compatible: bool = True
    min_ollama_version: str = ""
    use_case_en: str = ""
    use_case_zh: str = ""
    is_curated: bool = True


@dataclass(frozen=True)
class LocalDiscoveryResult:
    engine_version: str
    models: tuple[DiscoveredModelRow, ...]
    hidden_non_vision_count: int = 0


def _probe_nvidia_smi_vram() -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1.5,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if proc.returncode != 0 or not proc.stdout:
            return 0, ""
        best_bytes = 0
        best_name = ""
        for raw_line in proc.stdout.splitlines():
            parts = [p.strip() for p in raw_line.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            try:
                mib = float(parts[-1])
            except ValueError:
                continue
            if mib > 0:
                vram_b = int(mib * 1024 * 1024)
                if vram_b > best_bytes:
                    best_bytes = vram_b
                    best_name = ", ".join(parts[:-1])[:80]
        return best_bytes, best_name
    except Exception:
        return 0, ""


def _parse_winreg_qw_memory_size(raw_val: Any) -> int:
    if isinstance(raw_val, int) and not isinstance(raw_val, bool) and raw_val > 0:
        return int(raw_val)
    if isinstance(raw_val, (bytes, bytearray)) and len(raw_val) == 8:
        return int.from_bytes(raw_val, "little", signed=False)
    return 0


def _probe_winreg_vram() -> tuple[int, str]:
    if os.name != "nt":
        return 0, ""
    try:
        import winreg

        class_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        best_bytes = 0
        best_name = ""
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_path) as class_key:
            idx = 0
            while True:
                try:
                    sub_name = winreg.EnumKey(class_key, idx)
                except OSError:
                    break
                idx += 1
                if not sub_name.startswith("0"):
                    continue
                try:
                    with winreg.OpenKey(class_key, sub_name) as sub_key:
                        try:
                            desc_val, _ = winreg.QueryValueEx(sub_key, "DriverDesc")
                            desc = str(desc_val or "").strip()
                        except OSError:
                            desc = ""
                        low_desc = desc.lower()
                        if any(skip in low_desc for skip in ("basic render", "basic display", "virtual", "remote")):
                            continue
                        raw_mem, _ = winreg.QueryValueEx(sub_key, "HardwareInformation.qwMemorySize")
                        vram_b = _parse_winreg_qw_memory_size(raw_mem)
                        if vram_b > best_bytes:
                            best_bytes = vram_b
                            best_name = desc[:80]
                except OSError:
                    continue
        return best_bytes, best_name
    except Exception:
        return 0, ""


def _probe_system_ram() -> int:
    if os.name == "nt":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
        except Exception:
            pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return 0


def _probe_free_disk_bytes(model_dir: str | Path | None) -> int:
    try:
        candidate = Path(model_dir).expanduser() if model_dir else default_models_dir()
        while not candidate.exists() and candidate.parent != candidate:
            candidate = candidate.parent
        return int(shutil.disk_usage(candidate).free)
    except Exception:
        return 0


def detect_local_hardware(model_dir: str | Path | None = None) -> LocalHardwareInfo:
    """Detect dedicated VRAM (nvidia-smi -> winreg qwMemorySize max), system RAM, and free disk space."""
    nv_vram, nv_name = _probe_nvidia_smi_vram()
    reg_vram, reg_name = _probe_winreg_vram() if nv_vram <= 0 else (0, "")
    if nv_vram >= reg_vram and nv_vram > 0:
        vram_bytes, gpu_name = nv_vram, nv_name
    else:
        vram_bytes, gpu_name = reg_vram, reg_name
    return LocalHardwareInfo(
        system_ram_bytes=_probe_system_ram(),
        vram_bytes=vram_bytes,
        gpu_name=gpu_name,
        free_disk_bytes=_probe_free_disk_bytes(model_dir),
    )


def classify_hardware_fit(
    model: str | CuratedLocalModel,
    hardware: LocalHardwareInfo,
    *,
    is_installed: bool = False,
    installed_size_bytes: int = 0,
) -> HardwareFit:
    """Classify model memory/disk fit including ~2.5-3.0 GB vision/KV-cache headroom."""
    entry = model if isinstance(model, CuratedLocalModel) else get_curated_model(model)
    model_id = entry.id if entry is not None else str(model)
    download_bytes = entry.approx_download_bytes if entry is not None else (installed_size_bytes or MODEL_BYTES)
    headroom_bytes = (
        entry.kv_vision_headroom_bytes
        if entry is not None
        else (3_000_000_000 if download_bytes >= 15_000_000_000 else 2_500_000_000)
    )
    working_bytes = download_bytes + headroom_bytes
    reserve_bytes = 5_000_000_000 if download_bytes >= 15_000_000_000 else 2_000_000_000
    required_disk_bytes = 0 if is_installed else (download_bytes + reserve_bytes)
    disk_ok = True if (is_installed or hardware.free_disk_bytes <= 0) else (hardware.free_disk_bytes >= required_disk_bytes)

    if hardware.vram_bytes > 0:
        if working_bytes <= hardware.vram_bytes:
            mem_status = "fits_vram"
        elif hardware.system_ram_bytes <= 0 or working_bytes <= int(hardware.system_ram_bytes * 0.70):
            mem_status = "cpu_offload"
        else:
            mem_status = "heavy"
    else:
        if hardware.system_ram_bytes <= 0 or working_bytes <= int(hardware.system_ram_bytes * 0.55):
            mem_status = "fits_vram"
        elif working_bytes <= int(hardware.system_ram_bytes * 0.75):
            mem_status = "cpu_offload"
        else:
            mem_status = "heavy"

    status = "disk_low" if not disk_ok else mem_status
    return HardwareFit(
        model_id=model_id,
        status=status,
        working_bytes=working_bytes,
        required_disk_bytes=required_disk_bytes,
        disk_ok=disk_ok,
        is_installed=is_installed,
    )


def recommend_best_model(
    hardware: LocalHardwareInfo,
    installed_ids: set[str] | frozenset[str] = frozenset(),
) -> str:
    """Return the highest-tier curated model that fits comfortably on this machine."""
    for entry in reversed(LOCAL_MODEL_CATALOG):
        fit = classify_hardware_fit(entry, hardware, is_installed=entry.id in installed_ids)
        if fit.status == "fits_vram" and fit.disk_ok:
            return entry.id
    for entry in reversed(LOCAL_MODEL_CATALOG):
        fit = classify_hardware_fit(entry, hardware, is_installed=entry.id in installed_ids)
        if fit.status == "cpu_offload" and fit.disk_ok:
            return entry.id
    return LOCAL_MODEL_CATALOG[0].id


def _normalize_verified_models(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict) or len(raw) > 64:
        raise ValueError("Invalid local AI configuration.")
    cleaned: dict[str, dict[str, Any]] = {}
    for model_id, item in raw.items():
        if not valid_local_model_tag(model_id) or not isinstance(item, dict):
            raise ValueError("Invalid local AI configuration.")
        digest = item.get("digest", "")
        engine_version = item.get("engine_version", "")
        supports_tools = item.get("supports_tools", False)
        has_thinking = item.get("has_thinking", False)
        if (
            not isinstance(digest, str)
            or not isinstance(engine_version, str)
            or type(supports_tools) is not bool
            or type(has_thinking) is not bool
        ):
            raise ValueError("Invalid local AI configuration.")
        if digest:
            curated = get_curated_model(model_id)
            if curated is not None and not curated.supports_agent_tools:
                supports_tools = False
            cleaned[model_id] = {
                "digest": digest,
                "engine_version": engine_version,
                "supports_tools": supports_tools,
                "has_thinking": has_thinking,
            }
    return cleaned


@dataclass(frozen=True)
class LocalConfig:
    model_dir: str = ""
    model: str = MODEL
    call_limit: int = 100
    verified_digest: str = ""
    engine_version: str = ""
    engine_path: str = ""
    service_mode: str = "managed"
    port: int = DEFAULT_PORT
    supports_tools: bool | None = None
    has_thinking: bool | None = None
    verified_models: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if (
            not isinstance(self.model_dir, str)
            or (self.model_dir and not Path(self.model_dir).is_absolute())
            or not valid_local_model_tag(self.model)
            or type(self.call_limit) is not int
            or not 1 <= self.call_limit <= 10000
            or not isinstance(self.verified_digest, str)
            or not isinstance(self.engine_version, str)
            or not isinstance(self.engine_path, str)
            or (self.engine_path and not Path(self.engine_path).is_absolute())
            or self.service_mode not in ("managed", "existing")
            or not valid_loopback_port(self.port)
        ):
            raise ValueError("Invalid local AI configuration.")

        cleaned_map = _normalize_verified_models(self.verified_models)
        curated = get_curated_model(self.model)
        existing_entry = cleaned_map.get(self.model)

        if self.supports_tools is None:
            resolved_tools = (
                bool(existing_entry["supports_tools"])
                if existing_entry is not None
                else (bool(curated.supports_agent_tools) if curated is not None else False)
            )
        elif type(self.supports_tools) is bool:
            resolved_tools = self.supports_tools and (curated.supports_agent_tools if curated is not None else True)
        else:
            raise ValueError("Invalid local AI configuration.")

        if self.has_thinking is None:
            resolved_thinking = (
                bool(existing_entry["has_thinking"])
                if existing_entry is not None
                else (bool(curated.has_thinking) if curated is not None else False)
            )
        elif type(self.has_thinking) is bool:
            resolved_thinking = self.has_thinking
        else:
            raise ValueError("Invalid local AI configuration.")

        if self.verified_digest:
            cleaned_map[self.model] = {
                "digest": self.verified_digest,
                "engine_version": self.engine_version,
                "supports_tools": resolved_tools,
                "has_thinking": resolved_thinking,
            }
            register_model_capabilities(
                self.model,
                has_vision=True,
                supports_tools=resolved_tools,
                has_thinking=resolved_thinking,
                port=self.port,
            )
        else:
            cleaned_map.pop(self.model, None)

        object.__setattr__(self, "supports_tools", resolved_tools)
        object.__setattr__(self, "has_thinking", resolved_thinking)
        object.__setattr__(self, "verified_models", copy.deepcopy(cleaned_map))

    @property
    def endpoint(self) -> str:
        return endpoint_for_port(self.port)

    def with_active_model(self, model_id: str) -> LocalConfig:
        """Switch active local model while restoring its stored verification & capability flags."""
        if not valid_local_model_tag(model_id):
            raise ValueError("Invalid local AI configuration.")
        entry = self.verified_models.get(model_id)
        curated = get_curated_model(model_id)
        if entry is not None and entry.get("digest"):
            return replace(
                self,
                model=model_id,
                verified_digest=entry["digest"],
                engine_version=entry.get("engine_version", self.engine_version),
                supports_tools=bool(entry.get("supports_tools", False)),
                has_thinking=bool(entry.get("has_thinking", False)),
            )
        return replace(
            self,
            model=model_id,
            verified_digest="",
            supports_tools=bool(curated.supports_agent_tools) if curated is not None else False,
            has_thinking=bool(curated.has_thinking) if curated is not None else False,
        )


class LocalSettingsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> LocalConfig:
        if not self.path.exists():
            return LocalConfig()
        try:
            if self.path.stat().st_size > 32768:
                raise ValueError
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                raise ValueError
            schema_ver = doc.pop("schema_version", None)
            if schema_ver not in (1, 2):
                raise ValueError
            return LocalConfig(**doc)
        except (OSError, ValueError, TypeError, KeyError):
            raise ValueError("Local AI settings could not be read. Choose the model folder again.") from None

    def save(self, config: LocalConfig) -> None:
        if not isinstance(config, LocalConfig):
            raise ValueError("Invalid local AI configuration.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="local-ai-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"schema_version": 2, **asdict(config)}, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)


def find_ollama() -> str | None:
    binary = shutil.which("ollama")
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"
    return binary or (str(candidate) if candidate.is_file() else None)


class _WindowsJob:
    """Close kills only the process tree assigned to this private job."""

    def __init__(self, process):
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [
                ("ProcessTime", ctypes.c_int64),
                ("JobTime", ctypes.c_int64),
                ("Flags", w.DWORD),
                ("Minimum", ctypes.c_size_t),
                ("Maximum", ctypes.c_size_t),
                ("Active", w.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("Priority", w.DWORD),
                ("Scheduling", w.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOps", "WriteOps", "OtherOps", "Read", "Write", "Other")]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("Basic", Basic),
                ("IO", IO),
                ("ProcessMemory", ctypes.c_size_t),
                ("JobMemory", ctypes.c_size_t),
                ("PeakProcess", ctypes.c_size_t),
                ("PeakJob", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel, self.handle = kernel, kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (
            not self.handle
            or not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info))
            or not kernel.AssignProcessToJobObject(self.handle, int(process._handle))
        ):
            self.close()
            raise RuntimeError("Unable to own the local engine process tree.")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class LocalRuntime:
    def __init__(self, store: LocalSettingsStore):
        self.store = store
        self.process = None
        self.job = None
        self.model_dir = None
        self.port = DEFAULT_PORT
        self.version = ""
        self._lock_file = None
        self._guard = threading.Lock()
        self.inference_lock = threading.Lock()

    def start(self, config: LocalConfig, cancel: threading.Event) -> None:
        with self._guard:
            if config.service_mode == "existing":
                if self.process is not None:
                    self._stop()
                resp = _call_exchange("/api/version", cancel=cancel, timeout=3, port=config.port)
                ver = resp.get("version") if isinstance(resp, dict) else None
                if not isinstance(ver, str) or not ver.strip():
                    raise LocalError("response")
                self.version = ver.strip()
                self.port = config.port
                return

            if self.process is not None and self.process.poll() is None:
                if Path(self.model_dir).resolve() != Path(config.model_dir).resolve() or self.port != config.port:
                    raise RuntimeError("Stop Local AI before changing its model folder or port.")
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
                    probe.bind(("127.0.0.1", config.port))
            except OSError:
                lock.close()
                raise RuntimeError(
                    f"PhotoCoach's local engine is already in use, or port {config.port} is occupied."
                ) from None
            self._lock_file = lock
            env = dict(
                os.environ,
                OLLAMA_HOST=f"127.0.0.1:{config.port}",
                OLLAMA_MODELS=str(folder),
                OLLAMA_NO_CLOUD="1",
                OLLAMA_MAX_LOADED_MODELS="1",
                OLLAMA_NUM_PARALLEL="1",
            )
            try:
                self.process = subprocess.Popen(
                    [binary, "serve"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                if os.name == "nt":
                    self.job = _WindowsJob(self.process)
                self.model_dir = str(folder)
                self.port = config.port
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if cancel.is_set():
                        raise LocalError("cancelled")
                    if self.process.poll() is not None:
                        raise RuntimeError("Ollama could not start. Install the latest official version and detect again.")
                    try:
                        self.version = _call_exchange("/api/version", cancel=cancel, timeout=1, port=config.port)["version"]
                        if not isinstance(self.version, str):
                            raise ValueError
                        return
                    except LocalError:
                        cancel.wait(0.2)
                raise LocalError("timeout")
            except Exception:
                self._stop()
                raise

    def _stop(self) -> None:
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

    def stop(self) -> None:
        with self._guard:
            self._stop()

    def _call_model_identity(self, model: str, cancel: threading.Event, port: int) -> str:
        if port != DEFAULT_PORT:
            return model_identity(model, cancel=cancel, port=port)
        return model_identity(model, cancel=cancel)

    def _resolved_capabilities(self, model: str, port: int) -> tuple[bool, bool]:
        """Return (supports_tools, has_thinking) from /api/show cache or catalog fallback."""
        curated = get_curated_model(model)
        observed = _lookup_observed_capabilities(model, port)
        if observed is not None:
            has_thinking = "thinking" in observed
            supports_tools = ("tools" in observed) and (curated.supports_agent_tools if curated is not None else True)
            return supports_tools, has_thinking
        return (
            bool(curated.supports_agent_tools) if curated is not None else False,
            bool(curated.has_thinking) if curated is not None else False,
        )

    def discover_models(self, config: LocalConfig, cancel: threading.Event) -> LocalDiscoveryResult:
        """List curated models + installed local vision models via /api/tags and POST /api/show."""
        self.start(config, cancel)
        tags_doc = _call_exchange("/api/tags", cancel=cancel, port=config.port)
        raw_models = tags_doc.get("models", []) if isinstance(tags_doc, dict) else []
        installed_by_id: dict[str, dict[str, Any]] = {}
        hidden_non_vision = 0

        for item in raw_models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            digest = item.get("digest")
            if not name or is_cloud_proxied_model(name, item) or not isinstance(digest, str) or not digest.strip():
                if name:
                    hidden_non_vision += 1
                continue
            if not valid_local_model_tag(name):
                hidden_non_vision += 1
                continue
            installed_by_id[name] = item

        rows: list[DiscoveredModelRow] = []
        seen_ids: set[str] = set()

        for entry in LOCAL_MODEL_CATALOG:
            seen_ids.add(entry.id)
            inst = installed_by_id.get(entry.id)
            ver_ok, min_ver = is_ollama_version_compatible(self.version, entry.id)
            if inst is not None:
                digest = str(inst["digest"])
                size_b = int(inst.get("size") or entry.approx_download_bytes)
                try:
                    cap_info = inspect_model_capabilities(entry.id, cancel=cancel, port=config.port)
                    has_vision = bool(cap_info["has_vision"])
                    supports_tools = bool(cap_info["supports_tools"])
                    has_thinking = bool(cap_info["has_thinking"])
                except LocalError:
                    has_vision = True
                    supports_tools = entry.supports_agent_tools
                    has_thinking = entry.has_thinking
                v_entry = config.verified_models.get(entry.id)
                is_verified = bool(
                    has_vision
                    and v_entry is not None
                    and v_entry.get("digest") == digest
                    and v_entry.get("engine_version") == self.version
                )
                rows.append(
                    DiscoveredModelRow(
                        id=entry.id,
                        title_en=entry.title_en,
                        title_zh=entry.title_zh,
                        tier=entry.tier,
                        approx_bytes=size_b,
                        installed=True,
                        digest=digest,
                        has_vision=has_vision,
                        supports_tools=supports_tools,
                        has_thinking=has_thinking,
                        verified=is_verified,
                        version_compatible=ver_ok,
                        min_ollama_version=min_ver,
                        use_case_en=entry.use_case_en,
                        use_case_zh=entry.use_case_zh,
                        is_curated=True,
                    )
                )
            else:
                rows.append(
                    DiscoveredModelRow(
                        id=entry.id,
                        title_en=entry.title_en,
                        title_zh=entry.title_zh,
                        tier=entry.tier,
                        approx_bytes=entry.approx_download_bytes,
                        installed=False,
                        digest="",
                        has_vision=True,
                        supports_tools=entry.supports_agent_tools,
                        has_thinking=entry.has_thinking,
                        verified=False,
                        version_compatible=ver_ok,
                        min_ollama_version=min_ver,
                        use_case_en=entry.use_case_en,
                        use_case_zh=entry.use_case_zh,
                        is_curated=True,
                    )
                )

        # Inspect non-catalog installed models via POST /api/show; hide non-vision models by default.
        for name, inst in sorted(installed_by_id.items()):
            if name in seen_ids:
                continue
            try:
                cap_info = inspect_model_capabilities(name, cancel=cancel, port=config.port)
            except LocalError:
                hidden_non_vision += 1
                continue
            if not cap_info["has_vision"]:
                hidden_non_vision += 1
                continue
            digest = str(inst["digest"])
            size_b = int(inst.get("size") or MODEL_BYTES)
            v_entry = config.verified_models.get(name)
            is_verified = bool(
                v_entry is not None
                and v_entry.get("digest") == digest
                and v_entry.get("engine_version") == self.version
            )
            rows.append(
                DiscoveredModelRow(
                    id=name,
                    title_en=f"{name} (Installed)",
                    title_zh=f"{name}（已安装自定义）",
                    tier="custom",
                    approx_bytes=size_b,
                    installed=True,
                    digest=digest,
                    has_vision=True,
                    supports_tools=bool(cap_info["supports_tools"]),
                    has_thinking=bool(cap_info["has_thinking"]),
                    verified=is_verified,
                    version_compatible=True,
                    min_ollama_version="",
                    use_case_en="Installed local vision model verified via /api/show.",
                    use_case_zh="本机已安装且通过 /api/show 视觉能力检查的模型。",
                    is_curated=False,
                )
            )

        return LocalDiscoveryResult(
            engine_version=self.version,
            models=tuple(rows),
            hidden_non_vision_count=hidden_non_vision,
        )

    def inspect(self, config: LocalConfig, cancel: threading.Event) -> tuple[str, LocalConfig]:
        self.start(config, cancel)
        ver_ok, _ = is_ollama_version_compatible(self.version, config.model)
        if not ver_ok:
            raise LocalError("version")
        try:
            digest = self._call_model_identity(config.model, cancel, config.port)
        except LocalError as error:
            if error.code == "model":
                updated_map = dict(config.verified_models)
                updated_map.pop(config.model, None)
                return "not_downloaded", replace(
                    config,
                    verified_digest="",
                    engine_version="",
                    verified_models=updated_map,
                )
            raise
        supports_tools, has_thinking = self._resolved_capabilities(config.model, config.port)
        v_entry = config.verified_models.get(config.model)
        ready = (
            config.verified_digest == digest and config.engine_version == self.version
        ) or (
            v_entry is not None
            and v_entry.get("digest") == digest
            and v_entry.get("engine_version") == self.version
            and config.engine_version in ("", self.version)
        )
        if ready:
            updated_map = dict(config.verified_models)
            updated_map[config.model] = {
                "digest": digest,
                "engine_version": self.version,
                "supports_tools": supports_tools,
                "has_thinking": has_thinking,
            }
            return "ready", replace(
                config,
                verified_digest=digest,
                engine_version=self.version,
                supports_tools=supports_tools,
                has_thinking=has_thinking,
                verified_models=updated_map,
            )
        updated_map = dict(config.verified_models)
        updated_map.pop(config.model, None)
        return "unverified", replace(
            config,
            verified_digest="",
            engine_version="",
            supports_tools=supports_tools,
            has_thinking=has_thinking,
            verified_models=updated_map,
        )

    def download(self, config: LocalConfig, cancel: threading.Event, progress) -> tuple[str, LocalConfig]:
        self.start(config, cancel)
        ver_ok, _ = is_ollama_version_compatible(self.version, config.model)
        if not ver_ok:
            raise LocalError("version")
        try:
            self._call_model_identity(config.model, cancel, config.port)
        except LocalError as error:
            if error.code != "model":
                raise
        else:
            return self.verify(config, cancel)
        expected_bytes = model_download_bytes(config.model)
        reserve_bytes = 5_000_000_000 if expected_bytes >= 15_000_000_000 else 2_000_000_000
        if config.service_mode == "managed" or config.model_dir:
            target_dir = Path(config.model_dir) if config.model_dir else default_models_dir()
            blobs = target_dir / "blobs"
            cached = (
                sum(p.stat().st_size for p in blobs.iterdir() if p.is_file() and not p.is_symlink())
                if blobs.is_dir()
                else 0
            )
            required = expected_bytes + reserve_bytes - min(cached, expected_bytes)
            if shutil.disk_usage(target_dir).free < required:
                req_gb = round((expected_bytes + reserve_bytes) / 1e9)
                raise RuntimeError(
                    f"Allow {req_gb} GB free space for a fresh download; cached layers reduce the remaining space requirement."
                )
        pull_kwargs: dict[str, Any] = {"cancel": cancel, "timeout": 7200, "progress": progress}
        if config.port != DEFAULT_PORT:
            pull_kwargs["port"] = config.port
        exchange("/api/pull", {"model": config.model, "stream": True}, **pull_kwargs)
        return self.verify(config, cancel)

    def verify(self, config: LocalConfig, cancel: threading.Event) -> tuple[str, LocalConfig]:
        self.start(config, cancel)
        ver_ok, _ = is_ollama_version_compatible(self.version, config.model)
        if not ver_ok:
            raise LocalError("version")
        with self.inference_lock:
            digest = self._call_model_identity(config.model, cancel, config.port)
            supports_tools, has_thinking = self._resolved_capabilities(config.model, config.port)
            register_model_capabilities(
                config.model,
                has_vision=True,
                supports_tools=supports_tools,
                has_thinking=has_thinking,
                port=config.port,
            )
            if config.port != DEFAULT_PORT:
                verify_vision(config.model, cancel=cancel, port=config.port, has_thinking=has_thinking)
            else:
                verify_vision(config.model, cancel=cancel)
            if self._call_model_identity(config.model, cancel, config.port) != digest:
                raise LocalError("model")
        updated_map = dict(config.verified_models)
        updated_map[config.model] = {
            "digest": digest,
            "engine_version": self.version,
            "supports_tools": supports_tools,
            "has_thinking": has_thinking,
        }
        updated = replace(
            config,
            verified_digest=digest,
            engine_version=self.version,
            supports_tools=supports_tools,
            has_thinking=has_thinking,
            verified_models=updated_map,
        )
        self.store.save(updated)
        return "ready", updated

    def delete_model(self, config: LocalConfig, target_model: str | None, cancel: threading.Event) -> tuple[str, LocalConfig]:
        model_id = (target_model or config.model).strip()
        self.start(config, cancel)
        with self.inference_lock:
            delete_installed_model(model_id, cancel=cancel, port=config.port)
        updated_map = dict(config.verified_models)
        updated_map.pop(model_id, None)
        if model_id == config.model:
            updated = replace(
                config,
                verified_digest="",
                engine_version="",
                verified_models=updated_map,
            )
            state = "not_downloaded"
        else:
            updated = replace(config, verified_models=updated_map)
            state = "ready" if updated.verified_digest else "not_downloaded"
        self.store.save(updated)
        return state, updated

    def ensure_ready(self, config: LocalConfig, cancel: threading.Event) -> None:
        state, _ = self.inspect(config, cancel)
        if state != "ready":
            raise LocalError("vision")
