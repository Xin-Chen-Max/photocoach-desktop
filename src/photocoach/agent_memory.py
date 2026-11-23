"""Bounded Context Engineering: working memory, compaction, opt-in style memory, and Playbook RAG."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .agent_tools import ToolResult
from .state import EditState, EditValues


def _clean_summary_text(text: Any, limit: int) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or "")).strip()
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit]


def summarize_parameter_deltas(before_params: dict[str, Any], current_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return bounded, path-free parameter deltas between two validated EditValues dicts."""
    before_ev = EditValues.from_dict(before_params)
    current_ev = EditValues.from_dict(current_params)
    if before_ev == current_ev:
        return []
    b_dict = before_ev.to_dict()
    c_dict = current_ev.to_dict()
    deltas: list[dict[str, Any]] = []

    for name in ("exposure_ev", "contrast", "saturation", "highlights", "shadows", "whites", "blacks", "vibrance"):
        b_val = getattr(before_ev, name)
        c_val = getattr(current_ev, name)
        if b_val != c_val:
            deltas.append({
                "parameter": name,
                "before": round(float(b_val), 4),
                "after": round(float(c_val), 4),
                "delta": round(float(c_val - b_val), 4),
            })

    if before_ev.base_rendering != current_ev.base_rendering:
        deltas.append({
            "parameter": "base_rendering",
            "before": before_ev.base_rendering,
            "after": current_ev.base_rendering,
        })

    if before_ev.raw_options != current_ev.raw_options:
        deltas.append({
            "parameter": "white_balance",
            "before_mode": before_ev.raw_options.wb_mode,
            "after_mode": current_ev.raw_options.wb_mode,
            "before_gains": [round(float(g), 4) for g in before_ev.raw_options.wb_gains],
            "after_gains": [round(float(g), 4) for g in current_ev.raw_options.wb_gains],
        })

    if before_ev.composition != current_ev.composition:
        deltas.append({
            "parameter": "composition",
            "before_crop": [round(float(v), 4) for v in before_ev.composition.crop],
            "after_crop": [round(float(v), 4) for v in current_ev.composition.crop],
            "before_angle": round(float(before_ev.composition.angle), 3),
            "after_angle": round(float(current_ev.composition.angle), 3),
        })

    for field in ("distortion", "vignetting_ev"):
        b_val = getattr(before_ev.lens_corrections, field)
        c_val = getattr(current_ev.lens_corrections, field)
        if b_val != c_val:
            deltas.append({
                "parameter": f"lens_corrections.{field}",
                "before": round(float(b_val), 4),
                "after": round(float(c_val), 4),
                "delta": round(float(c_val - b_val), 4),
            })

    for field in ("sharpening", "masking", "luminance_nr", "chroma_nr"):
        b_val = getattr(before_ev.detail, field)
        c_val = getattr(current_ev.detail, field)
        if b_val != c_val:
            deltas.append({
                "parameter": f"detail.{field}",
                "before": round(float(b_val), 4),
                "after": round(float(c_val), 4),
                "delta": round(float(c_val - b_val), 4),
            })

    b_cm = b_dict["color_mixer"]
    c_cm = c_dict["color_mixer"]
    for band in b_cm:
        if b_cm[band] != c_cm[band]:
            deltas.append({
                "parameter": f"color_mixer.{band}",
                "before": b_cm[band],
                "after": c_cm[band],
            })

    b_cg = b_dict["color_grading"]
    c_cg = c_dict["color_grading"]
    for zone in ("shadows", "midtones", "highlights"):
        if b_cg[zone] != c_cg[zone]:
            deltas.append({
                "parameter": f"color_grading.{zone}",
                "before": b_cg[zone],
                "after": c_cg[zone],
            })
    for key in ("blend", "balance"):
        if b_cg[key] != c_cg[key]:
            deltas.append({
                "parameter": f"color_grading.{key}",
                "before": round(float(b_cg[key]), 4),
                "after": round(float(c_cg[key]), 4),
                "delta": round(float(c_cg[key] - b_cg[key]), 4),
            })

    if before_ev.rgb_curve != current_ev.rgb_curve:
        deltas.append({
            "parameter": "rgb_curve",
            "before_points": len(before_ev.rgb_curve),
            "after_points": len(current_ev.rgb_curve),
            "before_black_y": round(float(before_ev.rgb_curve[0][1]), 4),
            "after_black_y": round(float(current_ev.rgb_curve[0][1]), 4),
            "before_white_y": round(float(before_ev.rgb_curve[-1][1]), 4),
            "after_white_y": round(float(current_ev.rgb_curve[-1][1]), 4),
        })

    if before_ev.local_gradients != current_ev.local_gradients:
        before_by_id = {g.id: g for g in before_ev.local_gradients}
        after_by_id = {g.id: g for g in current_ev.local_gradients}
        deltas.append({
            "parameter": "local_gradients",
            "before_count": len(before_ev.local_gradients),
            "after_count": len(current_ev.local_gradients),
            "added": [_clean_summary_text(g.name, 40) for gid, g in after_by_id.items() if gid not in before_by_id][:4],
            "removed": [_clean_summary_text(g.name, 40) for gid, g in before_by_id.items() if gid not in after_by_id][:4],
            "changed": [_clean_summary_text(g.name, 40) for gid, g in after_by_id.items()
                        if gid in before_by_id and g != before_by_id[gid]][:4],
        })

    if before_ev.spots != current_ev.spots:
        deltas.append({
            "parameter": "spots",
            "before_count": len(before_ev.spots),
            "after_count": len(current_ev.spots),
            "after_heal": sum(1 for s in current_ev.spots if s.mode == "heal" and s.enabled),
            "after_clone": sum(1 for s in current_ev.spots if s.mode == "clone" and s.enabled),
        })

    if before_ev.goal != current_ev.goal:
        deltas.append({"parameter": "goal", "changed": True})

    return deltas[:16]


@dataclass(frozen=True)
class TurnSummary:
    state_version: int
    question: str
    decision: str
    objective: str
    tools_used: tuple[str, ...] = ()
    action_label: str = ""
    applied: bool = False
    parameters: dict[str, Any] | None = None
    proposed_actions: tuple[str, ...] = ()
    next_step: str = ""
    watch_stop: str = ""
    goal: str = ""


def _subsequent_edit_events(
    events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    photo_id: str,
    session_id: str,
    since_version: int,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("photo_id") != photo_id or event.get("session_id") != session_id:
            continue
        ver = event.get("state_version")
        if type(ver) is not int or ver <= since_version:
            continue
        etype = event.get("type")
        if etype in ("parameter", "undo", "redo", "reset", "snapshot_restore"):
            if event.get("phase") != "update":
                matched.append(event)
        elif etype == "gesture":
            if event.get("phase") == "end" and event.get("history_entry") is not None:
                matched.append(event)
    return matched


class SessionWorkingMemory:
    """In-memory ring buffer scoped to one (photo_id, session_id)."""

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max(1, int(max_turns))
        self._binding: tuple[str, str] | None = None
        self._entries: list[TurnSummary] = []

    def is_bound_to(self, photo_id: str, session_id: str) -> bool:
        return self._binding == (photo_id, session_id)

    def bind(self, photo_id: str, session_id: str) -> None:
        key = (photo_id, session_id)
        if self._binding != key:
            self._binding = key
            self._entries.clear()

    def clear(self) -> None:
        self._binding = None
        self._entries.clear()

    def record(self, photo_id: str, session_id: str, summary: TurnSummary) -> None:
        self.bind(photo_id, session_id)
        params = EditValues.from_dict(summary.parameters).to_dict() if summary.parameters is not None else None
        cleaned = TurnSummary(
            state_version=max(0, int(summary.state_version)),
            question=_clean_summary_text(summary.question, 400),
            decision=_clean_summary_text(summary.decision, 400),
            objective=_clean_summary_text(summary.objective, 400),
            tools_used=tuple(_clean_summary_text(t, 80) for t in summary.tools_used[:8] if _clean_summary_text(t, 80)),
            action_label=_clean_summary_text(summary.action_label, 200),
            applied=bool(summary.applied),
            parameters=params,
            proposed_actions=tuple(
                _clean_summary_text(a, 200) for a in summary.proposed_actions[:6] if _clean_summary_text(a, 200)
            ),
            next_step=_clean_summary_text(summary.next_step, 400),
            watch_stop=_clean_summary_text(summary.watch_stop, 400),
            goal=_clean_summary_text(summary.goal, 400),
        )
        self._entries.append(cleaned)
        if len(self._entries) > self.max_turns:
            self._entries = self._entries[-self.max_turns :]

    def mark_last_applied(self, action_label: str = "", *,
                          photo_id: str | None = None, session_id: str | None = None) -> None:
        if (photo_id is not None or session_id is not None) and self._binding != (photo_id, session_id):
            return
        if not self._entries:
            return
        last = self._entries[-1]
        cleaned_label = _clean_summary_text(action_label, 200) if action_label else ""
        self._entries[-1] = replace(
            last,
            applied=True,
            action_label=cleaned_label or last.action_label,
        )

    def entries(self) -> tuple[TurnSummary, ...]:
        return tuple(self._entries)

    def last_advice_for(self, photo_id: str, session_id: str) -> TurnSummary | None:
        if self._binding != (photo_id, session_id) or not self._entries:
            return None
        return self._entries[-1]

    def last_advice_summary(self, photo_id: str | None = None, session_id: str | None = None) -> dict[str, Any] | None:
        if (photo_id is not None or session_id is not None) and self._binding != (photo_id, session_id):
            return None
        if not self._entries:
            return None
        last = self._entries[-1]
        proposed = list(last.proposed_actions[:6]) if last.proposed_actions else (
            [last.action_label] if last.action_label else []
        )
        return {
            "state_version": last.state_version,
            "question": last.question,
            "decision": last.decision,
            "objective": last.objective,
            "next_step": last.next_step,
            "watch_stop": last.watch_stop,
            "proposed_actions": proposed[:6],
            "applied": bool(last.applied),
            "tools_used": list(last.tools_used[:8]),
            "has_parameters_snapshot": last.parameters is not None,
        }

    def has_subsequent_edits(
        self,
        photo_id: str,
        session_id: str,
        current_state_version: int,
        events: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        *,
        current_values: EditValues | None = None,
    ) -> bool:
        last = self.last_advice_for(photo_id, session_id)
        if last is None:
            return False
        if _subsequent_edit_events(events or (), photo_id, session_id, last.state_version):
            return True
        if current_values is not None and last.parameters is not None:
            try:
                if EditValues.from_dict(last.parameters).to_dict() != current_values.to_dict():
                    return True
            except (TypeError, ValueError):
                pass
        if not events and last.parameters is None and current_state_version > last.state_version:
            return True
        return False

    def follow_up_context(
        self,
        events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        current_state: EditState,
    ) -> dict[str, Any] | None:
        last = self.last_advice_for(current_state.photo_id, current_state.session_id)
        if last is None:
            return None
        prior_summary = self.last_advice_summary(current_state.photo_id, current_state.session_id)
        assert prior_summary is not None
        sub_events = _subsequent_edit_events(
            events or (), current_state.photo_id, current_state.session_id, last.state_version
        )
        subsequent_actions = [
            {
                "sequence": int(event["sequence"]),
                "state_version": int(event["state_version"]),
                "type": str(event["type"]),
                "parameter": _clean_summary_text(str(event.get("parameter") or "all"), 80),
            }
            for event in sub_events[-12:]
        ]
        undo_count = sum(1 for event in sub_events if event.get("type") == "undo")
        redo_count = sum(1 for event in sub_events if event.get("type") == "redo")
        reset_count = sum(1 for event in sub_events if event.get("type") == "reset")
        curr_dict = current_state.values.to_dict()
        deltas = summarize_parameter_deltas(last.parameters, curr_dict) if last.parameters is not None else []
        changed_since = bool(deltas) if last.parameters is not None else bool(sub_events)
        reverted = bool(last.parameters is not None and not deltas and bool(sub_events))
        payload = {
            "version": 1,
            "kind": "follow_up_review",
            "prior_advice": prior_summary,
            "current_state_version": int(current_state.revision),
            "revisions_elapsed": max(0, int(current_state.revision) - last.state_version),
            "subsequent_actions": subsequent_actions,
            "undo_count": undo_count,
            "redo_count": redo_count,
            "reset_count": reset_count,
            "parameter_deltas": deltas,
            "parameters_changed_since_advice": changed_since,
            "reverted_to_prior_parameters": reverted,
        }
        from .coach import validate_follow_up_context
        return validate_follow_up_context(payload, state_version=current_state.revision)

    def prompt_snippet(self, max_chars: int = 800) -> str:
        if not self._entries:
            return ""
        lines = []
        for item in self._entries:
            status = f"applied ({item.action_label})" if item.applied else (
                f"suggested ({item.action_label})" if item.action_label else "no edit"
            )
            tools = ", ".join(item.tools_used) if item.tools_used else "none"
            q = f"Q='{item.question[:80]}'" if item.question else "Q=review"
            lines.append(f"- v{item.state_version} {q} -> {item.decision}: {item.objective[:100]} [tools: {tools}; {status}]")
        joined = "\n".join(lines)
        return joined[:max_chars]


def _estimate_message_tokens(msg: dict[str, Any]) -> int:
    role = msg.get("role")
    if role == "user":
        text_len = len(msg.get("text", ""))
        img_count = len(msg.get("images", ()))
        return text_len // 4 + img_count * 256
    if role == "tool_results":
        total = 0
        for r in msg.get("results", ()):
            total += len(json.dumps(r.payload, ensure_ascii=False)) // 4
            total += len(r.images) * 256
        return total
    return len(json.dumps(msg, ensure_ascii=False, default=str)) // 4


def compact_conversation(messages: list[dict[str, Any]], max_est_tokens: int = 6000) -> list[dict[str, Any]]:
    """Evict older tool-returned images when context grows, keeping initial and latest images pinned."""
    copied = [dict(m) for m in messages]
    tool_img_indices = [
        idx for idx, msg in enumerate(copied)
        if msg.get("role") == "tool_results" and any(r.images for r in msg.get("results", ()))
    ]
    if len(tool_img_indices) <= 1 and sum(_estimate_message_tokens(m) for m in copied) <= max_est_tokens:
        return copied

    # Always preserve the initial user message (index 0) and the most recent tool_results turn's images.
    keep_idx = tool_img_indices[-1] if tool_img_indices else -1
    for idx in tool_img_indices:
        if idx == keep_idx:
            continue
        new_results = []
        for r in copied[idx].get("results", ()):
            if r.images:
                new_payload = dict(r.payload)
                new_payload["compacted_images"] = [img.label for img in r.images]
                new_results.append(ToolResult(r.call_id, r.name, r.ok, new_payload, images=(), error=r.error))
            else:
                new_results.append(r)
        copied[idx]["results"] = new_results
        if sum(_estimate_message_tokens(m) for m in copied) <= max_est_tokens and len(
            [i for i in tool_img_indices if i > idx]
        ) <= 1:
            break
    return copied


def _is_safe_preference(text: str) -> bool:
    if not isinstance(text, str):
        return False
    cleaned = text.strip()
    if not 3 <= len(cleaned) <= 160 or any(ord(c) < 32 for c in cleaned):
        return False
    low = cleaned.lower()
    forbidden = (":\\", ":/", "/", "\\", ".arw", ".nef", ".photocoach", "sk-", "aiza")
    if any(tok in low for tok in forbidden):
        return False
    if re.search(r"\b[0-9a-f]{32,}\b", low):
        return False
    return True


def _strict_json_load(raw: str) -> Any:
    def _no_dups(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("Duplicate JSON field.")
            out[k] = v
        return out

    def _no_const(token):
        raise ValueError(f"Non-finite JSON constant: {token}")

    return json.loads(raw, object_pairs_hook=_no_dups, parse_constant=_no_const)


class StyleMemoryStore:
    """Opt-in, bounded, path-free user editing style preference store."""

    MAX_BYTES = 32 * 1024
    MAX_PREFERENCES = 12

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from .settings import SettingsStore
            path = SettingsStore().path.with_name("agent_style_memory.json")
        self.path = Path(path)
        self._enabled = False
        self._preferences: list[str] = []
        self._loaded = False

    def load(self) -> None:
        self._loaded = True
        self._enabled = False
        self._preferences = []
        if not self.path.exists():
            return
        try:
            raw = self.path.read_bytes()
            if len(raw) > self.MAX_BYTES:
                raise ValueError
            doc = _strict_json_load(raw.decode("utf-8"))
            if (type(doc) is not dict or set(doc) != {"schema_version", "enabled", "preferences"}
                    or type(doc["schema_version"]) is not int or doc["schema_version"] != 1
                    or type(doc["enabled"]) is not bool or not isinstance(doc["preferences"], list)):
                raise ValueError
            prefs = [p.strip() for p in doc["preferences"] if _is_safe_preference(p)]
            self._enabled = doc["enabled"]
            self._preferences = prefs[: self.MAX_PREFERENCES]
        except (OSError, ValueError, TypeError, UnicodeError):
            self._enabled = False
            self._preferences = []

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    @property
    def enabled(self) -> bool:
        self._ensure_loaded()
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._ensure_loaded()
        self._enabled = bool(enabled)
        self.save()

    def preferences(self) -> tuple[str, ...]:
        self._ensure_loaded()
        return tuple(self._preferences)

    def add_preference(self, rule: str) -> bool:
        self._ensure_loaded()
        if not _is_safe_preference(rule):
            return False
        cleaned = rule.strip()
        if cleaned in self._preferences:
            self._preferences.remove(cleaned)
        self._preferences.append(cleaned)
        self._preferences = self._preferences[-self.MAX_PREFERENCES :]
        self.save()
        return True

    def record_applied_edit(self, before: EditValues, after: EditValues, goal: str = "") -> None:
        self._ensure_loaded()
        if not self._enabled:
            return
        candidates: list[str] = []
        if after.base_rendering != "legacy_v1":
            candidates.append(f"Uses {after.base_rendering} base rendering.")
        if after.highlights <= -0.10 and after.highlights < before.highlights:
            candidates.append("Protects bright highlights with negative linear Highlights.")
        if after.shadows >= 0.15 and after.shadows > before.shadows:
            candidates.append("Lifts dark regions gently via linear Shadows.")
        if after.vibrance > before.vibrance and after.saturation == before.saturation:
            candidates.append("Prefers muted-color Vibrance boost over global Saturation.")
        if len(after.local_gradients) > len(before.local_gradients):
            added = after.local_gradients[-1]
            if getattr(added, "warmth", 0.0) >= 0.2:
                candidates.append("Favors warm directional local lighting masks.")
        if after.color_grading != before.color_grading and not after.color_grading.is_identity:
            candidates.append("Uses subtle 3-way Color Grading tonal mood.")
        if not candidates and after != before:
            candidates.append("Applies conservative incremental adjustments.")
        for rule in candidates[:2]:
            self.add_preference(rule)

    def clear(self) -> None:
        self._loaded = True
        self._preferences = []
        self.save()

    def save(self) -> None:
        self._ensure_loaded()
        payload = {
            "schema_version": 1,
            "enabled": bool(self._enabled),
            "preferences": list(self._preferences[-self.MAX_PREFERENCES :]),
        }
        raw = (json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        if len(raw) > self.MAX_BYTES:
            raise ValueError("Style memory exceeds maximum allowed size.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".photocoach-style-", suffix=".tmp", dir=self.path.parent)
        tmp = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def prompt_snippet(self, max_chars: int = 400) -> str:
        self._ensure_loaded()
        if not self._enabled or not self._preferences:
            return ""
        joined = " ".join(self._preferences)
        return joined[:max_chars]


@dataclass(frozen=True)
class PlaybookCard:
    id: str
    title: str
    keywords: str
    guidance: str
    controls: tuple[str, ...]
    stop_when: str


PLAYBOOK_CARDS: tuple[PlaybookCard, ...] = (
    PlaybookCard(
        "pb_underexposure",
        "Underexposed RAW Midtone Recovery",
        "dark underexposed exposure lift shadows brighten noise 欠曝 提亮 曝光 阴影",
        "Lift global exposure_ev conservatively (+0.25 to +0.80 EV) in linear space, then check shadows (+0.15 to +0.35). Check 100% crop if ISO noise increases in deep shadows.",
        ("exposure_ev", "shadows"),
        "Stop lifting once display_ceiling_fraction rises sharply or shadow noise breaks down.",
    ),
    PlaybookCard(
        "pb_highlights_sky",
        "Bright Sky & Highlight Protection",
        "highlights blown sky overexposed bright white 高光 过曝 天空 压暗 死白",
        "Use negative linear Highlights (-0.15 to -0.45) before reducing global exposure_ev. For directional bright skies, consider a top-down linear mask (sky_filter).",
        ("highlights", "whites", "local_gradients"),
        "Stop if midtones become muddy or sky banding appears.",
    ),
    PlaybookCard(
        "pb_cfa_headroom",
        "Bayer Sensor Headroom vs Display Clipping",
        "raw clipping recoverability sensor headroom highlight recovery 传感器 宽容度 溢出 可恢复",
        "Call inspect_raw_headroom before claiming whether bright pixels are recoverable in RAW; display SDR ceiling (255) does not establish LibRaw sensor saturation.",
        ("highlights", "exposure_ev"),
        "Abstain from recovery claims when sensor_clipped_fraction > 0.",
    ),
    PlaybookCard(
        "pb_base_rendering",
        "Base Rendering Shoulder Semantics",
        "natural neutral legacy base rendering shoulder rolloff 基础渲染 自然 中性",
        "Natural v1 applies 2.5*Y/(1+1.5*Y) luminance rolloff with radial gamut compression. Never propose altering base_rendering via Coach actions (it is read-only).",
        ("highlights", "contrast", "whites"),
        "Do not duplicate Natural's midtone lift with aggressive extra contrast.",
    ),
    PlaybookCard(
        "pb_mountain_beam",
        "Golden-Hour Mountain Ridge Grazing Light",
        "mountain ridge peak sunlit beam golden warm 光束 丁达尔 山脊 山峰 金色 日落",
        "Call detect_scene_lighting_anchors first. Place an angled RadialGradient beam over the sunlit slope (+0.45 to +0.75 EV, warmth +0.35 to +0.56, feather >= 0.75) paired with a cool valley/sky falloff mask.",
        ("local_gradients",),
        "Stop if warm light spills onto cool blue sky or clips sunlit snow crests.",
    ),
    PlaybookCard(
        "pb_spotlight",
        "3D Subject Spotlight & Ambient Falloff",
        "subject spotlight portrait car street focus center 聚光 主体 立体感 压暗周围",
        "Detect subject anchor coordinates first, then combine a soft radial Key Light (+0.35 to +0.55 EV, feather ~0.70) with an inverted Ambient Falloff (-0.20 to -0.35 EV).",
        ("local_gradients",),
        "Stop if the transition edge reads as an artificial halo around the subject.",
    ),
    PlaybookCard(
        "pb_sunbeam",
        "Top-Down Sunbeam Glow",
        "sunbeam glow overhead ray atmosphere 耶稣光 顶光 透光",
        "Anchor a wide feathered radial mask (feather ~0.82, warmth ~0.48) bridging from the brightest upper horizon quadrant toward the subject.",
        ("local_gradients",),
        "Stop if contrast across the midtones washes out.",
    ),
    PlaybookCard(
        "pb_window_light",
        "Directional Side Window Light",
        "window side light directional portrait interior 窗户光 侧光 明暗对比",
        "Use two LinearGradients opposing each other: a warm key side (+0.40 to +0.55 EV, warmth +0.30) and a cool opposite shadow falloff (-0.20 to -0.30 EV).",
        ("local_gradients",),
        "Stop if shadow falloff crushes useful texture.",
    ),
    PlaybookCard(
        "pb_rim_backlight",
        "Backlit Rim & Halo Separation",
        "backlight rim halo silhouette 逆光 轮廓光",
        "Use a soft radial backlight glow (+0.45 to +0.65 EV, shadows +0.30, warmth +0.35 to +0.45) centered on the upper subject crest.",
        ("local_gradients",),
        "Stop if highlights around the rim clip to flat 255.",
    ),
    PlaybookCard(
        "pb_sky_filter",
        "Graduated Neutral / Cool Sky Filter",
        "graduated sky filter horizon dark blue 渐变滤镜 压暗天空 地平线",
        "Call detect_scene_lighting_anchors for horizon_y, then place a top-down LinearGradient ending near horizon_y + 0.06 (-0.30 to -0.50 EV, highlights -0.30).",
        ("local_gradients",),
        "Stop if tall mountain peaks or trees darken unnaturally across the horizon.",
    ),
    PlaybookCard(
        "pb_foreground_fill",
        "Foreground Fill Light",
        "foreground bottom dark road shadow fill 前景 补光 底部提亮",
        "Use a bottom-up LinearGradient ending just below the horizon (+0.25 to +0.45 EV, shadows +0.25 to +0.35).",
        ("local_gradients",),
        "Stop if foreground shadows become flat or noisy.",
    ),
    PlaybookCard(
        "pb_whites_blacks",
        "SDR Endpoint Shaping (Whites & Blacks)",
        "whites blacks dynamic range punch muddy 白色 黑色 黑位 白位 通透",
        "Whites and Blacks shape encoded endpoints c + 0.25*w*c^4 + 0.25*b*(1-c)^4 without RAW recovery. Small steps (±0.08 to ±0.25) set clean black/white anchor points.",
        ("whites", "blacks"),
        "Stop when display_floor_fraction or display_ceiling_fraction spikes past 1%.",
    ),
    PlaybookCard(
        "pb_contrast_curve",
        "S-Curve vs Global Contrast",
        "contrast curve flat gray midtones punch 对比度 曲线 发灰 通透",
        "Use gentle rgb_curve S-curves (e.g. [0.25, 0.22] and [0.75, 0.78]) for selective midtone separation instead of large global contrast (> 0.35), which can clip both ends.",
        ("rgb_curve", "contrast"),
        "Stop if skin or foliage shadows lose subtle tonal gradation.",
    ),
    PlaybookCard(
        "pb_faded_blacks",
        "Raised Curve Black Point / Gray Wash",
        "faded gray washed out matte black point 发灰 雾感 黑点",
        "Check rgb_curve[0][1]; values above 0.05 lift true blacks toward gray. Restore [0.0, 0.0] or lower Blacks slightly (-0.08) if crispness is desired.",
        ("rgb_curve", "blacks"),
        "Do not remove raised blacks if the user explicitly requested a matte film look.",
    ),
    PlaybookCard(
        "pb_vibrance_vs_saturation",
        "Vibrance vs Global Saturation",
        "vibrance saturation dull vivid muted color 自然饱和度 饱和度 鲜艳 色彩",
        "Prefer Vibrance (+0.10 to +0.30) to lift muted colors without clipping already vivid reds/blues; keep global Saturation near 1.00 (0.90 to 1.15).",
        ("vibrance", "saturation"),
        "Stop when any color channel clips or neon artifacts appear.",
    ),
    PlaybookCard(
        "pb_skin_hsl",
        "Skin-Tone Hue, Saturation & Lightness",
        "skin portrait face orange red complexion 肤色 人像 偏黄 偏红 橙色",
        "Inspect orange/red bands via analyze_color_zones. Adjust color_mixer.orange (lightness +5 to +15, saturation -10 to +10, subtle hue ±5) rather than heavy global saturation.",
        ("color_mixer",),
        "Remember PhotoCoach has no semantic skin detector; stop if background oranges shift unnaturally.",
    ),
    PlaybookCard(
        "pb_foliage_hsl",
        "Foliage & Landscape Greens",
        "foliage grass trees green yellow landscape 树叶 绿色 黄色 风景 植物",
        "Shift color_mixer.yellow toward +10..+20 or green saturation -10..-25 to tame harsh digital greens in sunlit foliage.",
        ("color_mixer",),
        "Stop if moss or grass loses natural warmth.",
    ),
    PlaybookCard(
        "pb_sky_hsl",
        "Sky & Water Blue/Aqua Depth",
        "blue aqua sky water sea cyan 蓝色 青色 天空 海水",
        "Deepen bright skies cleanly with color_mixer.blue.lightness (-10 to -25) and moderate saturation (+5 to +15) before resorting to heavy polarizer gradients.",
        ("color_mixer",),
        "Stop if sky noise or smooth-gradient banding appears.",
    ),
    PlaybookCard(
        "pb_color_grading",
        "3-Way Color Grading & Split Toning",
        "color grading split tone toning teal orange cinematic mood shadows midtones highlights blend balance hue wheel 色彩分级 分离色调 色轮 青橙 电影感 阴影偏冷 高光偏暖 混合 平衡",
        "Use color_grading.<shadows|midtones|highlights>.<hue|saturation|luminance> (hue 0..359, saturation 0..1, luminance -1..1) plus blend (0..1) and balance (-1..1). Always set saturation > 0 when introducing a hue tint, start subtle (saturation 0.12-0.30), and use visual verification checks for pure luma-neutral tints.",
        ("color_grading",),
        "Stop if neutral grays, snow, or skin tones pick up an unintended cast or midtone transitions turn muddy.",
    ),
    PlaybookCard(
        "pb_white_balance",
        "Sensor White Balance (Warmth/Tint & Neutral Picker)",
        "white balance wb warmth tint color cast yellow blue magenta green 白平衡 偏色 色温 色调",
        "Use analyze_color_zones to check midtone_cast_ab, and inspect_raw_headroom to verify neutral candidates. Adjust raw_options via relative sensor gains (mode='custom').",
        ("raw_options",),
        "Avoid shifting WB when warm golden light is intentional.",
    ),
    PlaybookCard(
        "pb_detail_sharpening",
        "Luminance Sharpening & Edge Masking",
        "sharpen sharpening blurry crisp halo edge masking 锐化 清晰度 边缘蒙版 光晕",
        "ALWAYS call inspect_region_100pct before proposing detail changes. Start with detail.sharpening 25-45 and detail.masking 10-30 so smooth skies stay unsharpened.",
        ("detail",),
        "Stop immediately if bright edge halos or sky grain amplification appear at 100%.",
    ),
    PlaybookCard(
        "pb_detail_nr",
        "High-ISO Luminance & Color Noise Reduction",
        "noise grain iso chroma luminance nr denoise 噪点 降噪 彩噪 明度降噪",
        "ALWAYS inspect a dark/midtone shadow area at 100% first. Prefer color_nr (detail.chroma_nr 20-35) to remove color speckles while keeping luminance_nr (0-20) conservative to preserve texture.",
        ("detail",),
        "Stop if fine fabric, hair, or foliage turns plastic/waxy.",
    ),
    PlaybookCard(
        "pb_camera_raw_preset",
        "Camera Raw Baseline Detail Preset",
        "camera raw preset default detail sharpening color nr 默认细节 预设",
        "PhotoCoach's calibrated Camera Raw detail baseline is sharpening=40, masking=10, luminance_nr=0, chroma_nr=25 (verified against A7C II XMP baseline), only after 100% crop verification.",
        ("detail",),
        "Reduce sharpening or raise masking on clean portraits.",
    ),
    PlaybookCard(
        "pb_lens_corrections",
        "Manual Lens Distortion & Vignetting",
        "lens distortion barrel pincushion vignette corners 镜头 畸变 暗角 边缘",
        "Use lens_corrections.distortion (-100..+100) for barrel/pincushion bowing and vignetting_ev (-2..+2 EV) for linear-light corner brightening or framing. Geometry changes invalidate regional luminance comparisons.",
        ("lens_corrections",),
        "Use visual verification checks when distortion changes.",
    ),
    PlaybookCard(
        "pb_stop_abstain",
        "Intentional Lighting & When to Stop",
        "stop finished good enough low key high key moody silhouette 停止 完成 暗调 剪影 不需要改",
        "If current parameters and preview already honor the user's goal (including intentional low-key shadows or silhouettes), choose decision='stop' with actions=[] instead of forcing unnecessary edits.",
        (),
        "Stop cleanly and explain why current tonal separation already works.",
    ),
    PlaybookCard(
        "pb_follow_up_review",
        "Follow-Up Diagnosis (Did My Adjustment Help?)",
        "follow up followup did it help adjustment review compare before after undo revert overshoot 复盘 效果 有帮助吗 变好了吗 有效吗 撤销",
        "When follow_up is present, give a direct verdict first (helped / mixed / neutral or reverted / regressed) comparing prior_advice, actual parameter_deltas, undo/redo history, and before-vs-current application_evidence. Separate measured pixel/histogram changes from subjective preference, and choose decision='stop' with actions=[] when the edit already met the goal or was reverted.",
        (),
        "Stop cleanly once the adjustment achieves the goal or was intentionally reverted, rather than stacking redundant edits.",
    ),
)


def _tokenize_bilingual(text: str) -> list[str]:
    low = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", low)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", low)
    for run in cjk_runs:
        for ch in run:
            tokens.append(ch)
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])
    return tokens


class PlaybookRAG:
    """Zero-dependency bilingual BM25 retriever over PhotoCoach-exact editing playbooks."""

    def __init__(self, cards: tuple[PlaybookCard, ...] = PLAYBOOK_CARDS) -> None:
        self.cards = cards
        self._docs = [
            _tokenize_bilingual(f"{c.title} {c.keywords} {c.guidance} {' '.join(c.controls)}")
            for c in cards
        ]
        self._tfs = [Counter(doc) for doc in self._docs]
        self._avg_dl = sum(len(d) for d in self._docs) / max(1, len(self._docs))
        n = len(self._docs)
        df: Counter[str] = Counter()
        for tf in self._tfs:
            for tok in tf:
                df[tok] += 1
        self._idf = {
            tok: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        q_tokens = _tokenize_bilingual(query)
        if not q_tokens:
            return []
        k1, b = 1.5, 0.75
        scored: list[tuple[float, PlaybookCard]] = []
        for idx, card in enumerate(self.cards):
            tf = self._tfs[idx]
            dl = max(1, len(self._docs[idx]))
            score = 0.0
            for tok in q_tokens:
                if tok not in tf:
                    continue
                f = tf[tok]
                idf = self._idf.get(tok, 0.0)
                score += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / self._avg_dl))
            if score > 0.0:
                scored.append((score, card))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            {
                "id": card.id,
                "title": card.title,
                "guidance": card.guidance,
                "controls": list(card.controls),
                "stop_when": card.stop_when,
                "score": round(score, 3),
            }
            for score, card in scored[: max(1, min(5, int(top_k)))]
        ]
