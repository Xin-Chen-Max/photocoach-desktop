"""Bounded coach context and local mock; provider transport lives separately."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
import math
import re
from typing import Any, Protocol, TYPE_CHECKING
from uuid import uuid4

import numpy as np

from .state import EDIT_PIPELINE_VERSION, EditState, EditValues
from .coach_actions import CoachAction, resolve_actions

if TYPE_CHECKING:
    from .coach_agent import AgentPlan, AgentReport


_FOLLOW_UP_TOP_KEYS = frozenset({
    "version",
    "kind",
    "prior_advice",
    "current_state_version",
    "revisions_elapsed",
    "subsequent_actions",
    "undo_count",
    "redo_count",
    "reset_count",
    "parameter_deltas",
    "parameters_changed_since_advice",
    "reverted_to_prior_parameters",
})
_PRIOR_ADVICE_KEYS = frozenset({
    "state_version",
    "question",
    "decision",
    "objective",
    "next_step",
    "watch_stop",
    "proposed_actions",
    "applied",
    "tools_used",
    "has_parameters_snapshot",
})
_SUBSEQUENT_ACTION_KEYS = frozenset({"sequence", "state_version", "type", "parameter"})
_SUBSEQUENT_ACTION_TYPES = frozenset({"parameter", "gesture", "undo", "redo", "reset", "snapshot_restore"})
_FORBIDDEN_FOLLOW_UP_KEYS = frozenset({"photo_id", "session_id", "source_path", "api_key", "path"})
_MAX_FOLLOW_UP_BYTES = 8192


def _validate_clean_text(value: Any, limit: int, field_name: str) -> str:
    if not isinstance(value, str) or len(value) > limit or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"Invalid follow-up text for {field_name}.")
    return value


def _validate_nonneg_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid non-negative integer for {field_name}.")
    return value


def _validate_json_safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        raise ValueError("Follow-up delta nesting depth exceeded.")
    if value is None or isinstance(value, bool):
        return value
    if type(value) is int:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Follow-up delta contains non-finite number.")
        return value
    if isinstance(value, str):
        return _validate_clean_text(value, 240, "delta")
    if isinstance(value, (list, tuple)):
        if len(value) > 16:
            raise ValueError("Follow-up delta list exceeds maximum length.")
        return [_validate_json_safe_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 16:
            raise ValueError("Follow-up delta mapping exceeds maximum keys.")
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = _validate_clean_text(k, 64, "delta key")
            if key.lower() in _FORBIDDEN_FOLLOW_UP_KEYS:
                raise ValueError(f"Forbidden key in follow-up context: {key}")
            out[key] = _validate_json_safe_value(v, depth=depth + 1)
        return out
    raise ValueError("Unsupported value type in follow-up delta.")


def validate_follow_up_context(follow_up: dict | None, *, state_version: int | None = None) -> dict[str, Any] | None:
    """Strictly validate and deep-copy bounded, path-free follow-up metadata."""
    if follow_up is None:
        return None
    if not isinstance(follow_up, dict) or set(follow_up) != _FOLLOW_UP_TOP_KEYS:
        raise ValueError("Invalid follow-up context structure.")
    if type(follow_up["version"]) is not int or follow_up["version"] != 1:
        raise ValueError("Unsupported follow-up context version.")
    if follow_up["kind"] != "follow_up_review":
        raise ValueError("Unsupported follow-up context kind.")

    current_v = _validate_nonneg_int(follow_up["current_state_version"], "current_state_version")
    if state_version is not None and current_v != state_version:
        raise ValueError("Follow-up state version does not match request state version.")
    revisions_elapsed = _validate_nonneg_int(follow_up["revisions_elapsed"], "revisions_elapsed")
    undo_count = _validate_nonneg_int(follow_up["undo_count"], "undo_count")
    redo_count = _validate_nonneg_int(follow_up["redo_count"], "redo_count")
    reset_count = _validate_nonneg_int(follow_up["reset_count"], "reset_count")
    if type(follow_up["parameters_changed_since_advice"]) is not bool:
        raise ValueError("parameters_changed_since_advice must be a bool.")
    if type(follow_up["reverted_to_prior_parameters"]) is not bool:
        raise ValueError("reverted_to_prior_parameters must be a bool.")

    raw_prior = follow_up["prior_advice"]
    if not isinstance(raw_prior, dict) or set(raw_prior) != _PRIOR_ADVICE_KEYS:
        raise ValueError("Invalid prior_advice structure in follow-up context.")
    prior_v = _validate_nonneg_int(raw_prior["state_version"], "prior_advice.state_version")
    if prior_v > current_v:
        raise ValueError("prior_advice.state_version cannot exceed current_state_version.")
    if type(raw_prior["applied"]) is not bool or type(raw_prior["has_parameters_snapshot"]) is not bool:
        raise ValueError("Invalid boolean flag in prior_advice.")
    raw_proposed = raw_prior["proposed_actions"]
    raw_tools = raw_prior["tools_used"]
    if not isinstance(raw_proposed, list) or len(raw_proposed) > 6:
        raise ValueError("Invalid proposed_actions in prior_advice.")
    if not isinstance(raw_tools, list) or len(raw_tools) > 8:
        raise ValueError("Invalid tools_used in prior_advice.")
    prior_advice = {
        "state_version": prior_v,
        "question": _validate_clean_text(raw_prior["question"], 400, "prior_advice.question"),
        "decision": _validate_clean_text(raw_prior["decision"], 400, "prior_advice.decision"),
        "objective": _validate_clean_text(raw_prior["objective"], 400, "prior_advice.objective"),
        "next_step": _validate_clean_text(raw_prior["next_step"], 400, "prior_advice.next_step"),
        "watch_stop": _validate_clean_text(raw_prior["watch_stop"], 400, "prior_advice.watch_stop"),
        "proposed_actions": [_validate_clean_text(item, 200, "proposed_action") for item in raw_proposed],
        "applied": raw_prior["applied"],
        "tools_used": [_validate_clean_text(item, 80, "tool_used") for item in raw_tools],
        "has_parameters_snapshot": raw_prior["has_parameters_snapshot"],
    }

    raw_subsequent = follow_up["subsequent_actions"]
    if not isinstance(raw_subsequent, list) or len(raw_subsequent) > 12:
        raise ValueError("Invalid subsequent_actions in follow-up context.")
    subsequent_actions: list[dict[str, Any]] = []
    for item in raw_subsequent:
        if not isinstance(item, dict) or set(item) != _SUBSEQUENT_ACTION_KEYS:
            raise ValueError("Invalid subsequent_actions entry.")
        if item["type"] not in _SUBSEQUENT_ACTION_TYPES:
            raise ValueError("Unsupported subsequent_actions type.")
        subsequent_actions.append({
            "sequence": _validate_nonneg_int(item["sequence"], "subsequent_actions.sequence"),
            "state_version": _validate_nonneg_int(item["state_version"], "subsequent_actions.state_version"),
            "type": item["type"],
            "parameter": _validate_clean_text(item["parameter"], 80, "subsequent_actions.parameter"),
        })

    raw_deltas = follow_up["parameter_deltas"]
    if not isinstance(raw_deltas, list) or len(raw_deltas) > 16:
        raise ValueError("Invalid parameter_deltas in follow-up context.")
    parameter_deltas: list[dict[str, Any]] = []
    for item in raw_deltas:
        if not isinstance(item, dict):
            raise ValueError("Each follow-up parameter delta must be a dict.")
        parameter_deltas.append(_validate_json_safe_value(item))

    cleaned = {
        "version": 1,
        "kind": "follow_up_review",
        "prior_advice": prior_advice,
        "current_state_version": current_v,
        "revisions_elapsed": revisions_elapsed,
        "subsequent_actions": subsequent_actions,
        "undo_count": undo_count,
        "redo_count": redo_count,
        "reset_count": reset_count,
        "parameter_deltas": parameter_deltas,
        "parameters_changed_since_advice": follow_up["parameters_changed_since_advice"],
        "reverted_to_prior_parameters": follow_up["reverted_to_prior_parameters"],
    }
    encoded = json.dumps(cleaned, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_FOLLOW_UP_BYTES:
        raise ValueError("Follow-up context exceeds byte limit.")
    return copy.deepcopy(cleaned)


@dataclass(frozen=True)
class CoachRequest:
    request_id: str
    photo_id: str
    session_id: str
    state_version: int
    before_state_version: int
    pipeline_version: str
    goal: str
    question: str
    parameters: dict
    before_parameters: dict
    recent_operations: tuple[dict, ...]
    before_preview: np.ndarray
    current_preview: np.ndarray
    source_size: tuple[int, int] | None = None
    follow_up: dict[str, Any] | None = None


def build_request(current: EditState, before: EditState, events: list[dict],
                  before_preview: np.ndarray, current_preview: np.ndarray, question="", *,
                  source_size=None, follow_up: dict[str, Any] | None = None) -> CoachRequest:
    if current.photo_id != before.photo_id or current.session_id != before.session_id:
        raise ValueError("Coach previews must belong to the same photo session.")
    if not isinstance(question, str) or len(question) > 2000:
        raise ValueError("The coach question must contain at most 2000 characters.")
    for pixels in (before_preview, current_preview):
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or max(pixels.shape[:2]) > 768:
            raise ValueError("Coach inputs require RGB previews no larger than 768 pixels.")
    validated_follow_up = validate_follow_up_context(follow_up, state_version=current.revision)
    # Crop changes can produce different aspect ratios. Each image belongs to
    # its own explicit parameter snapshot; never stretch one to match the other.
    operations = tuple({key: copy.deepcopy(event[key]) for key in
                        ("sequence", "type", "parameter", "before", "after", "state_version")}
                       for event in events if event["photo_id"] == current.photo_id and
                       event["session_id"] == current.session_id and event["type"] in
                       ("parameter", "reset", "undo", "redo", "snapshot_restore") and event["phase"] != "update")[-12:]
    # Committed drags have their final values on the gesture's end event.
    committed = [event for event in events if event["photo_id"] == current.photo_id and
                 event["session_id"] == current.session_id and event.get("history_entry")]
    combined = {item["sequence"]: item for item in operations}
    for event in committed[-12:]:
        combined[event["sequence"]] = {key: copy.deepcopy(event[key]) for key in
                                     ("sequence", "type", "parameter", "before", "after", "state_version")}
    operations = tuple(combined[key] for key in sorted(combined)[-12:])
    images = [np.array(pixels, copy=True) for pixels in (before_preview, current_preview)]
    for image in images:
        image.setflags(write=False)
    return CoachRequest(str(uuid4()), current.photo_id, current.session_id, current.revision, before.revision,
                        EDIT_PIPELINE_VERSION, current.goal, question, current.values.to_dict(),
                        before.values.to_dict(), operations, *images, source_size, validated_follow_up)


@dataclass(frozen=True)
class CoachFeedback:
    photo_id: str
    session_id: str
    state_version: int
    main_change: str
    reason: str
    next_step_or_stop: str
    observe_and_stop_when: str
    evidence: tuple[str, ...]
    source: str = "MOCK · Local rules"
    answer: str = ""
    question: str = ""
    actions: tuple[CoachAction, ...] = ()
    action_notice: str = ""
    agent_plan: AgentPlan | None = None
    agent_report: AgentReport | None = None
    agent_trace: tuple[Any, ...] = ()
    inspected_crops: tuple[tuple[str, np.ndarray], ...] = field(default=(), repr=False, compare=False)
    follow_up_context: dict[str, Any] | None = None

    def text(self) -> str:
        response = (f"Question: {self.question}\n\n" if self.question else "")
        if self.answer:
            response += f"Answer: {self.answer}\n\n"
        workflow = ""
        if self.agent_plan:
            workflow = (f"\n\nWorkflow: {self.agent_plan.decision} · {self.agent_plan.objective}\n"
                        + "\n".join(f"{f.region}: {f.observation} ({f.category}, {f.confidence})"
                                    for f in self.agent_plan.findings)
                        + f"\nUncertainty: {self.agent_plan.uncertainty}")
        if self.agent_report:
            workflow += "\n" + "\n".join(self.agent_report.lines())
        if self.agent_trace:
            workflow += "\nTrace: " + " → ".join(
                f"{getattr(s, 'name', s)}" for s in self.agent_trace
            )
        return (f"{self.source} · State v{self.state_version}\n\n" + response +
                f"Change: {self.main_change}\n\nWhy: {self.reason}\n\n"
                f"Next: {self.next_step_or_stop}\n\nWatch / stop: {self.observe_and_stop_when}\n\n"
                + "\n".join(self.evidence) + workflow)


class CoachAdapter(Protocol):
    def analyze(self, request: CoachRequest) -> CoachFeedback: ...


class MockCoach:
    def __init__(self):
        self.calls = 0
        self.limit = 100
        self.paused = False
        self.last_request: CoachRequest | None = None

    def analyze(self, request: CoachRequest) -> CoachFeedback:
        if self.paused:
            raise RuntimeError("The mock coach is paused.")
        if self.calls >= self.limit:
            raise RuntimeError("The session's mock analysis limit has been reached.")
        self.calls += 1
        self.last_request = request  # Only one bounded request is retained.
        if request.follow_up is not None:
            return self._analyze_follow_up(request)
        p, before = request.parameters, request.before_parameters
        changed = [key for key in dict.fromkeys(list(p) + list(before)) if p.get(key) != before.get(key)]
        delta = float(request.current_preview.astype(np.float32).mean() - request.before_preview.astype(np.float32).mean()) / 255
        observed = "The supplied preview is brighter overall." if delta > .01 else (
            "The supplied preview is darker overall." if delta < -.01 else "Average preview brightness changed little.")
        change = f"Changed {', '.join(changed)}. {observed}" if changed else f"No committed edit to compare. {observed}"
        proposals = []
        if p.get('local_gradients') != before.get('local_gradients'):
            items=p.get('local_gradients',[])
            change+=' Local gradients: '+('; '.join(f"{g['name']}: Exposure {g['exposure_ev']:+.2f} EV, Highlights {g['highlights']:+.2f}, Shadows {g['shadows']:+.2f}" for g in items) or 'none')
            reason='Manual gradient weights apply only by position, not subject detection. Local exposure adds in stops; overlapping tonal amounts combine with global settings and are limited.'
            next_step=f"Compare local gradient {items[-1]['name']} enabled and disabled before another adjustment." if items else 'Compare the current edit with Baseline before another adjustment.'
            watch='Inspect both sides of the transition. Stop if the light looks artificial, texture looks flat, or important bright areas lose separation.'
        elif p.get("spots") != before.get("spots"):
            items = p.get("spots", [])
            heal_count = sum(1 for s in items if s.get("mode") == "heal" and s.get("enabled", True))
            clone_count = sum(1 for s in items if s.get("mode") == "clone" and s.get("enabled", True))
            change += f" Spot removal: {len(items)} spot(s) ({heal_count} heal, {clone_count} clone)."
            reason = (
                "Spot removal applies circular heal or clone patches in original-oriented linear RGB; "
                "heal matches target perimeter brightness smoothly while clone copies sampled source pixels directly."
            )
            next_step = "Inspect repaired edges at 100% zoom and adjust Spot size, Feather, or Source position if needed."
            watch = "Watch for repeating texture or perimeter seams around high-contrast boundaries."
        elif p.get("detail") != before.get("detail"):
            det = p.get("detail", {"sharpening": 0, "masking": 0, "luminance_nr": 0, "chroma_nr": 0})
            change += (f" Detail — Sharpening {det['sharpening']:g}, Edge masking {det['masking']:g}, "
                       f"Luminance NR {det['luminance_nr']:g}, Color NR {det['chroma_nr']:g}.")
            reason = "Detail controls apply luminance unsharp masking and opponent chroma noise reduction in perceptual sRGB; inspect at 100% zoom."
            next_step = "Check fine edges and smooth sky or shadow areas at 100% zoom before increasing Sharpening further."
            watch = "Watch for bright edge halos or amplified grain in flat areas; raise Edge masking or lower Sharpening if halos appear."
        elif p["composition"] != before["composition"]:
            reason = "Composition changed the included scene or its angle. Different average brightness may come from framing rather than a tonal adjustment."
            next_step = "Check the horizon and frame edges before making further tonal changes."
            watch = "Keep important subjects clear of unintended edge cuts. Stop when the framing serves your goal."
        elif p["rgb_curve"][0][1] > .08:
            reason = f"The curve maps black to {p['rgb_curve'][0][1]:.2f}. Raised blacks can create a faded or gray appearance."
            next_step = "If the faded look conflicts with your goal, compare with a slightly lower black endpoint in RGB curve."
            watch = "Watch separation in dark areas. Stop before useful shadow differences disappear."
        elif p.get("color_grading") != before.get("color_grading"):
            cg = p["color_grading"]
            bcg = before["color_grading"]
            zones = [zone for zone in ("shadows", "midtones", "highlights") if cg[zone] != bcg[zone]]
            parts = [
                f"{zone.title()}: Hue {cg[zone]['hue']:g}°, Sat {cg[zone]['saturation']:.2f}, Lum {cg[zone]['luminance']:+.2f}"
                for zone in zones
            ]
            if cg["blend"] != bcg["blend"] or cg["balance"] != bcg["balance"] or not parts:
                parts.append(f"Blend {cg['blend']:.2f}, Balance {cg['balance']:+.2f}")
            change += " Color Grading — " + "; ".join(parts) + "."
            reason = (
                "Color Grading tints shadows, midtones, and highlights using a display-space luma proxy "
                "before Color Mixer and the RGB curve; pure tints preserve the display luma proxy before clipping, "
                "while luminance shifts adjust encoded SDR values."
            )
            next_step = "Compare Color Grading against its neutral settings before making another color shift."
            watch = "Watch neutral grays, skin, and tonal transitions for unwanted color casts or harsh rolloff. Stop when the mood fits your goal."
        elif p["color_mixer"] != before["color_mixer"]:
            colors = [color for color in p["color_mixer"] if p["color_mixer"][color] != before["color_mixer"][color]]
            description = "; ".join(f"{color.title()}: Hue {p['color_mixer'][color]['hue']:+g}°, "
                f"Saturation {p['color_mixer'][color]['saturation']:+g}, Lightness {p['color_mixer'][color]['lightness']:+g}"
                for color in colors)
            change += " Color Mixer — " + description
            reason = "HSL adjustments affect matching colors throughout the photo, blended with neighboring hues; near-neutral colors are protected. Lightness changes HSL lightness, not exposure."
            next_step = f"Compare Color Mixer {colors[0].title()} with its neutral settings before making another change."
            watch = "Watch neighboring colors and texture for unnatural transitions or washed-out color. Stop if the intended colors already fit your goal."
        elif any(p[name] != before[name] for name in ("whites", "blacks")):
            reason = f"Whites {p['whites']:+.2f} and Blacks {p['blacks']:+.2f} reshape the encoded RGB endpoints, not RAW detail recovery."
            control = "Whites" if p["whites"] != before["whites"] else "Blacks"
            next_step = f"Compare {control} with its neutral value 0.00. Keep the adjustment only if the endpoint separation better serves your goal; otherwise stop and reconsider."
            watch = "Keep useful separation in dark textures and bright surfaces. Stop if either end loses detail."
        elif p["vibrance"] != before["vibrance"]:
            reason = f"Vibrance {p['vibrance']:+.2f} changes muted colors more than already saturated colors, without skin detection."
            next_step = "Compare Vibrance with its neutral value 0.00 before adding more color."
            watch = "Watch familiar colors and neutral surfaces. Stop when color serves your goal rather than becoming uniformly stronger."
        elif p["saturation"] < .8:
            reason = f"Saturation is {p['saturation']:.2f}; values below 1 move RGB toward weighted gray. This may explain a gray appearance."
            next_step = "If that muted look conflicts with your goal, try bringing saturation slightly toward 1.00."
            watch = "Watch familiar colors and skin, if present. Stop when they feel believable, even below 1.00."
        elif p["contrast"] < -.3:
            reason = f"Contrast is {p['contrast']:+.2f}; a lower slope pulls encoded tones toward mid-gray."
            next_step = "Compare with slightly less negative contrast (-0.10) if the result feels too flat."
            watch = "Watch separation between adjacent tones. Stop before shadows or highlights lose useful detail."
        elif p["saturation"] > 1.5:
            reason = f"Saturation is {p['saturation']:.2f}; it increases separation from gray and can clip channels."
            proposals.append({"operation": "set", "parameter": "saturation", "value": 1.25})
            next_step = "Compare against a small saturation reduction toward 1.25; keep the stronger version only if it serves your goal."
            watch = "Watch highly colored areas for abrupt changes or lost texture. Stop when the color feels intentional."
        elif p["highlights"] or p["shadows"]:
            reason = (f"Highlights {p['highlights']:+.2f} and Shadows {p['shadows']:+.2f} apply luminance-weighted gains before display clipping. "
                      "They cannot restore information already lost during RAW decoding.")
            next_step = "Compare with Baseline and inspect the important bright and dark areas before changing another control."
            watch = "Watch bright texture and shadow separation. Stop when the result meets your goal; display-limit warnings alone do not mean a bad edit."
        elif p["exposure_ev"] > 1.5:
            reason = f"Exposure is {p['exposure_ev']:+.2f} EV, multiplying decoded linear RGB by {2 ** p['exposure_ev']:.2f}."
            next_step = "Inspect important bright areas at full resolution before increasing exposure_ev further."
            watch = "Stop adding exposure once important bright texture disappears. A small preview cannot establish RAW recoverability."
        else:
            reason = (f"Current exposure {p['exposure_ev']:+.2f} EV, contrast {p['contrast']:+.2f}, "
                      f"and saturation {p['saturation']:.2f} come directly from the editor state.")
            next_step = "Compare the subject and background against your goal before changing another control."
            watch = "Stop if the current separation already serves your intention."
            if "gray" in request.question.lower() or "flat" in request.question.lower():
                reason += " These rules do not identify a clear parameter cause for grayness."

        answer = ("This local demo uses parameter rules, not a language or vision model. "
                  "It cannot answer a custom question; the guidance below is a rule-based comparison.")
        if request.question:
            proposals = []
        evidence = ("Observed: brightness comparison uses only the two supplied reduced previews.",
                    "Inference: parameter rules cannot identify subjects or assess aesthetic quality.",
                    f"Preference / goal: {request.goal or 'No goal entered.'}",
                    f"Question: {request.question or 'Analyze the current edit.'}")
        from .coach_agent import AgentPlan, Finding, VerificationCheck
        plan = AgentPlan("recommend" if proposals else "inspect", "Compare the current edit with your goal.",
            (Finding("whole", observed, "This demo reads settings and preview measurements only.", "technical", "medium"),),
            (VerificationCheck("whole", "visual", "visual", "Keep the trial only if it serves your goal."),),
            "The local mock cannot recognize subjects or judge aesthetic quality.")
        return CoachFeedback(request.photo_id, request.session_id, request.state_version,
                             change, reason, next_step, watch, evidence,
                             answer=answer,
                             question=request.question, actions=resolve_actions(proposals, EditValues.from_dict(p), request.source_size),
                             agent_plan=plan)

    def _analyze_follow_up(self, request: CoachRequest) -> CoachFeedback:
        from .coach_agent import AgentPlan, Finding, VerificationCheck, measure_preview

        follow_up = validate_follow_up_context(request.follow_up, state_version=request.state_version)
        assert follow_up is not None
        prior = follow_up["prior_advice"]
        prior_v = prior["state_version"]
        deltas = follow_up["parameter_deltas"]
        delta_names = [str(item.get("parameter", "")) for item in deltas if isinstance(item, dict) and item.get("parameter")]
        delta_summary = ", ".join(delta_names[:5]) + ("…" if len(delta_names) > 5 else "")
        before_ev = EditValues.from_dict(request.before_parameters)
        curr_ev = EditValues.from_dict(request.parameters)
        aligned = (
            before_ev.composition == curr_ev.composition
            and before_ev.lens_corrections.distortion == curr_ev.lens_corrections.distortion
        )
        before_stats = measure_preview(request.before_preview)["whole"]
        curr_stats = measure_preview(request.current_preview)["whole"]
        luma_delta = curr_stats["mean_luminance"] - before_stats["mean_luminance"]
        ceil_delta = curr_stats["display_ceiling_fraction"] - before_stats["display_ceiling_fraction"]
        floor_delta = curr_stats["display_floor_fraction"] - before_stats["display_floor_fraction"]
        pixels_changed = (request.before_preview.shape != request.current_preview.shape
                          or not np.array_equal(request.before_preview, request.current_preview))
        undo_n = follow_up["undo_count"]
        redo_n = follow_up["redo_count"]
        reset_n = follow_up["reset_count"]
        zh = bool(re.search(r"[\u4e00-\u9fff]", request.question or "")) or (
            not (request.question or "").strip() and bool(re.search(r"[\u4e00-\u9fff]", request.goal or ""))
        )

        if follow_up["reverted_to_prior_parameters"]:
            decision = "stop"
            if zh:
                verdict = "中性（已撤回至上一轮建议前相同参数）。"
                answer = (
                    f"效果复盘（v{prior_v} → v{request.state_version}）：中性。后续共记录 {undo_n} 次撤销、"
                    f"{redo_n} 次重做、{reset_n} 次重置，当前编辑参数已回到上一轮建议（v{prior_v}）时的状态。"
                    "这是客观参数对比；是否需要重新尝试其他方向仍由你的目标决定。"
                )
                reason = "当前参数与上一轮建议时一致，预览像素未产生净偏移；本规则评估不自动追加调整。"
                next_step = "1. 保持当前状态停下，或明确你想继续改善的具体画面区域后再发起新分析。"
                watch = "对比原片基准与当前画面；如果已满足创作目标即可停止。"
            else:
                verdict = "Neutral (reverted to the same parameters as prior advice)."
                answer = (
                    f"Verdict (v{prior_v} -> v{request.state_version}): Neutral — parameters returned to the same values "
                    f"as your v{prior_v} advice after {undo_n} undo, {redo_n} redo, and {reset_n} reset step(s). "
                    "This is an objective parameter comparison; whether to try a different adjustment remains your preference."
                )
                reason = "No net parameter change remains relative to the prior advice state, so stopping without extra edits avoids oscillation."
                next_step = "1. Stop here if the v0/prior state already fits your goal, or specify a new target area before adjusting again."
                watch = "Compare Edited vs Baseline; stop if separation and mood already match your intention."
        elif not deltas and abs(luma_delta) < 0.002:
            decision = "inspect"
            if zh:
                verdict = "中性（未检测到相对上一轮建议的净参数变化）。"
                answer = (
                    f"效果复盘（v{prior_v} → v{request.state_version}）：中性。未检测到相对 v{prior_v} 的参数变化，"
                    f"缩略预览平均亮度变化为 {luma_delta:+.4f}。本地规则无法判断主观审美是否满意。"
                )
                reason = "前后参数快照相同且像素均值几乎不变。"
                next_step = "1. 先执行一项小幅度可撤销调整，或查看 100% 细节后再重新复盘。"
                watch = "确认确实调整了目标控件后再对比变化。"
            else:
                verdict = "Neutral (no net parameter change since prior advice)."
                answer = (
                    f"Verdict (v{prior_v} -> v{request.state_version}): Neutral — no net parameter change since v{prior_v} "
                    f"(mean luminance delta {luma_delta:+.4f}). Local rules cannot judge aesthetic satisfaction."
                )
                reason = "Current parameters match the prior advice snapshot with negligible preview difference."
                next_step = "1. Make one small reversible adjustment or inspect at 100% before requesting another follow-up."
                watch = "Verify the intended control changed before evaluating before/after impact."
        elif not aligned:
            decision = "inspect"
            if zh:
                verdict = "结果混合 / 需视觉复核（裁剪或几何畸变已改变，无法逐区域数值比较）。"
                answer = (
                    f"效果复盘（v{prior_v} → v{request.state_version}）：构图裁剪或镜头畸变已改动"
                    f"（改动项：{delta_summary or '构图/几何'}），前后像素区域不再一一对应。"
                    "请在主观上检查构图与主体边缘，客观统计不能单独判定几何修改是否变好。"
                )
                reason = "构图/畸变改变会移动取景范围，平均亮度差异可能源于裁掉了天空或暗部。"
                next_step = "1. 在当前取景下用原片基准或快照 A/B 对比，确认主体和地平线后再做进一步调色。"
                watch = "检查四边是否误切主体；若构图已符合目标即可停止。"
            else:
                verdict = "Mixed / visual check required (crop or distortion changed alignment)."
                answer = (
                    f"Verdict (v{prior_v} -> v{request.state_version}): Mixed / visual review needed — composition or "
                    f"distortion changed ({delta_summary or 'geometry'}), so regional measurements no longer align. "
                    "Judge framing visually; objective luminance deltas cannot evaluate framing quality."
                )
                reason = "Geometry changes shift what sits inside the frame, so average preview luminance reflects different scene content."
                next_step = "1. Visually compare with Baseline or a saved Snapshot at the same zoom before making further tonal edits."
                watch = "Check frame edges and horizon alignment; stop once framing supports your goal."
        elif ceil_delta > 0.02 or floor_delta > 0.02:
            decision = "stop"
            if zh:
                verdict = "结果混合 / 可能过度（调整带来了明显的变化，但高光溢出或黑位死黑比例上升超过 2%）。"
                answer = (
                    f"效果复盘（v{prior_v} → v{request.state_version}）：结果混合 / 存在过度调整风险。"
                    f"修改参数（{delta_summary}）使平均亮度变化 {luma_delta:+.4f}，但显示高光上限变化 {ceil_delta:+.2%}、"
                    f"黑位下限变化 {floor_delta:+.2%}。上述为客观像素统计；若该高反差并非你的主观偏好，建议回调。"
                )
                reason = "局部或全局调整幅度偏大导致 SDR 端点裁剪比例上升，缩略预览不能证明 RAW 高光可恢复性。"
                next_step = "1. 用 Ctrl+Z 撤销或保存快照对比，适当减小本次调整幅度后检查高光与暗部纹理。"
                watch = "观察最亮纹理与最暗阴影细节；若纹理丢失且不符合目标，立即回调并停止加量。"
            else:
                verdict = "Mixed / possible overshoot (display ceiling or floor clipping rose by > 2%)."
                answer = (
                    f"Verdict (v{prior_v} -> v{request.state_version}): Mixed / possible overshoot — adjustments to {delta_summary} "
                    f"shifted mean luminance by {luma_delta:+.4f}, while display ceiling changed {ceil_delta:+.2%} and floor "
                    f"changed {floor_delta:+.2%}. These are objective pixel measurements; dial back only if clipping conflicts with your subjective goal."
                )
                reason = "End-point clipping increased noticeably relative to the prior advice state, while reduced previews cannot establish RAW recoverability."
                next_step = "1. Toggle Undo (Ctrl+Z) or Baseline to inspect bright and dark textures; dial back the last step if important detail clipped."
                watch = "Inspect brightest highlights and deepest shadows at 100%; stop once texture separation meets your goal."
        elif deltas and (abs(luma_delta) >= 0.002 or pixels_changed):
            decision = "stop"
            if zh:
                verdict = "在可测指标上有帮助（已按可控方向改变画面，且未造成明显端点裁剪恶化）。"
                answer = (
                    f"效果复盘（v{prior_v} → v{request.state_version}）：客观测量方向有帮助。"
                    f"已调整参数（{delta_summary}；撤销 {undo_n} 次 / 重做 {redo_n} 次），"
                    f"全图线性亮度变化 {luma_delta:+.4f}，高光溢出 {ceil_delta:+.2%}、黑位 {floor_delta:+.2%}。"
                    "以上为客观像素变化；画面氛围与审美是否满足目标仍属于你的主观判断，当前可停下复核。"
                )
                reason = "调整确实改变了画面影调/色彩分布且未导致显著 SDR 端点挤压，继续叠加同类修改易出现过调。"
                next_step = "1. 对比原片基准或上一版本快照；若当前观感已符合编辑目标即可停止，不建议无依据继续叠加。"
                watch = "检查主体质感、肤色或中性灰是否自然；如已达到预期就到此为止。"
            else:
                verdict = "Helped on measured preview metrics without major endpoint clipping."
                answer = (
                    f"Verdict (v{prior_v} -> v{request.state_version}): Helped objectively — your adjustments ({delta_summary}; "
                    f"{undo_n} undo, {redo_n} redo) changed preview pixels with mean luminance delta {luma_delta:+.4f}, "
                    f"ceiling delta {ceil_delta:+.2%}, and floor delta {floor_delta:+.2%}. "
                    "Objective measurements confirm a clean directional shift; whether the look serves your creative preference is your call."
                )
                reason = "The committed parameter changes produced measurable pixel change without spiking SDR ceiling/floor clipping."
                next_step = "1. Compare Edited against Baseline or Undo; stop here if the photo now matches your editing goal."
                watch = "Check subject texture and neutral tones for unintended casts; stop rather than stacking redundant edits."
        else:
            decision = "inspect"
            if zh:
                verdict = "中性（参数已改动，但缩小预览的整体亮度变化微弱）。"
                answer = (
                    f"效果复盘（v{prior_v} → v{request.state_version}）：在 768px 缩小预览上变化微弱"
                    f"（参数：{delta_summary}；平均亮度变化 {luma_delta:+.4f}）。精细局部、污点修复或锐化降噪需在 100% 视图确认。"
                )
                reason = "微小局部修改、细节或温和色相调整在整图均值上几乎不改变平均亮度。"
                next_step = "1. 放大到 100% 直接查看目标区域，用 Undo 前后对比实际像素效果。"
                watch = "在 100% 视图核验边缘、噪点与过渡；如符合目标即可停止。"
            else:
                verdict = "Neutral on reduced preview (parameter changed, but whole-frame shift is subtle)."
                answer = (
                    f"Verdict (v{prior_v} -> v{request.state_version}): Neutral at reduced preview scale — parameters changed "
                    f"({delta_summary}), with whole-frame luminance delta {luma_delta:+.4f}. Fine detail, spot repairs, or small local masks "
                    "must be judged visually at 100% rather than by global preview averages."
                )
                reason = "Small local, detail, or hue adjustments barely alter whole-frame mean luminance in a reduced preview."
                next_step = "1. Zoom to 100% on the edited area and toggle Undo or Baseline to judge the actual change."
                watch = "Check fine texture, halos, or seams at 100%; stop if the localized change already works."

        main_change = (
            f"Follow-up v{prior_v} -> v{request.state_version}: {verdict} "
            f"Deltas: {delta_summary or 'none'} (undo {undo_n}, redo {redo_n}, reset {reset_n})."
        )
        evidence = (
            f"Observed: mean luminance delta {luma_delta:+.4f}, ceiling delta {ceil_delta:+.2%}, floor delta {floor_delta:+.2%} (v{prior_v} -> v{request.state_version}).",
            "Inference: local rules compare parameter snapshots and reduced-preview measurements; they cannot perceive subjects or aesthetic quality.",
            f"Preference / goal: {request.goal or 'No goal entered.'}",
            f"Prior advice (v{prior_v}): {prior['objective'] or prior['next_step'] or 'Review'}",
        )
        plan = AgentPlan(
            decision,
            f"Evaluate whether edits since v{prior_v} helped relative to the user's goal.",
            (Finding("whole", f"Mean luminance delta {luma_delta:+.4f}, ceiling {ceil_delta:+.2%}, floor {floor_delta:+.2%}.",
                     verdict, "technical", "medium"),),
            (VerificationCheck("whole", "visual", "visual", "Compare prior state vs current edit visually before making further changes."),),
            "Local mock cannot recognize subjects or judge aesthetic preference.",
        )
        return CoachFeedback(
            request.photo_id,
            request.session_id,
            request.state_version,
            main_change,
            reason,
            next_step,
            watch,
            evidence,
            answer=answer,
            question=request.question,
            actions=(),
            agent_plan=plan,
            follow_up_context=copy.deepcopy(follow_up),
        )
