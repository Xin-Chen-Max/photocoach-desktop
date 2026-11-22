"""Bounded Observe -> Diagnose -> Trial -> Verify workflow for photo coaching.

The model proposes one transaction. This module owns the evidence tools and
verification policy; it never owns the editor, credentials, transport or files.
"""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .gradients import RadialGradient
from .state import EditValues


REGIONS = ("whole", "top_left", "top_center", "top_right", "middle_left",
           "center", "middle_right", "bottom_left", "bottom_center", "bottom_right")
METRICS = ("mean_luminance", "display_ceiling_fraction", "display_floor_fraction", "visual")


def region_label(region, zh=False):
    names = ("全图", "左上", "上方中央", "右上", "左侧中央", "中央", "右侧中央", "左下", "下方中央", "右下")
    return names[REGIONS.index(region)] if zh else region.replace("_", " ")


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _enum(values):
    return {"type": "string", "enum": list(values)}


PLAN_SCHEMA = _object({
    "version": {"type": "integer", "enum": [1]},
    "decision": _enum(("recommend", "stop", "inspect", "clarify")),
    "objective": {"type": "string"},
    "findings": {"type": "array", "items": _object({
        "region": _enum(REGIONS), "observation": {"type": "string"},
        "interpretation": {"type": "string"},
        "category": _enum(("technical", "preference")),
        "confidence": _enum(("low", "medium", "high")),
    })},
    "checks": {"type": "array", "items": _object({
        "region": _enum(REGIONS), "metric": _enum(METRICS),
        "direction": _enum(("increase", "decrease", "preserve", "visual")),
        "expectation": {"type": "string"},
    })},
    "uncertainty": {"type": "string"},
})


@dataclass(frozen=True)
class Finding:
    region: str
    observation: str
    interpretation: str
    category: str
    confidence: str


@dataclass(frozen=True)
class VerificationCheck:
    region: str
    metric: str
    direction: str
    expectation: str


@dataclass(frozen=True)
class AgentPlan:
    decision: str
    objective: str
    findings: tuple[Finding, ...]
    checks: tuple[VerificationCheck, ...]
    uncertainty: str


def _bounded_text(value, limit=400):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Invalid workflow text.")
    return value.strip()


def parse_plan(document) -> AgentPlan:
    """Strictly parse the versioned model contract; never infer a plan from prose."""
    if not isinstance(document, dict) or set(document) != set(PLAN_SCHEMA["required"]):
        raise ValueError("Incomplete coaching workflow.")
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("Unsupported coaching workflow version.")
    if document["decision"] not in ("recommend", "stop", "inspect", "clarify"):
        raise ValueError("Invalid workflow decision.")
    findings, checks = [], []
    for name, target, cls in (("findings", findings, Finding), ("checks", checks, VerificationCheck)):
        items = document[name]
        if not isinstance(items, list) or not 1 <= len(items) <= 3:
            raise ValueError("A workflow needs one to three findings and checks.")
        for item in items:
            properties = PLAN_SCHEMA["properties"][name]["items"]["properties"]
            if not isinstance(item, dict) or set(item) != set(properties):
                raise ValueError("Incomplete finding or verification check.")
            data = {}
            for key, schema in properties.items():
                data[key] = _bounded_text(item[key])
                if "enum" in schema and data[key] not in schema["enum"]:
                    raise ValueError("Unknown workflow region, evidence type or check.")
            if name == "checks" and ((data["metric"] == "visual") != (data["direction"] == "visual")):
                raise ValueError("Visual checks cannot claim a numerical result.")
            target.append(cls(**data))
    return AgentPlan(document["decision"], _bounded_text(document["objective"]),
                     tuple(findings), tuple(checks), _bounded_text(document["uncertainty"]))


def _pixels(pixels):
    if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 3
            or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1 or max(pixels.shape[:2]) > 768):
        raise ValueError("Evidence requires a bounded RGB SDR preview.")
    return pixels


def _region(pixels, name):
    if name == "whole":
        return pixels
    index = REGIONS.index(name) - 1
    y, x = divmod(index, 3)
    h, w = pixels.shape[:2]
    return pixels[min(h-1, y*h//3):max(min(h, (y+1)*h//3), min(h-1, y*h//3)+1),
                  min(w-1, x*w//3):max(min(w, (x+1)*w//3), min(w-1, x*w//3)+1)]


def measure_preview(pixels) -> dict:
    """Measurements, not diagnoses: a bright pixel is not necessarily a defect."""
    _pixels(pixels)
    result = {}
    for name in REGIONS:
        patch = _region(pixels, name)
        rgb = patch.astype(np.float32) / 255
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        luminance = linear @ np.array([.2126, .7152, .0722], dtype=np.float32)
        result[name] = {
            "mean_luminance": float(luminance.mean()),
            "luminance_p05_p50_p95": [float(v) for v in np.percentile(luminance, [5, 50, 95])],
            "display_ceiling_fraction": float(np.any(patch == 255, axis=2).mean()),
            "display_floor_fraction": float(np.all(patch == 0, axis=2).mean()),
            "pixels": int(luminance.size),
        }
    return result


def evidence_context(request, include_before=True) -> dict:
    current = measure_preview(request.current_preview)
    context = {
        "version": 1, "source": "application measurements of supplied reduced SDR sRGB previews",
        "region_coordinates": "whole or a fixed 3x3 grid, top to bottom, left to right; not subject masks",
        "current": current,
        "limitations": ["Display endpoints do not establish sensor saturation or RAW recoverability.",
                        "Brightness statistics do not determine correct exposure or aesthetic quality.",
                        "A reduced preview cannot verify noise, sharpening, focus or fine texture."],
    }
    if include_before:
        context["before"] = measure_preview(request.before_preview)
        context["geometry_aligned"] = _aligned(EditValues.from_dict(request.before_parameters),
                                                EditValues.from_dict(request.parameters))
    return context


def _aligned(before, after):
    return (before.composition == after.composition and
            before.lens_corrections.distortion == after.lens_corrections.distortion)


def action_policy(plan: AgentPlan, actions, goal="", question="") -> str:
    if not actions:
        return ""
    if plan.decision != "recommend":
        return "The workflow requested stopping, inspection or clarification. No edit is enabled."
    if not any(f.confidence != "low" for f in plan.findings):
        return "Evidence is uncertain. Inspect the photo or clarify the goal before proposing an edit."
    if not (goal.strip() or question.strip()) and all(f.category == "preference" for f in plan.findings):
        return "Choose an editing goal before applying a preference-only recommendation."
    return ""


@dataclass(frozen=True)
class CheckResult:
    check: VerificationCheck
    before: float | None
    after: float | None
    verdict: str


@dataclass(frozen=True)
class AgentReport:
    status: str
    can_apply: bool = False
    checks: tuple[CheckResult, ...] = ()
    notices: tuple[tuple[str, str], ...] = ()
    trace: tuple[str, ...] = ("observe", "diagnose", "recommend")
    current_preview: np.ndarray | None = field(default=None, repr=False, compare=False)
    trial_preview: np.ndarray | None = field(default=None, repr=False, compare=False)

    def lines(self, zh=False):
        status = {"review": ("Trial rendered · Review before applying", "试调已生成 · 应用前请比较"),
                  "blocked": ("Suggestion held · Checks need attention", "建议暂不应用 · 请检查原因"),
                  "no_edit": ("No edit proposed", "本轮不修改照片")}[self.status]
        lines = [status[1 if zh else 0]]
        for result in self.checks:
            check = result.check
            if result.before is None:
                values = ""
            elif "fraction" in check.metric:
                values = f" {result.before:.2%} → {result.after:.2%}"
            else:
                values = f" {result.before:.4f} → {result.after:.4f}"
            labels = {"observed": ("Expected change observed", "已观察到预期变化"),
                      "failed": ("Expected change not observed", "未观察到预期变化"),
                      "visual": ("Visual judgment required", "需要人工视觉判断"),
                      "unaligned": ("Geometry changed; regions cannot be compared", "几何已变化，区域不可直接比较")}
            lines.append(f"{check.expectation} · {labels[result.verdict][1 if zh else 0]}{values}")
        lines.extend(n[1 if zh else 0] for n in self.notices)
        return tuple(lines)


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError()


def evaluate_trial_candidate(request, plan: AgentPlan, actions, render: Callable[[EditValues], np.ndarray],
                             *, cancel=None, allow_detail: bool = False) -> tuple[AgentReport, dict]:
    """Render and measure one candidate transaction without mutating StateStore."""
    _cancelled(cancel)
    if not actions:
        report = AgentReport("no_edit", trace=("observe", "diagnose", plan.decision))
        return report, {"can_apply": False, "failed_checks": [], "warnings": [], "reason": "no_edit"}
    notice = action_policy(plan, actions, request.goal, getattr(request, "question", ""))
    if notice or len(actions) != 1:
        msg_en = notice or "Only one trial transaction is supported."
        report = AgentReport("blocked", notices=((msg_en, "本轮需补充依据或缩减为一组可撤销的试调。"),))
        return report, {"can_apply": False, "failed_checks": ["policy"], "warnings": [msg_en], "reason": msg_en}
    baseline = EditValues.from_dict(request.parameters)
    action, = actions
    changes = action.changes or ((action.parameter, action.new_value),)
    candidate = replace(baseline, **dict(changes))
    detail_changed = candidate.detail != baseline.detail
    if detail_changed and not allow_detail:
        msg_en = "Inspect Detail at 100%. These reduced previews cannot verify sharpening or noise reduction."
        report = AgentReport("blocked", notices=((
            msg_en,
            "请在 100% 视图检查细节。缩略预览不能验证锐化或降噪效果。"),))
        return report, {"can_apply": False, "failed_checks": ["detail_requires_100pct"],
                        "warnings": [msg_en], "reason": msg_en}
    trace = ("observe", "diagnose", "recommend", "render_trial", "verify", "await_user")
    try:
        before = np.array(_pixels(render(baseline)), copy=True)
        _cancelled(cancel)
        after = np.array(_pixels(render(candidate)), copy=True)
        _cancelled(cancel)
    except CancelledError:
        raise
    except Exception:
        msg_en = "The trial could not be rendered. No edit was applied; inspect manually or analyze again."
        report = AgentReport("blocked", trace=trace[:4], notices=((
            msg_en,
            "无法生成试调。本轮未修改照片；请手动检查或重新分析。"),))
        return report, {"can_apply": False, "failed_checks": ["render_error"], "warnings": [msg_en], "reason": msg_en}
    for pixels in (before, after):
        pixels.setflags(write=False)
    same_geometry = before.shape == after.shape and _aligned(baseline, candidate)
    left, right = measure_preview(before), measure_preview(after)
    results = []
    failed_checks = []
    for check in plan.checks:
        if check.metric == "visual":
            results.append(CheckResult(check, None, None, "visual"))
        elif not same_geometry:
            results.append(CheckResult(check, None, None, "unaligned"))
            failed_checks.append(f"{check.region}.{check.metric}:unaligned")
        else:
            a, b = left[check.region][check.metric], right[check.region][check.metric]
            tolerance = 1e-5 if "fraction" in check.metric else .0005
            matched = (b > a + tolerance if check.direction == "increase" else
                       b < a - tolerance if check.direction == "decrease" else abs(b - a) <= .005)
            verdict = "observed" if matched else "failed"
            if not matched:
                failed_checks.append(f"{check.region}.{check.metric}:expected_{check.direction}({a:.4f}->{b:.4f})")
            results.append(CheckResult(check, a, b, verdict))
    notices = [("Measurements check the stated change, not aesthetic quality. Compare the trial and use Undo after applying.",
                "统计只验证所述变化，不代表审美改善。请比较试调，应用后可撤销。")]
    if detail_changed and allow_detail:
        notices.append(("Detail change tested after 100% crop inspection; confirm texture and noise in the native crop.",
                        "已在 100% 原生裁剪核验后生成细节试调；请结合细节缩略图复核纹理与噪点。"))
    failed = any(r.verdict in ("failed", "unaligned") for r in results)
    warnings: list[str] = []
    if before.shape == after.shape and np.array_equal(before, after):
        failed = True
        msg_en = "The trial has no visible pixel change at this preview resolution."
        notices.append((msg_en, "试调在当前预览分辨率没有像素变化。"))
        failed_checks.append("no_visible_change")
    for metric, en, zh in (("display_ceiling_fraction", "More pixels reach the display ceiling; inspect bright texture.",
                            "更多像素达到显示上限，请检查亮部纹理。"),
                           ("display_floor_fraction", "More pixels reach display black; inspect shadow texture.",
                            "更多像素达到显示黑位，请检查暗部纹理。")):
        if same_geometry and right["whole"][metric] > left["whole"][metric] + .01:
            notices.append((en, zh))
            warnings.append(en)
    existing_ids = {g.id for g in baseline.local_gradients}
    new_masks = tuple(g for g in candidate.local_gradients if g.id not in existing_ids)
    key_radials = [g for g in new_masks if isinstance(g, RadialGradient) and not g.inverted and g.exposure_ev >= 0.25]
    if key_radials:
        has_broad_key = any(
            abs(g.end[0] - g.start[0]) >= 0.20 and abs(g.end[1] - g.start[1]) >= 0.16
            for g in key_radials
        )
        if not has_broad_key:
            en = (
                "Radial spotlight mask is very small (< 0.20x0.16) and risks looking like a bright dot; "
                "widen start/end around the subject crest or use build_lighting_rig."
            )
            zh = "径向聚光蒙版过小（可能出现孤立亮斑），请扩大覆盖主体顶部区域或使用 build_lighting_rig 生成双蒙版。"
            notices.append((en, zh))
            warnings.append(en)
        if any(g.feather < 0.45 for g in key_radials):
            en = "Radial spotlight feather is below 0.45 and may create a hard halo; increase feather to >= 0.65."
            zh = "径向聚光蒙版羽化值低于 0.45，容易产生生硬边缘；建议提高至 0.65 以上。"
            notices.append((en, zh))
            warnings.append(en)
        if all(g.warmth > 0.15 and g.shadows <= 0.05 for g in key_radials) and (
            left["whole"]["mean_luminance"] < 0.18 or left["center"]["mean_luminance"] < 0.15
        ):
            en = (
                "Warm spotlight on a dim scene has no local shadow lift; add local shadows (+0.20..+0.38) "
                "so dark foreground subjects illuminate cleanly without blowing out bright ground."
            )
            zh = "偏暗画面中的暖色聚光灯缺少局部阴影提亮（shadows），建议补充 +0.20..+0.38 避免仅亮起浅色地面。"
            notices.append((en, zh))
            warnings.append(en)
    whole_luma_delta = (right["whole"]["mean_luminance"] - left["whole"]["mean_luminance"]) if same_geometry else None
    ceiling_delta = (right["whole"]["display_ceiling_fraction"] - left["whole"]["display_ceiling_fraction"]) if same_geometry else None
    floor_delta = (right["whole"]["display_floor_fraction"] - left["whole"]["display_floor_fraction"]) if same_geometry else None
    report = AgentReport("blocked" if failed else "review", not failed,
                         tuple(results), tuple(notices), trace, before, after)
    metrics_delta = {
        "can_apply": not failed,
        "same_geometry": same_geometry,
        "whole_luminance_delta": round(whole_luma_delta, 5) if whole_luma_delta is not None else None,
        "ceiling_delta": round(ceiling_delta, 5) if ceiling_delta is not None else None,
        "floor_delta": round(floor_delta, 5) if floor_delta is not None else None,
        "failed_checks": failed_checks,
        "warnings": warnings,
    }
    return report, metrics_delta


def build_reflexion_prompt_summary(report: AgentReport, metrics_delta: dict) -> dict:
    """Summarize trial check outcomes and visual self-critique prompts for the agent loop."""
    reasons = list(metrics_delta.get("failed_checks", [])) + list(metrics_delta.get("warnings", []))
    needs_repair = (not report.can_apply) or bool(metrics_delta.get("warnings"))
    if not report.can_apply:
        instruction = (
            "Trial checks FAILED (" + (", ".join(reasons) or "policy check failed") + "). "
            "Inspect the rendered trial image for artifacts, reduce magnitude or adjust checks/parameters, "
            "and call propose_and_preview_trial again or switch decision to stop/inspect if unrecoverable."
        )
    elif metrics_delta.get("warnings"):
        instruction = (
            "Numeric checks passed WITH endpoint clipping warnings (" + ", ".join(metrics_delta["warnings"]) + "). "
            "Visually inspect the rendered trial preview carefully: if highlight/shadow detail is lost or halos appear, "
            "dial back the adjustment and run one repaired propose_and_preview_trial."
        )
    else:
        instruction = (
            "Numeric checks passed cleanly. Inspect the rendered trial preview visually for halos, color casts, or "
            "unnatural transitions; if the result meets the user's goal cleanly, deliver your final structured feedback."
        )
    return {"needs_repair": needs_repair, "reasons": reasons, "instruction": instruction}


def check_trial(request, feedback, render: Callable[[EditValues], np.ndarray], *, cancel=None):
    """Run one bounded dry run, with no editor mutations or additional model calls.

    The renderer is owned by the existing serial image worker. Current and trial
    use its SAME formal path. Model judgments are never reported as measured facts.
    """
    _cancelled(cancel)
    plan = feedback.agent_plan
    if plan is None:
        return feedback  # Legacy replies stay readable under their existing contract.
    if (feedback.photo_id, feedback.session_id, feedback.state_version) != (
            request.photo_id, request.session_id, request.state_version):
        raise ValueError("Workflow belongs to another photo or edit.")
    report, _ = evaluate_trial_candidate(request, plan, feedback.actions, render, cancel=cancel, allow_detail=False)
    return replace(feedback, agent_report=report)
