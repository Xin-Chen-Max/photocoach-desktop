"""Deterministic state-graph runtime, RunBudget, checkpoints, and replay for PhotoCoach Agent."""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
import json
import math
import re
import threading
import time
from typing import Any, Callable

import numpy as np

from .agent_tools import ToolCall, ToolContext, ToolRegistry, ToolResult, build_default_tool_registry
from .api_coach import _FINAL_SCHEMA_KEYS, AGENT_INSTRUCTIONS, ApiCoach, ApiError, TransportProgress, request_context
from .budget import BudgetExceeded, SessionBudget
from .coach import CoachFeedback, CoachRequest
from .coach_actions import analyze_scene_lighting
from .coach_agent import AgentReport, evaluate_trial_candidate
from .state import EditState, EditValues


@dataclass(frozen=True)
class RunBudget:
    max_llm_turns: int = 3
    max_tool_calls: int = 8
    max_trial_renders: int = 3
    max_wall_seconds: float = 120.0
    max_run_usd: Decimal | None = None

    def __post_init__(self):
        if type(self.max_llm_turns) is not int or not 1 <= self.max_llm_turns <= 8:
            raise ValueError("max_llm_turns must be between 1 and 8.")
        if type(self.max_tool_calls) is not int or not 0 <= self.max_tool_calls <= 24:
            raise ValueError("max_tool_calls must be between 0 and 24.")
        if type(self.max_trial_renders) is not int or not 0 <= self.max_trial_renders <= 8:
            raise ValueError("max_trial_renders must be between 0 and 8.")
        if self.max_wall_seconds <= 0 or self.max_wall_seconds > 1200:
            raise ValueError("max_wall_seconds must be in (0, 1200].")


@dataclass(frozen=True)
class TraceSpan:
    step_index: int
    node: str  # observe | reason | tool_call | trial_render | reflect | finalize | abort
    name: str
    started_ms: float
    duration_ms: float
    status: str  # ok | blocked | cancelled | error
    input_tokens: int | None = None
    output_tokens: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentCheckpoint:
    step_index: int
    node: str
    turn: int
    tool_calls_used: int
    trial_renders_used: int
    tokens_in: int
    tokens_out: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRunResult:
    feedback: CoachFeedback
    spans: tuple[TraceSpan, ...]
    checkpoints: tuple[AgentCheckpoint, ...]
    total_input_tokens: int | None
    total_output_tokens: int | None
    turns_used: int
    tool_calls_used: int
    trial_renders_used: int
    live_photo_response: bool = False
    aborted_reason: str = ""

    def to_record(self, request: CoachRequest | None = None) -> dict[str, Any]:
        """Export a sanitized, path-free, credential-free replay record."""
        return {
            "schema_version": 1,
            "state_version": self.feedback.state_version,
            "pipeline_version": request.pipeline_version if request else "",
            "turns_used": self.turns_used,
            "tool_calls_used": self.tool_calls_used,
            "trial_renders_used": self.trial_renders_used,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "aborted_reason": self.aborted_reason,
            "decision": self.feedback.agent_plan.decision if self.feedback.agent_plan else "legacy",
            "can_apply": bool(self.feedback.agent_report and self.feedback.agent_report.can_apply),
            "actions_count": len(self.feedback.actions),
            "spans": [span.to_dict() for span in self.spans],
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
        }


def replay_run_record(record: dict[str, Any]) -> dict[str, Any]:
    """Deterministically validate and summarize an exported AgentRunResult record offline."""
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Unsupported agent trace record schema.")
    spans = record.get("spans", [])
    checkpoints = record.get("checkpoints", [])
    if not isinstance(spans, list) or not isinstance(checkpoints, list) or not spans:
        raise ValueError("Trace record contains invalid spans or checkpoints.")
    expected_nodes = {"observe", "reason", "tool_call", "trial_render", "reflect", "finalize", "abort"}
    total_in = 0
    total_out = 0
    has_token_counts = False
    for idx, span in enumerate(spans):
        if span.get("step_index") != idx or span.get("node") not in expected_nodes:
            raise ValueError("Corrupted trace span sequence.")
        if span.get("input_tokens") is not None:
            total_in += int(span["input_tokens"])
            has_token_counts = True
        if span.get("output_tokens") is not None:
            total_out += int(span["output_tokens"])
            has_token_counts = True
    if has_token_counts and record.get("total_input_tokens") is not None and total_in != record["total_input_tokens"]:
        raise ValueError("Token ledger mismatch in trace replay.")
    return {
        "valid": True,
        "nodes": [s["node"] for s in spans],
        "turns_used": record["turns_used"],
        "tool_calls_used": record["tool_calls_used"],
        "trial_renders_used": record["trial_renders_used"],
        "total_duration_ms": round(sum(float(s.get("duration_ms", 0.0)) for s in spans), 2),
        "tokens": (total_in, total_out) if has_token_counts else None,
        "can_apply": bool(record.get("can_apply")),
    }


class AgentProgress(TransportProgress):
    """Thread-safe transport + multi-step agent loop progress snapshot."""

    def __init__(self, max_turns: int = 1):
        super().__init__()
        self._agent_lock = threading.Lock()
        self._turn = 0
        self._max_turns = max(1, int(max_turns))
        self._step_label = ""
        self._active_detail = ""
        self._spans: tuple[TraceSpan, ...] = ()

    def set_agent_step(self, turn: int, max_turns: int, label: str, *, detail: str = "") -> None:
        with self._agent_lock:
            self._turn = int(turn)
            self._max_turns = max(1, int(max_turns))
            self._step_label = str(label)
            self._active_detail = str(detail)

    def agent_snapshot(self) -> tuple[int, int, str]:
        with self._agent_lock:
            return self._turn, self._max_turns, self._step_label

    def record_spans(self, spans: list[TraceSpan] | tuple[TraceSpan, ...]) -> None:
        with self._agent_lock:
            self._spans = tuple(spans)

    def live_trace_snapshot(self) -> tuple[int, int, str, str, tuple[TraceSpan, ...]]:
        with self._agent_lock:
            return self._turn, self._max_turns, self._step_label, self._active_detail, self._spans


def _safe_float(value: Any, default: float) -> float:
    try:
        num = float(value)
        return num if math.isfinite(num) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _summarize_tool_call(call: ToolCall) -> tuple[dict[str, Any], str]:
    """Extract sanitized display metadata and an active-step hint for a ToolCall."""
    args = call.arguments if isinstance(call.arguments, dict) else {}
    name = call.name
    if name == "inspect_region_100pct":
        x = round(_safe_float(args.get("x_norm"), 0.5), 2)
        y = round(_safe_float(args.get("y_norm"), 0.5), 2)
        size = _safe_int(args.get("size_px"), 512)
        return {"x_norm": x, "y_norm": y, "size_px": size}, f"({x:.2f}, {y:.2f})"
    if name == "inspect_raw_headroom":
        x = round(_safe_float(args.get("x_norm"), 0.5), 2)
        y = round(_safe_float(args.get("y_norm"), 0.5), 2)
        patch = _safe_int(args.get("patch_px"), 64)
        return {"x_norm": x, "y_norm": y, "patch_px": patch}, f"({x:.2f}, {y:.2f})"
    if name == "analyze_color_zones":
        region = str(args.get("region", "whole") or "whole")[:24]
        return {"region": region}, region
    if name == "detect_scene_lighting_anchors":
        return {}, ""
    if name == "search_editing_playbook":
        query = " ".join(str(args.get("query", "")).split())[:48]
        return {"query": query}, query
    if name == "build_lighting_rig":
        recipe = str(args.get("recipe", "spotlight"))[:24]
        direction = str(args.get("light_direction", "auto"))[:16]
        strength = str(args.get("strength", "moderate"))[:16]
        return {"recipe": recipe, "direction": direction, "strength": strength}, f"{recipe}, {direction}"
    if name == "propose_and_preview_trial":
        raw_actions = args.get("actions", [])
        params: list[str] = []
        if isinstance(raw_actions, list):
            for item in raw_actions:
                if isinstance(item, dict):
                    param = item.get("parameter")
                    if isinstance(param, str) and param and param not in params:
                        params.append(param)
                        if len(params) >= 4:
                            break
        wf = args.get("workflow", {})
        decision = str(wf.get("decision", ""))[:24] if isinstance(wf, dict) else ""
        return {"params": params, "decision": decision}, ", ".join(params)
    return {}, ""


def _actions_signature(actions: tuple[Any, ...]) -> tuple[Any, ...]:
    if not actions:
        return ()
    action = actions[0]
    changes = action.changes or ((action.parameter, action.new_value),)
    return tuple((k, repr(v)) for k, v in sorted(changes, key=lambda kv: kv[0]))


class AgentRuntime:
    """Bounded Observe -> Reason -> Tool -> Render -> Reflect -> Finalize loop."""

    def __init__(
        self,
        adapter: ApiCoach,
        registry: ToolRegistry | None = None,
        *,
        run_budget: RunBudget = RunBudget(),
        session_budget: SessionBudget | None = None,
        playbook: Any | None = None,
        style_memory_snippet: str = "",
        working_memory_snippet: str = "",
        compactor: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.adapter = adapter
        self.registry = registry if registry is not None else build_default_tool_registry()
        self.run_budget = run_budget
        self.session_budget = session_budget
        self.playbook = playbook
        self.style_memory_snippet = style_memory_snippet.strip()
        self.working_memory_snippet = working_memory_snippet.strip()
        self.compactor = compactor

    def run(
        self,
        request: CoachRequest,
        *,
        state: EditState | None = None,
        include_before: bool = True,
        render_trial: Callable[[EditValues], np.ndarray] | None = None,
        render_crop_100pct: Callable[..., tuple[np.ndarray, dict[str, Any]]] | None = None,
        inspect_cfa: Callable[..., dict[str, Any]] | None = None,
        cancel: threading.Event | None = None,
        progress: TransportProgress | None = None,
        stale_check: Callable[[], bool] | None = None,
        initial_ticket: str | None = None,
    ) -> AgentRunResult:
        cancel = cancel if cancel is not None else threading.Event()
        t_start = time.monotonic()
        deadline = t_start + self.run_budget.max_wall_seconds

        def ensure_active() -> None:
            if cancel.is_set() or (stale_check is not None and stale_check()):
                raise ApiError("cancelled")
            if time.monotonic() >= deadline:
                raise ApiError("timeout", elapsed_seconds=time.monotonic() - t_start,
                               timeout_seconds=self.run_budget.max_wall_seconds)

        ensure_active()
        ctx = ToolContext(
            request=request,
            state=state,
            render_trial=render_trial,
            render_crop_100pct=render_crop_100pct,
            inspect_cfa=inspect_cfa,
            playbook=self.playbook,
            cancel=cancel,
            max_trials=self.run_budget.max_trial_renders,
        )

        spans: list[TraceSpan] = []
        checkpoints: list[AgentCheckpoint] = []
        total_in = 0
        total_out = 0
        has_in = False
        has_out = False
        turns_used = 0
        tool_calls_used = 0

        def add_span(node: str, name: str, started: float, status: str, *,
                     in_tok: int | None = None, out_tok: int | None = None,
                     detail: dict[str, Any] | None = None) -> None:
            now = time.monotonic()
            idx = len(spans)
            spans.append(TraceSpan(
                step_index=idx,
                node=node,
                name=name,
                started_ms=round((started - t_start) * 1000.0, 2),
                duration_ms=round(max(0.0, (now - started) * 1000.0), 2),
                status=status,
                input_tokens=in_tok,
                output_tokens=out_tok,
                detail=detail or {},
            ))
            checkpoints.append(AgentCheckpoint(
                step_index=idx,
                node=node,
                turn=turns_used,
                tool_calls_used=tool_calls_used,
                trial_renders_used=ctx.trial_count,
                tokens_in=total_in,
                tokens_out=total_out,
                status=status,
            ))
            if isinstance(progress, AgentProgress):
                progress.record_spans(spans)

        # 1. Observe phase: prepare bounded context + optional playbook, pre-anchors & memory hints
        obs_start = time.monotonic()
        ctx_dict = request_context(request, include_before=include_before)
        query_text = f"{request.question} {request.goal}".strip()
        if re.search(
            r"\b(?:spotlight|key[ _]?light|sunbeam|backlight|window[ _]?light|mountain[ _]?beam|lighting|light|mask|gradient)s?\b|"
            r"聚光|追光|补光|人造光|逆光|侧光|丁达尔|耶稣光|打光|光照|局部|蒙版|渐变",
            query_text,
            re.I,
        ):
            try:
                pre_anchors = analyze_scene_lighting(request.current_preview, request.question, request.goal)
                if pre_anchors:
                    ctx_dict["precomputed_lighting_anchors"] = {
                        "source_x": pre_anchors.get("source_x"),
                        "source_y": pre_anchors.get("source_y"),
                        "horizon_y": pre_anchors.get("horizon_y"),
                        "subject_candidates": (pre_anchors.get("subject_candidates") or ())[:3],
                    }
            except Exception:
                pass
        if self.playbook is not None:
            pb_query = query_text or "exposure tone color balance"
            hints = self.playbook.search(pb_query, top_k=2)
            if hints:
                ctx_dict["playbook_hints"] = hints
        if self.working_memory_snippet:
            ctx_dict["session_working_memory"] = self.working_memory_snippet
        if self.style_memory_snippet:
            ctx_dict["user_style_preferences"] = self.style_memory_snippet
        user_text = json.dumps(ctx_dict, ensure_ascii=False, allow_nan=False)
        images = [("Before this editing round", request.before_preview)] if include_before else []
        images.append(("Current effect", request.current_preview))
        messages: list[dict[str, Any]] = [{"role": "user", "text": user_text, "images": images}]
        add_span("observe", "context_ready", obs_start, "ok",
                 detail={"include_before": include_before, "tools": len(self.registry.specs())})

        final_feedback: CoachFeedback | None = None
        live_response = False
        aborted_reason = ""
        verified_trial_ready = False
        tools = self.registry.specs()

        for turn in range(1, self.run_budget.max_llm_turns + 1):
            ensure_active()
            force_final = (
                verified_trial_ready
                or (turn == self.run_budget.max_llm_turns)
                or (tool_calls_used >= self.run_budget.max_tool_calls)
            )
            if isinstance(progress, AgentProgress):
                progress.set_agent_step(
                    turn, self.run_budget.max_llm_turns,
                    "Finalizing structured guidance" if force_final else "Reasoning and selecting tools",
                )

            response = None
            turn_in: int | None = None
            turn_out: int | None = None
            turn_start = time.monotonic()
            for attempt in range(2):
                ensure_active()
                ticket = None
                if self.session_budget is not None:
                    if turn == 1 and attempt == 0 and initial_ticket is not None:
                        ticket = initial_ticket
                    else:
                        try:
                            ticket = self.session_budget.reserve()
                        except BudgetExceeded as exc:
                            if turn == 1 and attempt == 0:
                                raise ApiError("rate_limit") from exc
                            aborted_reason = str(exc)
                            add_span("abort", "budget_exhausted", time.monotonic(), "blocked", detail={"reason": aborted_reason})
                            response = None
                            break

                turn_start = time.monotonic()
                turns_used = turn
                turn_ok = False
                turn_in = None
                turn_out = None
                response = None
                retry_exc: ApiError | None = None
                try:
                    try:
                        call_messages = self.compactor(messages) if self.compactor is not None else messages
                        response = self.adapter.step(
                            request,
                            AGENT_INSTRUCTIONS,
                            call_messages,
                            () if force_final else tools,
                            cancel_event=cancel,
                            progress=progress,
                            force_final_schema=force_final,
                        )
                        turn_ok = True
                        turn_in = response.input_tokens
                        turn_out = response.output_tokens
                        live_response = live_response or response.live_photo_response
                    except ApiError as exc:
                        if (
                            attempt == 0
                            and turn > 1
                            and exc.code == "rate_limit"
                            and not cancel.is_set()
                            and (deadline - time.monotonic() > 8.0)
                            and (self.session_budget is None or self.session_budget.can_authorize_turns(1) is None)
                        ):
                            retry_exc = exc
                        else:
                            recoverable = ("timeout", "unavailable", "rate_limit", "invalid_response", "output_limit", "http_error")
                            if turn > 1 and ctx.best_trial is not None and exc.code in recoverable and not cancel.is_set():
                                aborted_reason = str(exc)
                                add_span("abort", f"turn_{turn}_{exc.code}", turn_start, "blocked", detail={"reason": aborted_reason})
                                response = None
                            else:
                                raise
                finally:
                    if ticket is not None:
                        self.session_budget.settle(ticket, turn_in, turn_out, success=turn_ok)
                if retry_exc is not None:
                    if isinstance(progress, AgentProgress):
                        progress.set_agent_step(turn, self.run_budget.max_llm_turns, "Rate-limited; waiting briefly to retry…")
                    sleep_end = min(deadline - 5.0, time.monotonic() + 2.5)
                    while time.monotonic() < sleep_end:
                        ensure_active()
                        time.sleep(0.2)
                    continue
                break
            if response is None:
                break

            if turn_in is not None:
                total_in += turn_in
                has_in = True
            if turn_out is not None:
                total_out += turn_out
                has_out = True

            if response.tool_calls and not force_final:
                add_span("reason", f"turn_{turn}_tool_plan", turn_start, "ok",
                         in_tok=turn_in, out_tok=turn_out,
                         detail={"calls": [c.name for c in response.tool_calls]})
                messages.append(response.raw_assistant_message)
                tool_results: list[ToolResult] = []
                for call in response.tool_calls:
                    ensure_active()
                    if tool_calls_used >= self.run_budget.max_tool_calls:
                        tool_results.append(ToolResult(
                            call.call_id, call.name, False,
                            {"error": f"Per-run tool budget ({self.run_budget.max_tool_calls}) reached."},
                            error="tool_limit",
                        ))
                        continue
                    tool_calls_used += 1
                    summary_detail, active_detail = _summarize_tool_call(call)
                    if isinstance(progress, AgentProgress):
                        progress.set_agent_step(turn, self.run_budget.max_llm_turns, call.name, detail=active_detail)
                    tool_start = time.monotonic()
                    try:
                        result = self.registry.execute(call, ctx)
                    except CancelledError:
                        raise ApiError("cancelled") from None
                    node = "trial_render" if call.name == "propose_and_preview_trial" else "tool_call"
                    status = "ok" if result.ok else "blocked"
                    detail_info: dict[str, Any] = {"ok": result.ok, **summary_detail}
                    if result.error:
                        detail_info["error"] = result.error
                    if call.name == "propose_and_preview_trial" and result.ok:
                        detail_info["can_apply"] = bool(result.payload.get("can_apply"))
                        detail_info["trial_index"] = result.payload.get("trial_index")
                    add_span(node, call.name, tool_start, status, detail=detail_info)
                    if call.name == "propose_and_preview_trial" and result.ok:
                        ref_start = time.monotonic()
                        reflex = result.payload.get("reflexion", {})
                        needs_repair = bool(reflex.get("needs_repair"))
                        add_span("reflect", f"trial_{result.payload.get('trial_index', ctx.trial_count)}_check",
                                 ref_start, "ok" if not needs_repair else "blocked",
                                 detail={"passed": not needs_repair,
                                         "needs_repair": needs_repair,
                                         "reasons": list(reflex.get("reasons", []))})
                        raw_acts = call.arguments.get("actions") if isinstance(call.arguments, dict) else None
                        has_new_mask = isinstance(raw_acts, list) and any(
                            isinstance(it, dict) and it.get("operation") == "create_mask" for it in raw_acts
                        )
                        verified_trial_ready = bool((not needs_repair) and (not has_new_mask) and result.payload.get("can_apply"))
                    tool_results.append(result)
                messages.append({"role": "tool_results", "results": tool_results})
                continue

            if response.final_feedback is not None:
                add_span("reason", f"turn_{turn}_final", turn_start, "ok",
                         in_tok=turn_in, out_tok=turn_out)
                final_feedback = response.final_feedback
                break

            # Unstructured intermediate text (e.g. model forgot schema); ask strictly on next turn.
            add_span("reason", f"turn_{turn}_prose", turn_start, "blocked",
                     in_tok=turn_in, out_tok=turn_out)
            messages.append(response.raw_assistant_message)
            messages.append({
                "role": "user",
                "text": (
                    "Either invoke the appropriate diagnostic/trial tool OR emit a single strict JSON object "
                    f"(no prose/markdown outside JSON) containing ALL required top-level keys: {_FINAL_SCHEMA_KEYS}."
                ),
            })

        if final_feedback is None and ctx.best_trial is not None:
            # Synthesize bounded fallback only if we ran out of turns after a verified trial.
            actions, plan, report = ctx.best_trial
            final_feedback = CoachFeedback(
                photo_id=request.photo_id,
                session_id=request.session_id,
                state_version=request.state_version,
                main_change="Verified reversible trial prepared after multi-step inspection.",
                reason=plan.objective,
                next_step_or_stop=f"Compare and apply the verified trial: {actions[0].label}" if actions else "No edit.",
                observe_and_stop_when=plan.uncertainty,
                evidence=tuple(f"Observed ({f.region}): {f.observation}" for f in plan.findings),
                source="AI · " + self.adapter.config.model,
                answer=plan.objective,
                question=request.question,
                actions=actions,
                action_notice=aborted_reason,
                agent_plan=plan,
                agent_report=report,
            )

        if final_feedback is None:
            raise ApiError("invalid_response")

        # Ensure final feedback carries a verified AgentReport matching its exact actions.
        fin_start = time.monotonic()
        if final_feedback.agent_plan is not None:
            if not final_feedback.actions:
                report = AgentReport("no_edit", trace=("observe", "diagnose", final_feedback.agent_plan.decision))
                final_feedback = replace(final_feedback, agent_report=report)
            else:
                matched_report: AgentReport | None = None
                sig = _actions_signature(final_feedback.actions)
                for candidate in (ctx.best_trial, ctx.last_trial):
                    if candidate is None:
                        continue
                    c_actions, c_plan, c_report = candidate
                    if (_actions_signature(c_actions) == sig
                            and c_plan.decision == final_feedback.agent_plan.decision
                            and c_plan.checks == final_feedback.agent_plan.checks):
                        matched_report = c_report
                        break
                if matched_report is None and render_trial is not None:
                    matched_report, _ = evaluate_trial_candidate(
                        request, final_feedback.agent_plan, final_feedback.actions, render_trial,
                        cancel=cancel, allow_detail=ctx.inspected_100pct,
                    )
                if matched_report is not None:
                    final_feedback = replace(final_feedback, agent_report=matched_report)

        add_span("finalize", "ready_for_user", fin_start, "ok",
                 detail={"can_apply": bool(final_feedback.agent_report and final_feedback.agent_report.can_apply)})
        final_feedback = replace(
            final_feedback,
            agent_trace=tuple(spans),
            inspected_crops=tuple(ctx.inspected_crops),
        )
        return AgentRunResult(
            feedback=final_feedback,
            spans=tuple(spans),
            checkpoints=tuple(checkpoints),
            total_input_tokens=total_in if has_in else None,
            total_output_tokens=total_out if has_out else None,
            turns_used=turns_used,
            tool_calls_used=tool_calls_used,
            trial_renders_used=ctx.trial_count,
            live_photo_response=live_response,
            aborted_reason=aborted_reason,
        )
