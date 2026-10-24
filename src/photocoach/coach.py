"""Bounded coach context and local mock; provider transport lives separately."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Protocol
from uuid import uuid4

import numpy as np

from .state import EDIT_PIPELINE_VERSION, EditState


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


def build_request(current: EditState, before: EditState, events: list[dict],
                  before_preview: np.ndarray, current_preview: np.ndarray, question="") -> CoachRequest:
    if current.photo_id != before.photo_id or current.session_id != before.session_id:
        raise ValueError("Coach previews must belong to the same photo session.")
    if not isinstance(question, str) or len(question) > 2000:
        raise ValueError("The coach question must contain at most 2000 characters.")
    for pixels in (before_preview, current_preview):
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or max(pixels.shape[:2]) > 768:
            raise ValueError("Coach inputs require RGB previews no larger than 768 pixels.")
    # Crop changes can produce different aspect ratios. Each image belongs to
    # its own explicit parameter snapshot; never stretch one to match the other.
    operations = tuple({key: copy.deepcopy(event[key]) for key in
                        ("sequence", "type", "parameter", "before", "after", "state_version")}
                       for event in events if event["photo_id"] == current.photo_id and
                       event["session_id"] == current.session_id and event["type"] in
                       ("parameter", "reset", "undo", "redo") and event["phase"] != "update")[-12:]
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
                        before.values.to_dict(), operations, *images)


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

    def text(self) -> str:
        response = (f"Question: {self.question}\n\n" if self.question else "")
        if self.answer:
            response += f"Answer: {self.answer}\n\n"
        return (f"{self.source} · State v{self.state_version}\n\n" + response +
                f"Change: {self.main_change}\n\nWhy: {self.reason}\n\n"
                f"Next: {self.next_step_or_stop}\n\nWatch / stop: {self.observe_and_stop_when}\n\n"
                + "\n".join(self.evidence))


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
        p, before = request.parameters, request.before_parameters
        changed = [key for key in p if p[key] != before[key]]
        delta = float(request.current_preview.astype(np.float32).mean() - request.before_preview.astype(np.float32).mean()) / 255
        observed = "The supplied preview is brighter overall." if delta > .01 else (
            "The supplied preview is darker overall." if delta < -.01 else "Average preview brightness changed little.")
        change = f"Changed {', '.join(changed)}. {observed}" if changed else f"No committed edit to compare. {observed}"
        if p.get('local_gradients') != before.get('local_gradients'):
            items=p.get('local_gradients',[])
            change+=' Local gradients: '+('; '.join(f"{g['name']}: Exposure {g['exposure_ev']:+.2f} EV, Highlights {g['highlights']:+.2f}, Shadows {g['shadows']:+.2f}" for g in items) or 'none')
            reason='Manual gradient weights apply only by position, not subject detection. Local exposure adds in stops; overlapping tonal amounts combine with global settings and are limited.'
            next_step=f"Compare local gradient {items[-1]['name']} enabled and disabled before another adjustment." if items else 'Compare the current edit with Baseline before another adjustment.'
            watch='Inspect both sides of the transition. Stop if the light looks artificial, texture looks flat, or important bright areas lose separation.'
        elif p["composition"] != before["composition"]:
            reason = "Composition changed the included scene or its angle. Different average brightness may come from framing rather than a tonal adjustment."
            next_step = "Check the horizon and frame edges before making further tonal changes."
            watch = "Keep important subjects clear of unintended edge cuts. Stop when the framing serves your goal."
        elif p["rgb_curve"][0][1] > .08:
            reason = f"The curve maps black to {p['rgb_curve'][0][1]:.2f}. Raised blacks can create a faded or gray appearance."
            next_step = "If the faded look conflicts with your goal, compare with a slightly lower black endpoint."
            watch = "Watch separation in dark areas. Stop before useful shadow differences disappear."
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
            next_step = f"Compare {control} with its neutral value. Keep the adjustment only if the endpoint separation better serves your goal; otherwise stop and reconsider."
            watch = "Keep useful separation in dark textures and bright surfaces. Stop if either end loses detail."
        elif p["vibrance"] != before["vibrance"]:
            reason = f"Vibrance {p['vibrance']:+.2f} changes muted colors more than already saturated colors, without skin detection."
            next_step = "Compare Vibrance with its neutral value before adding more color."
            watch = "Watch familiar colors and neutral surfaces. Stop when color serves your goal rather than becoming uniformly stronger."
        elif p["saturation"] < .8:
            reason = f"Saturation is {p['saturation']:.2f}; values below 1 move RGB toward weighted gray. This may explain a gray appearance."
            next_step = "If that muted look conflicts with your goal, try bringing saturation slightly toward 1.00."
            watch = "Watch familiar colors and skin, if present. Stop when they feel believable, even below 1.00."
        elif p["contrast"] < -.3:
            reason = f"Contrast is {p['contrast']:+.2f}; a lower slope pulls encoded tones toward mid-gray."
            next_step = "Compare with slightly less negative contrast if the result feels too flat."
            watch = "Watch separation between adjacent tones. Stop before shadows or highlights lose useful detail."
        elif p["saturation"] > 1.5:
            reason = f"Saturation is {p['saturation']:.2f}; it increases separation from gray and can clip channels."
            next_step = "Compare against a small saturation reduction; keep the stronger version only if it serves your goal."
            watch = "Watch highly colored areas for abrupt changes or lost texture. Stop when the color feels intentional."
        elif p["highlights"] or p["shadows"]:
            reason = (f"Highlights {p['highlights']:+.2f} and Shadows {p['shadows']:+.2f} apply luminance-weighted gains before display clipping. "
                      "They cannot restore information already lost during RAW decoding.")
            next_step = "Compare with Baseline and inspect the important bright and dark areas before changing another control."
            watch = "Watch bright texture and shadow separation. Stop when the result meets your goal; display-limit warnings alone do not mean a bad edit."
        elif p["exposure_ev"] > 1.5:
            reason = f"Exposure is {p['exposure_ev']:+.2f} EV, multiplying decoded linear RGB by {2 ** p['exposure_ev']:.2f}."
            next_step = "Inspect important bright areas at full resolution before increasing exposure further."
            watch = "Stop adding exposure once important bright texture disappears. A small preview cannot establish RAW recoverability."
        else:
            reason = (f"Current exposure {p['exposure_ev']:+.2f} EV, contrast {p['contrast']:+.2f}, "
                      f"and saturation {p['saturation']:.2f} come directly from the editor state.")
            next_step = "Pause adjustments and compare the current result with your goal. No further change is justified by these rules alone."
            watch = "Check light, believable color, and important bright detail. Stop if the result already matches your intention."
            if any(word in request.question.lower() for word in ("gray", "grey", "flat")):
                reason += " These rules do not identify a clear parameter cause for grayness; lighting or other curve points may contribute."
                next_step = "Compare the current edit with the baseline and isolate one changed control at a time."
        evidence = ("Observed: brightness comparison uses only the two supplied reduced previews.",
                    "Inference: parameter rules cannot identify subjects or assess aesthetic quality.",
                    f"Preference / goal: {request.goal or 'No goal entered.'}",
                    f"Question: {request.question or 'Analyze the current edit.'}")
        return CoachFeedback(request.photo_id, request.session_id, request.state_version,
                             change, reason, next_step, watch, evidence,
                             answer="This local demo uses parameter rules, not a language or vision model. "
                                    "It cannot answer a custom question; the guidance below is a rule-based comparison.",
                             question=request.question)
