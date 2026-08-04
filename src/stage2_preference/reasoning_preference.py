"""Pure utilities shared by preference mining, validation, and tests."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable


ERROR_CATEGORIES = (
    "wrong_type",
    "trigger_drift",
    "argument_omission",
    "event_omission",
    "extra_frame",
)

PROFILE_WEIGHTS = {
    "e81": {"trigger": 0.40, "event": 0.35, "argument": 0.25},
    "g9": {"trigger": 0.15, "event": 0.50, "argument": 0.35},
}

COMPLETE_REASONING_RESPONSE = re.compile(
    r"\s*<thinking>.*?</thinking>\s*<final>.*?</final>\s*", re.DOTALL
)


def extract_tag(text: str, tag: str) -> str | None:
    pattern = re.compile(
        rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text or "")
    if match:
        return match.group(1).strip()
    start_match = re.search(rf"<\s*{re.escape(tag)}\s*>", text or "", re.IGNORECASE)
    if not start_match:
        return None
    return (text or "")[start_match.end() :].strip()


def extract_final_json(text: str) -> dict[str, Any] | None:
    final_text = extract_tag(text, "final")
    if final_text is None:
        return None
    start = final_text.find("{")
    end = final_text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(final_text[start : end + 1])
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def has_complete_reasoning_response(text: str) -> bool:
    return isinstance(text, str) and COMPLETE_REASONING_RESPONSE.fullmatch(text) is not None


def final_only_response(text: str) -> str | None:
    final_text = extract_tag(text, "final")
    if final_text is None:
        return None
    return f"<final>{final_text}</final>"


def norm_token(token: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", (token or "").lower())


def phrase_tokens(text: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", text or "")
    return [norm_token(token) for token in raw if norm_token(token)]


def parse_prompt_tokens(input_text: str) -> list[str]:
    match = re.search(
        r"Tokens:\n(.*?)(?:\n\nCandidate event types:|\n\nSchema cards:|\Z)",
        input_text or "",
        flags=re.DOTALL,
    )
    return [token for token in match.group(1).strip().split() if token] if match else []


def find_subsequences(haystack: list[str], needle: list[str]) -> list[tuple[int, int]]:
    if not haystack or not needle or len(needle) > len(haystack):
        return []
    width = len(needle)
    return [
        (index, index + width)
        for index in range(len(haystack) - width + 1)
        if haystack[index : index + width] == needle
    ]


def find_compact_spans(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    exact_phrase_tokens = [token.casefold() for token in (phrase or "").strip().split()]
    exact_spans = find_subsequences(
        [token.casefold() for token in tokens], exact_phrase_tokens
    )
    if exact_spans:
        return exact_spans
    target = norm_token(phrase)
    if not target:
        return []
    spans = []
    for start in range(len(tokens)):
        compact = ""
        first_content = None
        for end in range(start, len(tokens)):
            piece = norm_token(tokens[end])
            if piece and first_content is None:
                first_content = end
            compact += piece
            if len(compact) > len(target):
                break
            if compact == target and first_content is not None:
                spans.append((first_content, end + 1))
                break
    return list(dict.fromkeys(spans))


def choose_surface_span(
    tokens: list[str], surface: str, evidence: str
) -> tuple[tuple[int | None, int | None], dict[str, Any]]:
    surface_spans = find_compact_spans(tokens, surface)
    evidence_spans = find_compact_spans(tokens, evidence)
    diagnostics: dict[str, Any] = {
        "surface": surface,
        "evidence": evidence,
        "surface_candidates": len(surface_spans),
        "evidence_candidates": len(evidence_spans),
        "method": "none",
    }
    if not surface_spans:
        return (None, None), diagnostics
    for surface_span in surface_spans:
        if any(
            surface_span[0] >= evidence_span[0] and surface_span[1] <= evidence_span[1]
            for evidence_span in evidence_spans
        ):
            diagnostics["method"] = "surface_inside_evidence"
            return surface_span, diagnostics
    if evidence_spans:
        evidence_midpoint = sum(evidence_spans[0]) / 2
        chosen = min(
            surface_spans,
            key=lambda span: abs((span[0] + span[1]) / 2 - evidence_midpoint),
        )
        diagnostics["method"] = "nearest_surface_to_evidence"
        return chosen, diagnostics
    if len(surface_spans) == 1:
        diagnostics["method"] = "unique_surface"
        return surface_spans[0], diagnostics
    diagnostics["method"] = "first_surface_fallback"
    return surface_spans[0], diagnostics


def recover_offsets_from_evidence(
    surface_payload: dict[str, Any], input_text: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    tokens = parse_prompt_tokens(input_text)
    recovered_events: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"token_count": len(tokens), "items": [], "missing_offsets": 0}
    events = surface_payload.get("events", []) if isinstance(surface_payload, dict) else []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        trigger_span, trigger_diag = choose_surface_span(
            tokens, trigger.get("text") or "", trigger.get("evidence") or ""
        )
        trigger_diag.update({"kind": "trigger", "event_type": event.get("event_type")})
        diagnostics["items"].append(trigger_diag)
        if trigger_span[0] is None:
            diagnostics["missing_offsets"] += 1
        recovered_arguments = []
        arguments = event.get("arguments", [])
        for argument in arguments if isinstance(arguments, list) else []:
            if not isinstance(argument, dict):
                continue
            argument_span, argument_diag = choose_surface_span(
                tokens, argument.get("text") or "", argument.get("evidence") or ""
            )
            argument_diag.update(
                {
                    "kind": "argument",
                    "event_type": event.get("event_type"),
                    "role": argument.get("role"),
                }
            )
            diagnostics["items"].append(argument_diag)
            if argument_span[0] is None:
                diagnostics["missing_offsets"] += 1
            recovered_arguments.append(
                {
                    "role": argument.get("role"),
                    "text": argument.get("text"),
                    "start": argument_span[0],
                    "end": argument_span[1],
                }
            )
        recovered_events.append(
            {
                "event_type": event.get("event_type"),
                "trigger": {
                    "text": trigger.get("text"),
                    "start": trigger_span[0],
                    "end": trigger_span[1],
                },
                "arguments": recovered_arguments,
            }
        )
    return {"events": recovered_events}, diagnostics


def normalize_events(payload: dict[str, Any] | None) -> tuple[set[tuple], set[tuple], set[tuple]]:
    trigger_set: set[tuple] = set()
    argument_set: set[tuple] = set()
    event_set: set[tuple] = set()
    events = payload.get("events", []) if isinstance(payload, dict) else []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        trigger_tuple = (event_type, trigger.get("start"), trigger.get("end"))
        trigger_set.add(trigger_tuple)
        normalized_arguments = []
        arguments = event.get("arguments", [])
        for argument in arguments if isinstance(arguments, list) else []:
            if not isinstance(argument, dict):
                continue
            argument_tuple = (
                event_type,
                trigger.get("start"),
                trigger.get("end"),
                argument.get("role"),
                argument.get("start"),
                argument.get("end"),
            )
            argument_set.add(argument_tuple)
            normalized_arguments.append(
                (argument.get("role"), argument.get("start"), argument.get("end"))
            )
        event_set.add(
            (
                event_type,
                trigger.get("start"),
                trigger.get("end"),
                tuple(sorted(normalized_arguments, key=_nullable_tuple_key)),
            )
        )
    return trigger_set, argument_set, event_set


def _nullable_tuple_key(item: tuple[Any, Any, Any]) -> tuple[str, int, int]:
    return (
        item[0] or "",
        -1 if item[1] is None else int(item[1]),
        -1 if item[2] is None else int(item[2]),
    )


def f1(predicted: set[tuple], gold: set[tuple]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    precision = len(predicted & gold) / len(predicted)
    recall = len(predicted & gold) / len(gold)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def metric_f1s(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    pred_trigger, pred_argument, pred_event = normalize_events(predicted)
    gold_trigger, gold_argument, gold_event = normalize_events(gold)
    return {
        "trigger": f1(pred_trigger, gold_trigger),
        "argument": f1(pred_argument, gold_argument),
        "event": f1(pred_event, gold_event),
    }


def weighted_quality(predicted: dict[str, Any], gold: dict[str, Any], profile: str) -> float:
    if profile not in PROFILE_WEIGHTS:
        raise ValueError(f"unknown profile: {profile}")
    scores = metric_f1s(predicted, gold)
    return sum(PROFILE_WEIGHTS[profile][name] * scores[name] for name in scores)


def is_exact(predicted: dict[str, Any], gold: dict[str, Any]) -> bool:
    return normalize_events(predicted) == normalize_events(gold)


def offsets_complete(payload: dict[str, Any]) -> bool:
    events = payload.get("events", []) if isinstance(payload, dict) else []
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict):
            return False
        trigger = event.get("trigger")
        if not isinstance(trigger, dict) or trigger.get("start") is None or trigger.get("end") is None:
            return False
        arguments = event.get("arguments", [])
        if not isinstance(arguments, list):
            return False
        if any(
            not isinstance(argument, dict)
            or argument.get("start") is None
            or argument.get("end") is None
            for argument in arguments
        ):
            return False
    return True


def event_types_within_candidates(payload: dict[str, Any], candidate_types: Iterable[str]) -> bool:
    allowed = set(candidate_types)
    events = payload.get("events", []) if isinstance(payload, dict) else []
    return isinstance(events, list) and all(
        isinstance(event, dict) and event.get("event_type") in allowed for event in events
    )


def classify_single_error(predicted: dict[str, Any], gold: dict[str, Any]) -> str | None:
    pred_trigger, pred_argument, pred_event = normalize_events(predicted)
    gold_trigger, gold_argument, gold_event = normalize_events(gold)
    if (pred_trigger, pred_argument, pred_event) == (gold_trigger, gold_argument, gold_event):
        return "exact"

    pred_untyped = {(start, end) for _, start, end in pred_trigger}
    gold_untyped = {(start, end) for _, start, end in gold_trigger}
    pred_types = Counter(event_type for event_type, _, _ in pred_trigger)
    gold_types = Counter(event_type for event_type, _, _ in gold_trigger)

    if pred_untyped == gold_untyped and len(pred_trigger) == len(gold_trigger) and pred_trigger != gold_trigger:
        return "wrong_type"

    if pred_types == gold_types and len(pred_trigger) == len(gold_trigger) and pred_trigger != gold_trigger:
        unmatched_pred = pred_trigger - gold_trigger
        unmatched_gold = gold_trigger - pred_trigger
        if len(unmatched_pred) == len(unmatched_gold) == 1:
            pred_type, pred_start, pred_end = next(iter(unmatched_pred))
            gold_type, gold_start, gold_end = next(iter(unmatched_gold))
            if pred_type == gold_type and _nearby_spans(pred_start, pred_end, gold_start, gold_end):
                return "trigger_drift"

    if pred_trigger == gold_trigger and pred_argument < gold_argument and not (pred_argument - gold_argument):
        return "argument_omission"

    if pred_trigger < gold_trigger and not (pred_trigger - gold_trigger) and pred_argument <= gold_argument:
        return "event_omission"

    if gold_trigger < pred_trigger and not (gold_trigger - pred_trigger) and gold_argument <= pred_argument:
        return "extra_frame"

    return None


def _nearby_spans(
    pred_start: Any, pred_end: Any, gold_start: Any, gold_end: Any, tolerance: int = 2
) -> bool:
    if None in (pred_start, pred_end, gold_start, gold_end):
        return False
    overlaps = int(pred_start) < int(gold_end) and int(gold_start) < int(pred_end)
    boundary_distance = abs(int(pred_start) - int(gold_start)) + abs(int(pred_end) - int(gold_end))
    return overlaps or boundary_distance <= tolerance


def find_heldout_leaks(value: Any, heldout_types: Iterable[str], path: str = "$") -> list[dict[str, str]]:
    lowered = {event_type.lower(): event_type for event_type in heldout_types}
    leaks: list[dict[str, str]] = []
    if isinstance(value, str):
        lower_value = value.lower()
        for lower_type, original_type in lowered.items():
            if lower_type in lower_value:
                leaks.append({"path": path, "event_type": original_type})
    elif isinstance(value, dict):
        for key, child in value.items():
            leaks.extend(find_heldout_leaks(child, heldout_types, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaks.extend(find_heldout_leaks(child, heldout_types, f"{path}[{index}]"))
    return leaks


def valid_length_pair(
    chosen_tokens: int,
    rejected_tokens: int,
    chosen_total_tokens: int,
    rejected_total_tokens: int,
    cutoff_len: int,
    minimum_ratio: float = 0.7,
    maximum_ratio: float = 1.3,
) -> bool:
    if chosen_tokens <= 0 or rejected_tokens <= 0:
        return False
    ratio = chosen_tokens / rejected_tokens
    return (
        minimum_ratio <= ratio <= maximum_ratio
        and chosen_total_tokens <= cutoff_len
        and rejected_total_tokens <= cutoff_len
    )
