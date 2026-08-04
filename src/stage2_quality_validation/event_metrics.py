"""Lightweight exact event-normalization and set-PRF helpers."""

from __future__ import annotations

from typing import Any


def normalize_events(events_payload: Any) -> tuple[set[tuple], set[tuple], set[tuple]]:
    events = events_payload.get("events", []) if isinstance(events_payload, dict) else []
    trigger_set: set[tuple] = set()
    argument_set: set[tuple] = set()
    event_set: set[tuple] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        trigger = event.get("trigger", {})
        if not isinstance(trigger, dict):
            trigger = {}
        trig = (event_type, trigger.get("start"), trigger.get("end"))
        trigger_set.add(trig)
        arguments = []
        raw_arguments = event.get("arguments", [])
        if not isinstance(raw_arguments, list):
            raw_arguments = []
        for argument in raw_arguments:
            if not isinstance(argument, dict):
                continue
            argument_set.add(
                (
                    event_type,
                    trigger.get("start"),
                    trigger.get("end"),
                    argument.get("role"),
                    argument.get("start"),
                    argument.get("end"),
                )
            )
            arguments.append((argument.get("role"), argument.get("start"), argument.get("end")))
        sorted_arguments = tuple(
            sorted(
                arguments,
                key=lambda item: (
                    item[0] or "",
                    -1 if item[1] is None else item[1],
                    -1 if item[2] is None else item[2],
                ),
            )
        )
        event_set.add((event_type, trigger.get("start"), trigger.get("end"), sorted_arguments))
    return trigger_set, argument_set, event_set


def prf(pred_set: set, gold_set: set) -> dict[str, float]:
    if not pred_set and not gold_set:
        return {"p": 1.0, "r": 1.0, "f1": 1.0}
    if not pred_set or not gold_set:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    true_positive = len(pred_set & gold_set)
    precision = true_positive / len(pred_set)
    recall = true_positive / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"p": precision, "r": recall, "f1": f1}
