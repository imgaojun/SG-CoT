"""Atomic counterfactual construction for reasoning-path preferences."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

from src.stage2_preference.reasoning_preference import (
    ERROR_CATEGORIES,
    classify_single_error,
    find_compact_spans,
    norm_token,
)


ATOMIC_CATEGORIES = tuple(ERROR_CATEGORIES)
FORBIDDEN_LABEL_TERMS = ("chosen", "rejected", "gold", "error", "counterfactual")


def _span_index(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def event_key(event: dict[str, Any]) -> tuple[str, int, int]:
    trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
    return (
        str(event.get("event_type") or ""),
        _span_index(trigger.get("start", -1)),
        _span_index(trigger.get("end", -1)),
    )


def event_ref(event: dict[str, Any]) -> dict[str, Any]:
    event_type, start, end = event_key(event)
    return {"event_type": event_type, "start": start, "end": end}


def argument_key(argument: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(argument.get("role") or ""),
        _span_index(argument.get("start", -1)),
        _span_index(argument.get("end", -1)),
    )


def argument_ref(argument: dict[str, Any]) -> dict[str, Any]:
    role, start, end = argument_key(argument)
    return {"role": role, "start": start, "end": end}


def _event_sort_key(event: dict[str, Any]) -> tuple[int, int, str]:
    event_type, start, end = event_key(event)
    return start, end, event_type


def canonicalize_numeric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(payload)
    events = output.get("events", []) if isinstance(output, dict) else []
    clean_events = [event for event in events if isinstance(event, dict)]
    for event in clean_events:
        arguments = event.get("arguments", [])
        event["arguments"] = sorted(
            [argument for argument in arguments if isinstance(argument, dict)],
            key=lambda argument: (argument_key(argument)[1], argument_key(argument)[2], argument_key(argument)[0]),
        )
    output["events"] = sorted(clean_events, key=_event_sort_key)
    return output


def _find_event_index(events: list[dict[str, Any]], target: dict[str, Any]) -> int:
    target_key = (
        str(target.get("event_type") or ""),
        _span_index(target.get("start", -1)),
        _span_index(target.get("end", -1)),
    )
    for index, event in enumerate(events):
        if event_key(event) == target_key:
            return index
    raise ValueError(f"target event not found: {target}")


def _span_text(tokens: list[str], start: int, end: int) -> str:
    if not 0 <= start < end <= len(tokens):
        raise ValueError(f"invalid span [{start}, {end}) for {len(tokens)} tokens")
    return " ".join(tokens[start:end])


def apply_atomic_proposal(
    gold: dict[str, Any], proposal: dict[str, Any], tokens: list[str]
) -> dict[str, Any]:
    payload = canonicalize_numeric_payload(gold)
    events = payload["events"]
    category = str(proposal.get("category"))
    operation = proposal.get("operation") if isinstance(proposal.get("operation"), dict) else {}

    if category == "wrong_type":
        index = _find_event_index(events, operation["target_event"])
        events[index]["event_type"] = str(operation["new_type"])
    elif category == "trigger_drift":
        index = _find_event_index(events, operation["target_event"])
        start = int(operation["new_start"])
        end = int(operation["new_end"])
        events[index]["trigger"] = {
            "text": _span_text(tokens, start, end),
            "start": start,
            "end": end,
        }
    elif category == "argument_omission":
        index = _find_event_index(events, operation["target_event"])
        target_argument = operation["target_argument"]
        target_key = (
            str(target_argument.get("role") or ""),
            _span_index(target_argument.get("start", -1)),
            _span_index(target_argument.get("end", -1)),
        )
        arguments = events[index].get("arguments", [])
        events[index]["arguments"] = [
            argument for argument in arguments if argument_key(argument) != target_key
        ]
    elif category == "event_omission":
        index = _find_event_index(events, operation["target_event"])
        del events[index]
    elif category == "extra_frame":
        extra_event = copy.deepcopy(operation["extra_event"])
        events.append(extra_event)
    else:
        raise ValueError(f"unknown atomic category: {category}")
    return canonicalize_numeric_payload(payload)


def _nearby_spans(
    left_start: int, left_end: int, right_start: int, right_end: int, tolerance: int = 2
) -> bool:
    overlaps = left_start < right_end and right_start < left_end
    distance = abs(left_start - right_start) + abs(left_end - right_end)
    return overlaps or distance <= tolerance


def _proposal(
    category: str,
    operation: dict[str, Any],
    source: str,
    sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = {
        "category": category,
        "operation": operation,
        "proposal_source": source,
        "frequency": 1,
    }
    if sample is not None:
        output.update(
            {
                "source_sample_seed": sample.get("sample_seed"),
                "source_sample_round": sample.get("sample_round"),
                "source_sample_index": sample.get("sample_index"),
            }
        )
    return output


def extract_atomic_proposals(
    gold: dict[str, Any],
    predicted: dict[str, Any],
    candidate_types: Iterable[str],
    sample: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Factor an arbitrary prediction into atomic operations that can be replayed on gold."""

    gold_payload = canonicalize_numeric_payload(gold)
    predicted_payload = canonicalize_numeric_payload(predicted)
    gold_events = gold_payload.get("events", [])
    pred_events = predicted_payload.get("events", [])
    allowed = set(candidate_types)
    proposals: list[dict[str, Any]] = []

    gold_by_span: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    pred_by_span: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in gold_events:
        _, start, end = event_key(event)
        gold_by_span[(start, end)].append(event)
    for event in pred_events:
        _, start, end = event_key(event)
        pred_by_span[(start, end)].append(event)

    for span, span_gold_events in gold_by_span.items():
        for gold_event in span_gold_events:
            for pred_event in pred_by_span.get(span, []):
                pred_type = event_key(pred_event)[0]
                if pred_type and pred_type != event_key(gold_event)[0] and pred_type in allowed:
                    proposals.append(
                        _proposal(
                            "wrong_type",
                            {"target_event": event_ref(gold_event), "new_type": pred_type},
                            "observed_atomic",
                            sample,
                        )
                    )

    for gold_event in gold_events:
        gold_type, gold_start, gold_end = event_key(gold_event)
        for pred_event in pred_events:
            pred_type, pred_start, pred_end = event_key(pred_event)
            if (
                pred_type == gold_type
                and (pred_start, pred_end) != (gold_start, gold_end)
                and _nearby_spans(pred_start, pred_end, gold_start, gold_end)
            ):
                proposals.append(
                    _proposal(
                        "trigger_drift",
                        {
                            "target_event": event_ref(gold_event),
                            "new_start": pred_start,
                            "new_end": pred_end,
                        },
                        "observed_atomic",
                        sample,
                    )
                )

    pred_by_event = {event_key(event): event for event in pred_events}
    gold_keys = {event_key(event) for event in gold_events}
    for gold_event in gold_events:
        matching = pred_by_event.get(event_key(gold_event))
        if matching is None:
            if len(gold_events) >= 2:
                proposals.append(
                    _proposal(
                        "event_omission",
                        {"target_event": event_ref(gold_event)},
                        "observed_atomic",
                        sample,
                    )
                )
            continue
        predicted_arguments = {
            argument_key(argument) for argument in matching.get("arguments", [])
        }
        for argument in gold_event.get("arguments", []):
            if argument_key(argument) not in predicted_arguments:
                proposals.append(
                    _proposal(
                        "argument_omission",
                        {
                            "target_event": event_ref(gold_event),
                            "target_argument": argument_ref(argument),
                        },
                        "observed_atomic",
                        sample,
                    )
                )

    for pred_event in pred_events:
        if event_key(pred_event) not in gold_keys and event_key(pred_event)[0] in allowed:
            proposals.append(
                _proposal(
                    "extra_frame",
                    {"extra_event": pred_event},
                    "observed_atomic",
                    sample,
                )
            )
    return proposals


def _proposal_identity(proposal: dict[str, Any]) -> str:
    value = {"category": proposal["category"], "operation": proposal["operation"]}
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def aggregate_observed_proposals(
    gold: dict[str, Any],
    samples: list[dict[str, Any]],
    candidate_types: Iterable[str],
    tokens: list[str],
) -> dict[str, list[dict[str, Any]]]:
    aggregated: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    ordered_samples = sorted(
        samples,
        key=lambda sample: (
            int(sample.get("sample_round", 0)),
            int(sample.get("sample_index", 0)),
            int(sample.get("sample_seed", 0)),
        ),
    )
    for sample in ordered_samples:
        predicted = sample.get("recovered")
        if not isinstance(predicted, dict):
            continue
        for proposal in extract_atomic_proposals(gold, predicted, candidate_types, sample):
            try:
                mutated = apply_atomic_proposal(gold, proposal, tokens)
            except (KeyError, TypeError, ValueError):
                continue
            if classify_single_error(mutated, gold) != proposal["category"]:
                continue
            identity = _proposal_identity(proposal)
            current = aggregated[proposal["category"]].get(identity)
            if current is None:
                aggregated[proposal["category"]][identity] = proposal
            else:
                current["frequency"] = int(current.get("frequency", 1)) + 1
    return {
        category: sorted(
            proposals.values(),
            key=lambda proposal: (-int(proposal.get("frequency", 1)), _proposal_identity(proposal)),
        )
        for category, proposals in aggregated.items()
    }


def _alternative_type(
    target_event: dict[str, Any], gold: dict[str, Any], candidate_types: list[str]
) -> str | None:
    target_type, start, end = event_key(target_event)
    existing = {event_key(event) for event in gold.get("events", [])}
    family = target_type.split(":", 1)[0]
    alternatives = [
        event_type
        for event_type in candidate_types
        if event_type != target_type and (event_type, start, end) not in existing
    ]
    same_family = [event_type for event_type in alternatives if event_type.split(":", 1)[0] == family]
    return (same_family or alternatives or [None])[0]


def fallback_proposal(
    category: str,
    gold: dict[str, Any],
    candidate_types: list[str],
    tokens: list[str],
) -> dict[str, Any] | None:
    payload = canonicalize_numeric_payload(gold)
    events = payload.get("events", [])
    if not events:
        return None

    proposal: dict[str, Any] | None = None
    if category == "wrong_type":
        for event in events:
            alternative = _alternative_type(event, payload, candidate_types)
            if alternative:
                proposal = _proposal(
                    category,
                    {"target_event": event_ref(event), "new_type": alternative},
                    "deterministic_fallback",
                )
                break
    elif category == "trigger_drift":
        existing = {event_key(event) for event in events}
        for event in events:
            event_type, start, end = event_key(event)
            span_candidates = []
            if start > 0:
                span_candidates.append((start - 1, end))
            if end < len(tokens):
                span_candidates.append((start, end + 1))
            if start > 0 and end > start:
                span_candidates.append((start - 1, end - 1))
            if end < len(tokens):
                span_candidates.append((start + 1, end + 1))
            for new_start, new_end in span_candidates:
                if not 0 <= new_start < new_end <= len(tokens):
                    continue
                if (event_type, new_start, new_end) in existing:
                    continue
                if not any(norm_token(token) for token in tokens[new_start:new_end]):
                    continue
                proposal = _proposal(
                    category,
                    {
                        "target_event": event_ref(event),
                        "new_start": new_start,
                        "new_end": new_end,
                    },
                    "deterministic_fallback",
                )
                break
            if proposal:
                break
    elif category == "argument_omission":
        candidates = []
        for event in events:
            _, trigger_start, trigger_end = event_key(event)
            trigger_midpoint = (trigger_start + trigger_end) / 2
            for argument in event.get("arguments", []):
                _, start, end = argument_key(argument)
                distance = abs((start + end) / 2 - trigger_midpoint)
                candidates.append(
                    (
                        distance,
                        _event_sort_key(event),
                        argument_key(argument),
                        event,
                        argument,
                    )
                )
        if candidates:
            _, _, _, event, argument = max(
                candidates, key=lambda item: (item[0], item[1], item[2])
            )
            proposal = _proposal(
                category,
                {"target_event": event_ref(event), "target_argument": argument_ref(argument)},
                "deterministic_fallback",
            )
    elif category == "event_omission" and len(events) >= 2:
        event = max(events, key=_event_sort_key)
        proposal = _proposal(
            category,
            {"target_event": event_ref(event)},
            "deterministic_fallback",
        )
    elif category == "extra_frame":
        for event in events:
            alternative = _alternative_type(event, payload, candidate_types)
            if alternative:
                _, start, end = event_key(event)
                proposal = _proposal(
                    category,
                    {
                        "extra_event": {
                            "event_type": alternative,
                            "trigger": {
                                "text": _span_text(tokens, start, end),
                                "start": start,
                                "end": end,
                            },
                            "arguments": [],
                        }
                    },
                    "deterministic_fallback",
                )
                break
    if proposal is None:
        return None
    try:
        mutated = apply_atomic_proposal(payload, proposal, tokens)
    except (KeyError, TypeError, ValueError):
        return None
    return proposal if classify_single_error(mutated, payload) == category else None


def shortest_unique_evidence(tokens: list[str], start: int, end: int) -> str:
    if not 0 <= start < end <= len(tokens):
        raise ValueError(f"invalid token span [{start}, {end}) for {len(tokens)} tokens")
    for extra in range(len(tokens) + 1):
        candidates = []
        for left_extra in range(extra + 1):
            right_extra = extra - left_extra
            left = start - left_extra
            right = end + right_extra
            if left < 0 or right > len(tokens):
                continue
            evidence = " ".join(tokens[left:right])
            spans = find_compact_spans(tokens, evidence)
            if len(spans) == 1 and spans[0][0] <= start and spans[0][1] >= end:
                candidates.append((abs(left_extra - right_extra), left, right, evidence))
        if candidates:
            return min(candidates)[3]
    raise ValueError(f"no unique evidence for span [{start}, {end})")


def surface_payload(payload: dict[str, Any], tokens: list[str]) -> dict[str, Any]:
    events = []
    for event in canonicalize_numeric_payload(payload).get("events", []):
        event_type, start, end = event_key(event)
        arguments = []
        for argument in event.get("arguments", []):
            role, argument_start, argument_end = argument_key(argument)
            arguments.append(
                {
                    "role": role,
                    "text": _span_text(tokens, argument_start, argument_end),
                    "evidence": shortest_unique_evidence(tokens, argument_start, argument_end),
                }
            )
        events.append(
            {
                "event_type": event_type,
                "trigger": {
                    "text": _span_text(tokens, start, end),
                    "evidence": shortest_unique_evidence(tokens, start, end),
                },
                "arguments": arguments,
            }
        )
    return {"events": events}


def _quoted(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _frame_map(payload: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {event_key(event): event for event in payload.get("events", [])}


def _argument_universe(
    chosen_frames: dict[tuple[str, int, int], dict[str, Any]],
    rejected_frames: dict[tuple[str, int, int], dict[str, Any]],
) -> list[tuple[tuple[str, int, int], tuple[str, int, int]]]:
    values = set()
    for frames in (chosen_frames, rejected_frames):
        for frame_key, event in frames.items():
            for argument in event.get("arguments", []):
                values.add((frame_key, argument_key(argument)))
    return sorted(
        values,
        key=lambda value: (
            value[0][1],
            value[0][2],
            value[0][0],
            value[1][1],
            value[1][2],
            value[1][0],
        ),
    )


def _alternatives(event_type: str, candidate_types: list[str]) -> list[str]:
    family = event_type.split(":", 1)[0]
    others = [candidate for candidate in candidate_types if candidate != event_type]
    same_family = [candidate for candidate in others if candidate.split(":", 1)[0] == family]
    ordered = same_family + [candidate for candidate in others if candidate not in same_family]
    return ordered[:2]


def _render_thinking(
    state: dict[str, Any],
    universe: dict[tuple[str, int, int], dict[str, Any]],
    argument_universe: list[tuple[tuple[str, int, int], tuple[str, int, int]]],
    candidate_types: list[str],
    tokens: list[str],
) -> str:
    state_frames = _frame_map(state)
    frame_keys = sorted(universe, key=lambda key: (key[1], key[2], key[0]))
    considered = "; ".join(
        f"{_quoted(_span_text(tokens, key[1], key[2]))} for {key[0]}" for key in frame_keys
    )
    anchor_clauses = []
    type_clauses = []
    frame_clauses = []
    for key in frame_keys:
        event_type, start, end = key
        trigger_text = _quoted(_span_text(tokens, start, end))
        retained = key in state_frames
        anchor_clauses.append(
            f"{'Lock' if retained else 'Do not lock'} {trigger_text} as the minimal anchor for {event_type}."
        )
        alternatives = _alternatives(event_type, candidate_types)
        contrast = ", ".join(alternatives) if alternatives else "the remaining listed types"
        type_clauses.append(
            f"{'Select' if retained else 'Do not select'} {event_type} for {trigger_text} after comparison with {contrast}."
        )
        frame_clauses.append(
            f"{'Retain' if retained else 'Do not retain'} the {event_type} frame at {trigger_text}."
        )

    argument_clauses = []
    state_arguments = {
        (frame_key, argument_key(argument))
        for frame_key, event in state_frames.items()
        for argument in event.get("arguments", [])
    }
    for frame_key, arg_key in argument_universe:
        role, start, end = arg_key
        argument_text = _quoted(_span_text(tokens, start, end))
        event_type = frame_key[0]
        argument_clauses.append(
            f"{'Attach' if (frame_key, arg_key) in state_arguments else 'Do not attach'} {role} {argument_text} to {event_type}."
        )
    if not argument_clauses:
        argument_clauses.append("No local argument candidates are attached to the retained frames.")

    alignment_ledger = []
    for key in frame_keys:
        if key in state_frames:
            continue
        event_type, start, end = key
        alignment_ledger.append(
            "Inventory check: "
            f"{event_type} at {_quoted(_span_text(tokens, start, end))} with evidence "
            f"{_quoted(shortest_unique_evidence(tokens, start, end))} is inspected but not emitted."
        )
    for frame_key, arg_key in argument_universe:
        if (frame_key, arg_key) in state_arguments:
            continue
        role, start, end = arg_key
        alignment_ledger.append(
            "Attachment check: "
            f"{role} {_quoted(_span_text(tokens, start, end))} with evidence "
            f"{_quoted(shortest_unique_evidence(tokens, start, end))} is inspected but not emitted "
            f"for {frame_key[0]}."
        )
    if not alignment_ledger:
        alignment_ledger.append("Every retained inventory item is represented in the emitted structure.")

    return "\n\n".join(
        [
            f"Step 1: Event mention grounding. Audit these locally grounded frame candidates: {considered}.",
            "Step 2: Trigger anchor lock. " + " ".join(anchor_clauses),
            "Step 3: Schema-grounded type discrimination. " + " ".join(type_clauses),
            "Step 4: Event separation and completeness. " + " ".join(frame_clauses),
            "Step 5: Argument grounding. " + " ".join(argument_clauses),
            (
                "Step 6: Final alignment. Emit "
                f"{len(state_frames)} retained frame(s) and {len(state_arguments)} attached argument(s) "
                "with the audited anchors, types, and local evidence. "
                + " ".join(alignment_ledger)
            ),
        ]
    )


def render_canonical_pair(
    chosen_numeric: dict[str, Any],
    rejected_numeric: dict[str, Any],
    candidate_types: list[str],
    tokens: list[str],
) -> tuple[str, str]:
    chosen = canonicalize_numeric_payload(chosen_numeric)
    rejected = canonicalize_numeric_payload(rejected_numeric)
    chosen_frames = _frame_map(chosen)
    rejected_frames = _frame_map(rejected)
    universe = dict(chosen_frames)
    universe.update(rejected_frames)
    arguments = _argument_universe(chosen_frames, rejected_frames)

    def render(state: dict[str, Any]) -> str:
        thinking = _render_thinking(state, universe, arguments, candidate_types, tokens)
        final = json.dumps(surface_payload(state, tokens), ensure_ascii=False, separators=(",", ":"))
        return f"<thinking>{thinking}</thinking><final>{final}</final>"

    return render(chosen), render(rejected)


def label_leaks(response: str, source_text: str) -> list[str]:
    source_lower = source_text.lower()
    response_lower = response.lower()
    return [
        term
        for term in FORBIDDEN_LABEL_TERMS
        if re.search(rf"\b{re.escape(term)}\b", response_lower)
        and not re.search(rf"\b{re.escape(term)}\b", source_lower)
    ]


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int


def _add_flow_edge(
    graph: list[list[_FlowEdge]], source: int, target: int, capacity: int, cost: int
) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity, cost)
    backward = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(backward)
    return forward


def _stable_tie(seed: int, wnd_id: str, category: str) -> int:
    digest = hashlib.sha256(f"{seed}:{wnd_id}:{category}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1000


def select_quota_assignment(
    options_by_window: dict[str, dict[str, dict[str, Any]]],
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Solve a deterministic min-cost bipartite b-matching."""

    windows = sorted(options_by_window)
    categories = sorted(quotas)
    source = 0
    window_offset = 1
    category_offset = window_offset + len(windows)
    sink = category_offset + len(categories)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    category_nodes = {category: category_offset + index for index, category in enumerate(categories)}
    references: list[tuple[str, str, _FlowEdge]] = []

    for index, wnd_id in enumerate(windows):
        window_node = window_offset + index
        _add_flow_edge(graph, source, window_node, 1, 0)
        for category, option in sorted(options_by_window[wnd_id].items()):
            if category not in category_nodes:
                continue
            observed = option.get("proposal_source") == "observed_atomic"
            frequency = min(int(option.get("frequency", 1)), 999)
            base = 0 if observed else 10_000_000
            cost = base + (1000 - frequency) * 1000 + _stable_tie(seed, wnd_id, category)
            edge = _add_flow_edge(graph, window_node, category_nodes[category], 1, cost)
            references.append((wnd_id, category, edge))
    for category, quota in quotas.items():
        _add_flow_edge(graph, category_nodes[category], sink, int(quota), 0)

    target_flow = sum(quotas.values())
    flow = 0
    while flow < target_flow:
        distance = [10**30] * len(graph)
        in_queue = [False] * len(graph)
        previous_node = [-1] * len(graph)
        previous_edge = [-1] * len(graph)
        distance[source] = 0
        queue = deque([source])
        in_queue[source] = True
        while queue:
            node = queue.popleft()
            in_queue[node] = False
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                candidate = distance[node] + edge.cost
                if candidate >= distance[edge.to]:
                    continue
                distance[edge.to] = candidate
                previous_node[edge.to] = node
                previous_edge[edge.to] = edge_index
                if not in_queue[edge.to]:
                    queue.append(edge.to)
                    in_queue[edge.to] = True
        if previous_node[sink] < 0:
            break
        node = sink
        while node != source:
            parent = previous_node[node]
            edge_index = previous_edge[node]
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
        flow += 1

    if flow != target_flow:
        supplies = {
            category: sum(category in options for options in options_by_window.values())
            for category in categories
        }
        raise ValueError(f"quota assignment infeasible: flow={flow}/{target_flow}, supplies={supplies}")

    selected = []
    for wnd_id, category, edge in references:
        if edge.capacity == 0:
            selected.append(
                {
                    "wnd_id": wnd_id,
                    "error_category": category,
                    "option": options_by_window[wnd_id][category],
                }
            )
    counts = Counter(item["error_category"] for item in selected)
    if counts != Counter(quotas):
        raise AssertionError(f"assignment quota mismatch: {counts} != {quotas}")
    return sorted(selected, key=lambda item: item["wnd_id"])
