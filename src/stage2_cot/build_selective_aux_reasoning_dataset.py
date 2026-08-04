import argparse
from collections import Counter
import copy
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import (
    event_family,
    jaccard,
    load_jsonl,
    load_schema_map,
    normalize_cue_tokens,
    update_dataset_info,
    write_json,
)


REASON_CODES = [
    "family_boundary",
    "role_signature",
    "trigger_signature",
    "context_support",
]

CONFRARE_CONFUSION_WEIGHT = 0.55
CONFRARE_EVENT_TYPE_RARITY_WEIGHT = 0.30
CONFRARE_ROLE_SIGNATURE_RARITY_WEIGHT = 0.15
CONFRARE_CONFUSION_NORMALIZER = 14.0
RAREONLY_EVENT_TYPE_RARITY_WEIGHT = 0.40
RAREONLY_ROLE_SIGNATURE_RARITY_WEIGHT = 0.60
CONFROLE_CONFUSION_WEIGHT = 0.65
CONFROLE_ROLE_SIGNATURE_RARITY_WEIGHT = 0.35
ROLECONF_CONFUSION_WEIGHT = 0.35
ROLECONF_ROLE_SIGNATURE_RARITY_WEIGHT = 0.30
ROLECONF_CORE_ROLE_DENSITY_WEIGHT = 0.20
ROLECONF_MULTI_EVENT_TRIGGER_WEIGHT = 0.15
HARDCONF_CONFUSION_WEIGHT = 0.35
HARDCONF_ROLE_SIGNATURE_RARITY_WEIGHT = 0.20
HARDCONF_ROLE_DENSITY_WEIGHT = 0.20
HARDCONF_MULTI_EVENT_TRIGGER_WEIGHT = 0.15
HARDCONF_CORE_ROLE_ABSENCE_WEIGHT = 0.10
NO_ROLE_SIGNATURE = "NO_ROLES"
HYBRID_DUPAUX_TYPEROLELITE = "hybrid_dupaux_typerolelite"
DIRECT_DUP_2X = "direct_dup_2x"


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_output_events(row):
    payload = row["output"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload.get("events", [])


def normalize_trigger(trigger):
    return {
        "text": trigger["text"],
        "start": trigger["start"],
        "end": trigger["end"],
    }


def normalize_argument_span(arg):
    return {
        "text": arg["text"],
        "start": arg["start"],
        "end": arg["end"],
    }


def normalize_argument(arg):
    return {
        "role": arg["role"],
        "text": arg["text"],
        "start": arg["start"],
        "end": arg["end"],
    }


def normalize_event(event):
    return {
        "event_type": event["event_type"],
        "trigger": normalize_trigger(event["trigger"]),
        "arguments": [normalize_argument(arg) for arg in event.get("arguments", [])],
    }


def schema_roles(schema_by_type, event_type):
    return set(schema_by_type[event_type].get("core_roles", []))


def schema_cues(schema_by_type, event_type):
    return normalize_cue_tokens(schema_by_type[event_type].get("trigger_cues", []))


def type_confusion_score(gold_type, other_type, schema_by_type):
    same_family = 1.0 if event_family(gold_type) == event_family(other_type) else 0.0
    role_overlap = jaccard(schema_roles(schema_by_type, gold_type), schema_roles(schema_by_type, other_type))
    cue_overlap = jaccard(schema_cues(schema_by_type, gold_type), schema_cues(schema_by_type, other_type))
    return same_family * 10.0 + role_overlap * 3.0 + cue_overlap


def sample_confusion_score(row, schema_by_type):
    events = parse_output_events(row)
    gold_types = sorted({event["event_type"] for event in events})
    candidate_types = row.get("meta", {}).get("candidate_types", [])
    scores = [
        type_confusion_score(gold_type, candidate_type, schema_by_type)
        for gold_type in gold_types
        for candidate_type in candidate_types
        if candidate_type != gold_type
    ]
    return max(scores) if scores else 0.0


def event_role_signature(event):
    roles = sorted({arg["role"] for arg in event.get("arguments", [])})
    role_text = "|".join(roles) if roles else NO_ROLE_SIGNATURE
    return f"{event['event_type']}::{role_text}"


def build_confrare_stats(rows):
    event_type_counts = Counter()
    role_signature_counts = Counter()
    for row in rows:
        for event in parse_output_events(row):
            event_type_counts[event["event_type"]] += 1
            role_signature_counts[event_role_signature(event)] += 1

    return {
        "event_type_counts": event_type_counts,
        "role_signature_counts": role_signature_counts,
        "max_type_freq": max(event_type_counts.values(), default=0),
        "max_signature_freq": max(role_signature_counts.values(), default=0),
    }


def log_frequency_rarity(freq: int, max_freq: int):
    if freq <= 0 or max_freq <= 0:
        return 0.0
    return 1.0 - math.log1p(freq) / math.log1p(max_freq)


def sample_event_type_rarity(row, stats):
    events = parse_output_events(row)
    if not events:
        return 0.0
    event_type_counts = stats["event_type_counts"]
    max_type_freq = stats["max_type_freq"]
    return max(
        log_frequency_rarity(event_type_counts.get(event["event_type"], 0), max_type_freq)
        for event in events
    )


def sample_role_signature_rarity(row, stats):
    events = parse_output_events(row)
    if not events:
        return 0.0
    role_signature_counts = stats["role_signature_counts"]
    max_signature_freq = stats["max_signature_freq"]
    return max(
        log_frequency_rarity(role_signature_counts.get(event_role_signature(event), 0), max_signature_freq)
        for event in events
    )


def sample_core_role_density(row, schema_by_type):
    events = parse_output_events(row)
    observed = 0
    possible = 0
    for event in events:
        core_roles = schema_roles(schema_by_type, event["event_type"])
        if not core_roles:
            continue
        possible += len(core_roles)
        observed += len({arg["role"] for arg in event.get("arguments", []) if arg["role"] in core_roles})
    if possible == 0:
        return 0.0
    return observed / possible


def sample_role_density_norm(row):
    events = parse_output_events(row)
    num_args = sum(len(event.get("arguments", [])) for event in events)
    return min(num_args / 6.0, 1.0)


def sample_core_role_absence_risk(row, schema_by_type):
    events = parse_output_events(row)
    if not events:
        return 0.0
    missing = 0
    possible = 0
    for event in events:
        core_roles = schema_roles(schema_by_type, event["event_type"])
        if not core_roles:
            continue
        observed = {arg["role"] for arg in event.get("arguments", [])}
        possible += len(core_roles)
        missing += len(core_roles - observed)
    if possible == 0:
        return 0.0
    return missing / possible


def sample_multi_event_or_multi_trigger(row):
    events = parse_output_events(row)
    if len(events) > 1:
        return 1.0
    trigger_spans = {
        (event["trigger"].get("start"), event["trigger"].get("end"))
        for event in events
        if event.get("trigger")
    }
    return 1.0 if len(trigger_spans) > 1 else 0.0


def confrare_score_row(idx, row, schema_by_type, stats):
    confusion_score = sample_confusion_score(row, schema_by_type)
    confusion_norm = confusion_score / CONFRARE_CONFUSION_NORMALIZER
    event_type_rarity = sample_event_type_rarity(row, stats)
    role_signature_rarity = sample_role_signature_rarity(row, stats)
    confrare_score = (
        CONFRARE_CONFUSION_WEIGHT * confusion_norm
        + CONFRARE_EVENT_TYPE_RARITY_WEIGHT * event_type_rarity
        + CONFRARE_ROLE_SIGNATURE_RARITY_WEIGHT * role_signature_rarity
    )
    return {
        "idx": idx,
        "wnd_id": row_id(row),
        "score": confrare_score,
        "confusion_score": confusion_score,
        "confusion_norm": confusion_norm,
        "event_type_rarity": event_type_rarity,
        "role_signature_rarity": role_signature_rarity,
        "confrare_score": confrare_score,
    }


def rareonly_score_row(idx, row, stats):
    event_type_rarity = sample_event_type_rarity(row, stats)
    role_signature_rarity = sample_role_signature_rarity(row, stats)
    rareonly_score = (
        RAREONLY_EVENT_TYPE_RARITY_WEIGHT * event_type_rarity
        + RAREONLY_ROLE_SIGNATURE_RARITY_WEIGHT * role_signature_rarity
    )
    return {
        "idx": idx,
        "wnd_id": row_id(row),
        "score": rareonly_score,
        "event_type_rarity": event_type_rarity,
        "role_signature_rarity": role_signature_rarity,
        "rareonly_score": rareonly_score,
    }


def confrole_score_row(idx, row, schema_by_type, stats):
    confusion_score = sample_confusion_score(row, schema_by_type)
    confusion_norm = confusion_score / CONFRARE_CONFUSION_NORMALIZER
    role_signature_rarity = sample_role_signature_rarity(row, stats)
    confrole_score = (
        CONFROLE_CONFUSION_WEIGHT * confusion_norm
        + CONFROLE_ROLE_SIGNATURE_RARITY_WEIGHT * role_signature_rarity
    )
    return {
        "idx": idx,
        "wnd_id": row_id(row),
        "score": confrole_score,
        "confusion_score": confusion_score,
        "confusion_norm": confusion_norm,
        "role_signature_rarity": role_signature_rarity,
        "confrole_score": confrole_score,
    }


def roleconf_score_row(idx, row, schema_by_type, stats):
    confusion_score = sample_confusion_score(row, schema_by_type)
    confusion_norm = confusion_score / CONFRARE_CONFUSION_NORMALIZER
    role_signature_rarity = sample_role_signature_rarity(row, stats)
    core_role_density = sample_core_role_density(row, schema_by_type)
    multi_event_or_multi_trigger = sample_multi_event_or_multi_trigger(row)
    roleconf_score = (
        ROLECONF_CONFUSION_WEIGHT * confusion_norm
        + ROLECONF_ROLE_SIGNATURE_RARITY_WEIGHT * role_signature_rarity
        + ROLECONF_CORE_ROLE_DENSITY_WEIGHT * core_role_density
        + ROLECONF_MULTI_EVENT_TRIGGER_WEIGHT * multi_event_or_multi_trigger
    )
    return {
        "idx": idx,
        "wnd_id": row_id(row),
        "score": roleconf_score,
        "confusion_score": confusion_score,
        "confusion_norm": confusion_norm,
        "role_signature_rarity": role_signature_rarity,
        "core_role_density": core_role_density,
        "multi_event_or_multi_trigger": multi_event_or_multi_trigger,
        "roleconf_score": roleconf_score,
    }


def hardconf_score_row(idx, row, schema_by_type, stats):
    confusion_score = sample_confusion_score(row, schema_by_type)
    confusion_norm = confusion_score / CONFRARE_CONFUSION_NORMALIZER
    role_signature_rarity = sample_role_signature_rarity(row, stats)
    role_density_norm = sample_role_density_norm(row)
    multi_event_or_multi_trigger = sample_multi_event_or_multi_trigger(row)
    core_role_absence_risk = sample_core_role_absence_risk(row, schema_by_type)
    hardconf_score = (
        HARDCONF_CONFUSION_WEIGHT * confusion_norm
        + HARDCONF_ROLE_SIGNATURE_RARITY_WEIGHT * role_signature_rarity
        + HARDCONF_ROLE_DENSITY_WEIGHT * role_density_norm
        + HARDCONF_MULTI_EVENT_TRIGGER_WEIGHT * multi_event_or_multi_trigger
        + HARDCONF_CORE_ROLE_ABSENCE_WEIGHT * core_role_absence_risk
    )
    return {
        "idx": idx,
        "wnd_id": row_id(row),
        "score": hardconf_score,
        "confusion_score": confusion_score,
        "confusion_norm": confusion_norm,
        "role_signature_rarity": role_signature_rarity,
        "role_density_norm": role_density_norm,
        "multi_event_or_multi_trigger": multi_event_or_multi_trigger,
        "core_role_absence_risk": core_role_absence_risk,
        "hardconf_score": hardconf_score,
    }


def row_id(row):
    meta = row.get("meta", {})
    return meta.get("wnd_id") or f"{meta.get('doc_id', 'unknown')}::{hash(row.get('input', ''))}"


def select_auxiliary_indices(rows, schema_by_type, selection_strategy: str, aux_count: int, seed: int):
    if aux_count <= 0:
        return set(), []

    if selection_strategy == "confusion":
        scored = [
            {
                "idx": idx,
                "wnd_id": row_id(row),
                "score": sample_confusion_score(row, schema_by_type),
            }
            for idx, row in enumerate(rows)
        ]
        scored.sort(key=lambda item: (item["score"], item["wnd_id"]), reverse=True)
        selected = scored[:aux_count]
        return {item["idx"] for item in selected}, scored

    if selection_strategy == "random":
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        selected_indices = set(rng.sample(indices, min(aux_count, len(indices))))
        scored = [
            {
                "idx": idx,
                "wnd_id": row_id(row),
                "score": sample_confusion_score(row, schema_by_type),
                "random_selected": idx in selected_indices,
            }
            for idx, row in enumerate(rows)
        ]
        return selected_indices, scored

    if selection_strategy == "confrare":
        stats = build_confrare_stats(rows)
        scored = [
            confrare_score_row(idx, row, schema_by_type, stats)
            for idx, row in enumerate(rows)
        ]
        scored.sort(key=lambda item: (item["confrare_score"], item["confusion_score"], item["wnd_id"]), reverse=True)
        selected = scored[:aux_count]
        return {item["idx"] for item in selected}, scored

    if selection_strategy == "rareonly":
        stats = build_confrare_stats(rows)
        scored = [
            rareonly_score_row(idx, row, stats)
            for idx, row in enumerate(rows)
        ]
        scored.sort(key=lambda item: (item["rareonly_score"], item["wnd_id"]), reverse=True)
        selected = scored[:aux_count]
        return {item["idx"] for item in selected}, scored

    if selection_strategy == "confrole":
        stats = build_confrare_stats(rows)
        scored = [
            confrole_score_row(idx, row, schema_by_type, stats)
            for idx, row in enumerate(rows)
        ]
        scored.sort(key=lambda item: (item["confrole_score"], item["confusion_score"], item["wnd_id"]), reverse=True)
        selected = scored[:aux_count]
        return {item["idx"] for item in selected}, scored

    if selection_strategy == "roleconf":
        stats = build_confrare_stats(rows)
        scored = [
            roleconf_score_row(idx, row, schema_by_type, stats)
            for idx, row in enumerate(rows)
        ]
        scored.sort(key=lambda item: (item["roleconf_score"], item["confusion_score"], item["wnd_id"]), reverse=True)
        selected = scored[:aux_count]
        return {item["idx"] for item in selected}, scored

    if selection_strategy == "hardconf":
        stats = build_confrare_stats(rows)
        scored = [
            hardconf_score_row(idx, row, schema_by_type, stats)
            for idx, row in enumerate(rows)
        ]
        scored.sort(key=lambda item: (item["hardconf_score"], item["confusion_score"], item["wnd_id"]), reverse=True)
        selected = scored[:aux_count]
        return {item["idx"] for item in selected}, scored

    raise ValueError(f"Unsupported selection_strategy: {selection_strategy}")


def choose_contrast_type(event, candidate_types, schema_by_type):
    chosen_type = event["event_type"]
    ranked = []
    for candidate_type in candidate_types:
        if candidate_type == chosen_type:
            continue
        ranked.append((type_confusion_score(chosen_type, candidate_type, schema_by_type), candidate_type))
    if not ranked:
        return ""
    ranked.sort(reverse=True)
    best_score, best_type = ranked[0]
    return best_type if best_score > 0.0 else ""


def observed_roles(event):
    return {arg["role"] for arg in event.get("arguments", [])}


def observed_trigger_tokens(event):
    return normalize_cue_tokens([event["trigger"]["text"]])


def reason_code_for_event(event, contrast_type, schema_by_type):
    event_type = event["event_type"]
    if contrast_type and event_family(event_type) == event_family(contrast_type):
        return "family_boundary"

    role_overlap = jaccard(observed_roles(event), schema_roles(schema_by_type, event_type))
    cue_overlap = jaccard(observed_trigger_tokens(event), schema_cues(schema_by_type, event_type))
    if role_overlap > 0.0 and role_overlap >= cue_overlap:
        return "role_signature"
    if cue_overlap > 0.0:
        return "trigger_signature"
    return "context_support"


def build_decisions(events, candidate_types, schema_by_type):
    decisions = []
    for event in events:
        contrast_type = choose_contrast_type(event, candidate_types, schema_by_type)
        decisions.append(
            {
                "event_type": event["event_type"],
                "trigger": normalize_trigger(event["trigger"]),
                "contrast_type": contrast_type,
                "reason_code": reason_code_for_event(event, contrast_type, schema_by_type),
            }
        )
    return decisions


def role_priority(arg, core_roles):
    return 0 if arg["role"] in core_roles else 1


def build_role_checks(events, schema_by_type, max_role_checks: int):
    candidates = []
    order = 0
    for event in events:
        core_roles = schema_roles(schema_by_type, event["event_type"])
        for arg in event.get("arguments", []):
            candidates.append(
                {
                    "priority": role_priority(arg, core_roles),
                    "order": order,
                    "check": {
                        "event_type": event["event_type"],
                        "trigger": normalize_trigger(event["trigger"]),
                        "role": arg["role"],
                        "argument": normalize_argument_span(arg),
                    },
                }
            )
            order += 1
    candidates.sort(key=lambda item: (item["priority"], item["order"]))
    selected = sorted(candidates[:max_role_checks], key=lambda item: item["order"])
    return [item["check"] for item in selected]


def build_role_verify_checks(events, schema_by_type, max_role_checks: int):
    checks = []
    for event in events:
        trigger = normalize_trigger(event["trigger"])
        args_by_role = {}
        for arg in event.get("arguments", []):
            args_by_role.setdefault(arg["role"], []).append(arg)

        emitted_roles = set()
        for role in schema_by_type[event["event_type"]].get("core_roles", []):
            emitted_roles.add(role)
            args = args_by_role.get(role, [])
            if args:
                for arg in args:
                    checks.append(
                        {
                            "event_type": event["event_type"],
                            "trigger": trigger,
                            "role": role,
                            "status": "present",
                            "argument": normalize_argument_span(arg),
                        }
                    )
            else:
                checks.append(
                    {
                        "event_type": event["event_type"],
                        "trigger": trigger,
                        "role": role,
                        "status": "absent",
                        "argument": None,
                    }
                )

        for arg in event.get("arguments", []):
            if arg["role"] in emitted_roles:
                continue
            checks.append(
                {
                    "event_type": event["event_type"],
                    "trigger": trigger,
                    "role": arg["role"],
                    "status": "present",
                    "argument": normalize_argument_span(arg),
                }
            )

    return checks[:max_role_checks]


def render_auxiliary_instruction(target_style: str):
    if target_style == "type_lite":
        return (
            "Do event extraction with selective auxiliary reasoning. "
            "Use only the provided candidate event types and schema cards. "
            "Return exactly one JSON object with top-level keys `decisions` and `events`. "
            "`decisions` contains compact accepted event decisions only; each item has "
            "`event_type`, `trigger`, `contrast_type`, and `reason_code`. "
            "`events` is the final extracted event list in the standard schema. "
            "Do not add mode tokens or explanation outside the JSON object. "
            'If no valid event is supported, return {"decisions": [], "events": []}.'
        )

    if target_style == "role_lite":
        return (
            "Do event extraction with selective auxiliary reasoning. "
            "Use only the provided candidate event types and schema cards. "
            "Return exactly one JSON object with top-level keys `role_checks` and `events`. "
            "`role_checks` contains compact role-grounding checks for explicit argument spans. "
            "`events` is the final extracted event list in the standard schema. "
            "Do not add mode tokens or explanation outside the JSON object. "
            'If no valid event is supported, return {"role_checks": [], "events": []}.'
        )

    if target_style == "type_role_verify_lite":
        return (
            "Do event extraction with selective auxiliary reasoning. "
            "Use only the provided candidate event types and schema cards. "
            "Return exactly one JSON object with top-level keys `type_decisions`, `role_checks`, and `events`. "
            "`type_decisions` contains compact accepted event decisions only; each item has "
            "`event_type`, `trigger`, `contrast_type`, and `reason_code`. "
            "`role_checks` verifies core and observed roles with `status` as `present` or `absent`; "
            "present roles must include an explicit argument text/span and absent roles must use null argument. "
            "`events` is the final extracted event list in the standard schema. "
            "Do not add mode tokens or explanation outside the JSON object. "
            'If no valid event is supported, return {"type_decisions": [], "role_checks": [], "events": []}.'
        )

    if target_style != "type_role_lite":
        raise ValueError(f"Unsupported auxiliary target style: {target_style}")

    return (
        "Do event extraction with selective auxiliary reasoning. "
        "Use only the provided candidate event types and schema cards. "
        "Return exactly one JSON object with top-level keys `decisions`, `role_checks`, and `events`. "
        "`decisions` contains compact accepted event decisions only; each item has "
        "`event_type`, `trigger`, `contrast_type`, and `reason_code`. "
        "`role_checks` contains compact role-grounding checks for explicit argument spans. "
        "`events` is the final extracted event list in the standard schema. "
        "Do not add mode tokens or explanation outside the JSON object. "
        'If no valid event is supported, return {"decisions": [], "role_checks": [], "events": []}.'
    )


def build_auxiliary_payload(events, candidate_types, schema_by_type, max_role_checks: int, target_style: str):
    normalized_events = [normalize_event(event) for event in events]
    if target_style == "type_role_lite":
        return {
            "decisions": build_decisions(events, candidate_types, schema_by_type),
            "role_checks": build_role_checks(events, schema_by_type, max_role_checks),
            "events": normalized_events,
        }
    if target_style == "type_lite":
        return {
            "decisions": build_decisions(events, candidate_types, schema_by_type),
            "events": normalized_events,
        }
    if target_style == "type_role_verify_lite":
        return {
            "type_decisions": build_decisions(events, candidate_types, schema_by_type),
            "role_checks": build_role_verify_checks(events, schema_by_type, max_role_checks),
            "events": normalized_events,
        }
    if target_style == "role_lite":
        return {
            "role_checks": build_role_checks(events, schema_by_type, max_role_checks),
            "events": normalized_events,
        }
    raise ValueError(f"Unsupported auxiliary target style: {target_style}")


def build_auxiliary_row(row, schema_by_type, max_role_checks: int, selection_strategy: str, aux_rate: float, target_style: str):
    item = copy.deepcopy(row)
    meta = dict(item.get("meta", {}))
    meta.update(
        {
            "sar_source": "direct_duplicate_control" if target_style == "direct_dup" else "auxiliary_reasoning",
            "sar_selection_strategy": selection_strategy,
            "sar_aux_rate": aux_rate,
            "sar_target_style": target_style,
            "sar_confusion_score": sample_confusion_score(row, schema_by_type),
        }
    )
    item["meta"] = meta
    if target_style != "direct_dup":
        events = parse_output_events(row)
        candidate_types = item.get("meta", {}).get("candidate_types", [])
        payload = build_auxiliary_payload(events, candidate_types, schema_by_type, max_role_checks, target_style)
        item["instruction"] = render_auxiliary_instruction(target_style)
        item["output"] = json.dumps(payload, ensure_ascii=False)
    return item


def annotate_direct_row(row, *, dataset_role: str, selection_strategy: str, aux_rate: float, target_style: str):
    item = copy.deepcopy(row)
    meta = dict(item.get("meta", {}))
    meta.update(
        {
            "sar_source": "direct_anchor" if dataset_role == "train" else "direct_eval",
            "sar_selection_strategy": selection_strategy,
            "sar_aux_rate": aux_rate,
            "sar_target_style": target_style,
        }
    )
    item["meta"] = meta
    return item


def build_train_rows(
    rows,
    schema_by_type,
    selected_indices,
    selection_strategy: str,
    aux_rate: float,
    max_role_checks: int,
    seed: int,
    target_style: str,
):
    direct_rows = [
        annotate_direct_row(
            row,
            dataset_role="train",
            selection_strategy=selection_strategy,
            aux_rate=aux_rate,
            target_style=target_style,
        )
        for row in rows
    ]
    aux_rows = []
    for idx in sorted(selected_indices):
        if target_style == HYBRID_DUPAUX_TYPEROLELITE:
            dup_row = build_auxiliary_row(rows[idx], schema_by_type, max_role_checks, selection_strategy, aux_rate, "direct_dup")
            dup_row["meta"].update(
                {
                    "sar_target_style": target_style,
                    "sar_component": "direct_duplicate",
                    "sar_duplicate_round": 1,
                }
            )
            reasoning_row = build_auxiliary_row(
                rows[idx],
                schema_by_type,
                max_role_checks,
                selection_strategy,
                aux_rate,
                "type_role_lite",
            )
            reasoning_row["meta"].update(
                {
                    "sar_target_style": target_style,
                    "sar_component": "auxiliary_reasoning",
                }
            )
            aux_rows.extend([dup_row, reasoning_row])
            continue

        if target_style == DIRECT_DUP_2X:
            for duplicate_round in (1, 2):
                dup_row = build_auxiliary_row(
                    rows[idx],
                    schema_by_type,
                    max_role_checks,
                    selection_strategy,
                    aux_rate,
                    "direct_dup",
                )
                dup_row["meta"].update(
                    {
                        "sar_target_style": target_style,
                        "sar_component": "direct_duplicate",
                        "sar_duplicate_round": duplicate_round,
                    }
                )
                aux_rows.append(dup_row)
            continue

        aux_rows.append(
            build_auxiliary_row(rows[idx], schema_by_type, max_role_checks, selection_strategy, aux_rate, target_style)
        )
    mixed = direct_rows + aux_rows
    rng = random.Random(seed)
    rng.shuffle(mixed)
    return mixed, direct_rows, aux_rows


def register_dataset(dataset_dir: Path, dataset_name: str, rows, meta: dict):
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(dataset_dir / file_name, rows)
    update_dataset_info(dataset_dir, dataset_name, file_name)
    write_json(dataset_dir / f"{dataset_name}.meta.json", {"dataset_name": dataset_name, "file_name": file_name, **meta})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_path", required=True)
    parser.add_argument("--direct_train_jsonl", required=True)
    parser.add_argument("--direct_dev_jsonl", required=True)
    parser.add_argument("--direct_test_jsonl", required=True)
    parser.add_argument("--direct_test_seen_jsonl", required=True)
    parser.add_argument("--direct_test_unseen_jsonl", required=True)
    parser.add_argument("--dataset_dir", default="data/stage2_cot_datasets")
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--dev_dataset_name", required=True)
    parser.add_argument("--test_dataset_name", required=True)
    parser.add_argument("--test_seen_dataset_name", required=True)
    parser.add_argument("--test_unseen_dataset_name", required=True)
    parser.add_argument("--selection_strategy", choices=["confusion", "random", "confrare", "rareonly", "confrole", "roleconf", "hardconf"], required=True)
    parser.add_argument("--aux_rate", type=float, default=0.20)
    parser.add_argument(
        "--aux_target_style",
        choices=[
            "type_role_lite",
            "type_lite",
            "role_lite",
            "type_role_verify_lite",
            "direct_dup",
            HYBRID_DUPAUX_TYPEROLELITE,
            DIRECT_DUP_2X,
        ],
        default="type_role_lite",
    )
    parser.add_argument("--max_role_checks_per_sample", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    schema_by_type = load_schema_map(Path(args.schema_path))
    train_rows = load_jsonl(Path(args.direct_train_jsonl))
    aux_count = round(len(train_rows) * args.aux_rate)
    selected_indices, score_rows = select_auxiliary_indices(
        train_rows,
        schema_by_type,
        args.selection_strategy,
        aux_count,
        args.seed,
    )
    mixed_train_rows, direct_anchor_rows, aux_rows = build_train_rows(
        train_rows,
        schema_by_type,
        selected_indices,
        args.selection_strategy,
        args.aux_rate,
        args.max_role_checks_per_sample,
        args.seed,
        args.aux_target_style,
    )

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    shared_meta = {
        "schema_path": args.schema_path,
        "selection_strategy": args.selection_strategy,
        "aux_rate": args.aux_rate,
        "aux_target_style": args.aux_target_style,
        "max_role_checks_per_sample": args.max_role_checks_per_sample,
        "seed": args.seed,
        "direct_train_jsonl": args.direct_train_jsonl,
        "direct_dev_jsonl": args.direct_dev_jsonl,
        "direct_test_jsonl": args.direct_test_jsonl,
        "direct_test_seen_jsonl": args.direct_test_seen_jsonl,
        "direct_test_unseen_jsonl": args.direct_test_unseen_jsonl,
        "confrare_confusion_weight": CONFRARE_CONFUSION_WEIGHT,
        "confrare_event_type_rarity_weight": CONFRARE_EVENT_TYPE_RARITY_WEIGHT,
        "confrare_role_signature_rarity_weight": CONFRARE_ROLE_SIGNATURE_RARITY_WEIGHT,
        "confrare_confusion_normalizer": CONFRARE_CONFUSION_NORMALIZER,
        "rareonly_event_type_rarity_weight": RAREONLY_EVENT_TYPE_RARITY_WEIGHT,
        "rareonly_role_signature_rarity_weight": RAREONLY_ROLE_SIGNATURE_RARITY_WEIGHT,
        "confrole_confusion_weight": CONFROLE_CONFUSION_WEIGHT,
        "confrole_role_signature_rarity_weight": CONFROLE_ROLE_SIGNATURE_RARITY_WEIGHT,
        "confrole_confusion_normalizer": CONFRARE_CONFUSION_NORMALIZER,
    }
    train_meta = {
        **shared_meta,
        "dataset_role": "train",
        "direct_count": len(direct_anchor_rows),
        "auxiliary_count": len(aux_rows),
        "reasoning_auxiliary_count": sum(
            1 for row in aux_rows if row.get("meta", {}).get("sar_source") == "auxiliary_reasoning"
        ),
        "direct_duplicate_count": sum(
            1 for row in aux_rows if row.get("meta", {}).get("sar_source") == "direct_duplicate_control"
        ),
        "total_count": len(mixed_train_rows),
        "selected_aux_wnd_ids": [row_id(train_rows[idx]) for idx in sorted(selected_indices)],
        "score_rows": score_rows,
    }
    register_dataset(dataset_dir, args.train_dataset_name, mixed_train_rows, train_meta)

    eval_specs = [
        (args.direct_dev_jsonl, args.dev_dataset_name, "dev_seen"),
        (args.direct_test_jsonl, args.test_dataset_name, "test"),
        (args.direct_test_seen_jsonl, args.test_seen_dataset_name, "test_seen"),
        (args.direct_test_unseen_jsonl, args.test_unseen_dataset_name, "test_unseen"),
    ]
    for source_jsonl, dataset_name, role in eval_specs:
        rows = [
            annotate_direct_row(
                row,
                dataset_role=role,
                selection_strategy=args.selection_strategy,
                aux_rate=args.aux_rate,
                target_style=args.aux_target_style,
            )
            for row in load_jsonl(Path(source_jsonl))
        ]
        register_dataset(
            dataset_dir,
            dataset_name,
            rows,
            {
                **shared_meta,
                "dataset_role": role,
                "source_jsonl": source_jsonl,
                "num_examples": len(rows),
            },
        )

    print(
        json.dumps(
            {
                "train_dataset_name": args.train_dataset_name,
                "selection_strategy": args.selection_strategy,
                "aux_rate": args.aux_rate,
                "direct_count": len(direct_anchor_rows),
                "auxiliary_count": len(aux_rows),
                "total_count": len(mixed_train_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
