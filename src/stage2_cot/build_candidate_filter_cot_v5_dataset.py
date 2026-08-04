import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import (
    build_prediction_map,
    event_family,
    jaccard,
    load_json,
    load_jsonl,
    load_schema_map,
    normalize_cue_tokens,
    resolve_candidate_types,
    run_dir_name,
    select_rows,
    update_dataset_info,
    write_json,
)


def render_instruction():
    return (
        "Do event extraction with explicit candidate filtering under noisy candidate types. "
        "Use only the provided candidate event types and schema cards. "
        "Return exactly one JSON object with top-level keys `candidate_reviews` and `events`. "
        "`candidate_reviews` must cover every candidate type exactly once and in the same order as the candidate list. "
        "Each review must contain `event_type`, `status`, `reason_code`, `contrast_type`, `evidence_triggers`, and `matched_roles`. "
        "`status` must be `supported` or `rejected`. "
        "`reason_code` must be one of `supported`, `better_type_match`, `role_mismatch`, or `trigger_mismatch`. "
        "`events` is the final extracted event list. "
        "Do not add any explanation outside the JSON object. "
        'If no candidate type is supported, return all reviews as `rejected` and `{"events": []}`.'
    )


def sort_key_for_confusion(candidate_type, other_type, schema_by_type):
    left = schema_by_type[candidate_type]
    right = schema_by_type[other_type]
    same_family = 1.0 if event_family(candidate_type) == event_family(other_type) else 0.0
    role_overlap = jaccard(left.get("core_roles", []), right.get("core_roles", []))
    cue_overlap = jaccard(
        normalize_cue_tokens(left.get("trigger_cues", [])),
        normalize_cue_tokens(right.get("trigger_cues", [])),
    )
    return (same_family, role_overlap, cue_overlap, other_type)


def contrast_candidates(candidate_type, candidate_types, schema_by_type, max_items=3):
    others = [item for item in candidate_types if item != candidate_type]
    ranked = sorted(
        others,
        key=lambda other_type: sort_key_for_confusion(candidate_type, other_type, schema_by_type),
        reverse=True,
    )
    return ranked[:max_items]


def render_schema_cards(candidate_types, schema_by_type):
    cards = []
    for idx, event_type in enumerate(candidate_types, start=1):
        schema = schema_by_type[event_type]
        confusions = contrast_candidates(event_type, candidate_types, schema_by_type)
        confusion_text = ", ".join(confusions) if confusions else "none"
        cue_text = ", ".join(schema.get("trigger_cues", [])) or "none"
        role_text = ", ".join(schema.get("core_roles", [])) or "none"
        cards.append(
            "\n".join(
                [
                    f"[{idx}] {event_type}",
                    f"Family: {event_family(event_type)}",
                    f"Definition: {schema.get('definition', '')}",
                    f"Trigger cues: {cue_text}",
                    f"Core roles: {role_text}",
                    f"Potential confusions in current candidate list: {confusion_text}",
                ]
            )
        )
    return "\n\n".join(cards)


def render_input(row, candidate_types, schema_by_type):
    tokens_text = " ".join(row["tokens"])
    type_text = ", ".join(candidate_types)
    cards = render_schema_cards(candidate_types, schema_by_type)
    return (
        f"Text:\n{row['text']}\n\n"
        f"Tokens:\n{tokens_text}\n\n"
        f"Candidate event types:\n{type_text}\n\n"
        f"Schema cards:\n{cards}\n\n"
        "Filtering requirements:\n"
        "- Review every candidate type in order.\n"
        "- Use `supported` only when both trigger evidence and role pattern support the type.\n"
        "- Use `better_type_match` when a similar candidate is present but another event type fits better.\n"
        "- Use `role_mismatch` when trigger evidence is partially related but the role pattern does not support the type.\n"
        "- Use `trigger_mismatch` when the text does not support the candidate trigger pattern.\n"
        "- Keep `contrast_type` empty when it is not needed.\n"
        "- `events` must contain only the final accepted events.\n\n"
        "Return JSON only."
    )


def normalize_trigger(trigger):
    return {
        "text": trigger["text"],
        "start": trigger["start"],
        "end": trigger["end"],
    }


def normalize_event(event):
    return {
        "event_type": event["event_type"],
        "trigger": normalize_trigger(event["trigger"]),
        "arguments": [
            {
                "role": arg["role"],
                "text": arg["text"],
                "start": arg["start"],
                "end": arg["end"],
            }
            for arg in event["arguments"]
        ],
    }


def event_groups(events):
    grouped = {}
    for event in events:
        grouped.setdefault(event["event_type"], []).append(event)
    return grouped


def supported_review(candidate_type, events_for_type, schema_by_type):
    schema_roles = set(schema_by_type[candidate_type].get("core_roles", []))
    matched_roles = sorted(
        {
            arg["role"]
            for event in events_for_type
            for arg in event["arguments"]
            if not schema_roles or arg["role"] in schema_roles
        }
    )
    evidence_triggers = []
    seen = set()
    for event in events_for_type:
        trig = normalize_trigger(event["trigger"])
        key = (trig["start"], trig["end"], trig["text"])
        if key not in seen:
            seen.add(key)
            evidence_triggers.append(trig)
    return {
        "event_type": candidate_type,
        "status": "supported",
        "reason_code": "supported",
        "contrast_type": "",
        "evidence_triggers": evidence_triggers,
        "matched_roles": matched_roles,
    }


def nearest_gold_match(candidate_type, gold_events, schema_by_type):
    candidate_schema = schema_by_type[candidate_type]
    candidate_roles = set(candidate_schema.get("core_roles", []))
    candidate_cues = normalize_cue_tokens(candidate_schema.get("trigger_cues", []))
    best = None

    for event in gold_events:
        gold_type = event["event_type"]
        gold_schema = schema_by_type[gold_type]
        observed_roles = {arg["role"] for arg in event["arguments"]}
        observed_trigger_tokens = normalize_cue_tokens([event["trigger"]["text"]])
        same_family = 1.0 if event_family(candidate_type) == event_family(gold_type) else 0.0
        schema_role_overlap = jaccard(candidate_roles, gold_schema.get("core_roles", []))
        observed_role_overlap = jaccard(candidate_roles, observed_roles)
        observed_trigger_overlap = jaccard(candidate_cues, observed_trigger_tokens)
        schema_cue_overlap = jaccard(
            candidate_cues,
            normalize_cue_tokens(gold_schema.get("trigger_cues", [])),
        )
        score = same_family * 10.0 + observed_role_overlap * 4.0 + schema_role_overlap * 2.0 + max(
            observed_trigger_overlap,
            schema_cue_overlap,
        )
        record = {
            "gold_type": gold_type,
            "same_family": same_family,
            "schema_role_overlap": schema_role_overlap,
            "observed_role_overlap": observed_role_overlap,
            "observed_trigger_overlap": observed_trigger_overlap,
            "schema_cue_overlap": schema_cue_overlap,
            "overlap_roles": sorted(candidate_roles & observed_roles),
            "score": score,
        }
        if best is None or record["score"] > best["score"]:
            best = record
    return best


def rejected_reason(candidate_type, gold_events, schema_by_type):
    if not gold_events:
        return {
            "reason_code": "trigger_mismatch",
            "contrast_type": "",
            "matched_roles": [],
        }

    best = nearest_gold_match(candidate_type, gold_events, schema_by_type)
    if best is None:
        return {
            "reason_code": "trigger_mismatch",
            "contrast_type": "",
            "matched_roles": [],
        }

    if best["same_family"] > 0 or best["schema_role_overlap"] >= 0.5:
        return {
            "reason_code": "better_type_match",
            "contrast_type": best["gold_type"],
            "matched_roles": best["overlap_roles"],
        }
    if best["observed_trigger_overlap"] > 0 or best["schema_cue_overlap"] > 0 or best["observed_role_overlap"] > 0:
        return {
            "reason_code": "role_mismatch",
            "contrast_type": best["gold_type"],
            "matched_roles": best["overlap_roles"],
        }
    return {
        "reason_code": "trigger_mismatch",
        "contrast_type": "",
        "matched_roles": [],
    }


def rejected_review(candidate_type, gold_events, schema_by_type):
    reject = rejected_reason(candidate_type, gold_events, schema_by_type)
    return {
        "event_type": candidate_type,
        "status": "rejected",
        "reason_code": reject["reason_code"],
        "contrast_type": reject["contrast_type"],
        "evidence_triggers": [],
        "matched_roles": reject["matched_roles"],
    }


def render_output(candidate_types, events, schema_by_type):
    grouped = event_groups(events)
    candidate_reviews = []
    for candidate_type in candidate_types:
        if candidate_type in grouped:
            candidate_reviews.append(supported_review(candidate_type, grouped[candidate_type], schema_by_type))
        else:
            candidate_reviews.append(rejected_review(candidate_type, events, schema_by_type))

    return json.dumps(
        {
            "candidate_reviews": candidate_reviews,
            "events": [normalize_event(event) for event in events],
        },
        ensure_ascii=False,
    )


def build_samples(
    rows,
    schema_by_type,
    candidate_universe,
    candidate_source,
    top_k,
    prediction_map,
    seed,
    dataset_meta,
    candidate_order_mode,
):
    samples = []
    for row in rows:
        resolved = resolve_candidate_types(
            row=row,
            schema_by_type=schema_by_type,
            candidate_universe=candidate_universe,
            candidate_source=candidate_source,
            top_k=top_k,
            prediction_map=prediction_map,
            seed=seed,
            candidate_order_mode=candidate_order_mode,
        )
        candidate_types = resolved["candidate_types"]
        raw_predicted_topk = resolved["raw_predicted_topk"]
        noise_mode = resolved["noise_mode"]

        samples.append(
            {
                "instruction": render_instruction(),
                "input": render_input(row, candidate_types, schema_by_type),
                "output": render_output(candidate_types, row["event_mentions"], schema_by_type),
                "meta": {
                    "doc_id": row["doc_id"],
                    "wnd_id": row["wnd_id"],
                    "candidate_types": candidate_types,
                    "gold_event_types": sorted({ev["event_type"] for ev in row["event_mentions"]}),
                    "raw_predicted_topk": raw_predicted_topk,
                    "missing_gold_from_raw_predicted_topk": (
                        sorted(set(ev["event_type"] for ev in row["event_mentions"]) - set(raw_predicted_topk))
                        if raw_predicted_topk is not None
                        else None
                    ),
                    "noise_mode": noise_mode,
                    "candidate_order_mode": candidate_order_mode,
                    "prompt_style": "candidate_filter_cot_v5",
                    **dataset_meta,
                },
            }
        )
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/processed/type_holdout")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--schema_path", required=True)
    parser.add_argument(
        "--candidate_source",
        choices=[
            "predicted",
            "oracle",
            "oracle_clean",
            "all_schema",
            "oracle_anchor_predicted",
            "oracle_inject_missing_predicted",
            "oracle_random_noise",
            "oracle_hard_noise",
            "oracle_mixed_noise",
        ],
        required=True,
    )
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--candidate_order_mode", choices=["as_is", "shuffle", "sorted"], default="as_is")
    parser.add_argument("--prediction_tag", default="qwen8b_main")
    parser.add_argument("--prediction_model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--prediction_part", default=None)
    parser.add_argument("--selection_mode", choices=["positive_only", "all", "sampled_neg"], default="positive_only")
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dataset_dir", default="data/stage2_cot_datasets")
    parser.add_argument("--dataset_name", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.dataset / args.protocol / args.split
    rows = load_jsonl(data_dir / f"{args.part}.jsonl")
    rows = select_rows(rows, args.selection_mode, args.negative_ratio, args.seed)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    unseen_types = set(load_json(data_dir / "unseen_types.json"))
    seen_types = load_json(data_dir / "seen_types.json")
    candidate_universe = seen_types + sorted(unseen_types)
    schema_by_type = load_schema_map(Path(args.schema_path))

    missing = [event_type for event_type in candidate_universe if event_type not in schema_by_type]
    if missing:
        raise ValueError(f"Schema entries missing for candidate types: {missing}")

    prediction_map = None
    if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}:
        prediction_part = args.prediction_part or args.part
        prediction_dir = (
            Path("outputs/candidate_type_recall_runs")
            / run_dir_name(args.dataset, args.protocol, args.split, prediction_part, args.prediction_tag)
            / args.prediction_model.replace("/", "__")
        )
        prediction_map = build_prediction_map(prediction_dir / "predictions.jsonl")

    dataset_meta = {
        "source_dataset": args.dataset,
        "source_protocol": args.protocol,
        "source_split": args.split,
        "source_part": args.part,
        "candidate_source": args.candidate_source,
        "top_k": args.top_k,
        "candidate_order_mode": args.candidate_order_mode,
        "selection_mode": args.selection_mode,
        "prediction_model": args.prediction_model
        if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}
        else None,
        "prediction_part": (args.prediction_part or args.part)
        if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}
        else None,
    }
    samples = build_samples(
        rows=rows,
        schema_by_type=schema_by_type,
        candidate_universe=candidate_universe,
        candidate_source=args.candidate_source,
        top_k=args.top_k,
        prediction_map=prediction_map,
        seed=args.seed,
        dataset_meta=dataset_meta,
        candidate_order_mode=args.candidate_order_mode,
    )

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{args.dataset_name}.jsonl"
    output_path = dataset_dir / file_name
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    update_dataset_info(dataset_dir, args.dataset_name, file_name)
    meta = {
        "dataset_name": args.dataset_name,
        "file_name": file_name,
        "num_examples": len(samples),
        "data_root": args.data_root,
        "dataset": args.dataset,
        "protocol": args.protocol,
        "split": args.split,
        "part": args.part,
        "schema_path": args.schema_path,
        "candidate_source": args.candidate_source,
        "top_k": args.top_k,
        "candidate_order_mode": args.candidate_order_mode,
        "selection_mode": args.selection_mode,
        "seed": args.seed,
    }
    write_json(dataset_dir / f"{args.dataset_name}.meta.json", meta)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
