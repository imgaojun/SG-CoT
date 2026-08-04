import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import (
    build_prediction_map,
    load_json,
    load_jsonl,
    load_schema_map,
    resolve_candidate_types,
    run_dir_name,
    select_rows,
    update_dataset_info,
    write_json,
)


def render_instruction():
    return (
        "Do event extraction with a short checklist-style reasoning trace. "
        "Use only the provided candidate event types and schema cards. "
        "Return strict JSON only. "
        "First list event clues in `clues`, then map each clue to one event type in `decisions`, "
        "then provide the final extracted events in top-level `events`. "
        "Do not add free-form explanation outside the JSON schema. "
        "If no valid event is supported, return "
        '{"clues": [], "decisions": [], "events": []}.'
    )


def render_schema_cards(candidate_types, schema_by_type):
    cards = []
    for idx, event_type in enumerate(candidate_types, start=1):
        cards.append(f"[{idx}] {schema_by_type[event_type]['document']}")
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
        "Output requirements:\n"
        "- Return exactly one JSON object.\n"
        "- Top-level keys must be `clues`, `decisions`, and `events`.\n"
        "- `clues` is a list of trigger candidates with text and token spans.\n"
        "- `decisions` maps each clue to one event type and a status.\n"
        "- `events` is the final extracted event list.\n"
        "- Each event contains `event_type`, `trigger`, and `arguments`.\n"
        "- If no valid event is supported, all three lists must be empty.\n\n"
        "Return JSON only."
    )


def clue_for_event(event):
    return {
        "text": event["trigger"]["text"],
        "start": event["trigger"]["start"],
        "end": event["trigger"]["end"],
    }


def decision_for_event(event, clue_id):
    return {
        "clue_id": clue_id,
        "event_type": event["event_type"],
        "status": "supported",
    }


def render_output(events):
    if not events:
        return json.dumps({"clues": [], "decisions": [], "events": []}, ensure_ascii=False)

    clues = []
    decisions = []
    normalized_events = []
    for clue_id, event in enumerate(events):
        clues.append(clue_for_event(event))
        decisions.append(decision_for_event(event, clue_id))
        normalized_events.append(
            {
                "event_type": event["event_type"],
                "trigger": {
                    "text": event["trigger"]["text"],
                    "start": event["trigger"]["start"],
                    "end": event["trigger"]["end"],
                },
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
        )

    return json.dumps(
        {
            "clues": clues,
            "decisions": decisions,
            "events": normalized_events,
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
                "output": render_output(row["event_mentions"]),
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
                    "prompt_style": "checklist_cot_v3_noexample",
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
        "prediction_tag": args.prediction_tag
        if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}
        else None,
        "prediction_model": args.prediction_model
        if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}
        else None,
        "prediction_part": (args.prediction_part or args.part)
        if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}
        else None,
        "selection_mode": args.selection_mode,
        "negative_ratio": args.negative_ratio if args.selection_mode == "sampled_neg" else None,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "prompt_style": "checklist_cot_v3_noexample",
    }
    write_json(dataset_dir / f"{args.dataset_name}.meta.json", meta)
    print(f"wrote {output_path}")
    print(f"updated {dataset_dir / 'dataset_info.json'}")
    print(f"wrote {dataset_dir / f'{args.dataset_name}.meta.json'}")


if __name__ == "__main__":
    main()
