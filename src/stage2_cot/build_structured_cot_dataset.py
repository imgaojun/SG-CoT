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
    oracle_candidate_types,
    run_dir_name,
    select_rows,
    update_dataset_info,
    write_json,
)


def render_instruction():
    return (
        "You are doing event extraction with structured step-by-step reasoning. "
        "Use only the provided candidate event types and their schema cards. "
        "Return strict JSON only. "
        "For each supported event, first describe the trigger clue, then decide the event type, "
        "then summarize the relevant schema roles, then explain the argument-role assignments, "
        "and finally provide the extracted events in the top-level `events` field. "
        "If no valid event is expressed by the candidate set, return {\"reasoning\": [], \"events\": []}."
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
        "- Top-level keys must be `reasoning` and `events`.\n"
        "- `reasoning` is a structured list with trigger clue, type decision, schema focus, argument reasoning, and schema check.\n"
        "- `events` is the final extracted event list.\n"
        "- Each event contains `event_type`, `trigger`, and `arguments`.\n"
        "- If no valid event is supported, both lists must be empty.\n\n"
        "Return JSON only."
    )


def reasoning_for_event(event, schema_by_type):
    event_type = event["event_type"]
    trigger = event["trigger"]
    schema = schema_by_type[event_type]
    schema_roles = schema["core_roles"]
    args = event["arguments"]

    if args:
        role_bits = []
        for arg in args:
            role_bits.append(
                f'{arg["role"]} = "{arg["text"]}" at tokens [{arg["start"]}, {arg["end"]})'
            )
        arg_reasoning = "The text supports " + "; ".join(role_bits) + "."
    else:
        arg_reasoning = "No explicit arguments are confidently supported for this event."

    return {
        "trigger_clue": (
            f'The phrase "{trigger["text"]}" at tokens '
            f'[{trigger["start"]}, {trigger["end"]}) is an event clue.'
        ),
        "type_decision": (
            f'Among the candidate types, this clue best matches {event_type}.'
        ),
        "schema_focus": (
            f"For {event_type}, focus on core roles: {', '.join(schema_roles)}."
        ),
        "argument_reasoning": arg_reasoning,
        "schema_check": (
            f"The trigger and extracted arguments are consistent with the {event_type} schema."
        ),
    }


def render_output(events, schema_by_type):
    if not events:
        return json.dumps({"reasoning": [], "events": []}, ensure_ascii=False)

    normalized_events = []
    reasoning = []
    for event in events:
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
        reasoning.append(reasoning_for_event(event, schema_by_type))

    return json.dumps(
        {
            "reasoning": reasoning,
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
):
    import random

    rng = random.Random(seed)
    samples = []
    for row in rows:
        if candidate_source == "predicted":
            pred = prediction_map.get(row["wnd_id"])
            if pred is None:
                raise ValueError(f"Missing stage-1 prediction for wnd_id={row['wnd_id']}")
            candidate_types = pred["ranked_types"][:top_k]
        elif candidate_source == "oracle":
            candidate_types = oracle_candidate_types(row, candidate_universe, top_k, rng)
        elif candidate_source == "all_schema":
            candidate_types = candidate_universe[:]
        else:
            raise ValueError(f"Unsupported candidate_source: {candidate_source}")

        samples.append(
            {
                "instruction": render_instruction(),
                "input": render_input(row, candidate_types, schema_by_type),
                "output": render_output(row["event_mentions"], schema_by_type),
                "meta": {
                    "doc_id": row["doc_id"],
                    "wnd_id": row["wnd_id"],
                    "candidate_types": candidate_types,
                    "gold_event_types": sorted({ev["event_type"] for ev in row["event_mentions"]}),
                    "prompt_style": "structured_cot_v1_noexample",
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
    parser.add_argument("--candidate_source", choices=["predicted", "oracle", "all_schema"], required=True)
    parser.add_argument("--top_k", type=int, default=None)
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
    if args.candidate_source == "predicted":
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
        "selection_mode": args.selection_mode,
        "prediction_model": args.prediction_model if args.candidate_source == "predicted" else None,
        "prediction_part": (args.prediction_part or args.part) if args.candidate_source == "predicted" else None,
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
        "prediction_tag": args.prediction_tag if args.candidate_source == "predicted" else None,
        "prediction_model": args.prediction_model if args.candidate_source == "predicted" else None,
        "prediction_part": (args.prediction_part or args.part) if args.candidate_source == "predicted" else None,
        "selection_mode": args.selection_mode,
        "negative_ratio": args.negative_ratio if args.selection_mode == "sampled_neg" else None,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "prompt_style": "structured_cot_v1_noexample",
    }
    write_json(dataset_dir / f"{args.dataset_name}.meta.json", meta)
    print(f"wrote {output_path}")
    print(f"updated {dataset_dir / 'dataset_info.json'}")
    print(f"wrote {dataset_dir / f'{args.dataset_name}.meta.json'}")


if __name__ == "__main__":
    main()
