import argparse
import json
import random
from pathlib import Path

from common import load_json


def run_dir_name(dataset: str, protocol: str, split: str, part: str, tag: str):
    dataset_slug = dataset.replace("-", "_")
    protocol_slug = protocol.replace("-", "_")
    return f"{dataset_slug}_{protocol_slug}_{split}_{part}_{tag}"


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def schema_map(schema_path: Path):
    items = load_json(schema_path)
    return {item["event_type"]: item for item in items}


def render_schema_cards(candidate_types, schema_by_type):
    cards = []
    for idx, event_type in enumerate(candidate_types, start=1):
        doc = schema_by_type[event_type]["document"]
        cards.append(f"[{idx}] {doc}")
    return "\n\n".join(cards)


def render_output(events):
    normalized = {"events": []}
    for event in events:
        normalized["events"].append(
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
    return json.dumps(normalized, ensure_ascii=False)


def render_instruction():
    return (
        "You are doing event extraction. Use only the provided candidate event types and their schema cards. "
        "Extract all event mentions supported by the text and output strict JSON with token offsets. "
        "If no valid event is expressed by the candidate set, output {\"events\": []}."
    )


def render_input(row, candidate_types, schema_by_type):
    tokens_text = " ".join(row["tokens"])
    type_text = ", ".join(candidate_types)
    cards = render_schema_cards(candidate_types, schema_by_type)
    return (
        f"Text:\n{row['text']}\n\n"
        f"Tokens:\n{tokens_text}\n\n"
        f"Candidate event types:\n{type_text}\n\n"
        f"Schema cards:\n{cards}\n\n"
        "Return JSON only."
    )


def pick_examples(rows, max_examples, seed):
    rows = [row for row in rows if row["event_mentions"]]
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    return rows[:max_examples]


def join_predictions(slice_spec):
    source_dataset = slice_spec["source_dataset"]
    source_protocol = slice_spec["source_protocol"]
    source_split = slice_spec["source_split"]
    source_part = slice_spec["source_part"]
    prediction_tag = slice_spec["prediction_tag"]
    prediction_model = slice_spec["prediction_model"]
    top_k = slice_spec["top_k"]

    data_path = (
        Path("data/processed/type_holdout")
        / source_dataset
        / source_protocol
        / source_split
        / f"{source_part}.jsonl"
    )
    prediction_path = (
        Path("outputs/candidate_type_recall_runs")
        / run_dir_name(source_dataset, source_protocol, source_split, source_part, prediction_tag)
        / prediction_model.replace("/", "__")
        / "predictions.jsonl"
    )
    rows = load_jsonl(data_path)
    preds = load_jsonl(prediction_path)
    pred_map = {row["wnd_id"]: row for row in preds}

    enriched = []
    for row in rows:
        if not row["event_mentions"]:
            continue
        pred = pred_map.get(row["wnd_id"])
        if pred is None:
            continue
        candidate_types = pred["ranked_types"][:top_k]
        enriched.append(
            {
                **row,
                "candidate_types": candidate_types,
                "prediction_gold_types": pred["gold_types"],
                "prediction_unseen_gold_types": pred["unseen_gold_types"],
            }
        )
    return enriched


def build_slice(spec, slice_key):
    slice_spec = spec["benchmark_slices"][slice_key]
    rows = join_predictions(slice_spec)
    rows = pick_examples(rows, slice_spec["max_examples"], slice_spec["sample_seed"])
    schema_path = Path("data/schema") / f"{slice_spec['source_dataset']}.event_schema.json"
    schema_by_type = schema_map(schema_path)

    dataset_dir = Path(slice_spec["dataset_dir_host"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / slice_spec["dataset_file"]

    samples = []
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            sample = {
                "instruction": render_instruction(),
                "input": render_input(row, row["candidate_types"], schema_by_type),
                "output": render_output(row["event_mentions"]),
                "meta": {
                    "doc_id": row["doc_id"],
                    "wnd_id": row["wnd_id"],
                    "candidate_types": row["candidate_types"],
                    "source_dataset": slice_spec["source_dataset"],
                    "source_protocol": slice_spec["source_protocol"],
                    "source_split": slice_spec["source_split"],
                    "source_part": slice_spec["source_part"],
                    "prediction_model": slice_spec["prediction_model"],
                    "top_k": slice_spec["top_k"],
                },
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            samples.append(sample)

    info = {
        "slice_key": slice_key,
        "dataset_name": slice_spec["dataset_name"],
        "dataset_file": slice_spec["dataset_file"],
        "num_examples": len(samples),
        "top_k": slice_spec["top_k"],
        "source_dataset": slice_spec["source_dataset"],
        "source_protocol": slice_spec["source_protocol"],
        "source_split": slice_spec["source_split"],
        "source_part": slice_spec["source_part"],
        "prediction_model": slice_spec["prediction_model"],
    }
    info_path = dataset_dir / f"{slice_spec['dataset_name']}.meta.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"wrote {output_path}")
    print(f"wrote {info_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/stage2_llamafactory_benchmark_spec.json")
    parser.add_argument("--slice_key", required=True)
    parser.add_argument("--max_examples_override", type=int, default=None)
    args = parser.parse_args()

    spec = load_json(Path(args.spec))
    if args.max_examples_override is not None:
        spec["benchmark_slices"][args.slice_key]["max_examples"] = args.max_examples_override
    build_slice(spec, args.slice_key)


if __name__ == "__main__":
    main()
