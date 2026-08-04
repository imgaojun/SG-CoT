import argparse
import json
from collections import Counter
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def avg(values):
    return sum(values) / len(values) if values else 0.0


def summarize(rows):
    if not rows:
        return {
            "num_examples": 0,
            "json_valid_rate": 0.0,
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "avg_latency_sec": 0.0,
            "avg_num_gold_event_types": 0.0,
            "avg_min_train_freq": 0.0,
        }
    return {
        "num_examples": len(rows),
        "json_valid_rate": avg([1.0 if row["valid_json"] else 0.0 for row in rows]),
        "trigger_f1": avg([row["trigger_f1"] for row in rows]),
        "argument_f1": avg([row["argument_f1"] for row in rows]),
        "event_f1": avg([row["event_f1"] for row in rows]),
        "avg_latency_sec": avg([row["latency_sec"] for row in rows]),
        "avg_num_gold_event_types": avg([len(row["gold_event_types"]) for row in rows]),
        "avg_min_train_freq": avg([row["min_train_freq"] for row in rows]),
    }


def build_train_type_counter(train_jsonl: Path):
    counter = Counter()
    for row in load_jsonl(train_jsonl):
        for ev in row["event_mentions"]:
            counter[ev["event_type"]] += 1
    return counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_source_jsonl", required=True)
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    train_counter = build_train_type_counter(Path(args.train_source_jsonl))
    seen_freqs = sorted(train_counter.values())
    if not seen_freqs:
        raise ValueError("Empty training type counter.")
    median_seen_freq = seen_freqs[len(seen_freqs) // 2]

    dataset_rows = load_jsonl(Path(args.dataset_jsonl))
    pred_rows = load_jsonl(Path(args.predictions_jsonl))
    if len(dataset_rows) != len(pred_rows):
        raise ValueError(
            f"Length mismatch: dataset has {len(dataset_rows)} rows but predictions have {len(pred_rows)} rows."
        )

    groups = {"head": [], "tail": [], "unseen": []}
    for idx, (dataset_row, pred_row) in enumerate(zip(dataset_rows, pred_rows)):
        dataset_input = dataset_row["input"].strip()
        pred_input = pred_row["input"].strip()
        if dataset_input != pred_input:
            raise ValueError(f"Input mismatch at row {idx}.")

        gold_event_types = dataset_row["meta"]["gold_event_types"]
        freqs = [train_counter.get(event_type, 0) for event_type in gold_event_types]
        min_freq = min(freqs) if freqs else 0

        if any(freq == 0 for freq in freqs):
            group = "unseen"
        elif min_freq < median_seen_freq:
            group = "tail"
        else:
            group = "head"

        groups[group].append(
            {
                "wnd_id": dataset_row["meta"].get("wnd_id"),
                "gold_event_types": gold_event_types,
                "min_train_freq": min_freq,
                "valid_json": pred_row["valid_json"],
                "latency_sec": pred_row["latency_sec"],
                "trigger_f1": pred_row["trigger_f1"],
                "argument_f1": pred_row["argument_f1"],
                "event_f1": pred_row["event_f1"],
            }
        )

    payload = {
        "label": args.label,
        "train_source_jsonl": args.train_source_jsonl,
        "dataset_jsonl": args.dataset_jsonl,
        "predictions_jsonl": args.predictions_jsonl,
        "median_seen_type_frequency": median_seen_freq,
        "seen_type_frequency_min": seen_freqs[0],
        "seen_type_frequency_max": seen_freqs[-1],
        "num_seen_types": len(train_counter),
        "groups": {group: summarize(rows) for group, rows in groups.items()},
        "group_sizes": {group: len(rows) for group, rows in groups.items()},
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
