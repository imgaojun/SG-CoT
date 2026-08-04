import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def avg(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def safe_len(obj):
    if obj is None:
        return 0
    return len(obj)


def scaffold_stats(row):
    pred = row["predicted"]
    if not isinstance(pred, dict):
        return {
            "has_events_key": False,
            "num_events": 0,
            "has_clues_key": False,
            "num_clues": 0,
            "has_decisions_key": False,
            "num_decisions": 0,
            "events_only_chars": 0,
            "scaffold_overhead_chars": 0,
        }

    events = pred.get("events", [])
    clues = pred.get("clues", None)
    decisions = pred.get("decisions", None)
    events_only_json = json.dumps({"events": events}, ensure_ascii=False)
    events_only_chars = len(events_only_json)
    generated_chars = len(row["generated_text"])
    return {
        "has_events_key": "events" in pred,
        "num_events": safe_len(events),
        "has_clues_key": "clues" in pred,
        "num_clues": safe_len(clues),
        "has_decisions_key": "decisions" in pred,
        "num_decisions": safe_len(decisions),
        "events_only_chars": events_only_chars,
        "scaffold_overhead_chars": max(0, generated_chars - events_only_chars),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.predictions_jsonl))
    stats = []
    for row in rows:
        generated_text = row["generated_text"]
        item = {
            "valid_json": row["valid_json"],
            "generated_chars": len(generated_text),
            "generated_words": len(generated_text.split()),
            "latency_sec": row["latency_sec"],
            **scaffold_stats(row),
        }
        stats.append(item)

    valid_rows = [row for row in stats if row["valid_json"]]
    invalid_rows = [row for row in stats if not row["valid_json"]]

    payload = {
        "label": args.label,
        "predictions_jsonl": args.predictions_jsonl,
        "num_examples": len(rows),
        "json_valid_rate": avg([1.0 if row["valid_json"] else 0.0 for row in stats]),
        "avg_latency_sec": avg([row["latency_sec"] for row in stats]),
        "avg_generated_chars": avg([row["generated_chars"] for row in stats]),
        "p50_generated_chars": percentile([row["generated_chars"] for row in stats], 0.50),
        "p90_generated_chars": percentile([row["generated_chars"] for row in stats], 0.90),
        "avg_generated_words": avg([row["generated_words"] for row in stats]),
        "avg_generated_chars_valid": avg([row["generated_chars"] for row in valid_rows]),
        "avg_generated_chars_invalid": avg([row["generated_chars"] for row in invalid_rows]),
        "events_key_rate": avg([1.0 if row["has_events_key"] else 0.0 for row in stats]),
        "empty_events_rate": avg([1.0 if row["num_events"] == 0 else 0.0 for row in stats]),
        "nonempty_events_rate": avg([1.0 if row["num_events"] > 0 else 0.0 for row in stats]),
        "avg_num_events": avg([row["num_events"] for row in stats]),
        "clues_key_rate": avg([1.0 if row["has_clues_key"] else 0.0 for row in stats]),
        "decisions_key_rate": avg([1.0 if row["has_decisions_key"] else 0.0 for row in stats]),
        "avg_num_clues": avg([row["num_clues"] for row in stats]),
        "avg_num_decisions": avg([row["num_decisions"] for row in stats]),
        "avg_events_only_chars": avg([row["events_only_chars"] for row in valid_rows]),
        "avg_scaffold_overhead_chars": avg([row["scaffold_overhead_chars"] for row in valid_rows]),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
