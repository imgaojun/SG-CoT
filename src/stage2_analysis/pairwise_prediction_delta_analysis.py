import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def avg(values):
    return sum(values) / len(values) if values else 0.0


def extract_section(text: str, start_marker: str, end_marker: str):
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def metric(row, key: str):
    return float(row.get(key, 0.0) or 0.0)


def predicted_event_types(row):
    predicted = row.get("predicted") or {}
    events = predicted.get("events", []) if isinstance(predicted, dict) else []
    return Counter(event.get("event_type", "") for event in events if isinstance(event, dict))


def gold_event_types(row):
    gold = row.get("gold") or {}
    events = gold.get("events", []) if isinstance(gold, dict) else []
    return Counter(event.get("event_type", "") for event in events if isinstance(event, dict))


def row_digest(row):
    text = extract_section(row.get("input", ""), "Text:\n", "\n\nTokens:\n")
    tokens = extract_section(row.get("input", ""), "Tokens:\n", "\n\nCandidate event types:\n")
    gold = json.dumps(row.get("gold"), ensure_ascii=False, sort_keys=True)
    return f"{text}\n{tokens}\n{gold}"


def check_alignment(left_rows, right_rows):
    if len(left_rows) != len(right_rows):
        raise ValueError(f"Row count mismatch: {len(left_rows)} vs {len(right_rows)}")
    mismatches = []
    for idx, (left, right) in enumerate(zip(left_rows, right_rows)):
        if row_digest(left) != row_digest(right):
            mismatches.append(idx)
            if len(mismatches) >= 5:
                break
    if mismatches:
        raise ValueError(f"Prediction rows are not aligned at indices: {mismatches}")


def summarize_rows(left_rows, right_rows, left_label, right_label):
    rows = []
    for idx, (left, right) in enumerate(zip(left_rows, right_rows)):
        record = {
            "index": idx,
            "text": extract_section(left.get("input", ""), "Text:\n", "\n\nTokens:\n")[:500],
            "gold_event_types": dict(gold_event_types(left)),
            "left_valid_json": bool(left.get("valid_json", False)),
            "right_valid_json": bool(right.get("valid_json", False)),
            "left_trigger_f1": metric(left, "trigger_f1"),
            "right_trigger_f1": metric(right, "trigger_f1"),
            "left_argument_f1": metric(left, "argument_f1"),
            "right_argument_f1": metric(right, "argument_f1"),
            "left_event_f1": metric(left, "event_f1"),
            "right_event_f1": metric(right, "event_f1"),
            "left_pred_event_types": dict(predicted_event_types(left)),
            "right_pred_event_types": dict(predicted_event_types(right)),
            "left_generated_chars": len(left.get("generated_text", "")),
            "right_generated_chars": len(right.get("generated_text", "")),
            "left_generated_prefix": left.get("generated_payload", left.get("generated_text", ""))[:300],
            "right_generated_prefix": right.get("generated_payload", right.get("generated_text", ""))[:300],
        }
        for key in ["trigger_f1", "argument_f1", "event_f1"]:
            record[f"delta_{key}"] = record[f"right_{key}"] - record[f"left_{key}"]
        record["delta_generated_chars"] = record["right_generated_chars"] - record["left_generated_chars"]
        record["json_regression"] = record["left_valid_json"] and not record["right_valid_json"]
        record["json_improvement"] = (not record["left_valid_json"]) and record["right_valid_json"]
        rows.append(record)

    def count_where(predicate):
        return sum(1 for row in rows if predicate(row))

    summary = {
        "num_examples": len(rows),
        "left_label": left_label,
        "right_label": right_label,
        "left": {
            "json_valid_rate": avg([1.0 if r["left_valid_json"] else 0.0 for r in rows]),
            "trigger_f1": avg([r["left_trigger_f1"] for r in rows]),
            "argument_f1": avg([r["left_argument_f1"] for r in rows]),
            "event_f1": avg([r["left_event_f1"] for r in rows]),
            "avg_generated_chars": avg([r["left_generated_chars"] for r in rows]),
        },
        "right": {
            "json_valid_rate": avg([1.0 if r["right_valid_json"] else 0.0 for r in rows]),
            "trigger_f1": avg([r["right_trigger_f1"] for r in rows]),
            "argument_f1": avg([r["right_argument_f1"] for r in rows]),
            "event_f1": avg([r["right_event_f1"] for r in rows]),
            "avg_generated_chars": avg([r["right_generated_chars"] for r in rows]),
        },
        "delta_right_minus_left": {
            "json_valid_rate": avg([1.0 if r["right_valid_json"] else 0.0 for r in rows])
            - avg([1.0 if r["left_valid_json"] else 0.0 for r in rows]),
            "trigger_f1": avg([r["delta_trigger_f1"] for r in rows]),
            "argument_f1": avg([r["delta_argument_f1"] for r in rows]),
            "event_f1": avg([r["delta_event_f1"] for r in rows]),
            "avg_generated_chars": avg([r["delta_generated_chars"] for r in rows]),
        },
        "counts": {
            "json_regression": count_where(lambda r: r["json_regression"]),
            "json_improvement": count_where(lambda r: r["json_improvement"]),
            "trigger_gain": count_where(lambda r: r["delta_trigger_f1"] > 0),
            "trigger_loss": count_where(lambda r: r["delta_trigger_f1"] < 0),
            "argument_gain": count_where(lambda r: r["delta_argument_f1"] > 0),
            "argument_loss": count_where(lambda r: r["delta_argument_f1"] < 0),
            "event_gain": count_where(lambda r: r["delta_event_f1"] > 0),
            "event_loss": count_where(lambda r: r["delta_event_f1"] < 0),
            "trigger_gain_argument_loss": count_where(
                lambda r: r["delta_trigger_f1"] > 0 and r["delta_argument_f1"] < 0
            ),
            "trigger_gain_event_loss": count_where(
                lambda r: r["delta_trigger_f1"] > 0 and r["delta_event_f1"] < 0
            ),
        },
    }
    return summary, rows


def select_examples(rows, key, reverse, limit):
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:limit]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_predictions_jsonl", required=True)
    parser.add_argument("--right_predictions_jsonl", required=True)
    parser.add_argument("--left_label", default="left")
    parser.add_argument("--right_label", default="right")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--example_limit", type=int, default=10)
    args = parser.parse_args()

    left_rows = load_jsonl(Path(args.left_predictions_jsonl))
    right_rows = load_jsonl(Path(args.right_predictions_jsonl))
    check_alignment(left_rows, right_rows)
    summary, rows = summarize_rows(left_rows, right_rows, args.left_label, args.right_label)

    payload = {
        "left_predictions_jsonl": args.left_predictions_jsonl,
        "right_predictions_jsonl": args.right_predictions_jsonl,
        "summary": summary,
        "examples": {
            "largest_trigger_gains": select_examples(rows, "delta_trigger_f1", True, args.example_limit),
            "largest_argument_losses": select_examples(rows, "delta_argument_f1", False, args.example_limit),
            "largest_event_losses": select_examples(rows, "delta_event_f1", False, args.example_limit),
            "json_regressions": [row for row in rows if row["json_regression"]][: args.example_limit],
            "trigger_gain_argument_loss": [
                row for row in rows if row["delta_trigger_f1"] > 0 and row["delta_argument_f1"] < 0
            ][: args.example_limit],
        },
    }
    write_json(Path(args.output_json), payload)
    print(json.dumps({"output_json": args.output_json, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
