import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import load_jsonl, write_json


def extract_section(text: str, start_marker: str, end_marker: str):
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def canonical_gold_events(payload):
    if isinstance(payload, dict) and "events" in payload:
        payload = {"events": payload.get("events", [])}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def row_hash_from_prediction(row):
    digest = hashlib.sha256()
    text_section = extract_section(row["input"], "Text:\n", "\n\nTokens:\n")
    candidate_section = extract_section(row["input"], "Candidate event types:\n", "\n\nSchema cards:\n")
    digest.update(text_section.encode("utf-8"))
    digest.update(b"\n")
    digest.update(candidate_section.encode("utf-8"))
    digest.update(b"\n")
    digest.update(canonical_gold_events(row["gold"]).encode("utf-8"))
    return digest.hexdigest()


def load_prediction_map(path: Path):
    rows = load_jsonl(path)
    mapping = {}
    for row in rows:
        row_hash = row_hash_from_prediction(row)
        if row_hash in mapping:
            raise ValueError(f"Duplicate row_hash in predictions: {path} -> {row_hash}")
        mapping[row_hash] = row
    return mapping


def label_record(direct_row, cot_row, arg_gain_threshold: float, trig_slack: float):
    direct_tuple = (direct_row["event_f1"], direct_row["argument_f1"], direct_row["trigger_f1"])
    cot_tuple = (cot_row["event_f1"], cot_row["argument_f1"], cot_row["trigger_f1"])
    direct_better = direct_tuple > cot_tuple
    cot_better = cot_tuple > direct_tuple

    if cot_better:
        loose_label = "COT"
        loose_reason = "cot_tuple_better"
    else:
        loose_label = "DIRECT"
        loose_reason = "direct_tuple_better_or_tie"

    if cot_row["event_f1"] > direct_row["event_f1"]:
        strict_label = "COT"
        strict_reason = "event_win"
    elif (
        cot_row["event_f1"] == direct_row["event_f1"]
        and (cot_row["argument_f1"] - direct_row["argument_f1"]) >= arg_gain_threshold
        and cot_row["trigger_f1"] >= direct_row["trigger_f1"] - trig_slack
    ):
        strict_label = "COT"
        strict_reason = "argument_gain_with_trigger_guard"
    else:
        strict_label = "DIRECT"
        strict_reason = "default_direct"

    return {
        "loose_label": loose_label,
        "loose_reason": loose_reason,
        "strict_label": strict_label,
        "strict_reason": strict_reason,
        "direct_better": direct_better,
        "cot_better": cot_better,
    }


def summarize(rows):
    if not rows:
        return {
            "num_examples": 0,
            "loose_counts": {},
            "strict_counts": {},
        }
    loose_counts = {"DIRECT": 0, "COT": 0}
    strict_counts = {"DIRECT": 0, "COT": 0}
    loose_reason_counts = {}
    strict_reason_counts = {}
    for row in rows:
        loose_counts[row["loose_label"]] += 1
        strict_counts[row["strict_label"]] += 1
        loose_reason_counts[row["loose_reason"]] = loose_reason_counts.get(row["loose_reason"], 0) + 1
        strict_reason_counts[row["strict_reason"]] = strict_reason_counts.get(row["strict_reason"], 0) + 1

    def avg(key):
        return sum(row[key] for row in rows) / len(rows)

    return {
        "num_examples": len(rows),
        "loose_counts": loose_counts,
        "strict_counts": strict_counts,
        "loose_reason_counts": loose_reason_counts,
        "strict_reason_counts": strict_reason_counts,
        "avg_direct_event_f1": avg("direct_event_f1"),
        "avg_cot_event_f1": avg("cot_event_f1"),
        "avg_direct_argument_f1": avg("direct_argument_f1"),
        "avg_cot_argument_f1": avg("cot_argument_f1"),
        "avg_direct_trigger_f1": avg("direct_trigger_f1"),
        "avg_cot_trigger_f1": avg("cot_trigger_f1"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_predictions", required=True)
    parser.add_argument("--cot_predictions", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary_json", default=None)
    parser.add_argument("--arg_gain_threshold", type=float, default=0.25)
    parser.add_argument("--trig_slack", type=float, default=0.05)
    args = parser.parse_args()

    direct_rows = load_prediction_map(Path(args.direct_predictions))
    cot_rows = load_prediction_map(Path(args.cot_predictions))
    common_hashes = set(direct_rows) & set(cot_rows)
    if common_hashes != set(direct_rows) or common_hashes != set(cot_rows):
        raise ValueError(
            "Prediction rows do not align perfectly: "
            f"direct_only={len(set(direct_rows) - common_hashes)} "
            f"cot_only={len(set(cot_rows) - common_hashes)}"
        )

    records = []
    for row_hash in sorted(common_hashes):
        direct_row = direct_rows[row_hash]
        cot_row = cot_rows[row_hash]
        labels = label_record(direct_row, cot_row, args.arg_gain_threshold, args.trig_slack)
        records.append(
            {
                "row_hash": row_hash,
                "direct_event_f1": direct_row["event_f1"],
                "cot_event_f1": cot_row["event_f1"],
                "direct_argument_f1": direct_row["argument_f1"],
                "cot_argument_f1": cot_row["argument_f1"],
                "direct_trigger_f1": direct_row["trigger_f1"],
                "cot_trigger_f1": cot_row["trigger_f1"],
                "event_delta": cot_row["event_f1"] - direct_row["event_f1"],
                "argument_delta": cot_row["argument_f1"] - direct_row["argument_f1"],
                "trigger_delta": cot_row["trigger_f1"] - direct_row["trigger_f1"],
                **labels,
            }
        )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = Path(args.output_summary_json) if args.output_summary_json else output_path.with_suffix(".summary.json")
    write_json(
        summary_path,
        {
            "direct_predictions": args.direct_predictions,
            "cot_predictions": args.cot_predictions,
            "arg_gain_threshold": args.arg_gain_threshold,
            "trig_slack": args.trig_slack,
            **summarize(records),
        },
    )
    print(json.dumps({"output_jsonl": args.output_jsonl, "output_summary_json": summary_path.as_posix(), **summarize(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
