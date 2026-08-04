import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import load_jsonl, write_json


DIRECT_MODE = "<DIRECT>"
COT_MODE = "<COT>"


def extract_section(text: str, start_marker: str, end_marker: str):
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def canonical_gold_json(payload):
    if isinstance(payload, dict) and "events" in payload:
        payload = {"events": payload.get("events", [])}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def row_hash_from_eval_row(row):
    payload = row.get("gold_output", row["output"])
    if isinstance(payload, str):
        payload = json.loads(payload)
    digest = hashlib.sha256()
    text_section = extract_section(row["input"], "Text:\n", "\n\nTokens:\n")
    candidate_section = extract_section(row["input"], "Candidate event types:\n", "\n\nSchema cards:\n")
    digest.update(text_section.encode("utf-8"))
    digest.update(b"\n")
    digest.update(candidate_section.encode("utf-8"))
    digest.update(b"\n")
    digest.update(canonical_gold_json(payload).encode("utf-8"))
    return digest.hexdigest()


def row_hash_from_prediction(row):
    digest = hashlib.sha256()
    text_section = extract_section(row["input"], "Text:\n", "\n\nTokens:\n")
    candidate_section = extract_section(row["input"], "Candidate event types:\n", "\n\nSchema cards:\n")
    digest.update(text_section.encode("utf-8"))
    digest.update(b"\n")
    digest.update(candidate_section.encode("utf-8"))
    digest.update(b"\n")
    digest.update(canonical_gold_json(row["gold"]).encode("utf-8"))
    return digest.hexdigest()


def detect_mode(text: str):
    prefix = text.lstrip()
    if prefix.startswith(DIRECT_MODE):
        return "DIRECT"
    if prefix.startswith(COT_MODE):
        return "COT"
    match = re.match(r"<\s*(DIRECT|COT)\s*>", prefix)
    if match:
        return match.group(1)
    return "UNKNOWN"


def summarize(rows):
    if not rows:
        return {
            "num_examples": 0,
            "mode_counts": {},
            "mode_accuracy": 0.0,
        }
    mode_counts = {"DIRECT": 0, "COT": 0, "UNKNOWN": 0}
    oracle_total = 0
    oracle_correct = 0
    cot_tp = cot_fp = cot_fn = 0
    for row in rows:
        pred = row["predicted_mode"]
        mode_counts[pred] = mode_counts.get(pred, 0) + 1
        oracle = row["oracle_mode_label"]
        if oracle in {"DIRECT", "COT"}:
            oracle_total += 1
            oracle_correct += int(pred == oracle)
            cot_tp += int(pred == "COT" and oracle == "COT")
            cot_fp += int(pred == "COT" and oracle != "COT")
            cot_fn += int(pred != "COT" and oracle == "COT")
    cot_precision = cot_tp / (cot_tp + cot_fp) if (cot_tp + cot_fp) else 0.0
    cot_recall = cot_tp / (cot_tp + cot_fn) if (cot_tp + cot_fn) else 0.0
    return {
        "num_examples": len(rows),
        "mode_counts": mode_counts,
        "mode_rates": {k: v / len(rows) for k, v in mode_counts.items()},
        "mode_accuracy": oracle_correct / oracle_total if oracle_total else 0.0,
        "cot_precision": cot_precision,
        "cot_recall": cot_recall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    eval_map = {}
    for row in load_jsonl(Path(args.eval_jsonl)):
        row_hash = row_hash_from_eval_row(row)
        if row_hash in eval_map:
            raise ValueError(f"Duplicate row_hash in eval rows: {row_hash}")
        eval_map[row_hash] = row

    records = []
    for pred_row in load_jsonl(Path(args.predictions_jsonl)):
        row_hash = row_hash_from_prediction(pred_row)
        eval_row = eval_map.get(row_hash)
        if eval_row is None:
            raise ValueError(f"Prediction row missing from eval dataset: {row_hash}")
        generated_payload = pred_row.get("generated_payload", pred_row.get("generated_text", ""))
        records.append(
            {
                "row_hash": row_hash,
                "oracle_mode_label": eval_row.get("meta", {}).get("oracle_mode_label"),
                "predicted_mode": detect_mode(generated_payload),
                "generated_prefix": generated_payload[:80],
                "event_f1": pred_row["event_f1"],
                "argument_f1": pred_row["argument_f1"],
                "trigger_f1": pred_row["trigger_f1"],
            }
        )

    write_json(
        Path(args.output_json),
        {
            "eval_jsonl": args.eval_jsonl,
            "predictions_jsonl": args.predictions_jsonl,
            **summarize(records),
            "sample_rows": records[:10],
        },
    )
    print(json.dumps({"output_json": args.output_json, **summarize(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
