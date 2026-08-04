import argparse
import hashlib
import json
import sys
from collections import defaultdict
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


def canonical_gold_json(payload):
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
    digest.update(canonical_gold_json(row["gold"]).encode("utf-8"))
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


def load_score_rows(path: Path):
    rows = load_jsonl(path)
    mapping = {}
    for row in rows:
        row_hash = row["row_hash"]
        if row_hash in mapping:
            raise ValueError(f"Duplicate row_hash in scores: {path} -> {row_hash}")
        mapping[row_hash] = row
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router_scores", required=True)
    parser.add_argument("--direct_predictions", required=True)
    parser.add_argument("--cot_predictions", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_support", type=int, default=5)
    args = parser.parse_args()

    score_rows = load_score_rows(Path(args.router_scores))
    direct_rows = load_prediction_map(Path(args.direct_predictions))
    cot_rows = load_prediction_map(Path(args.cot_predictions))
    common_hashes = set(score_rows) & set(direct_rows) & set(cot_rows)
    if common_hashes != set(score_rows) or common_hashes != set(direct_rows) or common_hashes != set(cot_rows):
        raise ValueError(
            "Router scores and prediction rows do not align perfectly: "
            f"score_only={len(set(score_rows) - common_hashes)} "
            f"direct_only={len(set(direct_rows) - common_hashes)} "
            f"cot_only={len(set(cot_rows) - common_hashes)}"
        )

    pair_stats = defaultdict(
        lambda: {
            "n": 0,
            "cot_better_count": 0,
            "event_win_count": 0,
            "arg_better_count": 0,
            "delta_event_sum": 0.0,
            "delta_argument_sum": 0.0,
            "delta_trigger_sum": 0.0,
        }
    )
    global_cot_better_count = 0

    for row_hash in common_hashes:
        score_row = score_rows[row_hash]
        pair = score_row.get("canonical_top_pair")
        if not pair:
            continue
        key = " || ".join(pair)
        direct = direct_rows[row_hash]
        cot = cot_rows[row_hash]
        d_tuple = (direct["event_f1"], direct["argument_f1"], direct["trigger_f1"])
        c_tuple = (cot["event_f1"], cot["argument_f1"], cot["trigger_f1"])
        cot_better = c_tuple > d_tuple
        event_win = cot["event_f1"] > direct["event_f1"]
        arg_better = cot["argument_f1"] > direct["argument_f1"]

        record = pair_stats[key]
        record["n"] += 1
        record["cot_better_count"] += int(cot_better)
        record["event_win_count"] += int(event_win)
        record["arg_better_count"] += int(arg_better)
        record["delta_event_sum"] += cot["event_f1"] - direct["event_f1"]
        record["delta_argument_sum"] += cot["argument_f1"] - direct["argument_f1"]
        record["delta_trigger_sum"] += cot["trigger_f1"] - direct["trigger_f1"]
        global_cot_better_count += int(cot_better)

    global_cot_better_rate = global_cot_better_count / len(common_hashes) if common_hashes else 0.0
    payload_pairs = {}
    for key, row in sorted(pair_stats.items()):
        n = row["n"]
        cot_better_rate = row["cot_better_count"] / n
        event_win_rate = row["event_win_count"] / n
        arg_better_rate = row["arg_better_count"] / n
        if n >= args.min_support:
            pair_prior_score = max(0.0, (cot_better_rate - global_cot_better_rate) / max(1e-8, 1.0 - global_cot_better_rate))
        else:
            pair_prior_score = 0.0
        payload_pairs[key] = {
            "n": n,
            "cot_better_rate": cot_better_rate,
            "event_win_rate": event_win_rate,
            "arg_better_rate": arg_better_rate,
            "mean_delta_event": row["delta_event_sum"] / n,
            "mean_delta_argument": row["delta_argument_sum"] / n,
            "mean_delta_trigger": row["delta_trigger_sum"] / n,
            "pair_prior_score": pair_prior_score,
        }

    write_json(
        Path(args.output_json),
        {
            "router_scores": args.router_scores,
            "direct_predictions": args.direct_predictions,
            "cot_predictions": args.cot_predictions,
            "min_support": args.min_support,
            "global_cot_better_rate": global_cot_better_rate,
            "pair_stats": payload_pairs,
        },
    )
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "num_pairs": len(payload_pairs),
                "global_cot_better_rate": global_cot_better_rate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
