import argparse
import copy
import hashlib
import json
import random
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
            raise ValueError(f"Duplicate row_hash in router scores: {path} -> {row_hash}")
        mapping[row_hash] = row
    return mapping


def select_cot_hashes(score_rows, strategy: str, cot_rate: float, seed: int | None):
    row_hashes = sorted(score_rows)
    if strategy == "direct_only":
        return set()
    if strategy == "cot_only":
        return set(row_hashes)

    if cot_rate is None:
        raise ValueError(f"{strategy} requires --cot_rate")
    count = int(round(len(row_hashes) * cot_rate))
    count = max(0, min(len(row_hashes), count))

    if strategy == "random":
        rng = random.Random(seed)
        return set(rng.sample(row_hashes, count)) if count else set()

    if strategy == "heuristic":
        ranked = sorted(
            score_rows.values(),
            key=lambda item: (item[CURRENT_SCORE_KEY], item["row_hash"]),
            reverse=True,
        )
        return {row["row_hash"] for row in ranked[:count]}

    raise ValueError(f"Unsupported strategy: {strategy}")


def metric_average(rows, key: str):
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router_scores", required=True)
    parser.add_argument("--direct_predictions", required=True)
    parser.add_argument("--cot_predictions", required=True)
    parser.add_argument("--strategy", choices=["direct_only", "cot_only", "heuristic", "random"], required=True)
    parser.add_argument("--cot_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--score_key", default="ambiguity_score")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    global CURRENT_SCORE_KEY
    CURRENT_SCORE_KEY = args.score_key

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

    cot_hashes = select_cot_hashes(score_rows, args.strategy, args.cot_rate, args.seed)
    merged = []
    source_counts = {"direct": 0, "cot": 0}

    for row_hash in sorted(common_hashes):
        source = "cot" if row_hash in cot_hashes else "direct"
        selected = cot_rows[row_hash] if source == "cot" else direct_rows[row_hash]
        score_row = score_rows[row_hash]
        item = copy.deepcopy(selected)
        item["row_hash"] = row_hash
        item["router_strategy"] = args.strategy
        item["router_source"] = source
        item["router_ambiguity_score"] = score_row["ambiguity_score"]
        item["router_schema_score"] = score_row["schema_score"]
        item["router_text_score"] = score_row["text_score"]
        item["router_score_key"] = args.score_key
        item["router_score_value"] = score_row.get(args.score_key)
        item["router_top_confusion_pair"] = score_row.get("top_confusion_pair")
        merged.append(item)
        source_counts[source] += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "merged_predictions.jsonl", "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_examples": len(merged),
        "json_valid_rate": metric_average(merged, "valid_json"),
        "avg_latency_sec": metric_average(merged, "latency_sec"),
        "trigger_f1": metric_average(merged, "trigger_f1"),
        "argument_f1": metric_average(merged, "argument_f1"),
        "event_f1": metric_average(merged, "event_f1"),
        "cot_route_rate": source_counts["cot"] / len(merged) if merged else 0.0,
        "cot_selected_count": source_counts["cot"],
        "direct_selected_count": source_counts["direct"],
        "avg_ambiguity_score_selected_cot": (
            sum(row["router_ambiguity_score"] for row in merged if row["router_source"] == "cot") / source_counts["cot"]
            if source_counts["cot"]
            else 0.0
        ),
        "avg_ambiguity_score_selected_direct": (
            sum(row["router_ambiguity_score"] for row in merged if row["router_source"] == "direct") / source_counts["direct"]
            if source_counts["direct"]
            else 0.0
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "route_manifest.json",
        {
            "router_scores": args.router_scores,
            "direct_predictions": args.direct_predictions,
            "cot_predictions": args.cot_predictions,
            "strategy": args.strategy,
            "cot_rate": args.cot_rate,
            "seed": args.seed,
            "score_key": args.score_key,
            "selected_row_hashes": sorted(cot_hashes),
            "summary": summary,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
