import argparse
import json
from pathlib import Path


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
            "avg_worst_gold_rank": 0.0,
            "avg_best_gold_rank": 0.0,
        }
    return {
        "num_examples": len(rows),
        "json_valid_rate": avg([1.0 if row["valid_json"] else 0.0 for row in rows]),
        "trigger_f1": avg([row["trigger_f1"] for row in rows]),
        "argument_f1": avg([row["argument_f1"] for row in rows]),
        "event_f1": avg([row["event_f1"] for row in rows]),
        "avg_latency_sec": avg([row["latency_sec"] for row in rows]),
        "avg_num_gold_event_types": avg([len(row["gold_event_types"]) for row in rows]),
        "avg_worst_gold_rank": avg([row["worst_gold_rank"] for row in rows]),
        "avg_best_gold_rank": avg([row["best_gold_rank"] for row in rows]),
    }


def bucket_for_rank(rank: int):
    if rank == 1:
        return "1"
    if 2 <= rank <= 3:
        return "2-3"
    if 4 <= rank <= 5:
        return "4-5"
    if 6 <= rank <= 10:
        return "6-10"
    return "other"


def build_record(dataset_row, pred_row):
    candidate_types = dataset_row["meta"]["candidate_types"]
    gold_event_types = dataset_row["meta"]["gold_event_types"]
    candidate_rank = {event_type: idx + 1 for idx, event_type in enumerate(candidate_types)}
    missing_gold_types = [event_type for event_type in gold_event_types if event_type not in candidate_rank]
    if missing_gold_types:
        return None

    ranks = sorted(candidate_rank[event_type] for event_type in gold_event_types)
    best_gold_rank = min(ranks)
    worst_gold_rank = max(ranks)
    return {
        "wnd_id": dataset_row["meta"].get("wnd_id"),
        "doc_id": dataset_row["meta"].get("doc_id"),
        "gold_event_types": gold_event_types,
        "candidate_types": candidate_types,
        "gold_type_ranks": ranks,
        "best_gold_rank": best_gold_rank,
        "worst_gold_rank": worst_gold_rank,
        "bucket": bucket_for_rank(worst_gold_rank),
        "valid_json": pred_row["valid_json"],
        "latency_sec": pred_row["latency_sec"],
        "trigger_f1": pred_row["trigger_f1"],
        "argument_f1": pred_row["argument_f1"],
        "event_f1": pred_row["event_f1"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    dataset_rows = load_jsonl(Path(args.dataset_jsonl))
    pred_rows = load_jsonl(Path(args.predictions_jsonl))
    if len(dataset_rows) != len(pred_rows):
        raise ValueError(
            f"Length mismatch: dataset has {len(dataset_rows)} rows but predictions have {len(pred_rows)} rows."
        )

    covered_rows = []
    omitted_rows = 0
    for idx, (dataset_row, pred_row) in enumerate(zip(dataset_rows, pred_rows)):
        dataset_input = dataset_row["input"].strip()
        pred_input = pred_row["input"].strip()
        if dataset_input != pred_input:
            raise ValueError(f"Input mismatch at row {idx}.")
        record = build_record(dataset_row, pred_row)
        if record is None:
            omitted_rows += 1
            continue
        covered_rows.append(record)

    buckets = ["1", "2-3", "4-5", "6-10"]
    bucket_payload = {}
    for bucket in buckets:
        rows = [row for row in covered_rows if row["bucket"] == bucket]
        bucket_payload[bucket] = summarize(rows)

    payload = {
        "label": args.label,
        "dataset_jsonl": args.dataset_jsonl,
        "predictions_jsonl": args.predictions_jsonl,
        "num_total_examples": len(dataset_rows),
        "num_covered_examples": len(covered_rows),
        "num_omitted_examples": omitted_rows,
        "covered_ratio": len(covered_rows) / len(dataset_rows) if dataset_rows else 0.0,
        "omitted_ratio": omitted_rows / len(dataset_rows) if dataset_rows else 0.0,
        "covered_overall": summarize(covered_rows),
        "buckets": bucket_payload,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
