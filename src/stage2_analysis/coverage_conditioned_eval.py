import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def avg(values):
    return sum(values) / len(values) if values else 0.0


def summarize_group(rows):
    if not rows:
        return {
            "num_examples": 0,
            "json_valid_rate": 0.0,
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "avg_latency_sec": 0.0,
            "avg_num_gold_event_types": 0.0,
            "avg_num_candidate_types": 0.0,
            "avg_num_missing_gold_types": 0.0,
        }

    return {
        "num_examples": len(rows),
        "json_valid_rate": avg([1.0 if row["valid_json"] else 0.0 for row in rows]),
        "trigger_f1": avg([row["trigger_f1"] for row in rows]),
        "argument_f1": avg([row["argument_f1"] for row in rows]),
        "event_f1": avg([row["event_f1"] for row in rows]),
        "avg_latency_sec": avg([row["latency_sec"] for row in rows]),
        "avg_num_gold_event_types": avg([len(row["gold_event_types"]) for row in rows]),
        "avg_num_candidate_types": avg([len(row["candidate_types"]) for row in rows]),
        "avg_num_missing_gold_types": avg([len(row["missing_gold_types"]) for row in rows]),
    }


def build_record(dataset_row, pred_row):
    candidate_types = dataset_row["meta"]["candidate_types"]
    gold_event_types = dataset_row["meta"]["gold_event_types"]
    candidate_rank = {event_type: idx + 1 for idx, event_type in enumerate(candidate_types)}
    missing_gold_types = [event_type for event_type in gold_event_types if event_type not in candidate_rank]
    covered = len(missing_gold_types) == 0
    gold_ranks = sorted(candidate_rank[event_type] for event_type in gold_event_types if event_type in candidate_rank)
    return {
        "wnd_id": dataset_row["meta"].get("wnd_id"),
        "doc_id": dataset_row["meta"].get("doc_id"),
        "candidate_types": candidate_types,
        "gold_event_types": gold_event_types,
        "missing_gold_types": missing_gold_types,
        "covered": covered,
        "gold_type_ranks": gold_ranks,
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

    records = []
    for idx, (dataset_row, pred_row) in enumerate(zip(dataset_rows, pred_rows)):
        dataset_input = dataset_row["input"].strip()
        pred_input = pred_row["input"].strip()
        if dataset_input != pred_input:
            raise ValueError(f"Input mismatch at row {idx}.")
        records.append(build_record(dataset_row, pred_row))

    covered_rows = [row for row in records if row["covered"]]
    omitted_rows = [row for row in records if not row["covered"]]

    missing_counter = Counter()
    for row in omitted_rows:
        missing_counter.update(row["missing_gold_types"])

    payload = {
        "label": args.label,
        "dataset_jsonl": args.dataset_jsonl,
        "predictions_jsonl": args.predictions_jsonl,
        "overall": summarize_group(records),
        "covered": summarize_group(covered_rows),
        "omitted": summarize_group(omitted_rows),
        "coverage": {
            "covered_examples": len(covered_rows),
            "omitted_examples": len(omitted_rows),
            "covered_ratio": len(covered_rows) / len(records) if records else 0.0,
            "omitted_ratio": len(omitted_rows) / len(records) if records else 0.0,
        },
        "top_missing_gold_types": missing_counter.most_common(20),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
