import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key, score  # noqa: E402
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def is_valid(row):
    return bool(row.get("valid_final_json", row.get("valid_json", False)))


def metric_score(row):
    return {
        "trigger_f1": float(row.get("trigger_f1", 0.0) or 0.0),
        "argument_f1": float(row.get("argument_f1", 0.0) or 0.0),
        "event_f1": float(row.get("event_f1", 0.0) or 0.0),
        "score": score(row),
        "valid_json": is_valid(row),
    }


def build_labels(
    direct_rows,
    reason_rows,
    reason_rate_cap: float,
    margin: float,
    label_source: str,
    miner_checkpoint: str,
):
    candidates = []
    common_keys = sorted(set(direct_rows) & set(reason_rows))
    for key in common_keys:
        direct_metric = metric_score(direct_rows[key])
        reason_metric = metric_score(reason_rows[key])
        reason_gain = reason_metric["score"] - direct_metric["score"]
        if reason_metric["valid_json"] and reason_gain > margin:
            candidates.append(
                {
                    "wnd_id": key,
                    "reason_gain": reason_gain,
                    "direct_metric": direct_metric,
                    "reason_metric": reason_metric,
                }
            )

    candidates.sort(key=lambda row: (row["reason_gain"], row["wnd_id"]), reverse=True)
    cap = round(len(common_keys) * reason_rate_cap)
    reason_ids = {row["wnd_id"] for row in candidates[:cap]}

    labels = []
    for key in common_keys:
        direct_metric = metric_score(direct_rows[key])
        reason_metric = metric_score(reason_rows[key])
        reason_gain = reason_metric["score"] - direct_metric["score"]
        labels.append(
            {
                "wnd_id": key,
                "route_label": "reason" if key in reason_ids else "direct",
                "label_source": label_source,
                "reason_rate_cap": reason_rate_cap,
                "margin": margin,
                "reason_gain": reason_gain,
                "direct_score": direct_metric["score"],
                "reason_score": reason_metric["score"],
                "direct_argument_f1": direct_metric["argument_f1"],
                "reason_argument_f1": reason_metric["argument_f1"],
                "direct_event_f1": direct_metric["event_f1"],
                "reason_event_f1": reason_metric["event_f1"],
                "direct_trigger_f1": direct_metric["trigger_f1"],
                "reason_trigger_f1": reason_metric["trigger_f1"],
                "direct_valid_json": direct_metric["valid_json"],
                "reason_valid_json": reason_metric["valid_json"],
                "miner_checkpoint": miner_checkpoint,
            }
        )
    return labels, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forced_direct_predictions", required=True)
    parser.add_argument("--forced_reason_predictions", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--reason_rate_cap", type=float, required=True)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--label_source", required=True)
    parser.add_argument("--miner_checkpoint", required=True)
    args = parser.parse_args()

    direct_rows = load_prediction_map(Path(args.forced_direct_predictions))
    reason_rows = load_prediction_map(Path(args.forced_reason_predictions))
    labels, candidates = build_labels(
        direct_rows,
        reason_rows,
        args.reason_rate_cap,
        args.margin,
        args.label_source,
        args.miner_checkpoint,
    )
    write_jsonl(Path(args.output_jsonl), labels)
    reason_rows_out = [row for row in labels if row["route_label"] == "reason"]
    positive_gains = [row["reason_gain"] for row in labels if row["reason_gain"] > args.margin]
    summary = {
        "forced_direct_predictions": args.forced_direct_predictions,
        "forced_reason_predictions": args.forced_reason_predictions,
        "output_jsonl": args.output_jsonl,
        "label_source": args.label_source,
        "miner_checkpoint": args.miner_checkpoint,
        "reason_rate_cap": args.reason_rate_cap,
        "margin": args.margin,
        "num_examples": len(labels),
        "candidate_positive_count": len(candidates),
        "candidate_positive_rate": len(candidates) / len(labels) if labels else 0.0,
        "reason_count": len(reason_rows_out),
        "direct_count": len(labels) - len(reason_rows_out),
        "reason_rate": len(reason_rows_out) / len(labels) if labels else 0.0,
        "min_selected_reason_gain": min([row["reason_gain"] for row in reason_rows_out], default=None),
        "max_selected_reason_gain": max([row["reason_gain"] for row in reason_rows_out], default=None),
        "avg_selected_reason_gain": (
            sum(row["reason_gain"] for row in reason_rows_out) / len(reason_rows_out)
            if reason_rows_out
            else None
        ),
        "avg_positive_reason_gain": sum(positive_gains) / len(positive_gains) if positive_gains else None,
    }
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
