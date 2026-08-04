import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key, score  # noqa: E402
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


METRIC_KEYS = ["trigger_f1", "argument_f1", "event_f1"]


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


def metric_payload(row):
    payload = {key: float(row.get(key, 0.0) or 0.0) for key in METRIC_KEYS}
    payload["score"] = score(row)
    payload["valid_json"] = is_valid(row)
    return payload


def quantile(values, q):
    if not values:
        return None
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def build_labels(direct_rows, reason_rows, label_source, margin, source_split):
    labels = []
    common_keys = sorted(set(direct_rows) & set(reason_rows))
    for idx, key in enumerate(common_keys):
        direct_metric = metric_payload(direct_rows[key])
        reason_metric = metric_payload(reason_rows[key])
        gains = {key_name: reason_metric[key_name] - direct_metric[key_name] for key_name in METRIC_KEYS}
        reason_gain = reason_metric["score"] - direct_metric["score"]
        route_label = "reason" if reason_metric["valid_json"] and reason_gain > margin else "direct"
        labels.append(
            {
                "idx": idx,
                "wnd_id": key,
                "route_label": route_label,
                "label_source": label_source,
                "source_split": source_split,
                "margin": margin,
                "reason_gain": reason_gain,
                "direct_score": direct_metric["score"],
                "reason_score": reason_metric["score"],
                "direct_trigger_f1": direct_metric["trigger_f1"],
                "direct_argument_f1": direct_metric["argument_f1"],
                "direct_event_f1": direct_metric["event_f1"],
                "reason_trigger_f1": reason_metric["trigger_f1"],
                "reason_argument_f1": reason_metric["argument_f1"],
                "reason_event_f1": reason_metric["event_f1"],
                "trigger_gain": gains["trigger_f1"],
                "argument_gain": gains["argument_f1"],
                "event_gain": gains["event_f1"],
                "direct_valid_json": direct_metric["valid_json"],
                "reason_valid_json": reason_metric["valid_json"],
            }
        )
    return labels


def summarize(labels, direct_predictions, reason_predictions, output_jsonl, label_source, margin, source_split):
    total = len(labels)
    reason_rows = [row for row in labels if row["route_label"] == "reason"]
    direct_rows = [row for row in labels if row["route_label"] == "direct"]
    gains = [row["reason_gain"] for row in labels]
    positive_gains = [row["reason_gain"] for row in labels if row["reason_gain"] > margin]
    selected_gains = [row["reason_gain"] for row in reason_rows]
    return {
        "forced_direct_predictions": direct_predictions,
        "forced_reason_predictions": reason_predictions,
        "output_jsonl": output_jsonl,
        "label_source": label_source,
        "source_split": source_split,
        "margin": margin,
        "num_examples": total,
        "direct_count": len(direct_rows),
        "reason_count": len(reason_rows),
        "reason_rate": len(reason_rows) / total if total else 0.0,
        "positive_gain_count": len(positive_gains),
        "positive_gain_rate": len(positive_gains) / total if total else 0.0,
        "direct_valid_json_rate": (
            sum(1 for row in labels if row["direct_valid_json"]) / total if total else 0.0
        ),
        "reason_valid_json_rate": (
            sum(1 for row in labels if row["reason_valid_json"]) / total if total else 0.0
        ),
        "avg_reason_gain": sum(gains) / len(gains) if gains else None,
        "avg_selected_reason_gain": sum(selected_gains) / len(selected_gains) if selected_gains else None,
        "min_selected_reason_gain": min(selected_gains) if selected_gains else None,
        "max_selected_reason_gain": max(selected_gains) if selected_gains else None,
        "reason_gain_quantiles": {
            "p10": quantile(gains, 0.10),
            "p25": quantile(gains, 0.25),
            "p50": quantile(gains, 0.50),
            "p75": quantile(gains, 0.75),
            "p90": quantile(gains, 0.90),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forced_direct_predictions", required=True)
    parser.add_argument("--forced_reason_predictions", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--label_source", required=True)
    parser.add_argument("--source_split", required=True)
    parser.add_argument("--margin", type=float, default=0.0)
    args = parser.parse_args()

    direct_rows = load_prediction_map(Path(args.forced_direct_predictions))
    reason_rows = load_prediction_map(Path(args.forced_reason_predictions))
    labels = build_labels(direct_rows, reason_rows, args.label_source, args.margin, args.source_split)
    summary = summarize(
        labels,
        args.forced_direct_predictions,
        args.forced_reason_predictions,
        args.output_jsonl,
        args.label_source,
        args.margin,
        args.source_split,
    )
    write_jsonl(Path(args.output_jsonl), labels)
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
