import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

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


def prediction_key(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or meta.get("doc_id")


def load_sample_rows(root: Path):
    grouped = defaultdict(list)
    paths = sorted(root.glob("seed-*/predictions.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no seed-*/predictions.jsonl files under {root}")
    for path in paths:
        for row in load_jsonl(path):
            key = prediction_key(row)
            if key is None:
                raise ValueError(f"missing prediction key in {path}")
            grouped[key].append(row)
    return grouped, paths


def metric_row(row):
    trigger_f1 = float(row.get("trigger_f1", 0.0) or 0.0)
    argument_f1 = float(row.get("argument_f1", 0.0) or 0.0)
    event_f1 = float(row.get("event_f1", 0.0) or 0.0)
    return {
        "trigger_f1": trigger_f1,
        "argument_f1": argument_f1,
        "event_f1": event_f1,
        "score": argument_f1 + event_f1 + 0.25 * trigger_f1,
        "valid_json": bool(row.get("valid_final_json", row.get("valid_json", False))),
        "sample_id": row.get("sample_id"),
        "sample_seed": row.get("sample_seed"),
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def mean_metric(metrics, key):
    return mean([row[key] for row in metrics])


def classify(direct_metrics, reason_metrics, args):
    direct_scores = [row["score"] for row in direct_metrics]
    reason_scores = [row["score"] for row in reason_metrics]
    direct_triggers = [row["trigger_f1"] for row in direct_metrics]
    reason_triggers = [row["trigger_f1"] for row in reason_metrics]

    pairs = [(reason_row, direct_row) for reason_row in reason_metrics for direct_row in direct_metrics]
    p_win = (
        mean([1.0 if reason_row["score"] >= direct_row["score"] + args.pair_score_margin else 0.0 for reason_row, direct_row in pairs])
        if pairs
        else 0.0
    )
    p_trigger_noharm = (
        mean(
            [
                1.0 if reason_row["trigger_f1"] >= direct_row["trigger_f1"] - args.trigger_harm_tolerance else 0.0
                for reason_row, direct_row in pairs
            ]
        )
        if pairs
        else 0.0
    )
    reason_valid_rate = mean([1.0 if row["valid_json"] else 0.0 for row in reason_metrics])
    direct_valid_rate = mean([1.0 if row["valid_json"] else 0.0 for row in direct_metrics])
    direct_mean_score = mean(direct_scores)
    reason_mean_score = mean(reason_scores)
    mean_gain = reason_mean_score - direct_mean_score

    stable_reason = (
        reason_valid_rate >= args.reason_valid_rate_min
        and mean_gain >= args.mean_gain_min
        and p_win >= args.p_win_min
        and p_trigger_noharm >= args.p_trigger_noharm_min
    )
    stable_direct = (
        reason_valid_rate < args.direct_reason_valid_rate_max
        or mean_gain <= args.direct_mean_gain_max
        or p_win <= args.direct_p_win_max
        or p_trigger_noharm < args.direct_p_trigger_noharm_min
    )
    if stable_reason:
        label = "stable_reason"
        route_label = "reason"
    elif stable_direct:
        label = "stable_direct"
        route_label = "direct"
    else:
        label = "ambiguous"
        route_label = "ambiguous"

    return {
        "utility_label": label,
        "route_label": route_label,
        "direct_count": len(direct_metrics),
        "reason_count": len(reason_metrics),
        "direct_valid_rate": direct_valid_rate,
        "reason_valid_rate": reason_valid_rate,
        "direct_mean_score": direct_mean_score,
        "reason_mean_score": reason_mean_score,
        "mean_gain": mean_gain,
        "p_win": p_win,
        "p_trigger_noharm": p_trigger_noharm,
        "direct_mean_trigger_f1": mean(direct_triggers),
        "reason_mean_trigger_f1": mean(reason_triggers),
        "direct_mean_argument_f1": mean_metric(direct_metrics, "argument_f1"),
        "reason_mean_argument_f1": mean_metric(reason_metrics, "argument_f1"),
        "direct_mean_event_f1": mean_metric(direct_metrics, "event_f1"),
        "reason_mean_event_f1": mean_metric(reason_metrics, "event_f1"),
    }


def aggregate_policy(rows):
    if not rows:
        return {
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "score": 0.0,
        }
    return {
        "trigger_f1": mean([row["trigger_f1"] for row in rows]),
        "argument_f1": mean([row["argument_f1"] for row in rows]),
        "event_f1": mean([row["event_f1"] for row in rows]),
        "score": mean([row["score"] for row in rows]),
    }


def summarize(labels, direct_grouped, reason_grouped):
    count_by_label = defaultdict(int)
    routed_metrics = []
    direct_baseline_metrics = []
    reason_all_metrics = []
    stable_reason_gains = []
    incomplete = 0
    for label in labels:
        key = label["wnd_id"]
        count_by_label[label["utility_label"]] += 1
        direct_metrics = [metric_row(row) for row in direct_grouped[key]]
        reason_metrics = [metric_row(row) for row in reason_grouped[key]]
        if label["direct_count"] != label["expected_samples_per_route"] or label["reason_count"] != label["expected_samples_per_route"]:
            incomplete += 1
        direct_mean = {
            "trigger_f1": label["direct_mean_trigger_f1"],
            "argument_f1": label["direct_mean_argument_f1"],
            "event_f1": label["direct_mean_event_f1"],
            "score": label["direct_mean_score"],
        }
        reason_mean = {
            "trigger_f1": label["reason_mean_trigger_f1"],
            "argument_f1": label["reason_mean_argument_f1"],
            "event_f1": label["reason_mean_event_f1"],
            "score": label["reason_mean_score"],
        }
        direct_baseline_metrics.append(direct_mean)
        reason_all_metrics.append(reason_mean)
        routed_metrics.append(reason_mean if label["utility_label"] == "stable_reason" else direct_mean)
        if label["utility_label"] == "stable_reason":
            stable_reason_gains.append(label["mean_gain"])

    direct_summary = aggregate_policy(direct_baseline_metrics)
    reason_summary = aggregate_policy(reason_all_metrics)
    routed_summary = aggregate_policy(routed_metrics)
    deltas = {key: routed_summary[key] - direct_summary[key] for key in routed_summary}
    total = len(labels)
    return {
        "num_examples": total,
        "stable_reason_count": count_by_label["stable_reason"],
        "stable_direct_count": count_by_label["stable_direct"],
        "ambiguous_count": count_by_label["ambiguous"],
        "stable_reason_rate": count_by_label["stable_reason"] / total if total else 0.0,
        "stable_direct_rate": count_by_label["stable_direct"] / total if total else 0.0,
        "ambiguous_rate": count_by_label["ambiguous"] / total if total else 0.0,
        "incomplete_sample_count": incomplete,
        "stable_reason_mean_gain": mean(stable_reason_gains) if stable_reason_gains else None,
        "direct_baseline_mean": direct_summary,
        "reason_all_mean": reason_summary,
        "sampled_expected_routed_mean": routed_summary,
        "sampled_expected_routed_minus_direct": deltas,
    }


def build_labels(args):
    direct_grouped, direct_paths = load_sample_rows(Path(args.direct_root))
    reason_grouped, reason_paths = load_sample_rows(Path(args.reason_root))
    common_keys = sorted(set(direct_grouped) & set(reason_grouped))
    if not common_keys:
        raise ValueError("direct and reason sampled predictions have no shared keys")
    missing_direct = sorted(set(reason_grouped) - set(direct_grouped))
    missing_reason = sorted(set(direct_grouped) - set(reason_grouped))

    labels = []
    for key in common_keys:
        direct_metrics = [metric_row(row) for row in direct_grouped[key]]
        reason_metrics = [metric_row(row) for row in reason_grouped[key]]
        label = classify(direct_metrics, reason_metrics, args)
        label.update(
            {
                "wnd_id": key,
                "label_source": args.label_source,
                "expected_samples_per_route": args.expected_samples_per_route,
                "pair_score_margin": args.pair_score_margin,
                "trigger_harm_tolerance": args.trigger_harm_tolerance,
                "reason_valid_rate_min": args.reason_valid_rate_min,
                "mean_gain_min": args.mean_gain_min,
                "p_win_min": args.p_win_min,
                "p_trigger_noharm_min": args.p_trigger_noharm_min,
                "direct_reason_valid_rate_max": args.direct_reason_valid_rate_max,
                "direct_mean_gain_max": args.direct_mean_gain_max,
                "direct_p_win_max": args.direct_p_win_max,
                "direct_p_trigger_noharm_min": args.direct_p_trigger_noharm_min,
                "direct_sample_ids": sorted(str(row.get("sample_id")) for row in direct_grouped[key]),
                "reason_sample_ids": sorted(str(row.get("sample_id")) for row in reason_grouped[key]),
            }
        )
        labels.append(label)

    summary = summarize(labels, direct_grouped, reason_grouped)
    summary.update(
        {
            "label_source": args.label_source,
            "direct_root": args.direct_root,
            "reason_root": args.reason_root,
            "direct_seed_prediction_files": [path.as_posix() for path in direct_paths],
            "reason_seed_prediction_files": [path.as_posix() for path in reason_paths],
            "missing_direct_count": len(missing_direct),
            "missing_reason_count": len(missing_reason),
            "missing_direct_preview": missing_direct[:10],
            "missing_reason_preview": missing_reason[:10],
            "output_jsonl": args.output_jsonl,
            "summary_json": args.summary_json,
        }
    )
    return labels, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_root", required=True)
    parser.add_argument("--reason_root", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--label_source", default="sampled_counterfactual_utility_k8")
    parser.add_argument("--expected_samples_per_route", type=int, default=8)
    parser.add_argument("--pair_score_margin", type=float, default=0.2)
    parser.add_argument("--trigger_harm_tolerance", type=float, default=0.02)
    parser.add_argument("--reason_valid_rate_min", type=float, default=0.875)
    parser.add_argument("--mean_gain_min", type=float, default=0.35)
    parser.add_argument("--p_win_min", type=float, default=0.70)
    parser.add_argument("--p_trigger_noharm_min", type=float, default=0.75)
    parser.add_argument("--direct_reason_valid_rate_max", type=float, default=0.75)
    parser.add_argument("--direct_mean_gain_max", type=float, default=-0.20)
    parser.add_argument("--direct_p_win_max", type=float, default=0.25)
    parser.add_argument("--direct_p_trigger_noharm_min", type=float, default=0.50)
    args = parser.parse_args()

    labels, summary = build_labels(args)
    write_jsonl(Path(args.output_jsonl), labels)
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
