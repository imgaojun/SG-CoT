import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import (  # noqa: E402
    merge_metric_dict,
    normalize_events,
    prediction_key,
    prf,
    score,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


BUDGETS = [0.05, 0.10, 0.15, 0.20]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def parse_router_spec(spec: str):
    if "=" not in spec:
        raise ValueError(f"router spec must be NAME=PATH, got: {spec}")
    name, path = spec.split("=", 1)
    if not name:
        raise ValueError(f"empty router spec name in: {spec}")
    return name, Path(path)


def parse_score_router_spec(spec: str):
    # NAME=PATH[:budget]
    if "=" not in spec:
        raise ValueError(f"score-router spec must be NAME=PATH[:budget], got: {spec}")
    name, rest = spec.split("=", 1)
    if not name:
        raise ValueError(f"empty score-router spec name in: {spec}")
    path_text = rest
    budget = None
    if ":" in rest:
        path_text, budget_text = rest.rsplit(":", 1)
        try:
            budget = float(budget_text)
        except ValueError:
            path_text = rest
            budget = None
    return name, Path(path_text), budget


def metric_from_payload(predicted, gold):
    pred_trig, pred_arg, pred_event = normalize_events(predicted or {"events": []})
    gold_trig, gold_arg, gold_event = normalize_events(gold or {"events": []})
    return {
        "trigger": prf(pred_trig, gold_trig),
        "argument": prf(pred_arg, gold_arg),
        "event": prf(pred_event, gold_event),
    }


def row_metric(row):
    return metric_from_payload(row.get("predicted") or row.get("final_predicted"), row.get("gold") or {"events": []})


def summarize_metrics(metric_rows):
    if not metric_rows:
        return {
            "num_examples": 0,
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
        }
    return {
        "num_examples": len(metric_rows),
        "trigger_f1": merge_metric_dict([row["trigger"] for row in metric_rows], "f1"),
        "argument_f1": merge_metric_dict([row["argument"] for row in metric_rows], "f1"),
        "event_f1": merge_metric_dict([row["event"] for row in metric_rows], "f1"),
    }


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_oracle(common_keys, direct_rows, reason_rows, budget):
    scored = []
    for key in common_keys:
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        gain = score(reason_row) - score(direct_row)
        scored.append((gain, key))
    scored.sort(reverse=True)
    cap = round(len(common_keys) * budget)
    reason_keys = {key for gain, key in scored[:cap] if gain > 0}
    metrics = []
    for key in common_keys:
        source = reason_rows[key] if key in reason_keys else direct_rows[key]
        metrics.append(row_metric(source))
    summary = summarize_metrics(metrics)
    summary["reason_count"] = len(reason_keys)
    summary["reason_rate"] = len(reason_keys) / len(common_keys) if common_keys else 0.0
    return summary


def analyze_router(name, router_path, direct_rows, reason_rows):
    router_rows = load_prediction_map(router_path)
    common_keys = sorted(set(router_rows) & set(direct_rows) & set(reason_rows))
    routed_metrics = []
    direct_metrics = []
    reason_metrics = []
    route_counts = {"direct": 0, "reason": 0, "unknown": 0}
    label_counts = {"direct": 0, "reason": 0, "unknown": 0}
    correct = 0
    label_tp = label_fp = label_fn = 0
    helpful_tp = helpful_fp = helpful_fn = 0
    positive_helpful_count = 0
    selected_helpful_gains = []
    selected_harmful_gains = []
    selected_examples = []

    for key in common_keys:
        router_row = router_rows[key]
        pred_route = router_row.get("route_pred") or "unknown"
        if pred_route not in {"direct", "reason"}:
            pred_route = "unknown"
        exec_route = "reason" if pred_route == "reason" else "direct"
        gold_route = router_row.get("gold_route") or "unknown"
        if gold_route not in {"direct", "reason"}:
            gold_route = "unknown"
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        direct_score = score(direct_row)
        reason_score = score(reason_row)
        reason_gain = reason_score - direct_score
        reason_helpful = reason_gain > 0
        if reason_helpful:
            positive_helpful_count += 1

        route_counts[exec_route] += 1
        label_counts[gold_route] += 1
        if pred_route == gold_route:
            correct += 1
        if pred_route == "reason" and gold_route == "reason":
            label_tp += 1
        elif pred_route == "reason" and gold_route != "reason":
            label_fp += 1
        elif pred_route != "reason" and gold_route == "reason":
            label_fn += 1

        if exec_route == "reason" and reason_helpful:
            helpful_tp += 1
            selected_helpful_gains.append(reason_gain)
        elif exec_route == "reason" and not reason_helpful:
            helpful_fp += 1
            selected_harmful_gains.append(reason_gain)
        elif exec_route != "reason" and reason_helpful:
            helpful_fn += 1

        chosen_row = reason_row if exec_route == "reason" else direct_row
        routed_metrics.append(row_metric(chosen_row))
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        if exec_route == "reason":
            selected_examples.append(
                {
                    "wnd_id": key,
                    "gold_route": gold_route,
                    "reason_gain": reason_gain,
                    "direct_argument_f1": direct_row.get("argument_f1", 0.0),
                    "reason_argument_f1": reason_row.get("argument_f1", 0.0),
                    "direct_event_f1": direct_row.get("event_f1", 0.0),
                    "reason_event_f1": reason_row.get("event_f1", 0.0),
                    "text_prefix": (router_row.get("input") or "").splitlines()[1][:180]
                    if len((router_row.get("input") or "").splitlines()) > 1
                    else "",
                }
            )

    total = len(common_keys)
    routed = summarize_metrics(routed_metrics)
    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    oracle = {f"oracle{int(budget * 100)}": compute_oracle(common_keys, direct_rows, reason_rows, budget) for budget in BUDGETS}
    result = {
        "name": name,
        "router_predictions": router_path.as_posix(),
        "num_examples": total,
        "route_accuracy_vs_label": correct / total if total else 0.0,
        "label_reason_count": label_counts["reason"],
        "label_reason_rate": label_counts["reason"] / total if total else 0.0,
        "pred_reason_count": route_counts["reason"],
        "pred_reason_rate": route_counts["reason"] / total if total else 0.0,
        "route_vs_label": route_prf(label_tp, label_fp, label_fn),
        "positive_reason_helpful_count": positive_helpful_count,
        "positive_reason_helpful_rate": positive_helpful_count / total if total else 0.0,
        "route_vs_positive_reason_helpful": route_prf(helpful_tp, helpful_fp, helpful_fn),
        "selected_reason_avg_positive_gain": (
            sum(selected_helpful_gains) / len(selected_helpful_gains) if selected_helpful_gains else 0.0
        ),
        "selected_reason_avg_nonpositive_gain": (
            sum(selected_harmful_gains) / len(selected_harmful_gains) if selected_harmful_gains else 0.0
        ),
        "direct": direct,
        "forced_reason_all": reason,
        "routed": routed,
        "routed_delta_vs_direct": {
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
        },
        "oracle": oracle,
        "selected_reason_examples": sorted(selected_examples, key=lambda row: row["reason_gain"], reverse=True)[:20],
    }
    return result


def analyze_score_router(name, score_path, budget, direct_rows, reason_rows):
    score_rows = load_prediction_map(score_path)
    common_keys = sorted(set(score_rows) & set(direct_rows) & set(reason_rows))
    scored = []
    for key in common_keys:
        delta = score_rows[key].get("delta_direct_minus_reason_route_nll")
        if delta is None:
            delta = float("-inf")
        scored.append((float(delta), key))
    scored.sort(reverse=True)
    if budget is None:
        reason_keys = {key for delta, key in scored if delta > 0}
        budget_label = "argmin"
    else:
        cap = round(len(common_keys) * budget)
        reason_keys = {key for _, key in scored[:cap]}
        budget_label = f"top{int(budget * 100)}"

    router_like_path = score_path
    routed_metrics = []
    direct_metrics = []
    reason_metrics = []
    label_tp = label_fp = label_fn = correct = 0
    helpful_tp = helpful_fp = helpful_fn = positive_helpful_count = 0
    selected_helpful_gains = []
    selected_harmful_gains = []
    selected_examples = []
    label_counts = {"direct": 0, "reason": 0, "unknown": 0}

    for key in common_keys:
        score_row = score_rows[key]
        exec_route = "reason" if key in reason_keys else "direct"
        gold_route = score_row.get("gold_route") or "unknown"
        if gold_route not in {"direct", "reason"}:
            gold_route = "unknown"
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        reason_gain = score(reason_row) - score(direct_row)
        reason_helpful = reason_gain > 0
        if reason_helpful:
            positive_helpful_count += 1
        label_counts[gold_route] += 1
        if exec_route == gold_route:
            correct += 1
        if exec_route == "reason" and gold_route == "reason":
            label_tp += 1
        elif exec_route == "reason" and gold_route != "reason":
            label_fp += 1
        elif exec_route != "reason" and gold_route == "reason":
            label_fn += 1
        if exec_route == "reason" and reason_helpful:
            helpful_tp += 1
            selected_helpful_gains.append(reason_gain)
        elif exec_route == "reason" and not reason_helpful:
            helpful_fp += 1
            selected_harmful_gains.append(reason_gain)
        elif exec_route != "reason" and reason_helpful:
            helpful_fn += 1

        chosen_row = reason_row if exec_route == "reason" else direct_row
        routed_metrics.append(row_metric(chosen_row))
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        if exec_route == "reason":
            selected_examples.append(
                {
                    "wnd_id": key,
                    "gold_route": gold_route,
                    "route_delta_direct_minus_reason_nll": score_row.get("delta_direct_minus_reason_route_nll"),
                    "reason_gain": reason_gain,
                    "direct_argument_f1": direct_row.get("argument_f1", 0.0),
                    "reason_argument_f1": reason_row.get("argument_f1", 0.0),
                    "direct_event_f1": direct_row.get("event_f1", 0.0),
                    "reason_event_f1": reason_row.get("event_f1", 0.0),
                }
            )

    total = len(common_keys)
    routed = summarize_metrics(routed_metrics)
    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    oracle = {f"oracle{int(budget * 100)}": compute_oracle(common_keys, direct_rows, reason_rows, budget) for budget in BUDGETS}
    return {
        "name": f"{name}_{budget_label}",
        "router_predictions": router_like_path.as_posix(),
        "num_examples": total,
        "route_accuracy_vs_label": correct / total if total else 0.0,
        "label_reason_count": label_counts["reason"],
        "label_reason_rate": label_counts["reason"] / total if total else 0.0,
        "pred_reason_count": len(reason_keys),
        "pred_reason_rate": len(reason_keys) / total if total else 0.0,
        "route_vs_label": route_prf(label_tp, label_fp, label_fn),
        "positive_reason_helpful_count": positive_helpful_count,
        "positive_reason_helpful_rate": positive_helpful_count / total if total else 0.0,
        "route_vs_positive_reason_helpful": route_prf(helpful_tp, helpful_fp, helpful_fn),
        "selected_reason_avg_positive_gain": (
            sum(selected_helpful_gains) / len(selected_helpful_gains) if selected_helpful_gains else 0.0
        ),
        "selected_reason_avg_nonpositive_gain": (
            sum(selected_harmful_gains) / len(selected_harmful_gains) if selected_harmful_gains else 0.0
        ),
        "direct": direct,
        "forced_reason_all": reason,
        "routed": routed,
        "routed_delta_vs_direct": {
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
        },
        "oracle": oracle,
        "selected_reason_examples": sorted(
            selected_examples,
            key=lambda row: (
                row["route_delta_direct_minus_reason_nll"]
                if row["route_delta_direct_minus_reason_nll"] is not None
                else float("-inf")
            ),
            reverse=True,
        )[:20],
    }


def markdown_table(results):
    lines = [
        "| router | pred reason | label P/R/F1 | helpful P/R/F1 | direct arg/event | routed arg/event | routed delta | oracle15 arg/event |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        label = row["route_vs_label"]
        helpful = row["route_vs_positive_reason_helpful"]
        direct = row["direct"]
        routed = row["routed"]
        delta = row["routed_delta_vs_direct"]
        oracle15 = row["oracle"]["oracle15"]
        lines.append(
            "| {name} | {pred_rate:.1%} | {lp:.3f}/{lr:.3f}/{lf:.3f} | {hp:.3f}/{hr:.3f}/{hf:.3f} | "
            "{darg:.4f}/{devent:.4f} | {rarg:.4f}/{revent:.4f} | {garg:+.4f}/{gevent:+.4f} | "
            "{oarg:.4f}/{oevent:.4f} |".format(
                name=row["name"],
                pred_rate=row["pred_reason_rate"],
                lp=label["precision"],
                lr=label["recall"],
                lf=label["f1"],
                hp=helpful["precision"],
                hr=helpful["recall"],
                hf=helpful["f1"],
                darg=direct["argument_f1"],
                devent=direct["event_f1"],
                rarg=routed["argument_f1"],
                revent=routed["event_f1"],
                garg=delta["argument_f1"],
                gevent=delta["event_f1"],
                oarg=oracle15["argument_f1"],
                oevent=oracle15["event_f1"],
            )
        )
    return "\n".join(lines)


def render_report(payload):
    lines = [
        "# Adaptive Outcome Router Execution Analysis",
        "",
        "This report simulates executable routing by taking a route-classifier prediction and selecting either the existing forced-direct or forced-reason extraction output for the same sample.",
        "",
        "## Summary",
        "",
        markdown_table(payload["routers"]),
        "",
        "## Reading",
        "",
    ]
    best_arg = max(payload["routers"], key=lambda row: row["routed_delta_vs_direct"]["argument_f1"])
    best_event = max(payload["routers"], key=lambda row: row["routed_delta_vs_direct"]["event_f1"])
    lines.append(
        "- The routed simulations show the executable signal for the evaluated split; compare routed deltas against forced-direct before interpreting route metrics."
    )
    lines.append(
        f"- Best argument gain: `{best_arg['name']}` with `{best_arg['routed_delta_vs_direct']['argument_f1']:+.4f}`."
    )
    lines.append(
        f"- Best event gain: `{best_event['name']}` with `{best_event['routed_delta_vs_direct']['event_f1']:+.4f}`."
    )
    lines.append(
        "- Oracle15 remains the important reference: it shows the same forced-direct/reason experts still contain routeable headroom if the router can capture reason-helpful samples."
    )
    lines.append(
        "- The remaining gap is route precision/recall: this is promising enough for route calibration or loss-balanced router training, but still not strong enough to treat the raw generator as the final adaptive router."
    )
    lines.extend(["", "## Inputs", ""])
    lines.append(f"- forced direct: `{payload['forced_direct_predictions']}`")
    lines.append(f"- forced reason: `{payload['forced_reason_predictions']}`")
    for row in payload["routers"]:
        lines.append(f"- router `{row['name']}`: `{row['router_predictions']}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forced_direct_predictions", required=True)
    parser.add_argument("--forced_reason_predictions", required=True)
    parser.add_argument("--router", action="append", default=[], help="NAME=predictions.jsonl")
    parser.add_argument("--score_router", action="append", default=[], help="NAME=scores.jsonl[:budget]")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()
    if not args.router and not args.score_router:
        parser.error("provide at least one --router or --score_router")

    direct_path = Path(args.forced_direct_predictions)
    reason_path = Path(args.forced_reason_predictions)
    direct_rows = load_prediction_map(direct_path)
    reason_rows = load_prediction_map(reason_path)
    routers = []
    for spec in args.router:
        name, path = parse_router_spec(spec)
        routers.append(analyze_router(name, path, direct_rows, reason_rows))
    for spec in args.score_router:
        name, path, budget = parse_score_router_spec(spec)
        routers.append(analyze_score_router(name, path, budget, direct_rows, reason_rows))
    payload = {
        "forced_direct_predictions": direct_path.as_posix(),
        "forced_reason_predictions": reason_path.as_posix(),
        "routers": routers,
    }
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), render_report(payload))
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "num_routers": len(routers)}, indent=2))


if __name__ == "__main__":
    main()
