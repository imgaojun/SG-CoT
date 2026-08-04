import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_outcome_router_execution import (  # noqa: E402
    compute_oracle,
    load_prediction_map,
    route_prf,
    row_metric,
    summarize_metrics,
    write_json,
    write_text,
)
from src.stage2_analysis.analyze_adaptive_hardness_boundary import score  # noqa: E402


BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]
METRIC_KEYS = ["trigger_f1", "argument_f1", "event_f1"]


def resolve_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def metric_delta(left, right):
    return {key: left[key] - right[key] for key in METRIC_KEYS}


def finite_score(value):
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def common_inputs(split_cfg):
    direct_path = resolve_path(split_cfg["direct_predictions"])
    reason_path = resolve_path(split_cfg["reason_predictions"])
    scores_path = resolve_path(split_cfg["scores"])
    direct_rows = load_prediction_map(direct_path)
    reason_rows = load_prediction_map(reason_path)
    score_rows = load_prediction_map(scores_path)
    common_keys = sorted(set(direct_rows) & set(reason_rows) & set(score_rows))
    if not common_keys:
        raise ValueError(f"no common keys for {direct_path}, {reason_path}, {scores_path}")
    return {
        "paths": {
            "direct_predictions": direct_path.as_posix(),
            "reason_predictions": reason_path.as_posix(),
            "scores": scores_path.as_posix(),
        },
        "direct_rows": direct_rows,
        "reason_rows": reason_rows,
        "score_rows": score_rows,
        "common_keys": common_keys,
        "counts": {
            "direct": len(direct_rows),
            "reason": len(reason_rows),
            "scores": len(score_rows),
            "common": len(common_keys),
        },
    }


def row_wnd_id(key, direct_row, score_row):
    meta = direct_row.get("meta") or {}
    return meta.get("wnd_id") or score_row.get("wnd_id") or key


def analyze_threshold(split_name, split_cfg, threshold, score_field, return_decisions=False):
    inputs = common_inputs(split_cfg)
    direct_rows = inputs["direct_rows"]
    reason_rows = inputs["reason_rows"]
    score_rows = inputs["score_rows"]
    common_keys = inputs["common_keys"]

    reason_keys = set()
    direct_metrics = []
    reason_metrics = []
    routed_metrics = []
    helpful_tp = helpful_fp = helpful_fn = 0
    label_tp = label_fp = label_fn = label_correct = 0
    label_counts = {"direct": 0, "reason": 0, "unknown": 0}
    positive_helpful_count = 0
    selected_positive_gains = []
    selected_nonpositive_gains = []
    selected_examples = []
    decisions = []

    for key in common_keys:
        score_row = score_rows[key]
        delta = finite_score(score_row.get(score_field))
        use_reason = delta is not None and delta >= threshold
        if use_reason:
            reason_keys.add(key)

        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        direct_score = score(direct_row)
        reason_score = score(reason_row)
        reason_gain = reason_score - direct_score
        reason_helpful = reason_gain > 0
        if reason_helpful:
            positive_helpful_count += 1

        gold_route = score_row.get("gold_route") or "unknown"
        if gold_route not in label_counts:
            gold_route = "unknown"
        exec_route = "reason" if use_reason else "direct"
        label_counts[gold_route] += 1
        if exec_route == gold_route:
            label_correct += 1
        if exec_route == "reason" and gold_route == "reason":
            label_tp += 1
        elif exec_route == "reason" and gold_route != "reason":
            label_fp += 1
        elif exec_route != "reason" and gold_route == "reason":
            label_fn += 1

        if exec_route == "reason" and reason_helpful:
            helpful_tp += 1
            selected_positive_gains.append(reason_gain)
        elif exec_route == "reason" and not reason_helpful:
            helpful_fp += 1
            selected_nonpositive_gains.append(reason_gain)
        elif exec_route != "reason" and reason_helpful:
            helpful_fn += 1

        chosen_row = reason_row if use_reason else direct_row
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        routed_metrics.append(row_metric(chosen_row))

        if use_reason:
            selected_examples.append(
                {
                    "key": key,
                    "wnd_id": row_wnd_id(key, direct_row, score_row),
                    "score_delta": delta,
                    "gold_route": gold_route,
                    "reason_gain": reason_gain,
                    "direct_trigger_f1": direct_row.get("trigger_f1", 0.0),
                    "direct_argument_f1": direct_row.get("argument_f1", 0.0),
                    "direct_event_f1": direct_row.get("event_f1", 0.0),
                    "reason_trigger_f1": reason_row.get("trigger_f1", 0.0),
                    "reason_argument_f1": reason_row.get("argument_f1", 0.0),
                    "reason_event_f1": reason_row.get("event_f1", 0.0),
                }
            )

        if return_decisions:
            decisions.append(
                {
                    "split": split_name,
                    "key": key,
                    "wnd_id": row_wnd_id(key, direct_row, score_row),
                    "route": exec_route,
                    "score_delta": delta,
                    "threshold": threshold,
                    "gold_route": gold_route,
                    "reason_helpful": reason_helpful,
                    "reason_gain": reason_gain,
                    "direct_score": direct_score,
                    "reason_score": reason_score,
                    "direct_trigger_f1": direct_row.get("trigger_f1", 0.0),
                    "direct_argument_f1": direct_row.get("argument_f1", 0.0),
                    "direct_event_f1": direct_row.get("event_f1", 0.0),
                    "reason_trigger_f1": reason_row.get("trigger_f1", 0.0),
                    "reason_argument_f1": reason_row.get("argument_f1", 0.0),
                    "reason_event_f1": reason_row.get("event_f1", 0.0),
                }
            )

    total = len(common_keys)
    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    routed = summarize_metrics(routed_metrics)
    oracle = {
        f"oracle{int(budget * 100)}": compute_oracle(common_keys, direct_rows, reason_rows, budget)
        for budget in BUDGETS
    }
    result = {
        "split": split_name,
        "threshold": threshold,
        "score_field": score_field,
        "num_examples": total,
        "input_counts": inputs["counts"],
        "input_paths": inputs["paths"],
        "label_reason_count": label_counts["reason"],
        "label_reason_rate": label_counts["reason"] / total if total else 0.0,
        "pred_reason_count": len(reason_keys),
        "pred_reason_rate": len(reason_keys) / total if total else 0.0,
        "route_accuracy_vs_label": label_correct / total if total else 0.0,
        "route_vs_label": route_prf(label_tp, label_fp, label_fn),
        "positive_reason_helpful_count": positive_helpful_count,
        "positive_reason_helpful_rate": positive_helpful_count / total if total else 0.0,
        "route_vs_positive_reason_helpful": route_prf(helpful_tp, helpful_fp, helpful_fn),
        "selected_reason_avg_positive_gain": (
            sum(selected_positive_gains) / len(selected_positive_gains) if selected_positive_gains else 0.0
        ),
        "selected_reason_avg_nonpositive_gain": (
            sum(selected_nonpositive_gains) / len(selected_nonpositive_gains) if selected_nonpositive_gains else 0.0
        ),
        "direct": direct,
        "forced_reason_all": reason,
        "routed": routed,
        "routed_delta_vs_direct": metric_delta(routed, direct),
        "forced_reason_delta_vs_direct": metric_delta(reason, direct),
        "oracle": oracle,
        "selected_reason_examples": sorted(
            selected_examples,
            key=lambda row: (row["score_delta"] if row["score_delta"] is not None else float("-inf")),
            reverse=True,
        )[:20],
    }
    if return_decisions:
        result["decisions"] = decisions
    return result


def candidate_thresholds(split_cfg, score_field):
    inputs = common_inputs(split_cfg)
    values = []
    for key in inputs["common_keys"]:
        delta = finite_score(inputs["score_rows"][key].get(score_field))
        if delta is not None:
            values.append(delta)
    if not values:
        raise ValueError(f"no finite `{score_field}` values in {inputs['paths']['scores']}")
    thresholds = set(values)
    thresholds.add(0.0)
    thresholds.add(max(values) + 1e-9)
    thresholds.add(min(values) - 1e-9)
    return sorted(thresholds, reverse=True)


def passes_constraints(result, constraints):
    delta = result["routed_delta_vs_direct"]
    return (
        result["pred_reason_rate"] <= constraints.get("max_reason_rate", 1.0)
        and delta["argument_f1"] >= constraints.get("min_argument_delta", -1.0)
        and delta["event_f1"] >= constraints.get("min_event_delta", -1.0)
        and delta["trigger_f1"] >= constraints.get("min_trigger_delta", -1.0)
    )


def selection_key(result, constraints):
    delta = result["routed_delta_vs_direct"]
    helpful = result["route_vs_positive_reason_helpful"]
    passed = passes_constraints(result, constraints)
    objective = (
        delta["event_f1"],
        delta["argument_f1"],
        delta["trigger_f1"],
        helpful["f1"],
        helpful["precision"],
        -result["pred_reason_rate"],
    )
    if passed:
        return (1,) + objective
    # Fallback is deliberately conservative if no threshold satisfies constraints.
    deficit = 0.0
    deficit += max(0.0, result["pred_reason_rate"] - constraints.get("max_reason_rate", 1.0))
    deficit += max(0.0, constraints.get("min_argument_delta", -1.0) - delta["argument_f1"])
    deficit += max(0.0, constraints.get("min_event_delta", -1.0) - delta["event_f1"])
    deficit += max(0.0, constraints.get("min_trigger_delta", -1.0) - delta["trigger_f1"])
    return (0, -deficit) + objective


def select_dev_threshold(config):
    selection_cfg = config.get("selection", {})
    score_field = selection_cfg.get("score_field", "delta_direct_minus_reason_route_nll")
    dev_cfg = config["dev"]
    candidates = []
    for threshold in candidate_thresholds(dev_cfg, score_field):
        result = analyze_threshold("dev_seen", dev_cfg, threshold, score_field, return_decisions=False)
        result["passes_constraints"] = passes_constraints(result, selection_cfg)
        result["selection_tuple"] = selection_key(result, selection_cfg)
        candidates.append(result)
    selected = max(candidates, key=lambda row: row["selection_tuple"])
    candidates.sort(key=lambda row: row["selection_tuple"], reverse=True)
    return selected, candidates


def slim_result(result):
    kept = dict(result)
    kept.pop("decisions", None)
    kept.pop("selection_tuple", None)
    return kept


def format_float(value, digits=4):
    return f"{value:.{digits}f}"


def format_signed(value, digits=4):
    return f"{value:+.{digits}f}"


def format_percent(value):
    return f"{value:.1%}"


def metrics_triplet(row):
    return "{}/{}/{}".format(
        format_float(row["argument_f1"]),
        format_float(row["event_f1"]),
        format_float(row["trigger_f1"]),
    )


def delta_triplet(row):
    return "{}/{}/{}".format(
        format_signed(row["argument_f1"]),
        format_signed(row["event_f1"]),
        format_signed(row["trigger_f1"]),
    )


def helpful_triplet(row):
    return "{}/{}/{}".format(
        format_float(row["precision"], 3),
        format_float(row["recall"], 3),
        format_float(row["f1"], 3),
    )


def split_summary_table(results):
    lines = [
        "| split | n | reason rate | direct A/E/T | reason A/E/T | routed A/E/T | routed delta A/E/T | helpful P/R/F1 | oracle15 A/E |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        oracle15 = row["oracle"]["oracle15"]
        lines.append(
            "| {split} | {n} | {rate} | {direct} | {reason} | {routed} | {delta} | {helpful} | {oracle_arg}/{oracle_event} |".format(
                split=row["split"],
                n=row["num_examples"],
                rate=format_percent(row["pred_reason_rate"]),
                direct=metrics_triplet(row["direct"]),
                reason=metrics_triplet(row["forced_reason_all"]),
                routed=metrics_triplet(row["routed"]),
                delta=delta_triplet(row["routed_delta_vs_direct"]),
                helpful=helpful_triplet(row["route_vs_positive_reason_helpful"]),
                oracle_arg=format_float(oracle15["argument_f1"]),
                oracle_event=format_float(oracle15["event_f1"]),
            )
        )
    return "\n".join(lines)


def candidate_table(candidates, limit=20):
    lines = [
        "| rank | threshold | pass | reason rate | routed delta A/E/T | helpful P/R/F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(candidates[:limit], start=1):
        lines.append(
            "| {idx} | {threshold:.6f} | {passed} | {rate} | {delta} | {helpful} |".format(
                idx=idx,
                threshold=row["threshold"],
                passed="yes" if row.get("passes_constraints") else "no",
                rate=format_percent(row["pred_reason_rate"]),
                delta=delta_triplet(row["routed_delta_vs_direct"]),
                helpful=helpful_triplet(row["route_vs_positive_reason_helpful"]),
            )
        )
    return "\n".join(lines)


def render_report(payload):
    selected = payload["dev_selection"]["selected"]
    formal_results = payload["formal_results"]
    dev_result = payload["dev_selection"]["selected_full_result"]
    lines = [
        "# Modular Dual-Expert Threshold Diagnostic",
        "",
        "This no-training diagnostic separates the execution roles: a direct expert supplies all direct-route outputs, a reason expert supplies all reason-route outputs, and a route-NLL score chooses when to switch from direct to reason.",
        "",
        "The threshold is selected on dev only. No fixed top-k budget is used as the main rule.",
        "",
        "## Setup",
        "",
        f"- experiment id: `{payload['id']}`",
        f"- direct expert: `{payload['direct_expert']['name']}`",
        f"- reason expert: `{payload['reason_expert']['name']}`",
        f"- router score: `{payload['router_score']['name']}`",
        f"- rule: reason iff `{payload['selection']['score_field']} >= threshold`",
        "",
        "## Selected Dev Threshold",
        "",
        f"- threshold: `{selected['threshold']:.6f}`",
        f"- passes constraints: `{selected['passes_constraints']}`",
        f"- dev reason rate: `{format_percent(selected['pred_reason_rate'])}`",
        f"- dev routed delta A/E/T: `{delta_triplet(selected['routed_delta_vs_direct'])}`",
        f"- dev helpful P/R/F1: `{helpful_triplet(selected['route_vs_positive_reason_helpful'])}`",
        "",
        "## Routed Execution Results",
        "",
        split_summary_table([dev_result] + formal_results),
        "",
        "## Top Dev Candidate Thresholds",
        "",
        candidate_table(payload["dev_selection"]["top_candidates"]),
        "",
        "## Inputs",
        "",
        f"- config: `{payload['config_path']}`",
        f"- output root: `{payload['output_root']}`",
        f"- report json: `{payload['report_json']}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_config(config_path)
    output_cfg = config["outputs"]
    output_root = resolve_path(output_cfg["root"])
    report_json = resolve_path(output_cfg["report_json"])
    report_md = resolve_path(output_cfg["report_md"])
    score_field = config.get("selection", {}).get("score_field", "delta_direct_minus_reason_route_nll")

    selected, candidates = select_dev_threshold(config)
    threshold = selected["threshold"]

    dev_result = analyze_threshold("dev_seen", config["dev"], threshold, score_field, return_decisions=True)
    dev_result["passes_constraints"] = passes_constraints(dev_result, config.get("selection", {}))
    write_jsonl(output_root / "dev_seen_decisions.jsonl", dev_result["decisions"])
    dev_result_slim = slim_result(dev_result)

    formal_results = []
    for split_name, split_cfg in config.get("formal", {}).items():
        result = analyze_threshold(split_name, split_cfg, threshold, score_field, return_decisions=True)
        write_jsonl(output_root / f"{split_name}_decisions.jsonl", result["decisions"])
        formal_results.append(slim_result(result))

    top_candidates = [slim_result(row) for row in candidates[:50]]
    payload = {
        "id": config["id"],
        "config_path": config_path.as_posix(),
        "output_root": output_root.as_posix(),
        "report_json": report_json.as_posix(),
        "report_md": report_md.as_posix(),
        "direct_expert": config["direct_expert"],
        "reason_expert": config["reason_expert"],
        "router_score": config["router_score"],
        "selection": config["selection"],
        "dev_selection": {
            "selected": slim_result(selected),
            "selected_full_result": dev_result_slim,
            "top_candidates": top_candidates,
            "num_candidates": len(candidates),
        },
        "formal_results": formal_results,
    }
    write_json(report_json, payload)
    write_json(output_root / "summary.json", payload)
    write_json(
        output_root / "selected_threshold.json",
        {
            "threshold": threshold,
            "score_field": score_field,
            "selected_dev_reason_rate": selected["pred_reason_rate"],
            "selected_dev_routed_delta_vs_direct": selected["routed_delta_vs_direct"],
            "passes_constraints": selected["passes_constraints"],
        },
    )
    write_jsonl(output_root / "dev_seen_candidate_thresholds.jsonl", [slim_result(row) for row in candidates])
    write_text(report_md, render_report(payload))
    print(
        json.dumps(
            {
                "id": config["id"],
                "threshold": threshold,
                "dev_reason_rate": selected["pred_reason_rate"],
                "dev_delta": selected["routed_delta_vs_direct"],
                "report_md": report_md.as_posix(),
                "report_json": report_json.as_posix(),
                "output_root": output_root.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
