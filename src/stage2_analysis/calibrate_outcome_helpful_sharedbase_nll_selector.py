import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key, score  # noqa: E402
from src.stage2_analysis.analyze_adaptive_outcome_router_execution import (  # noqa: E402
    route_prf,
    row_metric,
    summarize_metrics,
    write_json,
    write_text,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DEFAULT_BRANCH = "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux"
DEFAULT_CKPT = "checkpoint-2058"


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def paths_for(split: str, branch: str, checkpoint: str):
    if split == "dev_seen":
        direct = (
            REPO_ROOT
            / f"outputs/stage2_adaptive_runs_user_devpick_frontier/{RUN_PREFIX}_{branch}_full_forced_direct_dev_seen_max512/{checkpoint}/predictions.jsonl"
        )
        reason = (
            REPO_ROOT
            / f"outputs/stage2_adaptive_runs_user_devpick_frontier/{RUN_PREFIX}_{branch}_full_forced_reason_dev_seen_max512/{checkpoint}/predictions.jsonl"
        )
        scores = (
            REPO_ROOT
            / f"outputs/stage2_adaptive_route_likelihood_probe/outcome_helpful_sharedbase_20260515/{branch}/{checkpoint}/dev_seen_scores.jsonl"
        )
        return direct, reason, scores
    direct = (
        REPO_ROOT
        / f"outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_{branch}/{checkpoint}/forced_direct/{split}/predictions.jsonl"
    )
    reason = (
        REPO_ROOT
        / f"outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_{branch}/{checkpoint}/forced_reason/{split}/predictions.jsonl"
    )
    scores = (
        REPO_ROOT
        / f"outputs/stage2_adaptive_route_likelihood_probe/outcome_helpful_sharedbase_formal_20260515/{branch}/{checkpoint}/{split}/scores.jsonl"
    )
    return direct, reason, scores


def split_rows(split: str, branch: str, checkpoint: str):
    direct_path, reason_path, score_path = paths_for(split, branch, checkpoint)
    direct = load_prediction_map(direct_path)
    reason = load_prediction_map(reason_path)
    scores = load_prediction_map(score_path)
    common = sorted(set(direct) & set(reason) & set(scores))
    rows = []
    for key in common:
        direct_row = direct[key]
        reason_row = reason[key]
        score_row = scores[key]
        delta = score_row.get("delta_direct_minus_reason_route_nll")
        rows.append(
            {
                "key": key,
                "direct": direct_row,
                "reason": reason_row,
                "delta": float(delta) if delta is not None else float("-inf"),
                "reason_gain": score(reason_row) - score(direct_row),
            }
        )
    return rows


def selected_keys(rows, rule):
    ranked = sorted(rows, key=lambda row: (row["delta"], row["key"]), reverse=True)
    if rule["kind"] == "topk":
        cap = round(len(rows) * rule["budget"])
        return {row["key"] for row in ranked[:cap]}
    if rule["kind"] == "threshold":
        return {row["key"] for row in rows if row["delta"] >= rule["threshold"]}
    if rule["kind"] == "topk_threshold":
        cap = round(len(rows) * rule["budget"])
        return {row["key"] for row in ranked[:cap] if row["delta"] >= rule["threshold"]}
    raise ValueError(f"unknown rule kind: {rule['kind']}")


def rule_label(rule):
    if rule["kind"] == "topk":
        return f"top{int(rule['budget'] * 100):02d}"
    if rule["kind"] == "threshold":
        return f"delta_ge_{rule['threshold']:.2f}"
    if rule["kind"] == "topk_threshold":
        return f"top{int(rule['budget'] * 100):02d}_delta_ge_{rule['threshold']:.2f}"
    raise ValueError(rule)


def evaluate_rule(split: str, rows, rule):
    selected = selected_keys(rows, rule)
    direct_metrics = []
    reason_metrics = []
    routed_metrics = []
    helpful_tp = helpful_fp = helpful_fn = 0
    positive_helpful_count = 0
    selected_positive = []
    selected_nonpositive = []
    for row in rows:
        is_reason = row["key"] in selected
        helpful = row["reason_gain"] > 0
        if helpful:
            positive_helpful_count += 1
        if is_reason and helpful:
            helpful_tp += 1
            selected_positive.append(row["reason_gain"])
        elif is_reason and not helpful:
            helpful_fp += 1
            selected_nonpositive.append(row["reason_gain"])
        elif (not is_reason) and helpful:
            helpful_fn += 1
        direct_metrics.append(row_metric(row["direct"]))
        reason_metrics.append(row_metric(row["reason"]))
        routed_metrics.append(row_metric(row["reason"] if is_reason else row["direct"]))
    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    routed = summarize_metrics(routed_metrics)
    return {
        "split": split,
        "rule": rule_label(rule),
        "rule_spec": rule,
        "num_examples": len(rows),
        "selected_reason_count": len(selected),
        "selected_reason_rate": len(selected) / len(rows) if rows else 0.0,
        "positive_reason_helpful_count": positive_helpful_count,
        "positive_reason_helpful_rate": positive_helpful_count / len(rows) if rows else 0.0,
        "route_vs_positive_reason_helpful": route_prf(helpful_tp, helpful_fp, helpful_fn),
        "selected_reason_avg_positive_gain": (
            sum(selected_positive) / len(selected_positive) if selected_positive else 0.0
        ),
        "selected_reason_avg_nonpositive_gain": (
            sum(selected_nonpositive) / len(selected_nonpositive) if selected_nonpositive else 0.0
        ),
        "direct": direct,
        "forced_reason_all": reason,
        "routed": routed,
        "routed_delta_vs_direct": {
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
        },
    }


def candidate_rules(dev_rows):
    budgets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rules = [{"kind": "topk", "budget": budget} for budget in budgets]
    thresholds = sorted({round(row["delta"], 4) for row in dev_rows if row["delta"] > float("-inf")}, reverse=True)
    # Keep threshold grid compact and deterministic.
    threshold_grid = []
    for threshold in thresholds:
        rate = sum(1 for row in dev_rows if row["delta"] >= threshold) / len(dev_rows)
        if 0.03 <= rate <= 0.35:
            threshold_grid.append(threshold)
    threshold_grid = threshold_grid[:: max(1, len(threshold_grid) // 24)] or [0.0]
    rules.extend({"kind": "threshold", "threshold": threshold} for threshold in threshold_grid)
    for budget in budgets:
        for threshold in threshold_grid:
            rules.append({"kind": "topk_threshold", "budget": budget, "threshold": threshold})
    dedup = {}
    for rule in rules:
        dedup[rule_label(rule)] = rule
    return list(dedup.values())


def robustness_ok(result):
    delta = result["routed_delta_vs_direct"]
    helpful = result["route_vs_positive_reason_helpful"]
    return (
        delta["argument_f1"] >= 0.0
        and delta["event_f1"] >= 0.005
        and delta["trigger_f1"] >= -0.010
        and helpful["recall"] >= 0.20
        and result["selected_reason_rate"] <= 0.30
    )


def selector_score(result):
    delta = result["routed_delta_vs_direct"]
    helpful = result["route_vs_positive_reason_helpful"]
    return (
        int(robustness_ok(result)),
        min(delta["argument_f1"], 0.02) + min(delta["event_f1"], 0.03),
        helpful["f1"],
        helpful["recall"],
        -max(0.0, -delta["trigger_f1"]),
        -abs(result["selected_reason_rate"] - 0.20),
    )


def render_markdown(payload):
    selected = payload["selected"]
    lines = [
        "# Outcome-Helpful Shared-Base NLL Selector Calibration",
        "",
        "## Scope",
        "",
        f"- branch: `{payload['branch']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- mode: `{payload['mode']}`",
        "",
        "## Decision",
        "",
        f"- selected rule: `{selected['rule']}`",
        f"- selected by dev-only robustness score: `{selected['selector_score']}`",
        f"- robustness pass: `{'yes' if selected['robustness_ok'] else 'no'}`",
        "",
        "## Dev Candidates",
        "",
        "| rule | selected | arg/event delta | trigger delta | helpful P/R/F1 | robustness |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dev_ranked"][:20]:
        delta = row["routed_delta_vs_direct"]
        helpful = row["route_vs_positive_reason_helpful"]
        lines.append(
            "| `{}` | {:.1%} | {:+.4f}/{:+.4f} | {:+.4f} | {:.3f}/{:.3f}/{:.3f} | {} |".format(
                row["rule"],
                row["selected_reason_rate"],
                delta["argument_f1"],
                delta["event_f1"],
                delta["trigger_f1"],
                helpful["precision"],
                helpful["recall"],
                helpful["f1"],
                "yes" if row["robustness_ok"] else "no",
            )
        )
    if payload["formal_selected"]:
        lines.extend(["", "## Formal Transfer", ""])
        lines.append("| split | rule | selected | direct arg/event | routed arg/event | delta arg/event | helpful P/R/F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for split, row in payload["formal_selected"].items():
            delta = row["routed_delta_vs_direct"]
            helpful = row["route_vs_positive_reason_helpful"]
            lines.append(
                "| `{}` | `{}` | {:.1%} | {:.4f}/{:.4f} | {:.4f}/{:.4f} | {:+.4f}/{:+.4f} | {:.3f}/{:.3f}/{:.3f} |".format(
                    split,
                    row["rule"],
                    row["selected_reason_rate"],
                    row["direct"]["argument_f1"],
                    row["direct"]["event_f1"],
                    row["routed"]["argument_f1"],
                    row["routed"]["event_f1"],
                    delta["argument_f1"],
                    delta["event_f1"],
                    helpful["precision"],
                    helpful["recall"],
                    helpful["f1"],
                )
            )
    lines.extend(["", "## Reading", ""])
    lines.append(
        "- This is a no-training calibration: the selected rule uses dev predictions and route-NLL scores only, then transfers unchanged to formal splits."
    )
    lines.append(
        "- Prefer rules that keep argument/event deltas non-negative, preserve helpful recall, and avoid large trigger loss."
    )
    if not selected["robustness_ok"]:
        lines.append(
            "- No candidate passed the stricter no-training robustness screen; treat this as a dev diagnostic rather than a formal-launch recommendation."
        )
    lines.append(
        "- If the selected rule transfers better than raw `top15`, the next training branch should target route ranking rather than extraction formatting."
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument(
        "--dev_only",
        action="store_true",
        help="Only calibrate on dev_seen. Use this while formal outputs are not available yet.",
    )
    parser.add_argument(
        "--output_json",
        default="reports/artifacts/2026-05-15_stage2_adaptive_outcome_helpful_sharedbase_nll_selector_calibration.json",
    )
    parser.add_argument(
        "--output_md",
        default="reports/2026-05-15_stage2_adaptive_outcome_helpful_sharedbase_nll_selector_calibration.md",
    )
    args = parser.parse_args()

    dev_rows = split_rows("dev_seen", args.branch, args.checkpoint)
    rules = candidate_rules(dev_rows)
    dev_results = []
    for rule in rules:
        result = evaluate_rule("dev_seen", dev_rows, rule)
        result["robustness_ok"] = robustness_ok(result)
        result["selector_score"] = selector_score(result)
        dev_results.append(result)
    dev_ranked = sorted(dev_results, key=lambda row: row["selector_score"], reverse=True)
    selected_rule = dev_ranked[0]["rule_spec"]
    formal_selected = {}
    if not args.dev_only:
        formal_selected = {
            split: evaluate_rule(split, split_rows(split, args.branch, args.checkpoint), selected_rule)
            for split in ["test", "test_seen", "test_unseen"]
        }
    payload = {
        "branch": args.branch,
        "checkpoint": args.checkpoint,
        "mode": "dev_only" if args.dev_only else "dev_plus_formal_transfer",
        "selected": dev_ranked[0],
        "dev_ranked": dev_ranked,
        "formal_selected": formal_selected,
        "candidate_count": len(rules),
    }
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), render_markdown(payload))
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "selected": dev_ranked[0]["rule"]}, indent=2))


if __name__ == "__main__":
    main()
