#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_formal_unseen_false_positives_20260519 import (  # noqa: E402
    aggregate_sample_rows,
    key_for,
    load_jsonl,
    load_sample_rows,
    metric_dict,
)
from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


SPLITS = ["test_seen", "test_unseen"]
ROUTES = ["direct", "reason"]
SEEDS = [21, 22]
THRESHOLD = 0.25
REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_formal_seen_false_positive_diagnosis.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_formal_seen_false_positive_diagnosis.json"
OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_seen_fp_diagnosis_20260519/sampledk2_seen_fp"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def stdev(values):
    vals = list(values)
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def summarize_values(values):
    vals = sorted(values)
    if not vals:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(vals),
        "median": statistics.median(vals),
        "min": vals[0],
        "max": vals[-1],
    }


def score(metrics):
    return metrics["argument_f1"] + metrics["event_f1"] + 0.25 * metrics["trigger_f1"]


def avg_metric_dict(rows):
    metrics = [metric_dict(row) for row in rows]
    return {
        name: mean(metric[name] for metric in metrics)
        for name in ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json"]
    }


def load_exec_rows(exec_root: Path, split: str, route: str):
    path = exec_root / f"forced_{route}" / split / "predictions.jsonl"
    return {key_for(row): row for row in load_jsonl(path)}


def load_margins(nll_root: Path, split: str):
    path = nll_root / "checkpoint-50" / split / "scores.jsonl"
    return {
        key_for(row): row.get("delta_direct_minus_reason_route_nll")
        for row in load_jsonl(path)
        if key_for(row)
    }


def feature_delta(reason_feat, direct_feat, name):
    return reason_feat.get(name, 0.0) - direct_feat.get(name, 0.0)


def build_cases(config):
    sample_root = repo_path(config["sample_root"])
    nll_root = repo_path(config["output_root"])
    exec_root = REPO / "outputs/stage2_adaptive_route_formal_execution_20260518/sampledk2_ckpt50_margin025"
    cases = []
    for split in SPLITS:
        margins = load_margins(nll_root, split)
        exec_rows = {route: load_exec_rows(exec_root, split, route) for route in ROUTES}
        sample_rows = {
            route: load_sample_rows(sample_root, split, route, SEEDS)
            for route in ROUTES
        }
        keys = set(margins) & set(exec_rows["direct"]) & set(exec_rows["reason"])
        for route in ROUTES:
            keys &= set(sample_rows[route])
        for key in sorted(keys):
            direct_samples = sample_rows["direct"][key]
            reason_samples = sample_rows["reason"][key]
            direct_sample_metrics = avg_metric_dict(direct_samples)
            reason_sample_metrics = avg_metric_dict(reason_samples)
            direct_exec = metric_dict(exec_rows["direct"][key])
            reason_exec = metric_dict(exec_rows["reason"][key])
            direct_feat = aggregate_sample_rows(direct_samples)
            reason_feat = aggregate_sample_rows(reason_samples)
            margin = margins[key]
            event_delta = feature_delta(reason_feat, direct_feat, "event_count_mean")
            selected_margin = margin >= THRESHOLD
            selected_locked = selected_margin and event_delta >= 0
            row = {
                "split": split,
                "case_id": f"{split}::{key}",
                "wnd_id": key,
                "fresh_margin": margin,
                "selected_margin": selected_margin,
                "selected_locked": selected_locked,
                "single_gen_direct": direct_exec,
                "single_gen_reason": reason_exec,
                "single_gen_score_gain": reason_exec["score"] - direct_exec["score"],
                "single_gen_argument_gain": reason_exec["argument_f1"] - direct_exec["argument_f1"],
                "single_gen_event_gain": reason_exec["event_f1"] - direct_exec["event_f1"],
                "single_gen_trigger_gain": reason_exec["trigger_f1"] - direct_exec["trigger_f1"],
                "k2_expected_direct": direct_sample_metrics,
                "k2_expected_reason": reason_sample_metrics,
                "k2_expected_score_gain": reason_sample_metrics["score"] - direct_sample_metrics["score"],
                "meta": exec_rows["direct"][key].get("meta", {}),
            }
            for feat in [
                "event_count_mean",
                "argument_count_mean",
                "trigger_count_mean",
                "event_type_count_mean",
                "output_chars_mean",
                "valid_mean",
                "full_consensus",
                "unique_signature_count",
            ]:
                row[f"reason_minus_direct_{feat}"] = feature_delta(reason_feat, direct_feat, feat)
                row[f"direct_{feat}"] = direct_feat.get(feat, 0.0)
                row[f"reason_{feat}"] = reason_feat.get(feat, 0.0)
            cases.append(row)
    return cases


def selected_category(row):
    if not row["selected_locked"]:
        return "not_selected"
    if row["single_gen_score_gain"] > 0:
        return "selected_helpful"
    if row["single_gen_score_gain"] < 0:
        return "selected_harmful"
    return "selected_neutral"


def summarize_group(rows, split, category):
    selected = [row for row in rows if row["split"] == split and selected_category(row) == category]
    denom = len([row for row in rows if row["split"] == split])
    if not selected:
        return {"split": split, "category": category, "count": 0, "rate": 0.0}
    out = {
        "split": split,
        "category": category,
        "count": len(selected),
        "rate": len(selected) / denom if denom else 0.0,
        "single_gen_score_gain": summarize_values(row["single_gen_score_gain"] for row in selected),
        "k2_expected_score_gain": summarize_values(row["k2_expected_score_gain"] for row in selected),
        "fresh_margin": summarize_values(row["fresh_margin"] for row in selected),
        "single_gen_argument_gain_mean": mean(row["single_gen_argument_gain"] for row in selected),
        "single_gen_event_gain_mean": mean(row["single_gen_event_gain"] for row in selected),
        "single_gen_trigger_gain_mean": mean(row["single_gen_trigger_gain"] for row in selected),
    }
    for feat in FEATURE_NAMES:
        out[f"feature_mean_{feat}"] = mean(row.get(feat, 0.0) for row in selected)
    return out


FEATURE_NAMES = [
    "fresh_margin",
    "k2_expected_score_gain",
    "reason_minus_direct_event_count_mean",
    "reason_minus_direct_argument_count_mean",
    "reason_minus_direct_trigger_count_mean",
    "reason_minus_direct_event_type_count_mean",
    "reason_minus_direct_output_chars_mean",
    "reason_minus_direct_valid_mean",
    "reason_minus_direct_full_consensus",
    "reason_minus_direct_unique_signature_count",
    "direct_event_count_mean",
    "reason_event_count_mean",
    "direct_argument_count_mean",
    "reason_argument_count_mean",
    "direct_unique_signature_count",
    "reason_unique_signature_count",
]


def feature_tests(rows):
    comparisons = []
    for split in ["test_seen", "test_unseen", "test"]:
        split_rows = rows if split == "test" else [row for row in rows if row["split"] == split]
        harmful = [row for row in split_rows if selected_category(row) == "selected_harmful"]
        helpful = [row for row in split_rows if selected_category(row) == "selected_helpful"]
        neutral = [row for row in split_rows if selected_category(row) == "selected_neutral"]
        for name, left, right in [
            ("harmful_vs_helpful", harmful, helpful),
            ("harmful_vs_nonharmful_selected", harmful, helpful + neutral),
        ]:
            for feat in FEATURE_NAMES:
                left_vals = [row.get(feat, 0.0) for row in left]
                right_vals = [row.get(feat, 0.0) for row in right]
                left_mean = mean(left_vals)
                right_mean = mean(right_vals)
                pooled = stdev(left_vals + right_vals)
                comparisons.append(
                    {
                        "split": split,
                        "comparison": name,
                        "feature": feat,
                        "left_count": len(left),
                        "right_count": len(right),
                        "left_mean": left_mean,
                        "right_mean": right_mean,
                        "diff_left_minus_right": left_mean - right_mean,
                        "abs_standardized_diff": abs(left_mean - right_mean) / pooled if pooled else 0.0,
                    }
                )
    comparisons.sort(key=lambda row: (row["split"], row["comparison"], -row["abs_standardized_diff"], row["feature"]))
    return comparisons


def policy_reason(row, policy):
    if policy == "locked_event_guard":
        return row["fresh_margin"] >= THRESHOLD and row["reason_minus_direct_event_count_mean"] >= 0
    if policy == "margin_only":
        return row["fresh_margin"] >= THRESHOLD
    if policy == "locked_and_nonnegative_k2_gain":
        return (
            row["fresh_margin"] >= THRESHOLD
            and row["reason_minus_direct_event_count_mean"] >= 0
            and row["k2_expected_score_gain"] >= 0
        )
    if policy == "locked_and_no_argument_count_drop":
        return (
            row["fresh_margin"] >= THRESHOLD
            and row["reason_minus_direct_event_count_mean"] >= 0
            and row["reason_minus_direct_argument_count_mean"] >= 0
        )
    if policy == "locked_and_no_trigger_count_drop":
        return (
            row["fresh_margin"] >= THRESHOLD
            and row["reason_minus_direct_event_count_mean"] >= 0
            and row["reason_minus_direct_trigger_count_mean"] >= 0
        )
    if policy == "locked_and_reason_not_more_variable":
        return (
            row["fresh_margin"] >= THRESHOLD
            and row["reason_minus_direct_event_count_mean"] >= 0
            and row["reason_minus_direct_unique_signature_count"] <= 0
        )
    raise KeyError(policy)


POLICIES = [
    "margin_only",
    "locked_event_guard",
    "locked_and_nonnegative_k2_gain",
    "locked_and_no_argument_count_drop",
    "locked_and_no_trigger_count_drop",
    "locked_and_reason_not_more_variable",
]


def summarize_policy(rows, split, policy, source):
    split_rows = rows if split == "test" else [row for row in rows if row["split"] == split]
    direct_rows = []
    routed_rows = []
    selected_gains = []
    helpful = set()
    selected = set()
    for row in split_rows:
        direct = row[f"{source}_direct"]
        reason = row[f"{source}_reason"]
        gain = score(reason) - score(direct)
        direct_rows.append(direct)
        if gain > 0:
            helpful.add(row["case_id"])
        if policy_reason(row, policy):
            selected.add(row["case_id"])
            selected_gains.append(gain)
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)
    direct_avg = {
        metric: mean(row[metric] for row in direct_rows)
        for metric in ["argument_f1", "event_f1", "trigger_f1", "score"]
    }
    routed_avg = {
        metric: mean(row[metric] for row in routed_rows)
        for metric in ["argument_f1", "event_f1", "trigger_f1", "score"]
    }
    tp = len(selected & helpful)
    fp = len(selected - helpful)
    fn = len(helpful - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "split": split,
        "policy": policy,
        "source": source,
        "num_examples": len(split_rows),
        "pred_reason_count": len(selected),
        "pred_reason_rate": len(selected) / len(split_rows) if split_rows else 0.0,
        "selected_reason_score_gain_mean": mean(selected_gains),
        "selected_reason_harm_rate": mean(1.0 if gain < 0 else 0.0 for gain in selected_gains),
        "route_vs_helpful": {"precision": precision, "recall": recall, "f1": f1},
        "routed_minus_direct": {
            metric: routed_avg[metric] - direct_avg[metric]
            for metric in ["argument_f1", "event_f1", "trigger_f1", "score"]
        },
    }


def compact_case(row):
    meta = row.get("meta", {})
    return {
        "case_id": row["case_id"],
        "wnd_id": row["wnd_id"],
        "split": row["split"],
        "category": selected_category(row),
        "fresh_margin": row["fresh_margin"],
        "k2_expected_score_gain": row["k2_expected_score_gain"],
        "single_gen_score_gain": row["single_gen_score_gain"],
        "single_gen_argument_gain": row["single_gen_argument_gain"],
        "single_gen_event_gain": row["single_gen_event_gain"],
        "single_gen_trigger_gain": row["single_gen_trigger_gain"],
        "candidate_types": meta.get("candidate_types"),
        "gold_event_types": meta.get("gold_event_types"),
        "doc_id": meta.get("doc_id"),
        "source_part": meta.get("source_part"),
        "features": {feat: row.get(feat) for feat in FEATURE_NAMES},
    }


def render_group_table(rows):
    lines = [
        "| split | category | count | rate | single gain | K2 gain | margin | A/E/T gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"test_seen": 0, "test_unseen": 1}
    cat_order = {"selected_harmful": 0, "selected_helpful": 1, "selected_neutral": 2, "not_selected": 3}
    for row in sorted(rows, key=lambda r: (order[r["split"]], cat_order[r["category"]])):
        if row["count"] == 0:
            lines.append(
                f"| `{row['split']}` | `{row['category']}` | 0 | {pct(row['rate'])} | "
                "NA | NA | NA | NA/NA/NA |"
            )
            continue
        lines.append(
            f"| `{row['split']}` | `{row['category']}` | {row['count']} | {pct(row['rate'])} | "
            f"{signed(row['single_gen_score_gain']['mean'])} | {signed(row['k2_expected_score_gain']['mean'])} | "
            f"{fmt(row['fresh_margin']['mean'])} | "
            f"{signed(row['single_gen_argument_gain_mean'])}/{signed(row['single_gen_event_gain_mean'])}/{signed(row['single_gen_trigger_gain_mean'])} |"
        )
    return "\n".join(lines)


def render_feature_table(rows, split, comparison, limit=10):
    selected = [
        row for row in rows
        if row["split"] == split and row["comparison"] == comparison
    ][:limit]
    lines = [
        "| feature | harmful mean | reference mean | diff | std diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| `{row['feature']}` | {fmt(row['left_mean'])} | {fmt(row['right_mean'])} | "
            f"{signed(row['diff_left_minus_right'])} | {fmt(row['abs_standardized_diff'])} |"
        )
    return "\n".join(lines)


def render_policy_table(rows, source):
    lines = [
        "| split | policy | reason rate | score delta | A/E/T delta | selected gain | harm rate | P/R/F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"test": 0, "test_seen": 1, "test_unseen": 2}
    for row in sorted([r for r in rows if r["source"] == source], key=lambda r: (order[r["split"]], r["policy"])):
        prf = row["route_vs_helpful"]
        delta = row["routed_minus_direct"]
        lines.append(
            f"| `{row['split']}` | `{row['policy']}` | {pct(row['pred_reason_rate'])} | "
            f"{signed(delta['score'])} | {signed(delta['argument_f1'])}/{signed(delta['event_f1'])}/{signed(delta['trigger_f1'])} | "
            f"{signed(row['selected_reason_score_gain_mean'])} | {pct(row['selected_reason_harm_rate'])} | "
            f"{fmt(prf['precision'])}/{fmt(prf['recall'])}/{fmt(prf['f1'])} |"
        )
    return "\n".join(lines)


def render_case_table(rows, split, category, limit=14):
    selected = [row for row in rows if row["split"] == split and row["category"] == category]
    selected.sort(key=lambda row: row["single_gen_score_gain"])
    lines = [
        "| wnd_id | single gain | K2 gain | margin | event/arg/trig count delta | gold types |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in selected[:limit]:
        feats = row["features"]
        gold = ",".join(row.get("gold_event_types") or [])
        lines.append(
            f"| `{row['wnd_id']}` | {signed(row['single_gen_score_gain'])} | {signed(row['k2_expected_score_gain'])} | "
            f"{fmt(row['fresh_margin'])} | "
            f"{signed(feats['reason_minus_direct_event_count_mean'])}/{signed(feats['reason_minus_direct_argument_count_mean'])}/{signed(feats['reason_minus_direct_trigger_count_mean'])} | "
            f"`{gold}` |"
        )
    return "\n".join(lines)


def render_report(payload):
    policy_rows = payload["policy_evaluations"]
    locked_single = next(
        row for row in policy_rows
        if row["source"] == "single_gen" and row["split"] == "test_seen" and row["policy"] == "locked_event_guard"
    )
    best_single = max(
        [row for row in policy_rows if row["source"] == "single_gen" and row["split"] == "test" and row["policy"] != "margin_only"],
        key=lambda row: row["routed_minus_direct"]["score"],
    )
    lines = [
        "# Sampled K2 Formal Seen False-Positive Diagnosis",
        "",
        "This report diagnoses why the fresh seedpair21/22 locked event-count guard has positive K2 expected deltas but negative single-generation execution on `test_seen`.",
        "",
        f"- output root: `{payload['output_root']}`",
        "",
        "## Selected Case Groups",
        "",
        render_group_table(payload["group_summaries"]),
        "",
        "## Separating Features",
        "",
        "### Test Seen: Harmful vs Helpful",
        "",
        render_feature_table(payload["feature_tests"], "test_seen", "harmful_vs_helpful"),
        "",
        "### Test Seen: Harmful vs Non-Harmful Selected",
        "",
        render_feature_table(payload["feature_tests"], "test_seen", "harmful_vs_nonharmful_selected"),
        "",
        "## Guard Probes: Single Generation",
        "",
        render_policy_table(policy_rows, "single_gen"),
        "",
        "## Guard Probes: K2 Expected",
        "",
        render_policy_table(policy_rows, "k2_expected"),
        "",
        "## Worst Selected Seen Cases",
        "",
        render_case_table(payload["cases"], "test_seen", "selected_harmful"),
        "",
        "## Reading",
        "",
        f"- Locked guard single-gen `test_seen` score delta is `{locked_single['routed_minus_direct']['score']:+.4f}` with selected harm rate `{locked_single['selected_reason_harm_rate']:.1%}`.",
        f"- Best non-margin guard by aggregate single-gen score is `{best_single['policy']}` with test score delta `{best_single['routed_minus_direct']['score']:+.4f}` and reason rate `{best_single['pred_reason_rate']:.1%}`.",
        "- Treat these guards as diagnostic only; they are selected after formal outcomes and require locked validation before training a selector.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['report_json']}`",
        f"- cases JSONL: `{payload['cases_jsonl']}`",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    config = load_json(repo_path(args.config))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = build_cases(config)
    compact_cases = [compact_case(row) for row in cases]
    cases_jsonl = OUTPUT_ROOT / "cases.jsonl"
    write_jsonl(cases_jsonl, compact_cases)
    categories = ["selected_harmful", "selected_helpful", "selected_neutral", "not_selected"]
    group_summaries = [
        summarize_group(cases, split, category)
        for split in SPLITS
        for category in categories
    ]
    feats = feature_tests(cases)
    policy_evaluations = [
        summarize_policy(cases, split, policy, source)
        for source in ["single_gen", "k2_expected"]
        for split in ["test", "test_seen", "test_unseen"]
        for policy in POLICIES
    ]
    payload = {
        "config": repo_path(args.config).as_posix(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "seeds": SEEDS,
        "threshold": THRESHOLD,
        "num_cases": len(cases),
        "group_summaries": group_summaries,
        "feature_tests": feats,
        "policy_evaluations": policy_evaluations,
        "cases": compact_cases,
        "cases_jsonl": cases_jsonl.as_posix(),
        "report_md": REPORT_MD.as_posix(),
        "report_json": REPORT_JSON.as_posix(),
    }
    write_json(REPORT_JSON, payload)
    write_json(OUTPUT_ROOT / "summary.json", payload)
    write_text(REPORT_MD, render_report(payload))
    print(json.dumps({"report_md": REPORT_MD.as_posix(), "report_json": REPORT_JSON.as_posix()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
