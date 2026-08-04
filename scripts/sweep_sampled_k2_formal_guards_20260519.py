#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_formal_unseen_false_positives_20260519 import (  # noqa: E402
    aggregate_sample_rows,
    load_sample_rows,
)
from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


SPLITS = ["test_seen", "test_unseen"]
ROUTES = ["direct", "reason"]
SEEDS = [17, 18, 19, 20]
METRICS = ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json_rate"]
OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_guard_sweep_20260519/sampledk2_guard_sweep"
REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_formal_guard_sweep.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_formal_guard_sweep.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def key_for(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or row.get("wnd_id")


def score_value(row):
    return row.get("argument_f1", 0.0) + row.get("event_f1", 0.0) + 0.25 * row.get("trigger_f1", 0.0)


def metric_dict(row):
    return {
        "argument_f1": row.get("argument_f1", 0.0),
        "event_f1": row.get("event_f1", 0.0),
        "trigger_f1": row.get("trigger_f1", 0.0),
        "score": score_value(row),
        "valid_json_rate": 1.0 if row.get("valid_final_json", row.get("valid_json")) else 0.0,
    }


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def avg_metric_dict(rows):
    if not rows:
        return {metric: 0.0 for metric in METRICS}
    return {metric: mean(row[metric] for row in rows) for metric in METRICS}


def load_sampled_metrics(sample_root: Path, split: str, route: str):
    grouped = defaultdict(list)
    for seed in SEEDS:
        path = sample_root / split / route / f"seed-{seed}" / "predictions.jsonl"
        for row in load_jsonl(path):
            key = key_for(row)
            if key:
                grouped[key].append(metric_dict(row))
    out = {}
    for key, rows in grouped.items():
        if len(rows) != len(SEEDS):
            raise ValueError(f"{split}/{route}/{key}: expected {len(SEEDS)} rows, got {len(rows)}")
        out[key] = avg_metric_dict(rows)
    return out


def load_execution_metrics(execution_root: Path, split: str, route: str):
    path = execution_root / f"forced_{route}" / split / "predictions.jsonl"
    return {key_for(row): metric_dict(row) for row in load_jsonl(path) if key_for(row)}


def load_margins(root: Path, split: str):
    path = root / "checkpoint-50" / split / "scores.jsonl"
    return {
        key_for(row): row.get("delta_direct_minus_reason_route_nll")
        for row in load_jsonl(path)
        if key_for(row)
    }


def build_cases(config):
    sample_root = repo_path(config["sample_root"])
    execution_root = repo_path(config["execution_root"])
    old_root = repo_path(config["route_nll_roots"]["seedpair17_18"])
    new_root = repo_path(config["route_nll_roots"]["seedpair19_20"])
    cases = {}
    for split in SPLITS:
        old_margins = load_margins(old_root, split)
        new_margins = load_margins(new_root, split)
        sampled_metrics = {
            route: load_sampled_metrics(sample_root, split, route)
            for route in ROUTES
        }
        exec_metrics = {
            route: load_execution_metrics(execution_root, split, route)
            for route in ROUTES
        }
        sample_features = {
            route: load_sample_rows(sample_root, split, route, SEEDS)
            for route in ROUTES
        }
        keys = set(old_margins) & set(new_margins)
        for route in ROUTES:
            keys &= set(sampled_metrics[route])
            keys &= set(exec_metrics[route])
            keys &= set(sample_features[route])
        split_cases = []
        for key in sorted(keys):
            direct_feat = aggregate_sample_rows(sample_features["direct"][key])
            reason_feat = aggregate_sample_rows(sample_features["reason"][key])
            old = old_margins[key]
            new = new_margins[key]
            case = {
                "split": split,
                "key": key,
                "case_id": f"{split}::{key}",
                "old_margin": old,
                "new_margin": new,
                "avg_margin": (old + new) / 2,
                "min_margin": min(old, new),
                "max_margin": max(old, new),
                "margin_gap_new_minus_old": new - old,
                "abs_margin_gap": abs(new - old),
                "k2_expected_direct": sampled_metrics["direct"][key],
                "k2_expected_reason": sampled_metrics["reason"][key],
                "single_gen_execution_direct": exec_metrics["direct"][key],
                "single_gen_execution_reason": exec_metrics["reason"][key],
            }
            for feat in [
                "event_count_mean",
                "argument_count_mean",
                "trigger_count_mean",
                "output_chars_mean",
                "full_consensus",
                "unique_signature_count",
            ]:
                case[f"reason_minus_direct_{feat}"] = reason_feat.get(feat, 0.0) - direct_feat.get(feat, 0.0)
            split_cases.append(case)
        cases[split] = split_cases
    return cases


def guard_reason(case, policy):
    old = case["old_margin"]
    new = case["new_margin"]
    avg = case["avg_margin"]
    min_margin = case["min_margin"]
    gap = case["margin_gap_new_minus_old"]
    abs_gap = case["abs_margin_gap"]
    reason_args = case["reason_minus_direct_argument_count_mean"]
    reason_events = case["reason_minus_direct_event_count_mean"]
    reason_cons = case["reason_minus_direct_full_consensus"]

    if policy == "old_main":
        return old >= 0.25
    if policy == "new_main":
        return new >= 0.25
    if policy == "both_main":
        return old >= 0.25 and new >= 0.25
    if policy == "avg_main":
        return avg >= 0.25
    if policy == "either_main":
        return old >= 0.25 or new >= 0.25
    if policy == "old_and_new_negative":
        return old >= 0.25 and new < 0.0
    if policy == "old_and_new_below_010":
        return old >= 0.25 and new < 0.10
    if policy == "old_and_new_below_025":
        return old >= 0.25 and new < 0.25
    if policy == "old_and_gap_negative":
        return old >= 0.25 and gap < 0.0
    if policy == "old_and_gap_le_minus_025":
        return old >= 0.25 and gap <= -0.25
    if policy == "old_and_abs_gap_ge_025_new_lower":
        return old >= 0.25 and gap < 0.0 and abs_gap >= 0.25
    if policy == "old_and_avg_below_025":
        return old >= 0.25 and avg < 0.25
    if policy == "old_and_min_below_0":
        return old >= 0.25 and min_margin < 0.0
    if policy == "old_and_reason_more_args":
        return old >= 0.25 and reason_args > 0
    if policy == "old_and_reason_not_fewer_events":
        return old >= 0.25 and reason_events >= 0
    if policy == "old_and_reason_consensus_not_higher":
        return old >= 0.25 and reason_cons <= 0
    if policy == "avg_and_reason_more_args":
        return avg >= 0.25 and reason_args > 0
    if policy == "both_and_reason_not_fewer_events":
        return old >= 0.25 and new >= 0.25 and reason_events >= 0
    raise KeyError(policy)


POLICIES = [
    "old_main",
    "new_main",
    "both_main",
    "avg_main",
    "either_main",
    "old_and_new_negative",
    "old_and_new_below_010",
    "old_and_new_below_025",
    "old_and_gap_negative",
    "old_and_gap_le_minus_025",
    "old_and_abs_gap_ge_025_new_lower",
    "old_and_avg_below_025",
    "old_and_min_below_0",
    "old_and_reason_more_args",
    "old_and_reason_not_fewer_events",
    "old_and_reason_consensus_not_higher",
    "avg_and_reason_more_args",
    "both_and_reason_not_fewer_events",
]


def avg_metrics(rows):
    if not rows:
        return {metric: 0.0 for metric in METRICS}
    return {metric: mean(row[metric] for row in rows) for metric in METRICS}


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize(cases, split, source, policy):
    direct_rows = []
    reason_rows = []
    routed_rows = []
    selected = []
    selected_set = set()
    helpful = set()
    for case in cases:
        direct = case[f"{source}_direct"]
        reason = case[f"{source}_reason"]
        direct_rows.append(direct)
        reason_rows.append(reason)
        gain = reason["score"] - direct["score"]
        if gain > 0:
            helpful.add(case["case_id"])
        if guard_reason(case, policy):
            selected_set.add(case["case_id"])
            selected.append(gain)
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)
    direct_summary = avg_metrics(direct_rows)
    reason_summary = avg_metrics(reason_rows)
    routed_summary = avg_metrics(routed_rows)
    tp = len(selected_set & helpful)
    fp = len(selected_set - helpful)
    fn = len(helpful - selected_set)
    return {
        "split": split,
        "source": source,
        "policy": policy,
        "num_examples": len(cases),
        "pred_reason_count": len(selected_set),
        "pred_reason_rate": len(selected_set) / len(cases) if cases else 0.0,
        "selected_reason_score_gain_mean": mean(selected),
        "selected_reason_harm_rate": mean(1.0 if gain < 0 else 0.0 for gain in selected),
        "route_vs_helpful": route_prf(tp, fp, fn),
        "direct": direct_summary,
        "reason_all": reason_summary,
        "routed": routed_summary,
        "routed_minus_direct": {
            metric: routed_summary[metric] - direct_summary[metric]
            for metric in METRICS
        },
    }


def policy_screen(rows, source):
    by_policy = defaultdict(dict)
    for row in rows:
        if row["source"] == source:
            by_policy[row["policy"]][row["split"]] = row
    screened = []
    for policy, splits in by_policy.items():
        if not {"test", "test_seen", "test_unseen"} <= set(splits):
            continue
        test = splits["test"]["routed_minus_direct"]["score"]
        seen = splits["test_seen"]["routed_minus_direct"]["score"]
        unseen = splits["test_unseen"]["routed_minus_direct"]["score"]
        screened.append(
            {
                "source": source,
                "policy": policy,
                "test_score_delta": test,
                "seen_score_delta": seen,
                "unseen_score_delta": unseen,
                "test_reason_rate": splits["test"]["pred_reason_rate"],
                "seen_reason_rate": splits["test_seen"]["pred_reason_rate"],
                "unseen_reason_rate": splits["test_unseen"]["pred_reason_rate"],
                "passes": test > 0 and unseen >= 0 and seen > 0,
            }
        )
    screened.sort(
        key=lambda row: (
            not row["passes"],
            -row["test_score_delta"],
            -row["unseen_score_delta"],
            row["test_reason_rate"],
        )
    )
    return screened


def metric_cell(row):
    return f"{fmt(row['argument_f1'])}/{fmt(row['event_f1'])}/{fmt(row['trigger_f1'])}/{fmt(row['score'])}"


def delta_cell(row):
    return f"{signed(row['argument_f1'])}/{signed(row['event_f1'])}/{signed(row['trigger_f1'])}/{signed(row['score'])}"


def render_screen_table(rows, source):
    lines = [
        "| policy | pass | reason rate test/seen/unseen | score delta test/seen/unseen |",
        "|---|---:|---:|---:|",
    ]
    for row in [item for item in rows if item["source"] == source][:18]:
        lines.append(
            f"| `{row['policy']}` | `{row['passes']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} |"
        )
    return "\n".join(lines)


def render_detail_table(rows, source, split):
    selected = [row for row in rows if row["source"] == source and row["split"] == split]
    selected.sort(key=lambda row: (-row["routed_minus_direct"]["score"], row["pred_reason_rate"]))
    lines = [
        "| policy | reason rate | routed A/E/T/Score | delta A/E/T/Score | selected gain | harm rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected[:18]:
        lines.append(
            f"| `{row['policy']}` | {pct(row['pred_reason_rate'])} | {metric_cell(row['routed'])} | "
            f"{delta_cell(row['routed_minus_direct'])} | {signed(row['selected_reason_score_gain_mean'])} | "
            f"{pct(row['selected_reason_harm_rate'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    lines = [
        "# Sampled K2 Formal Guard Sweep",
        "",
        "This offline sweep tests margin-disagreement and stability guards across `test_seen`, `test_unseen`, and aggregated `test`. Rules use no gold metrics.",
        "",
        f"- output root: `{payload['output_root']}`",
        "",
        "## Policy Screen: Single-Generation Execution",
        "",
        render_screen_table(payload["screen"], "single_gen_execution"),
        "",
        "## Policy Screen: K2 Expected 17-20",
        "",
        render_screen_table(payload["screen"], "k2_expected"),
        "",
        "## Best Single-Generation Test Policies",
        "",
        render_detail_table(payload["results"], "single_gen_execution", "test"),
        "",
        "## Single-Generation Unseen Policies",
        "",
        render_detail_table(payload["results"], "single_gen_execution", "test_unseen"),
        "",
        "## Reading",
        "",
    ]
    exec_screen = [row for row in payload["screen"] if row["source"] == "single_gen_execution"]
    passing = [row for row in exec_screen if row["passes"]]
    if passing:
        best = passing[0]
        lines.append(
            f"- Best passing single-gen guard: `{best['policy']}` with test/seen/unseen score deltas "
            f"`{best['test_score_delta']:+.4f}/{best['seen_score_delta']:+.4f}/{best['unseen_score_delta']:+.4f}`."
        )
    else:
        lines.append("- No single-gen guard passed all criteria: aggregate positive, seen positive, unseen nonnegative.")
    old = next(row for row in exec_screen if row["policy"] == "old_main")
    lines.append(
        f"- Baseline `old_main`: score deltas test/seen/unseen "
        f"`{old['test_score_delta']:+.4f}/{old['seen_score_delta']:+.4f}/{old['unseen_score_delta']:+.4f}`."
    )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['report_json']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    config = load_json(repo_path(args.config))
    cases_by_split = build_cases(config)
    case_sets = {"test": [case for split in SPLITS for case in cases_by_split[split]], **cases_by_split}
    results = []
    for split, cases in case_sets.items():
        for source in ["k2_expected", "single_gen_execution"]:
            for policy in POLICIES:
                results.append(summarize(cases, split, source, policy))
    screen = []
    for source in ["k2_expected", "single_gen_execution"]:
        screen.extend(policy_screen(results, source))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": repo_path(args.config).as_posix(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "policies": POLICIES,
        "splits": {split: len(cases) for split, cases in cases_by_split.items()},
        "results": results,
        "screen": screen,
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
