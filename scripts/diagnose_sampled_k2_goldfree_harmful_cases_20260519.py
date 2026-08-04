#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_formal_unseen_false_positives_20260519 import (  # noqa: E402
    events_from_row,
    key_for,
    load_jsonl,
    load_sample_rows,
    metric_dict,
)
from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


SPLITS = ["test_seen", "test_unseen"]
ROUTES = ["direct", "reason"]
FRESH_SEEDS = [21, 22]
CHECKPOINT = "checkpoint-50"
EXEC_ROOT = REPO / "outputs/stage2_adaptive_route_formal_execution_20260518/sampledk2_ckpt50_margin025"
OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_goldfree_harm_diagnosis_20260519/sampledk2_harmful_cases"
REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_goldfree_harmful_case_feature_diagnosis.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_goldfree_harmful_case_feature_diagnosis.json"


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


def score(metrics):
    return metrics["argument_f1"] + metrics["event_f1"] + 0.25 * metrics["trigger_f1"]


def load_exec_rows(split: str, route: str):
    path = EXEC_ROOT / f"forced_{route}" / split / "predictions.jsonl"
    return {key_for(row): row for row in load_jsonl(path)}


def load_margins(root: Path, split: str, checkpoint: str = CHECKPOINT):
    path = root / checkpoint / split / "scores.jsonl"
    return {
        key_for(row): row.get("delta_direct_minus_reason_route_nll")
        for row in load_jsonl(path)
        if key_for(row)
    }


def arg_items(event):
    args = event.get("arguments") if isinstance(event, dict) else []
    out = []
    if isinstance(args, list):
        for arg in args:
            if isinstance(arg, dict):
                out.append(arg)
    return out


def event_output_features(row):
    events = events_from_row(row)
    event_types = []
    triggers = []
    arg_roles = []
    arg_texts = []
    arg_spans = []
    event_type_role_pairs = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        if event_type:
            event_types.append(str(event_type))
        trigger = event.get("trigger")
        if isinstance(trigger, dict):
            trig_text = trigger.get("text")
            if trig_text:
                triggers.append(str(trig_text).lower())
        for arg in arg_items(event):
            role = arg.get("role")
            text = arg.get("text")
            start = arg.get("start")
            end = arg.get("end")
            if role:
                arg_roles.append(str(role))
                event_type_role_pairs.append(f"{event_type}::{role}")
            if text:
                arg_texts.append(str(text).lower())
            if start is not None and end is not None:
                arg_spans.append(f"{start}:{end}")
    return {
        "event_count": len(events),
        "event_types": set(event_types),
        "triggers": set(triggers),
        "arg_roles": set(arg_roles),
        "arg_texts": set(arg_texts),
        "arg_spans": set(arg_spans),
        "event_type_role_pairs": set(event_type_role_pairs),
        "argument_count": len(arg_roles),
    }


def jaccard(left, right):
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def safe_ratio(num, den):
    return num / den if den else 0.0


def pair_features(direct_row, reason_row):
    direct = event_output_features(direct_row)
    reason = event_output_features(reason_row)
    out = {
        "event_count_delta": reason["event_count"] - direct["event_count"],
        "argument_count_delta": reason["argument_count"] - direct["argument_count"],
        "event_type_jaccard": jaccard(direct["event_types"], reason["event_types"]),
        "trigger_text_jaccard": jaccard(direct["triggers"], reason["triggers"]),
        "arg_role_jaccard": jaccard(direct["arg_roles"], reason["arg_roles"]),
        "arg_text_jaccard": jaccard(direct["arg_texts"], reason["arg_texts"]),
        "arg_span_jaccard": jaccard(direct["arg_spans"], reason["arg_spans"]),
        "event_type_role_jaccard": jaccard(direct["event_type_role_pairs"], reason["event_type_role_pairs"]),
        "reason_new_event_type_count": len(reason["event_types"] - direct["event_types"]),
        "reason_dropped_event_type_count": len(direct["event_types"] - reason["event_types"]),
        "reason_new_arg_role_count": len(reason["arg_roles"] - direct["arg_roles"]),
        "reason_dropped_arg_role_count": len(direct["arg_roles"] - reason["arg_roles"]),
        "reason_new_arg_text_count": len(reason["arg_texts"] - direct["arg_texts"]),
        "reason_dropped_arg_text_count": len(direct["arg_texts"] - reason["arg_texts"]),
        "reason_arg_text_retention": safe_ratio(len(reason["arg_texts"] & direct["arg_texts"]), len(direct["arg_texts"])),
        "reason_event_type_retention": safe_ratio(len(reason["event_types"] & direct["event_types"]), len(direct["event_types"])),
    }
    return out


def aggregate_pair_features(direct_rows, reason_rows):
    feats = [pair_features(d, r) for d, r in zip(direct_rows, reason_rows)]
    out = {}
    for key in feats[0]:
        vals = [feat[key] for feat in feats]
        out[f"{key}_mean"] = mean(vals)
        out[f"{key}_std"] = stdev(vals)
        out[f"{key}_min"] = min(vals)
        out[f"{key}_max"] = max(vals)
    return out


def avg_metric_dict(rows):
    metrics = [metric_dict(row) for row in rows]
    return {
        name: mean(metric[name] for metric in metrics)
        for name in ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json"]
    }


def build_cases(fresh_config, consensus_config):
    sample_root = repo_path(fresh_config["sample_root"])
    fresh_nll_root = repo_path(fresh_config["output_root"])
    old_nll_root = repo_path(consensus_config["route_nll_roots"]["seedpair17_18"])
    new_nll_root = repo_path(consensus_config["route_nll_roots"]["seedpair19_20"])
    cases = []
    for split in SPLITS:
        margins = {
            "fresh_margin": load_margins(fresh_nll_root, split),
            "old17_18_margin": load_margins(old_nll_root, split),
            "new19_20_margin": load_margins(new_nll_root, split),
        }
        exec_rows = {route: load_exec_rows(split, route) for route in ROUTES}
        sample_rows = {
            route: load_sample_rows(sample_root, split, route, FRESH_SEEDS)
            for route in ROUTES
        }
        keys = set(exec_rows["direct"]) & set(exec_rows["reason"])
        for margin_rows in margins.values():
            keys &= set(margin_rows)
        for route in ROUTES:
            keys &= set(sample_rows[route])
        for key in sorted(keys):
            old_margin = margins["old17_18_margin"][key]
            new_margin = margins["new19_20_margin"][key]
            fresh_margin = margins["fresh_margin"][key]
            margin_values = [old_margin, new_margin, fresh_margin]
            direct_samples = sample_rows["direct"][key]
            reason_samples = sample_rows["reason"][key]
            direct_exec = metric_dict(exec_rows["direct"][key])
            reason_exec = metric_dict(exec_rows["reason"][key])
            sample_pair = aggregate_pair_features(direct_samples, reason_samples)
            exec_pair = pair_features(exec_rows["direct"][key], exec_rows["reason"][key])
            selected_best = (
                fresh_margin >= 0.25
                and max(margin_values) - min(margin_values) <= 0.50
                and sum(1 for val in margin_values if val >= 0.25) >= 2
            )
            row = {
                "split": split,
                "case_id": f"{split}::{key}",
                "wnd_id": key,
                "selected_best": selected_best,
                "fresh_margin": fresh_margin,
                "old17_18_margin": old_margin,
                "new19_20_margin": new_margin,
                "avg_margin": mean(margin_values),
                "margin_range": max(margin_values) - min(margin_values),
                "num_margins_ge_0p25": sum(1 for val in margin_values if val >= 0.25),
                "single_gen_direct": direct_exec,
                "single_gen_reason": reason_exec,
                "single_gen_score_gain": reason_exec["score"] - direct_exec["score"],
                "k2_expected_direct": avg_metric_dict(direct_samples),
                "k2_expected_reason": avg_metric_dict(reason_samples),
                "meta": exec_rows["direct"][key].get("meta", {}),
            }
            row["k2_expected_score_gain"] = row["k2_expected_reason"]["score"] - row["k2_expected_direct"]["score"]
            for name, value in sample_pair.items():
                row[f"sample_{name}"] = value
            for name, value in exec_pair.items():
                row[f"exec_{name}"] = value
            cases.append(row)
    return cases


def category(row):
    if not row["selected_best"]:
        return "not_selected"
    if row["single_gen_score_gain"] > 0:
        return "selected_helpful"
    if row["single_gen_score_gain"] < 0:
        return "selected_harmful"
    return "selected_neutral"


FEATURES = [
    "fresh_margin",
    "avg_margin",
    "margin_range",
    "sample_arg_text_jaccard_mean",
    "sample_arg_span_jaccard_mean",
    "sample_arg_role_jaccard_mean",
    "sample_event_type_jaccard_mean",
    "sample_event_type_role_jaccard_mean",
    "sample_trigger_text_jaccard_mean",
    "sample_reason_arg_text_retention_mean",
    "sample_reason_event_type_retention_mean",
    "sample_reason_new_arg_text_count_mean",
    "sample_reason_dropped_arg_text_count_mean",
    "sample_reason_new_arg_role_count_mean",
    "sample_reason_dropped_arg_role_count_mean",
    "sample_reason_new_event_type_count_mean",
    "sample_reason_dropped_event_type_count_mean",
    "sample_argument_count_delta_mean",
    "sample_event_count_delta_mean",
    "exec_arg_text_jaccard",
    "exec_arg_span_jaccard",
    "exec_arg_role_jaccard",
    "exec_event_type_jaccard",
    "exec_event_type_role_jaccard",
    "exec_reason_arg_text_retention",
    "exec_reason_new_arg_text_count",
    "exec_reason_dropped_arg_text_count",
]


def compare_features(rows, split, left_cat, right_cats):
    split_rows = rows if split == "test" else [row for row in rows if row["split"] == split]
    left = [row for row in split_rows if category(row) == left_cat]
    right = [row for row in split_rows if category(row) in set(right_cats)]
    out = []
    for feat in FEATURES:
        left_vals = [row.get(feat, 0.0) for row in left]
        right_vals = [row.get(feat, 0.0) for row in right]
        left_mean = mean(left_vals)
        right_mean = mean(right_vals)
        pooled = stdev(left_vals + right_vals)
        out.append(
            {
                "split": split,
                "comparison": f"{left_cat}_vs_{'+'.join(right_cats)}",
                "feature": feat,
                "left_count": len(left),
                "right_count": len(right),
                "left_mean": left_mean,
                "right_mean": right_mean,
                "diff_left_minus_right": left_mean - right_mean,
                "abs_standardized_diff": abs(left_mean - right_mean) / pooled if pooled else 0.0,
            }
        )
    out.sort(key=lambda row: (-row["abs_standardized_diff"], row["feature"]))
    return out


def summarize_groups(rows):
    out = []
    for split in ["test", "test_seen", "test_unseen"]:
        split_rows = rows if split == "test" else [row for row in rows if row["split"] == split]
        for cat in ["selected_harmful", "selected_helpful", "selected_neutral", "not_selected"]:
            selected = [row for row in split_rows if category(row) == cat]
            out.append(
                {
                    "split": split,
                    "category": cat,
                    "count": len(selected),
                    "rate": len(selected) / len(split_rows) if split_rows else 0.0,
                    "single_gain_mean": mean(row["single_gen_score_gain"] for row in selected),
                    "k2_gain_mean": mean(row["k2_expected_score_gain"] for row in selected),
                }
            )
    return out


def type_counter(rows, cat):
    counter = Counter()
    for row in rows:
        if category(row) != cat:
            continue
        for event_type in row.get("meta", {}).get("gold_event_types") or []:
            counter[event_type] += 1
    return counter.most_common(12)


def compact_case(row):
    return {
        "case_id": row["case_id"],
        "wnd_id": row["wnd_id"],
        "split": row["split"],
        "selected_best": row["selected_best"],
        "category": category(row),
        "single_gen_score_gain": row["single_gen_score_gain"],
        "k2_expected_score_gain": row["k2_expected_score_gain"],
        "fresh_margin": row["fresh_margin"],
        "avg_margin": row["avg_margin"],
        "margin_range": row["margin_range"],
        "num_margins_ge_0p25": row["num_margins_ge_0p25"],
        "gold_event_types": row.get("meta", {}).get("gold_event_types"),
        "candidate_types": row.get("meta", {}).get("candidate_types"),
        "features": {feat: row.get(feat) for feat in FEATURES},
    }


def render_group_table(rows):
    lines = [
        "| split | category | count | rate | single gain | K2 gain |",
        "|---|---|---:|---:|---:|---:|",
    ]
    order = {"test": 0, "test_seen": 1, "test_unseen": 2}
    cat_order = {"selected_harmful": 0, "selected_helpful": 1, "selected_neutral": 2, "not_selected": 3}
    for row in sorted(rows, key=lambda r: (order[r["split"]], cat_order[r["category"]])):
        lines.append(
            f"| `{row['split']}` | `{row['category']}` | {row['count']} | {pct(row['rate'])} | "
            f"{signed(row['single_gain_mean'])} | {signed(row['k2_gain_mean'])} |"
        )
    return "\n".join(lines)


def render_feature_table(rows, limit=16):
    lines = [
        "| feature | harmful mean | reference mean | diff | std diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['feature']}` | {fmt(row['left_mean'])} | {fmt(row['right_mean'])} | "
            f"{signed(row['diff_left_minus_right'])} | {fmt(row['abs_standardized_diff'])} |"
        )
    return "\n".join(lines)


def render_worst_cases(rows, limit=16):
    selected = [row for row in rows if row["selected_best"]]
    selected.sort(key=lambda row: row["single_gen_score_gain"])
    lines = [
        "| split | wnd_id | gain | K2 gain | margin avg/range | arg-text J | event-role J | gold types |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected[:limit]:
        gold = ",".join(row.get("gold_event_types") or [])
        feats = row.get("features") or {}
        lines.append(
            f"| `{row['split']}` | `{row['wnd_id']}` | {signed(row['single_gen_score_gain'])} | "
            f"{signed(row['k2_expected_score_gain'])} | {fmt(row['avg_margin'])}/{fmt(row['margin_range'])} | "
            f"{fmt(feats.get('sample_arg_text_jaccard_mean'))} | {fmt(feats.get('sample_event_type_role_jaccard_mean'))} | `{gold}` |"
        )
    return "\n".join(lines)


def render_report(payload):
    seen_feats = payload["feature_comparisons"]["test_seen_harmful_vs_nonharmful"]
    test_feats = payload["feature_comparisons"]["test_harmful_vs_nonharmful"]
    lines = [
        "# Sampled K2 Gold-Free Harmful Case Feature Diagnosis",
        "",
        "This report inspects selected harmful cases under the best margin-stability proxy from the previous sweep.",
        "",
        f"- output root: `{payload['output_root']}`",
        "",
        "## Group Summary",
        "",
        render_group_table(payload["group_summary"]),
        "",
        "## Feature Separators",
        "",
        "### Test Seen: Harmful vs Non-Harmful Selected",
        "",
        render_feature_table(seen_feats),
        "",
        "### Aggregate Test: Harmful vs Non-Harmful Selected",
        "",
        render_feature_table(test_feats),
        "",
        "## Worst Selected Cases",
        "",
        render_worst_cases(payload["cases"]),
        "",
        "## Event-Type Concentration",
        "",
        f"- selected harmful: `{payload['event_type_counts']['selected_harmful']}`",
        f"- selected helpful: `{payload['event_type_counts']['selected_helpful']}`",
        "",
        "## Reading",
        "",
    ]
    if seen_feats:
        top = seen_feats[0]
        lines.append(
            f"- Strongest seen separator is `{top['feature']}`: harmful mean `{top['left_mean']:.4f}` vs reference `{top['right_mean']:.4f}`."
        )
    lines.extend(
        [
            "- The next proxy sweep should add the strongest separators as explicit guard predicates, then retest the positive-delta and low-harm screens.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['report_json']}`",
            f"- cases JSONL: `{payload['cases_jsonl']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    fresh_config = load_json(repo_path(args.fresh_config))
    consensus_config = load_json(repo_path(args.consensus_config))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = build_cases(fresh_config, consensus_config)
    compact_cases = [compact_case(row) for row in cases]
    cases_jsonl = OUTPUT_ROOT / "cases.jsonl"
    write_jsonl(cases_jsonl, compact_cases)
    payload = {
        "fresh_config": repo_path(args.fresh_config).as_posix(),
        "consensus_config": repo_path(args.consensus_config).as_posix(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "num_cases": len(cases),
        "group_summary": summarize_groups(cases),
        "feature_comparisons": {
            "test_seen_harmful_vs_nonharmful": compare_features(
                cases, "test_seen", "selected_harmful", ["selected_helpful", "selected_neutral"]
            ),
            "test_harmful_vs_nonharmful": compare_features(
                cases, "test", "selected_harmful", ["selected_helpful", "selected_neutral"]
            ),
        },
        "event_type_counts": {
            "selected_harmful": type_counter(cases, "selected_harmful"),
            "selected_helpful": type_counter(cases, "selected_helpful"),
        },
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
    parser.add_argument("--fresh-config", required=True)
    parser.add_argument("--consensus-config", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
