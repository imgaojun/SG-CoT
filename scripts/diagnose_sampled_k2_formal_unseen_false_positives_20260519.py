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

from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


SPLIT = "test_unseen"
ROUTES = ["direct", "reason"]
SEEDPAIRS = {
    "old17_18": [17, 18],
    "new19_20": [19, 20],
    "all17_20": [17, 18, 19, 20],
}
THRESHOLD = 0.25
OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_unseen_fp_diagnosis_20260519/sampledk2_unseen_fp"
REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_formal_unseen_false_positive_diagnosis.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_formal_unseen_false_positive_diagnosis.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def key_for(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or row.get("wnd_id")


def score_value(row):
    return row.get("argument_f1", 0.0) + row.get("event_f1", 0.0) + 0.25 * row.get("trigger_f1", 0.0)


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


def metric_dict(row):
    return {
        "argument_f1": row.get("argument_f1", 0.0),
        "event_f1": row.get("event_f1", 0.0),
        "trigger_f1": row.get("trigger_f1", 0.0),
        "score": score_value(row),
        "valid_json": 1.0 if row.get("valid_final_json", row.get("valid_json")) else 0.0,
    }


def events_from_row(row):
    pred = row.get("final_predicted") or row.get("predicted") or {}
    events = pred.get("events") if isinstance(pred, dict) else []
    return events if isinstance(events, list) else []


def output_signature(row):
    events = events_from_row(row)
    compact = []
    for event in events:
        trigger = event.get("trigger") if isinstance(event, dict) else {}
        args = event.get("arguments") if isinstance(event, dict) else []
        arg_sig = []
        if isinstance(args, list):
            for arg in args:
                if isinstance(arg, dict):
                    arg_sig.append(
                        (
                            arg.get("role"),
                            arg.get("start"),
                            arg.get("end"),
                            arg.get("text"),
                        )
                    )
        compact.append(
            (
                event.get("event_type") if isinstance(event, dict) else None,
                trigger.get("start") if isinstance(trigger, dict) else None,
                trigger.get("end") if isinstance(trigger, dict) else None,
                tuple(sorted(arg_sig)),
            )
        )
    return json.dumps(sorted(compact), ensure_ascii=False, sort_keys=True)


def row_features(row):
    events = events_from_row(row)
    event_types = []
    argument_count = 0
    trigger_count = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        event_types.append(event.get("event_type"))
        if event.get("trigger"):
            trigger_count += 1
        args = event.get("arguments")
        if isinstance(args, list):
            argument_count += len(args)
    text = row.get("generated_text") or row.get("generated_payload") or ""
    return {
        "valid": 1.0 if row.get("valid_final_json", row.get("valid_json")) else 0.0,
        "event_count": len(events),
        "trigger_count": trigger_count,
        "argument_count": argument_count,
        "event_type_count": len(set(event_types)),
        "output_chars": len(text),
        "signature": output_signature(row),
    }


def aggregate_sample_rows(rows):
    feats = [row_features(row) for row in rows]
    signatures = [feat["signature"] for feat in feats]
    out = {}
    for name in ["valid", "event_count", "trigger_count", "argument_count", "event_type_count", "output_chars"]:
        vals = [feat[name] for feat in feats]
        out[f"{name}_mean"] = mean(vals)
        out[f"{name}_std"] = stdev(vals)
        out[f"{name}_min"] = min(vals) if vals else 0.0
        out[f"{name}_max"] = max(vals) if vals else 0.0
    out["full_consensus"] = 1.0 if len(set(signatures)) <= 1 else 0.0
    out["unique_signature_count"] = len(set(signatures))
    return out


def load_sample_rows(sample_root: Path, split: str, route: str, seeds):
    grouped = defaultdict(list)
    for seed in seeds:
        path = sample_root / split / route / f"seed-{seed}" / "predictions.jsonl"
        for row in load_jsonl(path):
            key = key_for(row)
            if key:
                grouped[key].append(row)
    for key, rows in grouped.items():
        if len(rows) != len(seeds):
            raise ValueError(f"{split}/{route}/{key}: expected {len(seeds)} samples, got {len(rows)}")
    return grouped


def load_exec_rows(execution_root: Path, split: str, route: str):
    path = execution_root / f"forced_{route}" / split / "predictions.jsonl"
    return {key_for(row): row for row in load_jsonl(path)}


def load_margins(root: Path):
    path = root / "checkpoint-50" / SPLIT / "scores.jsonl"
    return {
        key_for(row): row.get("delta_direct_minus_reason_route_nll")
        for row in load_jsonl(path)
        if key_for(row)
    }


def classify_group(old_margin, new_margin):
    old_on = old_margin >= THRESHOLD
    new_on = new_margin >= THRESHOLD
    if old_on and new_on:
        return "both_ge_0p25"
    if old_on:
        return "old_only_ge_0p25"
    if new_on:
        return "new_only_ge_0p25"
    return "neither_ge_0p25"


def metric_average(rows):
    metrics = [metric_dict(row) for row in rows]
    return {
        name: mean(metric[name] for metric in metrics)
        for name in ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json"]
    }


def feature_delta(reason_feat, direct_feat, name):
    return reason_feat.get(name, 0.0) - direct_feat.get(name, 0.0)


def build_cases(config):
    sample_root = repo_path(config["sample_root"])
    execution_root = repo_path(config["execution_root"])
    old_root = repo_path(config["route_nll_roots"]["seedpair17_18"])
    new_root = repo_path(config["route_nll_roots"]["seedpair19_20"])
    old_margins = load_margins(old_root)
    new_margins = load_margins(new_root)
    exec_rows = {route: load_exec_rows(execution_root, SPLIT, route) for route in ROUTES}
    sample_rows = {
        seedpair: {
            route: load_sample_rows(sample_root, SPLIT, route, seeds)
            for route in ROUTES
        }
        for seedpair, seeds in SEEDPAIRS.items()
    }
    keys = set(old_margins) & set(new_margins) & set(exec_rows["direct"]) & set(exec_rows["reason"])
    for seedpair in SEEDPAIRS:
        for route in ROUTES:
            keys &= set(sample_rows[seedpair][route])

    cases = []
    for key in sorted(keys):
        old_margin = old_margins[key]
        new_margin = new_margins[key]
        exec_direct = metric_dict(exec_rows["direct"][key])
        exec_reason = metric_dict(exec_rows["reason"][key])
        row = {
            "case_id": f"{SPLIT}::{key}",
            "split": SPLIT,
            "wnd_id": key,
            "group": classify_group(old_margin, new_margin),
            "old17_18_margin": old_margin,
            "new19_20_margin": new_margin,
            "avg_margin": (old_margin + new_margin) / 2,
            "margin_gap_new_minus_old": new_margin - old_margin,
            "abs_margin_gap": abs(new_margin - old_margin),
            "single_gen_direct": exec_direct,
            "single_gen_reason": exec_reason,
            "single_gen_score_gain": exec_reason["score"] - exec_direct["score"],
            "single_gen_argument_gain": exec_reason["argument_f1"] - exec_direct["argument_f1"],
            "single_gen_event_gain": exec_reason["event_f1"] - exec_direct["event_f1"],
            "single_gen_trigger_gain": exec_reason["trigger_f1"] - exec_direct["trigger_f1"],
            "meta": exec_rows["direct"][key].get("meta", {}),
        }
        for seedpair in SEEDPAIRS:
            direct_rows = sample_rows[seedpair]["direct"][key]
            reason_rows = sample_rows[seedpair]["reason"][key]
            direct_metrics = metric_average(direct_rows)
            reason_metrics = metric_average(reason_rows)
            direct_feat = aggregate_sample_rows(direct_rows)
            reason_feat = aggregate_sample_rows(reason_rows)
            row[f"{seedpair}_direct_metrics"] = direct_metrics
            row[f"{seedpair}_reason_metrics"] = reason_metrics
            row[f"{seedpair}_k2_score_gain"] = reason_metrics["score"] - direct_metrics["score"]
            row[f"{seedpair}_direct_features"] = direct_feat
            row[f"{seedpair}_reason_features"] = reason_feat
            for feat in [
                "event_count_mean",
                "argument_count_mean",
                "trigger_count_mean",
                "output_chars_mean",
                "full_consensus",
                "unique_signature_count",
            ]:
                row[f"{seedpair}_reason_minus_direct_{feat}"] = feature_delta(
                    reason_feat, direct_feat, feat
                )
        cases.append(row)
    return cases


def summarize_group(rows, group):
    selected = [row for row in rows if row["group"] == group]
    if not selected:
        return {"group": group, "count": 0, "rate": 0.0}
    out = {
        "group": group,
        "count": len(selected),
        "rate": len(selected) / len(rows),
        "single_gen_score_gain": summarize_values(row["single_gen_score_gain"] for row in selected),
        "single_gen_argument_gain_mean": mean(row["single_gen_argument_gain"] for row in selected),
        "single_gen_event_gain_mean": mean(row["single_gen_event_gain"] for row in selected),
        "single_gen_trigger_gain_mean": mean(row["single_gen_trigger_gain"] for row in selected),
        "single_gen_harm_rate": mean(1.0 if row["single_gen_score_gain"] < 0 else 0.0 for row in selected),
        "old17_18_margin": summarize_values(row["old17_18_margin"] for row in selected),
        "new19_20_margin": summarize_values(row["new19_20_margin"] for row in selected),
        "avg_margin": summarize_values(row["avg_margin"] for row in selected),
        "abs_margin_gap": summarize_values(row["abs_margin_gap"] for row in selected),
    }
    for seedpair in SEEDPAIRS:
        out[f"{seedpair}_k2_score_gain"] = summarize_values(
            row[f"{seedpair}_k2_score_gain"] for row in selected
        )
        for feat in [
            "reason_minus_direct_event_count_mean",
            "reason_minus_direct_argument_count_mean",
            "reason_minus_direct_trigger_count_mean",
            "reason_minus_direct_output_chars_mean",
            "reason_minus_direct_full_consensus",
            "reason_minus_direct_unique_signature_count",
        ]:
            out[f"{seedpair}_{feat}"] = mean(row[f"{seedpair}_{feat}"] for row in selected)
    return out


def candidate_feature_tests(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    comparisons = [
        ("old_only_vs_new_only", groups["old_only_ge_0p25"], groups["new_only_ge_0p25"]),
        ("both_vs_old_only", groups["both_ge_0p25"], groups["old_only_ge_0p25"]),
        ("harmful_selected_vs_helpful_selected", [
            row for row in rows if row["group"] != "neither_ge_0p25" and row["single_gen_score_gain"] < 0
        ], [
            row for row in rows if row["group"] != "neither_ge_0p25" and row["single_gen_score_gain"] > 0
        ]),
    ]
    feature_names = [
        "old17_18_margin",
        "new19_20_margin",
        "avg_margin",
        "margin_gap_new_minus_old",
        "abs_margin_gap",
        "old17_18_k2_score_gain",
        "new19_20_k2_score_gain",
        "all17_20_k2_score_gain",
        "old17_18_reason_minus_direct_event_count_mean",
        "new19_20_reason_minus_direct_event_count_mean",
        "all17_20_reason_minus_direct_event_count_mean",
        "old17_18_reason_minus_direct_argument_count_mean",
        "new19_20_reason_minus_direct_argument_count_mean",
        "all17_20_reason_minus_direct_argument_count_mean",
        "old17_18_reason_minus_direct_full_consensus",
        "new19_20_reason_minus_direct_full_consensus",
        "all17_20_reason_minus_direct_full_consensus",
        "old17_18_reason_minus_direct_unique_signature_count",
        "new19_20_reason_minus_direct_unique_signature_count",
        "all17_20_reason_minus_direct_unique_signature_count",
    ]
    out = []
    for name, left, right in comparisons:
        for feat in feature_names:
            left_vals = [row.get(feat, 0.0) for row in left]
            right_vals = [row.get(feat, 0.0) for row in right]
            left_mean = mean(left_vals)
            right_mean = mean(right_vals)
            pooled = stdev(left_vals + right_vals)
            out.append(
                {
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
    out.sort(key=lambda row: (row["comparison"], -row["abs_standardized_diff"], row["feature"]))
    return out


def compact_case(row):
    meta = row.get("meta", {})
    return {
        "case_id": row["case_id"],
        "wnd_id": row["wnd_id"],
        "group": row["group"],
        "single_gen_score_gain": row["single_gen_score_gain"],
        "single_gen_argument_gain": row["single_gen_argument_gain"],
        "single_gen_event_gain": row["single_gen_event_gain"],
        "single_gen_trigger_gain": row["single_gen_trigger_gain"],
        "old17_18_margin": row["old17_18_margin"],
        "new19_20_margin": row["new19_20_margin"],
        "avg_margin": row["avg_margin"],
        "old17_18_k2_score_gain": row["old17_18_k2_score_gain"],
        "new19_20_k2_score_gain": row["new19_20_k2_score_gain"],
        "all17_20_k2_score_gain": row["all17_20_k2_score_gain"],
        "candidate_types": meta.get("candidate_types"),
        "gold_event_types": meta.get("gold_event_types"),
        "doc_id": meta.get("doc_id"),
        "source_part": meta.get("source_part"),
        "features": {
            name: row.get(name)
            for name in [
                "old17_18_reason_minus_direct_event_count_mean",
                "new19_20_reason_minus_direct_event_count_mean",
                "all17_20_reason_minus_direct_event_count_mean",
                "old17_18_reason_minus_direct_argument_count_mean",
                "new19_20_reason_minus_direct_argument_count_mean",
                "all17_20_reason_minus_direct_argument_count_mean",
                "old17_18_reason_minus_direct_full_consensus",
                "new19_20_reason_minus_direct_full_consensus",
                "all17_20_reason_minus_direct_full_consensus",
            ]
        },
    }


def render_group_table(rows):
    lines = [
        "| group | count | single-gen gain mean | harm rate | old margin | new margin | old K2 gain | new K2 gain | all K2 gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['group']}` | {row['count']} | {signed(row['single_gen_score_gain']['mean'])} | "
            f"{pct(row['single_gen_harm_rate'])} | {fmt(row['old17_18_margin']['mean'])} | "
            f"{fmt(row['new19_20_margin']['mean'])} | {signed(row['old17_18_k2_score_gain']['mean'])} | "
            f"{signed(row['new19_20_k2_score_gain']['mean'])} | {signed(row['all17_20_k2_score_gain']['mean'])} |"
        )
    return "\n".join(lines)


def render_feature_table(rows, comparison, limit=10):
    selected = [row for row in rows if row["comparison"] == comparison][:limit]
    lines = [
        "| feature | left mean | right mean | diff | std diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| `{row['feature']}` | {fmt(row['left_mean'])} | {fmt(row['right_mean'])} | "
            f"{signed(row['diff_left_minus_right'])} | {fmt(row['abs_standardized_diff'])} |"
        )
    return "\n".join(lines)


def render_case_table(rows, group, limit=12):
    selected = [row for row in rows if row["group"] == group]
    selected.sort(key=lambda row: row["single_gen_score_gain"])
    lines = [
        "| wnd_id | single gain | old margin | new margin | old K2 gain | new K2 gain | gold types |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected[:limit]:
        gold = ",".join(row.get("gold_event_types") or [])
        lines.append(
            f"| `{row['wnd_id']}` | {signed(row['single_gen_score_gain'])} | {fmt(row['old17_18_margin'])} | "
            f"{fmt(row['new19_20_margin'])} | {signed(row['old17_18_k2_score_gain'])} | "
            f"{signed(row['new19_20_k2_score_gain'])} | `{gold}` |"
        )
    return "\n".join(lines)


def render_report(payload):
    lines = [
        "# Sampled K2 Formal Unseen False-Positive Diagnosis",
        "",
        "This report diagnoses `test_unseen` route failures using existing formal route-NLL scores, K2 sampled outputs, and deterministic execution outputs. Gold metrics are used only for diagnosis labels; candidate guard features must be gold-free.",
        "",
        f"- output root: `{payload['output_root']}`",
        "",
        "## Group Summary",
        "",
        render_group_table(payload["group_summaries"]),
        "",
        "## Separating Features",
        "",
        "### Old-Only vs New-Only",
        "",
        render_feature_table(payload["feature_tests"], "old_only_vs_new_only"),
        "",
        "### Both vs Old-Only",
        "",
        render_feature_table(payload["feature_tests"], "both_vs_old_only"),
        "",
        "### Harmful Selected vs Helpful Selected",
        "",
        render_feature_table(payload["feature_tests"], "harmful_selected_vs_helpful_selected"),
        "",
        "## Worst Cases",
        "",
        "### New-Only",
        "",
        render_case_table(payload["cases"], "new_only_ge_0p25"),
        "",
        "### Both",
        "",
        render_case_table(payload["cases"], "both_ge_0p25"),
        "",
        "### Old-Only",
        "",
        render_case_table(payload["cases"], "old_only_ge_0p25"),
        "",
        "## Reading",
        "",
    ]
    group_by_name = {row["group"]: row for row in payload["group_summaries"]}
    old_only = group_by_name["old_only_ge_0p25"]
    new_only = group_by_name["new_only_ge_0p25"]
    both = group_by_name["both_ge_0p25"]
    lines.extend(
        [
            f"- `old_only_ge_0p25` is the useful unseen set: count `{old_only['count']}`, single-gen gain `{old_only['single_gen_score_gain']['mean']:+.4f}`, harm `{old_only['single_gen_harm_rate']:.1%}`.",
            f"- `new_only_ge_0p25` is strongly harmful: count `{new_only['count']}`, single-gen gain `{new_only['single_gen_score_gain']['mean']:+.4f}`, harm `{new_only['single_gen_harm_rate']:.1%}`.",
            f"- `both_ge_0p25` is small but harmful: count `{both['count']}`, single-gen gain `{both['single_gen_score_gain']['mean']:+.4f}`, harm `{both['single_gen_harm_rate']:.1%}`.",
            "- If the top separating features are deployable gold-free evidence statistics, use them to propose an unseen-risk guard; otherwise do case-level manual inspection before another training run.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['report_json']}`",
            f"- cases JSONL: `{payload['cases_jsonl']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    config = load_json(repo_path(args.config))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = build_cases(config)
    compact_cases = [compact_case(row) for row in cases]
    groups = ["old_only_ge_0p25", "new_only_ge_0p25", "both_ge_0p25", "neither_ge_0p25"]
    group_summaries = [summarize_group(cases, group) for group in groups]
    feature_tests = candidate_feature_tests(cases)
    cases_jsonl = OUTPUT_ROOT / "test_unseen_cases.jsonl"
    write_jsonl(cases_jsonl, compact_cases)
    payload = {
        "config": repo_path(args.config).as_posix(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "split": SPLIT,
        "threshold": THRESHOLD,
        "num_cases": len(cases),
        "group_summaries": group_summaries,
        "feature_tests": feature_tests,
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
