#!/usr/bin/env python3
import argparse
import itertools
import json
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
FRESH_SEEDS = [21, 22]
CHECKPOINT = "checkpoint-50"
EXEC_ROOT = REPO / "outputs/stage2_adaptive_route_formal_execution_20260518/sampledk2_ckpt50_margin025"
OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_goldfree_proxy_sweep_20260519/sampledk2_goldfree_proxy"
REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_goldfree_stability_proxy_sweep.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_goldfree_stability_proxy_sweep.json"


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


def score(metrics):
    return metrics["argument_f1"] + metrics["event_f1"] + 0.25 * metrics["trigger_f1"]


def avg_metric_dict(rows):
    metrics = [metric_dict(row) for row in rows]
    return {
        name: mean(metric[name] for metric in metrics)
        for name in ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json"]
    }


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


def feature_delta(reason_feat, direct_feat, name):
    return reason_feat.get(name, 0.0) - direct_feat.get(name, 0.0)


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
            direct_samples = sample_rows["direct"][key]
            reason_samples = sample_rows["reason"][key]
            direct_sample_metrics = avg_metric_dict(direct_samples)
            reason_sample_metrics = avg_metric_dict(reason_samples)
            direct_exec = metric_dict(exec_rows["direct"][key])
            reason_exec = metric_dict(exec_rows["reason"][key])
            direct_feat = aggregate_sample_rows(direct_samples)
            reason_feat = aggregate_sample_rows(reason_samples)
            old_margin = margins["old17_18_margin"][key]
            new_margin = margins["new19_20_margin"][key]
            fresh_margin = margins["fresh_margin"][key]
            margin_values = [old_margin, new_margin, fresh_margin]
            row = {
                "split": split,
                "case_id": f"{split}::{key}",
                "wnd_id": key,
                "fresh_margin": fresh_margin,
                "old17_18_margin": old_margin,
                "new19_20_margin": new_margin,
                "avg_margin": mean(margin_values),
                "min_margin": min(margin_values),
                "max_margin": max(margin_values),
                "margin_range": max(margin_values) - min(margin_values),
                "num_margins_ge_0p25": sum(1 for val in margin_values if val >= 0.25),
                "num_margins_ge_0": sum(1 for val in margin_values if val >= 0.0),
                "same_margin_sign": all(val >= 0 for val in margin_values) or all(val < 0 for val in margin_values),
                "single_gen_direct": direct_exec,
                "single_gen_reason": reason_exec,
                "single_gen_score_gain": reason_exec["score"] - direct_exec["score"],
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


def base_predicates():
    predicates = [
        ("fresh_margin_ge_0p25", lambda row: row["fresh_margin"] >= 0.25),
        ("fresh_margin_ge_0p30", lambda row: row["fresh_margin"] >= 0.30),
        ("fresh_margin_ge_0p35", lambda row: row["fresh_margin"] >= 0.35),
        ("fresh_margin_ge_0p40", lambda row: row["fresh_margin"] >= 0.40),
        ("event_no_drop", lambda row: row["reason_minus_direct_event_count_mean"] >= 0.0),
        ("arg_no_drop", lambda row: row["reason_minus_direct_argument_count_mean"] >= 0.0),
        ("trigger_no_drop", lambda row: row["reason_minus_direct_trigger_count_mean"] >= 0.0),
        ("event_type_no_drop", lambda row: row["reason_minus_direct_event_type_count_mean"] >= 0.0),
        ("reason_not_more_variable", lambda row: row["reason_minus_direct_unique_signature_count"] <= 0.0),
        ("reason_full_consensus_ge_direct", lambda row: row["reason_minus_direct_full_consensus"] >= 0.0),
        ("reason_chars_not_plus_128", lambda row: row["reason_minus_direct_output_chars_mean"] <= 128.0),
        ("reason_chars_not_plus_64", lambda row: row["reason_minus_direct_output_chars_mean"] <= 64.0),
        ("avg_margin_ge_0p20", lambda row: row["avg_margin"] >= 0.20),
        ("avg_margin_ge_0p25", lambda row: row["avg_margin"] >= 0.25),
        ("min_margin_ge_0", lambda row: row["min_margin"] >= 0.0),
        ("min_margin_ge_neg0p10", lambda row: row["min_margin"] >= -0.10),
        ("margin_range_le_0p50", lambda row: row["margin_range"] <= 0.50),
        ("margin_range_le_0p35", lambda row: row["margin_range"] <= 0.35),
        ("two_of_three_margins_ge_0p25", lambda row: row["num_margins_ge_0p25"] >= 2),
        ("all_three_margins_nonnegative", lambda row: row["num_margins_ge_0"] == 3),
        ("same_margin_sign", lambda row: row["same_margin_sign"]),
    ]
    return predicates


def generate_rules():
    predicates = dict(base_predicates())
    def make_rule(name, parts):
        return {
            "id": name,
            "parts": list(parts),
            "uses_gold": False,
            "fn": lambda row, ps=list(parts): all(predicates[p](row) for p in ps),
        }

    hand_rules = [
        ("margin_only", ["fresh_margin_ge_0p25"]),
        ("locked_event_guard", ["fresh_margin_ge_0p25", "event_no_drop"]),
        ("locked_event_arg_guard", ["fresh_margin_ge_0p25", "event_no_drop", "arg_no_drop"]),
        ("locked_event_trigger_guard", ["fresh_margin_ge_0p25", "event_no_drop", "trigger_no_drop"]),
        ("locked_event_arg_trigger_guard", ["fresh_margin_ge_0p25", "event_no_drop", "arg_no_drop", "trigger_no_drop"]),
        ("locked_event_reason_not_more_variable", ["fresh_margin_ge_0p25", "event_no_drop", "reason_not_more_variable"]),
        ("strong030_event_guard", ["fresh_margin_ge_0p30", "event_no_drop"]),
        ("strong035_event_guard", ["fresh_margin_ge_0p35", "event_no_drop"]),
        ("strong040_event_guard", ["fresh_margin_ge_0p40", "event_no_drop"]),
        ("strong030_event_arg_guard", ["fresh_margin_ge_0p30", "event_no_drop", "arg_no_drop"]),
        ("strong030_event_variable_guard", ["fresh_margin_ge_0p30", "event_no_drop", "reason_not_more_variable"]),
        ("stable_margin_event_guard", ["fresh_margin_ge_0p25", "event_no_drop", "min_margin_ge_0"]),
        ("avg_margin_event_guard", ["fresh_margin_ge_0p25", "event_no_drop", "avg_margin_ge_0p25"]),
        ("range_event_guard", ["fresh_margin_ge_0p25", "event_no_drop", "margin_range_le_0p35"]),
        ("two_seed_event_guard", ["fresh_margin_ge_0p25", "event_no_drop", "two_of_three_margins_ge_0p25"]),
        ("all_nonneg_event_guard", ["fresh_margin_ge_0p25", "event_no_drop", "all_three_margins_nonnegative"]),
        ("stable_margin_event_arg_guard", ["fresh_margin_ge_0p25", "event_no_drop", "arg_no_drop", "min_margin_ge_0"]),
        ("avg_margin_event_arg_guard", ["fresh_margin_ge_0p25", "event_no_drop", "arg_no_drop", "avg_margin_ge_0p25"]),
    ]
    rules = [make_rule(name, parts) for name, parts in hand_rules]
    atoms = [
        "fresh_margin_ge_0p25",
        "fresh_margin_ge_0p30",
        "fresh_margin_ge_0p35",
        "event_no_drop",
        "arg_no_drop",
        "trigger_no_drop",
        "reason_not_more_variable",
        "reason_full_consensus_ge_direct",
        "min_margin_ge_0",
        "min_margin_ge_neg0p10",
        "avg_margin_ge_0p20",
        "avg_margin_ge_0p25",
        "margin_range_le_0p50",
        "two_of_three_margins_ge_0p25",
    ]
    seen = {rule["id"] for rule in rules}
    for size in [2, 3, 4]:
        for combo in itertools.combinations(atoms, size):
            if not any(part.startswith("fresh_margin_ge") for part in combo):
                continue
            if "fresh_margin_ge_0p25" in combo and "fresh_margin_ge_0p30" in combo:
                continue
            if "fresh_margin_ge_0p30" in combo and "fresh_margin_ge_0p35" in combo:
                continue
            name = "auto_" + "__".join(combo)
            if name in seen:
                continue
            rules.append(make_rule(name, combo))
            seen.add(name)
    return rules


def summarize_metrics(rows):
    return {
        metric: mean(row[metric] for row in rows)
        for metric in ["argument_f1", "event_f1", "trigger_f1", "score"]
    }


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_rule(cases, split, rule, source):
    split_cases = cases if split == "test" else [row for row in cases if row["split"] == split]
    direct_rows = []
    routed_rows = []
    selected = set()
    helpful = set()
    selected_gains = []
    for row in split_cases:
        direct = row[f"{source}_direct"]
        reason = row[f"{source}_reason"]
        gain = score(reason) - score(direct)
        direct_rows.append(direct)
        if gain > 0:
            helpful.add(row["case_id"])
        if rule["fn"](row):
            selected.add(row["case_id"])
            selected_gains.append(gain)
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)
    direct_avg = summarize_metrics(direct_rows)
    routed_avg = summarize_metrics(routed_rows)
    tp = len(selected & helpful)
    fp = len(selected - helpful)
    fn = len(helpful - selected)
    return {
        "rule_id": rule["id"],
        "parts": rule["parts"],
        "split": split,
        "source": source,
        "num_examples": len(split_cases),
        "pred_reason_count": len(selected),
        "pred_reason_rate": len(selected) / len(split_cases) if split_cases else 0.0,
        "selected_reason_score_gain_mean": mean(selected_gains),
        "selected_reason_harm_rate": mean(1.0 if gain < 0 else 0.0 for gain in selected_gains),
        "route_vs_helpful": route_prf(tp, fp, fn),
        "routed_minus_direct": {
            metric: routed_avg[metric] - direct_avg[metric]
            for metric in ["argument_f1", "event_f1", "trigger_f1", "score"]
        },
    }


def consolidate_rule(rows, rule):
    by_source_split = {(row["source"], row["split"]): row for row in rows}
    single_test = by_source_split[("single_gen", "test")]
    single_seen = by_source_split[("single_gen", "test_seen")]
    single_unseen = by_source_split[("single_gen", "test_unseen")]
    k2_test = by_source_split[("k2_expected", "test")]
    passes = (
        single_test["routed_minus_direct"]["score"] > 0
        and single_seen["routed_minus_direct"]["score"] > 0
        and single_unseen["routed_minus_direct"]["score"] >= 0
        and 0.03 <= single_test["pred_reason_rate"] <= 0.12
    )
    return {
        "rule_id": rule["id"],
        "parts": rule["parts"],
        "single_gen_test_score_delta": single_test["routed_minus_direct"]["score"],
        "single_gen_seen_score_delta": single_seen["routed_minus_direct"]["score"],
        "single_gen_unseen_score_delta": single_unseen["routed_minus_direct"]["score"],
        "single_gen_test_aet_delta": {
            metric: single_test["routed_minus_direct"][metric]
            for metric in ["argument_f1", "event_f1", "trigger_f1"]
        },
        "single_gen_test_reason_rate": single_test["pred_reason_rate"],
        "single_gen_seen_reason_rate": single_seen["pred_reason_rate"],
        "single_gen_unseen_reason_rate": single_unseen["pred_reason_rate"],
        "single_gen_test_harm_rate": single_test["selected_reason_harm_rate"],
        "single_gen_seen_harm_rate": single_seen["selected_reason_harm_rate"],
        "single_gen_unseen_harm_rate": single_unseen["selected_reason_harm_rate"],
        "k2_expected_test_score_delta": k2_test["routed_minus_direct"]["score"],
        "passes_proxy_screen": passes,
        "screen_score": min(
            single_test["routed_minus_direct"]["score"],
            single_seen["routed_minus_direct"]["score"],
            single_unseen["routed_minus_direct"]["score"],
        ),
    }


def compact_case(row):
    return {
        "case_id": row["case_id"],
        "wnd_id": row["wnd_id"],
        "split": row["split"],
        "single_gen_score_gain": row["single_gen_score_gain"],
        "k2_expected_score_gain": row["k2_expected_score_gain"],
        "fresh_margin": row["fresh_margin"],
        "old17_18_margin": row["old17_18_margin"],
        "new19_20_margin": row["new19_20_margin"],
        "avg_margin": row["avg_margin"],
        "min_margin": row["min_margin"],
        "margin_range": row["margin_range"],
        "num_margins_ge_0p25": row["num_margins_ge_0p25"],
        "reason_minus_direct_event_count_mean": row["reason_minus_direct_event_count_mean"],
        "reason_minus_direct_argument_count_mean": row["reason_minus_direct_argument_count_mean"],
        "reason_minus_direct_trigger_count_mean": row["reason_minus_direct_trigger_count_mean"],
        "reason_minus_direct_unique_signature_count": row["reason_minus_direct_unique_signature_count"],
        "reason_minus_direct_full_consensus": row["reason_minus_direct_full_consensus"],
        "candidate_types": row.get("meta", {}).get("candidate_types"),
        "gold_event_types": row.get("meta", {}).get("gold_event_types"),
    }


def signed_cell(value):
    return signed(value)


def render_leaderboard(rows, limit=25):
    lines = [
        "| rank | rule | pass | reason test/seen/unseen | score delta test/seen/unseen | A/E/T test | harm test/seen/unseen |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows[:limit], 1):
        aet = row["single_gen_test_aet_delta"]
        lines.append(
            f"| {idx} | `{row['rule_id']}` | `{row['passes_proxy_screen']}` | "
            f"{pct(row['single_gen_test_reason_rate'])}/{pct(row['single_gen_seen_reason_rate'])}/{pct(row['single_gen_unseen_reason_rate'])} | "
            f"{signed_cell(row['single_gen_test_score_delta'])}/{signed_cell(row['single_gen_seen_score_delta'])}/{signed_cell(row['single_gen_unseen_score_delta'])} | "
            f"{signed_cell(aet['argument_f1'])}/{signed_cell(aet['event_f1'])}/{signed_cell(aet['trigger_f1'])} | "
            f"{pct(row['single_gen_test_harm_rate'])}/{pct(row['single_gen_seen_harm_rate'])}/{pct(row['single_gen_unseen_harm_rate'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    passing = [row for row in payload["consolidated"] if row["passes_proxy_screen"]]
    best = payload["consolidated"][0]
    locked = next(row for row in payload["consolidated"] if row["rule_id"] == "locked_event_guard")
    lines = [
        "# Sampled K2 Gold-Free Stability Proxy Sweep",
        "",
        "This offline sweep evaluates gold-free routing guards. Gold metrics are used only for evaluation, not in the rules.",
        "",
        f"- rules evaluated: `{payload['num_rules']}`",
        f"- cases: `{payload['num_cases']}`",
        f"- output root: `{payload['output_root']}`",
        "",
        "## Leaderboard",
        "",
        render_leaderboard(payload["consolidated"]),
        "",
        "## Passing Rules",
        "",
    ]
    if passing:
        lines.append(render_leaderboard(passing, limit=20))
    else:
        lines.append("No gold-free rule passed the proxy screen.")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- Best ranked rule: `{best['rule_id']}` with single-gen score delta test/seen/unseen `{best['single_gen_test_score_delta']:+.4f}/{best['single_gen_seen_score_delta']:+.4f}/{best['single_gen_unseen_score_delta']:+.4f}`.",
            f"- Locked event guard baseline: `{locked['single_gen_test_score_delta']:+.4f}/{locked['single_gen_seen_score_delta']:+.4f}/{locked['single_gen_unseen_score_delta']:+.4f}`.",
            f"- Passing rule count: `{len(passing)}`.",
        ]
    )
    if passing:
        first = passing[0]
        lines.append(
            f"- Recommended locked-validation candidate: `{first['rule_id']}` with parts `{first['parts']}`."
        )
    else:
        lines.append("- No deployable proxy is ready for locked validation; the next step should add stronger gold-free evidence features.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['report_json']}`",
            f"- cases JSONL: `{payload['cases_jsonl']}`",
            f"- evaluations JSONL: `{payload['evaluations_jsonl']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    fresh_config = load_json(repo_path(args.fresh_config))
    consensus_config = load_json(repo_path(args.consensus_config))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = build_cases(fresh_config, consensus_config)
    rules = generate_rules()
    evaluations = []
    by_rule = defaultdict(list)
    for rule in rules:
        for source in ["single_gen", "k2_expected"]:
            for split in ["test", "test_seen", "test_unseen"]:
                row = evaluate_rule(cases, split, rule, source)
                evaluations.append(row)
                by_rule[rule["id"]].append(row)
    rule_by_id = {rule["id"]: rule for rule in rules}
    consolidated = [
        consolidate_rule(rows, rule_by_id[rule_id])
        for rule_id, rows in by_rule.items()
    ]
    consolidated.sort(
        key=lambda row: (
            row["passes_proxy_screen"],
            row["screen_score"],
            row["single_gen_test_score_delta"],
            -row["single_gen_test_reason_rate"],
        ),
        reverse=True,
    )
    cases_jsonl = OUTPUT_ROOT / "cases.jsonl"
    eval_jsonl = OUTPUT_ROOT / "evaluations.jsonl"
    write_jsonl(cases_jsonl, [compact_case(row) for row in cases])
    write_jsonl(eval_jsonl, evaluations)
    payload = {
        "fresh_config": repo_path(args.fresh_config).as_posix(),
        "consensus_config": repo_path(args.consensus_config).as_posix(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "num_cases": len(cases),
        "num_rules": len(rules),
        "consolidated": consolidated,
        "top_evaluations": evaluations[:20],
        "cases_jsonl": cases_jsonl.as_posix(),
        "evaluations_jsonl": eval_jsonl.as_posix(),
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
