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

from scripts.summarize_sampled_confident_router_dev_20260518 import pct, signed, write_json, write_text  # noqa: E402
from scripts.summarize_sampled_k2_structural_proxy_locked_validation_20260519 import (  # noqa: E402
    DEFAULT_FRESH_NLL_ROOT,
    DEFAULT_NEW_NLL_ROOT,
    DEFAULT_OLD_NLL_ROOT,
    DEFAULT_SAMPLE_ROOT,
    build_cases,
)


OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_structproxy_relaxed_selector_sweep_20260519"
REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_structproxy_relaxed_selector_sweep.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_structproxy_relaxed_selector_sweep.json"
SPLITS = ["test", "test_seen", "test_unseen"]
SOURCES = ["single_gen_execution", "k2_expected"]
METRICS = ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json"]


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def avg_metrics(rows):
    return {metric: mean(row[metric] for row in rows) for metric in METRICS}


def score(metrics):
    return metrics["score"]


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def locked_rule(case):
    return (
        case["fresh_margin"] >= 0.25
        and case["margin_range"] <= 0.50
        and case["num_margins_ge_0p25"] >= 2
        and case["sample_arg_text_jaccard_mean"] >= 0.40
        and case["sample_event_count_delta_mean"] <= 0.0
    )


def relaxed_rule(case, spec):
    return (
        case["fresh_margin"] >= spec["fresh_margin_min"]
        and case["margin_range"] <= spec["margin_range_max"]
        and case["num_margins_ge_0p25"] >= spec["num_margins_ge_0p25_min"]
        and case["sample_arg_text_jaccard_mean"] >= spec["arg_text_jaccard_min"]
        and case["sample_event_count_delta_mean"] <= spec["event_count_delta_max"]
        and (case["fresh_margin"] + case["old17_18_margin"] + case["new19_20_margin"]) / 3.0 >= spec["avg_margin_min"]
    )


def rule_id(spec):
    return (
        f"fm{spec['fresh_margin_min']:.2f}_"
        f"mr{spec['margin_range_max']:.2f}_"
        f"n{spec['num_margins_ge_0p25_min']}_"
        f"aj{spec['arg_text_jaccard_min']:.2f}_"
        f"ed{spec['event_count_delta_max']:.2f}_"
        f"am{spec['avg_margin_min']:.2f}"
    ).replace(".", "p").replace("-", "m")


def generate_rules():
    rules = [
        {
            "rule_id": "locked_structural_proxy",
            "kind": "baseline",
            "spec": {
                "fresh_margin_min": 0.25,
                "margin_range_max": 0.50,
                "num_margins_ge_0p25_min": 2,
                "arg_text_jaccard_min": 0.40,
                "event_count_delta_max": 0.0,
                "avg_margin_min": -999.0,
            },
            "fn": locked_rule,
        }
    ]
    seen = {"locked_structural_proxy"}
    for fresh, margin_range, num_ge, arg_j, event_delta, avg_margin in itertools.product(
        [0.25, 0.20, 0.15, 0.10, 0.00],
        [0.50, 0.60, 0.75, 1.00],
        [2, 1],
        [0.40, 0.35, 0.30, 0.25],
        [0.00, 0.25],
        [0.15, 0.10, 0.00, -999.0],
    ):
        spec = {
            "fresh_margin_min": fresh,
            "margin_range_max": margin_range,
            "num_margins_ge_0p25_min": num_ge,
            "arg_text_jaccard_min": arg_j,
            "event_count_delta_max": event_delta,
            "avg_margin_min": avg_margin,
        }
        rid = rule_id(spec)
        if rid in seen:
            continue
        seen.add(rid)
        rules.append({"rule_id": rid, "kind": "relaxed_grid", "spec": spec, "fn": lambda case, s=spec: relaxed_rule(case, s)})
    return rules


def evaluate_rule(cases_by_split, rule, split, source):
    split_cases = cases_by_split["test_seen"] + cases_by_split["test_unseen"] if split == "test" else cases_by_split[split]
    direct_rows = []
    routed_rows = []
    selected = []
    selected_set = set()
    helpful = set()
    for case in split_cases:
        direct = case[f"{source}_direct"]
        reason = case[f"{source}_reason"]
        gain = score(reason) - score(direct)
        direct_rows.append(direct)
        if gain > 0:
            helpful.add(case["case_id"])
        if rule["fn"](case):
            selected_set.add(case["case_id"])
            selected.append(gain)
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)
    direct_avg = avg_metrics(direct_rows)
    routed_avg = avg_metrics(routed_rows)
    tp = len(selected_set & helpful)
    fp = len(selected_set - helpful)
    fn = len(helpful - selected_set)
    return {
        "rule_id": rule["rule_id"],
        "kind": rule["kind"],
        "spec": rule["spec"],
        "split": split,
        "source": source,
        "num_examples": len(split_cases),
        "pred_reason_count": len(selected_set),
        "pred_reason_rate": len(selected_set) / len(split_cases) if split_cases else 0.0,
        "selected_reason_score_gain_mean": mean(selected),
        "selected_reason_harm_rate": mean(1.0 if gain < 0 else 0.0 for gain in selected),
        "selected_reason_helpful_count": sum(1 for gain in selected if gain > 0),
        "selected_reason_harmful_count": sum(1 for gain in selected if gain < 0),
        "selected_reason_neutral_count": sum(1 for gain in selected if gain == 0),
        "route_vs_helpful": route_prf(tp, fp, fn),
        "routed_minus_direct": {
            metric: routed_avg[metric] - direct_avg[metric]
            for metric in METRICS
        },
    }


def consolidate(rule, rows):
    by_key = {(row["source"], row["split"]): row for row in rows}
    single_test = by_key[("single_gen_execution", "test")]
    single_seen = by_key[("single_gen_execution", "test_seen")]
    single_unseen = by_key[("single_gen_execution", "test_unseen")]
    k2_test = by_key[("k2_expected", "test")]
    target_coverage = 0.06 <= single_test["pred_reason_rate"] <= 0.10
    beats_locked_candidate = (
        single_test["routed_minus_direct"]["score"] > 0.0085
        and single_seen["routed_minus_direct"]["score"] >= 0.0
        and single_unseen["routed_minus_direct"]["score"] >= 0.0
    )
    robust_positive = (
        single_test["routed_minus_direct"]["score"] > 0
        and single_seen["routed_minus_direct"]["score"] >= 0
        and single_unseen["routed_minus_direct"]["score"] >= 0
    )
    return {
        "rule_id": rule["rule_id"],
        "kind": rule["kind"],
        "spec": rule["spec"],
        "target_coverage": target_coverage,
        "robust_positive": robust_positive,
        "beats_locked_candidate": beats_locked_candidate,
        "test_score_delta": single_test["routed_minus_direct"]["score"],
        "seen_score_delta": single_seen["routed_minus_direct"]["score"],
        "unseen_score_delta": single_unseen["routed_minus_direct"]["score"],
        "k2_test_score_delta": k2_test["routed_minus_direct"]["score"],
        "test_reason_rate": single_test["pred_reason_rate"],
        "seen_reason_rate": single_seen["pred_reason_rate"],
        "unseen_reason_rate": single_unseen["pred_reason_rate"],
        "test_harm_rate": single_test["selected_reason_harm_rate"],
        "seen_harm_rate": single_seen["selected_reason_harm_rate"],
        "unseen_harm_rate": single_unseen["selected_reason_harm_rate"],
        "test_aet_delta": {
            metric: single_test["routed_minus_direct"][metric]
            for metric in ["argument_f1", "event_f1", "trigger_f1"]
        },
        "screen_score": min(
            single_test["routed_minus_direct"]["score"],
            single_seen["routed_minus_direct"]["score"],
            single_unseen["routed_minus_direct"]["score"],
        ),
    }


def compact_case(case):
    margins = [case["old17_18_margin"], case["new19_20_margin"], case["fresh_margin"]]
    return {
        "case_id": case["case_id"],
        "key": case["key"],
        "split": case["split"],
        "fresh_margin": case["fresh_margin"],
        "avg_margin": mean(margins),
        "margin_range": case["margin_range"],
        "num_margins_ge_0p25": case["num_margins_ge_0p25"],
        "sample_arg_text_jaccard_mean": case["sample_arg_text_jaccard_mean"],
        "sample_event_count_delta_mean": case["sample_event_count_delta_mean"],
        "single_gen_score_gain": case["single_gen_execution_reason"]["score"] - case["single_gen_execution_direct"]["score"],
        "k2_expected_score_gain": case["k2_expected_reason"]["score"] - case["k2_expected_direct"]["score"],
        "locked_structural_proxy": locked_rule(case),
    }


def render_table(rows, limit=30):
    lines = [
        "| rank | rule | target | robust | beats locked | reason test/seen/unseen | score test/seen/unseen | harm test/seen/unseen | A/E/T test |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows[:limit], 1):
        aet = row["test_aet_delta"]
        lines.append(
            f"| {idx} | `{row['rule_id']}` | `{row['target_coverage']}` | `{row['robust_positive']}` | `{row['beats_locked_candidate']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} | "
            f"{pct(row['test_harm_rate'])}/{pct(row['seen_harm_rate'])}/{pct(row['unseen_harm_rate'])} | "
            f"{signed(aet['argument_f1'])}/{signed(aet['event_f1'])}/{signed(aet['trigger_f1'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    locked = next(row for row in payload["consolidated"] if row["rule_id"] == "locked_structural_proxy")
    target = [row for row in payload["consolidated"] if row["target_coverage"]]
    robust_target = [row for row in target if row["robust_positive"]]
    beaters = [row for row in target if row["beats_locked_candidate"]]
    lines = [
        "# Sampled K2 StructProxy Relaxed Selector Sweep",
        "",
        "This sweep relaxes the locked structural proxy to test whether no-training selector coverage can increase from 3-4% toward 6-10%.",
        "",
        f"- rules evaluated: `{payload['num_rules']}`",
        f"- cases: `{payload['num_cases']}`",
        f"- target-coverage rules: `{len(target)}`",
        f"- robust target-coverage rules: `{len(robust_target)}`",
        f"- target-coverage rules beating locked score: `{len(beaters)}`",
        "",
        "## Locked Baseline",
        "",
        render_table([locked], limit=1),
        "",
        "## Leaderboard",
        "",
        render_table(payload["consolidated"], limit=30),
        "",
        "## Target Coverage Candidates",
        "",
    ]
    lines.append(render_table(target, limit=30) if target else "No rule reached the 6-10% target coverage band.")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- Locked structural proxy score delta test/seen/unseen: `{locked['test_score_delta']:+.4f}/{locked['seen_score_delta']:+.4f}/{locked['unseen_score_delta']:+.4f}` at reason rate `{locked['test_reason_rate']:.1%}`.",
            "- A relaxed selector is only promising if it reaches 6-10% reason rate, keeps seen non-negative, and improves aggregate single-generation score.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['report_json']}`",
            f"- cases JSONL: `{payload['cases_jsonl']}`",
            f"- evaluations JSONL: `{payload['evaluations_jsonl']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def sort_key(row):
    return (
        row["beats_locked_candidate"],
        row["target_coverage"] and row["robust_positive"],
        row["robust_positive"],
        row["target_coverage"],
        row["screen_score"],
        row["test_score_delta"],
        -row["test_harm_rate"],
    )


def run(args):
    build_args = argparse.Namespace(
        seeds=args.seeds,
        sample_root=args.sample_root,
        fresh_nll_root=args.fresh_nll_root,
        old_nll_root=args.old_nll_root,
        new_nll_root=args.new_nll_root,
        checkpoint=args.checkpoint,
    )
    cases_by_split = build_cases(build_args)
    rules = generate_rules()
    evaluations = []
    grouped = defaultdict(list)
    for rule in rules:
        for source in SOURCES:
            for split in SPLITS:
                row = evaluate_rule(cases_by_split, rule, split, source)
                evaluations.append(row)
                grouped[rule["rule_id"]].append(row)
    rule_by_id = {rule["rule_id"]: rule for rule in rules}
    consolidated = [consolidate(rule_by_id[rid], rows) for rid, rows in grouped.items()]
    consolidated.sort(key=sort_key, reverse=True)
    all_cases = cases_by_split["test_seen"] + cases_by_split["test_unseen"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases_jsonl = args.output_root / "cases.jsonl"
    eval_jsonl = args.output_root / "evaluations.jsonl"
    write_jsonl(cases_jsonl, [compact_case(case) for case in all_cases])
    write_jsonl(eval_jsonl, evaluations)
    payload = {
        "checkpoint": args.checkpoint,
        "seeds": args.seeds,
        "sample_root": args.sample_root.as_posix(),
        "fresh_nll_root": args.fresh_nll_root.as_posix(),
        "old_nll_root": args.old_nll_root.as_posix(),
        "new_nll_root": args.new_nll_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "num_cases": len(all_cases),
        "num_rules": len(rules),
        "consolidated": consolidated,
        "report_md": args.report_md.as_posix(),
        "report_json": args.report_json.as_posix(),
        "cases_jsonl": cases_jsonl.as_posix(),
        "evaluations_jsonl": eval_jsonl.as_posix(),
    }
    write_json(args.report_json, payload)
    write_json(args.output_root / "summary.json", payload)
    write_text(args.report_md, render_report(payload))
    print(json.dumps({"report_md": args.report_md.as_posix(), "report_json": args.report_json.as_posix()}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 24])
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--fresh-nll-root", type=Path, default=DEFAULT_FRESH_NLL_ROOT)
    parser.add_argument("--old-nll-root", type=Path, default=DEFAULT_OLD_NLL_ROOT)
    parser.add_argument("--new-nll-root", type=Path, default=DEFAULT_NEW_NLL_ROOT)
    parser.add_argument("--checkpoint", default="checkpoint-50")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
