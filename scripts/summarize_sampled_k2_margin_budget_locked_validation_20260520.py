#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402
from scripts.summarize_sampled_k2_structural_proxy_locked_validation_20260519 import (  # noqa: E402
    DEFAULT_NEW_NLL_ROOT,
    DEFAULT_OLD_NLL_ROOT,
    DEFAULT_SAMPLE_ROOT,
    METRICS,
    SPLITS,
    avg_metrics,
    build_cases,
)


DEFAULT_FRESH_NLL_ROOT = REPO / (
    "outputs/stage2_adaptive_route_formal_nll_margin_budget_seedpair25_26_20260520/"
    "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
)
DEFAULT_OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_margin_budget_validation_20260520/seedpair25_26_top15"
DEFAULT_REPORT_MD = REPO / "reports/2026-05-20_stage2_locked_margin_budget_top15_seedpair25_26_validation.md"
DEFAULT_REPORT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_locked_margin_budget_top15_seedpair25_26_validation.json"


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def avg_margin(case):
    return mean([case["fresh_margin"], case["old17_18_margin"], case["new19_20_margin"]])


def relaxed_selector(case):
    return (
        case["fresh_margin"] >= 0.25
        and case["margin_range"] <= 0.75
        and case["num_margins_ge_0p25"] >= 1
        and case["sample_arg_text_jaccard_mean"] >= 0.40
        and case["sample_event_count_delta_mean"] <= 0.0
        and avg_margin(case) >= 0.0
    )


def locked_structural_proxy(case):
    return (
        case["fresh_margin"] >= 0.25
        and case["margin_range"] <= 0.50
        and case["num_margins_ge_0p25"] >= 2
        and case["sample_arg_text_jaccard_mean"] >= 0.40
        and case["sample_event_count_delta_mean"] <= 0.0
    )


def ranked_relaxed(cases):
    selected = [case for case in cases if relaxed_selector(case)]
    return sorted(
        selected,
        key=lambda case: (
            -case["num_margins_ge_0p25"],
            -avg_margin(case),
            case["margin_range"],
            case["case_id"],
        ),
    )


def selected_ids(cases, policy):
    if policy == "direct_only":
        return set()
    if policy == "relaxed_full_reason":
        return {case["case_id"] for case in cases if relaxed_selector(case)}
    if policy == "locked_structural_proxy":
        return {case["case_id"] for case in cases if locked_structural_proxy(case)}
    if policy.startswith("margin_budget_top"):
        budget = int(policy.removeprefix("margin_budget_top"))
        return {case["case_id"] for case in ranked_relaxed(cases)[:budget]}
    raise KeyError(policy)


def summarize(cases, split, source, policy):
    selected = selected_ids(cases, policy)
    direct_rows = []
    routed_rows = []
    gains = []
    helpful = set()
    for case in cases:
        direct = case[f"{source}_direct"]
        reason = case[f"{source}_reason"]
        gain = reason["score"] - direct["score"]
        direct_rows.append(direct)
        if gain > 0:
            helpful.add(case["case_id"])
        if case["case_id"] in selected:
            gains.append(gain)
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)
    direct_avg = avg_metrics(direct_rows)
    routed_avg = avg_metrics(routed_rows)
    buckets = Counter("helpful" if gain > 0 else "harmful" if gain < 0 else "neutral" for gain in gains)
    tp = len(selected & helpful)
    fp = len(selected - helpful)
    fn = len(helpful - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "split": split,
        "source": source,
        "policy": policy,
        "num_examples": len(cases),
        "pred_reason_count": len(selected),
        "pred_reason_rate": len(selected) / len(cases) if cases else 0.0,
        "selected_helpful": buckets["helpful"],
        "selected_harmful": buckets["harmful"],
        "selected_neutral": buckets["neutral"],
        "selected_reason_score_gain_mean": mean(gains),
        "selected_reason_harm_rate": buckets["harmful"] / len(gains) if gains else 0.0,
        "route_vs_helpful": {"precision": precision, "recall": recall, "f1": f1},
        "direct": direct_avg,
        "routed": routed_avg,
        "routed_minus_direct": {metric: routed_avg[metric] - direct_avg[metric] for metric in METRICS},
    }


def screen_rows(results, source):
    by_policy = defaultdict(dict)
    for row in results:
        if row["source"] == source:
            by_policy[row["policy"]][row["split"]] = row
    out = []
    for policy, splits in by_policy.items():
        test = splits["test"]
        seen = splits["test_seen"]
        unseen = splits["test_unseen"]
        out.append(
            {
                "source": source,
                "policy": policy,
                "test_score_delta": test["routed_minus_direct"]["score"],
                "seen_score_delta": seen["routed_minus_direct"]["score"],
                "unseen_score_delta": unseen["routed_minus_direct"]["score"],
                "test_reason_rate": test["pred_reason_rate"],
                "seen_reason_rate": seen["pred_reason_rate"],
                "unseen_reason_rate": unseen["pred_reason_rate"],
                "test_harm_rate": test["selected_reason_harm_rate"],
                "seen_harm_rate": seen["selected_reason_harm_rate"],
                "unseen_harm_rate": unseen["selected_reason_harm_rate"],
                "passes_target": (
                    test["routed_minus_direct"]["score"] > 0.0085
                    and seen["routed_minus_direct"]["score"] >= 0.0042
                    and test["selected_reason_harm_rate"] <= 0.16
                    and seen["selected_reason_harm_rate"] <= 0.20
                    and unseen["routed_minus_direct"]["score"] >= 0.0200
                ),
            }
        )
    return out


def metric_cell(row):
    return f"{fmt(row['argument_f1'])}/{fmt(row['event_f1'])}/{fmt(row['trigger_f1'])}/{fmt(row['score'])}"


def delta_cell(row):
    return f"{signed(row['argument_f1'])}/{signed(row['event_f1'])}/{signed(row['trigger_f1'])}/{signed(row['score'])}"


def render_screen(rows):
    lines = [
        "| source | policy | pass | reason test/seen/unseen | score test/seen/unseen | harm test/seen/unseen |",
        "|---|---|---:|---:|---:|---:|",
    ]
    order = {"margin_budget_top15": 0, "margin_budget_top10": 1, "margin_budget_top20": 2}
    for row in sorted(rows, key=lambda r: (r["source"], order.get(r["policy"], 9), r["policy"])):
        lines.append(
            f"| `{row['source']}` | `{row['policy']}` | `{row['passes_target']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} | "
            f"{pct(row['test_harm_rate'])}/{pct(row['seen_harm_rate'])}/{pct(row['unseen_harm_rate'])} |"
        )
    return "\n".join(lines)


def render_table(rows, source):
    lines = [
        "| split | policy | reason rate | delta A/E/T/Score | selected H/h/N | selected gain | harm | P/R/F1 vs helpful |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    split_order = {"test": 0, "test_seen": 1, "test_unseen": 2}
    policy_order = {"margin_budget_top15": 0, "margin_budget_top10": 1, "margin_budget_top20": 2}
    for row in sorted([r for r in rows if r["source"] == source], key=lambda r: (split_order[r["split"]], policy_order.get(r["policy"], 9), r["policy"])):
        prf = row["route_vs_helpful"]
        lines.append(
            f"| `{row['split']}` | `{row['policy']}` | {pct(row['pred_reason_rate'])} | "
            f"{delta_cell(row['routed_minus_direct'])} | "
            f"{row['selected_helpful']}/{row['selected_harmful']}/{row['selected_neutral']} | "
            f"{signed(row['selected_reason_score_gain_mean'])} | {pct(row['selected_reason_harm_rate'])} | "
            f"{fmt(prf['precision'])}/{fmt(prf['recall'])}/{fmt(prf['f1'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    main = next(row for row in payload["screen"] if row["source"] == "single_gen_execution" and row["policy"] == "margin_budget_top15")
    lines = [
        "# Locked Margin-Budget Top15 Seedpair Validation",
        "",
        "This validates the locked relaxed-selector plus margin-budget top15 policy. Top10/top20 are reported only as robustness checks.",
        "",
        "Locked policy:",
        "",
        "```text",
        "candidate filter: relaxed selector",
        "rank: num_margins_ge_0p25 desc, avg_margin desc, margin_range asc, case_id asc",
        "budget: top15",
        "```",
        "",
        f"- seeds: `{payload['seeds']}`",
        f"- sample root: `{payload['sample_root']}`",
        f"- fresh route-NLL root: `{payload['fresh_nll_root']}`",
        "",
        "## Validation Screen",
        "",
        render_screen(payload["screen"]),
        "",
        "## Single-Generation Execution",
        "",
        render_table(payload["results"], "single_gen_execution"),
        "",
        "## K2 Expected",
        "",
        render_table(payload["results"], "k2_expected"),
        "",
        "## Reading",
        "",
        f"- Locked top15 single-gen score delta test/seen/unseen: `{main['test_score_delta']:+.4f}/{main['seen_score_delta']:+.4f}/{main['unseen_score_delta']:+.4f}`.",
        f"- Locked top15 harm test/seen/unseen: `{main['test_harm_rate']:.1%}/{main['seen_harm_rate']:.1%}/{main['unseen_harm_rate']:.1%}`.",
        f"- Locked top15 passes target: `{main['passes_target']}`.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['report_json']}`",
        f"- output summary: `{Path(payload['output_root']) / 'summary.json'}`",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    cases_by_split = build_cases(args)
    case_sets = {"test": [case for split in SPLITS for case in cases_by_split[split]], **cases_by_split}
    policies = [
        "margin_budget_top15",
        "margin_budget_top10",
        "margin_budget_top20",
        "relaxed_full_reason",
        "locked_structural_proxy",
        "direct_only",
    ]
    results = []
    for split, cases in case_sets.items():
        for source in ["single_gen_execution", "k2_expected"]:
            for policy in policies:
                results.append(summarize(cases, split, source, policy))
    screen = []
    for source in ["single_gen_execution", "k2_expected"]:
        screen.extend(screen_rows(results, source))
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": args.seeds,
        "sample_root": args.sample_root.as_posix(),
        "fresh_nll_root": args.fresh_nll_root.as_posix(),
        "old_nll_root": args.old_nll_root.as_posix(),
        "new_nll_root": args.new_nll_root.as_posix(),
        "checkpoint": args.checkpoint,
        "output_root": args.output_root.as_posix(),
        "splits": {split: len(cases) for split, cases in cases_by_split.items()},
        "results": results,
        "screen": screen,
        "report_md": args.report_md.as_posix(),
        "report_json": args.report_json.as_posix(),
    }
    write_json(args.report_json, payload)
    write_json(args.output_root / "summary.json", payload)
    write_text(args.report_md, render_report(payload))
    print(json.dumps({"report_md": args.report_md.as_posix(), "report_json": args.report_json.as_posix()}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[25, 26])
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--fresh-nll-root", type=Path, default=DEFAULT_FRESH_NLL_ROOT)
    parser.add_argument("--old-nll-root", type=Path, default=DEFAULT_OLD_NLL_ROOT)
    parser.add_argument("--new-nll-root", type=Path, default=DEFAULT_NEW_NLL_ROOT)
    parser.add_argument("--checkpoint", default="checkpoint-50")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
