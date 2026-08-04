#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_goldfree_harmful_cases_20260519 import (  # noqa: E402
    aggregate_pair_features,
    load_exec_rows,
    load_margins,
    load_sample_rows,
    avg_metric_dict,
)
from scripts.diagnose_sampled_k2_formal_unseen_false_positives_20260519 import metric_dict  # noqa: E402
from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


SPLITS = ["test_seen", "test_unseen"]
ROUTES = ["direct", "reason"]
METRICS = ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json"]
DEFAULT_SAMPLE_ROOT = REPO / (
    "outputs/stage2_modular_dualexpert/formal_k2_counterfactual_utility_20260518/"
    "sampled_reason_expert_forcedreason_from_noaux_20260517_checkpoint-258"
)
DEFAULT_FRESH_NLL_ROOT = REPO / (
    "outputs/stage2_adaptive_route_formal_nll_structural_proxy_seedpair23_24_20260519/"
    "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
)
DEFAULT_OLD_NLL_ROOT = REPO / (
    "outputs/stage2_adaptive_route_formal_nll_20260518/"
    "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
)
DEFAULT_NEW_NLL_ROOT = REPO / (
    "outputs/stage2_adaptive_route_formal_nll_seedpair19_20_20260518/"
    "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
)
DEFAULT_OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_structural_proxy_validation_20260519/seedpair23_24"
DEFAULT_REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_structural_proxy_locked_seedpair23_24_validation.md"
DEFAULT_REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_structural_proxy_locked_seedpair23_24_validation.json"


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def avg_metrics(rows):
    if not rows:
        return {metric: 0.0 for metric in METRICS}
    return {metric: mean(row[metric] for row in rows) for metric in METRICS}


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def build_cases(args):
    cases = {}
    for split in SPLITS:
        fresh = load_margins(args.fresh_nll_root, split, args.checkpoint)
        old = load_margins(args.old_nll_root, split, args.checkpoint)
        new = load_margins(args.new_nll_root, split, args.checkpoint)
        sample_rows = {
            route: load_sample_rows(args.sample_root, split, route, args.seeds)
            for route in ROUTES
        }
        exec_rows = {route: load_exec_rows(split, route) for route in ROUTES}
        keys = set(fresh) & set(old) & set(new)
        for route in ROUTES:
            keys &= set(sample_rows[route])
            keys &= set(exec_rows[route])
        split_cases = []
        for key in sorted(keys):
            direct_samples = sample_rows["direct"][key]
            reason_samples = sample_rows["reason"][key]
            pair = aggregate_pair_features(direct_samples, reason_samples)
            margins = [old[key], new[key], fresh[key]]
            split_cases.append(
                {
                    "split": split,
                    "key": key,
                    "case_id": f"{split}::{key}",
                    "fresh_margin": fresh[key],
                    "old17_18_margin": old[key],
                    "new19_20_margin": new[key],
                    "margin_range": max(margins) - min(margins),
                    "num_margins_ge_0p25": sum(1 for value in margins if value >= 0.25),
                    "sample_arg_text_jaccard_mean": pair["arg_text_jaccard_mean"],
                    "sample_event_count_delta_mean": pair["event_count_delta_mean"],
                    "k2_expected_direct": avg_metric_dict(direct_samples),
                    "k2_expected_reason": avg_metric_dict(reason_samples),
                    "single_gen_execution_direct": metric_dict(exec_rows["direct"][key]),
                    "single_gen_execution_reason": metric_dict(exec_rows["reason"][key]),
                }
            )
        if not split_cases:
            raise ValueError(f"no cases for {split}")
        cases[split] = split_cases
    return cases


def route_reason(case, policy):
    if policy == "fresh_margin":
        return case["fresh_margin"] >= 0.25
    if policy == "base_margin_stability":
        return (
            case["fresh_margin"] >= 0.25
            and case["margin_range"] <= 0.50
            and case["num_margins_ge_0p25"] >= 2
        )
    if policy == "locked_structural_proxy":
        return (
            case["fresh_margin"] >= 0.25
            and case["margin_range"] <= 0.50
            and case["num_margins_ge_0p25"] >= 2
            and case["sample_arg_text_jaccard_mean"] >= 0.40
            and case["sample_event_count_delta_mean"] <= 0.0
        )
    raise KeyError(policy)


def summarize(cases, split, source, policy):
    direct_rows = []
    routed_rows = []
    selected = []
    selected_set = set()
    helpful = set()
    for case in cases:
        direct = case[f"{source}_direct"]
        reason = case[f"{source}_reason"]
        gain = reason["score"] - direct["score"]
        direct_rows.append(direct)
        if gain > 0:
            helpful.add(case["case_id"])
        if route_reason(case, policy):
            selected_set.add(case["case_id"])
            selected.append(gain)
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)
    direct_summary = avg_metrics(direct_rows)
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
        "routed": routed_summary,
        "routed_minus_direct": {
            metric: routed_summary[metric] - direct_summary[metric]
            for metric in METRICS
        },
    }


def screen_rows(results, source):
    by_policy = defaultdict(dict)
    for row in results:
        if row["source"] == source:
            by_policy[row["policy"]][row["split"]] = row
    screen = []
    for policy, splits in by_policy.items():
        test = splits["test"]["routed_minus_direct"]["score"]
        seen = splits["test_seen"]["routed_minus_direct"]["score"]
        unseen = splits["test_unseen"]["routed_minus_direct"]["score"]
        screen.append(
            {
                "source": source,
                "policy": policy,
                "test_score_delta": test,
                "seen_score_delta": seen,
                "unseen_score_delta": unseen,
                "test_reason_rate": splits["test"]["pred_reason_rate"],
                "seen_reason_rate": splits["test_seen"]["pred_reason_rate"],
                "unseen_reason_rate": splits["test_unseen"]["pred_reason_rate"],
                "test_harm_rate": splits["test"]["selected_reason_harm_rate"],
                "seen_harm_rate": splits["test_seen"]["selected_reason_harm_rate"],
                "unseen_harm_rate": splits["test_unseen"]["selected_reason_harm_rate"],
                "passes_locked_validation": test > 0 and seen > 0 and unseen >= 0,
                "passes_low_harm": (
                    test > 0
                    and seen > 0
                    and unseen >= 0
                    and splits["test"]["selected_reason_harm_rate"] <= 0.15
                    and splits["test_seen"]["selected_reason_harm_rate"] <= 0.15
                ),
            }
        )
    return screen


def metric_cell(row):
    return f"{fmt(row['argument_f1'])}/{fmt(row['event_f1'])}/{fmt(row['trigger_f1'])}/{fmt(row['score'])}"


def delta_cell(row):
    return f"{signed(row['argument_f1'])}/{signed(row['event_f1'])}/{signed(row['trigger_f1'])}/{signed(row['score'])}"


def render_table(rows, source):
    lines = [
        "| split | policy | reason rate | routed A/E/T/Score | delta A/E/T/Score | selected gain | harm rate | P/R/F1 vs helpful |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"test": 0, "test_seen": 1, "test_unseen": 2}
    for row in sorted([r for r in rows if r["source"] == source], key=lambda r: (order[r["split"]], r["policy"])):
        prf = row["route_vs_helpful"]
        lines.append(
            f"| `{row['split']}` | `{row['policy']}` | {pct(row['pred_reason_rate'])} | "
            f"{metric_cell(row['routed'])} | {delta_cell(row['routed_minus_direct'])} | "
            f"{signed(row['selected_reason_score_gain_mean'])} | {pct(row['selected_reason_harm_rate'])} | "
            f"{fmt(prf['precision'])}/{fmt(prf['recall'])}/{fmt(prf['f1'])} |"
        )
    return "\n".join(lines)


def render_screen(rows):
    lines = [
        "| source | policy | pass | low harm | reason test/seen/unseen | score test/seen/unseen | harm test/seen/unseen |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['source']}` | `{row['policy']}` | `{row['passes_locked_validation']}` | `{row['passes_low_harm']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} | "
            f"{pct(row['test_harm_rate'])}/{pct(row['seen_harm_rate'])}/{pct(row['unseen_harm_rate'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    structural = next(
        row for row in payload["screen"]
        if row["source"] == "single_gen_execution" and row["policy"] == "locked_structural_proxy"
    )
    lines = [
        "# Sampled K2 Structural Proxy Locked Seedpair23/24 Validation",
        "",
        "This validates the exact structural proxy rule on fresh sampled evidence. No thresholds are swept or retuned.",
        "",
        "Locked rule:",
        "",
        "```text",
        "fresh_margin >= 0.25",
        "and margin_range(old17/18, new19/20, fresh23/24) <= 0.50",
        "and at least 2 of 3 margins >= 0.25",
        "and sample_arg_text_jaccard_mean >= 0.40",
        "and sample_event_count_delta_mean <= 0",
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
        f"- Locked structural proxy single-gen score delta test/seen/unseen: `{structural['test_score_delta']:+.4f}/{structural['seen_score_delta']:+.4f}/{structural['unseen_score_delta']:+.4f}`.",
        f"- Locked structural proxy harm test/seen/unseen: `{structural['test_harm_rate']:.1%}/{structural['seen_harm_rate']:.1%}/{structural['unseen_harm_rate']:.1%}`.",
    ]
    if structural["passes_low_harm"]:
        lines.append("- The locked structural proxy passes the low-harm validation criterion.")
    elif structural["passes_locked_validation"]:
        lines.append("- The locked structural proxy passes positive-delta validation but not low-harm validation.")
    else:
        lines.append("- The locked structural proxy does not pass locked validation.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['report_json']}`",
            f"- output summary: `{Path(payload['output_root']) / 'summary.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    cases_by_split = build_cases(args)
    case_sets = {"test": [case for split in SPLITS for case in cases_by_split[split]], **cases_by_split}
    results = []
    for split, cases in case_sets.items():
        for source in ["k2_expected", "single_gen_execution"]:
            for policy in ["fresh_margin", "base_margin_stability", "locked_structural_proxy"]:
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 24])
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
