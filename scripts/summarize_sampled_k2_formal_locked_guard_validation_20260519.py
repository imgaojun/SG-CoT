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
METRICS = ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json_rate"]
DEFAULT_BRANCH = "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
DEFAULT_RUN_ID = "sampled_reason_expert_forcedreason_from_noaux_20260517_checkpoint-258"
DEFAULT_SAMPLE_ROOT = REPO / "outputs/stage2_modular_dualexpert/formal_k2_counterfactual_utility_20260518" / DEFAULT_RUN_ID
DEFAULT_NLL_ROOT = REPO / "outputs/stage2_adaptive_route_formal_nll_locked_guard_seedpair21_22_20260519" / DEFAULT_BRANCH
DEFAULT_EXEC_ROOT = REPO / "outputs/stage2_adaptive_route_formal_execution_20260518/sampledk2_ckpt50_margin025"
DEFAULT_OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_locked_guard_validation_20260519/seedpair21_22"
DEFAULT_REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_formal_locked_guard_seedpair21_22_validation.md"
DEFAULT_REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_formal_locked_guard_seedpair21_22_validation.json"


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def avg_metrics(rows):
    if not rows:
        return {metric: 0.0 for metric in METRICS}
    return {metric: mean(row[metric] for row in rows) for metric in METRICS}


def load_sampled_metrics(sample_root: Path, split: str, route: str, seeds):
    grouped = defaultdict(list)
    for seed in seeds:
        path = sample_root / split / route / f"seed-{seed}" / "predictions.jsonl"
        for row in load_jsonl(path):
            key = key_for(row)
            if key:
                grouped[key].append(metric_dict(row))
    out = {}
    for key, rows in grouped.items():
        if len(rows) != len(seeds):
            raise ValueError(f"{split}/{route}/{key}: expected {len(seeds)} samples, got {len(rows)}")
        out[key] = avg_metrics(rows)
    return out


def load_execution_metrics(exec_root: Path, split: str, route: str):
    path = exec_root / f"forced_{route}" / split / "predictions.jsonl"
    return {key_for(row): metric_dict(row) for row in load_jsonl(path) if key_for(row)}


def load_margins(nll_root: Path, checkpoint: str, split: str):
    path = nll_root / checkpoint / split / "scores.jsonl"
    return {
        key_for(row): row.get("delta_direct_minus_reason_route_nll")
        for row in load_jsonl(path)
        if key_for(row)
    }


def build_cases(args):
    cases = {}
    for split in SPLITS:
        margins = load_margins(args.nll_root, args.checkpoint, split)
        sampled_metrics = {
            route: load_sampled_metrics(args.sample_root, split, route, args.seeds)
            for route in ROUTES
        }
        sample_rows = {
            route: load_sample_rows(args.sample_root, split, route, args.seeds)
            for route in ROUTES
        }
        exec_metrics = {
            route: load_execution_metrics(args.exec_root, split, route)
            for route in ROUTES
        }
        keys = set(margins)
        for route in ROUTES:
            keys &= set(sampled_metrics[route])
            keys &= set(sample_rows[route])
            keys &= set(exec_metrics[route])
        split_cases = []
        for key in sorted(keys):
            direct_feat = aggregate_sample_rows(sample_rows["direct"][key])
            reason_feat = aggregate_sample_rows(sample_rows["reason"][key])
            event_delta = reason_feat.get("event_count_mean", 0.0) - direct_feat.get("event_count_mean", 0.0)
            argument_delta = reason_feat.get("argument_count_mean", 0.0) - direct_feat.get("argument_count_mean", 0.0)
            split_cases.append(
                {
                    "split": split,
                    "key": key,
                    "case_id": f"{split}::{key}",
                    "fresh_margin": margins[key],
                    "reason_minus_direct_event_count_mean": event_delta,
                    "reason_minus_direct_argument_count_mean": argument_delta,
                    "k2_expected_direct": sampled_metrics["direct"][key],
                    "k2_expected_reason": sampled_metrics["reason"][key],
                    "single_gen_execution_direct": exec_metrics["direct"][key],
                    "single_gen_execution_reason": exec_metrics["reason"][key],
                }
            )
        if not split_cases:
            raise ValueError(f"no cases for {split}")
        cases[split] = split_cases
    return cases


def route_reason(case, policy):
    if policy == "fresh_margin":
        return case["fresh_margin"] >= 0.25
    if policy == "locked_event_guard":
        return case["fresh_margin"] >= 0.25 and case["reason_minus_direct_event_count_mean"] >= 0
    raise KeyError(policy)


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
        if route_reason(case, policy):
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
        "reason_all": avg_metrics(reason_rows),
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
                "passes_locked_validation": test > 0 and seen > 0 and unseen >= 0,
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


def render_report(payload):
    screen = payload["screen"]
    single_guard = next(
        row for row in screen
        if row["source"] == "single_gen_execution" and row["policy"] == "locked_event_guard"
    )
    k2_guard = next(
        row for row in screen
        if row["source"] == "k2_expected" and row["policy"] == "locked_event_guard"
    )
    lines = [
        "# Sampled K2 Formal Locked Guard Seedpair21/22 Validation",
        "",
        "This run validates a fixed rule on fresh sampled evidence. It does not sweep thresholds or introduce new policies.",
        "",
        "Locked rule:",
        "",
        "```text",
        "fresh checkpoint-50 route-NLL margin >= 0.25",
        "and reason_minus_direct_event_count_mean >= 0",
        "```",
        "",
        f"- seeds: `{payload['seeds']}`",
        f"- sample root: `{payload['sample_root']}`",
        f"- route-NLL root: `{payload['nll_root']}`",
        "",
        "## Validation Screen",
        "",
        "| source | policy | pass | reason rate test/seen/unseen | score delta test/seen/unseen |",
        "|---|---|---:|---:|---:|",
    ]
    for row in screen:
        lines.append(
            f"| `{row['source']}` | `{row['policy']}` | `{row['passes_locked_validation']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} |"
        )
    lines.extend(
        [
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
            f"- Single-generation locked guard score delta test/seen/unseen: `{single_guard['test_score_delta']:+.4f}/{single_guard['seen_score_delta']:+.4f}/{single_guard['unseen_score_delta']:+.4f}`.",
            f"- K2-expected locked guard score delta test/seen/unseen: `{k2_guard['test_score_delta']:+.4f}/{k2_guard['seen_score_delta']:+.4f}/{k2_guard['unseen_score_delta']:+.4f}`.",
        ]
    )
    if single_guard["passes_locked_validation"] and k2_guard["passes_locked_validation"]:
        lines.append("- The locked guard passes this fresh-seed validation criterion.")
    else:
        lines.append("- The locked guard does not pass the fresh-seed validation criterion; do not promote it to trainable supervision yet.")
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
            for policy in ["fresh_margin", "locked_event_guard"]:
                results.append(summarize(cases, split, source, policy))
    screen = []
    for source in ["single_gen_execution", "k2_expected"]:
        screen.extend(screen_rows(results, source))
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": args.seeds,
        "sample_root": args.sample_root.as_posix(),
        "nll_root": args.nll_root.as_posix(),
        "exec_root": args.exec_root.as_posix(),
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[21, 22])
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--nll-root", type=Path, default=DEFAULT_NLL_ROOT)
    parser.add_argument("--exec-root", type=Path, default=DEFAULT_EXEC_ROOT)
    parser.add_argument("--checkpoint", default="checkpoint-50")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
