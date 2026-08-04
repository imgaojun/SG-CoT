#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.analyze_sampled_k2_seedpair_robustness_20260518 import SEED_PAIRS  # noqa: E402
from scripts.summarize_sampled_confident_router_dev_20260518 import (  # noqa: E402
    LABEL_PATH,
    fmt,
    load_json,
    load_jsonl,
    load_label_map,
    pct,
    signed,
    summarize_routes,
    write_json,
    write_text,
)


BRANCH = "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
DEFAULT_CHECKPOINTS = ["checkpoint-25", "checkpoint-50", "checkpoint-75", "checkpoint-100"]
DEFAULT_OUTPUT_ROOT = f"outputs/stage2_adaptive_route_seedpair_nll_20260518/{BRANCH}"
DEFAULT_REPORT_STEM = "2026-05-18_stage2_sampled_k2_seedpair_route_nll_margin_calibration"
MARGIN_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
TOP_BUDGETS = [0.03, 0.05, 0.076, 0.10, 0.15, 0.20]


def route_from_threshold(score_rows, threshold):
    return {
        row["wnd_id"]: (
            "reason"
            if row.get("delta_direct_minus_reason_route_nll") is not None
            and row["delta_direct_minus_reason_route_nll"] >= threshold
            else "direct"
        )
        for row in score_rows
        if row.get("wnd_id")
    }


def route_from_top_budget(score_rows, budget):
    sortable = [
        (float(row["delta_direct_minus_reason_route_nll"]), row["wnd_id"])
        for row in score_rows
        if row.get("wnd_id") and row.get("delta_direct_minus_reason_route_nll") is not None
    ]
    sortable.sort(reverse=True)
    cap = round(len(sortable) * budget)
    reason_keys = {key for _delta, key in sortable[:cap]}
    return {key: ("reason" if key in reason_keys else "direct") for _delta, key in sortable}


def aggregate_results(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["checkpoint"], row["policy"])].append(row)

    aggregates = []
    for (checkpoint, policy), rows in sorted(
        grouped.items(), key=lambda item: (int(item[0][0].split("-", 1)[1]), item[0][1])
    ):
        scores = [row["sampled_expected_routed_minus_direct"]["score"] for row in rows]
        triggers = [row["sampled_expected_routed_minus_direct"]["trigger_f1"] for row in rows]
        arguments = [row["sampled_expected_routed_minus_direct"]["argument_f1"] for row in rows]
        events = [row["sampled_expected_routed_minus_direct"]["event_f1"] for row in rows]
        f1s = [row["route_vs_confident_label"]["f1"] for row in rows]
        precisions = [row["route_vs_confident_label"]["precision"] for row in rows]
        recalls = [row["route_vs_confident_label"]["recall"] for row in rows]
        pred_rates = [row["pred_reason_rate"] for row in rows]
        aggregates.append(
            {
                "checkpoint": checkpoint,
                "policy": policy,
                "policy_kind": rows[0]["policy_kind"],
                "seed_pair_count": len(rows),
                "pred_reason_mean": sum(pred_rates) / len(pred_rates),
                "precision_mean": sum(precisions) / len(precisions),
                "recall_mean": sum(recalls) / len(recalls),
                "reason_f1_mean": sum(f1s) / len(f1s),
                "score_min": min(scores),
                "score_mean": sum(scores) / len(scores),
                "score_max": max(scores),
                "trigger_min": min(triggers),
                "trigger_mean": sum(triggers) / len(triggers),
                "argument_mean": sum(arguments) / len(arguments),
                "event_mean": sum(events) / len(events),
            }
        )
    return aggregates


def policy_rank(row):
    return (
        row["score_min"],
        row["score_mean"],
        row["trigger_min"],
        row["trigger_mean"],
        row["reason_f1_mean"],
    )


def render_report(payload):
    aggregates = payload["aggregates"]
    margin_aggs = [row for row in aggregates if row["policy_kind"] == "margin"]
    top_aggs = [row for row in aggregates if row["policy_kind"] == "top_budget"]
    top_margin = sorted(margin_aggs, key=policy_rank, reverse=True)[:20]
    top_any = sorted(aggregates, key=policy_rank, reverse=True)[:20]

    lines = [
        "# Sampled K2 Seed-Pair Route-NLL Margin Calibration",
        "",
        "This report evaluates fixed route-NLL margin thresholds on K2 compact-evidence dev prompts recomputed from multiple seed pairs. Top-k budgets are included as diagnostics, but the decision target is a non-top-k policy.",
        "",
        f"- branch: `{payload['branch']}`",
        f"- output root: `{payload['output_root']}`",
        "",
        "## Best Fixed-Margin Policies",
        "",
        "| checkpoint | policy | pred reason mean | P/R/F1 mean | score min/mean/max | trigger min/mean | argument mean | event mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_margin:
        lines.append(
            f"| `{row['checkpoint']}` | `{row['policy']}` | {pct(row['pred_reason_mean'])} | "
            f"{fmt(row['precision_mean'], 3)}/{fmt(row['recall_mean'], 3)}/{fmt(row['reason_f1_mean'], 3)} | "
            f"{signed(row['score_min'])}/{signed(row['score_mean'])}/{signed(row['score_max'])} | "
            f"{signed(row['trigger_min'])}/{signed(row['trigger_mean'])} | "
            f"{signed(row['argument_mean'])} | {signed(row['event_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Best Overall Diagnostics",
            "",
            "| checkpoint | policy | kind | pred reason mean | P/R/F1 mean | score min/mean/max | trigger min/mean |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_any:
        lines.append(
            f"| `{row['checkpoint']}` | `{row['policy']}` | `{row['policy_kind']}` | {pct(row['pred_reason_mean'])} | "
            f"{fmt(row['precision_mean'], 3)}/{fmt(row['recall_mean'], 3)}/{fmt(row['reason_f1_mean'], 3)} | "
            f"{signed(row['score_min'])}/{signed(row['score_mean'])}/{signed(row['score_max'])} | "
            f"{signed(row['trigger_min'])}/{signed(row['trigger_mean'])} |"
        )

    passing = [
        row
        for row in margin_aggs
        if row["pred_reason_mean"] > 0.0
        and row["score_min"] >= 0.0
        and row["trigger_min"] >= -0.002
    ]
    near_passing = [
        row
        for row in margin_aggs
        if row["pred_reason_mean"] > 0.0
        and row["score_min"] >= 0.0
        and row["trigger_min"] >= -0.003
    ]
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- nontrivial fixed-margin policies meeting score_min >= 0 and trigger_min >= -0.002: `{len(passing)}`.",
        ]
    )
    if passing:
        best_pass = sorted(passing, key=policy_rank, reverse=True)[0]
        lines.append(
            f"- best passing fixed-margin policy: `{best_pass['checkpoint']}/{best_pass['policy']}` "
            f"with score min/mean `{best_pass['score_min']:+.4f}/{best_pass['score_mean']:+.4f}` "
            f"and trigger min/mean `{best_pass['trigger_min']:+.4f}/{best_pass['trigger_mean']:+.4f}`."
        )
    else:
        best_margin = top_margin[0] if top_margin else None
        if best_margin:
            lines.append(
                f"- best fixed-margin policy by robust score is `{best_margin['checkpoint']}/{best_margin['policy']}`, "
                f"but it has score min/mean `{best_margin['score_min']:+.4f}/{best_margin['score_mean']:+.4f}` "
                f"and trigger min/mean `{best_margin['trigger_min']:+.4f}/{best_margin['trigger_mean']:+.4f}`."
            )
    if near_passing:
        best_near = sorted(near_passing, key=policy_rank, reverse=True)[0]
        lines.append(
            f"- best near-passing fixed-margin policy with trigger_min >= -0.003: "
            f"`{best_near['checkpoint']}/{best_near['policy']}`, score min/mean "
            f"`{best_near['score_min']:+.4f}/{best_near['score_mean']:+.4f}`, trigger min/mean "
            f"`{best_near['trigger_min']:+.4f}/{best_near['trigger_mean']:+.4f}`."
        )
    lines.extend(
        [
            "- Formal expansion gate: prefer a fixed-margin policy with nonnegative score on every seed pair and controlled trigger harm.",
            "",
            "## Inputs",
            "",
            f"- label path: `{payload['label_path']}`",
            f"- JSON: `{payload['output_json']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    output_root = REPO / args.output_root
    label_path = REPO / LABEL_PATH
    label_map = load_label_map(label_path)
    results = []
    for checkpoint in args.checkpoints:
        for pair_name, _seeds in SEED_PAIRS:
            pair_slug = f"seedpair{pair_name}"
            score_path = output_root / checkpoint / pair_slug / "scores.jsonl"
            summary_path = output_root / checkpoint / pair_slug / "summary.json"
            if not score_path.exists():
                raise FileNotFoundError(score_path)
            score_rows = load_jsonl(score_path)
            score_summary = load_json(summary_path) if summary_path.exists() else {}
            for threshold in args.margin_thresholds:
                policy = f"margin_ge_{threshold:.2f}".replace(".", "p")
                pred_routes = route_from_threshold(score_rows, threshold)
                result = summarize_routes(f"{checkpoint}_nll_{policy}_{pair_slug}", pred_routes, label_map)
                result.update(
                    {
                        "checkpoint": checkpoint,
                        "seed_pair": pair_name,
                        "policy": policy,
                        "policy_kind": "margin",
                        "threshold": threshold,
                        "score_summary": score_summary,
                    }
                )
                results.append(result)
            for budget in args.top_budgets:
                policy = f"top{int(budget * 1000):03d}"
                pred_routes = route_from_top_budget(score_rows, budget)
                result = summarize_routes(f"{checkpoint}_nll_{policy}_{pair_slug}", pred_routes, label_map)
                result.update(
                    {
                        "checkpoint": checkpoint,
                        "seed_pair": pair_name,
                        "policy": policy,
                        "policy_kind": "top_budget",
                        "budget": budget,
                        "score_summary": score_summary,
                    }
                )
                results.append(result)

    output_json = Path(args.output_json) if args.output_json else REPO / f"reports/artifacts/{args.report_stem}.json"
    output_md = Path(args.output_md) if args.output_md else REPO / f"reports/{args.report_stem}.md"
    payload = {
        "branch": args.branch,
        "checkpoints": args.checkpoints,
        "output_root": output_root.as_posix(),
        "label_path": label_path.as_posix(),
        "margin_thresholds": args.margin_thresholds,
        "top_budgets": args.top_budgets,
        "results": results,
        "aggregates": aggregate_results(results),
        "output_json": output_json.as_posix(),
        "output_md": output_md.as_posix(),
    }
    write_json(output_json, payload)
    write_text(output_md, render_report(payload))
    print(json.dumps({"output_json": output_json.as_posix(), "output_md": output_md.as_posix()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--margin_thresholds", nargs="+", type=float, default=MARGIN_THRESHOLDS)
    parser.add_argument("--top_budgets", nargs="+", type=float, default=TOP_BUDGETS)
    parser.add_argument("--report_stem", default=DEFAULT_REPORT_STEM)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
