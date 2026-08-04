#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_sampled_confident_router_dev_20260518 import (  # noqa: E402
    fmt,
    load_json,
    load_jsonl,
    pct,
    signed,
    write_json,
    write_text,
)


BRANCH = "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
DEFAULT_OUTPUT_ROOT = f"outputs/stage2_adaptive_route_formal_nll_20260518/{BRANCH}"
DEFAULT_REPORT_STEM = "2026-05-18_stage2_sampled_k2_formal_route_nll_probe"
CHECKPOINTS = ["checkpoint-50", "checkpoint-75"]
SPLITS = ["test_seen", "test_unseen"]
MARGIN_THRESHOLDS = [0.0, 0.05, 0.10, 0.20, 0.25, 0.30]
PRE_REGISTERED = {
    ("checkpoint-50", "margin_ge_0p25"),
    ("checkpoint-75", "margin_ge_0p05"),
}


def prediction_key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id")


def score_value(row):
    return row.get("trigger_f1", 0.0) * 0.25 + row.get("argument_f1", 0.0) + row.get("event_f1", 0.0)


def load_sample_metrics(sample_root: Path):
    metrics = {}
    for split in SPLITS:
        metrics[split] = {}
        for route in ["direct", "reason"]:
            by_key = defaultdict(list)
            for seed in [17, 18]:
                path = sample_root / split / route / f"seed-{seed}" / "predictions.jsonl"
                if not path.exists():
                    raise FileNotFoundError(path)
                for row in load_jsonl(path):
                    key = prediction_key(row)
                    if key:
                        by_key[key].append(row)
            metrics[split][route] = {
                key: {
                    "trigger_f1": sum(row.get("trigger_f1", 0.0) for row in rows) / len(rows),
                    "argument_f1": sum(row.get("argument_f1", 0.0) for row in rows) / len(rows),
                    "event_f1": sum(row.get("event_f1", 0.0) for row in rows) / len(rows),
                    "score": sum(score_value(row) for row in rows) / len(rows),
                    "valid_json_rate": sum(1 for row in rows if row.get("valid_final_json", row.get("valid_json"))) / len(rows),
                    "sample_count": len(rows),
                }
                for key, rows in by_key.items()
            }
    return metrics


def route_from_threshold(score_rows, threshold):
    routes = {}
    for row in score_rows:
        key = prediction_key(row)
        if not key:
            continue
        delta = row.get("delta_direct_minus_reason_route_nll")
        routes[key] = "reason" if delta is not None and delta >= threshold else "direct"
    return routes


def summarize_split(checkpoint, split, policy, threshold, pred_routes, metrics):
    keys = sorted(set(pred_routes) & set(metrics[split]["direct"]) & set(metrics[split]["reason"]))
    if not keys:
        raise ValueError(f"no common keys for {checkpoint}/{split}/{policy}")
    sums = defaultdict(float)
    selected_reason = 0
    selected_reason_gains = []
    for key in keys:
        direct = metrics[split]["direct"][key]
        reason = metrics[split]["reason"][key]
        route = pred_routes.get(key, "direct")
        routed = reason if route == "reason" else direct
        if route == "reason":
            selected_reason += 1
            selected_reason_gains.append(reason["score"] - direct["score"])
        for metric in ["trigger_f1", "argument_f1", "event_f1", "score", "valid_json_rate"]:
            sums[f"direct_{metric}"] += direct[metric]
            sums[f"reason_{metric}"] += reason[metric]
            sums[f"routed_{metric}"] += routed[metric]
    n = len(keys)
    out = {
        "checkpoint": checkpoint,
        "split": split,
        "policy": policy,
        "threshold": threshold,
        "num_examples": n,
        "pred_reason_count": selected_reason,
        "pred_reason_rate": selected_reason / n,
        "selected_reason_avg_k2_score_gain": (
            sum(selected_reason_gains) / len(selected_reason_gains) if selected_reason_gains else 0.0
        ),
    }
    for route in ["direct", "reason", "routed"]:
        out[route] = {
            metric: sums[f"{route}_{metric}"] / n
            for metric in ["trigger_f1", "argument_f1", "event_f1", "score", "valid_json_rate"]
        }
    out["routed_minus_direct"] = {
        metric: out["routed"][metric] - out["direct"][metric]
        for metric in ["trigger_f1", "argument_f1", "event_f1", "score", "valid_json_rate"]
    }
    out["routed_minus_reason_all"] = {
        metric: out["routed"][metric] - out["reason"][metric]
        for metric in ["trigger_f1", "argument_f1", "event_f1", "score", "valid_json_rate"]
    }
    return out


def aggregate_test(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] in SPLITS:
            grouped[(row["checkpoint"], row["policy"])].append(row)
    out = []
    for (checkpoint, policy), items in grouped.items():
        total = sum(row["num_examples"] for row in items)
        agg = {
            "checkpoint": checkpoint,
            "split": "test",
            "policy": policy,
            "threshold": items[0]["threshold"],
            "num_examples": total,
            "pred_reason_count": sum(row["pred_reason_count"] for row in items),
        }
        agg["pred_reason_rate"] = agg["pred_reason_count"] / total if total else 0.0
        for key in ["selected_reason_avg_k2_score_gain"]:
            denom = sum(row["pred_reason_count"] for row in items)
            agg[key] = (
                sum(row[key] * row["pred_reason_count"] for row in items) / denom
                if denom
                else 0.0
            )
        for route in ["direct", "reason", "routed"]:
            agg[route] = {}
            for metric in ["trigger_f1", "argument_f1", "event_f1", "score", "valid_json_rate"]:
                agg[route][metric] = sum(row[route][metric] * row["num_examples"] for row in items) / total
        agg["routed_minus_direct"] = {
            metric: agg["routed"][metric] - agg["direct"][metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1", "score", "valid_json_rate"]
        }
        agg["routed_minus_reason_all"] = {
            metric: agg["routed"][metric] - agg["reason"][metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1", "score", "valid_json_rate"]
        }
        out.append(agg)
    return out


def render_table(rows):
    lines = [
        "| checkpoint | policy | split | reason rate | routed A/E/T/Score | delta vs direct A/E/T/Score | delta vs reason-all Score |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        routed = row["routed"]
        delta = row["routed_minus_direct"]
        reason_delta = row["routed_minus_reason_all"]
        lines.append(
            f"| `{row['checkpoint']}` | `{row['policy']}` | `{row['split']}` | {pct(row['pred_reason_rate'])} | "
            f"{fmt(routed['argument_f1'])}/{fmt(routed['event_f1'])}/{fmt(routed['trigger_f1'])}/{fmt(routed['score'])} | "
            f"{signed(delta['argument_f1'])}/{signed(delta['event_f1'])}/{signed(delta['trigger_f1'])}/{signed(delta['score'])} | "
            f"{signed(reason_delta['score'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    prereg = [
        row
        for row in payload["results"]
        if (row["checkpoint"], row["policy"]) in PRE_REGISTERED
    ]
    prereg.sort(key=lambda row: (row["checkpoint"], row["policy"], row["split"]))
    diagnostics = sorted(
        payload["results"],
        key=lambda row: (
            row["split"] != "test",
            row["routed_minus_direct"]["score"],
            row["routed_minus_direct"]["event_f1"],
        ),
        reverse=True,
    )[:24]
    lines = [
        "# Sampled K2 Formal Route-NLL Probe",
        "",
        "This report evaluates pre-registered fixed route-NLL margin policies on formal K2 compact-evidence prompts. Full `test` is aggregated from `test_seen` and `test_unseen`; it is not separately sampled.",
        "",
        f"- branch: `{payload['branch']}`",
        f"- sample root: `{payload['sample_root']}`",
        f"- route-NLL root: `{payload['output_root']}`",
        "",
        "## Pre-Registered Policies",
        "",
        render_table(prereg),
        "",
        "## Best Diagnostics",
        "",
        render_table(diagnostics),
        "",
        "## Reading",
        "",
    ]
    test_rows = [row for row in prereg if row["split"] == "test"]
    for row in test_rows:
        delta = row["routed_minus_direct"]
        lines.append(
            f"- `{row['checkpoint']}/{row['policy']}` on aggregated `test`: reason rate `{row['pred_reason_rate']:.1%}`, "
            f"score delta `{delta['score']:+.4f}`, argument/event/trigger deltas "
            f"`{delta['argument_f1']:+.4f}/{delta['event_f1']:+.4f}/{delta['trigger_f1']:+.4f}`."
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- JSON: `{payload['output_json']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    output_root = REPO / args.output_root
    sample_root = REPO / args.sample_root
    metrics = load_sample_metrics(sample_root)
    results = []
    for checkpoint in args.checkpoints:
        for split in SPLITS:
            score_path = output_root / checkpoint / split / "scores.jsonl"
            summary_path = output_root / checkpoint / split / "summary.json"
            if not score_path.exists():
                raise FileNotFoundError(score_path)
            score_rows = load_jsonl(score_path)
            score_summary = load_json(summary_path) if summary_path.exists() else {}
            for threshold in args.margin_thresholds:
                policy = f"margin_ge_{threshold:.2f}".replace(".", "p")
                pred_routes = route_from_threshold(score_rows, threshold)
                row = summarize_split(checkpoint, split, policy, threshold, pred_routes, metrics)
                row["score_summary"] = score_summary
                row["pre_registered"] = (checkpoint, policy) in PRE_REGISTERED
                results.append(row)
    results.extend(aggregate_test(results))
    for row in results:
        row["pre_registered"] = (row["checkpoint"], row["policy"]) in PRE_REGISTERED

    output_json = Path(args.output_json) if args.output_json else REPO / f"reports/artifacts/{args.report_stem}.json"
    output_md = Path(args.output_md) if args.output_md else REPO / f"reports/{args.report_stem}.md"
    payload = {
        "branch": args.branch,
        "checkpoints": args.checkpoints,
        "splits": SPLITS,
        "sample_root": sample_root.as_posix(),
        "output_root": output_root.as_posix(),
        "margin_thresholds": args.margin_thresholds,
        "pre_registered_policies": sorted([list(item) for item in PRE_REGISTERED]),
        "results": results,
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
    parser.add_argument(
        "--sample_root",
        default=(
            "outputs/stage2_modular_dualexpert/formal_k2_counterfactual_utility_20260518/"
            "sampled_reason_expert_forcedreason_from_noaux_20260517_checkpoint-258"
        ),
    )
    parser.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    parser.add_argument("--margin_thresholds", nargs="+", type=float, default=MARGIN_THRESHOLDS)
    parser.add_argument("--report_stem", default=DEFAULT_REPORT_STEM)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
