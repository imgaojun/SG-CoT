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
    load_generated_pred_routes,
    load_json,
    load_label_map,
    pct,
    signed,
    summarize_routes,
    write_json,
    write_text,
)


BRANCH = "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
DEFAULT_CHECKPOINTS = ["checkpoint-50"]
DEFAULT_OUTPUT_ROOT = (
    "outputs/stage2_adaptive_route_seedpair_transfer_20260518/"
    f"{BRANCH}"
)
DEFAULT_REPORT_STEM = "2026-05-18_stage2_sampled_k2_seedpair_transfer_router_checkpoint50"


def labels_cell(labels):
    return (
        f"R={labels.get('stable_reason', 0)}, "
        f"D={labels.get('stable_direct', 0)}, "
        f"A={labels.get('ambiguous', 0)}"
    )


def delta_cell(delta):
    return (
        f"{signed(delta['argument_f1'])}/"
        f"{signed(delta['event_f1'])}/"
        f"{signed(delta['trigger_f1'])}/"
        f"{signed(delta['score'])}"
    )


def aggregate_by_checkpoint(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[row["checkpoint"]].append(row)
    aggregates = []
    for checkpoint in sorted(grouped, key=lambda name: int(name.split("-", 1)[1])):
        rows = grouped[checkpoint]
        scores = [row["sampled_expected_routed_minus_direct"]["score"] for row in rows]
        triggers = [row["sampled_expected_routed_minus_direct"]["trigger_f1"] for row in rows]
        arguments = [row["sampled_expected_routed_minus_direct"]["argument_f1"] for row in rows]
        events = [row["sampled_expected_routed_minus_direct"]["event_f1"] for row in rows]
        f1s = [row["route_vs_confident_label"]["f1"] for row in rows]
        pred_rates = [row["pred_reason_rate"] for row in rows]
        aggregates.append(
            {
                "checkpoint": checkpoint,
                "seed_pair_count": len(rows),
                "pred_reason_mean": sum(pred_rates) / len(pred_rates),
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


def render_report(payload):
    title = (
        "Sampled K2 Seed-Pair Transfer Router Checkpoint-50"
        if payload["checkpoints"] == ["checkpoint-50"]
        else "Sampled K2 Seed-Pair Transfer Router Checkpoint Sweep"
    )
    desc = (
        "This report evaluates the already-trained K=2 compact-evidence router `checkpoint-50` "
        "on dev_seen prompts whose K=2 evidence is recomputed from different seed pairs."
        if payload["checkpoints"] == ["checkpoint-50"]
        else (
            "This report evaluates multiple checkpoints from the already-trained K=2 compact-evidence router "
            "on dev_seen prompts whose K=2 evidence is recomputed from different seed pairs."
        )
    )
    lines = [
        f"# {title}",
        "",
        f"{desc} The K=8 stable_reason/stable_direct labels remain the evaluation target.",
        "",
        f"- branch: `{payload['branch']}`",
        "",
        "## Results",
        "",
        "| checkpoint | seed pair | pred reason | P/R/F1 vs K8 stable_reason | selected labels | routed delta A/E/T/Score | avg selected gain | route accuracy |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["results"]:
        prf = row["route_vs_confident_label"]
        delta = row["sampled_expected_routed_minus_direct"]
        lines.append(
            f"| `{row['checkpoint']}` | `{row['seed_pair']}` | {pct(row['pred_reason_rate'])} | "
            f"{fmt(prf['precision'], 3)}/{fmt(prf['recall'], 3)}/{fmt(prf['f1'], 3)} | "
            f"{labels_cell(row['selected_reason_utility_labels'])} | "
            f"{delta_cell(delta)} | "
            f"{fmt(row['selected_reason_avg_sampled_gain'], 4)} | "
            f"{fmt(row['route_accuracy_vs_confident_label'], 3)} |"
        )

    aggregates = aggregate_by_checkpoint(payload["results"])
    score_deltas = [row["sampled_expected_routed_minus_direct"]["score"] for row in payload["results"]]
    trigger_deltas = [row["sampled_expected_routed_minus_direct"]["trigger_f1"] for row in payload["results"]]
    lines.extend(
        [
            "",
            "## Checkpoint Aggregate",
            "",
            "| checkpoint | pred reason mean | F1 mean | score min/mean/max | trigger min/mean | argument mean | event mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| `{row['checkpoint']}` | {pct(row['pred_reason_mean'])} | {fmt(row['reason_f1_mean'], 3)} | "
            f"{signed(row['score_min'])}/{signed(row['score_mean'])}/{signed(row['score_max'])} | "
            f"{signed(row['trigger_min'])}/{signed(row['trigger_mean'])} | "
            f"{signed(row['argument_mean'])} | {signed(row['event_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
        f"- score delta min/mean/max over all rows: `{min(score_deltas):+.4f}` / `{sum(score_deltas) / len(score_deltas):+.4f}` / `{max(score_deltas):+.4f}`.",
        f"- trigger delta min/mean/max over all rows: `{min(trigger_deltas):+.4f}` / `{sum(trigger_deltas) / len(trigger_deltas):+.4f}` / `{max(trigger_deltas):+.4f}`.",
            "- Passing criterion for formal expansion: all seed pairs should keep nonnegative score delta, and most should keep nonnegative trigger delta.",
            "",
            "## Inputs",
            "",
            f"- branch: `{payload['branch']}`",
            f"- output root: `{payload['output_root']}`",
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
        checkpoint_root = output_root / checkpoint
        for pair_name, _seeds in SEED_PAIRS:
            pair_slug = f"seedpair{pair_name}"
            pred_path = checkpoint_root / pair_slug / "predictions.jsonl"
            summary_path = checkpoint_root / pair_slug / "summary.json"
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            pred_routes = load_generated_pred_routes(pred_path)
            result = summarize_routes(f"{checkpoint}_gen_{pair_slug}", pred_routes, label_map)
            result["checkpoint"] = checkpoint
            result["seed_pair"] = pair_name
            result["generation_summary"] = load_json(summary_path) if summary_path.exists() else {}
            results.append(result)

    output_json = Path(args.output_json) if args.output_json else REPO / f"reports/artifacts/{args.report_stem}.json"
    output_md = Path(args.output_md) if args.output_md else REPO / f"reports/{args.report_stem}.md"
    payload = {
        "branch": args.branch,
        "checkpoints": args.checkpoints,
        "output_root": output_root.as_posix(),
        "label_path": label_path.as_posix(),
        "results": results,
        "checkpoint_aggregates": aggregate_by_checkpoint(results),
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
    parser.add_argument("--report_stem", default=DEFAULT_REPORT_STEM)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
