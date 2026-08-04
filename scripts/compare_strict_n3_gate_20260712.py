#!/usr/bin/env python3
"""Aggregate three matched strict-protocol seeds and apply the fixed n=3 gate."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.compare_preference_run_gate_20260712 import (
    METRICS,
    as_object,
    filter_type,
    integrity,
    load_run,
    macro_values,
    micro_values,
    verify_pairing,
)


def load_seed_pairs(
    baseline_dirs: list[Path], candidate_dirs: list[Path]
) -> list[dict[str, Any]]:
    if len(baseline_dirs) != 3 or len(candidate_dirs) != 3:
        raise ValueError("strict n=3 comparison requires exactly three baseline and candidate runs")
    pairs = []
    reference_gold = None
    for baseline_dir, candidate_dir in zip(baseline_dirs, candidate_dirs):
        baseline_summary, baseline_rows = load_run(baseline_dir)
        candidate_summary, candidate_rows = load_run(candidate_dir)
        verify_pairing(
            baseline_rows,
            candidate_rows,
            require_wnd_id=True,
            require_input_match=True,
        )
        if reference_gold is not None:
            verify_pairing(
                reference_gold,
                baseline_rows,
                require_wnd_id=True,
                require_input_match=True,
            )
        else:
            reference_gold = baseline_rows
        pairs.append(
            {
                "baseline_dir": str(baseline_dir.resolve()),
                "candidate_dir": str(candidate_dir.resolve()),
                "baseline_summary": baseline_summary,
                "candidate_summary": candidate_summary,
                "baseline_rows": baseline_rows,
                "candidate_rows": candidate_rows,
            }
        )
    return pairs


def metric_mean(values: list[dict[str, float]]) -> dict[str, float]:
    return {metric: mean(item[metric] for item in values) for metric in METRICS}


def seed_aggregate(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_macro = [macro_values(pair["baseline_summary"]) for pair in pairs]
    candidate_macro = [macro_values(pair["candidate_summary"]) for pair in pairs]
    baseline_micro = [micro_values(pair["baseline_rows"]) for pair in pairs]
    candidate_micro = [micro_values(pair["candidate_rows"]) for pair in pairs]
    baseline_integrity = [integrity(pair["baseline_summary"]) for pair in pairs]
    candidate_integrity = [integrity(pair["candidate_summary"]) for pair in pairs]
    macro_base_mean = metric_mean(baseline_macro)
    macro_candidate_mean = metric_mean(candidate_macro)
    micro_base_mean = metric_mean(baseline_micro)
    micro_candidate_mean = metric_mean(candidate_micro)
    macro_delta = {
        metric: macro_candidate_mean[metric] - macro_base_mean[metric] for metric in METRICS
    }
    micro_delta = {
        metric: micro_candidate_mean[metric] - micro_base_mean[metric] for metric in METRICS
    }
    return {
        "seed_runs": [
            {
                "baseline": pair["baseline_dir"],
                "candidate": pair["candidate_dir"],
                "baseline_macro": baseline_macro[index],
                "candidate_macro": candidate_macro[index],
                "macro_delta": {
                    metric: candidate_macro[index][metric] - baseline_macro[index][metric]
                    for metric in METRICS
                },
                "baseline_micro": baseline_micro[index],
                "candidate_micro": candidate_micro[index],
            }
            for index, pair in enumerate(pairs)
        ],
        "baseline_macro_mean": macro_base_mean,
        "candidate_macro_mean": macro_candidate_mean,
        "macro_mean_delta": macro_delta,
        "baseline_micro_mean": micro_base_mean,
        "candidate_micro_mean": micro_candidate_mean,
        "micro_mean_delta": micro_delta,
        "macro_micro_directions_match": all(
            (macro_delta[metric] >= 0) == (micro_delta[metric] >= 0) for metric in METRICS
        ),
        "baseline_integrity_mean": {
            key: mean(item[key] for item in baseline_integrity) for key in ("json", "offset")
        },
        "candidate_integrity_mean": {
            key: mean(item[key] for item in candidate_integrity) for key in ("json", "offset")
        },
    }


def paired_bootstrap_n3(
    pairs: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, dict[str, float]]:
    row_count = len(pairs[0]["baseline_rows"])
    rng = random.Random(seed)
    distributions = {metric: [] for metric in METRICS}
    for _ in range(samples):
        indices = [rng.randrange(row_count) for _ in range(row_count)]
        per_seed_delta = []
        for pair in pairs:
            baseline = micro_values(pair["baseline_rows"], indices)
            candidate = micro_values(pair["candidate_rows"], indices)
            per_seed_delta.append(
                {metric: candidate[metric] - baseline[metric] for metric in METRICS}
            )
        for metric in METRICS:
            distributions[metric].append(mean(item[metric] for item in per_seed_delta))
    result = {}
    for metric in METRICS:
        ordered = sorted(distributions[metric])
        result[metric] = {
            "point": mean(
                micro_values(pair["candidate_rows"])[metric]
                - micro_values(pair["baseline_rows"])[metric]
                for pair in pairs
            ),
            "lower_95": ordered[int(0.025 * samples)],
            "upper_95": ordered[min(int(0.975 * samples), samples - 1)],
        }
    return result


def per_type_mean_delta(pairs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    event_types = sorted(
        {
            event.get("event_type")
            for row in pairs[0]["baseline_rows"]
            for event in as_object(row.get("gold")).get("events", [])
            if isinstance(event, dict) and event.get("event_type")
        }
    )
    output = {}
    for event_type in event_types:
        seed_deltas = []
        for pair in pairs:
            baseline_filtered = []
            candidate_filtered = []
            for baseline_row, candidate_row in zip(
                pair["baseline_rows"], pair["candidate_rows"]
            ):
                gold = filter_type(as_object(baseline_row.get("gold")), event_type)
                baseline_filtered.append(
                    {
                        "gold": gold,
                        "predicted": filter_type(
                            as_object(baseline_row.get("predicted")), event_type
                        ),
                    }
                )
                candidate_filtered.append(
                    {
                        "gold": gold,
                        "predicted": filter_type(
                            as_object(candidate_row.get("predicted")), event_type
                        ),
                    }
                )
            baseline_values = micro_values(baseline_filtered)
            candidate_values = micro_values(candidate_filtered)
            seed_deltas.append(
                {
                    metric: candidate_values[metric] - baseline_values[metric]
                    for metric in METRICS
                }
            )
        values = metric_mean(seed_deltas)
        values["mean"] = mean(values.values())
        output[event_type] = values
    return output


def apply_gate(unseen: dict[str, Any], per_type: dict[str, dict[str, float]]) -> dict[str, Any]:
    nonnegative_types = sum(values["mean"] >= 0 for values in per_type.values())
    bootstrap = unseen["paired_bootstrap"]
    checks = {
        "all_three_macro_means_above_direct": all(
            value > 0 for value in unseen["macro_mean_delta"].values()
        ),
        "macro_micro_directions_match": unseen["macro_micro_directions_match"],
        "argument_ci_lower_positive": bootstrap["argument"]["lower_95"] > 0,
        "trigger_ci_lower_positive": bootstrap["trigger"]["lower_95"] > 0,
        "event_point_positive": bootstrap["event"]["point"] > 0,
        "at_least_six_nonnegative_types": nonnegative_types >= 6,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "nonnegative_type_count": nonnegative_types,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Strict Protocol N=3 Comparison", ""]
    for split in ("seen", "unseen"):
        values = report["splits"][split]
        lines.extend(
            [
                f"## {split.title()}",
                "",
                "| metric | Direct mean | candidate mean | macro delta | micro delta | paired 95% CI |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric in METRICS:
            ci = values["paired_bootstrap"][metric]
            lines.append(
                f"| {metric} | {values['baseline_macro_mean'][metric]:.4f} | "
                f"{values['candidate_macro_mean'][metric]:.4f} | "
                f"{values['macro_mean_delta'][metric]:+.4f} | "
                f"{values['micro_mean_delta'][metric]:+.4f} | "
                f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
            )
        lines.append("")
    lines.extend(["## Gate", "", f"Passed: **{report['gate']['passed']}**", ""])
    for name, passed in report["gate"]["checks"].items():
        lines.append(f"- `{name}`: {passed}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_seen", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline_unseen", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate_seen", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate_unseen", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    seen_pairs = load_seed_pairs(args.baseline_seen, args.candidate_seen)
    unseen_pairs = load_seed_pairs(args.baseline_unseen, args.candidate_unseen)
    seen = seed_aggregate(seen_pairs)
    unseen = seed_aggregate(unseen_pairs)
    seen["paired_bootstrap"] = paired_bootstrap_n3(
        seen_pairs, args.bootstrap_samples, args.seed
    )
    unseen["paired_bootstrap"] = paired_bootstrap_n3(
        unseen_pairs, args.bootstrap_samples, args.seed + 1
    )
    per_type = per_type_mean_delta(unseen_pairs)
    report = {
        "splits": {"seen": seen, "unseen": unseen},
        "unseen_per_type_mean_delta": per_type,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }
    report["gate"] = apply_gate(unseen, per_type)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report)
    (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if report["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
