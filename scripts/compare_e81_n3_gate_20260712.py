#!/usr/bin/env python3
"""Aggregate matched E81 start/ORPO runs for seeds 42, 8322, and 8333."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.compare_strict_n3_gate_20260712 import (  # noqa: E402
    load_seed_pairs,
    paired_bootstrap_n3,
    seed_aggregate,
)


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
    unseen_deltas = list(unseen["macro_mean_delta"].values())
    integrity_checks = {}
    for split_name, values in (("seen", seen), ("unseen", unseen)):
        for key in ("json", "offset"):
            integrity_checks[f"{split_name}_{key}_drop_at_most_0.01"] = (
                values["candidate_integrity_mean"][key]
                - values["baseline_integrity_mean"][key]
                >= -0.01
            )
    checks = {
        "unseen_mean_delta_at_least_0.015": sum(unseen_deltas) / 3 >= 0.015,
        "at_least_two_unseen_metrics_improve": sum(value > 0 for value in unseen_deltas)
        >= 2,
        "no_unseen_metric_below_minus_0.010": min(unseen_deltas) >= -0.010,
        "no_seen_metric_below_minus_0.015": min(seen["macro_mean_delta"].values())
        >= -0.015,
        **integrity_checks,
    }
    report = {
        "splits": {"seen": seen, "unseen": unseen},
        "gate": {"passed": all(checks.values()), "checks": checks},
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }
    lines = ["# E81 ORPO N=3 Comparison", ""]
    for split_name in ("seen", "unseen"):
        values = report["splits"][split_name]
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                "| metric | start mean | ORPO mean | delta | paired 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in ("argument", "event", "trigger"):
            ci = values["paired_bootstrap"][metric]
            lines.append(
                f"| {metric} | {values['baseline_macro_mean'][metric]:.4f} | "
                f"{values['candidate_macro_mean'][metric]:.4f} | "
                f"{values['macro_mean_delta'][metric]:+.4f} | "
                f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
            )
        lines.append("")
    lines.extend(["## Gate", "", f"Passed: **{report['gate']['passed']}**", ""])
    for name, passed in checks.items():
        lines.append(f"- `{name}`: {passed}")
    lines.append("")
    markdown = "\n".join(lines)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if report["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
