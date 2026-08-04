#!/usr/bin/env python3
"""Aggregate the three G9 ORPO seeds and apply the fixed GoLLIE Event-gap gate."""

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
    parser.add_argument("--g9_start_seen", type=Path, nargs="+", required=True)
    parser.add_argument("--g9_start_unseen", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate_seen", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate_unseen", type=Path, nargs="+", required=True)
    parser.add_argument("--gollie_seen", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    seen_pairs = load_seed_pairs(args.g9_start_seen, args.candidate_seen)
    unseen_pairs = load_seed_pairs(args.g9_start_unseen, args.candidate_unseen)
    gollie_pairs = load_seed_pairs(args.gollie_seen, args.candidate_seen)
    seen = seed_aggregate(seen_pairs)
    unseen = seed_aggregate(unseen_pairs)
    seen["paired_bootstrap"] = paired_bootstrap_n3(
        seen_pairs, args.bootstrap_samples, args.seed
    )
    unseen["paired_bootstrap"] = paired_bootstrap_n3(
        unseen_pairs, args.bootstrap_samples, args.seed + 1
    )
    versus_gollie = seed_aggregate(gollie_pairs)
    versus_gollie["paired_bootstrap"] = paired_bootstrap_n3(
        gollie_pairs, args.bootstrap_samples, args.seed + 2
    )
    event_gap = (
        versus_gollie["baseline_macro_mean"]["event"]
        - versus_gollie["candidate_macro_mean"]["event"]
    )
    event_ci = versus_gollie["paired_bootstrap"]["event"]
    ci_intersects_zero = event_ci["lower_95"] <= 0 <= event_ci["upper_95"]
    checks = {
        "gollie_seen_event_gap_at_most_0.020_or_ci_intersects_zero": (
            event_gap <= 0.020 or ci_intersects_zero
        )
    }
    report = {
        "g9_start_vs_orpo": {"seen": seen, "unseen": unseen},
        "gollie_vs_orpo_seen": versus_gollie,
        "gollie_seen_event_gap": event_gap,
        "gollie_event_ci_intersects_zero": ci_intersects_zero,
        "gate": {"passed": all(checks.values()), "checks": checks},
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }
    markdown = "\n".join(
        [
            "# G9 ORPO N=3 Gate",
            "",
            f"G9 start mean seen Event: `{seen['baseline_macro_mean']['event']:.4f}`",
            f"G9 ORPO mean seen Event: `{seen['candidate_macro_mean']['event']:.4f}`",
            f"GoLLIE mean seen Event: `{versus_gollie['baseline_macro_mean']['event']:.4f}`",
            f"GoLLIE minus ORPO Event gap: `{event_gap:+.4f}`",
            f"ORPO minus GoLLIE paired 95% CI: `[{event_ci['lower_95']:+.4f}, {event_ci['upper_95']:+.4f}]`",
            "",
            f"Gate passed: **{report['gate']['passed']}**",
            "",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if report["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
