#!/usr/bin/env python3
"""Compare paired EE runs, compute macro/micro deltas, bootstrap CIs, and apply fixed gates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.rescore_surface_predictions_20260712 import (  # noqa: E402
    as_object,
    f1_from_counts,
    micro_counts,
)

METRICS = ("argument", "event", "trigger")


def load_run(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "predictions.jsonl").open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return summary, rows


def macro_values(summary: dict[str, Any]) -> dict[str, float]:
    return {metric: float(summary[f"{metric}_f1"]) for metric in METRICS}


def aggregate_counts(rows: list[dict[str, Any]], indices: list[int] | None = None):
    totals = {metric: [0, 0, 0] for metric in METRICS}
    selected = range(len(rows)) if indices is None else indices
    for index in selected:
        row = rows[index]
        counts = micro_counts(as_object(row.get("predicted")), as_object(row.get("gold")))
        for metric in METRICS:
            totals[metric] = [
                left + right for left, right in zip(totals[metric], counts[metric])
            ]
    return totals


def micro_values(rows: list[dict[str, Any]], indices: list[int] | None = None) -> dict[str, float]:
    totals = aggregate_counts(rows, indices)
    return {metric: f1_from_counts(totals[metric]) for metric in METRICS}


def gold_signature(row: dict[str, Any]) -> str:
    return json.dumps(as_object(row.get("gold")), sort_keys=True, separators=(",", ":"))


def verify_pairing(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    require_wnd_id: bool = False,
    require_input_match: bool = False,
) -> None:
    if len(baseline) != len(candidate):
        raise ValueError(f"row count mismatch: {len(baseline)} != {len(candidate)}")
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        left_id = left.get("meta", {}).get("wnd_id")
        right_id = right.get("meta", {}).get("wnd_id")
        if require_wnd_id and (not left_id or not right_id):
            raise ValueError(f"missing wnd_id at row {index}")
        if left_id and right_id and left_id != right_id:
            raise ValueError(f"wnd_id mismatch at row {index}: {left_id} != {right_id}")
        if require_input_match and left.get("input") != right.get("input"):
            raise ValueError(f"input mismatch at row {index}")
        if gold_signature(left) != gold_signature(right):
            raise ValueError(f"gold mismatch at row {index}")


def bootstrap_deltas(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    verify_pairing(baseline, candidate)
    rng = random.Random(seed)
    distributions = {metric: [] for metric in METRICS}
    for _ in range(samples):
        indices = [rng.randrange(len(baseline)) for _ in range(len(baseline))]
        baseline_values = micro_values(baseline, indices)
        candidate_values = micro_values(candidate, indices)
        for metric in METRICS:
            distributions[metric].append(candidate_values[metric] - baseline_values[metric])
    point_baseline = micro_values(baseline)
    point_candidate = micro_values(candidate)
    result = {}
    for metric in METRICS:
        ordered = sorted(distributions[metric])
        lower = ordered[int(0.025 * samples)]
        upper = ordered[min(int(0.975 * samples), samples - 1)]
        result[metric] = {
            "point": point_candidate[metric] - point_baseline[metric],
            "lower_95": lower,
            "upper_95": upper,
        }
    return result


def filter_type(payload: dict[str, Any], event_type: str) -> dict[str, Any]:
    events = payload.get("events", []) if isinstance(payload, dict) else []
    return {
        "events": [
            event
            for event in events
            if isinstance(event, dict) and event.get("event_type") == event_type
        ]
    }


def per_type_deltas(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    event_types = sorted(
        {
            event.get("event_type")
            for row in baseline
            for event in as_object(row.get("gold")).get("events", [])
            if isinstance(event, dict) and event.get("event_type")
        }
    )
    output = {}
    for event_type in event_types:
        baseline_rows = []
        candidate_rows = []
        for left, right in zip(baseline, candidate):
            gold = filter_type(as_object(left.get("gold")), event_type)
            baseline_rows.append(
                {"gold": gold, "predicted": filter_type(as_object(left.get("predicted")), event_type)}
            )
            candidate_rows.append(
                {"gold": gold, "predicted": filter_type(as_object(right.get("predicted")), event_type)}
            )
        base_values = micro_values(baseline_rows)
        candidate_values = micro_values(candidate_rows)
        deltas = {
            metric: candidate_values[metric] - base_values[metric] for metric in METRICS
        }
        deltas["mean"] = sum(deltas.values()) / len(METRICS)
        output[event_type] = deltas
    return output


def integrity(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "json": float(summary.get("final_json_valid_rate", summary.get("json_valid_rate", 0.0))),
        "offset": float(summary.get("offset_recovery_full_rate", 0.0)),
    }


def e81_gate(comparison: dict[str, Any]) -> dict[str, Any]:
    unseen = comparison["splits"]["unseen"]["macro_delta"]
    seen = comparison["splits"]["seen"]["macro_delta"]
    unseen_values = [unseen[metric] for metric in METRICS]
    checks = {
        "unseen_mean_delta_at_least_0.015": sum(unseen_values) / 3 >= 0.015,
        "at_least_two_unseen_metrics_improve": sum(value > 0 for value in unseen_values) >= 2,
        "no_unseen_metric_below_minus_0.010": min(unseen_values) >= -0.010,
        "no_seen_metric_below_minus_0.015": min(seen.values()) >= -0.015,
    }
    for split in ("seen", "unseen"):
        for name in ("json", "offset"):
            checks[f"{split}_{name}_drop_at_most_0.01"] = (
                comparison["splits"][split]["integrity_delta"][name] >= -0.01
            )
    return {"passed": all(checks.values()), "checks": checks}


def strict_gate(comparison: dict[str, Any]) -> dict[str, Any]:
    unseen = comparison["splits"]["unseen"]
    per_type = comparison["unseen_per_type_delta"]
    nonnegative_types = sum(values["mean"] >= 0 for values in per_type.values())
    checks = {
        "all_macro_metrics_positive": all(value > 0 for value in unseen["macro_delta"].values()),
        "macro_micro_directions_match": unseen["macro_micro_directions_match"],
        "argument_ci_lower_positive": unseen["paired_bootstrap"]["argument"]["lower_95"] > 0,
        "trigger_ci_lower_positive": unseen["paired_bootstrap"]["trigger"]["lower_95"] > 0,
        "event_point_positive": unseen["paired_bootstrap"]["event"]["point"] > 0,
        "at_least_six_nonnegative_types": nonnegative_types >= 6,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "nonnegative_type_count": nonnegative_types,
    }


def g9_gate(comparison: dict[str, Any], reference_sc_seen_event_gain: float) -> dict[str, Any]:
    seen = comparison["splits"]["seen"]["macro_delta"]
    unseen = comparison["splits"]["unseen"]["macro_delta"]
    event_recovers_half_sc = seen["event"] >= 0.5 * reference_sc_seen_event_gain
    checks = {
        "seen_event_plus_0.020_or_half_sc_gain": seen["event"] >= 0.020
        or event_recovers_half_sc,
        "seen_argument_drop_at_most_0.010": seen["argument"] >= -0.010,
        "seen_trigger_drop_at_most_0.010": seen["trigger"] >= -0.010,
        "no_unseen_metric_below_minus_0.010": min(unseen.values()) >= -0.010,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reference_sc_seen_event_gain": reference_sc_seen_event_gain,
        "half_sc_seen_event_gain": 0.5 * reference_sc_seen_event_gain,
        "event_recovers_half_sc_gain": event_recovers_half_sc,
    }


def compare_split(
    baseline_dir: Path, candidate_dir: Path, bootstrap_samples: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_summary, baseline_rows = load_run(baseline_dir)
    candidate_summary, candidate_rows = load_run(candidate_dir)
    verify_pairing(baseline_rows, candidate_rows)
    baseline_macro = macro_values(baseline_summary)
    candidate_macro = macro_values(candidate_summary)
    baseline_micro = micro_values(baseline_rows)
    candidate_micro = micro_values(candidate_rows)
    macro_delta = {
        metric: candidate_macro[metric] - baseline_macro[metric] for metric in METRICS
    }
    micro_delta = {
        metric: candidate_micro[metric] - baseline_micro[metric] for metric in METRICS
    }
    baseline_integrity = integrity(baseline_summary)
    candidate_integrity = integrity(candidate_summary)
    result = {
        "baseline_macro": baseline_macro,
        "candidate_macro": candidate_macro,
        "macro_delta": macro_delta,
        "baseline_micro": baseline_micro,
        "candidate_micro": candidate_micro,
        "micro_delta": micro_delta,
        "macro_micro_directions_match": all(
            (macro_delta[metric] >= 0) == (micro_delta[metric] >= 0) for metric in METRICS
        ),
        "baseline_integrity": baseline_integrity,
        "candidate_integrity": candidate_integrity,
        "integrity_delta": {
            name: candidate_integrity[name] - baseline_integrity[name]
            for name in baseline_integrity
        },
        "paired_bootstrap": bootstrap_deltas(
            baseline_rows, candidate_rows, bootstrap_samples, seed
        ),
    }
    return result, baseline_rows, candidate_rows


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = ["# Preference Run Comparison", ""]
    for split in ("seen", "unseen"):
        values = comparison["splits"][split]
        lines.extend(
            [
                f"## {split.title()}",
                "",
                "| metric | baseline macro | candidate macro | delta | micro delta | bootstrap 95% CI |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric in METRICS:
            ci = values["paired_bootstrap"][metric]
            lines.append(
                f"| {metric} | {values['baseline_macro'][metric]:.4f} | "
                f"{values['candidate_macro'][metric]:.4f} | {values['macro_delta'][metric]:+.4f} | "
                f"{values['micro_delta'][metric]:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
            )
        lines.append("")
    gate = comparison.get("gate")
    if gate:
        lines.extend(["## Gate", "", f"Passed: **{gate['passed']}**", ""])
        for name, passed in gate["checks"].items():
            lines.append(f"- `{name}`: {passed}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_seen", type=Path, required=True)
    parser.add_argument("--baseline_unseen", type=Path, required=True)
    parser.add_argument("--candidate_seen", type=Path, required=True)
    parser.add_argument("--candidate_unseen", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gate", choices=["none", "e81", "strict", "g9"], default="none")
    parser.add_argument("--reference_sc_seen_event_gain", type=float, default=0.0)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    seen, _, _ = compare_split(
        args.baseline_seen, args.candidate_seen, args.bootstrap_samples, args.seed
    )
    unseen, unseen_baseline, unseen_candidate = compare_split(
        args.baseline_unseen, args.candidate_unseen, args.bootstrap_samples, args.seed + 1
    )
    comparison = {
        "baseline": {
            "seen": str(args.baseline_seen.resolve()),
            "unseen": str(args.baseline_unseen.resolve()),
        },
        "candidate": {
            "seen": str(args.candidate_seen.resolve()),
            "unseen": str(args.candidate_unseen.resolve()),
        },
        "splits": {"seen": seen, "unseen": unseen},
        "unseen_per_type_delta": per_type_deltas(unseen_baseline, unseen_candidate),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }
    if args.gate == "e81":
        comparison["gate"] = e81_gate(comparison)
    elif args.gate == "strict":
        comparison["gate"] = strict_gate(comparison)
    elif args.gate == "g9":
        comparison["gate"] = g9_gate(comparison, args.reference_sc_seen_event_gain)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison.md").write_text(render_markdown(comparison), encoding="utf-8")
    print(render_markdown(comparison))
    if comparison.get("gate") and not comparison["gate"]["passed"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
