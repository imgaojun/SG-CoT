#!/usr/bin/env python3
"""Aggregate the frozen E121 seed pairs and apply its preregistered gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.compare_strict_n3_gate_20260712 import (  # noqa: E402
    METRICS,
    load_seed_pairs,
    paired_bootstrap_n3,
    per_type_mean_delta,
    seed_aggregate,
)
from scripts.compare_preference_run_gate_20260712 import as_object  # noqa: E402


def aggregate(
    baseline_dirs: list[Path],
    candidate_dirs: list[Path],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = load_seed_pairs(baseline_dirs, candidate_dirs)
    result = seed_aggregate(pairs)
    result["paired_bootstrap"] = paired_bootstrap_n3(
        pairs, bootstrap_samples, bootstrap_seed
    )
    result["completion_diagnostics"] = completion_diagnostics(pairs)
    return result, pairs


def completion_diagnostics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    required = {
        "max_new_tokens",
        "generated_token_count_mean",
        "generated_token_count_p95",
        "generated_token_count_max",
        "hit_max_new_tokens_count",
        "hit_max_new_tokens_rate",
    }
    for method in ("baseline", "candidate"):
        summaries = [pair[f"{method}_summary"] for pair in pairs]
        available = all(required <= set(summary) for summary in summaries)
        values: dict[str, Any] = {"available": available}
        if available:
            values.update(
                {
                    "max_new_tokens": sorted(
                        {int(summary["max_new_tokens"]) for summary in summaries}
                    ),
                    "generated_token_count_mean_across_seeds": mean(
                        float(summary["generated_token_count_mean"])
                        for summary in summaries
                    ),
                    "generated_token_count_p95_across_seeds": mean(
                        float(summary["generated_token_count_p95"])
                        for summary in summaries
                    ),
                    "generated_token_count_max_across_seeds": max(
                        int(summary["generated_token_count_max"])
                        for summary in summaries
                    ),
                    "hit_max_new_tokens_count_across_seeds": sum(
                        int(summary["hit_max_new_tokens_count"])
                        for summary in summaries
                    ),
                    "hit_max_new_tokens_rate_mean": mean(
                        float(summary["hit_max_new_tokens_rate"])
                        for summary in summaries
                    ),
                }
            )
            contract_required = {
                "final_tag_complete_rate",
                "reasoning_tag_complete_rate",
                "surface_event_list_valid_rate",
                "candidate_type_valid_rate",
            }
            contract_available = all(
                contract_required <= set(summary) for summary in summaries
            )
            values["output_contract_available"] = contract_available
            if contract_available:
                for key in (
                    "final_tag_complete_rate",
                    "surface_event_list_valid_rate",
                    "candidate_type_valid_rate",
                ):
                    values[f"{key}_mean"] = mean(
                        float(summary[key]) for summary in summaries
                    )
                reasoning_rates = [
                    summary["reasoning_tag_complete_rate"]
                    for summary in summaries
                    if summary["reasoning_tag_complete_rate"] is not None
                ]
                values["reasoning_tag_complete_rate_mean"] = (
                    mean(float(value) for value in reasoning_rates)
                    if reasoning_rates
                    else None
                )
        output[method] = values
    return output


def protocol_integrity_checks(
    split_pairs: dict[str, list[dict[str, Any]]],
    expected_types: set[str],
    eval_config: dict[str, Any],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    expected_examples = eval_config["expected_examples"]
    expected_protocol = eval_config["protocol"]
    completion_fields = set(eval_config["completion_diagnostics"])
    output_contract_fields = set(eval_config["output_contract_diagnostics"])
    expected_max_new_tokens = int(eval_config["decode"]["max_new_tokens"])
    for split_name, pairs in split_pairs.items():
        expected = int(expected_examples[split_name])
        checks[f"{split_name}_all_runs_expected_examples"] = all(
            len(pair[side]) == expected
            for pair in pairs
            for side in ("baseline_rows", "candidate_rows")
        )
        checks[f"{split_name}_protocol_exact"] = all(
            row.get("meta", {}).get("source_protocol") == expected_protocol
            for pair in pairs
            for side in ("baseline_rows", "candidate_rows")
            for row in pair[side]
        )
        checks[f"{split_name}_completion_diagnostics_present"] = all(
            completion_fields | output_contract_fields <= set(row)
            for pair in pairs
            for side in ("baseline_rows", "candidate_rows")
            for row in pair[side]
        )
        checks[f"{split_name}_decode_cap_exact"] = all(
            int(pair[f"{method}_summary"].get("max_new_tokens", -1))
            == expected_max_new_tokens
            for pair in pairs
            for method in ("baseline", "candidate")
        )

    pooled_rows = split_pairs["pooled_unseen"][0]["baseline_rows"]
    pooled_wnd_ids = [row.get("meta", {}).get("wnd_id") for row in pooled_rows]
    checks["pooled_all_wnd_ids_present"] = all(pooled_wnd_ids)
    checks["pooled_unique_wnd_ids_exact"] = (
        len(set(pooled_wnd_ids))
        == int(eval_config["expected_pooled_unique_wnd_ids"])
        == len(pooled_rows)
    )
    pooled_gold_events = sum(
        len(as_object(row.get("gold")).get("events", [])) for row in pooled_rows
    )
    checks["pooled_gold_event_count_exact"] = pooled_gold_events == int(
        eval_config["expected_pooled_gold_events"]
    )
    checks["heldout_type_count_exact"] = len(expected_types) == int(
        eval_config["expected_heldout_types"]
    )
    return checks


def integrity_checks(
    split: dict[str, Any], prefix: str, maximum_drop: float
) -> dict[str, bool]:
    checks = {}
    for name in ("json", "offset"):
        delta = (
            split["candidate_integrity_mean"][name]
            - split["baseline_integrity_mean"][name]
        )
        checks[f"{prefix}_{name}_registered_drop_limit"] = delta >= -maximum_drop
    return checks


def apply_gate(
    seen: dict[str, Any],
    split1_unseen: dict[str, Any],
    pooled: dict[str, Any],
    per_type: dict[str, dict[str, float]],
    expected_types: set[str],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    macro = pooled["macro_mean_delta"]
    micro = pooled["micro_mean_delta"]
    seed_mean_deltas = [
        mean(run["macro_delta"][metric] for metric in METRICS)
        for run in pooled["seed_runs"]
    ]
    positive_cells = sum(
        run["macro_delta"][metric] > 0
        for run in pooled["seed_runs"]
        for metric in METRICS
    )
    observed_types = set(per_type)
    nonnegative_types = sum(values["mean"] >= 0 for values in per_type.values())
    worst_type_mean = min(values["mean"] for values in per_type.values())
    seen_values = seen["macro_mean_delta"]
    checks = {
        "pooled_all_macro_metrics_positive": all(macro[metric] > 0 for metric in METRICS),
        "pooled_macro_mean_delta_at_least_registered_minimum": mean(macro.values())
        >= float(gate_config["pooled_macro_mean_delta_min"]),
        "pooled_all_micro_metrics_positive": all(micro[metric] > 0 for metric in METRICS),
        "pooled_macro_micro_directions_match": pooled["macro_micro_directions_match"],
        "pooled_argument_ci_lower_positive": pooled["paired_bootstrap"]["argument"]["lower_95"] > 0,
        "pooled_trigger_ci_lower_positive": pooled["paired_bootstrap"]["trigger"]["lower_95"] > 0,
        "pooled_event_point_positive": pooled["paired_bootstrap"]["event"]["point"] > 0,
        "every_seed_macro_mean_delta_positive": all(value > 0 for value in seed_mean_deltas),
        "registered_seed_metric_cells_positive": positive_cells
        >= int(gate_config["required_positive_seed_metric_cells"]),
        "per_type_set_exact": observed_types == expected_types,
        "registered_nonnegative_type_count": nonnegative_types
        >= int(gate_config["required_nonnegative_types"]),
        "registered_per_type_floor": worst_type_mean
        >= float(gate_config["minimum_per_type_mean_delta"]),
        "registered_seen_mean_floor": mean(seen_values.values())
        >= float(gate_config["minimum_seen_mean_delta"]),
        "registered_seen_metric_floor": min(seen_values.values())
        >= float(gate_config["minimum_seen_metric_delta"]),
        **integrity_checks(
            seen, "seen", float(gate_config["maximum_integrity_drop"])
        ),
        **integrity_checks(
            split1_unseen,
            "split1_unseen",
            float(gate_config["maximum_integrity_drop"]),
        ),
        **integrity_checks(
            pooled, "pooled", float(gate_config["maximum_integrity_drop"])
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "positive_seed_metric_cells": positive_cells,
        "seed_macro_mean_deltas": seed_mean_deltas,
        "nonnegative_type_count": nonnegative_types,
        "worst_type_mean_delta": worst_type_mean,
    }


def render(report: dict[str, Any]) -> str:
    lines = [f"# {report['id']} N=3 Confirmation", ""]
    for split_name in ("seen", "split1_unseen", "pooled_unseen"):
        split = report["splits"][split_name]
        lines.extend(
            [
                f"## {split_name.replace('_', ' ').title()}",
                "",
                "| metric | Direct macro | Auto SG-CoT macro | macro delta | micro delta | paired 95% CI |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric in METRICS:
            ci = split["paired_bootstrap"][metric]
            lines.append(
                f"| {metric} | {split['baseline_macro_mean'][metric]:.4f} | "
                f"{split['candidate_macro_mean'][metric]:.4f} | "
                f"{split['macro_mean_delta'][metric]:+.4f} | "
                f"{split['micro_mean_delta'][metric]:+.4f} | "
                f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
            )
        completion = split["completion_diagnostics"]
        if completion["baseline"]["available"] and completion["candidate"]["available"]:
            lines.extend(
                [
                    "",
                    "Completion cap hits across three seeds: "
                    f"Direct {completion['baseline']['hit_max_new_tokens_count_across_seeds']}, "
                    f"Auto SG-CoT {completion['candidate']['hit_max_new_tokens_count_across_seeds']} "
                    f"(max-new-tokens {completion['candidate']['max_new_tokens']}).",
                ]
            )
            if completion["candidate"].get("output_contract_available"):
                reasoning_rate = completion["candidate"][
                    "reasoning_tag_complete_rate_mean"
                ]
                lines.append(
                    "Auto SG-CoT contract rates (final tag / reasoning tag / event list / candidate type): "
                    f"{completion['candidate']['final_tag_complete_rate_mean']:.4f} / "
                    f"{reasoning_rate:.4f} / "
                    f"{completion['candidate']['surface_event_list_valid_rate_mean']:.4f} / "
                    f"{completion['candidate']['candidate_type_valid_rate_mean']:.4f}."
                )
        lines.append("")
    lines.extend(["## Gate", "", f"Passed: **{report['gate']['passed']}**", ""])
    for name, passed in report["gate"]["checks"].items():
        lines.append(f"- `{name}`: {passed}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("baseline_seen", "baseline_unseen", "baseline_pooled"):
        parser.add_argument(f"--{name}", type=Path, nargs="+", required=True)
    for name in ("candidate_seen", "candidate_unseen", "candidate_pooled"):
        parser.add_argument(f"--{name}", type=Path, nargs="+", required=True)
    parser.add_argument("--heldout_types_json", type=Path, required=True)
    parser.add_argument("--gate_config", type=Path, required=True)
    parser.add_argument("--eval_config", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260712)
    args = parser.parse_args()

    seen, seen_pairs = aggregate(
        args.baseline_seen,
        args.candidate_seen,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    split1_unseen, split1_unseen_pairs = aggregate(
        args.baseline_unseen,
        args.candidate_unseen,
        args.bootstrap_samples,
        args.bootstrap_seed + 1,
    )
    pooled, pooled_pairs = aggregate(
        args.baseline_pooled,
        args.candidate_pooled,
        args.bootstrap_samples,
        args.bootstrap_seed + 2,
    )
    per_type = per_type_mean_delta(pooled_pairs)
    expected_types = set(
        json.loads(args.heldout_types_json.read_text(encoding="utf-8"))
    )
    gate_config = json.loads(args.gate_config.read_text(encoding="utf-8"))
    eval_config = (
        json.loads(args.eval_config.read_text(encoding="utf-8"))
        if args.eval_config
        else None
    )
    report = {
        "id": (
            eval_config["id"] if eval_config else "e121e_frozen_n3_confirmation"
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "splits": {
            "seen": seen,
            "split1_unseen": split1_unseen,
            "pooled_unseen": pooled,
        },
        "pooled_unseen_per_type_mean_delta": per_type,
        "expected_heldout_types": sorted(expected_types),
        "gate_config": gate_config,
        "eval_config": eval_config,
    }
    gate = apply_gate(
        seen, split1_unseen, pooled, per_type, expected_types, gate_config
    )
    if eval_config is not None:
        protocol_checks = protocol_integrity_checks(
            {
                "test_seen": seen_pairs,
                "test_unseen": split1_unseen_pairs,
                "pooled_unseen": pooled_pairs,
            },
            expected_types,
            eval_config,
        )
        protocol_checks["bootstrap_samples_exact"] = (
            args.bootstrap_samples
            == int(eval_config["bootstrap"]["samples"])
            == int(gate_config["bootstrap_samples"])
        )
        protocol_checks["bootstrap_seed_exact"] = (
            args.bootstrap_seed
            == int(eval_config["bootstrap"]["seed"])
            == int(gate_config["bootstrap_seed"])
        )
        gate["checks"] = {**protocol_checks, **gate["checks"]}
        gate["passed"] = all(gate["checks"].values())
        report["protocol_integrity_checks"] = protocol_checks
    report["gate"] = gate
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render(report)
    (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if report["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
