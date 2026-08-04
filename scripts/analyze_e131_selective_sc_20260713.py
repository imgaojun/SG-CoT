#!/usr/bin/env python3
"""Evaluate E131's preregistered two-path early exit on saved E101 samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.rescore_self_consistency_samples_20260712 import (
    aggregate,
    as_payload,
    load_jsonl,
    normalize,
    recover_offsets_from_evidence,
    vote,
)


METRICS = ("argument", "event", "trigger")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_structure(payload: dict[str, Any]) -> tuple[Any, ...]:
    events = []
    for event_key, arguments in normalize(payload):
        events.append((event_key, tuple(sorted(arguments, key=repr))))
    return tuple(sorted(events, key=repr))


def evaluate_gate(
    greedy_macro: dict[str, float],
    full_macro: dict[str, float],
    adaptive_macro: dict[str, float],
    mean_paths: float,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    cost_pass = mean_paths <= float(protocol["max_mean_paths"])
    metric_checks = {}
    for metric in METRICS:
        greedy = float(greedy_macro[metric])
        full = float(full_macro[metric])
        adaptive = float(adaptive_macro[metric])
        full_delta = full - greedy
        adaptive_delta = adaptive - greedy
        if full_delta > 0:
            reference_pass = (
                adaptive_delta + 1e-12
                >= float(protocol["positive_gain_retention"]) * full_delta
            )
            criterion = "retain_positive_full_sc_gain"
        else:
            reference_pass = (
                adaptive + 1e-12
                >= full - float(protocol["nonpositive_full_sc_tolerance"])
            )
            criterion = "match_nonpositive_full_sc"
        greedy_floor_pass = (
            adaptive + 1e-12
            >= greedy - float(protocol["max_macro_regression_from_greedy"])
        )
        metric_checks[metric] = {
            "greedy": greedy,
            "full_sc": full,
            "adaptive": adaptive,
            "full_sc_delta": full_delta,
            "adaptive_delta": adaptive_delta,
            "criterion": criterion,
            "reference_pass": reference_pass,
            "greedy_floor_pass": greedy_floor_pass,
            "passed": reference_pass and greedy_floor_pass,
        }
    passed = cost_pass and all(check["passed"] for check in metric_checks.values())
    return {
        "passed": passed,
        "cost_pass": cost_pass,
        "mean_paths": mean_paths,
        "max_mean_paths": float(protocol["max_mean_paths"]),
        "metrics": metric_checks,
    }


def recover(surface: dict[str, Any], input_text: str) -> tuple[dict[str, Any], int]:
    payload, diagnostics = recover_offsets_from_evidence(surface or {"events": []}, input_text)
    return payload, int(diagnostics.get("missing_offsets", 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_jsonl", type=Path, required=True)
    parser.add_argument("--eval_jsonl", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("id") != "e131_selective_sc_v1":
        raise ValueError("unexpected protocol id")
    if sha256_file(args.samples_jsonl) != protocol["saved_samples_sha256"]:
        raise ValueError("saved-sample hash does not match frozen protocol")
    if sha256_file(args.eval_jsonl) != protocol["eval_rows_sha256"]:
        raise ValueError("evaluation-row hash does not match frozen protocol")

    saved_rows = load_jsonl(args.samples_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl)
    if len(saved_rows) != len(eval_rows):
        raise ValueError(f"row count mismatch: {len(saved_rows)} != {len(eval_rows)}")

    expected_samples = int(protocol["n_samples"])
    prefix_samples = int(protocol["prefix_samples"])
    vote_k = int(protocol["vote_k"])
    if prefix_samples != 2 or expected_samples != 8 or vote_k != 3:
        raise ValueError("E131 v1 requires the frozen 2-to-8, k=3 policy")

    scored_rows = []
    escalated = 0
    missing_offsets = {"greedy": 0, "samples": 0}
    for index, (saved, row) in enumerate(zip(saved_rows, eval_rows)):
        if int(saved.get("idx", index)) != index:
            raise ValueError(f"sample index mismatch at row {index}")
        surfaces = saved.get("sample_surfaces") or []
        if len(surfaces) != expected_samples:
            raise ValueError(f"expected {expected_samples} samples at row {index}, got {len(surfaces)}")
        input_text = row["input"]
        greedy, missing = recover(saved.get("greedy_surface") or {}, input_text)
        missing_offsets["greedy"] += missing
        samples = []
        for surface in surfaces:
            payload, missing = recover(surface or {}, input_text)
            samples.append(payload)
            missing_offsets["samples"] += missing

        prefix_agrees = canonical_structure(samples[0]) == canonical_structure(samples[1])
        full_prediction = as_payload(vote([normalize(sample) for sample in samples], vote_k))
        if prefix_agrees:
            adaptive_prediction = as_payload(normalize(samples[0]))
            route = "early_exit_2"
        else:
            adaptive_prediction = full_prediction
            route = "escalate_8"
            escalated += 1

        gold_raw = row.get("gold_output", row["output"])
        gold = json.loads(gold_raw) if isinstance(gold_raw, str) else gold_raw
        scored_rows.append(
            {
                "idx": index,
                "route": route,
                "meta": row.get("meta", {}),
                "gold": gold,
                "greedy_predicted": greedy,
                "full_sc_predicted": full_prediction,
                "adaptive_predicted": adaptive_prediction,
            }
        )

    total = len(scored_rows)
    escalation_rate = escalated / total if total else 0.0
    mean_paths = prefix_samples + (expected_samples - prefix_samples) * escalation_rate
    greedy_summary = aggregate(scored_rows, "greedy_predicted")
    full_summary = aggregate(scored_rows, "full_sc_predicted")
    adaptive_summary = aggregate(scored_rows, "adaptive_predicted")
    gate = evaluate_gate(
        greedy_summary["macro"],
        full_summary["macro"],
        adaptive_summary["macro"],
        mean_paths,
        protocol,
    )
    summary = {
        "id": "e131_selective_sc_dev_gate_v1",
        "rows": total,
        "policy": {
            "prefix_samples": prefix_samples,
            "n_samples": expected_samples,
            "vote_k": vote_k,
            "agreement": "exact_recovered_event_type_trigger_and_argument_structure",
        },
        "routes": {"early_exit_2": total - escalated, "escalate_8": escalated},
        "escalation_rate": escalation_rate,
        "mean_paths": mean_paths,
        "relative_cost_vs_full_sc": mean_paths / expected_samples,
        "greedy": greedy_summary,
        "full_sc": full_summary,
        "adaptive": adaptive_summary,
        "macro_delta_vs_greedy": {
            metric: adaptive_summary["macro"][metric] - greedy_summary["macro"][metric]
            for metric in METRICS
        },
        "missing_offsets": missing_offsets,
        "gate": gate,
        "test_rows_read": 0,
        "input_sha256": {
            "samples": protocol["saved_samples_sha256"],
            "eval_rows": protocol["eval_rows_sha256"],
        },
    }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_pass and not gate["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
