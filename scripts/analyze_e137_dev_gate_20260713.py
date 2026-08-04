#!/usr/bin/env python3
"""Compare E137 against E81 on oracle and predicted dev_seen, then apply the frozen gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.stage2_quality_validation.event_metrics import normalize_events, prf


METRICS = ("argument", "event", "trigger")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assert_aligned(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], label: str
) -> None:
    if len(baseline) != len(candidate):
        raise ValueError(f"{label} row count mismatch")
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        left_meta, right_meta = left.get("meta", {}), right.get("meta", {})
        if (
            left_meta.get("wnd_id") != right_meta.get("wnd_id")
            or left.get("input") != right.get("input")
            or left.get("gold") != right.get("gold")
        ):
            raise ValueError(f"{label} alignment mismatch at row {index}")


def standard_summary(output_dir: Path) -> dict[str, Any]:
    value = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "rows": int(value["num_examples"]),
        "macro": {metric: float(value[f"{metric}_f1"]) for metric in METRICS},
        "json_valid_rate": float(value["json_valid_rate"]),
        "offset_recovery_full_rate": float(value["offset_recovery_full_rate"]),
    }


def row_metric(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    pred_sets = normalize_events(predicted)
    gold_sets = normalize_events(gold)
    mapping = {"trigger": 0, "argument": 1, "event": 2}
    return {metric: prf(pred_sets[index], gold_sets[index])["f1"] for metric, index in mapping.items()}


def supported_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {metric: 0.0 for metric in METRICS}
    abstention_rows = 0
    exact_empty = 0
    candidate_inconsistent_events = 0
    predicted_events = 0
    for row in rows:
        candidates = set(row.get("meta", {}).get("candidate_types", []))
        gold_events = [
            event
            for event in (row.get("gold") or {}).get("events", [])
            if event.get("event_type") in candidates
        ]
        predicted = row.get("predicted") or {"events": []}
        metrics = row_metric(predicted, {"events": gold_events})
        for metric in METRICS:
            totals[metric] += metrics[metric]
        current_predicted = predicted.get("events", [])
        predicted_events += len(current_predicted)
        candidate_inconsistent_events += sum(
            event.get("event_type") not in candidates for event in current_predicted
        )
        if not gold_events:
            abstention_rows += 1
            exact_empty += int(not current_predicted)
    count = len(rows)
    return {
        "rows": count,
        "macro": {metric: totals[metric] / count for metric in METRICS},
        "full_abstention_rows": abstention_rows,
        "full_abstention_exact_empty_rate": exact_empty / abstention_rows
        if abstention_rows
        else 0.0,
        "predicted_events": predicted_events,
        "candidate_inconsistent_events": candidate_inconsistent_events,
    }


def deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro": {
            metric: candidate["macro"][metric] - baseline["macro"][metric]
            for metric in METRICS
        },
        "json_valid_rate": candidate.get("json_valid_rate", 0.0)
        - baseline.get("json_valid_rate", 0.0),
        "offset_recovery_full_rate": candidate.get("offset_recovery_full_rate", 0.0)
        - baseline.get("offset_recovery_full_rate", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    for model in ("baseline", "candidate"):
        for regime in ("oracle", "predicted"):
            parser.add_argument(f"--{model}_{regime}", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse E137 dev gate output: {args.output_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    gate = protocol["dev_gate"]
    dirs = {
        model: {regime: getattr(args, f"{model}_{regime}") for regime in ("oracle", "predicted")}
        for model in ("baseline", "candidate")
    }
    standard = {
        model: {regime: standard_summary(dirs[model][regime]) for regime in dirs[model]}
        for model in dirs
    }
    predictions = {
        model: {
            regime: load_jsonl(dirs[model][regime] / "predictions.jsonl")
            for regime in dirs[model]
        }
        for model in dirs
    }
    for regime in ("oracle", "predicted"):
        assert_aligned(predictions["baseline"][regime], predictions["candidate"][regime], regime)
        if len(predictions["candidate"][regime]) != protocol["expected"][f"{regime}_dev_seen_rows"]:
            raise ValueError(f"unexpected {regime} development row count")

    supported = {
        model: supported_summary(predictions[model]["predicted"])
        for model in ("baseline", "candidate")
    }
    standard_delta = {
        regime: deltas(standard["candidate"][regime], standard["baseline"][regime])
        for regime in ("oracle", "predicted")
    }
    supported_delta = deltas(supported["candidate"], supported["baseline"])
    predicted_improved = sum(
        value >= gate["predicted_improvement_min"]
        for value in standard_delta["predicted"]["macro"].values()
    )
    supported_improved = sum(
        value >= gate["candidate_supported_improvement_min"]
        for value in supported_delta["macro"].values()
    )
    checks = {
        "oracle_each_macro_retained": all(
            value >= gate["oracle_each_macro_delta_min"]
            for value in standard_delta["oracle"]["macro"].values()
        ),
        "predicted_each_macro_retained": all(
            value >= gate["predicted_each_macro_delta_min"]
            for value in standard_delta["predicted"]["macro"].values()
        ),
        "predicted_minimum_macro_improvement": predicted_improved
        >= gate["predicted_minimum_improved_macro_cells"],
        "candidate_supported_each_macro_retained": all(
            value >= gate["candidate_supported_each_macro_delta_min"]
            for value in supported_delta["macro"].values()
        ),
        "candidate_supported_minimum_macro_improvement": supported_improved
        >= gate["candidate_supported_minimum_improved_macro_cells"],
        "full_abstention_empty_rate_retained": (
            supported["candidate"]["full_abstention_exact_empty_rate"]
            - supported["baseline"]["full_abstention_exact_empty_rate"]
        )
        >= gate["full_abstention_empty_rate_delta_min"],
        "oracle_json_retained": standard_delta["oracle"]["json_valid_rate"]
        >= gate["json_valid_rate_delta_min"],
        "predicted_json_retained": standard_delta["predicted"]["json_valid_rate"]
        >= gate["json_valid_rate_delta_min"],
        "oracle_recovery_retained": standard_delta["oracle"]["offset_recovery_full_rate"]
        >= gate["offset_recovery_full_rate_delta_min"],
        "predicted_recovery_retained": standard_delta["predicted"]["offset_recovery_full_rate"]
        >= gate["offset_recovery_full_rate_delta_min"],
    }
    payload = {
        "id": "e137_mixed_oracle_retrieved_dev_gate_v1",
        "protocol_sha256": sha256_file(args.protocol),
        "standard": standard,
        "candidate_supported_predicted": supported,
        "standard_delta": standard_delta,
        "candidate_supported_delta": supported_delta,
        "predicted_improved_macro_cells": predicted_improved,
        "candidate_supported_improved_macro_cells": supported_improved,
        "checks": checks,
        "test_rows_read": 0,
        "passed": all(checks.values()),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
