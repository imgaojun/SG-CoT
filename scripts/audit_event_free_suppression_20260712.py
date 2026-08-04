#!/usr/bin/env python3
"""Audit event suppression on negative windows without crediting parse failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "direct": {
        "base": "outputs/strengthen_20260709/withneg/direct_base/test_unseen/predictions.jsonl",
        "repeat1": "outputs/strengthen_20260709/withneg/direct_repeat1/test_unseen/predictions.jsonl",
        "repeat2": "outputs/strengthen_20260709/withneg/direct_repeat2/test_unseen/predictions.jsonl",
    },
    "sgcot": {
        "base": "outputs/strengthen_20260709/withneg/e81_base/test_unseen/predictions.jsonl",
        "r1": "outputs/strengthen_20260709/withneg/e81_r1/test_unseen/predictions.jsonl",
        "r2": "outputs/strengthen_20260709/withneg/e81_r2/test_unseen/predictions.jsonl",
        "r3": "outputs/strengthen_20260709/withneg/e81_r3/test_unseen/predictions.jsonl",
        "r4": "outputs/strengthen_20260709/withneg/e81_r4/test_unseen/predictions.jsonl",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def event_list(payload: Any, field: str, line_number: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError(f"line {line_number}: {field}.events must be a list")
    events = payload["events"]
    if not all(isinstance(event, dict) for event in events):
        raise ValueError(f"line {line_number}: {field}.events contains a non-object")
    return events


def valid_prediction(row: dict[str, Any]) -> bool:
    if "valid_final_json" in row:
        return bool(row["valid_final_json"])
    if "valid_json" in row:
        return bool(row["valid_json"])
    raise ValueError("prediction row has no JSON-validity field")


def audit_run(path: Path, expected_negative_rows: int | None = None) -> dict[str, Any]:
    total_rows = 0
    negative_rows = 0
    valid_rows = 0
    parsed_empty_rows = 0
    strict_empty_rows = 0
    predicted_event_total = 0
    negative_inputs: list[str] = []

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            gold_events = event_list(row.get("gold"), "gold", line_number)
            if gold_events:
                continue

            negative_rows += 1
            predicted_events = event_list(row.get("predicted"), "predicted", line_number)
            is_valid = valid_prediction(row)
            is_empty = len(predicted_events) == 0
            valid_rows += int(is_valid)
            parsed_empty_rows += int(is_empty)
            strict_empty_rows += int(is_valid and is_empty)
            predicted_event_total += len(predicted_events)
            negative_inputs.append(str(row.get("input") or ""))

    if negative_rows == 0:
        raise ValueError(f"{path}: no negative rows")
    if expected_negative_rows is not None and negative_rows != expected_negative_rows:
        raise ValueError(
            f"{path}: expected {expected_negative_rows} negative rows, found {negative_rows}"
        )

    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "total_rows": total_rows,
        "negative_rows": negative_rows,
        "positive_rows_ignored": total_rows - negative_rows,
        "negative_input_sequence_sha256": sequence_sha256(negative_inputs),
        "valid_rows": valid_rows,
        "invalid_rows": negative_rows - valid_rows,
        "valid_rate": valid_rows / negative_rows,
        "parsed_empty_rows": parsed_empty_rows,
        "parsed_empty_rate": parsed_empty_rows / negative_rows,
        "strict_valid_empty_rows": strict_empty_rows,
        "strict_valid_empty_rate": strict_empty_rows / negative_rows,
        "predicted_event_total": predicted_event_total,
        "mean_predicted_events": predicted_event_total / negative_rows,
    }


def aggregate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "valid_rate",
        "parsed_empty_rate",
        "strict_valid_empty_rate",
        "mean_predicted_events",
    )
    result: dict[str, Any] = {"seed_count": len(runs)}
    for metric in metrics:
        values = [float(run[metric]) for run in runs.values()]
        result[metric] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "per_seed": values,
        }
    return result


def build_report(
    root: Path,
    run_paths: dict[str, dict[str, str]],
    expected_negative_rows: int,
) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    input_hashes: set[str] = set()
    for system, seeds in run_paths.items():
        runs = {
            seed: audit_run(root / relative_path, expected_negative_rows)
            for seed, relative_path in seeds.items()
        }
        input_hashes.update(
            run["negative_input_sequence_sha256"] for run in runs.values()
        )
        systems[system] = {"runs": runs, "aggregate": aggregate(runs)}

    aligned = len(input_hashes) == 1
    return {
        "definition": (
            "successful suppression requires a valid decoded JSON object and an empty "
            "predicted events list; parser failures receive no credit"
        ),
        "expected_negative_rows_per_run": expected_negative_rows,
        "negative_input_alignment_passed": aligned,
        "negative_input_sequence_sha256": next(iter(input_hashes)) if aligned else None,
        "systems": systems,
        "passed": aligned,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("reports/artifacts/2026-07-12_event_free_suppression_audit.json"),
    )
    parser.add_argument("--expected_negative_rows", type=int, default=1085)
    args = parser.parse_args()

    report = build_report(args.root, DEFAULT_RUNS, args.expected_negative_rows)
    output_path = args.root / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
