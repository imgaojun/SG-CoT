#!/usr/bin/env python3
"""Verify E132 evaluation provenance and apply its frozen effectiveness gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("argument_f1", "event_f1", "trigger_f1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold_object(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("gold_output", row["output"])
    return json.loads(value) if isinstance(value, str) else value


def validate_predictions(
    predictions: list[dict[str, Any]], source_rows: list[dict[str, Any]], label: str
) -> None:
    if len(predictions) != len(source_rows):
        raise ValueError(f"{label} prediction count mismatch")
    seen = set()
    for index, (prediction, source) in enumerate(zip(predictions, source_rows)):
        predicted_meta = prediction.get("meta") or {}
        source_meta = source.get("meta") or {}
        wnd_id = source_meta.get("wnd_id")
        if predicted_meta.get("wnd_id") != wnd_id:
            raise ValueError(f"{label} wnd_id mismatch at row {index}")
        if wnd_id in seen:
            raise ValueError(f"{label} duplicate wnd_id: {wnd_id}")
        seen.add(wnd_id)
        if predicted_meta.get("candidate_types") != source_meta.get("candidate_types"):
            raise ValueError(f"{label} candidate order mismatch at row {index}")
        if prediction.get("instruction") != source.get("instruction"):
            raise ValueError(f"{label} instruction mismatch at row {index}")
        if prediction.get("input") != source.get("input"):
            raise ValueError(f"{label} input mismatch at row {index}")
        if prediction.get("gold") != gold_object(source):
            raise ValueError(f"{label} gold mismatch at row {index}")


def build_gate(
    phase: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    deltas = {metric: float(candidate[metric]) - float(baseline[metric]) for metric in METRICS}
    json_delta = float(candidate["final_json_valid_rate"]) - float(
        baseline["final_json_valid_rate"]
    )
    recovery_delta = float(candidate["offset_recovery_full_rate"]) - float(
        baseline["offset_recovery_full_rate"]
    )
    if phase == "dev_seen":
        maximum = float(rules["maximum_macro_regression_per_metric"])
        checks = {
            f"{metric}_retention": delta >= -maximum - 1e-12
            for metric, delta in deltas.items()
        }
    elif phase == "test_unseen":
        checks = {
            "argument_retention": deltas["argument_f1"]
            >= -float(rules["maximum_argument_regression"]) - 1e-12,
            "event_retention": deltas["event_f1"]
            >= -float(rules["maximum_event_regression"]) - 1e-12,
            "trigger_absolute": float(candidate["trigger_f1"])
            >= float(rules["minimum_trigger_f1"]) - 1e-12,
            "trigger_delta": deltas["trigger_f1"]
            >= float(rules["minimum_trigger_delta"]) - 1e-12,
        }
    else:
        raise ValueError(f"unknown phase: {phase}")
    checks["json_valid_retention"] = json_delta >= -float(
        rules["maximum_json_valid_rate_regression"]
    ) - 1e-12
    checks["offset_recovery_retention"] = recovery_delta >= -float(
        rules["maximum_offset_recovery_rate_regression"]
    ) - 1e-12
    return {
        "id": f"e132_{phase}_effectiveness_gate_v1",
        "phase": phase,
        "passed": all(checks.values()),
        "checks": checks,
        "baseline": {key: baseline[key] for key in (*METRICS, "final_json_valid_rate", "offset_recovery_full_rate")},
        "candidate": {key: candidate[key] for key in (*METRICS, "final_json_valid_rate", "offset_recovery_full_rate")},
        "deltas": {
            **deltas,
            "final_json_valid_rate": json_delta,
            "offset_recovery_full_rate": recovery_delta,
        },
        "rules": rules,
    }


def exact_path(path: Path, registered: str, label: str) -> None:
    if path.resolve() != (REPO_ROOT / registered).resolve():
        raise ValueError(f"{label} path differs from frozen protocol")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dev_seen", "test_unseen"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--candidate_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()

    if args.output_json.exists():
        raise SystemExit(f"refusing to overwrite gate: {args.output_json}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("id") != "e132_trigger_cue_enrichment_v1":
        raise ValueError("unexpected protocol id")
    evaluation = protocol["effectiveness_evaluation"]
    section_name = "dev_seen" if args.phase == "dev_seen" else "test_unseen_after_dev_pass"
    section = evaluation[section_name]
    exact_path(args.baseline_dir, section["baseline_output"], "baseline output")
    exact_path(args.candidate_dir, section["candidate_output"], "candidate output")
    exact_path(args.output_json, section["gate_output"], "gate output")

    if args.phase == "test_unseen":
        dev_gate_path = REPO_ROOT / evaluation["dev_seen"]["gate_output"]
        if not dev_gate_path.is_file():
            raise ValueError("test gate requires the frozen dev gate")
        dev_gate = json.loads(dev_gate_path.read_text(encoding="utf-8"))
        if dev_gate.get("id") != "e132_dev_seen_effectiveness_gate_v1" or not dev_gate.get(
            "passed"
        ):
            raise ValueError("test gate requires a passing frozen dev gate")
        baseline_summary_path = args.baseline_dir / "summary.json"
        if sha256_file(baseline_summary_path) != section["baseline_summary_sha256"]:
            raise ValueError("historical unseen baseline summary hash mismatch")
        baseline_rows_path = REPO_ROOT / section["source_rows"]
        candidate_rows_path = REPO_ROOT / section["candidate_rows"]
    else:
        baseline_rows_path = REPO_ROOT / section["baseline_rows"]
        candidate_rows_path = REPO_ROOT / section["candidate_rows"]
        if sha256_file(baseline_rows_path) != section["baseline_rows_sha256"]:
            raise ValueError("dev baseline rows hash mismatch")
        if sha256_file(candidate_rows_path) != section["candidate_rows_sha256"]:
            raise ValueError("dev candidate rows hash mismatch")

    baseline_rows = load_jsonl(baseline_rows_path)
    candidate_rows = load_jsonl(candidate_rows_path)
    expected_rows = int(section["rows"])
    if len(baseline_rows) != expected_rows or len(candidate_rows) != expected_rows:
        raise ValueError("evaluation source row count mismatch")
    if [row["meta"]["wnd_id"] for row in baseline_rows] != [
        row["meta"]["wnd_id"] for row in candidate_rows
    ]:
        raise ValueError("baseline/candidate window pairing mismatch")
    if [gold_object(row) for row in baseline_rows] != [gold_object(row) for row in candidate_rows]:
        raise ValueError("baseline/candidate gold pairing mismatch")

    baseline_predictions = load_jsonl(args.baseline_dir / "predictions.jsonl")
    candidate_predictions = load_jsonl(args.candidate_dir / "predictions.jsonl")
    validate_predictions(baseline_predictions, baseline_rows, "baseline")
    validate_predictions(candidate_predictions, candidate_rows, "candidate")
    baseline_summary = json.loads((args.baseline_dir / "summary.json").read_text())
    candidate_summary = json.loads((args.candidate_dir / "summary.json").read_text())
    if int(baseline_summary.get("num_examples", -1)) != expected_rows:
        raise ValueError("baseline summary row count mismatch")
    if int(candidate_summary.get("num_examples", -1)) != expected_rows:
        raise ValueError("candidate summary row count mismatch")

    gate = build_gate(args.phase, baseline_summary, candidate_summary, section)
    gate["rows"] = expected_rows
    gate["test_rows_read"] = 0 if args.phase == "dev_seen" else expected_rows
    gate["input_sha256"] = {
        "protocol": sha256_file(args.protocol),
        "baseline_rows": sha256_file(baseline_rows_path),
        "candidate_rows": sha256_file(candidate_rows_path),
        "baseline_summary": sha256_file(args.baseline_dir / "summary.json"),
        "candidate_summary": sha256_file(args.candidate_dir / "summary.json"),
        "baseline_predictions": sha256_file(args.baseline_dir / "predictions.jsonl"),
        "candidate_predictions": sha256_file(args.candidate_dir / "predictions.jsonl"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if args.require_pass and not gate["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
