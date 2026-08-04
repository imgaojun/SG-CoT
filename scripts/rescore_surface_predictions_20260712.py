#!/usr/bin/env python3
"""Rescore saved surface predictions with the shared offset-recovery implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_preference.reasoning_preference import (  # noqa: E402
    metric_f1s,
    recover_offsets_from_evidence,
)


def as_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def micro_counts(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, list[int]]:
    def normalize(payload: dict[str, Any]):
        triggers = []
        arguments = []
        events = []
        for event in payload.get("events", []) if isinstance(payload, dict) else []:
            if not isinstance(event, dict):
                continue
            event_type = event.get("event_type")
            trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
            trigger_key = (trigger.get("start"), trigger.get("end"), event_type)
            triggers.append(trigger_key)
            event_arguments = []
            for argument in event.get("arguments", []) if isinstance(event.get("arguments"), list) else []:
                if not isinstance(argument, dict):
                    continue
                argument_key = (
                    trigger_key,
                    argument.get("role"),
                    argument.get("start"),
                    argument.get("end"),
                )
                arguments.append(argument_key)
                event_arguments.append(argument_key[1:])
            events.append((trigger_key, tuple(sorted(event_arguments, key=str))))
        return {"trigger": triggers, "argument": arguments, "event": events}

    pred = normalize(predicted)
    target = normalize(gold)
    counts = {}
    for name in ("argument", "event", "trigger"):
        remaining = list(target[name])
        true_positive = 0
        false_positive = 0
        for item in pred[name]:
            if item in remaining:
                true_positive += 1
                remaining.remove(item)
            else:
                false_positive += 1
        counts[name] = [true_positive, false_positive, len(remaining)]
    return counts


def f1_from_counts(counts: list[int]) -> float:
    true_positive, false_positive, false_negative = counts
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    macro = {name: [] for name in ("argument", "event", "trigger")}
    totals = {name: [0, 0, 0] for name in ("argument", "event", "trigger")}
    valid = 0
    complete = 0
    with args.input_predictions.open("r", encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]
    for source in source_rows:
        surface = as_object(source.get("surface_final_predicted"))
        gold = as_object(source.get("gold"))
        is_valid = bool(source.get("valid_final_json", source.get("valid_json", bool(surface))))
        if is_valid:
            valid += 1
        recovered, diagnostics = recover_offsets_from_evidence(surface or {"events": []}, source["input"])
        if is_valid and diagnostics["missing_offsets"] == 0:
            complete += 1
        scores = metric_f1s(recovered, gold)
        counts = micro_counts(recovered, gold)
        for name in macro:
            macro[name].append(scores[name])
            totals[name] = [left + right for left, right in zip(totals[name], counts[name])]
        row = dict(source)
        row["predicted"] = recovered
        row["recovery_diagnostics"] = diagnostics
        row.update({f"{name}_f1": scores[name] for name in scores})
        rows.append(row)

    output_predictions = args.output_dir / "predictions.jsonl"
    with output_predictions.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "num_examples": len(rows),
        "final_json_valid_rate": valid / len(rows) if rows else 0.0,
        "offset_recovery_full_rate": complete / len(rows) if rows else 0.0,
        "argument_f1": sum(macro["argument"]) / len(rows) if rows else 0.0,
        "event_f1": sum(macro["event"]) / len(rows) if rows else 0.0,
        "trigger_f1": sum(macro["trigger"]) / len(rows) if rows else 0.0,
        "corpus_micro_argument_f1": f1_from_counts(totals["argument"]),
        "corpus_micro_event_f1": f1_from_counts(totals["event"]),
        "corpus_micro_trigger_f1": f1_from_counts(totals["trigger"]),
        "micro_counts": totals,
        "recovery_implementation": "src.stage2_preference.reasoning_preference",
        "source_predictions": str(args.input_predictions.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
