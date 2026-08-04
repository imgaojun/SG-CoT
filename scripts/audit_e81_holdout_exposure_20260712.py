#!/usr/bin/env python3
"""Audit positive-target holdout and schema-string exposure in an SG-CoT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


FINAL_RE = re.compile(r"<final>\s*(\{.*?\})\s*</final>", re.DOTALL)


def event_types(payload: Any) -> list[str]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return []
    return [
        str(event.get("event_type") or event.get("type"))
        for event in payload.get("events", [])
        if event.get("event_type") or event.get("type")
    ]


def final_event_types(output: str) -> list[str]:
    match = FINAL_RE.search(output)
    if not match:
        raise ValueError("missing <final> JSON block")
    return event_types(match.group(1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument("--unseen_types_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    unseen = json.loads(args.unseen_types_json.read_text(encoding="utf-8"))
    if not isinstance(unseen, list) or not all(isinstance(item, str) for item in unseen):
        raise ValueError("unseen_types_json must contain a list of strings")
    counters = {
        name: Counter({event_type: 0 for event_type in unseen})
        for name in (
            "candidate_rows",
            "input_rows",
            "instruction_rows",
            "reasoning_output_rows",
            "meta_gold_targets",
            "gold_output_targets",
            "final_output_targets",
        )
    }
    any_counts = Counter()
    row_count = 0

    with args.dataset_jsonl.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            meta = row.get("meta") or {}
            candidates = set(meta.get("candidate_types") or [])
            meta_gold = list(meta.get("gold_event_types") or [])
            gold_types = event_types(row.get("gold_output") or {"events": []})
            output = str(row.get("output") or "")
            try:
                final_types = final_event_types(output)
            except Exception as exc:
                raise ValueError(f"row {line_number}: {exc}") from exc

            surfaces = {
                "candidate_rows": candidates,
                "input_rows": str(row.get("input") or ""),
                "instruction_rows": str(row.get("instruction") or ""),
                "reasoning_output_rows": output.split("<final>", 1)[0],
                "meta_gold_targets": meta_gold,
                "gold_output_targets": gold_types,
                "final_output_targets": final_types,
            }
            for name, surface in surfaces.items():
                row_has_unseen = False
                for event_type in unseen:
                    if isinstance(surface, str):
                        present = event_type in surface
                    else:
                        present = event_type in surface
                    if present:
                        counters[name][event_type] += 1
                        row_has_unseen = True
                if row_has_unseen:
                    any_counts[name] += 1

            if set(meta_gold) != set(gold_types) or set(gold_types) != set(final_types):
                raise ValueError(f"row {line_number}: meta/gold/final target types do not align")

    positive_target_fields = (
        "meta_gold_targets",
        "gold_output_targets",
        "final_output_targets",
    )
    positive_target_violations = sum(
        sum(counters[name].values()) for name in positive_target_fields
    )
    schema_or_reasoning_exposure = (
        any_counts["candidate_rows"] > 0
        or any_counts["input_rows"] > 0
        or any_counts["instruction_rows"] > 0
        or any_counts["reasoning_output_rows"] > 0
    )
    result = {
        "dataset": str(args.dataset_jsonl),
        "dataset_sha256": sha256(args.dataset_jsonl),
        "unseen_types_source": str(args.unseen_types_json),
        "unseen_types": unseen,
        "row_count": row_count,
        "per_type_row_counts": {
            name: dict(counter) for name, counter in counters.items()
        },
        "rows_with_any_unseen_type": dict(any_counts),
        "positive_target_violations": positive_target_violations,
        "positive_target_holdout_passed": positive_target_violations == 0,
        "schema_or_reasoning_exposure_present": schema_or_reasoning_exposure,
        "strict_string_exclusion_passed": not schema_or_reasoning_exposure,
        "protocol_interpretation": (
            "positive-target holdout with candidate-schema exposure"
            if positive_target_violations == 0 and schema_or_reasoning_exposure
            else "audit assumptions not satisfied"
        ),
    }
    if positive_target_violations:
        raise ValueError("held-out types occur in positive target fields")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
