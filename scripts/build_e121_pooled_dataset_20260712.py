#!/usr/bin/env python3
"""Deduplicate E121 unseen-positive evaluation rows across window splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.build_surface_evidence_dataset_20260712 import register_dataset


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_signature(row: dict[str, Any]) -> str:
    payload = {
        "instruction": row.get("instruction"),
        "input": row.get("input"),
        "output": row.get("output"),
        "gold_output": row.get("gold_output"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def as_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("gold_output must be a JSON object")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_info", type=Path, required=True)
    parser.add_argument("--heldout_types_json", type=Path, required=True)
    parser.add_argument("--expected_raw_rows", type=int, required=True)
    parser.add_argument("--expected_unique_rows", type=int, required=True)
    parser.add_argument("--expected_event_mentions", type=int, required=True)
    args = parser.parse_args()

    heldout_types = set(
        json.loads(args.heldout_types_json.read_text(encoding="utf-8"))
    )
    by_id: dict[str, dict[str, Any]] = {}
    signatures: dict[str, str] = {}
    source_splits: dict[str, set[str]] = {}
    raw_rows = 0
    conflicts = []
    for path in args.input_jsonl:
        for row in load_jsonl(path):
            raw_rows += 1
            meta = row.get("meta", {})
            wnd_id = meta.get("wnd_id")
            if not wnd_id:
                raise ValueError(f"missing meta.wnd_id in {path}")
            signature = stable_signature(row)
            if wnd_id in signatures and signatures[wnd_id] != signature:
                conflicts.append({"wnd_id": wnd_id, "input": str(path.resolve())})
                continue
            signatures[wnd_id] = signature
            by_id.setdefault(wnd_id, row)
            source_splits.setdefault(wnd_id, set()).add(str(meta.get("source_split", "unknown")))
    if conflicts:
        raise ValueError(f"conflicting duplicate windows: {conflicts[:20]}")

    output = []
    event_mentions = 0
    event_type_counts: dict[str, int] = {}
    for wnd_id in sorted(by_id):
        row = json.loads(json.dumps(by_id[wnd_id], ensure_ascii=False))
        gold = as_object(row.get("gold_output"))
        events = gold.get("events", [])
        if not events:
            raise ValueError(f"pooled positive row has no gold events: {wnd_id}")
        for event in events:
            event_type = event.get("event_type")
            if event_type not in heldout_types:
                raise ValueError(f"non-held-out gold type in pooled row {wnd_id}: {event_type}")
            event_mentions += 1
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        row.setdefault("meta", {})["pooled_source_splits"] = sorted(source_splits[wnd_id])
        row["meta"]["pooled_deduplicate_key"] = "wnd_id"
        output.append(row)

    observed = (raw_rows, len(output), event_mentions)
    expected = (
        args.expected_raw_rows,
        args.expected_unique_rows,
        args.expected_event_mentions,
    )
    if observed != expected:
        raise ValueError(f"pooled cardinality mismatch: observed={observed}, expected={expected}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    register_dataset(args.dataset_info, args.dataset_name, args.output_jsonl)
    summary = {
        "inputs": [str(path.resolve()) for path in args.input_jsonl],
        "output": str(args.output_jsonl.resolve()),
        "raw_rows": raw_rows,
        "unique_rows": len(output),
        "event_mentions": event_mentions,
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "duplicate_conflicts": 0,
        "deduplicate_by": "wnd_id",
    }
    args.output_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
