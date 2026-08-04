#!/usr/bin/env python3
"""Build candidate-consistent train-only manifests for E130."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO / "configs/generated/stage2_revision/e130_retrieved_abstention_protocol.json"
DEFAULT_OUTPUT_DIR = REPO / "data/stage2_revision_e130"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def payload(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("gold_output", row.get("output", {"events": []}))
    value = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    if not isinstance(value, dict) or not isinstance(value.get("events"), list):
        raise ValueError(f"invalid event payload for {row.get('meta', {}).get('wnd_id')}")
    return value


def event_types(value: dict[str, Any]) -> set[str]:
    return {
        str(event["event_type"])
        for event in value.get("events", [])
        if isinstance(event, dict) and event.get("event_type")
    }


def adapt_row(row: dict[str, Any], sample_index: int) -> dict[str, Any]:
    item = copy.deepcopy(row)
    meta = item.setdefault("meta", {})
    candidates = [str(value) for value in meta.get("candidate_types", [])]
    if not candidates:
        raise ValueError(f"missing candidate_types for {meta.get('wnd_id')}")
    original = payload(item)
    original_types = event_types(original)
    candidate_set = set(candidates)
    supported_events = [
        copy.deepcopy(event)
        for event in original["events"]
        if isinstance(event, dict) and event.get("event_type") in candidate_set
    ]
    supported = {"events": supported_events}
    supported_types = event_types(supported)
    missing_types = sorted(original_types - candidate_set)
    if not missing_types:
        target_mode = "gold_present"
    elif supported_events:
        target_mode = "partial_supported"
    else:
        target_mode = "abstain"

    compact_target = json.dumps(supported, ensure_ascii=False, separators=(",", ":"))
    item["output"] = compact_target
    if "gold_output" in item:
        item["gold_output"] = compact_target
    meta.update(
        {
            "e40_sample_id": f"e130_retrieved_abstention_{sample_index:04d}",
            "e40_source_index": sample_index,
            "e130_target_mode": target_mode,
            "e130_original_gold_event_types": sorted(original_types),
            "e130_supported_target_types": sorted(supported_types),
            "e130_missing_gold_types": missing_types,
            "e130_original_gold_event_count": len(original["events"]),
            "e130_target_event_count": len(supported_events),
            "e130_candidate_consistent": supported_types <= candidate_set,
        }
    )
    return item


def stable_key(row: dict[str, Any], seed: int) -> str:
    wnd_id = str(row.get("meta", {}).get("wnd_id", ""))
    return hashlib.sha256(f"{seed}:{wnd_id}".encode("utf-8")).hexdigest()


def select_smoke(rows: list[dict[str, Any]], seed: int, changed_count: int, unchanged_count: int) -> list[dict[str, Any]]:
    changed = sorted(
        (row for row in rows if row["meta"]["e130_target_mode"] != "gold_present"),
        key=lambda row: stable_key(row, seed),
    )
    unchanged = sorted(
        (row for row in rows if row["meta"]["e130_target_mode"] == "gold_present"),
        key=lambda row: stable_key(row, seed),
    )
    if len(changed) < changed_count or len(unchanged) < unchanged_count:
        raise ValueError(
            f"insufficient smoke strata: changed={len(changed)}, unchanged={len(unchanged)}"
        )
    selected = changed[:changed_count] + unchanged[:unchanged_count]
    return sorted(selected, key=lambda row: stable_key(row, seed + 1))


def normalized_sha(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_manifest = load_jsonl(REPO / protocol["source_manifest"])
    predicted_rows = load_jsonl(REPO / protocol["predicted_train"])
    predicted_by_id = {row.get("meta", {}).get("wnd_id"): row for row in predicted_rows}
    if len(predicted_by_id) != len(predicted_rows):
        raise ValueError("predicted training rows have missing or duplicate wnd_id")
    selected_ids = [row.get("meta", {}).get("wnd_id") for row in source_manifest]
    if len(selected_ids) != protocol["full_manifest_rows"] or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("source manifest size or uniqueness mismatch")
    missing = [wnd_id for wnd_id in selected_ids if wnd_id not in predicted_by_id]
    if missing:
        raise ValueError(f"predicted train is missing {len(missing)} source windows")

    full = [adapt_row(predicted_by_id[wnd_id], index) for index, wnd_id in enumerate(selected_ids)]
    smoke = select_smoke(
        full,
        seed=int(protocol["smoke_seed"]),
        changed_count=int(protocol["smoke_changed_rows"]),
        unchanged_count=int(protocol["smoke_unchanged_rows"]),
    )
    mode_counts = Counter(row["meta"]["e130_target_mode"] for row in full)
    smoke_counts = Counter(row["meta"]["e130_target_mode"] for row in smoke)
    audit = {
        "id": "e130_retrieved_abstention_manifest_audit_v1",
        "full_rows": len(full),
        "smoke_rows": len(smoke),
        "source_predicted_rows": len(predicted_rows),
        "full_mode_counts": dict(sorted(mode_counts.items())),
        "smoke_mode_counts": dict(sorted(smoke_counts.items())),
        "candidate_inconsistent_rows": sum(
            not row["meta"]["e130_candidate_consistent"] for row in full
        ),
        "duplicate_full_wnd_ids": len(full)
        - len({row["meta"]["wnd_id"] for row in full}),
        "smoke_changed_rows": sum(
            row["meta"]["e130_target_mode"] != "gold_present" for row in smoke
        ),
        "smoke_unchanged_rows": sum(
            row["meta"]["e130_target_mode"] == "gold_present" for row in smoke
        ),
        "full_sha256": normalized_sha(full),
        "smoke_sha256": normalized_sha(smoke),
        "test_rows_read": 0,
    }
    expected = {
        "full_rows": int(protocol["full_manifest_rows"]),
        "smoke_rows": int(protocol["smoke_rows"]),
        "candidate_inconsistent_rows": 0,
        "duplicate_full_wnd_ids": 0,
        "smoke_changed_rows": int(protocol["smoke_changed_rows"]),
        "smoke_unchanged_rows": int(protocol["smoke_unchanged_rows"]),
        "test_rows_read": 0,
    }
    failures = {
        key: {"expected": value, "actual": audit[key]}
        for key, value in expected.items()
        if audit[key] != value
    }
    audit["failures"] = failures
    audit["gate_ready"] = not failures
    if failures:
        raise ValueError(f"E130 manifest audit failed: {failures}")
    return full, smoke, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    full, smoke, audit = build(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "e130_retrieved_train_full1500.jsonl", full)
    write_jsonl(args.output_dir / "e130_retrieved_train_smoke40.jsonl", smoke)
    (args.output_dir / "e130_manifest_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
