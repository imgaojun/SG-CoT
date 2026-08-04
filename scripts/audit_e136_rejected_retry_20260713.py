#!/usr/bin/env python3
"""Combine E134 locked accepts with E136 bounded retries and apply the frozen gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODES = ("gold_present", "partial_supported", "abstain")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_records(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        index = int(record["source_index"])
        if index in indexed:
            raise ValueError(f"duplicate raw source index: {index}")
        indexed[index] = record
    return indexed


def index_train_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["meta"]["e40_source_index"])
        if index in indexed:
            raise ValueError(f"duplicate train source index: {index}")
        indexed[index] = row
    return indexed


def choose_final_record(
    parent: dict[str, Any], retry: dict[str, Any] | None
) -> dict[str, Any]:
    if parent.get("accepted") or retry is None:
        return parent
    if retry.get("accepted"):
        return retry
    if (
        parent.get("error_stage") == "verifier_parse"
        and retry.get("error_stage") != "verifier_parse"
    ):
        return retry
    return parent


def candidate_inconsistent(record: dict[str, Any], source: dict[str, Any]) -> bool:
    candidates = set(source["meta"]["candidate_types"])
    final_obj = record.get("final_obj") if isinstance(record.get("final_obj"), dict) else {}
    predicted = {
        event.get("event_type")
        for event in final_obj.get("events", [])
        if isinstance(event, dict)
    }
    return not predicted <= candidates


def compute_combined_gate(
    rows: list[dict[str, Any]],
    parent_by_index: dict[int, dict[str, Any]],
    retry_by_index: dict[int, dict[str, Any]],
    retry_indices: set[int],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], set[int]]:
    accepted_modes = {mode: 0 for mode in MODES}
    accepted_indices: set[int] = set()
    selected_parse_errors = 0
    candidate_errors = 0
    locked_parent_rows = 0
    retry_accepts = 0
    for index, source in enumerate(rows):
        parent = parent_by_index[index]
        retry = retry_by_index.get(index)
        if parent.get("accepted"):
            locked_parent_rows += 1
        chosen = choose_final_record(parent, retry)
        selected_parse_errors += int(chosen.get("error_stage") == "verifier_parse")
        if not chosen.get("accepted"):
            continue
        accepted_indices.add(index)
        mode = source["meta"]["e130_target_mode"]
        accepted_modes[mode] += 1
        candidate_errors += int(candidate_inconsistent(chosen, source))
        retry_accepts += int(not parent.get("accepted"))

    accepted_total = sum(accepted_modes.values())
    frozen = protocol["gate"]
    checks = {
        "locked_parent_rows_exact": locked_parent_rows
        == int(frozen["required_locked_parent_rows"]),
        "retry_coverage_exact": len(retry_indices)
        == int(frozen["required_retry_coverage"]),
        "accepted_total": accepted_total >= int(frozen["minimum_accepted_total"]),
        "accepted_gold_present": accepted_modes["gold_present"]
        >= int(frozen["minimum_gold_present_accepted"]),
        "accepted_partial_supported": accepted_modes["partial_supported"]
        >= int(frozen["minimum_partial_supported_accepted"]),
        "accepted_abstain": accepted_modes["abstain"]
        >= int(frozen["minimum_abstain_accepted"]),
        "zero_selected_final_verifier_parse_errors": selected_parse_errors
        <= int(frozen["maximum_selected_final_verifier_parse_errors"]),
        "candidate_consistent": candidate_errors
        <= int(frozen["maximum_accepted_candidate_inconsistent"]),
    }
    result = {
        "locked_parent_rows": locked_parent_rows,
        "retry_rows": len(retry_indices),
        "retry_accepted_rows": retry_accepts,
        "accepted_mode_counts": accepted_modes,
        "accepted_total": accepted_total,
        "selected_final_verifier_parse_errors": selected_parse_errors,
        "accepted_candidate_inconsistent": candidate_errors,
        "checks": checks,
    }
    return result, accepted_indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--retry_raw_jsonl", type=Path, required=True)
    parser.add_argument("--retry_train_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("id") != "e136_rejected_row_bounded_retry_v1":
        raise ValueError("unexpected protocol id")
    if protocol["retry_manifest_sha256"] == "TO_BE_FROZEN_BEFORE_GENERATION":
        raise ValueError("retry manifest hash is not frozen")
    frozen_paths = (
        (Path(protocol["source_rows"]), protocol["source_rows_sha256"]),
        (Path(protocol["parent_raw"]), protocol["parent_raw_sha256"]),
        (Path(protocol["parent_gate"]), protocol["parent_gate_sha256"]),
        (Path(protocol["parent_train"]), protocol["parent_train_sha256"]),
        (Path(protocol["retry_manifest"]), protocol["retry_manifest_sha256"]),
    )
    for path, expected in frozen_paths:
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")

    rows = load_jsonl(Path(protocol["source_rows"]))
    parent_raw = load_jsonl(Path(protocol["parent_raw"]))
    retry_manifest = load_jsonl(Path(protocol["retry_manifest"]))
    retry_raw = load_jsonl(args.retry_raw_jsonl)
    parent_by_index = index_records(parent_raw)
    retry_by_index = index_records(retry_raw)
    expected_all = set(range(int(protocol["source_row_count"])))
    if set(parent_by_index) != expected_all:
        raise ValueError("parent raw coverage drift")
    retry_indices = {
        int(row["meta"]["e136_original_source_index"]) for row in retry_manifest
    }
    if len(retry_indices) != len(retry_manifest):
        raise ValueError("duplicate retry manifest source index")
    if set(retry_by_index) != retry_indices:
        raise ValueError("retry raw does not exactly cover the frozen retry manifest")
    if any(parent_by_index[index].get("accepted") for index in retry_indices):
        raise ValueError("retry attempted a locked parent-accepted row")

    result, accepted_indices = compute_combined_gate(
        rows, parent_by_index, retry_by_index, retry_indices, protocol
    )
    parent_train_path = Path(protocol["parent_train"])
    parent_train = load_jsonl(parent_train_path)
    retry_train = load_jsonl(args.retry_train_jsonl)
    parent_train_by_index = index_train_rows(parent_train)
    retry_train_by_index = index_train_rows(retry_train)
    parent_accepted_indices = {
        index for index, record in parent_by_index.items() if record.get("accepted")
    }
    retry_accepted_indices = {
        index for index, record in retry_by_index.items() if record.get("accepted")
    }
    train_pairing_exact = (
        set(parent_train_by_index) == parent_accepted_indices
        and set(retry_train_by_index) == retry_accepted_indices
        and parent_accepted_indices.isdisjoint(retry_accepted_indices)
        and parent_accepted_indices | retry_accepted_indices == accepted_indices
    )
    duplicate_source_indices = (
        len(parent_train) + len(retry_train) - len(accepted_indices)
    )
    result["checks"]["train_pairing_exact"] = train_pairing_exact
    result["checks"]["duplicate_source_indices"] = duplicate_source_indices <= int(
        protocol["gate"]["maximum_duplicate_source_indices"]
    )
    result["duplicate_source_indices"] = duplicate_source_indices

    args.output_dir.mkdir(parents=True)
    combined_path = args.output_dir / "combined_accepted_train.jsonl"
    parent_bytes = parent_train_path.read_bytes()
    retry_bytes = args.retry_train_jsonl.read_bytes()
    if parent_bytes and not parent_bytes.endswith(b"\n"):
        raise ValueError("parent train file lacks terminal newline")
    combined_path.write_bytes(parent_bytes + retry_bytes)
    combined_rows = load_jsonl(combined_path)
    prefix_locked = hashlib.sha256(
        combined_path.read_bytes()[: len(parent_bytes)]
    ).hexdigest() == protocol["parent_train_sha256"]
    result["checks"]["parent_train_byte_prefix_locked"] = prefix_locked
    result["checks"]["combined_train_count_exact"] = len(combined_rows) == result[
        "accepted_total"
    ]
    result.update(
        {
            "id": "e136_rejected_row_bounded_retry_gate_v1",
            "parent_raw_sha256": protocol["parent_raw_sha256"],
            "retry_manifest_sha256": protocol["retry_manifest_sha256"],
            "retry_raw_sha256": sha256_file(args.retry_raw_jsonl),
            "retry_train_sha256": sha256_file(args.retry_train_jsonl),
            "combined_train_sha256": sha256_file(combined_path),
            "combined_train_rows": len(combined_rows),
            "test_rows_read": 0,
        }
    )
    result["passed"] = all(result["checks"].values())
    gate_path = args.output_dir / "gate.json"
    gate_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_pass and not result["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
