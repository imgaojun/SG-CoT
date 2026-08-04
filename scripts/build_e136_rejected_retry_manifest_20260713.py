#!/usr/bin/env python3
"""Freeze the exact E134 rejected-row retry manifest for E136."""

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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_raw(raw: list[dict[str, Any]], expected: int) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in raw:
        index = int(record["source_index"])
        if index in indexed:
            raise ValueError(f"duplicate parent source index: {index}")
        indexed[index] = record
    if set(indexed) != set(range(expected)):
        raise ValueError("parent raw does not exactly cover the frozen source order")
    return indexed


def select_retry_rows(
    rows: list[dict[str, Any]], raw: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = int(protocol["source_row_count"])
    if len(rows) != expected or len(raw) != expected:
        raise ValueError("source/parent row count mismatch")
    raw_by_index = index_raw(raw, expected)
    selected: list[dict[str, Any]] = []
    mode_counts = {mode: 0 for mode in MODES}
    parse_indices = []
    parent_accepted = 0
    parent_accepted_modes = {mode: 0 for mode in MODES}
    for index, source in enumerate(rows):
        record = raw_by_index[index]
        mode = source["meta"]["e130_target_mode"]
        if mode not in mode_counts:
            raise ValueError(f"unexpected mode at source index {index}: {mode}")
        if record.get("error_stage") == "verifier_parse":
            parse_indices.append(index)
        if record.get("accepted"):
            parent_accepted += 1
            parent_accepted_modes[mode] += 1
            continue
        retry = json.loads(json.dumps(source, ensure_ascii=False))
        retry["meta"]["e136_original_source_index"] = index
        retry["meta"]["e136_retry_rank"] = len(selected)
        retry["meta"]["e136_parent_rejected"] = True
        selected.append(retry)
        mode_counts[mode] += 1

    if parent_accepted != int(protocol["parent_accepted_rows"]):
        raise ValueError("parent accepted count drift")
    if parent_accepted_modes != protocol["parent_accepted_mode_counts"]:
        raise ValueError("parent accepted mode-count drift")
    if len(selected) != int(protocol["retry_rows"]):
        raise ValueError("retry row count drift")
    if mode_counts != protocol["retry_mode_counts"]:
        raise ValueError("retry mode-count drift")
    if parse_indices != protocol["parent_final_verifier_parse_indices"]:
        raise ValueError("parent verifier-parse index drift")
    audit = {
        "id": "e136_rejected_retry_manifest_audit_v1",
        "source_rows": len(rows),
        "parent_accepted_rows": parent_accepted,
        "parent_accepted_mode_counts": parent_accepted_modes,
        "retry_rows": len(selected),
        "retry_mode_counts": mode_counts,
        "parent_final_verifier_parse_indices": parse_indices,
        "test_rows_read": 0,
        "passed": True,
    }
    return selected, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--verify_existing", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.verify_existing:
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    if args.verify_existing and not args.output_dir.is_dir():
        raise ValueError("--verify_existing requires an existing output directory")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("id") != "e136_rejected_row_bounded_retry_v1":
        raise ValueError("unexpected protocol id")
    source_path = Path(protocol["source_rows"])
    raw_path = Path(protocol["parent_raw"])
    gate_path = Path(protocol["parent_gate"])
    train_path = Path(protocol["parent_train"])
    for path, expected in (
        (source_path, protocol["source_rows_sha256"]),
        (raw_path, protocol["parent_raw_sha256"]),
        (gate_path, protocol["parent_gate_sha256"]),
        (train_path, protocol["parent_train_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    parent_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if parent_gate.get("passed") is not False or parent_gate.get("test_rows_read") != 0:
        raise ValueError("E136 requires the frozen failed train-only E134 gate")

    selected, audit = select_retry_rows(
        load_jsonl(source_path), load_jsonl(raw_path), protocol
    )
    manifest_path = args.output_dir / "retry_rows.jsonl"
    if args.verify_existing:
        if load_jsonl(manifest_path) != selected:
            raise ValueError("existing retry manifest content drift")
    else:
        args.output_dir.mkdir(parents=True)
        write_jsonl(manifest_path, selected)
    manifest_hash = sha256_file(manifest_path)
    frozen_hash = protocol["retry_manifest_sha256"]
    audit["retry_manifest_sha256"] = manifest_hash
    audit["protocol_manifest_sha256"] = frozen_hash
    audit["manifest_hash_frozen"] = manifest_hash == frozen_hash
    if frozen_hash != "TO_BE_FROZEN_BEFORE_GENERATION" and manifest_hash != frozen_hash:
        raise ValueError("retry manifest does not match frozen protocol hash")
    (args.output_dir / "manifest_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
