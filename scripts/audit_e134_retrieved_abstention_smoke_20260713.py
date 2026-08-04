#!/usr/bin/env python3
"""Apply the preregistered E134 stratified smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_gate(rows: list[dict[str, Any]], raw: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    if len(rows) != int(protocol["smoke_rows"]) or len(raw) != len(rows):
        raise ValueError(f"row/raw count mismatch: {len(rows)} / {len(raw)}")
    raw_by_suffix = {}
    for record in raw:
        suffix = str(record["sample_id"]).rsplit("_", 1)[-1]
        raw_by_suffix[int(suffix)] = record
    if set(raw_by_suffix) != set(range(len(rows))):
        raise ValueError("raw sample ids do not cover the frozen row order")

    counts = {
        "changed_total": 0,
        "changed_accepted": 0,
        "unchanged_total": 0,
        "unchanged_accepted": 0,
        "accepted": 0,
        "final_verifier_parse_errors": 0,
        "accepted_candidate_inconsistent": 0,
    }
    failures = []
    for index, row in enumerate(rows):
        record = raw_by_suffix[index]
        mode = row["meta"]["e130_target_mode"]
        stratum = "unchanged" if mode == "gold_present" else "changed"
        counts[f"{stratum}_total"] += 1
        if record.get("error_stage") == "verifier_parse":
            counts["final_verifier_parse_errors"] += 1
        if not record.get("accepted"):
            failures.append({"index": index, "mode": mode, "reason": record.get("error") or record.get("hard_errors")})
            continue
        counts["accepted"] += 1
        counts[f"{stratum}_accepted"] += 1
        candidates = set(row["meta"]["candidate_types"])
        final_obj = record.get("final_obj") if isinstance(record.get("final_obj"), dict) else {}
        predicted_types = {
            event.get("event_type")
            for event in final_obj.get("events", [])
            if isinstance(event, dict)
        }
        if not predicted_types <= candidates:
            counts["accepted_candidate_inconsistent"] += 1

    checks = {
        "accepted": counts["accepted"] >= int(protocol["smoke_min_accepted"]),
        "changed_accepted": counts["changed_accepted"] >= int(protocol["smoke_min_changed_accepted"]),
        "unchanged_accepted": counts["unchanged_accepted"] >= int(protocol["smoke_min_unchanged_accepted"]),
        "zero_final_verifier_parse_errors": counts["final_verifier_parse_errors"] == 0,
        "candidate_consistent": counts["accepted_candidate_inconsistent"] == 0,
        "strata_exact": (
            counts["changed_total"] == int(protocol["smoke_changed_rows"])
            and counts["unchanged_total"] == int(protocol["smoke_unchanged_rows"])
        ),
    }
    return {"counts": counts, "checks": checks, "failures": failures, "passed": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rows_jsonl", type=Path, required=True)
    parser.add_argument("--raw_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("id") != "e134_retrieved_abstention_glm_verifier_v1":
        raise ValueError("unexpected protocol id")
    if sha256_file(args.rows_jsonl) != protocol["source_smoke_sha256"]:
        raise ValueError("smoke input hash does not match frozen protocol")
    result = compute_gate(load_jsonl(args.rows_jsonl), load_jsonl(args.raw_jsonl), protocol)
    result.update(
        {
            "id": "e134_retrieved_abstention_smoke_gate_v1",
            "source_smoke_sha256": protocol["source_smoke_sha256"],
            "raw_sha256": sha256_file(args.raw_jsonl),
            "test_rows_read": 0,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_pass and not result["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
