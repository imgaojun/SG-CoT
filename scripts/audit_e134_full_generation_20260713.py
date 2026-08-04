#!/usr/bin/env python3
"""Apply the frozen E134 full train-only generation gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.audit_e134_retrieved_abstention_smoke_20260713 import load_jsonl, sha256_file


MODES = ("gold_present", "partial_supported", "abstain")


def compute_full_gate(
    rows: list[dict[str, Any]], raw: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    expected_rows = int(protocol["full_rows"])
    if len(rows) != expected_rows or len(raw) != expected_rows:
        raise ValueError(f"row/raw count mismatch: {len(rows)} / {len(raw)}")
    raw_by_index = {}
    for record in raw:
        index = int(str(record["sample_id"]).rsplit("_", 1)[-1])
        raw_by_index[index] = record
    if set(raw_by_index) != set(range(expected_rows)):
        raise ValueError("raw ids do not exactly cover the frozen source order")

    totals = {mode: 0 for mode in MODES}
    accepted = {mode: 0 for mode in MODES}
    final_verifier_parse_errors = 0
    accepted_candidate_inconsistent = 0
    for index, row in enumerate(rows):
        mode = row["meta"]["e130_target_mode"]
        if mode not in totals:
            raise ValueError(f"unexpected target mode: {mode}")
        totals[mode] += 1
        record = raw_by_index[index]
        final_verifier_parse_errors += int(record.get("error_stage") == "verifier_parse")
        if not record.get("accepted"):
            continue
        accepted[mode] += 1
        candidates = set(row["meta"]["candidate_types"])
        final_obj = record.get("final_obj") if isinstance(record.get("final_obj"), dict) else {}
        predicted_types = {
            event.get("event_type")
            for event in final_obj.get("events", [])
            if isinstance(event, dict)
        }
        accepted_candidate_inconsistent += int(not predicted_types <= candidates)

    accepted_total = sum(accepted.values())
    checks = {
        "source_mode_counts_exact": (
            totals["gold_present"] == int(protocol["full_gold_present_rows"])
            and totals["partial_supported"] == int(protocol["full_partial_supported_rows"])
            and totals["abstain"] == int(protocol["full_abstain_rows"])
        ),
        "accepted_total": accepted_total >= int(protocol["full_min_accepted"]),
        "accepted_gold_present": accepted["gold_present"]
        >= int(protocol["full_min_gold_present_accepted"]),
        "accepted_partial_supported": accepted["partial_supported"]
        >= int(protocol["full_min_partial_supported_accepted"]),
        "accepted_abstain": accepted["abstain"] >= int(protocol["full_min_abstain_accepted"]),
        "zero_final_verifier_parse_errors": final_verifier_parse_errors == 0,
        "candidate_consistent": accepted_candidate_inconsistent == 0,
    }
    return {
        "source_mode_counts": totals,
        "accepted_mode_counts": accepted,
        "accepted_total": accepted_total,
        "final_verifier_parse_errors": final_verifier_parse_errors,
        "accepted_candidate_inconsistent": accepted_candidate_inconsistent,
        "checks": checks,
        "passed": all(checks.values()),
    }


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
    if sha256_file(args.rows_jsonl) != protocol["full_source_sha256"]:
        raise ValueError("full source hash does not match frozen protocol")
    result = compute_full_gate(load_jsonl(args.rows_jsonl), load_jsonl(args.raw_jsonl), protocol)
    result.update(
        {
            "id": "e134_retrieved_abstention_full_gate_v1",
            "source_sha256": protocol["full_source_sha256"],
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
