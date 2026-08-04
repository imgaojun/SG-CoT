#!/usr/bin/env python3
"""Audit paired Direct/SG-CoT E121 datasets before any model evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_preference.reasoning_preference import (  # noqa: E402
    extract_final_json,
    find_heldout_leaks,
    is_exact,
    recover_offsets_from_evidence,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def as_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("expected a JSON object")


def candidates(input_text: str) -> list[str]:
    match = re.search(
        r"Candidate event types:\n(.*?)(?:\n\nSchema cards:|\Z)", input_text, re.S
    )
    if not match:
        raise ValueError("missing Candidate event types section")
    return [item.strip() for item in match.group(1).replace("\n", " ").split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--sgcot_jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--heldout_types_json", type=Path, required=True)
    parser.add_argument("--seen_types_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    if len(args.direct_jsonl) != len(args.sgcot_jsonl):
        raise ValueError("Direct and SG-CoT file lists must have the same length")

    heldout = set(json.loads(args.heldout_types_json.read_text(encoding="utf-8")))
    seen = set(json.loads(args.seen_types_json.read_text(encoding="utf-8")))
    allowed = heldout | seen
    counters = {
        "files": 0,
        "rows": 0,
        "paired_rows": 0,
        "train_rows": 0,
        "train_heldout_leaks": 0,
        "candidate_violations": 0,
        "gold_candidate_violations": 0,
        "surface_recovery_failures": 0,
        "paired_final_mismatches": 0,
        "paired_input_mismatches": 0,
        "missing_labels": 0,
    }
    examples: dict[str, list[dict[str, Any]]] = {}

    def record(name: str, payload: dict[str, Any]) -> None:
        examples.setdefault(name, [])
        if len(examples[name]) < 20:
            examples[name].append(payload)

    for direct_path, sgcot_path in zip(args.direct_jsonl, args.sgcot_jsonl):
        direct_rows = load_jsonl(direct_path)
        sgcot_rows = load_jsonl(sgcot_path)
        if len(direct_rows) != len(sgcot_rows):
            raise ValueError(f"row-count mismatch: {direct_path} vs {sgcot_path}")
        counters["files"] += 2
        for index, (direct, sgcot) in enumerate(zip(direct_rows, sgcot_rows)):
            counters["rows"] += 2
            counters["paired_rows"] += 1
            wnd_id = direct.get("meta", {}).get("wnd_id")
            if wnd_id != sgcot.get("meta", {}).get("wnd_id"):
                raise ValueError(f"wnd_id mismatch at {direct_path}:{index}")
            if direct.get("input") != sgcot.get("input") or direct.get("gold_output") != sgcot.get("gold_output"):
                counters["paired_input_mismatches"] += 1
                record("paired_input_mismatches", {"wnd_id": wnd_id, "file": str(direct_path)})
            direct_final = extract_final_json(direct.get("output", ""))
            sgcot_final = extract_final_json(sgcot.get("output", ""))
            if direct_final is None or sgcot_final is None or direct_final != sgcot_final:
                counters["paired_final_mismatches"] += 1
                record("paired_final_mismatches", {"wnd_id": wnd_id, "file": str(direct_path)})

            gold_raw = direct.get("gold_output")
            if gold_raw is None or direct_final is None:
                counters["missing_labels"] += 1
                continue
            gold = as_object(gold_raw)
            row_candidates = candidates(direct.get("input", ""))
            invalid_candidates = sorted(set(row_candidates) - allowed)
            if invalid_candidates:
                counters["candidate_violations"] += 1
                record("candidate_violations", {"wnd_id": wnd_id, "types": invalid_candidates})
            gold_types = {
                event.get("event_type") for event in gold.get("events", []) if isinstance(event, dict)
            }
            missing_candidates = sorted(gold_types - set(row_candidates))
            if missing_candidates:
                counters["gold_candidate_violations"] += 1
                record("gold_candidate_violations", {"wnd_id": wnd_id, "types": missing_candidates})
            recovered, diagnostics = recover_offsets_from_evidence(direct_final, direct.get("input", ""))
            if diagnostics["missing_offsets"] or not is_exact(recovered, gold):
                counters["surface_recovery_failures"] += 1
                record("surface_recovery_failures", {"wnd_id": wnd_id, "diagnostics": diagnostics})

            if direct.get("meta", {}).get("source_part") == "train":
                counters["train_rows"] += 1
                leaks = find_heldout_leaks(direct, heldout) + find_heldout_leaks(sgcot, heldout)
                if leaks:
                    counters["train_heldout_leaks"] += len(leaks)
                    record("train_heldout_leaks", {"wnd_id": wnd_id, "leaks": leaks[:10]})
                if set(row_candidates) - seen:
                    counters["candidate_violations"] += 1
                    record(
                        "candidate_violations",
                        {"wnd_id": wnd_id, "train_nonseen": sorted(set(row_candidates) - seen)},
                    )

    required_zero = [
        "train_heldout_leaks",
        "candidate_violations",
        "gold_candidate_violations",
        "surface_recovery_failures",
        "paired_final_mismatches",
        "paired_input_mismatches",
        "missing_labels",
    ]
    report = {
        **counters,
        "heldout_types": sorted(heldout),
        "all_checks_pass": all(counters[name] == 0 for name in required_zero),
        "examples": examples,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_pass"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
