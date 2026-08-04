#!/usr/bin/env python3
"""Audit strict-unseen JSONL data for label leakage and surface recovery integrity."""

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
    extract_final_json,
    find_heldout_leaks,
    is_exact,
    recover_offsets_from_evidence,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--heldout_types_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--require_zero_leaks", action="store_true")
    parser.add_argument("--require_exact_surface_recovery", action="store_true")
    args = parser.parse_args()

    heldout_types = json.loads(args.heldout_types_json.read_text(encoding="utf-8"))
    if not isinstance(heldout_types, list):
        raise ValueError("heldout_types_json must contain a JSON list")
    report: dict[str, Any] = {
        "heldout_types": heldout_types,
        "files": {},
        "total_rows": 0,
        "total_leaks": 0,
        "surface_rows": 0,
        "exact_surface_recovery_rows": 0,
        "surface_recovery_failures": [],
    }
    for path in args.input_jsonl:
        rows = load_jsonl(path)
        file_leaks = []
        for index, row in enumerate(rows):
            for leak in find_heldout_leaks(row, heldout_types):
                file_leaks.append({"row_index": index, **leak})
            final_payload = extract_final_json(row.get("output", ""))
            gold_raw = row.get("gold_output")
            if final_payload is not None and gold_raw is not None:
                report["surface_rows"] += 1
                gold = json.loads(gold_raw) if isinstance(gold_raw, str) else gold_raw
                recovered, diagnostics = recover_offsets_from_evidence(final_payload, row.get("input", ""))
                if diagnostics["missing_offsets"] == 0 and is_exact(recovered, gold):
                    report["exact_surface_recovery_rows"] += 1
                else:
                    report["surface_recovery_failures"].append(
                        {
                            "file": str(path.resolve()),
                            "row_index": index,
                            "wnd_id": row.get("meta", {}).get("wnd_id"),
                            "missing_offsets": diagnostics["missing_offsets"],
                        }
                    )
        report["files"][str(path.resolve())] = {
            "rows": len(rows),
            "leaks": len(file_leaks),
            "leak_examples": file_leaks[:50],
        }
        report["total_rows"] += len(rows)
        report["total_leaks"] += len(file_leaks)
    report["zero_leaks"] = report["total_leaks"] == 0
    report["all_surface_rows_recover_exactly"] = (
        not report["surface_recovery_failures"]
        and report["surface_rows"] == report["exact_surface_recovery_rows"]
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failed = (
        args.require_zero_leaks and not report["zero_leaks"]
    ) or (
        args.require_exact_surface_recovery and not report["all_surface_rows_recover_exactly"]
    )
    return 5 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

