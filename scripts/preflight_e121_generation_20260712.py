#!/usr/bin/env python3
"""Freeze E121 teacher sampling and audit every prompt before API access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

import scripts.generate_strategy_variants_cot_e47_20260606 as generator  # noqa: E402
from scripts.audit_e121_confirmation_data_20260712 import candidates  # noqa: E402
from src.stage2_preference.reasoning_preference import find_heldout_leaks  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest_ids(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--heldout_types_json", type=Path, required=True)
    parser.add_argument("--seen_types_json", type=Path, required=True)
    parser.add_argument("--auto_cluster_map", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--run_name", default="e121_autocluster_confirmation")
    args = parser.parse_args()

    source = load_jsonl(args.input_jsonl)
    selected = generator.e40.sample_rows(source, args.limit, args.seed, args.run_name)
    heldout = set(json.loads(args.heldout_types_json.read_text(encoding="utf-8")))
    seen = set(json.loads(args.seen_types_json.read_text(encoding="utf-8")))
    generator.AUTO_CLUSTER_MAP_PATH = str(args.auto_cluster_map.resolve())
    generator.AUTO_CLUSTER_MAP_CACHE = None

    selected_ids: list[str] = []
    source_leaks = []
    prompt_leaks = []
    candidate_violations = []
    for row in selected:
        wnd_id = str(row.get("meta", {}).get("wnd_id", ""))
        if not wnd_id:
            raise ValueError("selected row lacks meta.wnd_id")
        selected_ids.append(wnd_id)
        row_leaks = find_heldout_leaks(row, heldout)
        if row_leaks:
            source_leaks.append({"wnd_id": wnd_id, "leaks": row_leaks[:10]})
        row_candidates = candidates(row.get("input", ""))
        invalid = sorted(set(row_candidates) - seen)
        if invalid:
            candidate_violations.append({"wnd_id": wnd_id, "types": invalid})
        prompt = generator.generator_prompt(
            row, prompt_profile="e95_trigger_locked_autocluster"
        )
        leaks = find_heldout_leaks(prompt, heldout)
        if leaks:
            prompt_leaks.append({"wnd_id": wnd_id, "leaks": leaks[:10]})

    checks = {
        "selected_count_exact": len(selected) == args.limit,
        "unique_windows": len(set(selected_ids)) == len(selected_ids),
        "source_heldout_leaks_zero": not source_leaks,
        "prompt_heldout_leaks_zero": not prompt_leaks,
        "candidate_violations_zero": not candidate_violations,
    }
    report = {
        "id": "e121c_generation_preflight_v1",
        "input_jsonl": str(args.input_jsonl.resolve()),
        "source_rows": len(source),
        "selection_mode": "e40_priority_sample_v1",
        "limit": args.limit,
        "seed": args.seed,
        "run_name": args.run_name,
        "prompt_profile": "e95_trigger_locked_autocluster",
        "selected_count": len(selected),
        "selected_wnd_ids_sha256": digest_ids(selected_ids),
        "selected_wnd_ids": selected_ids,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "failures": {
            "source_heldout_leaks": source_leaks[:20],
            "prompt_heldout_leaks": prompt_leaks[:20],
            "candidate_violations": candidate_violations[:20],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "selected_wnd_ids"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["all_checks_pass"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
