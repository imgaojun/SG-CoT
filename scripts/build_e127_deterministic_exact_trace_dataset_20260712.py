#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.generate_strategy_variants_cot_e47_20260606 as generator
from scripts.build_e126_deterministic_trace_dataset_20260712 import (
    reconstructed_output,
    sha256_file,
)
from scripts.reverify_e125_role_alias_20260712 import build_items, load_jsonl, write_jsonl
from src.stage2_preference.reasoning_preference import (
    find_heldout_leaks,
    is_exact,
    recover_offsets_from_evidence,
)


def build_dataset(
    items: list[dict], heldout_types: list[str], run_name: str, minimum_rows: int
) -> tuple[list[dict], list[dict], list[dict], dict]:
    generator.ACTIVE_GENERATOR_MODEL = "glm-5.1"
    generator.ACTIVE_VERIFIER_MODEL = "deterministic-exact-gate-v1"
    output_rows = []
    manifest = []
    excluded = []
    for item in items:
        attempt = item["selected_attempt"]
        if attempt is None:
            excluded.append(
                {"sample_id": item["sample_id"], "reason": "no_existing_hard_valid_attempt"}
            )
            continue
        thinking, final_obj, hard_errors = generator.hard_verify(
            item["source_row"], reconstructed_output(attempt)
        )
        if hard_errors:
            excluded.append(
                {
                    "sample_id": item["sample_id"],
                    "reason": "hard_reverify_failure",
                    "details": hard_errors,
                }
            )
            continue
        recovered, diagnostics = recover_offsets_from_evidence(
            final_obj, item["source_row"]["input"]
        )
        gold = json.loads(item["source_row"]["gold_output"])
        if diagnostics["missing_offsets"] or not is_exact(recovered, gold):
            excluded.append(
                {
                    "sample_id": item["sample_id"],
                    "reason": "raw_final_not_exact_gold",
                    "missing_offsets": diagnostics["missing_offsets"],
                }
            )
            continue
        output_row = generator.make_evidence_row(
            item["source_row"], thinking, final_obj, "train", run_name
        )
        leaks = find_heldout_leaks(output_row, heldout_types)
        if leaks:
            excluded.append(
                {
                    "sample_id": item["sample_id"],
                    "wnd_id": (item["source_row"].get("meta") or {}).get("wnd_id"),
                    "reason": "heldout_string_leak",
                    "details": leaks,
                }
            )
            continue
        output_rows.append(output_row)
        manifest.append(
            {
                "sample_id": item["sample_id"],
                "wnd_id": (item["source_row"].get("meta") or {}).get("wnd_id"),
                "source_index": item["source_index"],
                "selected_attempt_number": item["selected_attempt_number"],
                "originally_accepted": item["originally_accepted"],
                "raw_final_exact_gold": True,
                "heldout_string_leaks": 0,
                "output_sha256": hashlib.sha256(output_row["output"].encode()).hexdigest(),
            }
        )
    reasons = Counter(row["reason"] for row in excluded)
    checks = {
        "exact_1500_source_rows": len(items) == 1500,
        "minimum_deterministic_exact_zero_leak_rows": len(output_rows) >= minimum_rows,
        "all_kept_raw_finals_exact_gold": all(
            row["raw_final_exact_gold"] for row in manifest
        ),
        "all_kept_zero_heldout_leaks": all(
            row["heldout_string_leaks"] == 0 for row in manifest
        ),
        "unique_sample_ids": len({row["sample_id"] for row in manifest}) == len(manifest),
        "unique_wnd_ids": len({row["wnd_id"] for row in manifest}) == len(manifest),
        "complete_partition": len(output_rows) + len(excluded) == len(items),
    }
    summary = {
        "protocol": "e127-deterministic-exact-zero-leak-existing-traces-v1",
        "counts": {
            "source_rows": len(items),
            "kept_rows": len(output_rows),
            "excluded_rows": len(excluded),
            "originally_accepted_kept": sum(
                row["originally_accepted"] for row in manifest
            ),
            "originally_rejected_kept": sum(
                not row["originally_accepted"] for row in manifest
            ),
            "exclusion_reasons": dict(sorted(reasons.items())),
        },
        "minimum_rows": minimum_rows,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return output_rows, manifest, excluded, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_jsonl", type=Path, required=True)
    parser.add_argument("--sampled_rows_jsonl", type=Path, required=True)
    parser.add_argument("--raw_sha256", required=True)
    parser.add_argument("--sampled_rows_sha256", required=True)
    parser.add_argument("--heldout_types_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_name", default="e127_deterministic_exact_zero_leak")
    parser.add_argument("--minimum_rows", type=int, default=1400)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    observed_raw_sha = sha256_file(args.raw_jsonl)
    observed_sampled_sha = sha256_file(args.sampled_rows_jsonl)
    if observed_raw_sha != args.raw_sha256:
        raise SystemExit(
            f"raw SHA mismatch: expected {args.raw_sha256}, observed {observed_raw_sha}"
        )
    if observed_sampled_sha != args.sampled_rows_sha256:
        raise SystemExit(
            "sampled-row SHA mismatch: "
            f"expected {args.sampled_rows_sha256}, observed {observed_sampled_sha}"
        )
    heldout_types = json.loads(args.heldout_types_json.read_text(encoding="utf-8"))
    items = build_items(load_jsonl(args.raw_jsonl), load_jsonl(args.sampled_rows_jsonl))
    rows, manifest, excluded, summary = build_dataset(
        items, heldout_types, args.run_name, args.minimum_rows
    )
    args.output_dir.mkdir(parents=True)
    dataset_path = args.output_dir / "deterministic_exact_evidence_cot.jsonl"
    manifest_path = args.output_dir / "selection_manifest.jsonl"
    excluded_path = args.output_dir / "excluded_manifest.jsonl"
    write_jsonl(dataset_path, rows)
    write_jsonl(manifest_path, manifest)
    write_jsonl(excluded_path, excluded)
    summary.update(
        {
            "raw_sha256": observed_raw_sha,
            "sampled_rows_sha256": observed_sampled_sha,
            "dataset_sha256": sha256_file(dataset_path),
            "selection_manifest_sha256": sha256_file(manifest_path),
            "excluded_manifest_sha256": sha256_file(excluded_path),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 6 if args.require_pass and not summary["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
