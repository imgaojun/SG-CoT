#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.generate_strategy_variants_cot_e47_20260606 as generator
from scripts.reverify_e125_role_alias_20260712 import build_items, load_jsonl, write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstructed_output(attempt: dict) -> str:
    return (
        f"<thinking>{attempt['thinking']}</thinking>"
        f"<final>{json.dumps(attempt['final_obj'], ensure_ascii=False)}</final>"
    )


def build_dataset(items: list[dict], run_name: str) -> tuple[list[dict], list[dict], dict]:
    generator.ACTIVE_GENERATOR_MODEL = "glm-5.1"
    generator.ACTIVE_VERIFIER_MODEL = "deterministic-hard-gate-v1"
    output_rows = []
    manifest = []
    missing = []
    reverify_failures = []
    for item in items:
        attempt = item["selected_attempt"]
        if attempt is None:
            missing.append(item["sample_id"])
            continue
        thinking, final_obj, hard_errors = generator.hard_verify(
            item["source_row"], reconstructed_output(attempt)
        )
        if hard_errors:
            reverify_failures.append(
                {"sample_id": item["sample_id"], "hard_errors": hard_errors}
            )
            continue
        output_row = generator.make_evidence_row(
            item["source_row"], thinking, final_obj, "train", run_name
        )
        output_rows.append(output_row)
        manifest.append(
            {
                "sample_id": item["sample_id"],
                "wnd_id": (item["source_row"].get("meta") or {}).get("wnd_id"),
                "source_index": item["source_index"],
                "selected_attempt_number": item["selected_attempt_number"],
                "originally_accepted": item["originally_accepted"],
                "output_sha256": hashlib.sha256(
                    output_row["output"].encode()
                ).hexdigest(),
            }
        )
    summary = {
        "protocol": "e126-deterministic-hard-valid-existing-traces-v1",
        "counts": {
            "source_rows": len(items),
            "selected_hard_valid": len(output_rows),
            "missing_hard_valid": len(missing),
            "hard_reverify_failures": len(reverify_failures),
            "originally_accepted": sum(row["originally_accepted"] for row in manifest),
            "originally_rejected": sum(not row["originally_accepted"] for row in manifest),
        },
        "checks": {
            "exact_1500_source_rows": len(items) == 1500,
            "minimum_1499_hard_valid": len(output_rows) >= 1499,
            "all_selected_reverify_hard_valid": not reverify_failures,
            "unique_sample_ids": len({row["sample_id"] for row in manifest}) == len(manifest),
            "unique_wnd_ids": len({row["wnd_id"] for row in manifest}) == len(manifest),
        },
        "missing_sample_ids": missing,
        "hard_reverify_failures": reverify_failures,
    }
    summary["passed"] = all(summary["checks"].values())
    return output_rows, manifest, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_jsonl", type=Path, required=True)
    parser.add_argument("--sampled_rows_jsonl", type=Path, required=True)
    parser.add_argument("--raw_sha256", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_name", default="e126_deterministic_hard_valid")
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    observed_sha = sha256_file(args.raw_jsonl)
    if observed_sha != args.raw_sha256:
        raise SystemExit(
            f"raw SHA mismatch: expected {args.raw_sha256}, observed {observed_sha}"
        )
    items = build_items(load_jsonl(args.raw_jsonl), load_jsonl(args.sampled_rows_jsonl))
    rows, manifest, summary = build_dataset(items, args.run_name)
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "hard_valid_evidence_cot.jsonl", rows)
    write_jsonl(args.output_dir / "selection_manifest.jsonl", manifest)
    summary["raw_sha256"] = observed_sha
    summary["dataset_sha256"] = sha256_file(
        args.output_dir / "hard_valid_evidence_cot.jsonl"
    )
    summary["selection_manifest_sha256"] = sha256_file(
        args.output_dir / "selection_manifest.jsonl"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 6 if args.require_pass and not summary["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
