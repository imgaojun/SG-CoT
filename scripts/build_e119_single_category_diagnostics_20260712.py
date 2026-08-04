#!/usr/bin/env python3
"""Freeze E119 single-category slices from the immutable E115 smoke set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_CATEGORIES = ("extra_frame", "wrong_type", "argument_omission")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_jsonl", type=Path, required=True)
    parser.add_argument("--expected_sha256", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_per_category", type=int, default=8)
    args = parser.parse_args()

    source_hash = sha256(args.source_jsonl)
    if source_hash != args.expected_sha256:
        raise ValueError(f"source SHA256 mismatch: {source_hash}")
    rows = load_jsonl(args.source_jsonl)
    counts = Counter(str(row["meta"]["error_category"]) for row in rows)
    for category in TARGET_CATEGORIES:
        if counts[category] != args.expected_per_category:
            raise ValueError(f"{category}: expected 8 rows, found {counts[category]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = {}
    manifest = []
    artifact_hashes = {}
    for category in TARGET_CATEGORIES:
        selected = [
            row for row in rows if row["meta"]["error_category"] == category
        ]
        windows = [str(row["meta"]["wnd_id"]) for row in selected]
        if len(set(windows)) != len(selected):
            raise ValueError(f"duplicate windows in {category}")
        file_name = f"{category}.jsonl"
        path = args.output_dir / file_name
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )
        dataset_name = f"e119a_{category}"
        dataset_info[dataset_name] = {
            "file_name": file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
            },
            "ranking": True,
        }
        artifact_hashes[category] = sha256(path)
        manifest.append(
            {
                "category": category,
                "dataset_name": dataset_name,
                "path": str(path),
                "sha256": artifact_hashes[category],
                "pairs": len(selected),
                "wnd_ids": windows,
            }
        )

    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "slice_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    frozen = {
        "frozen": True,
        "protocol": "E119 single-category masked-SimPO transfer diagnostic",
        "source_jsonl": str(args.source_jsonl),
        "source_sha256": source_hash,
        "target_categories": list(TARGET_CATEGORIES),
        "pairs_per_category": args.expected_per_category,
        "artifact_sha256": artifact_hashes,
        "test_data_access": False,
    }
    (args.output_dir / "frozen_artifacts.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(frozen, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
