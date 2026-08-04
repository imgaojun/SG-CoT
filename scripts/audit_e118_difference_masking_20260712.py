#!/usr/bin/env python3
"""Freeze and audit the token-difference mask used by E118."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from score_e115_training_diagnostics_20260712 import encoded_conversation, load_jsonl
from src.stage2_preference.difference_masking import divergent_token_indices


EXPECTED_CATEGORIES = {
    "argument_omission",
    "event_omission",
    "extra_frame",
    "trigger_drift",
    "wrong_type",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def response_ids(tokenizer: Any, row: dict[str, Any], response: str) -> list[int]:
    prompt_ids, full_ids = encoded_conversation(tokenizer, row, response)
    ids = full_ids[len(prompt_ids) :]
    if not ids:
        raise ValueError("assistant response tokenization is empty")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference_jsonl", type=Path, required=True)
    parser.add_argument("--expected_sha256", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--context_tokens", type=int, default=1)
    parser.add_argument("--expected_pairs", type=int, default=40)
    parser.add_argument("--expected_per_category", type=int, default=8)
    args = parser.parse_args()

    source_hash = sha256(args.preference_jsonl)
    if source_hash != args.expected_sha256:
        raise ValueError(
            f"source SHA256 mismatch: {source_hash} != {args.expected_sha256}"
        )
    rows = load_jsonl(args.preference_jsonl)
    if len(rows) != args.expected_pairs:
        raise ValueError(f"expected {args.expected_pairs} pairs, found {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    manifest = []
    counts: Counter[str] = Counter()
    windows: set[str] = set()
    documents: set[str] = set()
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("meta", {})
        wnd_id = str(meta.get("wnd_id", ""))
        category = str(meta.get("error_category", ""))
        if not wnd_id or category not in EXPECTED_CATEGORIES:
            raise ValueError(f"invalid pair metadata: {wnd_id!r}, {category!r}")
        if wnd_id in windows:
            raise ValueError(f"duplicate window: {wnd_id}")
        windows.add(wnd_id)
        document_id = wnd_id.rsplit("-", 1)[0]
        documents.add(document_id)
        counts[category] += 1

        chosen_ids = response_ids(tokenizer, row, row["chosen"])
        rejected_ids = response_ids(tokenizer, row, row["rejected"])
        chosen_keep, rejected_keep = divergent_token_indices(
            chosen_ids, rejected_ids, args.context_tokens
        )
        item = {
            "wnd_id": wnd_id,
            "document_id": document_id,
            "error_category": category,
            "chosen_response_tokens": len(chosen_ids),
            "rejected_response_tokens": len(rejected_ids),
            "chosen_kept_tokens": len(chosen_keep),
            "rejected_kept_tokens": len(rejected_keep),
            "chosen_keep_ratio": len(chosen_keep) / len(chosen_ids),
            "rejected_keep_ratio": len(rejected_keep) / len(rejected_ids),
            "chosen_keep_indices": chosen_keep,
            "rejected_keep_indices": rejected_keep,
        }
        manifest.append(item)
        category_rows[category].append(item)

    expected_counts = Counter(
        {category: args.expected_per_category for category in EXPECTED_CATEGORIES}
    )
    if counts != expected_counts:
        raise ValueError(f"category counts differ: {dict(counts)}")
    if len(documents) != args.expected_pairs:
        raise ValueError("smoke pairs are not globally document-diverse")

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "pairs": len(items),
            "mean_chosen_response_tokens": mean(
                [float(item["chosen_response_tokens"]) for item in items]
            ),
            "mean_rejected_response_tokens": mean(
                [float(item["rejected_response_tokens"]) for item in items]
            ),
            "mean_chosen_kept_tokens": mean(
                [float(item["chosen_kept_tokens"]) for item in items]
            ),
            "mean_rejected_kept_tokens": mean(
                [float(item["rejected_kept_tokens"]) for item in items]
            ),
            "mean_chosen_keep_ratio": mean(
                [float(item["chosen_keep_ratio"]) for item in items]
            ),
            "mean_rejected_keep_ratio": mean(
                [float(item["rejected_keep_ratio"]) for item in items]
            ),
        }

    summary = {
        "valid": True,
        "protocol": "E118 difference-masked atomic SimPO smoke",
        "source_preference": str(args.preference_jsonl),
        "source_sha256": source_hash,
        "model_path": args.model_path,
        "context_tokens": args.context_tokens,
        "pairs": len(manifest),
        "unique_windows": len(windows),
        "unique_documents": len(documents),
        "category_counts": dict(sorted(counts.items())),
        "overall": aggregate(manifest),
        "by_category": {
            category: aggregate(items)
            for category, items in sorted(category_rows.items())
        },
        "no_dev_or_test_data_read": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "difference_mask_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in manifest),
        encoding="utf-8",
    )
    manifest_hash = sha256(manifest_path)
    summary["manifest_sha256"] = manifest_hash
    (args.output_dir / "mask_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    frozen = {
        "frozen": True,
        "protocol": summary["protocol"],
        "source_preference_sha256": source_hash,
        "mask_manifest_sha256": manifest_hash,
        "context_tokens": args.context_tokens,
        "objective": "SimPO average log-probability over divergent labels only",
        "gates": {
            "minimum_masked_margin_delta": 0.005,
            "all_five_masked_categories_positive": True,
            "full_response_overall_delta_positive": True,
            "required_full_response_categories_positive": [
                "extra_frame",
                "trigger_drift",
            ],
            "maximum_chosen_full_logp_drop": 0.02,
        },
        "test_data_access": False,
    }
    (args.output_dir / "frozen_artifacts.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
