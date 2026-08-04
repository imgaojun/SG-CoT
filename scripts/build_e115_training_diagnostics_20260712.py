#!/usr/bin/env python3
"""Build the frozen, training-only E115 diagnostic subsets from E114A."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.mine_reasoning_preferences_e110_20260711 import (  # noqa: E402
    load_jsonl,
    load_sample_records,
    register_dataset,
    write_jsonl,
)
from src.stage2_preference.atomic_counterfactual import (  # noqa: E402
    ATOMIC_CATEGORIES,
    select_quota_assignment,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, *parts: object) -> str:
    payload = "\0".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def document_id(wnd_id: str) -> str:
    head, separator, tail = wnd_id.rpartition("-")
    return head if separator and tail else wnd_id


def select_doc_diverse_smoke(
    rows: list[dict[str, Any]], per_category: int, seed: int
) -> list[dict[str, Any]]:
    """Select balanced rows with a globally unique source document."""

    by_doc_category: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        meta = row.get("meta", {})
        wnd_id = str(meta["wnd_id"])
        category = str(meta["error_category"])
        if category not in ATOMIC_CATEGORIES:
            raise ValueError(f"unsupported E115 category: {category}")
        by_doc_category[document_id(wnd_id)][category].append(row)

    options_by_document: dict[str, dict[str, dict[str, Any]]] = {}
    for doc_id, category_rows in by_doc_category.items():
        options = {}
        for category, candidates in category_rows.items():
            selected = min(
                candidates,
                key=lambda row: stable_key(
                    seed, "smoke-row", category, row["meta"]["wnd_id"]
                ),
            )
            options[category] = {
                "row": selected,
                "proposal_source": "e115_training_diagnostic",
                "frequency": 1,
            }
        options_by_document[doc_id] = options

    quotas = {category: per_category for category in ATOMIC_CATEGORIES}
    assignment = select_quota_assignment(options_by_document, quotas, seed)
    selected_rows = [item["option"]["row"] for item in assignment]
    selected_rows.sort(
        key=lambda row: stable_key(seed, "smoke-order", row["meta"]["wnd_id"])
    )
    return selected_rows


def sample_lookup(
    samples_by_window: dict[str, list[dict[str, Any]]]
) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    lookup = {}
    for wnd_id, samples in samples_by_window.items():
        for sample in samples:
            key = (
                wnd_id,
                int(sample.get("sample_seed", 0)),
                int(sample.get("sample_round", 0)),
                int(sample.get("sample_index", 0)),
            )
            if key in lookup:
                raise ValueError(f"duplicate raw sample key: {key}")
            lookup[key] = sample
    return lookup


def matched_style_rows(
    preference_rows: list[dict[str, Any]],
    samples_by_window: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    lookup = sample_lookup(samples_by_window)
    matched = []
    for row in preference_rows:
        meta = row["meta"]
        eligibility = meta.get("eligibility_exact_sample")
        if not isinstance(eligibility, dict):
            raise ValueError(f"missing E114 eligibility record: {meta.get('wnd_id')}")
        key = (
            str(meta["wnd_id"]),
            int(eligibility["sample_seed"]),
            int(eligibility["sample_round"]),
            int(eligibility["sample_index"]),
        )
        sample = lookup.get(key)
        if sample is None:
            raise ValueError(f"eligibility raw sample is unavailable: {key}")
        native = sample.get("raw_response")
        if not isinstance(native, str) or not native:
            raise ValueError(f"eligibility raw response is empty: {key}")
        matched.append(
            {
                "instruction": row["instruction"],
                "input": row["input"],
                "canonical": row["chosen"],
                "native": native,
                "meta": {
                    "wnd_id": meta["wnd_id"],
                    "document_id": document_id(str(meta["wnd_id"])),
                    "error_category": meta["error_category"],
                    "eligibility_exact_sample": eligibility,
                    "canonical_renderer": meta.get("renderer_version"),
                },
            }
        )
    return matched


def select_balanced_style_diagnostic(
    rows: list[dict[str, Any]], per_category: int, seed: int
) -> list[dict[str, Any]]:
    """Select one window per (category, document), balanced across categories."""

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        meta = row["meta"]
        grouped[str(meta["error_category"])][str(meta["document_id"])].append(row)

    selected = []
    for category in ATOMIC_CATEGORIES:
        document_rows = []
        for doc_id, candidates in grouped[category].items():
            document_rows.append(
                min(
                    candidates,
                    key=lambda row: stable_key(
                        seed, "style-row", category, doc_id, row["meta"]["wnd_id"]
                    ),
                )
            )
        document_rows.sort(
            key=lambda row: stable_key(
                seed,
                "style-document",
                category,
                row["meta"]["document_id"],
            )
        )
        if len(document_rows) < per_category:
            raise ValueError(
                f"insufficient style documents for {category}: "
                f"{len(document_rows)} < {per_category}"
            )
        selected.extend(document_rows[:per_category])
    selected.sort(
        key=lambda row: stable_key(seed, "style-order", row["meta"]["wnd_id"])
    )
    return selected


def validate_selection(
    smoke_rows: list[dict[str, Any]],
    style_rows: list[dict[str, Any]],
    smoke_per_category: int,
    style_per_category: int,
) -> dict[str, Any]:
    smoke_counts = Counter(row["meta"]["error_category"] for row in smoke_rows)
    style_counts = Counter(row["meta"]["error_category"] for row in style_rows)
    expected_smoke = Counter(
        {category: smoke_per_category for category in ATOMIC_CATEGORIES}
    )
    expected_style = Counter(
        {category: style_per_category for category in ATOMIC_CATEGORIES}
    )
    smoke_windows = [str(row["meta"]["wnd_id"]) for row in smoke_rows]
    smoke_documents = [document_id(wnd_id) for wnd_id in smoke_windows]
    style_windows = [str(row["meta"]["wnd_id"]) for row in style_rows]
    if smoke_counts != expected_smoke:
        raise ValueError(f"smoke quota mismatch: {smoke_counts} != {expected_smoke}")
    if style_counts != expected_style:
        raise ValueError(f"style quota mismatch: {style_counts} != {expected_style}")
    if len(set(smoke_windows)) != len(smoke_windows):
        raise ValueError("smoke selection contains duplicate windows")
    if len(set(smoke_documents)) != len(smoke_documents):
        raise ValueError("smoke selection contains duplicate documents")
    if len(set(style_windows)) != len(style_windows):
        raise ValueError("style selection contains duplicate windows")
    for category in ATOMIC_CATEGORIES:
        docs = [
            row["meta"]["document_id"]
            for row in style_rows
            if row["meta"]["error_category"] == category
        ]
        if len(set(docs)) != len(docs):
            raise ValueError(f"style selection repeats a document within {category}")
    return {
        "smoke_category_counts": dict(sorted(smoke_counts.items())),
        "smoke_unique_windows": len(set(smoke_windows)),
        "smoke_unique_documents": len(set(smoke_documents)),
        "style_category_counts": dict(sorted(style_counts.items())),
        "style_unique_windows": len(set(style_windows)),
        "style_unique_documents": len(
            {row["meta"]["document_id"] for row in style_rows}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference_jsonl", type=Path, required=True)
    parser.add_argument("--expected_preference_sha256", required=True)
    parser.add_argument("--samples_glob", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--dataset_info", type=Path, required=True)
    parser.add_argument("--smoke_name", required=True)
    parser.add_argument("--smoke_per_category", type=int, default=8)
    parser.add_argument("--style_per_category", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1150)
    args = parser.parse_args()

    actual_source_hash = sha256_file(args.preference_jsonl)
    if actual_source_hash != args.expected_preference_sha256:
        raise ValueError(
            "frozen E114 preference hash mismatch: "
            f"{actual_source_hash} != {args.expected_preference_sha256}"
        )
    freeze_path = args.output_dir / "frozen_artifacts.json"
    if freeze_path.exists():
        raise FileExistsError(
            f"E115A is already frozen; refusing to overwrite: {freeze_path}"
        )

    preference_rows = load_jsonl(args.preference_jsonl)
    if len(preference_rows) != 900:
        raise ValueError(f"expected 900 frozen E114 pairs, found {len(preference_rows)}")
    smoke_rows = select_doc_diverse_smoke(
        preference_rows, args.smoke_per_category, args.seed
    )
    all_matched = matched_style_rows(
        preference_rows, load_sample_records(args.samples_glob, None)
    )
    style_rows = select_balanced_style_diagnostic(
        all_matched, args.style_per_category, args.seed
    )
    validation = validate_selection(
        smoke_rows,
        style_rows,
        args.smoke_per_category,
        args.style_per_category,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = args.dataset_dir / f"{args.smoke_name}.jsonl"
    style_path = args.output_dir / "canonical_native_matched200.jsonl"
    manifest_path = args.output_dir / "selection_manifest.jsonl"
    summary_path = args.output_dir / "build_summary.json"
    write_jsonl(smoke_path, smoke_rows)
    write_jsonl(style_path, style_rows)
    manifest = [
        {
            "purpose": "smoke40",
            "wnd_id": row["meta"]["wnd_id"],
            "document_id": document_id(str(row["meta"]["wnd_id"])),
            "error_category": row["meta"]["error_category"],
        }
        for row in smoke_rows
    ] + [
        {
            "purpose": "style200",
            "wnd_id": row["meta"]["wnd_id"],
            "document_id": row["meta"]["document_id"],
            "error_category": row["meta"]["error_category"],
            "eligibility_exact_sample": row["meta"]["eligibility_exact_sample"],
        }
        for row in style_rows
    ]
    write_jsonl(manifest_path, manifest)
    register_dataset(args.dataset_info, args.smoke_name, smoke_path.name, ranking=True)

    hashes = {
        "source_preference": actual_source_hash,
        "smoke40": sha256_file(smoke_path),
        "style200": sha256_file(style_path),
        "selection_manifest": sha256_file(manifest_path),
    }
    summary = {
        "valid": True,
        "protocol": "E115 training-only corrected smoke",
        "selection_seed": args.seed,
        "source_pairs": len(preference_rows),
        "smoke_pairs": len(smoke_rows),
        "style_pairs": len(style_rows),
        **validation,
        "no_test_data_read": True,
        "paths": {
            "smoke40": str(smoke_path),
            "style200": str(style_path),
            "selection_manifest": str(manifest_path),
        },
        "sha256": hashes,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    freeze = {
        "frozen": True,
        "protocol": summary["protocol"],
        "selection_seed": args.seed,
        "selection_rules": {
            "smoke": "8 per category; globally unique document; stable hash only",
            "style": "40 per category; unique document within category; stable hash only",
        },
        "gates": {
            "overall_mean_margin_delta_gt": 0.0,
            "minimum_improved_categories": 4,
            "required_improved_categories": ["extra_frame", "trigger_drift"],
            "maximum_canonical_minus_native_nll": 0.3,
        },
        "artifacts": {
            name: {"path": summary["paths"].get(name), "sha256": value}
            for name, value in hashes.items()
            if name != "source_preference"
        },
        "source_preference": {
            "path": str(args.preference_jsonl),
            "sha256": actual_source_hash,
        },
    }
    freeze_path.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
