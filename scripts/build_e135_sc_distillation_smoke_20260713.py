#!/usr/bin/env python3
"""Build E135's deterministic 64-row train-only SC-distillation smoke manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} is not an object")
            rows.append(row)
    return rows


def selection_key(seed: int, source_index: int, wnd_id: str) -> str:
    payload = f"{seed}\0{source_index}\0{wnd_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def experiment_prefix(protocol: dict[str, Any]) -> str:
    prefix = protocol.get("request_prefix", "e135")
    if not isinstance(prefix, str) or re.fullmatch(r"e\d+", prefix) is None:
        raise ValueError(f"invalid experiment prefix: {prefix!r}")
    if not str(protocol.get("id", "")).startswith(prefix + "_"):
        raise ValueError("protocol id/request prefix mismatch")
    return prefix


def select_rows(
    rows: list[dict[str, Any]],
    seed: int,
    count: int,
    excluded_wnd_ids: set[str] | None = None,
    prefix: str = "e135",
) -> list[dict[str, Any]]:
    ranked = []
    seen_wnd_ids = set()
    excluded_wnd_ids = excluded_wnd_ids or set()
    for source_index, row in enumerate(rows):
        meta = row.get("meta") or {}
        wnd_id = str(meta.get("wnd_id") or "")
        if not wnd_id:
            raise ValueError(f"missing wnd_id at source index {source_index}")
        if wnd_id in seen_wnd_ids:
            raise ValueError(f"duplicate source wnd_id: {wnd_id}")
        seen_wnd_ids.add(wnd_id)
        if meta.get("source_part") != "train":
            raise ValueError(f"non-train row at source index {source_index}: {wnd_id}")
        if wnd_id in excluded_wnd_ids:
            continue
        ranked.append((selection_key(seed, source_index, wnd_id), source_index, row))
    if count > len(ranked):
        raise ValueError(f"cannot select {count} rows from {len(ranked)}")

    selected = []
    for selection_rank, (key, source_index, source) in enumerate(sorted(ranked)[:count]):
        row = json.loads(json.dumps(source, ensure_ascii=False))
        row["meta"][f"{prefix}_source_index"] = source_index
        row["meta"][f"{prefix}_selection_rank"] = selection_rank
        row["meta"][f"{prefix}_selection_key"] = key
        selected.append(row)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    prefix = experiment_prefix(protocol)
    source = REPO_ROOT / protocol["source_train_jsonl"]
    if sha256_file(source) != protocol["source_train_sha256"]:
        raise ValueError("source training hash mismatch")
    rows = load_jsonl(source)
    if len(rows) != int(protocol["source_train_rows"]):
        raise ValueError("source row count mismatch")

    outputs = {
        "rows": args.output_dir / "selected_rows.jsonl",
        "audit": args.output_dir / "selection_audit.json",
    }
    if args.output_dir.exists() or any(path.exists() for path in outputs.values()):
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    selection = protocol["selection"]
    exclusion = protocol.get("exclusion")
    exclusions = protocol.get("exclusions")
    if exclusion and exclusions:
        raise ValueError("protocol cannot define both exclusion and exclusions")
    if exclusions is not None:
        if not isinstance(exclusions, list) or not exclusions:
            raise ValueError("exclusions must be a non-empty list")
        exclusion_specs = exclusions
    else:
        exclusion_specs = [exclusion] if exclusion else []
    excluded_wnd_ids: set[str] = set()
    exclusion_rows_total = 0
    exclusion_audit = []
    for exclusion_spec in exclusion_specs:
        exclusion_path = REPO_ROOT / exclusion_spec["manifest_jsonl"]
        if sha256_file(exclusion_path) != exclusion_spec["manifest_sha256"]:
            raise ValueError("exclusion manifest hash mismatch")
        excluded_rows = load_jsonl(exclusion_path)
        manifest_wnd_ids = {
            str(row.get("meta", {}).get("wnd_id") or "") for row in excluded_rows
        }
        if (
            len(excluded_rows) != int(exclusion_spec["rows"])
            or len(manifest_wnd_ids) != len(excluded_rows)
            or "" in manifest_wnd_ids
        ):
            raise ValueError("exclusion manifest row/identity mismatch")
        overlap = len(excluded_wnd_ids & manifest_wnd_ids)
        exclusion_rows_total += len(excluded_rows)
        excluded_wnd_ids.update(manifest_wnd_ids)
        exclusion_audit.append(
            {
                "manifest_jsonl": exclusion_spec["manifest_jsonl"],
                "manifest_sha256": exclusion_spec["manifest_sha256"],
                "rows": len(excluded_rows),
                "overlap_with_prior_manifests": overlap,
            }
        )
    exclusion_cross_manifest_overlap = exclusion_rows_total - len(excluded_wnd_ids)
    exclusion_union = protocol.get("exclusion_union", {})
    if exclusion_specs:
        required_union_rows = int(
            exclusion_union.get("required_rows", len(excluded_wnd_ids))
        )
        required_cross_overlap = int(
            exclusion_union.get(
                "required_cross_manifest_overlap", exclusion_cross_manifest_overlap
            )
        )
        if len(excluded_wnd_ids) != required_union_rows:
            raise ValueError("exclusion union row count mismatch")
        if exclusion_cross_manifest_overlap != required_cross_overlap:
            raise ValueError("exclusion cross-manifest overlap mismatch")
    selected = select_rows(
        rows,
        int(selection["seed"]),
        int(selection["rows"]),
        excluded_wnd_ids,
        prefix,
    )

    args.output_dir.mkdir(parents=True)
    with outputs["rows"].open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    wnd_ids = [row["meta"]["wnd_id"] for row in selected]
    audit = {
        "id": protocol.get("report_ids", {}).get(
            "selection", "e135_sc_distillation_smoke_selection_v1"
        ),
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "selection_seed": int(selection["seed"]),
        "selection_method": selection["method"],
        "unique_wnd_ids": len(set(wnd_ids)),
        "source_parts": sorted({row["meta"]["source_part"] for row in selected}),
        "test_rows_read": 0,
        "source_sha256": protocol["source_train_sha256"],
        "output_sha256": sha256_file(outputs["rows"]),
        "passed": (
            len(selected) == int(selection["rows"])
            and len(set(wnd_ids)) == len(selected)
            and all(row["meta"]["source_part"] == "train" for row in selected)
        ),
    }
    if exclusion_specs:
        audit.update(
            {
                "excluded_rows": len(excluded_wnd_ids),
                "exclusion_manifest_rows": exclusion_rows_total,
                "exclusion_cross_manifest_overlap": exclusion_cross_manifest_overlap,
                "exclusion_manifests": exclusion_audit,
                "selected_exclusion_overlap": len(set(wnd_ids) & excluded_wnd_ids),
            }
        )
        if exclusion:
            audit["exclusion_sha256"] = exclusion["manifest_sha256"]
        audit["passed"] = audit["passed"] and not audit["selected_exclusion_overlap"]
    outputs["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
