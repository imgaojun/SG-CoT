#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wnd_id(row: dict) -> str:
    value = (row.get("meta") or {}).get("wnd_id")
    if not isinstance(value, str) or not value:
        raise ValueError("row is missing meta.wnd_id")
    return value


def stable_key(value: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()
    return digest, value


def select_disjoint_rows(
    source_rows: list[dict], excluded_ids: set[str], count: int, seed: int
) -> list[dict]:
    source_ids = [wnd_id(row) for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source rows contain duplicate wnd_id values")
    eligible = [row for row in source_rows if wnd_id(row) not in excluded_ids]
    selected = sorted(eligible, key=lambda row: stable_key(wnd_id(row), seed))[:count]
    if len(selected) != count:
        raise ValueError(f"needed {count} rows but only found {len(selected)}")
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_jsonl", type=Path, required=True)
    parser.add_argument("--exclude_preflight_json", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--audit_json", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1240)
    args = parser.parse_args()

    source_rows = load_jsonl(args.source_jsonl)
    exclude_report = json.loads(args.exclude_preflight_json.read_text())
    excluded_ids = set(exclude_report.get("selected_wnd_ids") or [])
    if len(excluded_ids) != int(exclude_report.get("selected_count", -1)):
        raise ValueError("exclude preflight count and IDs do not match")
    selected = select_disjoint_rows(source_rows, excluded_ids, args.count, args.seed)
    selected_ids = [wnd_id(row) for row in selected]
    overlap = sorted(set(selected_ids) & excluded_ids)
    if overlap:
        raise ValueError(f"selected rows overlap excluded IDs: {overlap[:3]}")
    write_jsonl(args.output_jsonl, selected)

    digest = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    audit = {
        "protocol": "e124a-disjoint-stable-hash-trainonly-v1",
        "source_jsonl": str(args.source_jsonl),
        "exclude_preflight_json": str(args.exclude_preflight_json),
        "source_rows": len(source_rows),
        "excluded_rows": len(excluded_ids),
        "selected_rows": len(selected),
        "seed": args.seed,
        "selection": "ascending sha256(seed\\0wnd_id), then wnd_id",
        "selected_wnd_ids_sha256": digest,
        "selected_wnd_ids": selected_ids,
        "overlap_with_e123": overlap,
        "checks": {
            "selected_count_exact": len(selected) == args.count,
            "unique_selected": len(set(selected_ids)) == args.count,
            "zero_e123_overlap": not overlap,
        },
    }
    audit["passed"] = all(audit["checks"].values())
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
