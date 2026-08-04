#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROLE_LABEL_CONFLICT = re.compile(
    r"(role.{0,40}(schema.{0,20}(mismatch|conflict)|not in schema|absent from|deviat)"
    r"|schema.{0,30}role.{0,30}(mismatch|invalid|not)"
    r"|non[_ -]?schema[_ -]?role|invalid[_ -]?role"
    r"|Entity.{0,40}(Participant|Communicator)|Participant.{0,40}Entity)",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rejection_text(record: dict) -> str:
    parts = list(record.get("hard_errors") or []) + list(record.get("semantic_errors") or [])
    if record.get("error"):
        parts.append(str(record["error"]))
    return " | ".join(map(str, parts))


def classify_rejection(record: dict) -> str:
    text = rejection_text(record)
    if ROLE_LABEL_CONFLICT.search(text):
        return "role_label_schema_name_conflict"
    if "thinking_too_long" in text:
        return "thinking_word_limit"
    if record.get("error") or record.get("hard_errors"):
        return "other_hard_or_interface_failure"
    return "other_semantic_failure"


def audit(rows: list[dict], raw_sha256: str) -> dict:
    rejected = [row for row in rows if not row.get("accepted")]
    categories = {name: [] for name in (
        "role_label_schema_name_conflict",
        "thinking_word_limit",
        "other_hard_or_interface_failure",
        "other_semantic_failure",
    )}
    for row in rejected:
        categories[classify_rejection(row)].append(row.get("sample_id"))
    counts = Counter({name: len(ids) for name, ids in categories.items()})
    return {
        "protocol": "e124b-frozen-rejection-cause-audit-v1",
        "raw_sha256": raw_sha256,
        "counts": {
            "rows": len(rows),
            "accepted": len(rows) - len(rejected),
            "rejected": len(rejected),
            **dict(counts),
        },
        "checks": {
            "exact_1500_rows": len(rows) == 1500,
            "unique_sample_ids": len({row.get("sample_id") for row in rows}) == len(rows),
            "categories_partition_rejections": sum(counts.values()) == len(rejected),
        },
        "sample_ids_by_category": categories,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--require_raw_sha256")
    parser.add_argument("--require_checks", action="store_true")
    args = parser.parse_args()
    observed_sha = sha256_file(args.raw_jsonl)
    if args.require_raw_sha256 and observed_sha != args.require_raw_sha256:
        raise SystemExit(
            f"raw SHA mismatch: expected {args.require_raw_sha256}, observed {observed_sha}"
        )
    result = audit(load_jsonl(args.raw_jsonl), observed_sha)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 6 if args.require_checks and not all(result["checks"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
