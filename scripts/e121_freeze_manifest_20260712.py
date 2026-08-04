#!/usr/bin/env python3
"""Build or verify the immutable input manifest required by E121 test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(root: Path, paths: list[str], globs: list[str]) -> list[Path]:
    selected: set[Path] = set()
    for value in paths:
        path = (root / value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.add(path)
    for pattern in globs:
        matches = [path.resolve() for path in root.glob(pattern) if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"freeze glob matched no files: {pattern}")
        selected.update(matches)
    root_resolved = root.resolve()
    for path in selected:
        if not path.is_relative_to(root_resolved):
            raise ValueError(f"freeze path is outside repository root: {path}")
    return sorted(selected)


def record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def manifest_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build(args: argparse.Namespace) -> int:
    files = collect(args.root, args.path, args.glob)
    records = [record(args.root, path) for path in files]
    report = {
        "id": getattr(args, "manifest_id", "e121_frozen_inputs_v1"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()),
        "file_count": len(records),
        "files": records,
        "records_sha256": manifest_digest(records),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def verify(args: argparse.Namespace) -> int:
    report = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    current_records = []
    for expected in report.get("files", []):
        path = (args.root / expected["path"]).resolve()
        if not path.is_file():
            failures.append({"path": expected["path"], "error": "missing"})
            continue
        current = record(args.root, path)
        current_records.append(current)
        if current != expected:
            failures.append(
                {
                    "path": expected["path"],
                    "error": "content_changed",
                    "expected_size": expected["size"],
                    "observed_size": current["size"],
                    "expected_sha256": expected["sha256"],
                    "observed_sha256": current["sha256"],
                }
            )
    observed_digest = manifest_digest(current_records) if not failures else None
    if observed_digest is not None and observed_digest != report.get("records_sha256"):
        failures.append({"error": "records_digest_changed"})
    result = {
        "manifest": str(args.manifest.resolve()),
        "verified": not failures,
        "file_count": len(report.get("files", [])),
        "records_sha256": report.get("records_sha256"),
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verified"] else 6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "verify"], required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest_id", default="e121_frozen_inputs_v1")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--glob", action="append", default=[])
    args = parser.parse_args()
    if args.mode == "build" and not (args.path or args.glob):
        parser.error("build mode requires at least one --path or --glob")
    return build(args) if args.mode == "build" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
