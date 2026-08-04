#!/usr/bin/env python3
"""Independently audit the E77b row-matched surface control."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
SOURCE_BRANCH = "e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot"
CONTROL_BRANCH = "e77b_e81_rowmatched_control"
EXPECTED_COUNTS = {
    "train": 1448,
    "dev_seen": 197,
    "test_seen": 361,
    "test_unseen": 82,
}
SOURCE_RESPONSE_RE = re.compile(
    r"^<thinking>.*?</thinking>\s*(<final>(.*)</final>)$", re.S
)
CONTROL_RESPONSE_RE = re.compile(r"^<final>(.*)</final>$", re.S)
THINKING_SENTENCE_RE = re.compile(
    r"First output `<thinking>[^`]*`[^.]*\. Then output", re.S
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def expected_control_instruction(source: str) -> str:
    result, count = THINKING_SENTENCE_RE.subn("Output", source)
    if count != 1:
        raise ValueError("source instruction does not contain one reasoning directive")
    return result


def audit_split(
    source_path: Path,
    control_path: Path,
    *,
    expected_count: int,
) -> dict[str, Any]:
    source_rows = load_jsonl(source_path)
    control_rows = load_jsonl(control_path)
    problems: list[str] = []
    if len(source_rows) != expected_count:
        problems.append(f"source_count={len(source_rows)} expected={expected_count}")
    if len(control_rows) != expected_count:
        problems.append(f"control_count={len(control_rows)} expected={expected_count}")
    if len(source_rows) != len(control_rows):
        problems.append("source/control row counts differ")

    source_finals: list[str] = []
    control_outputs: list[str] = []
    source_ids: list[str] = []
    control_ids: list[str] = []
    exact_rows = 0
    for index, (source, control) in enumerate(
        zip(source_rows, control_rows, strict=False)
    ):
        row_problems: list[str] = []
        source_match = SOURCE_RESPONSE_RE.fullmatch(str(source.get("output", "")))
        control_match = CONTROL_RESPONSE_RE.fullmatch(str(control.get("output", "")))
        if source_match is None:
            row_problems.append("invalid source response shape")
            source_final = ""
        else:
            source_final = source_match.group(1)
        if control_match is None:
            row_problems.append("invalid control response shape")
        else:
            try:
                json.loads(control_match.group(1))
            except json.JSONDecodeError:
                row_problems.append("invalid control final JSON")

        source_finals.append(source_final)
        control_output = str(control.get("output", ""))
        control_outputs.append(control_output)
        if source_final != control_output:
            row_problems.append("final response is not byte-identical")
        if source.get("input") != control.get("input"):
            row_problems.append("input differs")
        if source.get("gold_output") != control.get("gold_output"):
            row_problems.append("gold_output differs")
        try:
            expected_instruction = expected_control_instruction(
                str(source.get("instruction", ""))
            )
        except ValueError as error:
            row_problems.append(str(error))
            expected_instruction = ""
        if control.get("instruction") != expected_instruction:
            row_problems.append("instruction differs beyond reasoning removal")
        if "<thinking>" in str(control.get("instruction", "")):
            row_problems.append("thinking tag remains in control instruction")
        if "<thinking>" in control_output:
            row_problems.append("thinking tag remains in control response")

        source_meta = source.get("meta") or {}
        control_meta = control.get("meta") or {}
        source_id = str(source_meta.get("wnd_id", ""))
        control_id = str(control_meta.get("wnd_id", ""))
        source_ids.append(source_id)
        control_ids.append(control_id)
        for key in ("doc_id", "wnd_id", "e40_source_index"):
            if source_meta.get(key) != control_meta.get(key):
                row_problems.append(f"meta.{key} differs")
        if control_meta.get("control_changed_variable") != "remove_thinking_keep_final":
            row_problems.append("missing control provenance marker")

        if row_problems:
            if len(problems) < 20:
                problems.extend(f"row={index}: {problem}" for problem in row_problems)
        else:
            exact_rows += 1

    if len(set(source_ids)) != len(source_ids):
        problems.append("duplicate source wnd_id")
    if source_ids != control_ids:
        problems.append("wnd_id order differs")

    return {
        "source_path": str(source_path),
        "control_path": str(control_path),
        "source_sha256": sha256(source_path),
        "control_sha256": sha256(control_path),
        "expected_count": expected_count,
        "source_count": len(source_rows),
        "control_count": len(control_rows),
        "exact_rows": exact_rows,
        "source_final_sequence_sha256": sequence_sha256(source_finals),
        "control_output_sequence_sha256": sequence_sha256(control_outputs),
        "passed": not problems and exact_rows == expected_count,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--source_branch", default=SOURCE_BRANCH)
    parser.add_argument("--control_branch", default=CONTROL_BRANCH)
    args = parser.parse_args()

    split_reports = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        source_path = (
            args.dataset_dir
            / f"{DATA_PREFIX}_{args.source_branch}_{split}_pos.jsonl"
        )
        control_path = (
            args.dataset_dir
            / f"{DATA_PREFIX}_{args.control_branch}_{split}_pos.jsonl"
        )
        split_reports[split] = audit_split(
            source_path, control_path, expected_count=expected_count
        )

    passed = all(report["passed"] for report in split_reports.values())
    result = {
        "protocol": "E77b independent row-matched surface-control audit",
        "source_branch": args.source_branch,
        "control_branch": args.control_branch,
        "changed_variable": "remove thinking directive/response; preserve row and final",
        "passed": passed,
        "total_expected_rows": sum(EXPECTED_COUNTS.values()),
        "total_exact_rows": sum(
            report["exact_rows"] for report in split_reports.values()
        ),
        "splits": split_reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
