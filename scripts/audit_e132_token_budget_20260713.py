#!/usr/bin/env python3
"""Audit full chat-template token lengths for frozen E95 and E132 input-only variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lengths(path: Path, tokenizer: Any) -> list[int]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = [
                {"role": "user", "content": row["instruction"] + "\n" + row["input"]},
                {"role": "assistant", "content": row["output"]},
            ]
            values.append(
                len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
            )
    return sorted(values)


def summarize(values: list[int], cutoff: int) -> dict[str, Any]:
    last = len(values) - 1
    return {
        "rows": len(values),
        "min": values[0],
        "p50": values[int(last * 0.50)],
        "p90": values[int(last * 0.90)],
        "p95": values[int(last * 0.95)],
        "p99": values[int(last * 0.99)],
        "max": values[-1],
        "over_cutoff": sum(value > cutoff for value in values),
        "over_2048": sum(value > 2048 for value in values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_jsonl", type=Path, required=True)
    parser.add_argument("--candidate_jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--cutoff", type=int, default=1536)
    parser.add_argument("--max_candidate_over_cutoff", type=int, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    if args.output_json.exists():
        raise SystemExit(f"refusing to overwrite token audit: {args.output_json}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    baseline_values = lengths(args.baseline_jsonl, tokenizer)
    candidate_values = lengths(args.candidate_jsonl, tokenizer)
    baseline = summarize(baseline_values, args.cutoff)
    candidate = summarize(candidate_values, args.cutoff)
    checks = {
        "row_count_exact": baseline["rows"] == candidate["rows"] == 1320,
        "candidate_over_cutoff_bounded": candidate["over_cutoff"]
        <= args.max_candidate_over_cutoff,
        "candidate_zero_over_2048": candidate["over_2048"] == 0,
    }
    result = {
        "id": "e132_compact_v3_token_budget_gate_v1",
        "tokenizer": args.tokenizer,
        "cutoff": args.cutoff,
        "max_candidate_over_cutoff": args.max_candidate_over_cutoff,
        "baseline": baseline,
        "candidate": candidate,
        "input_sha256": {
            "baseline": sha256_file(args.baseline_jsonl),
            "candidate": sha256_file(args.candidate_jsonl),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_pass and not result["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
