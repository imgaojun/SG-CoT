#!/usr/bin/env python3
"""Audit E137 chat-template lengths by oracle/retrieved candidate regime."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: list[int], cutoff: int) -> dict[str, Any]:
    values = sorted(values)
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    if args.output_json.exists():
        raise SystemExit(f"refusing to overwrite token audit: {args.output_json}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cutoff = int(protocol["training"]["cutoff_len"])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    grouped: dict[str, list[int]] = defaultdict(list)
    with args.train_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            family = row.get("meta", {}).get("e137_candidate_regime")
            if family not in {"oracle", "retrieved"}:
                raise SystemExit(f"invalid E137 candidate regime: {family}")
            messages = [
                {"role": "user", "content": row["instruction"] + "\n" + row["input"]},
                {"role": "assistant", "content": row["output"]},
            ]
            grouped[family].append(
                len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
            )
    grouped["total"] = grouped["oracle"] + grouped["retrieved"]
    summaries = {name: summarize(values, cutoff) for name, values in grouped.items()}
    gate = protocol["token_gate"]
    expected = protocol["expected"]
    checks = {
        "oracle_rows_exact": summaries["oracle"]["rows"] == expected["oracle_train_rows"],
        "retrieved_rows_exact": summaries["retrieved"]["rows"]
        == expected["retrieved_train_rows"],
        "total_rows_exact": summaries["total"]["rows"] == expected["mixed_train_rows"],
        "total_over_cutoff_bounded": summaries["total"]["over_cutoff"]
        <= gate["maximum_total_over_1536"],
        "retrieved_over_cutoff_bounded": summaries["retrieved"]["over_cutoff"]
        <= gate["maximum_retrieved_over_1536"],
        "zero_over_2048": summaries["total"]["over_2048"] <= gate["maximum_over_2048"],
    }
    result = {
        "id": "e137_mixed_oracle_retrieved_token_gate_v1",
        "protocol_sha256": sha256_file(args.protocol),
        "train_sha256": sha256_file(args.train_jsonl),
        "tokenizer": args.tokenizer,
        "cutoff": cutoff,
        "summaries": summaries,
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
