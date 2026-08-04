#!/usr/bin/env python3
"""Audit full chat-template conversation lengths before SFT registration/use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.normalize_strict_sgcot_dataset_20260712 import conversation_token_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cutoff_len", type=int, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--require_all_fit", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    over_limit = []
    maximum = 0
    rows = 0
    with args.input_jsonl.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            total = conversation_token_count(tokenizer, row)
            maximum = max(maximum, total)
            rows += 1
            if total > args.cutoff_len:
                over_limit.append(
                    {
                        "row_index": index,
                        "wnd_id": row.get("meta", {}).get("wnd_id"),
                        "total_tokens": total,
                    }
                )
    report = {
        "input_jsonl": str(args.input_jsonl.resolve()),
        "rows": rows,
        "cutoff_len": args.cutoff_len,
        "maximum_conversation_tokens": maximum,
        "over_limit_rows": len(over_limit),
        "over_limit_examples": over_limit[:50],
        "all_rows_fit": not over_limit,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 4 if args.require_all_fit and over_limit else 0


if __name__ == "__main__":
    raise SystemExit(main())
