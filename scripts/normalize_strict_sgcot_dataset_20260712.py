#!/usr/bin/env python3
"""Replace generated SG-CoT finals with the shared deterministic surface protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.build_surface_evidence_dataset_20260712 import (  # noqa: E402
    SGCOT_INSTRUCTION,
    register_dataset,
)
from src.stage2_preference.reasoning_preference import (  # noqa: E402
    extract_final_json,
    extract_tag,
    find_heldout_leaks,
    is_exact,
    recover_offsets_from_evidence,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def conversation_token_count(tokenizer: Any, row: dict[str, Any]) -> int:
    user_content = f"{row['instruction']}\n{row['input']}"
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["output"]},
    ]
    if getattr(tokenizer, "chat_template", None):
        return len(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        )
    return len(tokenizer(user_content + "\n" + row["output"], add_special_tokens=True)["input_ids"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_jsonl", type=Path, required=True)
    parser.add_argument("--reference_surface_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_info", type=Path)
    parser.add_argument("--heldout_types_json", type=Path)
    parser.add_argument("--require_zero_leaks", action="store_true")
    parser.add_argument("--min_rows", type=int, default=0)
    parser.add_argument("--model_path")
    parser.add_argument("--cutoff_len", type=int)
    args = parser.parse_args()

    if (args.model_path is None) != (args.cutoff_len is None):
        parser.error("--model_path and --cutoff_len must be provided together")
    tokenizer = None
    if args.model_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    generated = load_jsonl(args.generated_jsonl)
    references = {
        row.get("meta", {}).get("wnd_id"): row for row in load_jsonl(args.reference_surface_jsonl)
    }
    if None in references:
        raise ValueError("reference rows require meta.wnd_id")
    heldout_types = (
        json.loads(args.heldout_types_json.read_text(encoding="utf-8"))
        if args.heldout_types_json
        else []
    )
    output = []
    leak_records = []
    length_filtered = []
    maximum_conversation_tokens = 0
    for index, row in enumerate(generated):
        wnd_id = row.get("meta", {}).get("wnd_id")
        reference = references.get(wnd_id)
        if reference is None:
            raise ValueError(f"missing reference surface row for wnd_id={wnd_id}")
        thinking = extract_tag(row.get("output", ""), "thinking")
        reference_final = extract_final_json(reference.get("output", ""))
        if not thinking or reference_final is None:
            raise ValueError(f"missing generated thinking or reference final for wnd_id={wnd_id}")
        normalized = json.loads(json.dumps(row, ensure_ascii=False))
        normalized["instruction"] = SGCOT_INSTRUCTION
        normalized["output"] = (
            f"<thinking>{thinking}</thinking>\n"
            f"<final>{json.dumps(reference_final, ensure_ascii=False, separators=(',', ':'))}</final>"
        )
        normalized["gold_output"] = reference["gold_output"]
        normalized.setdefault("meta", {}).update(
            {
                "surface_protocol": "shortest_unique_evidence_v1",
                "surface_mode": "sgcot",
                "strict_final_normalized": True,
                "offset_recovery_verified": True,
            }
        )
        if tokenizer is not None:
            total_tokens = conversation_token_count(tokenizer, normalized)
            maximum_conversation_tokens = max(maximum_conversation_tokens, total_tokens)
            if total_tokens > args.cutoff_len:
                length_filtered.append(
                    {"row_index": index, "wnd_id": wnd_id, "total_tokens": total_tokens}
                )
                continue
        recovered, diagnostics = recover_offsets_from_evidence(reference_final, normalized["input"])
        gold = json.loads(normalized["gold_output"])
        if diagnostics["missing_offsets"] or not is_exact(recovered, gold):
            raise ValueError(f"normalized final does not recover gold for wnd_id={wnd_id}")
        for leak in find_heldout_leaks(normalized, heldout_types):
            leak_records.append({"row_index": index, "wnd_id": wnd_id, **leak})
        output.append(normalized)

    summary = {
        "generated_rows": len(generated),
        "normalized_rows": len(output),
        "length_filtered_rows": len(length_filtered),
        "length_filtered_examples": length_filtered[:50],
        "maximum_conversation_tokens_before_filter": maximum_conversation_tokens,
        "cutoff_len": args.cutoff_len,
        "heldout_leaks": len(leak_records),
        "zero_heldout_leaks": not leak_records,
        "minimum_rows_required": args.min_rows,
        "meets_minimum_rows": len(output) >= args.min_rows,
        "all_finals_recover_gold": True,
        "surface_protocol": "shortest_unique_evidence_v1",
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_zero_leaks and leak_records:
        return 5
    if len(output) < args.min_rows:
        return 6
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.dataset_name:
        if args.dataset_info is None:
            parser.error("--dataset_info is required with --dataset_name")
        register_dataset(args.dataset_info, args.dataset_name, args.output_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
