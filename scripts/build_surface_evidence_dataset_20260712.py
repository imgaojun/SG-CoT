#!/usr/bin/env python3
"""Convert numeric-offset EE targets to deterministic surface/evidence targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_preference.reasoning_preference import (  # noqa: E402
    find_compact_spans,
    find_heldout_leaks,
    is_exact,
    parse_prompt_tokens,
    recover_offsets_from_evidence,
)


DIRECT_INSTRUCTION = (
    "You are doing event extraction. Use only the provided candidate event types and schema cards. "
    "Output exactly `<final>{...}</final>` with a surface-only JSON event list: each trigger and argument "
    "must include `text` and the shortest unique contiguous local `evidence` quote that identifies its "
    "token span. Do not output `<thinking>`, numeric offsets, token indices, or text outside the lowercase "
    "`<final>` tag."
)

SGCOT_INSTRUCTION = (
    "You are doing event extraction. Use only the provided candidate event types and schema cards. First "
    "output `<thinking>...</thinking>` with schema-grounded reasoning: audit plausible frames, lock minimal "
    "trigger anchors, contrast only candidate types using their definitions, cues, and roles, then attach "
    "locally supported arguments. Do not use any type or cluster not shown in the candidates. Then output "
    "`<final>{...}</final>` with the same surface-only JSON protocol: every trigger and argument must include "
    "`text` and the shortest unique contiguous local `evidence` quote that identifies its token span. Do not "
    "output numeric offsets, token indices, or text outside the lowercase tags."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def shortest_unique_evidence(
    tokens: list[str], start: int, end: int, surface_text: str | None = None
) -> str:
    if not 0 <= start < end <= len(tokens):
        raise ValueError(f"invalid token span [{start}, {end}) for {len(tokens)} tokens")
    surface_text = surface_text or " ".join(tokens[start:end])
    for extra in range(0, len(tokens) + 1):
        candidates = []
        for left_extra in range(extra + 1):
            right_extra = extra - left_extra
            left = start - left_extra
            right = end + right_extra
            if left < 0 or right > len(tokens):
                continue
            evidence = " ".join(tokens[left:right])
            spans = find_compact_spans(tokens, evidence)
            target_spans = [
                span
                for span in find_compact_spans(tokens, surface_text)
                if span[0] >= left and span[1] <= right
            ]
            if (
                len(spans) == 1
                and spans[0][0] <= start
                and spans[0][1] >= end
                and target_spans == [(start, end)]
            ):
                candidates.append((abs(left_extra - right_extra), left, right, evidence))
        if candidates:
            return min(candidates)[3]
    raise ValueError(f"could not construct unique evidence for token span [{start}, {end})")


def offset_aligned_surface_text(
    tokens: list[str], start: int, end: int, annotated_text: str | None
) -> str:
    """Keep the annotation text only when it resolves to the annotated token span."""
    if annotated_text and (start, end) in find_compact_spans(tokens, annotated_text):
        return annotated_text
    return " ".join(tokens[start:end])


def surface_payload(gold: dict[str, Any], tokens: list[str]) -> dict[str, Any]:
    events = []
    for event in gold.get("events", []):
        trigger = event["trigger"]
        trigger_start = int(trigger["start"])
        trigger_end = int(trigger["end"])
        arguments = []
        for argument in event.get("arguments", []):
            argument_start = int(argument["start"])
            argument_end = int(argument["end"])
            argument_text = offset_aligned_surface_text(
                tokens,
                argument_start,
                argument_end,
                argument.get("text"),
            )
            arguments.append(
                {
                    "role": argument.get("role"),
                    "text": argument_text,
                    "evidence": shortest_unique_evidence(
                        tokens, argument_start, argument_end, argument_text
                    ),
                }
            )
        trigger_text = offset_aligned_surface_text(
            tokens, trigger_start, trigger_end, trigger.get("text")
        )
        events.append(
            {
                "event_type": event.get("event_type"),
                "trigger": {
                    "text": trigger_text,
                    "evidence": shortest_unique_evidence(
                        tokens, trigger_start, trigger_end, trigger_text
                    ),
                },
                "arguments": arguments,
            }
        )
    return {"events": events}


def convert_row(row: dict[str, Any], mode: str) -> dict[str, Any]:
    gold_raw = row.get("gold_output", row["output"])
    gold = json.loads(gold_raw) if isinstance(gold_raw, str) else gold_raw
    tokens = parse_prompt_tokens(row["input"])
    surface = surface_payload(gold, tokens)
    recovered, diagnostics = recover_offsets_from_evidence(surface, row["input"])
    if diagnostics["missing_offsets"] or not is_exact(recovered, gold):
        raise ValueError(
            f"surface recovery mismatch for wnd_id={row.get('meta', {}).get('wnd_id')}: {diagnostics}"
        )
    converted = json.loads(json.dumps(row, ensure_ascii=False))
    converted["instruction"] = DIRECT_INSTRUCTION if mode == "direct" else SGCOT_INSTRUCTION
    converted["output"] = f"<final>{compact_json(surface)}</final>"
    converted["gold_output"] = compact_json(gold)
    converted.setdefault("meta", {}).update(
        {
            "surface_protocol": "shortest_unique_evidence_v1",
            "surface_mode": mode,
            "offset_recovery_verified": True,
        }
    )
    return converted


def register_dataset(dataset_info_path: Path, dataset_name: str, output_path: Path) -> None:
    info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else {}
    entry = {
        "file_name": output_path.name,
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
    existing = info.get(dataset_name)
    if existing is not None and existing != entry:
        raise ValueError(f"dataset_info already contains a different entry for {dataset_name}")
    info[dataset_name] = entry
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=["direct", "sgcot"], required=True)
    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_info", type=Path)
    parser.add_argument("--heldout_types_json", type=Path)
    parser.add_argument("--require_no_heldout_leak", action="store_true")
    args = parser.parse_args()

    rows = [convert_row(row, args.mode) for row in load_jsonl(args.input_jsonl)]
    heldout_types = (
        json.loads(args.heldout_types_json.read_text(encoding="utf-8"))
        if args.heldout_types_json
        else []
    )
    leak_records = []
    if heldout_types:
        for index, row in enumerate(rows):
            for leak in find_heldout_leaks(row, heldout_types):
                leak_records.append({"row_index": index, **leak})
    if args.require_no_heldout_leak and leak_records:
        raise ValueError(f"held-out type leakage detected: {leak_records[:20]}")

    write_jsonl(args.output_jsonl, rows)
    if args.dataset_name:
        if args.dataset_info is None:
            parser.error("--dataset_info is required with --dataset_name")
        register_dataset(args.dataset_info, args.dataset_name, args.output_jsonl)
    summary = {
        "input": str(args.input_jsonl.resolve()),
        "output": str(args.output_jsonl.resolve()),
        "mode": args.mode,
        "rows": len(rows),
        "heldout_leaks": len(leak_records),
        "recovery_verified_rows": len(rows),
    }
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
