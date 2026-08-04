#!/usr/bin/env python3
"""Independently audit a reasoning-path preference dataset before training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_preference.reasoning_preference import (  # noqa: E402
    ERROR_CATEGORIES,
    classify_single_error,
    event_types_within_candidates,
    extract_final_json,
    has_complete_reasoning_response,
    is_exact,
    offsets_complete,
    recover_offsets_from_evidence,
    valid_length_pair,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("meta", {}).get("wnd_id") or row.get("wnd_id") or f"row-{index:06d}")


def parse_gold(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("gold_output", row.get("output", "{}"))
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("gold output is not a JSON object")
    return payload


def build_prompt(tokenizer: Any, instruction: str, input_text: str) -> str:
    content = f"{instruction}\n{input_text}"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def conversation_token_count(
    tokenizer: Any, instruction: str, input_text: str, response: str
) -> int:
    user_content = f"{instruction}\n{input_text}"
    if getattr(tokenizer, "chat_template", None):
        return len(
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
        )
    return token_count(tokenizer, user_content + "\n" + response)


def add_problem(problems: list[dict[str, Any]], index: int, wnd_id: str, check: str) -> None:
    problems.append({"row_index": index, "wnd_id": wnd_id, "check": check})


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    pairs = load_jsonl(args.preference_jsonl)
    source_rows = load_jsonl(args.input_jsonl)
    by_wnd = {row_id(row, index): row for index, row in enumerate(source_rows)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    problems: list[dict[str, Any]] = []
    seen_wnd_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    response_ratios: list[float] = []
    maximum_total_tokens = 0
    teacher_pairs_with_k8 = 0

    for index, pair in enumerate(pairs):
        meta = pair.get("meta") if isinstance(pair.get("meta"), dict) else {}
        wnd_id = str(meta.get("wnd_id", ""))
        if not wnd_id or wnd_id not in by_wnd:
            add_problem(problems, index, wnd_id, "unknown_or_missing_wnd_id")
            continue
        if wnd_id in seen_wnd_ids:
            add_problem(problems, index, wnd_id, "duplicate_wnd_id")
        seen_wnd_ids.add(wnd_id)

        row = by_wnd[wnd_id]
        if pair.get("instruction") != row.get("instruction") or pair.get("input") != row.get("input"):
            add_problem(problems, index, wnd_id, "prompt_mismatch")

        chosen = pair.get("chosen")
        rejected = pair.get("rejected")
        if not isinstance(chosen, str) or not has_complete_reasoning_response(chosen):
            add_problem(problems, index, wnd_id, "chosen_tags_incomplete_or_nonlowercase")
            continue
        if not isinstance(rejected, str) or not has_complete_reasoning_response(rejected):
            add_problem(problems, index, wnd_id, "rejected_tags_incomplete_or_nonlowercase")
            continue

        chosen_payload = extract_final_json(chosen)
        rejected_payload = extract_final_json(rejected)
        if chosen_payload is None:
            add_problem(problems, index, wnd_id, "chosen_final_json_invalid")
            continue
        if rejected_payload is None:
            add_problem(problems, index, wnd_id, "rejected_final_json_invalid")
            continue

        chosen_recovered, chosen_recovery = recover_offsets_from_evidence(chosen_payload, row["input"])
        rejected_recovered, rejected_recovery = recover_offsets_from_evidence(
            rejected_payload, row["input"]
        )
        gold = parse_gold(row)
        candidate_types = row.get("meta", {}).get("candidate_types", [])
        if chosen_recovery.get("missing_offsets") or not offsets_complete(chosen_recovered):
            add_problem(problems, index, wnd_id, "chosen_offsets_incomplete")
        if rejected_recovery.get("missing_offsets") or not offsets_complete(rejected_recovered):
            add_problem(problems, index, wnd_id, "rejected_offsets_incomplete")
        if not event_types_within_candidates(chosen_recovered, candidate_types):
            add_problem(problems, index, wnd_id, "chosen_type_outside_candidates")
        if not event_types_within_candidates(rejected_recovered, candidate_types):
            add_problem(problems, index, wnd_id, "rejected_type_outside_candidates")
        if not is_exact(chosen_recovered, gold):
            add_problem(problems, index, wnd_id, "chosen_not_exact")

        classified = classify_single_error(rejected_recovered, gold)
        recorded_category = meta.get("error_category")
        if classified not in ERROR_CATEGORIES:
            add_problem(problems, index, wnd_id, "rejected_not_single_error")
        if classified != recorded_category:
            add_problem(problems, index, wnd_id, "rejected_category_mismatch")

        chosen_tokens = token_count(tokenizer, chosen)
        rejected_tokens = token_count(tokenizer, rejected)
        chosen_total = conversation_token_count(
            tokenizer, row["instruction"], row["input"], chosen
        )
        rejected_total = conversation_token_count(
            tokenizer, row["instruction"], row["input"], rejected
        )
        if not valid_length_pair(
            chosen_tokens,
            rejected_tokens,
            chosen_total,
            rejected_total,
            args.cutoff_len,
            args.min_length_ratio,
            args.max_length_ratio,
        ):
            add_problem(problems, index, wnd_id, "length_or_cutoff_violation")
        response_ratios.append(chosen_tokens / rejected_tokens)
        maximum_total_tokens = max(maximum_total_tokens, chosen_total, rejected_total)

        source_counts[str(meta.get("chosen_source"))] += 1
        category_counts[str(recorded_category)] += 1
        if meta.get("chosen_source") == "verified_teacher_trace":
            observed_rounds = meta.get("sample_rounds_observed", [])
            has_k8_round = isinstance(observed_rounds, list) and any(
                isinstance(value, int) and value >= 1 for value in observed_rounds
            )
            if has_k8_round:
                teacher_pairs_with_k8 += 1
            else:
                add_problem(problems, index, wnd_id, "teacher_used_before_k8_topup")

    pair_count = len(pairs)
    teacher_fraction = (
        source_counts["verified_teacher_trace"] / pair_count if pair_count else 0.0
    )
    if teacher_fraction > args.max_teacher_fraction + 1e-12:
        add_problem(problems, -1, "*", "teacher_fraction_exceeded")

    category_fractions = {
        category: count / pair_count if pair_count else 0.0
        for category, count in sorted(category_counts.items())
    }
    if args.profile == "e81":
        for category, fraction in category_fractions.items():
            if fraction > args.max_e81_category_fraction + 1e-12:
                add_problem(problems, -1, "*", f"category_fraction_exceeded:{category}")
    else:
        omission_fraction = (
            category_counts["argument_omission"] + category_counts["event_omission"]
        ) / pair_count if pair_count else 0.0
        if omission_fraction > args.max_g9_omission_fraction + 1e-12:
            add_problem(problems, -1, "*", "g9_omission_fraction_exceeded")

    if pair_count < args.min_pairs:
        add_problem(problems, -1, "*", "minimum_pair_count_not_met")

    return {
        "valid": not problems,
        "preference_jsonl": str(args.preference_jsonl.resolve()),
        "input_jsonl": str(args.input_jsonl.resolve()),
        "profile": args.profile,
        "pairs": pair_count,
        "unique_windows": len(seen_wnd_ids),
        "minimum_required_pairs": args.min_pairs,
        "chosen_sources": dict(sorted(source_counts.items())),
        "teacher_fraction": teacher_fraction,
        "teacher_pairs_with_k8_topup": teacher_pairs_with_k8,
        "error_categories": dict(sorted(category_counts.items())),
        "error_category_fractions": category_fractions,
        "response_length_ratio_min": min(response_ratios) if response_ratios else None,
        "response_length_ratio_max": max(response_ratios) if response_ratios else None,
        "maximum_prompt_plus_response_tokens": maximum_total_tokens,
        "cutoff_len": args.cutoff_len,
        "problem_count": len(problems),
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference_jsonl", type=Path, required=True)
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--profile", choices=["e81", "g9"], required=True)
    parser.add_argument("--cutoff_len", type=int, required=True)
    parser.add_argument("--min_pairs", type=int, default=900)
    parser.add_argument("--min_length_ratio", type=float, default=0.7)
    parser.add_argument("--max_length_ratio", type=float, default=1.3)
    parser.add_argument("--max_teacher_fraction", type=float, default=0.30)
    parser.add_argument("--max_e81_category_fraction", type=float, default=0.40)
    parser.add_argument("--max_g9_omission_fraction", type=float, default=0.60)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    report = audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
