#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.generate_strategy_variants_cot_e47_20260606 as generator
from src.stage2_preference.reasoning_preference import (
    extract_final_json,
    extract_tag,
    is_exact,
    recover_offsets_from_evidence,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wnd_id(row: dict) -> str:
    value = (row.get("meta") or {}).get("wnd_id")
    if not isinstance(value, str) or not value:
        raise ValueError("audit row is missing meta.wnd_id")
    return value


def select_audit_rows(
    rows: list[dict], count: int, seed: int, excluded_wnd_ids: set[str] | None = None
) -> list[dict]:
    ids = [wnd_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("audit source contains duplicate wnd_id values")
    excluded_wnd_ids = excluded_wnd_ids or set()
    ranked = sorted(
        [row for row in rows if wnd_id(row) not in excluded_wnd_ids],
        key=lambda row: (
            hashlib.sha256(f"{seed}\0{wnd_id(row)}".encode()).hexdigest(),
            wnd_id(row),
        ),
    )
    if len(ranked) < count:
        raise ValueError(f"needed {count} audit rows but only found {len(ranked)}")
    return ranked[:count]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalized_surface_hard_verify(row: dict) -> tuple[str | None, dict | None, list[str]]:
    output = row.get("output") or ""
    thinking = extract_tag(output, "thinking")
    final_obj = extract_final_json(output)
    errors = []
    if not thinking:
        errors.append("missing_thinking")
    if final_obj is None:
        errors.append("missing_or_invalid_final")
    gold_raw = row.get("gold_output")
    if gold_raw is None:
        errors.append("missing_gold_output")
    if errors:
        return thinking, final_obj, errors
    try:
        gold = json.loads(gold_raw) if isinstance(gold_raw, str) else gold_raw
        recovered, diagnostics = recover_offsets_from_evidence(
            final_obj, row.get("input") or ""
        )
    except Exception as exc:
        return thinking, final_obj, [f"surface_recovery_exception:{type(exc).__name__}"]
    if diagnostics["missing_offsets"]:
        errors.append("surface_missing_offsets")
    if not is_exact(recovered, gold):
        errors.append("surface_final_not_exact_gold")
    return thinking, final_obj, errors


def audit_one(row: dict, args: argparse.Namespace, api_key: str) -> dict:
    sample_id = wnd_id(row)
    hard_profile = getattr(args, "hard_profile", "generator_hard_verify")
    if hard_profile == "normalized_surface_exact":
        thinking, final_obj, hard_errors = normalized_surface_hard_verify(row)
    elif hard_profile == "generator_hard_verify":
        thinking, final_obj, hard_errors = generator.hard_verify(
            row, row.get("output") or ""
        )
    else:
        raise ValueError(f"unknown hard profile: {hard_profile}")
    result = {
        "sample_id": sample_id,
        "hard_ok": not hard_errors,
        "hard_errors": hard_errors,
        "semantic_ok": False,
        "attempts": [],
    }
    if hard_errors:
        return result
    messages = [
        {
            "role": "system",
            "content": "You are a strict verifier for event-extraction CoT/evidence data. Return strict JSON only.",
        },
        {
            "role": "user",
            "content": generator.verifier_prompt(
                row,
                thinking or "",
                final_obj or {},
                verifier_profile=getattr(
                    args, "verifier_profile", "strict_schema_labels"
                ),
            ),
        },
    ]
    for attempt_index in range(args.max_attempts):
        verifier = None
        try:
            verifier = generator.call_model(
                args.base_url,
                api_key,
                args.model,
                messages,
                args.max_tokens,
                args.timeout,
                args.reasoning_effort,
            )
            verifier_obj = generator.e40.extract_json_obj(verifier.get("content") or "")
            semantic_ok, semantic_errors = generator.semantic_pass(
                verifier_obj,
                semantic_profile=getattr(args, "semantic_profile", "full_v1"),
            )
            attempt = {
                "attempt": attempt_index + 1,
                "verifier": verifier,
                "verifier_obj": verifier_obj,
                "semantic_ok": semantic_ok,
                "semantic_errors": semantic_errors,
            }
            result["attempts"].append(attempt)
            result.update(
                {
                    "semantic_ok": semantic_ok,
                    "semantic_errors": semantic_errors,
                    "verifier_obj": verifier_obj,
                }
            )
            # A valid semantic judgment is final, whether pass or reject.
            return result
        except Exception as exc:
            attempt = {
                "attempt": attempt_index + 1,
                "error": repr(exc),
                "error_stage": "verifier_parse" if verifier is not None else "verifier_call",
            }
            if verifier is not None:
                attempt["verifier"] = verifier
            result["attempts"].append(attempt)
    return result


def summarize(results: list[dict], selected_ids: list[str], args: argparse.Namespace) -> dict:
    attempts = [attempt for result in results for attempt in result.get("attempts", [])]
    verifier_attempts = [attempt for attempt in attempts if isinstance(attempt.get("verifier"), dict)]
    semantic_passes = sum(result.get("semantic_ok") is True for result in results)
    hard_valid = sum(result.get("hard_ok") is True for result in results)
    valid_judgments = sum("verifier_obj" in result for result in results)
    parse_errors = sum(a.get("error_stage") == "verifier_parse" for a in attempts)
    empty_content = sum(
        not isinstance(a["verifier"].get("content"), str)
        or not a["verifier"].get("content", "").strip()
        for a in verifier_attempts
    )
    length_finishes = sum(
        a["verifier"].get("finish_reason") == "length" for a in verifier_attempts
    )
    checks = {
        "exact_sample_size": len(results) == args.sample_size,
        "unique_sample_ids": len(set(selected_ids)) == args.sample_size,
        "all_hard_valid": hard_valid == args.sample_size,
        "all_receive_valid_judgment": valid_judgments == args.sample_size,
        "minimum_semantic_pass": semantic_passes >= args.min_semantic_pass,
        "attempt_bounds": all(
            1 <= len(result.get("attempts", [])) <= args.max_attempts for result in results
        ),
    }
    digest = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    return {
        "protocol": getattr(args, "protocol", "e124c-independent-deepseek-audit100-v1"),
        "passed": all(checks.values()),
        "checks": checks,
        "sample_wnd_ids_sha256": digest,
        "verifier": {
            "model": args.model,
            "profile": getattr(args, "verifier_profile", "strict_schema_labels"),
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
            "max_attempts": args.max_attempts,
            "workers": args.workers,
            "hard_profile": getattr(args, "hard_profile", "generator_hard_verify"),
            "semantic_profile": getattr(args, "semantic_profile", "full_v1"),
        },
        "counts": {
            "sampled": len(results),
            "hard_valid": hard_valid,
            "valid_semantic_judgments": valid_judgments,
            "semantic_passes": semantic_passes,
            "semantic_rejects": valid_judgments - semantic_passes,
            "attempts": len(attempts),
            "verifier_parse_errors": parse_errors,
            "empty_verifier_content": empty_content,
            "verifier_length_finishes": length_finishes,
        },
        "semantic_error_counts": dict(
            sorted(
                Counter(
                    error
                    for result in results
                    for error in result.get("semantic_errors", [])
                ).items()
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--exclude_jsonl", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", default="e124c-independent-deepseek-audit100-v1"
    )
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1242)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--reasoning_effort", default="high")
    parser.add_argument("--verifier_profile", default="strict_schema_labels")
    parser.add_argument(
        "--hard_profile",
        choices=["generator_hard_verify", "normalized_surface_exact"],
        default="generator_hard_verify",
    )
    parser.add_argument(
        "--semantic_profile",
        choices=["full_v1", "core_reasoning_v1"],
        default="full_v1",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min_semantic_pass", type=int, default=95)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--base_url", default=generator.e40.DEFAULT_BASE_URL)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse audit output directory: {args.output_dir}")
    api_key = generator.resolve_api_key()
    if not api_key:
        raise SystemExit("a LiteLLM API key is required")
    excluded_wnd_ids = (
        {wnd_id(row) for row in load_jsonl(args.exclude_jsonl)}
        if args.exclude_jsonl
        else set()
    )
    selected = select_audit_rows(
        load_jsonl(args.input_jsonl),
        args.sample_size,
        args.seed,
        excluded_wnd_ids=excluded_wnd_ids,
    )
    args.output_dir.mkdir(parents=True)
    selected_ids = [wnd_id(row) for row in selected]
    write_jsonl(args.output_dir / "sampled_rows.jsonl", selected)
    results = []
    raw_path = args.output_dir / "audit_raw.jsonl"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_one, row, args, api_key) for row in selected]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "sample_id": result["sample_id"],
                        "hard_ok": result["hard_ok"],
                        "semantic_ok": result["semantic_ok"],
                        "attempts": len(result["attempts"]),
                    },
                    ensure_ascii=False,
                )
            )
    results.sort(key=lambda result: result["sample_id"])
    summary = summarize(results, selected_ids, args)
    summary["input_jsonl_sha256"] = hashlib.sha256(
        args.input_jsonl.read_bytes()
    ).hexdigest()
    summary["excluded_wnd_ids"] = len(excluded_wnd_ids)
    summary["sample_overlap_with_excluded"] = len(
        set(selected_ids) & excluded_wnd_ids
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 6 if args.require_pass and not summary["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
