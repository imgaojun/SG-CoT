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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def select_last_hard_valid_attempt(record: dict) -> dict | None:
    for attempt in reversed(record.get("attempts") or []):
        if (
            attempt.get("hard_ok") is True
            and isinstance(attempt.get("thinking"), str)
            and attempt.get("thinking")
            and isinstance(attempt.get("final_obj"), dict)
        ):
            return attempt
    return None


def build_items(raw_rows: list[dict], sampled_rows: list[dict]) -> list[dict]:
    sampled_by_id = {
        (row.get("meta") or {}).get("e40_sample_id"): row for row in sampled_rows
    }
    if None in sampled_by_id or len(sampled_by_id) != len(sampled_rows):
        raise ValueError("sampled rows must have unique meta.e40_sample_id values")
    items = []
    for record in sorted(raw_rows, key=lambda row: row.get("sample_id") or ""):
        sample_id = record.get("sample_id")
        if sample_id not in sampled_by_id:
            raise ValueError(f"missing sampled source row for {sample_id}")
        selected = select_last_hard_valid_attempt(record)
        items.append(
            {
                "sample_id": sample_id,
                "source_row": sampled_by_id[sample_id],
                "source_index": record.get("source_index"),
                "originally_accepted": bool(record.get("accepted")),
                "selected_attempt": selected,
                "selected_attempt_number": selected.get("attempt") if selected else None,
            }
        )
    if len({item["sample_id"] for item in items}) != len(items):
        raise ValueError("raw rows contain duplicate sample IDs")
    return items


def stable_select(items: list[dict], sample_size: int, seed: int) -> list[dict]:
    eligible = [item for item in items if item["selected_attempt"] is not None]
    ranked = sorted(
        eligible,
        key=lambda item: (
            hashlib.sha256(f"{seed}\0{item['sample_id']}".encode()).hexdigest(),
            item["sample_id"],
        ),
    )
    if sample_size <= 0:
        return items
    if len(ranked) < sample_size:
        raise ValueError(f"requested {sample_size} rows but only {len(ranked)} are hard-valid")
    return ranked[:sample_size]


def verify_one(item: dict, args: argparse.Namespace, api_key: str) -> dict:
    selected = item["selected_attempt"]
    result = {
        "sample_id": item["sample_id"],
        "source_index": item["source_index"],
        "originally_accepted": item["originally_accepted"],
        "selected_attempt_number": item["selected_attempt_number"],
        "hard_ok": selected is not None,
        "semantic_ok": False,
        "attempts": [],
        "verifier_profile": args.verifier_profile,
    }
    if selected is None:
        result["hard_errors"] = ["no_existing_hard_valid_attempt"]
        return result
    prompt = generator.verifier_prompt(
        item["source_row"],
        selected["thinking"],
        selected["final_obj"],
        verifier_profile=args.verifier_profile,
    )
    messages = [
        {
            "role": "system",
            "content": "You are a strict verifier for event-extraction CoT/evidence data. Return strict JSON only.",
        },
        {"role": "user", "content": prompt},
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
            semantic_ok, semantic_errors = generator.semantic_pass(verifier_obj)
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
            # A valid semantic decision is final; retries only repair call/parse failures.
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


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(results: list[dict], selected_ids: list[str], args: argparse.Namespace) -> dict:
    attempts = [attempt for result in results for attempt in result.get("attempts", [])]
    verifier_attempts = [attempt for attempt in attempts if isinstance(attempt.get("verifier"), dict)]
    tokens = [
        value
        for attempt in verifier_attempts
        for value in [(attempt["verifier"].get("usage") or {}).get("completion_tokens")]
        if isinstance(value, int)
    ]
    parse_errors = sum(a.get("error_stage") == "verifier_parse" for a in attempts)
    call_errors = sum(a.get("error_stage") == "verifier_call" for a in attempts)
    empty_content = sum(
        not isinstance(a["verifier"].get("content"), str)
        or not a["verifier"].get("content", "").strip()
        for a in verifier_attempts
    )
    length_finishes = sum(
        a["verifier"].get("finish_reason") == "length" for a in verifier_attempts
    )
    valid_judgments = sum("verifier_obj" in result for result in results)
    semantic_passes = sum(result.get("semantic_ok") is True for result in results)
    hard_valid = sum(result.get("hard_ok") is True for result in results)
    allowed_failures = int(len(results) * args.max_failure_rate)
    p99 = percentile(tokens, 0.99)
    checks = {
        "exact_sample_size": len(results) == len(selected_ids),
        "unique_sample_ids": len(set(selected_ids)) == len(selected_ids),
        "minimum_hard_valid": hard_valid >= args.min_hard_valid,
        "minimum_valid_judgments": valid_judgments >= args.min_valid_judgments,
        "minimum_semantic_pass": semantic_passes >= args.min_semantic_pass,
        "attempt_bounds": all(
            (not result.get("hard_ok") and not result.get("attempts"))
            or 1 <= len(result.get("attempts", [])) <= args.max_attempts
            for result in results
        ),
        "verifier_call_error_rate": call_errors <= allowed_failures,
        "verifier_parse_error_rate": parse_errors <= allowed_failures,
        "empty_verifier_content_rate": empty_content <= allowed_failures,
        "verifier_length_finish_rate": length_finishes <= allowed_failures,
        "verifier_p99_token_headroom": (
            p99 is not None and p99 <= args.max_tokens - args.min_p99_headroom_tokens
        ),
    }
    return {
        "protocol": "e125-target-role-alias-reverification-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "selected_sample_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode()
        ).hexdigest(),
        "counts": {
            "rows": len(results),
            "hard_valid": hard_valid,
            "valid_semantic_judgments": valid_judgments,
            "semantic_passes": semantic_passes,
            "semantic_rejects": valid_judgments - semantic_passes,
            "attempts": len(attempts),
            "verifier_call_errors": call_errors,
            "verifier_parse_errors": parse_errors,
            "empty_verifier_content": empty_content,
            "verifier_length_finishes": length_finishes,
            "allowed_per_failure_kind": allowed_failures,
        },
        "completion_tokens": {
            "median": percentile(tokens, 0.5),
            "p95": percentile(tokens, 0.95),
            "p99": p99,
            "max": max(tokens) if tokens else None,
            "configured_max": args.max_tokens,
            "required_p99_headroom": args.min_p99_headroom_tokens,
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
    parser.add_argument("--raw_jsonl", type=Path, required=True)
    parser.add_argument("--sampled_rows_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_name", default="e125_role_alias_reverified")
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1250)
    parser.add_argument("--model", default="glm-5.1")
    parser.add_argument("--reasoning_effort")
    parser.add_argument("--verifier_profile", default="target_role_alias_v1")
    parser.add_argument("--max_tokens", type=int, default=6144)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--base_url", default=generator.e40.DEFAULT_BASE_URL)
    parser.add_argument("--min_hard_valid", type=int, required=True)
    parser.add_argument("--min_valid_judgments", type=int, required=True)
    parser.add_argument("--min_semantic_pass", type=int, required=True)
    parser.add_argument("--max_failure_rate", type=float, default=0.01)
    parser.add_argument("--min_p99_headroom_tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    if not args.output_dir.exists() and args.resume:
        raise SystemExit(f"cannot resume missing output directory: {args.output_dir}")
    api_key = generator.resolve_api_key()
    if not api_key:
        raise SystemExit("a LiteLLM API key is required")
    items = build_items(load_jsonl(args.raw_jsonl), load_jsonl(args.sampled_rows_jsonl))
    selected = stable_select(items, args.sample_size, args.seed)
    selected_ids = [item["sample_id"] for item in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "selection_manifest.jsonl",
        [
            {
                "sample_id": item["sample_id"],
                "source_index": item["source_index"],
                "originally_accepted": item["originally_accepted"],
                "selected_attempt_number": item["selected_attempt_number"],
                "hard_ok": item["selected_attempt"] is not None,
            }
            for item in selected
        ],
    )
    raw_path = args.output_dir / "reverify_raw.jsonl"
    existing = {
        row["sample_id"]: row for row in load_jsonl(raw_path)
    } if args.resume and raw_path.exists() else {}
    results = [existing[item["sample_id"]] for item in selected if item["sample_id"] in existing]
    pending = [item for item in selected if item["sample_id"] not in existing]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(verify_one, item, args, api_key) for item in pending]
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
    write_jsonl(raw_path, results)
    result_by_id = {result["sample_id"]: result for result in results}
    item_by_id = {item["sample_id"]: item for item in selected}
    generator.ACTIVE_GENERATOR_MODEL = "glm-5.1"
    generator.ACTIVE_VERIFIER_MODEL = args.model
    accepted_rows = []
    for sample_id in selected_ids:
        result = result_by_id[sample_id]
        item = item_by_id[sample_id]
        if not result.get("semantic_ok"):
            continue
        attempt = item["selected_attempt"]
        accepted_rows.append(
            generator.make_evidence_row(
                item["source_row"],
                attempt["thinking"],
                attempt["final_obj"],
                "train",
                args.run_name,
            )
        )
    write_jsonl(args.output_dir / "accepted_evidence_cot.jsonl", accepted_rows)
    summary = summarize(results, selected_ids, args)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 6 if args.require_pass and not summary["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
