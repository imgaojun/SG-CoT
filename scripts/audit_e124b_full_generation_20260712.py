#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def evaluate_full(
    rows: list[dict],
    summary: dict,
    *,
    expected_rows: int,
    min_accepted: int,
    max_attempts: int,
    max_verifier_failure_rate: float,
    verify_max_tokens: int,
    min_p99_headroom_tokens: int,
) -> dict:
    sample_ids = [row.get("sample_id") for row in rows]
    accepted = [row for row in rows if row.get("accepted")]
    attempts = [attempt for row in rows for attempt in row.get("attempts", [])]
    verifier_attempts = [attempt for attempt in attempts if isinstance(attempt.get("verifier"), dict)]
    completion_tokens = [
        value
        for attempt in verifier_attempts
        for value in [(attempt["verifier"].get("usage") or {}).get("completion_tokens")]
        if isinstance(value, int)
    ]
    parse_errors = sum(a.get("error_stage") == "verifier_parse" for a in attempts)
    empty_content = sum(
        not isinstance(a["verifier"].get("content"), str)
        or not a["verifier"].get("content", "").strip()
        for a in verifier_attempts
    )
    length_finishes = sum(
        a["verifier"].get("finish_reason") == "length" for a in verifier_attempts
    )
    allowed_verifier_failures = int(expected_rows * max_verifier_failure_rate)
    p99 = percentile(completion_tokens, 0.99)
    invalid_attempt_counts = [
        row.get("sample_id")
        for row in rows
        if not 1 <= len(row.get("attempts", [])) <= max_attempts
    ]
    invalid_accepted = [
        row.get("sample_id")
        for row in accepted
        if not row.get("hard_ok") or not row.get("semantic_ok")
    ]
    checks = {
        "exact_row_count": len(rows) == expected_rows,
        "unique_sample_ids": len(set(sample_ids)) == expected_rows and None not in sample_ids,
        "summary_counts_match": (
            summary.get("sampled") == expected_rows
            and summary.get("accepted") == len(accepted)
        ),
        "minimum_accepted": len(accepted) >= min_accepted,
        "attempt_bounds": not invalid_attempt_counts,
        "accepted_hard_semantic_valid": not invalid_accepted,
        "verifier_parse_error_rate": parse_errors <= allowed_verifier_failures,
        "empty_verifier_content_rate": empty_content <= allowed_verifier_failures,
        "verifier_length_finish_rate": length_finishes <= allowed_verifier_failures,
        "verifier_p99_token_headroom": (
            p99 is not None
            and p99 <= verify_max_tokens - min_p99_headroom_tokens
        ),
    }
    return {
        "protocol": "e124b-glm51-selfverifier4096-strict-full1500",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "rows": len(rows),
            "accepted": len(accepted),
            "rejected": len(rows) - len(accepted),
            "attempts": len(attempts),
            "verifier_attempts": len(verifier_attempts),
            "verifier_parse_errors": parse_errors,
            "empty_verifier_content": empty_content,
            "verifier_length_finishes": length_finishes,
            "allowed_per_verifier_failure_kind": allowed_verifier_failures,
        },
        "attempt_error_stages": dict(
            sorted(Counter(a.get("error_stage") for a in attempts if a.get("error_stage")).items())
        ),
        "completion_tokens": {
            "median": percentile(completion_tokens, 0.5),
            "p95": percentile(completion_tokens, 0.95),
            "p99": p99,
            "max": max(completion_tokens) if completion_tokens else None,
            "configured_max": verify_max_tokens,
            "required_p99_headroom": min_p99_headroom_tokens,
        },
        "invalid_attempt_count_sample_ids": invalid_attempt_counts,
        "invalid_accepted_sample_ids": invalid_accepted,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--expected_rows", type=int, default=1500)
    parser.add_argument("--min_accepted", type=int, default=1400)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_verifier_failure_rate", type=float, default=0.01)
    parser.add_argument("--verify_max_tokens", type=int, default=4096)
    parser.add_argument("--min_p99_headroom_tokens", type=int, default=256)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_full(
        load_jsonl(args.raw_jsonl),
        json.loads(args.summary_json.read_text()),
        expected_rows=args.expected_rows,
        min_accepted=args.min_accepted,
        max_attempts=args.max_attempts,
        max_verifier_failure_rate=args.max_verifier_failure_rate,
        verify_max_tokens=args.verify_max_tokens,
        min_p99_headroom_tokens=args.min_p99_headroom_tokens,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 6 if args.require_pass and not result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
