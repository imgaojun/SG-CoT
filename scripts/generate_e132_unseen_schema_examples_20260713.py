#!/usr/bin/env python3
"""Generate and locally verify E132 unseen-type trigger cues and examples."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


API_KEY_ENV_NAMES = ("LITELLM_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")


def resolve_api_key() -> str | None:
    for name in API_KEY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def call_model(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choice = payload["choices"][0]
    return {
        "content": choice.get("message", {}).get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage", {}),
        "latency_sec": time.time() - started,
    }


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("teacher response has no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("teacher response is not a JSON object")
    return value


def contains_trigger(sentence: str, trigger: str) -> bool:
    return bool(
        re.search(
            r"(?<!\w)" + re.escape(trigger.strip()) + r"(?!\w)",
            sentence,
            flags=re.IGNORECASE,
        )
    )


def hard_verify(
    request: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    errors = []
    if payload.get("event_type") != request["event_type"]:
        errors.append("event_type_mismatch")
    cues = payload.get("trigger_cues")
    if not isinstance(cues, list) or not all(isinstance(cue, str) and cue.strip() for cue in cues):
        errors.append("trigger_cues_invalid")
        cues = []
    normalized_cues = [cue.casefold().strip() for cue in cues]
    if not request["requested_trigger_cues_min"] <= len(cues) <= request["requested_trigger_cues_max"]:
        errors.append("trigger_cue_count")
    if len(set(normalized_cues)) != len(normalized_cues):
        errors.append("trigger_cues_duplicate")
    examples = payload.get("examples")
    if not isinstance(examples, list) or len(examples) != request["requested_examples"]:
        errors.append("example_count")
        examples = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            errors.append(f"example_{index}_not_object")
            continue
        sentence = example.get("sentence")
        trigger = example.get("trigger")
        if not isinstance(sentence, str) or not sentence.strip():
            errors.append(f"example_{index}_sentence_invalid")
            continue
        if not isinstance(trigger, str) or not trigger.strip():
            errors.append(f"example_{index}_trigger_invalid")
            continue
        if not contains_trigger(sentence, trigger):
            errors.append(f"example_{index}_trigger_not_locatable")
        if trigger.casefold().strip() not in normalized_cues:
            errors.append(f"example_{index}_trigger_not_in_cues")
    return errors


def teacher_prompt(request: dict[str, Any], errors: list[str] | None = None) -> str:
    repair = ""
    if errors:
        repair = "\nThe prior answer failed these checks: " + ", ".join(errors) + ". Return a corrected object."
    return (
        "Create a compact trigger-cue enrichment for one event schema. Use only the schema information "
        "below; do not assume access to labeled examples. Propose lexical trigger cues that directly evoke "
        "this event type, then write exactly two short natural sentences. Each example's `trigger` must occur "
        "verbatim in its `sentence` and must also be one of `trigger_cues`. Return strict JSON only with keys "
        "`event_type`, `trigger_cues`, and `examples`; each example has `sentence` and `trigger`.\n\n"
        f"Event type: {request['event_type']}\n"
        f"Definition: {request['definition']}\n"
        f"Core roles: {json.dumps(request['core_roles'], ensure_ascii=False)}\n"
        f"Existing schema cues: {json.dumps(request['original_trigger_cues'], ensure_ascii=False)}\n"
        f"Required trigger cue count: {request['requested_trigger_cues_min']} to "
        f"{request['requested_trigger_cues_max']}\n"
        f"Required example count: {request['requested_examples']}"
        + repair
    )


def generate_one(
    request: dict[str, Any],
    protocol: dict[str, Any],
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    attempts = []
    errors = None
    for attempt in range(1, int(protocol["max_attempts"]) + 1):
        response = None
        try:
            response = call_model(
                protocol["api_base_url"],
                api_key,
                protocol["teacher_model"],
                [
                    {"role": "system", "content": "Return one strict JSON object and no markdown."},
                    {"role": "user", "content": teacher_prompt(request, errors)},
                ],
                int(protocol["teacher_max_tokens"]),
                timeout,
            )
            payload = extract_json_object(response["content"])
            errors = hard_verify(request, payload)
            attempt_record = {
                "attempt": attempt,
                "response": response,
                "payload": payload,
                "hard_errors": errors,
            }
            attempts.append(attempt_record)
            if not errors:
                return {
                    "event_type": request["event_type"],
                    "accepted": True,
                    "attempts": attempts,
                    "payload": payload,
                    "hard_errors": [],
                }
        except Exception as exc:
            errors = [f"{type(exc).__name__}:{exc}"]
            record = {"attempt": attempt, "error": repr(exc), "hard_errors": errors}
            if response is not None:
                record["response"] = response
            attempts.append(record)
    return {
        "event_type": request["event_type"],
        "accepted": False,
        "attempts": attempts,
        "hard_errors": errors or ["unknown_failure"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--requests_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    requests = load_jsonl(args.requests_jsonl)
    if protocol.get("id") != "e132_trigger_cue_enrichment_v1":
        raise ValueError("unexpected protocol id")
    if len(requests) != int(protocol["unseen_types_expected"]):
        raise ValueError(f"unexpected synthesis request count: {len(requests)}")
    api_key = resolve_api_key()
    if not api_key:
        raise SystemExit("a LiteLLM API key is required through the process environment")
    args.output_dir.mkdir(parents=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_one, request, protocol, api_key, args.timeout) for request in requests]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({key: result[key] for key in ("event_type", "accepted", "hard_errors")}))
    results.sort(key=lambda item: item["event_type"])
    with (args.output_dir / "raw.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    accepted = [result["payload"] for result in results if result["accepted"]]
    with (args.output_dir / "accepted_cards.jsonl").open("w", encoding="utf-8") as handle:
        for payload in accepted:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    minimum = int(protocol["synthesis_min_hard_valid"])
    summary = {
        "id": "e132_unseen_schema_synthesis_gate_v1",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "requests": len(requests),
        "accepted": len(accepted),
        "rejected": len(requests) - len(accepted),
        "total_attempts": sum(len(result["attempts"]) for result in results),
        "minimum_accepted": minimum,
        "test_rows_read": 0,
        "passed": len(accepted) >= minimum,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_pass and not summary["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
