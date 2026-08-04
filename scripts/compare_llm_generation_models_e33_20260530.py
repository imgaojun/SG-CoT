#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.candidate_type_recall.schema_library import SCHEMA_LIBRARY


DEFAULT_SEED = REPO / "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_train_pos.jsonl"
DEFAULT_OUT = REPO / "outputs/stage2_1_7b_llm_reconstruction_e33/model_compare_smoke_20260530"
DEFAULT_BASE_URL = "${LLM_BASE_URL}"
DEFAULT_MODELS = ["gemini-3.5-flash", "deepseek-v4-pro"]


def load_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_text(input_text):
    match = re.search(r"Text:\n(.*?)\n\nTokens:", input_text, flags=re.S)
    return match.group(1).strip() if match else input_text.strip()


def schema_card(event_type):
    entry = SCHEMA_LIBRARY[event_type]
    return {
        "event_type": event_type,
        "definition": entry["definition"],
        "trigger_cues": entry.get("trigger_cues", []),
        "core_roles": entry.get("core_roles", []),
    }


def choose_rows(rows, n, seed):
    rng = random.Random(seed)
    usable = []
    for row in rows:
        gold = row.get("meta", {}).get("gold_event_types") or []
        if gold and all(t in SCHEMA_LIBRARY for t in gold):
            usable.append(row)
    return rng.sample(usable, min(n, len(usable)))


def build_prompt(row, sample_id):
    seed_text = extract_text(row["input"])
    gold = row.get("meta", {}).get("gold_event_types") or []
    target_type = gold[0]
    distractors = [t for t in row.get("meta", {}).get("candidate_types", []) if t in SCHEMA_LIBRARY and t != target_type][:4]
    cards = [schema_card(target_type)] + [schema_card(t) for t in distractors]
    seed_output = json.loads(row["output"])
    plan = {
        "sample_id": sample_id,
        "target_event_type": target_type,
        "allowed_event_types": [c["event_type"] for c in cards],
        "difficulty": "balanced_reconstruction_with_one_hard_distractor",
        "requirements": [
            "write a new English news-style passage, not a paraphrase",
            "include exactly one supported target event of target_event_type",
            "include at least one plausible distractor phrase that does not create another gold event",
            "all trigger and argument surface strings must appear verbatim in text",
            "use only roles listed in the schema card",
        ],
    }
    prompt = {
        "task": "Generate one verified synthetic event-extraction training example.",
        "seed_text": seed_text,
        "seed_gold_events": seed_output.get("events", []),
        "generation_plan": plan,
        "schema_cards": cards,
        "output_contract": {
            "text": "string",
            "events": [
                {
                    "event_type": target_type,
                    "trigger": {"text": "surface string appearing in text"},
                    "arguments": [
                        {"role": "schema role", "text": "surface string appearing in text"}
                    ],
                }
            ],
            "reasoning": "brief natural-language rationale grounded only in the generated text",
            "generation_plan": plan,
        },
        "strict_rules": [
            "Return JSON only. Do not wrap in markdown.",
            "Do not include token offsets.",
            "Do not invent event types outside allowed_event_types.",
            "Every trigger.text and argument.text must be an exact substring of text.",
        ],
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def extract_json(text):
    if text is None:
        raise ValueError("empty content")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_model(base_url, api_key, model, prompt, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You generate high-precision JSON datasets for event extraction.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    latency = time.time() - started
    data = json.loads(raw)
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content")
    return {
        "raw_response": data,
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
        "latency_sec": latency,
    }


def validate_generated(obj, allowed_types):
    errors = []
    text = obj.get("text")
    events = obj.get("events")
    if not isinstance(text, str) or not text.strip():
        errors.append("missing_text")
        text = ""
    if not isinstance(events, list):
        errors.append("events_not_list")
        events = []
    for i, event in enumerate(events):
        etype = event.get("event_type")
        if etype not in allowed_types:
            errors.append(f"event_{i}_type_not_allowed")
            continue
        schema_roles = set(SCHEMA_LIBRARY[etype].get("core_roles", []))
        trigger = event.get("trigger", {})
        trigger_text = trigger.get("text") if isinstance(trigger, dict) else None
        if not trigger_text or trigger_text not in text:
            errors.append(f"event_{i}_trigger_missing_from_text")
        args = event.get("arguments", [])
        if not isinstance(args, list):
            errors.append(f"event_{i}_arguments_not_list")
            continue
        for j, arg in enumerate(args):
            role = arg.get("role")
            arg_text = arg.get("text")
            if role not in schema_roles:
                errors.append(f"event_{i}_arg_{j}_role_illegal")
            if not arg_text or arg_text not in text:
                errors.append(f"event_{i}_arg_{j}_text_missing_from_text")
    return {
        "valid": not errors,
        "errors": errors,
        "num_events": len(events),
        "text_chars": len(text),
    }


def summarize(records):
    by_model = {}
    for rec in records:
        stats = by_model.setdefault(
            rec["model"],
            {
                "model": rec["model"],
                "calls": 0,
                "api_success": 0,
                "json_success": 0,
                "valid_success": 0,
                "content_null": 0,
                "latencies": [],
                "finish_reasons": {},
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "text_tokens": 0,
                "validation_errors": {},
            },
        )
        stats["calls"] += 1
        if rec.get("api_ok"):
            stats["api_success"] += 1
        if rec.get("json_ok"):
            stats["json_success"] += 1
        if rec.get("validation", {}).get("valid"):
            stats["valid_success"] += 1
        if rec.get("content") is None:
            stats["content_null"] += 1
        if rec.get("latency_sec") is not None:
            stats["latencies"].append(rec["latency_sec"])
        fr = rec.get("finish_reason")
        stats["finish_reasons"][fr] = stats["finish_reasons"].get(fr, 0) + 1
        usage = rec.get("usage") or {}
        stats["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
        stats["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        details = usage.get("completion_tokens_details") or {}
        stats["reasoning_tokens"] += details.get("reasoning_tokens", 0) or 0
        stats["text_tokens"] += details.get("text_tokens", 0) or 0
        for err in rec.get("validation", {}).get("errors", []):
            stats["validation_errors"][err] = stats["validation_errors"].get(err, 0) + 1
    for stats in by_model.values():
        calls = max(stats["calls"], 1)
        lat = stats.pop("latencies")
        stats["api_success_rate"] = stats["api_success"] / calls
        stats["json_success_rate"] = stats["json_success"] / calls
        stats["valid_success_rate"] = stats["valid_success"] / calls
        stats["avg_latency_sec"] = sum(lat) / len(lat) if lat else None
    return list(by_model.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-jsonl", type=Path, default=DEFAULT_SEED)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--sample-seed", type=int, default=33)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENAI_API_KEY is required")

    rows = choose_rows(load_jsonl(args.seed_jsonl), args.num_samples, args.sample_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = []
    for idx, row in enumerate(rows):
        prompt = build_prompt(row, f"e33-smoke-{idx:03d}")
        prompts.append(
            {
                "sample_id": f"e33-smoke-{idx:03d}",
                "seed_meta": row.get("meta", {}),
                "prompt": prompt,
            }
        )
    (args.output_dir / "prompts.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in prompts) + "\n"
    )

    if args.dry_run:
        print(json.dumps({"output_dir": str(args.output_dir), "num_prompts": len(prompts)}, indent=2))
        return

    records = []
    for model in args.models:
        for prompt_rec in prompts:
            allowed = json.loads(prompt_rec["prompt"])["generation_plan"]["allowed_event_types"]
            rec = {
                "model": model,
                "sample_id": prompt_rec["sample_id"],
                "seed_gold_event_types": prompt_rec["seed_meta"].get("gold_event_types", []),
                "allowed_event_types": allowed,
                "api_ok": False,
                "json_ok": False,
            }
            try:
                response = call_model(
                    args.base_url,
                    api_key,
                    model,
                    prompt_rec["prompt"],
                    args.max_tokens,
                    args.timeout,
                )
                rec.update(
                    {
                        "api_ok": True,
                        "content": response["content"],
                        "finish_reason": response["finish_reason"],
                        "usage": response["usage"],
                        "latency_sec": response["latency_sec"],
                        "raw_response": response["raw_response"],
                    }
                )
                obj = extract_json(response["content"])
                rec["json_ok"] = True
                rec["generated"] = obj
                rec["validation"] = validate_generated(obj, set(allowed))
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, KeyError) as exc:
                rec["error"] = repr(exc)
            records.append(rec)
            with (args.output_dir / "records.jsonl").open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "base_url": args.base_url,
        "models": args.models,
        "num_samples": len(prompts),
        "max_tokens": args.max_tokens,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "by_model": summarize(records),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
