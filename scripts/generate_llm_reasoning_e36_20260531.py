#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO / "scripts"
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

import prepare_1_7b_paired_augmentation_e27_20260527 as e27  # noqa: E402


DEFAULT_BASE_URL = "${LLM_BASE_URL}"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
OUT_ROOT = REPO / "outputs/stage2_llm_reasoning_e36/llm_reasoning_smoke100_20260531"
CONFIG_PATH = REPO / "configs/generated/stage2_adaptive/e36_llm_reasoning_smoke100_20260531.json"
REPORT_PATH = REPO / "reports/2026-05-31_e36_llm_reasoning_supervision_smoke.md"
EXPERIMENT_PATH = REPO / "experiments/2026-05-31_e36_llm_reasoning_supervision_smoke100.md"
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_text(input_text):
    match = re.search(r"Text:\n(.*?)\n\nTokens:", input_text, flags=re.S)
    return match.group(1).strip() if match else input_text.strip()


def extract_tokens(input_text):
    match = re.search(r"\n\nTokens:\n(.*?)\n\nCandidate event types:", input_text, flags=re.S)
    return match.group(1).strip().split() if match else []


def extract_schema(input_text):
    match = re.search(r"Candidate event types:\n(.*?)\n\nSchema cards:\n(.*?)\n\nReturn JSON only\.", input_text, flags=re.S)
    if not match:
        return [], ""
    candidates = [x.strip() for x in match.group(1).split(",") if x.strip()]
    return candidates, match.group(2).strip()


def gold_json(row):
    return json.loads(row.get("gold_output") or row["output"])


def surface_gold_json(row):
    payload = gold_json(row)
    events = []
    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            args.append({"role": arg.get("role"), "text": arg.get("text")})
        events.append(
            {
                "event_type": event.get("event_type"),
                "trigger": {"text": trigger.get("text")},
                "arguments": args,
            }
        )
    return {"events": events}


def event_stats(row):
    events = [e for e in gold_json(row).get("events", []) if isinstance(e, dict)]
    args = []
    roles = set()
    for event in events:
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict):
                args.append(arg)
                if arg.get("role"):
                    roles.add(arg.get("role"))
    return {"event_count": len(events), "argument_count": len(args), "role_count": len(roles)}


def row_priority(row):
    stats = event_stats(row)
    candidate_types = (row.get("meta") or {}).get("candidate_types") or []
    gold_types = (row.get("meta") or {}).get("gold_event_types") or []
    rare_bonus = sum(1 for t in gold_types if t.startswith(("Justice:", "Movement:", "Transaction:", "Contact:")))
    return stats["event_count"] * 4 + stats["argument_count"] * 3 + stats["role_count"] + rare_bonus + min(len(candidate_types), 10) * 0.1


def sample_rows(rows, limit, seed):
    rng = random.Random(seed)
    positives = [row for row in rows if event_stats(row)["event_count"] > 0]
    ranked = sorted(enumerate(positives), key=lambda item: (-row_priority(item[1]), item[0]))
    head_n = min(len(ranked), max(limit // 2, 1))
    head = ranked[:head_n]
    rest = ranked[head_n:]
    rng.shuffle(rest)
    picked = head + rest[: max(0, limit - len(head))]
    picked = sorted(picked, key=lambda item: item[0])
    out = []
    for local_idx, (source_index, row) in enumerate(picked):
        rec = json.loads(json.dumps(row, ensure_ascii=False))
        meta = rec.setdefault("meta", {})
        meta["e36_source_index"] = source_index
        meta["e36_sample_id"] = f"e36_smoke_{local_idx:04d}"
        out.append(rec)
    return out


def apply_run_sample_ids(rows, run_name):
    out = []
    for idx, row in enumerate(rows):
        rec = json.loads(json.dumps(row, ensure_ascii=False))
        rec.setdefault("meta", {})["e36_sample_id"] = f"{run_name}_{idx:04d}"
        out.append(rec)
    return out


def reasoning_prompt(row):
    candidates, schema_cards = extract_schema(row["input"])
    payload = {
        "task": "Create faithful step-by-step reasoning supervision for event extraction. The final gold events are already provided; do not change them.",
        "rules": [
            "Return strict JSON only.",
            "The reasoning must be faithful to gold_events and the text.",
            "Do not invent events, triggers, arguments, roles, or facts.",
            "Do not output the final JSON event list.",
            "Do not include token offsets, token indices, token counts, character offsets, or words such as token/start/end/index/offset in the reasoning.",
            "Use only natural surface text such as exact phrase \"John Smith\"; do not say token, span, start, end, index, offset, one-token, two-token, or multi-token.",
            "Use concise natural-language checklist lines.",
            "Mention check should identify gold triggers and event types.",
            "Type check should briefly explain why each gold event type is supported, and may contrast only candidate types if useful.",
            "Role coverage should mention gold arguments and roles; if there are no arguments, say no gold argument is supported.",
            "Boundary check should state exact argument surface text and nearby exclusions when useful.",
            "Event separation should state whether this is a single event or how multiple triggers/events stay separate.",
            "Keep the reasoning between 50 and 220 English words unless the example is very simple.",
        ],
        "output_contract": {
            "reasoning": "string containing exactly five checklist lines: Mention check, Type check, Role coverage, Boundary check, Event separation",
            "covered_event_types": ["gold event type strings"],
            "covered_triggers": ["gold trigger text strings"],
            "covered_arguments": ["gold argument text strings"],
        },
        "input": {
            "text": extract_text(row["input"]),
            "candidate_event_types": candidates,
            "schema_cards": schema_cards,
            "gold_events_without_offsets": surface_gold_json(row).get("events", []),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_model(base_url, api_key, model, prompt, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You write faithful event-extraction reasoning supervision. You output strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    choice = data["choices"][0]
    return {
        "latency_sec": time.time() - started,
        "finish_reason": choice.get("finish_reason"),
        "content": choice.get("message", {}).get("content"),
        "usage": data.get("usage", {}),
        "raw_response": data,
    }


def extract_json(text):
    if not text:
        raise ValueError("empty_content")
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


def gold_strings(row):
    events = gold_json(row).get("events", []) or []
    event_types = []
    triggers = []
    arguments = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("event_type"):
            event_types.append(event["event_type"])
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        if trigger.get("text"):
            triggers.append(trigger["text"])
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict) and arg.get("text"):
                arguments.append(arg["text"])
    return event_types, triggers, arguments


def validate_reasoning(row, obj):
    errors = []
    if not isinstance(obj, dict):
        return ["json_not_object"]
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return ["missing_reasoning"]
    if "<FINAL>" in reasoning or '"events"' in reasoning:
        errors.append("contains_final_or_json")
    if re.search(r"\b(start|end|offset|index|token index|token span)\s*[:=]?\s*\d+\b", reasoning, flags=re.I):
        errors.append("contains_offsets")
    if re.search(r"\btoken\s+\d+\b", reasoning, flags=re.I):
        errors.append("contains_offsets")
    if re.search(r"\b\d+\s*[- ]?(token|tokens)\b", reasoning, flags=re.I):
        errors.append("contains_offsets")
    if re.search(r"\b(one|two|three|four|five|single|multi)[- ]token\b", reasoning, flags=re.I):
        errors.append("contains_offsets")
    if re.search(r"\b(tokens?|offsets?|indices|indexes)\b", reasoning, flags=re.I):
        errors.append("contains_offsets")
    required_sections = ["Mention check", "Type check", "Role coverage", "Boundary check", "Event separation"]
    for section in required_sections:
        if section.lower() not in reasoning.lower():
            errors.append(f"missing_{section.lower().replace(' ', '_')}")
    word_count = len(reasoning.split())
    if word_count < 30:
        errors.append("too_short")
    if word_count > 260:
        errors.append("too_long")
    event_types, triggers, arguments = gold_strings(row)
    for typ in set(event_types):
        if typ not in reasoning:
            errors.append(f"missing_event_type:{typ}")
    for trig in set(triggers):
        if trig and trig not in reasoning:
            errors.append(f"missing_trigger:{trig}")
    if arguments:
        covered = sum(1 for arg in arguments if arg in reasoning)
        if covered / max(1, len(set(arguments))) < 0.6:
            errors.append("low_argument_coverage")
    candidates, _ = extract_schema(row["input"])
    allowed_types = set(candidates)
    mentioned_types = set(re.findall(r"\b[A-Z][A-Za-z]+:[A-Za-z-]+", reasoning))
    unsupported = sorted(mentioned_types - allowed_types)
    if unsupported:
        errors.append("unsupported_event_types:" + ",".join(unsupported))
    return errors


def make_reason_row(row, reasoning, dataset_role):
    out = json.loads(json.dumps(row, ensure_ascii=False))
    final = json.dumps(gold_json(row), ensure_ascii=False, separators=(",", ":"))
    out["gold_output"] = final
    out["instruction"] = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<STEP_REASONING>...</STEP_REASONING>` with five concise checklist lines: "
        "Mention check, Type check, Role coverage, Boundary check, and Event separation. "
        "Then output `<FINAL>{...}</FINAL>` with the complete strict JSON event list. "
        "Do not output token offsets in the reasoning block. Do not output text outside the requested tags."
    )
    out["output"] = f"<STEP_REASONING>{reasoning.strip()}</STEP_REASONING>\n<FINAL>{final}</FINAL>"
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "llm_reasoning_e36_smoke",
            "adaptive_target_style": "llm_checklist_reasoning",
            "adaptive_dataset_role": dataset_role,
            "e36_reasoning_source": "deepseek-v4-pro",
        }
    )
    return out


def make_direct_row(row, dataset_role):
    out = json.loads(json.dumps(row, ensure_ascii=False))
    final = json.dumps(gold_json(row), ensure_ascii=False, separators=(",", ":"))
    out["gold_output"] = final
    out["instruction"] = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "Output `<FINAL>{...}</FINAL>` with the complete strict JSON event list. "
        "Do not output text outside the requested tag."
    )
    out["output"] = f"<FINAL>{final}</FINAL>"
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "llm_reasoning_e36_smoke",
            "adaptive_target_style": "direct_final_only",
            "adaptive_dataset_role": dataset_role,
        }
    )
    return out


def process_one(row, args, api_key):
    sample_id = row["meta"]["e36_sample_id"]
    rec = {
        "sample_id": sample_id,
        "source_index": row["meta"].get("e36_source_index"),
        "api_ok": False,
        "json_ok": False,
        "accepted": False,
        "validation_errors": [],
    }
    try:
        response = call_model(args.base_url, api_key, args.model, reasoning_prompt(row), args.max_tokens, args.timeout)
        rec.update(response)
        rec["api_ok"] = True
        obj = extract_json(response["content"])
        rec["json_ok"] = True
        rec["annotation"] = obj
        rec["validation_errors"] = validate_reasoning(row, obj)
        rec["accepted"] = not rec["validation_errors"]
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, KeyError) as exc:
        rec["error"] = repr(exc)
    return rec


def run_generation(rows, args):
    raw_path = args.output_dir / "llm_reasoning_raw.jsonl"
    existing = {}
    if args.reuse_existing and raw_path.exists():
        for rec in load_jsonl(raw_path):
            if args.retry_rejected and not rec.get("accepted"):
                continue
            existing[rec["sample_id"]] = rec

    pending = [row for row in rows if row["meta"]["e36_sample_id"] not in existing]
    results = [existing[row["meta"]["e36_sample_id"]] for row in rows if row["meta"]["e36_sample_id"] in existing]
    if pending:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for pending LLM calls")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(process_one, row, args, api_key) for row in pending]
            for fut in concurrent.futures.as_completed(futs):
                rec = fut.result()
                results.append(rec)
                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(json.dumps({"sample_id": rec["sample_id"], "accepted": rec["accepted"], "errors": rec.get("validation_errors")}, ensure_ascii=False))
    results = sorted(results, key=lambda r: r["sample_id"])
    write_jsonl(raw_path, results)
    return results


def write_datasets(sampled_rows, results, args):
    by_id = {row["meta"]["e36_sample_id"]: row for row in sampled_rows}
    accepted = [rec for rec in results if rec.get("accepted")]
    accepted_rows = []
    direct_rows = []
    reason_rows = []
    for rec in accepted:
        row = by_id[rec["sample_id"]]
        reasoning = rec["annotation"]["reasoning"]
        accepted_rows.append({**rec, "input": row["input"], "gold_output": row["output"], "meta": row.get("meta", {})})
        direct_rows.append(make_direct_row(row, "train"))
        reason_rows.append(make_reason_row(row, reasoning, "train"))
    write_jsonl(args.output_dir / "accepted_reasoning.jsonl", accepted_rows)
    direct_name = f"{DATA_PREFIX}_adaptive_{args.run_name}_direct_final_only_train_pos"
    reason_name = f"{DATA_PREFIX}_adaptive_{args.run_name}_llm_checklist_reason_train_pos"
    e27.write_dataset(direct_name, direct_rows)
    e27.write_dataset(reason_name, reason_rows)

    eval_datasets = []
    for split in ["dev_seen", "test_seen", "test_unseen"]:
        rows = load_jsonl(REPO / f"data/stage2_formal_datasets/{DATA_PREFIX}_{split}_pos.jsonl")
        direct_eval_name = f"{DATA_PREFIX}_adaptive_{args.run_name}_direct_final_only_{split}_pos"
        reason_eval_name = f"{DATA_PREFIX}_adaptive_{args.run_name}_llm_checklist_reason_{split}_pos"
        e27.write_dataset(direct_eval_name, [make_direct_row(row, split) for row in rows])
        e27.write_dataset(reason_eval_name, [make_reason_row(row, "Mention check: use the text-supported event mentions.\nType check: choose only schema-supported event types.\nRole coverage: include only text-supported gold roles.\nBoundary check: use exact surface phrases and exclude nearby nonparticipants.\nEvent separation: keep distinct triggers and event frames separate.", split) for row in rows])
        eval_datasets.extend([direct_eval_name, reason_eval_name])

    direct_config = write_train_config(
        f"{args.run_name}_direct_final_only",
        direct_name,
        f"{DATA_PREFIX}_adaptive_{args.run_name}_direct_final_only_dev_seen_pos",
    )
    reason_config = write_train_config(
        f"{args.run_name}_llm_checklist_reason",
        reason_name,
        f"{DATA_PREFIX}_adaptive_{args.run_name}_llm_checklist_reason_dev_seen_pos",
    )
    return {
        "accepted_count": len(accepted),
        "direct_dataset": direct_name,
        "reason_dataset": reason_name,
        "direct_rows": len(direct_rows),
        "reason_rows": len(reason_rows),
        "eval_datasets": eval_datasets,
        "direct_config": direct_config.as_posix(),
        "reason_config": reason_config.as_posix(),
    }


def write_train_config(branch, train_name, dev_name):
    out_config = REPO / f"configs/generated/stage2_adaptive/{e27.RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    config = {
        "model_name_or_path": e27.WARM_START,
        "template": "qwen",
        "dataset_dir": "/workspace/project/data/stage2_adaptive_datasets",
        "dataset": train_name,
        "eval_dataset": dev_name,
        "output_dir": f"/workspace/project/outputs/stage2_adaptive_runs_user/{e27.RUN_PREFIX}_{branch}_full",
        "stage": "sft",
        "do_train": True,
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "logging_steps": 1,
        "report_to": "none",
        "finetuning_type": "full",
        "cutoff_len": 1024,
        "max_samples": 20000,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "packing": False,
        "learning_rate": 3.0e-6,
        "warmup_ratio": 0.05,
        "bf16": True,
        "val_size": 0.0,
        "eval_steps": 10,
        "do_eval": True,
        "save_only_model": True,
        "num_train_epochs": 3.0,
        "load_best_model_at_end": False,
        "deepspeed": "/workspace/project/configs/deepspeed/zero2_optimizer_offload_cpu.json",
    }
    e27.write_yaml(out_config, config)
    return out_config


def summarize(sampled_rows, results, dataset_info, args):
    api_ok = sum(1 for r in results if r.get("api_ok"))
    json_ok = sum(1 for r in results if r.get("json_ok"))
    accepted = sum(1 for r in results if r.get("accepted"))
    errors = Counter()
    latencies = []
    for rec in results:
        errors.update(rec.get("validation_errors") or [])
        if rec.get("latency_sec") is not None:
            latencies.append(rec["latency_sec"])
    stats = Counter()
    for row in sampled_rows:
        s = event_stats(row)
        stats[f"events_{s['event_count']}"] += 1
        stats[f"args_{min(s['argument_count'], 5)}"] += 1
    summary = {
        "created_at": now_iso(),
        "model": args.model,
        "base_url": args.base_url,
        "sampled": len(sampled_rows),
        "api_ok": api_ok,
        "json_ok": json_ok,
        "accepted": accepted,
        "accept_rate": accepted / max(1, len(sampled_rows)),
        "avg_latency_sec": sum(latencies) / len(latencies) if latencies else None,
        "validation_error_counts": dict(errors.most_common()),
        "sample_stats": dict(stats),
        "datasets": dataset_info,
    }
    write_json(args.output_dir / "summary.json", summary)
    lines = [
        "# E36 LLM Reasoning Supervision Smoke",
        "",
        f"- model: `{args.model}`",
        f"- output: `{args.output_dir}`",
        f"- sampled/API/JSON/accepted: `{len(sampled_rows)} / {api_ok} / {json_ok} / {accepted}`",
        f"- accept rate: `{summary['accept_rate']:.3f}`",
    ]
    if summary["avg_latency_sec"] is not None:
        lines.append(f"- avg latency sec: `{summary['avg_latency_sec']:.2f}`")
    lines.extend(
        [
            "",
            "## Datasets",
            "",
            f"- direct: `{dataset_info.get('direct_dataset')}` rows `{dataset_info.get('direct_rows')}`",
            f"- reason: `{dataset_info.get('reason_dataset')}` rows `{dataset_info.get('reason_rows')}`",
            "",
            "## Validation Errors",
            "",
        ]
    )
    if errors:
        for key, value in errors.most_common(20):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Reading", "", "Pending manual audit of accepted reasoning traces.", ""])
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary


def write_experiment_note(args):
    if args.experiment_path.exists() and not args.prepare_only:
        return
    timestamp = now_iso()
    body = f"""---
id: {args.run_name}_llm_reasoning_supervision
title: E36 LLM Reasoning Supervision {args.run_name}
kind: experiment
status: running
created_at: {timestamp}
updated_at: {timestamp}
owners:
  - codex
tags:
  - e36
  - llm-reasoning
  - data-generation
objective: Test whether DeepSeek-generated checklist reasoning can provide faithful extra supervision before scaling E36.
artifacts:
  configs:
    - {args.config_path}
  outputs:
    - {args.output_dir}
related:
  plans:
    - {REPO / 'PLANS.md'}
  docs:
    - {REPO / 'docs/llm_generation_service.md'}
context:
  model: {args.model}
  base_url_env: OPENAI_BASE_URL
  api_key_env: OPENAI_API_KEY
  run_name: {args.run_name}
  limit: {args.limit}
---

# E36 LLM Reasoning Supervision {args.run_name}

## Goal

Generate and verify a small batch of LLM-authored checklist reasoning targets while keeping gold final labels unchanged.

## Setup

- source train split: `data/stage2_formal_datasets/{DATA_PREFIX}_train_pos.jsonl`
- output root: `{args.output_dir}`
- model: `{args.model}`
- verifier: rule checks for JSON validity, required sections, gold event/trigger coverage, argument coverage, and unsupported event types.

## Commands

```bash
cd {REPO}
export OPENAI_BASE_URL=\"${LLM_BASE_URL}\"
export OPENAI_API_KEY=\"<virtual-key>\"
python3 scripts/generate_llm_reasoning_e36_20260531.py --limit {args.limit} --workers {args.workers} --reuse-existing
```

## Run Log

### {datetime.now(TZ).strftime('%Y-%m-%d %H:%M %z')}

- created E36 LLM reasoning smoke note/config and started generation.

## Result

Pending.

## Conclusion

Pending.

## Next

- manually audit accepted reasoning traces.
- if accept rate and quality are sufficient, scale to S0/S1/S3.
"""
    args.experiment_path.parent.mkdir(parents=True, exist_ok=True)
    args.experiment_path.write_text(body, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="e36_smoke100")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--config-path", type=Path)
    ap.add_argument("--report-path", type=Path)
    ap.add_argument("--experiment-path", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--retry-rejected", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    if args.config_path is None:
        args.config_path = REPO / f"configs/generated/stage2_adaptive/{args.run_name}_llm_reasoning_20260603.json"
    if args.report_path is None:
        args.report_path = REPO / f"reports/2026-06-03_{args.run_name}_llm_reasoning_supervision.md"
    if args.experiment_path is None:
        args.experiment_path = REPO / f"experiments/2026-06-03_{args.run_name}_llm_reasoning_supervision.md"

    train_path = REPO / f"data/stage2_formal_datasets/{DATA_PREFIX}_train_pos.jsonl"
    rows = load_jsonl(train_path)
    sampled = apply_run_sample_ids(sample_rows(rows, args.limit, args.seed), args.run_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "sampled_rows.jsonl", sampled)
    write_json(
        args.config_path,
        {
            "created_at": now_iso(),
            "script": "scripts/generate_llm_reasoning_e36_20260531.py",
            "source_train": train_path.as_posix(),
            "output_dir": args.output_dir.as_posix(),
            "run_name": args.run_name,
            "limit": args.limit,
            "seed": args.seed,
            "model": args.model,
            "base_url": args.base_url,
            "workers": args.workers,
            "timeout": args.timeout,
            "max_tokens": args.max_tokens,
        },
    )
    write_experiment_note(args)
    if args.prepare_only:
        print(json.dumps({"sampled": len(sampled), "output_dir": args.output_dir.as_posix(), "config": args.config_path.as_posix()}, indent=2))
        return
    results = run_generation(sampled, args)
    dataset_info = write_datasets(sampled, results, args)
    summary = summarize(sampled, results, dataset_info, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
