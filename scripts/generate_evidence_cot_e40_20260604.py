#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
from collections import Counter
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.generate_strategy_natural_cot_e37_20260604 import (  # noqa: E402
    ADAPTIVE_PREFIX,
    CONFIG_DIR,
    DATA_DIR,
    DATA_PREFIX,
    DEFAULT_BASE_URL,
    FORMAL_DATA_DIR,
    RUN_PREFIX,
    WARM_START,
    call_model,
    compact_json,
    event_stats,
    extract_json_obj,
    extract_schema,
    extract_text,
    extract_tag,
    gold_json,
    load_jsonl,
    now_iso,
    now_log,
    register_dataset,
    row_priority,
    strip_fence,
    surface_gold_json,
    write_json,
    write_jsonl,
    write_yaml,
)


OUT_ROOT = REPO / "outputs/stage2_strategy_cot_e40"
EXPERIMENT_DIR = REPO / "experiments"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def sample_rows(rows: list[dict], limit: int, seed: int, run_name: str) -> list[dict]:
    import random

    rng = random.Random(seed)
    positives = [row for row in rows if event_stats(row)["event_count"] > 0]
    ranked = sorted(enumerate(positives), key=lambda item: (-row_priority(item[1]), item[0]))
    head_n = min(len(ranked), max(limit // 2, 1))
    head = ranked[:head_n]
    rest = ranked[head_n:]
    rng.shuffle(rest)
    picked = sorted(head + rest[: max(0, limit - len(head))], key=lambda item: item[0])
    out = []
    for idx, (source_index, row) in enumerate(picked):
        rec = json.loads(json.dumps(row, ensure_ascii=False))
        rec.setdefault("meta", {})["e40_source_index"] = source_index
        rec["meta"]["e40_sample_id"] = f"{run_name}_{idx:04d}"
        out.append(rec)
    return out


def surface_with_empty_evidence(row: dict) -> dict:
    out = []
    for event in surface_gold_json(row).get("events", []) or []:
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict):
                args.append({"role": arg.get("role"), "text": arg.get("text"), "evidence": ""})
        out.append(
            {
                "event_type": event.get("event_type"),
                "trigger": {"text": trigger.get("text"), "evidence": ""},
                "arguments": args,
            }
        )
    return {"events": out}


def project_surface(payload: dict) -> dict:
    out = []
    for event in payload.get("events", []) if isinstance(payload, dict) else []:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict):
                args.append({"role": arg.get("role"), "text": arg.get("text")})
        out.append({"event_type": event.get("event_type"), "trigger": {"text": trigger.get("text")}, "arguments": args})
    return {"events": out}


def canonical_surface(payload: dict) -> str:
    events = []
    for event in project_surface(payload).get("events", []):
        args = sorted(event.get("arguments", []), key=lambda x: (x.get("role") or "", x.get("text") or ""))
        events.append({"event_type": event.get("event_type"), "trigger": event.get("trigger"), "arguments": args})
    events = sorted(events, key=lambda e: (e.get("event_type") or "", (e.get("trigger") or {}).get("text") or "", json.dumps(e.get("arguments", []), sort_keys=True)))
    return json.dumps({"events": events}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def contains_forbidden_offset_key(obj) -> bool:
    forbidden = {"start", "end", "offset", "offsets", "index", "indices", "token_index", "token_indices", "char_offset"}
    if isinstance(obj, dict):
        for key, val in obj.items():
            if str(key).lower() in forbidden:
                return True
            if contains_forbidden_offset_key(val):
                return True
    elif isinstance(obj, list):
        return any(contains_forbidden_offset_key(x) for x in obj)
    return False


def evidence_records(payload: dict) -> list[tuple[str, str, str]]:
    records = []
    for event in payload.get("events", []) if isinstance(payload, dict) else []:
        if not isinstance(event, dict):
            continue
        trig = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        records.append(("trigger", trig.get("text") or "", trig.get("evidence") or ""))
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict):
                records.append((arg.get("role") or "argument", arg.get("text") or "", arg.get("evidence") or ""))
    return records


def evidence_is_grounded(text: str, surface: str, evidence: str) -> tuple[bool, str | None]:
    if not surface or not evidence:
        return False, "missing_surface_or_evidence"
    text_n = normalize_text(text)
    surface_n = normalize_text(surface)
    evidence_n = normalize_text(evidence)
    if surface_n not in evidence_n:
        return False, "surface_not_in_evidence"
    if evidence_n not in text_n:
        return False, "evidence_not_in_text"
    if len(evidence.split()) > 45:
        return False, "evidence_too_long"
    return True, None


def generator_prompt(row: dict) -> str:
    candidates, schema_cards = extract_schema(row["input"])
    payload = {
        "task": "Generate evidence-grounded natural-language chain-of-thought supervision for event extraction.",
        "positioning": [
            "You are not labeling events; the gold events are authoritative.",
            "Do not add, remove, reorder, or modify event types, trigger text, argument text, or roles.",
            "Your job is to produce a faithful natural-language reasoning trace and a surface-only final answer with local evidence.",
        ],
        "required_output": "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only.",
        "final_json_schema": {
            "events": [
                {
                    "event_type": "copy from gold",
                    "trigger": {"text": "copy trigger text", "evidence": "minimal contiguous quote from Text that contains the trigger text"},
                    "arguments": [
                        {"role": "copy role", "text": "copy argument text", "evidence": "minimal contiguous quote from Text that contains the argument text"}
                    ],
                }
            ]
        },
        "thinking_strategy": [
            "Ground each trigger in its local evidence.",
            "Discriminate the event type against plausible candidate types using the schema.",
            "Construct the role frame and ground each argument role in local evidence.",
            "Control boundaries by explaining why the exact surface text is used and nearby words are excluded when useful.",
            "Separate multiple events and explain why no extra event is added.",
            "Ensure the final JSON is exactly supported by the reasoning and evidence.",
        ],
        "constraints": [
            "Do not output token indices, numeric start/end offsets, character offsets, or span indices anywhere.",
            "Every evidence string must be an exact contiguous quote from Text and must contain the corresponding text field.",
            "Prefer short clause-level evidence, not a whole sentence, unless the whole sentence is necessary.",
            "Keep <thinking> fluent natural language between 100 and 300 English words.",
            "Return no markdown and no text outside the two tags.",
        ],
        "input": {
            "text": extract_text(row["input"]),
            "candidate_event_types": candidates,
            "schema_cards": schema_cards,
            "gold_surface_events_to_copy": surface_gold_json(row),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def verifier_prompt(row: dict, thinking: str, final_obj: dict) -> str:
    candidates, schema_cards = extract_schema(row["input"])
    payload = {
        "task": "Strictly verify evidence-grounded CoT training data for event extraction.",
        "instruction": "Do not repair the answer. Return strict JSON only.",
        "pass_requirements": [
            "The final surface events exactly match the gold surface events.",
            "Each evidence string is local, discriminative, and supports the trigger or argument role.",
            "The reasoning is faithful to the gold events, schema, evidence, and final JSON.",
            "The response does not rely on numeric offsets or token indices.",
        ],
        "return_contract": {
            "pass": "boolean",
            "scores": {
                "trigger_evidence": "integer 1-5",
                "type_discrimination": "integer 1-5",
                "argument_evidence": "integer 1-5",
                "boundary_control": "integer 1-5",
                "event_completeness": "integer 1-5",
                "final_structure_consistency": "integer 1-5",
                "evidence_locality": "integer 1-5",
            },
            "errors": ["short error labels, empty if pass"],
            "reason": "one concise sentence",
        },
        "input": {
            "text": extract_text(row["input"]),
            "candidate_event_types": candidates,
            "schema_cards": schema_cards,
            "gold_surface_events": surface_gold_json(row),
            "generated_thinking": thinking,
            "generated_final": final_obj,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def hard_verify(row: dict, content: str) -> tuple[str | None, dict | None, list[str]]:
    errors = []
    content = strip_fence(content)
    if re.search(r"<\s*/?\s*(STEP_REASONING|ROUTE|REASONING_BUDGET)\s*>", content, re.I) or re.search(r"<\s*/?\s*FINAL\s*>", content):
        errors.append("contains_old_or_upper_tags")
    thinking = extract_tag(content, "thinking")
    final_text = extract_tag(content, "final")
    if not thinking:
        errors.append("missing_thinking")
    if not final_text:
        errors.append("missing_final")
    final_obj = None
    if final_text:
        try:
            final_obj = json.loads(final_text)
        except Exception:
            errors.append("final_json_parse_failed")
    if final_obj is not None:
        if contains_forbidden_offset_key(final_obj):
            errors.append("final_contains_offset_key")
        if canonical_surface(final_obj) != canonical_surface(surface_gold_json(row)):
            errors.append("surface_final_not_equal_gold")
        text = extract_text(row["input"])
        for label, surface, evidence in evidence_records(final_obj):
            ok, err = evidence_is_grounded(text, surface, evidence)
            if not ok:
                errors.append(f"{label}_{err}")
    if thinking:
        if re.search(r"\b(token|character)\s+(index|indices|offset|offsets|position|positions|count)\b", thinking, re.I):
            errors.append("thinking_mentions_offsets")
        if re.search(r"\b(start|end|offset|index|indices)\s*[:=]\s*\d+\b", thinking, re.I):
            errors.append("thinking_mentions_offsets")
        word_count = len(thinking.split())
        if word_count < 60:
            errors.append("thinking_too_short")
        if word_count > 360:
            errors.append("thinking_too_long")
    return thinking, final_obj, errors


def semantic_pass(verifier_obj: dict) -> tuple[bool, list[str]]:
    errors = []
    if not isinstance(verifier_obj, dict):
        return False, ["verifier_not_object"]
    if verifier_obj.get("pass") is not True:
        errors.append("semantic_pass_false")
    scores = verifier_obj.get("scores") if isinstance(verifier_obj.get("scores"), dict) else {}
    for key in [
        "trigger_evidence",
        "type_discrimination",
        "argument_evidence",
        "boundary_control",
        "event_completeness",
        "evidence_locality",
    ]:
        try:
            val = int(scores.get(key))
        except Exception:
            val = 0
        if val < 4:
            errors.append(f"low_{key}:{val}")
    try:
        final_consistency = int(scores.get("final_structure_consistency"))
    except Exception:
        final_consistency = 0
    if final_consistency < 5:
        errors.append(f"low_final_structure_consistency:{final_consistency}")
    verifier_errors = verifier_obj.get("errors")
    if isinstance(verifier_errors, list) and verifier_errors:
        errors.append("semantic_errors:" + ",".join(map(str, verifier_errors[:5])))
    return not errors, errors


def process_one(row: dict, args, api_key: str) -> dict:
    sample_id = row["meta"]["e40_sample_id"]
    rec = {"sample_id": sample_id, "source_index": row["meta"].get("e40_source_index"), "accepted": False}
    try:
        gen = call_model(
            args.base_url,
            api_key,
            args.model,
            [
                {"role": "system", "content": "You create faithful evidence-grounded CoT supervision for event extraction. Output only the requested tags."},
                {"role": "user", "content": generator_prompt(row)},
            ],
            args.gen_max_tokens,
            args.timeout,
        )
        rec["generator"] = gen
        rec["api_ok"] = True
        thinking, final_obj, hard_errors = hard_verify(row, gen.get("content") or "")
        rec["thinking"] = thinking
        rec["final_obj"] = final_obj
        rec["hard_errors"] = hard_errors
        rec["hard_ok"] = not hard_errors
        if hard_errors:
            return rec
        ver = call_model(
            args.base_url,
            api_key,
            args.verifier_model,
            [
                {"role": "system", "content": "You are a strict verifier for event-extraction CoT/evidence data. Return strict JSON only."},
                {"role": "user", "content": verifier_prompt(row, thinking or "", final_obj or {})},
            ],
            args.verify_max_tokens,
            args.timeout,
        )
        rec["verifier"] = ver
        rec["verifier_api_ok"] = True
        verifier_obj = extract_json_obj(ver.get("content") or "")
        rec["verifier_obj"] = verifier_obj
        ok, semantic_errors = semantic_pass(verifier_obj)
        rec["semantic_errors"] = semantic_errors
        rec["semantic_ok"] = ok
        rec["accepted"] = rec["hard_ok"] and ok
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, KeyError) as exc:
        rec["error"] = repr(exc)
    return rec


def run_generation(rows: list[dict], args) -> list[dict]:
    raw_path = args.output_dir / "e40_raw.jsonl"
    existing = {}
    if args.reuse_existing and raw_path.exists():
        for rec in load_jsonl(raw_path):
            if args.retry_rejected and not rec.get("accepted"):
                continue
            existing[rec["sample_id"]] = rec
    pending = [row for row in rows if row["meta"]["e40_sample_id"] not in existing]
    results = [existing[row["meta"]["e40_sample_id"]] for row in rows if row["meta"]["e40_sample_id"] in existing]
    if pending:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(process_one, row, args, api_key) for row in pending]
            for fut in concurrent.futures.as_completed(futs):
                rec = fut.result()
                results.append(rec)
                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(json.dumps({"sample_id": rec["sample_id"], "accepted": rec.get("accepted"), "hard": rec.get("hard_errors"), "semantic": rec.get("semantic_errors"), "error": rec.get("error")}, ensure_ascii=False))
    results = sorted(results, key=lambda r: r["sample_id"])
    write_jsonl(raw_path, results)
    return results


def write_dataset(name: str, rows: list[dict]) -> None:
    path = DATA_DIR / f"{name}.jsonl"
    write_jsonl(path, rows)
    register_dataset(name, path.name)


def make_evidence_row(row: dict, thinking: str, final_obj: dict, dataset_role: str, run_name: str) -> dict:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["gold_output"] = compact_json(gold_json(row))
    out["instruction"] = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<thinking>...</thinking>` with natural-language reasoning grounded in local textual evidence. "
        "Then output `<final>{...}</final>` with a surface-only JSON event list: each trigger and argument must include "
        "`text` and a short contiguous `evidence` quote from the input text. Do not output numeric offsets, token indices, "
        "or text outside these lowercase tags."
    )
    out["output"] = f"<thinking>{thinking.strip()}</thinking>\n<final>{compact_json(final_obj)}</final>"
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "evidence_cot_e40",
            "adaptive_target_style": "thinking_surface_evidence_cot",
            "adaptive_dataset_role": dataset_role,
            "e40_run_name": run_name,
            "e40_generator_model": "deepseek-v4-pro",
            "e40_verifier_model": "deepseek-v4-pro",
        }
    )
    return out


def make_eval_evidence_row(row: dict, dataset_role: str, run_name: str) -> dict:
    placeholder = (
        "Ground each trigger and argument in short local evidence, discriminate event types using the schema, "
        "check role completeness, and keep the final answer surface-only without numeric offsets."
    )
    return make_evidence_row(row, placeholder, surface_with_empty_evidence(row), dataset_role, run_name)


def write_train_config(branch: str, train_name: str, dev_name: str) -> Path:
    path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    config = {
        "model_name_or_path": WARM_START,
        "template": "qwen",
        "dataset_dir": "/workspace/project/data/stage2_adaptive_datasets",
        "dataset": train_name,
        "eval_dataset": dev_name,
        "output_dir": f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full",
        "stage": "sft",
        "do_train": True,
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "logging_steps": 1,
        "report_to": "none",
        "finetuning_type": "full",
        "cutoff_len": 1536,
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
    write_yaml(path, config)
    return path


def write_datasets(sampled: list[dict], results: list[dict], args) -> dict:
    by_id = {r["meta"]["e40_sample_id"]: r for r in sampled}
    accepted = [r for r in results if r.get("accepted")]
    accepted_rows = []
    train_rows = []
    for rec in accepted:
        row = by_id[rec["sample_id"]]
        accepted_rows.append({**rec, "input": row["input"], "gold_output": compact_json(gold_json(row)), "meta": row.get("meta", {})})
        train_rows.append(make_evidence_row(row, rec["thinking"], rec["final_obj"], "train", args.run_name))
    write_jsonl(args.output_dir / "accepted_evidence_cot.jsonl", accepted_rows)
    branch = f"{args.run_name}_thinking_evidence_cot"
    train_name = f"{ADAPTIVE_PREFIX}_{branch}_train_pos"
    write_dataset(train_name, train_rows)
    eval_names = []
    for split in ["dev_seen", "test_seen", "test_unseen"]:
        rows = load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_{split}_pos.jsonl")
        name = f"{ADAPTIVE_PREFIX}_{branch}_{split}_pos"
        write_dataset(name, [make_eval_evidence_row(row, split, args.run_name) for row in rows])
        eval_names.append(name)
    config = write_train_config(branch, train_name, f"{ADAPTIVE_PREFIX}_{branch}_dev_seen_pos")
    return {
        "branch": branch,
        "accepted_count": len(accepted),
        "train_dataset": train_name,
        "train_rows": len(train_rows),
        "eval_datasets": eval_names,
        "train_config": config.as_posix(),
    }


def summarize(sampled: list[dict], results: list[dict], dataset_info: dict | None, args) -> dict:
    hard_ok = sum(1 for r in results if r.get("hard_ok"))
    semantic_ok = sum(1 for r in results if r.get("semantic_ok"))
    accepted = sum(1 for r in results if r.get("accepted"))
    api_ok = sum(1 for r in results if r.get("api_ok"))
    err = Counter()
    lat = []
    for rec in results:
        err.update(rec.get("hard_errors") or [])
        err.update(rec.get("semantic_errors") or [])
        if rec.get("error"):
            err.update([rec["error"]])
        if rec.get("generator", {}).get("latency_sec") is not None:
            lat.append(rec["generator"]["latency_sec"])
    return {
        "created_at": now_iso(),
        "run_name": args.run_name,
        "mode": "evidence_cot_e40",
        "sampled": len(sampled),
        "api_ok": api_ok,
        "hard_ok": hard_ok,
        "semantic_ok": semantic_ok,
        "accepted": accepted,
        "accept_rate": accepted / len(sampled) if sampled else 0.0,
        "avg_generator_latency_sec": sum(lat) / len(lat) if lat else None,
        "top_errors": err.most_common(30),
        "dataset_info": dataset_info,
        "output_dir": args.output_dir.as_posix(),
    }


def write_experiment_note(args, summary: dict | None) -> Path:
    path = EXPERIMENT_DIR / f"2026-06-04_{args.run_name}_evidence_cot.md"
    status = "completed" if summary and summary.get("dataset_info") else "running"
    now = now_iso()
    created = now
    if path.exists():
        text = path.read_text(encoding="utf-8")
        match = re.search(r"created_at:\s*(.+)", text)
        if match:
            created = match.group(1).strip()
    dataset_info = (summary or {}).get("dataset_info") or {}
    body = f"""---
id: 2026-06-04_{args.run_name}_evidence_cot
title: E40 Evidence-Grounded Natural CoT ({args.run_name})
kind: experiment
status: {status}
created_at: {created}
updated_at: {now}
owners:
  - codex
tags:
  - e40
  - evidence-cot
  - natural-cot
objective: Train and evaluate surface-only natural CoT event extraction with local evidence and deterministic offset recovery.
artifacts:
  configs:
    - {dataset_info.get("train_config", "pending")}
  outputs:
    - {args.output_dir.as_posix()}
related:
  plans:
    - /mnt/disk/gaojun/research/progressive-ee/PLANS.md
context:
  model: deepseek-v4-pro
  run_name: {args.run_name}
  target_style: thinking_surface_evidence_cot
---

# E40 Evidence-Grounded Natural CoT ({args.run_name})

## Goal

Construct natural-language CoT supervision where the final answer does not contain numeric offsets. Each trigger and argument is represented by surface text plus short local evidence, and evaluation recovers offsets deterministically from evidence.

## Setup

- source dataset: `data/stage2_formal_datasets/{DATA_PREFIX}_train_pos.jsonl`
- generator/verifier: `deepseek-v4-pro` through an OpenAI-compatible endpoint
- output directory: `{args.output_dir.as_posix()}`
- target accepted samples: `{args.limit}`

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
OPENAI_API_KEY=<virtual-key> python3 scripts/generate_evidence_cot_e40_20260604.py \\
  --run_name {args.run_name} \\
  --limit {args.limit} \\
  --seed {args.seed} \\
  --workers {args.workers} \\
  --base_url {args.base_url} \\
  --model {args.model} \\
  --verifier_model {args.verifier_model}
```

## Run Log

### {now_log()}

- generated or refreshed E40 evidence-CoT data artifacts
- sampled: `{(summary or {}).get("sampled", "pending")}`
- accepted: `{(summary or {}).get("accepted", "pending")}`
- accept_rate: `{(summary or {}).get("accept_rate", "pending")}`

## Result

{json.dumps(summary, ensure_ascii=False, indent=2) if summary else "Pending."}

## Conclusion

Pending training and formal evidence-aware evaluation.

## Next

- train the generated E40 branch if smoke quality and acceptance are sufficient
- evaluate with `eval_adaptive_route_generation_evidence.py` so surface/evidence final answers are mapped back to offsets before scoring
"""
    path.write_text(body, encoding="utf-8")
    return path


def update_plans(args, summary: dict) -> None:
    path = REPO / "PLANS.md"
    marker = "## 2026-06-04 E40 Evidence-Grounded CoT"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"""{marker}

- status: data construction completed for `{args.run_name}`
- objective: avoid forcing the model to generate numeric offsets; train it to produce `<thinking>` plus surface-only `<final>` with local evidence, then recover offsets deterministically.
- artifacts:
  - generation output: `{args.output_dir.as_posix()}`
  - accepted data: `{(args.output_dir / "accepted_evidence_cot.jsonl").as_posix()}`
  - train config: `{(summary.get("dataset_info") or {}).get("train_config")}`
- current result:
  - sampled/api_ok/hard_ok/semantic_ok/accepted: `{summary.get("sampled")}` / `{summary.get("api_ok")}` / `{summary.get("hard_ok")}` / `{summary.get("semantic_ok")}` / `{summary.get("accepted")}`
  - accept_rate: `{summary.get("accept_rate")}`
- next:
  - train E40 if accepted data quality is acceptable.
  - use evidence-aware evaluator to score recovered offsets against Direct baseline.
"""
    if marker in text:
        text = re.sub(rf"{re.escape(marker)}\n.*?(?=\n## |\Z)", block.rstrip() + "\n", text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="e40_smoke100")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--verifier_model", default="deepseek-v4-pro")
    parser.add_argument("--gen_max_tokens", type=int, default=1800)
    parser.add_argument("--verify_max_tokens", type=int, default=800)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--reuse_existing", action="store_true", default=True)
    parser.add_argument("--retry_rejected", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = OUT_ROOT / f"{args.run_name}_20260604"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    sampled = sample_rows(source_rows, args.limit, args.seed, args.run_name)
    write_jsonl(args.output_dir / "sampled_rows.jsonl", sampled)
    write_experiment_note(args, None)
    results = run_generation(sampled, args)
    dataset_info = write_datasets(sampled, results, args)
    summary = summarize(sampled, results, dataset_info, args)
    write_json(args.output_dir / "e40_summary.json", summary)
    write_experiment_note(args, summary)
    update_plans(args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
