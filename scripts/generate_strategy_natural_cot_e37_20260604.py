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

import yaml


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
FORMAL_DATA_DIR = REPO / "data/stage2_formal_datasets"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
OUT_ROOT = REPO / "outputs/stage2_strategy_cot_e37"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPTIVE_PREFIX = f"{DATA_PREFIX}_adaptive"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
WARM_START = (
    "/workspace/project/outputs/stage2_adaptive_teacher_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_direct_teacher_full/checkpoint-258"
)
DEFAULT_BASE_URL = "${LLM_BASE_URL}"
TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def now_log() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M %z")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def extract_text(input_text: str) -> str:
    match = re.search(r"Text:\n(.*?)\n\nTokens:", input_text, flags=re.S)
    return match.group(1).strip() if match else input_text.strip()


def extract_schema(input_text: str) -> tuple[list[str], str]:
    match = re.search(r"Candidate event types:\n(.*?)\n\nSchema cards:\n(.*?)\n\nReturn JSON only\.", input_text, flags=re.S)
    if not match:
        return [], ""
    candidates = [x.strip() for x in match.group(1).split(",") if x.strip()]
    return candidates, match.group(2).strip()


def gold_json(row: dict) -> dict:
    return json.loads(row.get("gold_output") or extract_final_text(row["output"]) or row["output"])


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def compact_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def extract_final_text(text: str) -> str | None:
    for tag in ("final", "FINAL"):
        pattern = re.compile(rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>", re.I | re.S)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def extract_tag(text: str, tag: str) -> str | None:
    pattern = re.compile(rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>", re.I | re.S)
    match = pattern.search(text or "")
    return match.group(1).strip() if match else None


def surface_gold_json(row: dict) -> dict:
    payload = gold_json(row)
    out = []
    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict):
                args.append({"role": arg.get("role"), "text": arg.get("text")})
        out.append({"event_type": event.get("event_type"), "trigger": {"text": trigger.get("text")}, "arguments": args})
    return {"events": out}


def gold_strings(row: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    types, triggers, args, roles = [], [], [], []
    for event in gold_json(row).get("events", []) or []:
        if not isinstance(event, dict):
            continue
        if event.get("event_type"):
            types.append(event["event_type"])
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        if trigger.get("text"):
            triggers.append(trigger["text"])
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict):
                if arg.get("text"):
                    args.append(arg["text"])
                if arg.get("role"):
                    roles.append(arg["role"])
    return types, triggers, args, roles


def event_stats(row: dict) -> dict:
    events = [e for e in gold_json(row).get("events", []) if isinstance(e, dict)]
    arg_count = sum(len(e.get("arguments", []) or []) for e in events)
    role_count = len({a.get("role") for e in events for a in (e.get("arguments", []) or []) if isinstance(a, dict) and a.get("role")})
    return {"event_count": len(events), "argument_count": arg_count, "role_count": role_count}


def row_priority(row: dict) -> float:
    stats = event_stats(row)
    gold_types = (row.get("meta") or {}).get("gold_event_types") or []
    rare_bonus = sum(1 for t in gold_types if t.startswith(("Justice:", "Movement:", "Transaction:", "Contact:")))
    return stats["event_count"] * 5 + stats["argument_count"] * 3 + stats["role_count"] + rare_bonus


def sample_rows(rows: list[dict], limit: int, seed: int, run_name: str) -> list[dict]:
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
        rec.setdefault("meta", {})["e37_source_index"] = source_index
        rec["meta"]["e37_sample_id"] = f"{run_name}_{idx:04d}"
        out.append(rec)
    return out


def call_model(base_url: str, api_key: str, model: str, messages: list[dict], max_tokens: int, timeout: int) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
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
    }


def strip_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|xml|text)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_obj(text: str) -> dict:
    text = strip_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def generator_prompt(row: dict) -> str:
    candidates, schema_cards = extract_schema(row["input"])
    payload = {
        "task": "Generate strategy-constrained natural-language chain-of-thought supervision for event extraction.",
        "important_positioning": [
            "You are not labeling events.",
            "The gold final JSON is authoritative.",
            "Do not add, remove, or modify any event, trigger, argument, role, or offset in the final JSON.",
            "Your job is to verbalize predefined event-extraction strategies into faithful natural-language reasoning.",
        ],
        "required_output": "Return exactly <thinking>...</thinking> followed by <final>{gold JSON}</final>.",
        "thinking_strategy": [
            "Trigger grounding: identify text expressions that trigger each gold event.",
            "Event type discrimination: explain why each trigger matches the gold event type and not plausible competing candidate types.",
            "Argument role grounding: explain each gold argument role using the text.",
            "Boundary control: mention exact argument surface text and nearby words that should not be included when useful.",
            "Event separation or no-extra-event: explain why events are separate, or why no additional candidate event should be added.",
            "Final consistency: every event, trigger, argument, and role discussed must match the final JSON.",
        ],
        "style_constraints": [
            "Use fluent natural language, not a rigid checklist.",
            "Do not mention token indices, start/end offsets, character offsets, token counts, or span indices in <thinking>.",
            "Do not use old tags such as <STEP_REASONING> or <FINAL>.",
            "Keep <thinking> between 80 and 260 English words unless the example is very simple.",
        ],
        "input": {
            "text": extract_text(row["input"]),
            "candidate_event_types": candidates,
            "schema_cards": schema_cards,
            "gold_events_without_offsets": surface_gold_json(row)["events"],
            "gold_final_json_must_copy_exactly": gold_json(row),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def verifier_prompt(row: dict, thinking: str) -> str:
    candidates, schema_cards = extract_schema(row["input"])
    payload = {
        "task": "Strictly verify whether the natural-language thinking faithfully supports the gold event extraction JSON.",
        "instruction": "Be strict. This is filtering data for training, not improving or rewriting the answer.",
        "reject_if": [
            "thinking conflicts with the gold final JSON",
            "thinking invents extra events, triggers, arguments, roles, or facts",
            "thinking omits important gold events or core arguments",
            "thinking gives a wrong or vague event type explanation",
            "thinking fails to ground argument roles in the text",
            "thinking fails to discuss boundary control for nontrivial arguments",
            "thinking is generic and not grounded in the given text",
            "thinking mentions token indices, start/end offsets, character offsets, or token counts",
        ],
        "return_contract": {
            "pass": "boolean",
            "scores": {
                "trigger_grounding": "integer 1-5",
                "type_discrimination": "integer 1-5",
                "argument_grounding": "integer 1-5",
                "boundary_control": "integer 1-5",
                "event_separation": "integer 1-5",
                "final_consistency": "integer 1-5",
            },
            "errors": ["short error labels, empty if pass"],
            "reason": "one concise sentence",
        },
        "input": {
            "text": extract_text(row["input"]),
            "candidate_event_types": candidates,
            "schema_cards": schema_cards,
            "gold_final_json": gold_json(row),
            "thinking": thinking,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def hard_verify(row: dict, content: str) -> tuple[str | None, dict | None, list[str]]:
    errors = []
    content = strip_fence(content)
    if re.search(r"<\s*/?\s*STEP_REASONING\s*>|<\s*/?\s*FINAL\s*>", content):
        errors.append("contains_old_tags")
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
    if final_obj is not None and canonical_json(final_obj) != canonical_json(gold_json(row)):
        errors.append("final_not_equal_gold")
    if thinking:
        if re.search(r"\b(token|character)\s+(index|indices|offset|offsets|position|positions|count)\b", thinking, re.I):
            errors.append("thinking_mentions_offsets")
        if re.search(r"\b(start|end|offset|index|indices)\s*[:=]\s*\d+\b", thinking, re.I):
            errors.append("thinking_mentions_offsets")
        if re.search(r"\btoken\s+\d+\b", thinking, re.I):
            errors.append("thinking_mentions_offsets")
        word_count = len(thinking.split())
        if word_count < 45:
            errors.append("thinking_too_short")
        if word_count > 320:
            errors.append("thinking_too_long")
        event_types, triggers, arguments, roles = gold_strings(row)
        for typ in set(event_types):
            if typ not in thinking:
                errors.append(f"missing_event_type:{typ}")
        for trig in set(triggers):
            if trig and trig not in thinking:
                errors.append(f"missing_trigger:{trig}")
        if arguments:
            unique_args = set(arguments)
            covered = sum(1 for arg in unique_args if arg and arg in thinking)
            if covered / max(1, len(unique_args)) < 0.5:
                errors.append("low_argument_surface_coverage")
        candidates, _ = extract_schema(row["input"])
        mentioned_types = set(re.findall(r"\b[A-Z][A-Za-z]+:[A-Za-z-]+", thinking))
        unsupported = sorted(mentioned_types - set(candidates))
        if unsupported:
            errors.append("unsupported_event_types:" + ",".join(unsupported))
    return thinking, final_obj, errors


def semantic_pass(verifier_obj: dict) -> tuple[bool, list[str]]:
    errors = []
    if not isinstance(verifier_obj, dict):
        return False, ["verifier_not_object"]
    if verifier_obj.get("pass") is not True:
        errors.append("semantic_pass_false")
    scores = verifier_obj.get("scores") if isinstance(verifier_obj.get("scores"), dict) else {}
    required = [
        "trigger_grounding",
        "type_discrimination",
        "argument_grounding",
        "boundary_control",
        "event_separation",
        "final_consistency",
    ]
    for key in required:
        try:
            val = int(scores.get(key))
        except Exception:
            val = 0
        if key == "final_consistency":
            if val < 5:
                errors.append(f"low_{key}:{val}")
        elif val < 4:
            errors.append(f"low_{key}:{val}")
    verifier_errors = verifier_obj.get("errors")
    if isinstance(verifier_errors, list) and verifier_errors:
        errors.append("semantic_errors:" + ",".join(map(str, verifier_errors[:5])))
    return not errors, errors


def process_one(row: dict, args, api_key: str) -> dict:
    sample_id = row["meta"]["e37_sample_id"]
    rec = {"sample_id": sample_id, "source_index": row["meta"].get("e37_source_index"), "accepted": False}
    try:
        gen = call_model(
            args.base_url,
            api_key,
            args.model,
            [
                {"role": "system", "content": "You create faithful strategy-constrained CoT supervision for event extraction. Output only the requested tags."},
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
                {"role": "system", "content": "You are a strict verifier for event-extraction reasoning data. Return strict JSON only."},
                {"role": "user", "content": verifier_prompt(row, thinking or "")},
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


def run_probe(args) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = "Return exactly: {\"ok\":true}"
    levels = [int(x) for x in args.probe_workers.split(",") if x.strip()]
    summary = {"created_at": now_iso(), "model": args.model, "levels": []}
    for workers in levels:
        started = time.time()
        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(
                    call_model,
                    args.base_url,
                    api_key,
                    args.model,
                    [{"role": "user", "content": prompt}],
                    64,
                    args.timeout,
                )
                for _ in range(args.probe_requests)
            ]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    rec = fut.result()
                    rec["ok"] = True
                except Exception as exc:
                    rec = {"ok": False, "error": repr(exc)}
                records.append(rec)
        latencies = [r["latency_sec"] for r in records if r.get("ok") and r.get("latency_sec") is not None]
        level = {
            "workers": workers,
            "requests": len(records),
            "ok": sum(1 for r in records if r.get("ok")),
            "errors": [r.get("error") for r in records if not r.get("ok")],
            "wall_sec": time.time() - started,
            "avg_latency_sec": sum(latencies) / len(latencies) if latencies else None,
            "max_latency_sec": max(latencies) if latencies else None,
        }
        summary["levels"].append(level)
        write_json(out_dir / "concurrency_probe_summary.json", summary)
        print(json.dumps(level, ensure_ascii=False))
    return summary


def run_generation(rows: list[dict], args) -> list[dict]:
    raw_path = args.output_dir / "e37_raw.jsonl"
    existing = {}
    if args.reuse_existing and raw_path.exists():
        for rec in load_jsonl(raw_path):
            if args.retry_rejected and not rec.get("accepted"):
                continue
            existing[rec["sample_id"]] = rec
    pending = [row for row in rows if row["meta"]["e37_sample_id"] not in existing]
    results = [existing[row["meta"]["e37_sample_id"]] for row in rows if row["meta"]["e37_sample_id"] in existing]
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


def register_dataset(name: str, file_name: str) -> None:
    info_path = DATA_DIR / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info[name] = {"file_name": file_name, "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dataset(name: str, rows: list[dict]) -> None:
    path = DATA_DIR / f"{name}.jsonl"
    write_jsonl(path, rows)
    register_dataset(name, path.name)


def make_thinking_row(row: dict, thinking: str, dataset_role: str, run_name: str) -> dict:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    final = compact_json(gold_json(row))
    out["gold_output"] = final
    out["instruction"] = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<thinking>...</thinking>` with natural-language reasoning that follows the event extraction strategy: "
        "trigger grounding, event type discrimination, argument role grounding, boundary control, and event separation or no-extra-event. "
        "Then output `<final>{...}</final>` with the complete strict JSON event list. "
        "Do not output text outside these lowercase tags."
    )
    out["output"] = f"<thinking>{thinking.strip()}</thinking>\n<final>{final}</final>"
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "strategy_natural_cot_e37",
            "adaptive_target_style": "thinking_natural_cot",
            "adaptive_dataset_role": dataset_role,
            "e37_run_name": run_name,
            "e37_generator_model": "deepseek-v4-pro",
            "e37_verifier_model": "deepseek-v4-pro",
        }
    )
    return out


def make_eval_thinking_row(row: dict, dataset_role: str, run_name: str) -> dict:
    placeholder = (
        "Identify text-grounded event triggers, distinguish their event types from competing candidates, "
        "ground each argument role and boundary in the text, and explain why no extra event should be added."
    )
    return make_thinking_row(row, placeholder, dataset_role, run_name)


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
    write_yaml(path, config)
    return path


def write_datasets(sampled: list[dict], results: list[dict], args) -> dict:
    by_id = {r["meta"]["e37_sample_id"]: r for r in sampled}
    accepted = [r for r in results if r.get("accepted")]
    accepted_rows = []
    train_rows = []
    for rec in accepted:
        row = by_id[rec["sample_id"]]
        accepted_rows.append({**rec, "input": row["input"], "gold_output": compact_json(gold_json(row)), "meta": row.get("meta", {})})
        train_rows.append(make_thinking_row(row, rec["thinking"], "train", args.run_name))
    write_jsonl(args.output_dir / "accepted_thinking.jsonl", accepted_rows)
    branch = f"{args.run_name}_thinking_natural_cot"
    train_name = f"{ADAPTIVE_PREFIX}_{branch}_train_pos"
    write_dataset(train_name, train_rows)
    eval_names = []
    for split in ["dev_seen", "test_seen", "test_unseen"]:
        rows = load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_{split}_pos.jsonl")
        name = f"{ADAPTIVE_PREFIX}_{branch}_{split}_pos"
        write_dataset(name, [make_eval_thinking_row(row, split, args.run_name) for row in rows])
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
            err[rec["error"]] += 1
        gen = rec.get("generator") or {}
        ver = rec.get("verifier") or {}
        if gen.get("latency_sec") is not None:
            lat.append(gen["latency_sec"])
        if ver.get("latency_sec") is not None:
            lat.append(ver["latency_sec"])
    summary = {
        "created_at": now_iso(),
        "run_name": args.run_name,
        "model": args.model,
        "verifier_model": args.verifier_model,
        "sampled": len(sampled),
        "api_ok": api_ok,
        "hard_ok": hard_ok,
        "semantic_ok": semantic_ok,
        "accepted": accepted,
        "accept_rate": accepted / max(1, len(sampled)),
        "avg_call_latency_sec": sum(lat) / len(lat) if lat else None,
        "error_counts": dict(err.most_common(30)),
        "datasets": dataset_info or {},
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def write_experiment_note(args, status: str = "running") -> Path:
    path = EXPERIMENT_DIR / f"2026-06-04_{args.run_name}_strategy_natural_cot.md"
    if path.exists():
        return path
    ts = now_iso()
    exp_label = args.run_name.split("_", 1)[0].upper()
    body = f"""---
id: 2026-06-04_{args.run_name}_strategy_natural_cot
title: {exp_label} Strategy-Constrained Natural CoT {args.run_name}
kind: experiment
status: {status}
created_at: {ts}
updated_at: {ts}
owners:
  - codex
tags:
  - {args.run_name.split("_", 1)[0]}
  - deepseek-v4-pro
  - natural-cot
  - event-extraction
objective: Generate strategy-constrained natural-language CoT supervision with DeepSeek V4 Pro and verify it with hard checks plus DeepSeek semantic verification.
artifacts:
  configs:
    - /mnt/disk/gaojun/research/progressive-ee/configs/generated/stage2_adaptive/{RUN_PREFIX}_{args.run_name}_thinking_natural_cot_full_stepmatch.yaml
  outputs:
    - {args.output_dir}
related:
  plans:
    - /mnt/disk/gaojun/research/progressive-ee/PLANS.md
  docs:
    - /mnt/disk/gaojun/research/progressive-ee/docs/llm_generation_service.md
context:
  model: {args.model}
  verifier_model: {args.verifier_model}
  run_name: {args.run_name}
  limit: {args.limit}
  workers: {args.workers}
---

# {exp_label} Strategy-Constrained Natural CoT {args.run_name}

## Goal

Construct faithful natural-language `<thinking>...</thinking><final>...</final>` supervision from fixed gold event labels.

## Setup

- source train split: `data/stage2_formal_datasets/{DATA_PREFIX}_train_pos.jsonl`
- generator: `{args.model}`
- verifier: hard checks + `{args.verifier_model}`
- strategy: trigger grounding, type discrimination, argument role grounding, boundary control, event separation/no-extra-event, final consistency.

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
export OPENAI_BASE_URL=\"${LLM_BASE_URL}\"
export OPENAI_API_KEY=\"<virtual-key>\"
python3 scripts/generate_strategy_natural_cot_e37_20260604.py generate --run-name {args.run_name} --limit {args.limit} --workers {args.workers} --reuse-existing
```

## Run Log

### {now_log()}

- created experiment note and started generation.

## Result

Pending.

## Conclusion

Pending.

## Next

- manually audit accepted `<thinking>` traces.
- train Qwen3-1.7B if quality is acceptable.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--run-name", default="e37_smoke100")
    base.add_argument("--limit", type=int, default=100)
    base.add_argument("--seed", type=int, default=20260604)
    base.add_argument("--model", default="deepseek-v4-pro")
    base.add_argument("--verifier-model", default="deepseek-v4-pro")
    base.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    base.add_argument("--output-dir", type=Path)
    base.add_argument("--workers", type=int, default=4)
    base.add_argument("--timeout", type=int, default=240)
    base.add_argument("--gen-max-tokens", type=int, default=1200)
    base.add_argument("--verify-max-tokens", type=int, default=700)
    base.add_argument("--reuse-existing", action="store_true")
    base.add_argument("--retry-rejected", action="store_true")
    base.add_argument("--prepare-only", action="store_true")
    p_probe = sub.add_parser("probe", parents=[base])
    p_probe.add_argument("--probe-workers", default="1,2,4,6,8")
    p_probe.add_argument("--probe-requests", type=int, default=8)
    sub.add_parser("generate", parents=[base])
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = OUT_ROOT / f"{args.run_name}_20260604"
    if args.cmd == "probe":
        run_probe(args)
        return

    rows = load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    sampled = sample_rows(rows, args.limit, args.seed, args.run_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "sampled_rows.jsonl", sampled)
    write_experiment_note(args)
    config = {
        "created_at": now_iso(),
        "script": "scripts/generate_strategy_natural_cot_e37_20260604.py",
        "cmd": args.cmd,
        "run_name": args.run_name,
        "limit": args.limit,
        "seed": args.seed,
        "model": args.model,
        "verifier_model": args.verifier_model,
        "workers": args.workers,
        "output_dir": args.output_dir.as_posix(),
    }
    write_json(args.output_dir / "config.json", config)
    if args.prepare_only:
        print(json.dumps({"sampled": len(sampled), "output_dir": args.output_dir.as_posix()}, indent=2))
        return
    results = run_generation(sampled, args)
    dataset_info = write_datasets(sampled, results, args)
    summary = summarize(sampled, results, dataset_info, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
