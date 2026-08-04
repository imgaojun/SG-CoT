import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
SCRIPT_DIR = REPO / "scripts"
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

import prepare_1_7b_explicit_reason_forms_e21_20260525 as e21  # noqa: E402


DATA_DIR = REPO / "data/stage2_adaptive_datasets"
FORMAL_DATA_DIR = REPO / "data/stage2_formal_datasets"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPTIVE_PREFIX = f"{DATA_PREFIX}_adaptive"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
WARM_START = (
    "/workspace/project/outputs/stage2_adaptive_teacher_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_direct_teacher_full/checkpoint-258"
)
TZ = timezone(timedelta(hours=8))
FORMAL_BUDGETS = ["none", "standard"]
AUGMENT_LIMIT = 900
RNG = random.Random(20260527)

SPECS = {
    "e27a": {
        "branch": "eventmentions_budget_e27a_none_aug",
        "title": "E27A None Augmentation Baseline",
        "objective": "Measure data augmentation gains without a reasoning block.",
        "description": "Original plus targeted augmented inputs, trained only with forced none outputs.",
        "train_budgets": ["none"],
        "devpick_budget": "none",
        "target_style": "none_aug",
    },
    "e27b": {
        "branch": "eventmentions_budget_e27b_span_reason_aug",
        "title": "E27B Span-Reason Augmentation",
        "objective": "Test whether span-grounded reasoning adds value over the same augmented input distribution.",
        "description": "Original plus targeted augmented inputs, trained with forced standard outputs and compact span hints.",
        "train_budgets": ["standard"],
        "devpick_budget": "standard",
        "target_style": "span_reason_aug",
    },
    "e27c": {
        "branch": "eventmentions_budget_e27c_paired_none_standard_aug",
        "title": "E27C Paired None/Standard Augmentation",
        "objective": "Fairly compare none and standard budgets inside one model using paired augmented inputs.",
        "description": "Each original and augmented input is paired with both none and standard targets.",
        "train_budgets": ["none", "standard"],
        "devpick_budget": "standard",
        "target_style": "paired_none_standard_aug",
    },
}


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def parse_input(input_text):
    match = re.match(r"Text:\n(?P<text>.*?)\n\nTokens:\n(?P<tokens>.*?)\n\n(?P<rest>Candidate event types:\n.*)", input_text, re.S)
    if not match:
        raise ValueError("unexpected input format")
    tokens = match.group("tokens").strip().split()
    return tokens, match.group("rest")


def rebuild_input(tokens, rest):
    text = " ".join(tokens)
    return f"Text:\n{text}\n\nTokens:\n{' '.join(tokens)}\n\n{rest}"


def shift_index(index, insertions, is_end=False):
    if index is None:
        return None
    total = 0
    for pos, toks in insertions:
        if pos < index or (pos == index and not is_end):
            total += len(toks)
    return index + total


def shifted_gold(row, insertions):
    payload = e21.gold_json(row)
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    for event in payload.get("events", []):
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        if trigger:
            trigger["start"] = shift_index(trigger.get("start"), insertions, is_end=False)
            trigger["end"] = shift_index(trigger.get("end"), insertions, is_end=True)
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            arg["start"] = shift_index(arg.get("start"), insertions, is_end=False)
            arg["end"] = shift_index(arg.get("end"), insertions, is_end=True)
    return payload


def apply_insertions(tokens, insertions):
    out = list(tokens)
    added = 0
    for pos, toks in sorted(insertions, key=lambda item: item[0]):
        insert_at = max(0, min(len(out), pos + added))
        out[insert_at:insert_at] = toks
        added += len(toks)
    return out


def first_event(row):
    events = e21.gold_json(row).get("events", [])
    return events[0] if events else None


def first_argument(event):
    for arg in event.get("arguments", []) or []:
        if isinstance(arg, dict) and arg.get("start") is not None:
            return arg
    return None


def make_augmented_row(row, kind, aug_id):
    tokens, rest = parse_input(row["input"])
    event = first_event(row)
    if not event:
        return None
    trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
    arg = first_argument(event)
    insertions = []
    if kind == "boundary":
        if trigger.get("start") is None:
            return None
        insertions.append((trigger["start"], ["clearly"]))
        if arg and arg.get("start") is not None:
            insertions.append((arg["start"], ["the"]))
    elif kind == "role_contrast":
        insertions.append((len(tokens), ["Meanwhile", "other", "officials", "and", "civilians", "were", "mentioned", "."]))
    elif kind == "hard_negative":
        candidates = (row.get("meta") or {}).get("candidate_types") or []
        neg_words = []
        for cand in candidates[:4]:
            tail = cand.split(":")[-1].lower().replace("-", " ")
            neg_words.extend(tail.split()[:2])
        insertions.append((len(tokens), ["This", "context", "does", "not", "necessarily", "mean"] + neg_words[:8] + ["."]))
    else:
        raise ValueError(f"unknown augmentation kind: {kind}")

    new_tokens = apply_insertions(tokens, insertions)
    new_gold = shifted_gold(row, insertions)
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["input"] = rebuild_input(new_tokens, rest)
    out["output"] = json.dumps(new_gold, ensure_ascii=False)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "e27_augmented": True,
            "e27_augmentation_kind": kind,
            "e27_aug_id": aug_id,
            "e27_insertions": [{"pos": pos, "tokens": toks} for pos, toks in insertions],
        }
    )
    return out


def augmentation_candidates(rows):
    scored = []
    for idx, row in enumerate(rows):
        stats = e21.event_stats(row)
        if stats["event_count"] <= 0:
            continue
        score = stats["argument_count"] * 3 + stats["event_count"] * 2 + stats["role_count"]
        types = (row.get("meta") or {}).get("gold_event_types") or []
        if any(t.startswith(("Justice:", "Movement:", "Transaction:", "Life:")) for t in types):
            score += 2
        scored.append((score, idx, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored]


def build_augmented_pool(train_rows):
    pool = []
    for row in augmentation_candidates(train_rows):
        if len(pool) >= AUGMENT_LIMIT:
            break
        stats = e21.event_stats(row)
        kinds = ["boundary"]
        if stats["argument_count"] >= 2:
            kinds.append("role_contrast")
        if len((row.get("meta") or {}).get("candidate_types") or []) >= 5:
            kinds.append("hard_negative")
        for kind in kinds:
            if len(pool) >= AUGMENT_LIMIT:
                break
            aug = make_augmented_row(row, kind, f"aug{len(pool):04d}")
            if aug is not None:
                pool.append(aug)
    return pool


def event_mentions_from_payload(payload):
    return {
        "events": [
            {
                "event_type": event.get("event_type"),
                "trigger": event.get("trigger", {}),
            }
            for event in payload.get("events", [])
            if isinstance(event, dict)
        ]
    }


def span_hint(payload):
    events = []
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            args.append(
                {
                    "role": arg.get("role"),
                    "text": arg.get("text"),
                    "start": arg.get("start"),
                    "end": arg.get("end"),
                }
            )
        events.append(
            {
                "event_type": event.get("event_type"),
                "trigger": {"text": trigger.get("text"), "start": trigger.get("start"), "end": trigger.get("end")},
                "arguments": args,
            }
        )
    return {"events": events}


def budget_instruction(budget):
    if budget == "none":
        middle = "Do not output an intermediate reasoning block."
    elif budget == "standard":
        middle = "After the budget tag, output `<SPAN_HINT>{...}</SPAN_HINT>` with compact trigger and argument span checks."
    else:
        raise ValueError(f"unsupported budget: {budget}")
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
        f"Then output `<REASONING_BUDGET>{budget}</REASONING_BUDGET>`. "
        f"{middle} "
        "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        "Do not output text outside the requested tags."
    )


def output_text(row, budget):
    final_payload = e21.gold_json(row)
    mentions = json.dumps(event_mentions_from_payload(final_payload), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(final_payload, ensure_ascii=False, separators=(",", ":"))
    chunks = [f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>", f"<REASONING_BUDGET>{budget}</REASONING_BUDGET>"]
    if budget == "standard":
        hint = json.dumps(span_hint(final_payload), ensure_ascii=False, separators=(",", ":"))
        chunks.append(f"<SPAN_HINT>{hint}</SPAN_HINT>")
    chunks.append(f"<FINAL>{final}</FINAL>")
    return "\n".join(chunks)


def clone(row, budget, role, variant, source_kind):
    spec = SPECS[variant]
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["instruction"] = budget_instruction(budget)
    out["output"] = output_text(row, budget)
    out["gold_output"] = json.dumps(e21.gold_json(row), ensure_ascii=False)
    stats = e21.event_stats(row)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "paired_augmentation_e27",
            "adaptive_target_style": spec["target_style"],
            "adaptive_dataset_role": role,
            "adaptive_reasoning_budget": budget,
            "adaptive_budget_label": budget,
            "e27_variant": variant,
            "e27_branch": spec["branch"],
            "e27_source_kind": source_kind,
            "e27_event_count": stats["event_count"],
            "e27_argument_count": stats["argument_count"],
            "e27_role_count": stats["role_count"],
        }
    )
    return out


def write_dataset(name, rows):
    file_name = f"{name}.jsonl"
    e21.e15.write_jsonl(DATA_DIR / file_name, rows)
    e21.e15.update_dataset_info(name, file_name)


def write_config(variant, train_name, dev_name):
    branch = SPECS[variant]["branch"]
    out_config = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
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
        "logging_steps": 5,
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
    write_yaml(out_config, config)
    return out_config


def write_note(variant, train_name, dev_name, audit):
    timestamp = now_iso()
    spec = SPECS[variant]
    branch = spec["branch"]
    exp_id = f"2026-05-27_stage2_1_7b_{branch}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    out_dir = REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full"
    config_path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    log_stamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M %z")
    body = f"""---
id: {exp_id}
title: Stage2 1.7B {spec['title']}
kind: experiment
status: planned
created_at: {timestamp}
updated_at: {timestamp}
owners:
  - codex
tags:
  - stage2
  - qwen3-1.7b
  - paired-augmentation
  - forced-reasoning
objective: {spec['objective']}
artifacts:
  configs:
    - {config_path}
  outputs:
    - {out_dir}
related:
  plans:
    - {REPO / 'PLANS.md'}
context:
  dataset: RichERE split1 oracle_mixed_noise_top10_shuffle
  base_model: Qwen3-1.7B
  warm_start: {WARM_START}
  branch: {branch}
  variant: {variant}
---

# Stage2 1.7B {spec['title']}

## Goal

{spec['description']}

## Setup

- train dataset: `{train_name}`
- dev dataset: `{dev_name}`
- audit: `{json.dumps(audit, ensure_ascii=False, sort_keys=True)}`

## Commands

```bash
cd {REPO}
python3 scripts/prepare_1_7b_paired_augmentation_e27_20260527.py
bash scripts/launch_1_7b_paired_augmentation_e27_20260527.sh train {variant} 1
bash scripts/launch_1_7b_paired_augmentation_e27_20260527.sh devpick {variant} 1
bash scripts/launch_1_7b_paired_augmentation_e27_20260527.sh formal {variant} 1 2 3 4 7
```

## Run Log

### {log_stamp}

- prepared paired augmentation dataset/config/note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- run dev checkpoint selection and formal forced-budget evaluation.
"""
    note.write_text(body, encoding="utf-8")
    return note


def prepare_variant(variant, train_rows, aug_rows):
    spec = SPECS[variant]
    branch = spec["branch"]
    train = []
    for row in train_rows:
        for budget in spec["train_budgets"]:
            train.append(clone(row, budget, "train", variant, "original"))
    for row in aug_rows:
        for budget in spec["train_budgets"]:
            train.append(clone(row, budget, "train", variant, "augmented"))
    RNG.shuffle(train)
    train_name = f"{ADAPTIVE_PREFIX}_{branch}_train_pos"
    write_dataset(train_name, train)

    dev_rows = e21.e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_budget = spec["devpick_budget"]
    dev_name = f"{ADAPTIVE_PREFIX}_{branch}_forced_{dev_budget}_dev_seen_pos"
    write_dataset(dev_name, [clone(row, dev_budget, "dev_seen", variant, "original") for row in dev_rows])

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        rows = e21.e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_{split}_pos.jsonl")
        for budget in FORMAL_BUDGETS:
            name = f"{ADAPTIVE_PREFIX}_{branch}_forced_{budget}_{split}_pos"
            write_dataset(name, [clone(row, budget, split, variant, "original") for row in rows])
            eval_names.append(name)

    source_counts = Counter(row["meta"].get("e27_source_kind", "unknown") for row in train)
    budget_counts = Counter(row["meta"]["adaptive_reasoning_budget"] for row in train)
    target_tokens = Counter()
    for row in train:
        target_tokens[row["meta"]["adaptive_reasoning_budget"]] += len(row["output"].split())
    audit = {
        "recipe": spec["description"],
        "variant": variant,
        "branch": branch,
        "original_train_count": len(train_rows),
        "augmented_input_count": len(aug_rows),
        "train_budgets": spec["train_budgets"],
        "formal_budgets": FORMAL_BUDGETS,
        "total_train_rows": len(train),
        "train_source_counts": dict(source_counts),
        "train_budget_counts": dict(budget_counts),
        "approx_target_token_counts": dict(target_tokens),
        "augmentation": {
            "limit": AUGMENT_LIMIT,
            "kinds": dict(Counter((row.get("meta") or {}).get("e27_augmentation_kind") for row in aug_rows)),
        },
        "training_recipe": {"learning_rate": 3.0e-6, "num_train_epochs": 3.0},
    }
    config = write_config(variant, train_name, dev_name)
    note = write_note(variant, train_name, dev_name, audit)
    e21.e15.write_json(DATA_DIR / f"{train_name}.meta.json", {"audit": audit, "created_at": now_iso()})
    return {
        "name": variant,
        "branch": branch,
        "train_dataset": train_name,
        "dev_dataset": dev_name,
        "eval_datasets": eval_names,
        "config": config.as_posix(),
        "note": note.as_posix(),
        "audit": audit,
    }


def main():
    train_rows = e21.e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    aug_rows = build_augmented_pool(train_rows)
    payload = [prepare_variant(variant, train_rows, aug_rows) for variant in ["e27a", "e27b", "e27c"]]
    e21.e15.write_json(
        DATA_DIR / f"{ADAPTIVE_PREFIX}_eventmentions_budget_e27_augmentation_pool.meta.json",
        {
            "created_at": now_iso(),
            "original_train_count": len(train_rows),
            "augmented_input_count": len(aug_rows),
            "augmentation_kind_counts": dict(Counter((row.get("meta") or {}).get("e27_augmentation_kind") for row in aug_rows)),
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
