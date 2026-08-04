import json
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
SCRIPT_DIR = REPO / "scripts"
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

import prepare_4b_reason_format_ablation_e15_20260522 as e15  # noqa: E402


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

BRANCH = "eventmentions_budget_e21a_explicit_forms"
BUDGETS = ["none", "light", "standard", "deep"]


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def gold_json(row):
    if "gold_output" in row:
        return json.loads(row["gold_output"])
    return json.loads(row["output"])


def event_mentions_from_gold(row):
    events = []
    for event in gold_json(row).get("events", []):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        events.append(
            {
                "event_type": event.get("event_type"),
                "trigger": {
                    "text": trigger.get("text"),
                    "start": trigger.get("start"),
                    "end": trigger.get("end"),
                },
            }
        )
    return {"events": events}


def event_stats(row):
    events = [event for event in gold_json(row).get("events", []) if isinstance(event, dict)]
    arg_count = 0
    role_count = 0
    event_types = []
    for event in events:
        event_type = event.get("event_type")
        if event_type:
            event_types.append(event_type)
        args = [arg for arg in event.get("arguments", []) if isinstance(arg, dict)]
        arg_count += len(args)
        role_count += len({arg.get("role") for arg in args if arg.get("role")})
    return {"event_count": len(events), "argument_count": arg_count, "role_count": role_count, "event_types": event_types}


def schema_check(row):
    mentions = event_mentions_from_gold(row).get("events", [])
    event_types = []
    for event in mentions:
        event_type = event.get("event_type")
        if event_type and event_type not in event_types:
            event_types.append(event_type)
    return {"event_types": event_types, "trigger_count": len(mentions)}


def role_table(row):
    out = []
    for event in gold_json(row).get("events", []):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        roles = []
        seen = set()
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            if not role or role in seen:
                continue
            seen.add(role)
            roles.append({"role": role, "filled": True})
        out.append({"event_type": event.get("event_type"), "trigger": trigger.get("text"), "roles": roles})
    return {"events": out}


def argument_verify(row):
    out = []
    for event in gold_json(row).get("events", []):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            args.append({"role": arg.get("role"), "text": arg.get("text"), "status": "kept"})
        checks = ["schema_consistent"] if event.get("event_type") and trigger else []
        out.append(
            {
                "event_type": event.get("event_type"),
                "trigger": trigger.get("text"),
                "arguments": args,
                "checks": checks,
            }
        )
    return {"events": out}


def budget_instruction(budget):
    block = {
        "none": "Do not output an intermediate reasoning block.",
        "light": "After the budget tag, output `<SCHEMA_CHECK>{...}</SCHEMA_CHECK>` with event_types and trigger_count.",
        "standard": "After the budget tag, output `<ROLE_TABLE>{...}</ROLE_TABLE>` with event_type, trigger, and filled roles.",
        "deep": "After the budget tag, output `<ARGUMENT_VERIFY>{...}</ARGUMENT_VERIFY>` with kept arguments and compact schema checks.",
    }[budget]
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
        f"Then output `<REASONING_BUDGET>{budget}</REASONING_BUDGET>`. {block} "
        "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        "Do not output text outside the requested tags. "
        "If no valid event is expressed by the candidate set, all JSON objects must use empty event/type lists as appropriate."
    )


def output_text(row, budget):
    mentions = json.dumps(event_mentions_from_gold(row), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(gold_json(row), ensure_ascii=False, separators=(",", ":"))
    chunks = [f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>", f"<REASONING_BUDGET>{budget}</REASONING_BUDGET>"]
    if budget == "light":
        chunks.append(f"<SCHEMA_CHECK>{json.dumps(schema_check(row), ensure_ascii=False, separators=(',', ':'))}</SCHEMA_CHECK>")
    elif budget == "standard":
        chunks.append(f"<ROLE_TABLE>{json.dumps(role_table(row), ensure_ascii=False, separators=(',', ':'))}</ROLE_TABLE>")
    elif budget == "deep":
        chunks.append(f"<ARGUMENT_VERIFY>{json.dumps(argument_verify(row), ensure_ascii=False, separators=(',', ':'))}</ARGUMENT_VERIFY>")
    chunks.append(f"<FINAL>{final}</FINAL>")
    return "\n".join(chunks)


def clone(row, budget, role):
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["instruction"] = budget_instruction(budget)
    out["output"] = output_text(row, budget)
    out["gold_output"] = json.dumps(gold_json(row), ensure_ascii=False)
    meta = out.setdefault("meta", {})
    stats = event_stats(row)
    meta.update(
        {
            "adaptive_source": "explicit_reason_forms_e21",
            "adaptive_target_style": "explicit_reason_form_budget",
            "adaptive_dataset_role": role,
            "adaptive_reasoning_budget": budget,
            "adaptive_budget_label": budget,
            "e21_branch": BRANCH,
            "e21_budget": budget,
            "e21_event_count": stats["event_count"],
            "e21_argument_count": stats["argument_count"],
            "e21_role_count": stats["role_count"],
        }
    )
    return out


def write_dataset(name, rows):
    file_name = f"{name}.jsonl"
    e15.write_jsonl(DATA_DIR / file_name, rows)
    e15.update_dataset_info(name, file_name)
    return file_name


def write_config(train_name, dev_name):
    out_config = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
    config = {
        "model_name_or_path": WARM_START,
        "template": "qwen",
        "dataset_dir": "/workspace/project/data/stage2_adaptive_datasets",
        "dataset": train_name,
        "eval_dataset": dev_name,
        "output_dir": f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_full",
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


def write_note(train_name, dev_name, audit):
    timestamp = now_iso()
    exp_id = f"2026-05-25_stage2_1_7b_{BRANCH}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    out_dir = REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_full"
    config_path = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    body = f"""---
id: {exp_id}
title: Stage2 1.7B E21A Explicit Reasoning Forms
kind: experiment
status: planned
created_at: {timestamp}
updated_at: {timestamp}
owners:
  - codex
tags:
  - stage2
  - qwen3-1.7b
  - event-mentions
  - reasoning-budget
  - explicit-reasoning
objective: Test whether different visible reasoning forms make reasoning budgets meaningfully diverge.
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
  branch: {BRANCH}
---

# Stage2 1.7B E21A Explicit Reasoning Forms

## Goal

Train one model with four visible budget-specific forms: none, light schema check, standard role table, and deep argument verification.

## Setup

- train dataset: `{train_name}`
- dev dataset: `{dev_name}`
- audit: `{json.dumps(audit, ensure_ascii=False, sort_keys=True)}`

## Commands

```bash
cd {REPO}
python3 scripts/prepare_1_7b_explicit_reason_forms_e21_20260525.py
bash scripts/launch_1_7b_explicit_reason_forms_e21_20260525.sh train e21a 0
bash scripts/launch_1_7b_explicit_reason_forms_e21_20260525.sh devpick e21a 0
bash scripts/launch_1_7b_explicit_reason_forms_e21_20260525.sh formal e21a 0 1 2 3
```

## Run Log

### {datetime.now(TZ).strftime('%Y-%m-%d %H:%M %z')[:-2]}:{datetime.now(TZ).strftime('%z')[-2:]}

- prepared dataset/config/note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- run dev checkpoint selection and forced-budget formal evaluation.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    train_rows = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    rng = random.Random(20260525)
    train = []
    for row in train_rows:
        for budget in BUDGETS:
            train.append(clone(row, budget, "train"))
    rng.shuffle(train)
    train_name = f"{ADAPTIVE_PREFIX}_{BRANCH}_train_pos"
    write_dataset(train_name, train)

    dev_rows = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_name = f"{ADAPTIVE_PREFIX}_{BRANCH}_forced_standard_dev_seen_pos"
    write_dataset(dev_name, [clone(row, "standard", "dev_seen") for row in dev_rows])

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        rows = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_{split}_pos.jsonl")
        for budget in BUDGETS:
            name = f"{ADAPTIVE_PREFIX}_{BRANCH}_forced_{budget}_{split}_pos"
            write_dataset(name, [clone(row, budget, split) for row in rows])
            eval_names.append(name)

    audit = {
        "recipe": "explicit visible reasoning forms for none/light/standard/deep",
        "formal_train_count": len(train_rows),
        "budgets": BUDGETS,
        "rows_per_budget": len(train_rows),
        "total_train_rows": len(train),
        "form_tags": {
            "none": [],
            "light": ["SCHEMA_CHECK"],
            "standard": ["ROLE_TABLE"],
            "deep": ["ARGUMENT_VERIFY"],
        },
        "train_budget_counts": dict(Counter(row["meta"]["adaptive_reasoning_budget"] for row in train)),
        "training_recipe": {"learning_rate": 3.0e-6, "num_train_epochs": 3.0},
    }
    config = write_config(train_name, dev_name)
    note = write_note(train_name, dev_name, audit)
    e15.write_json(DATA_DIR / f"{train_name}.meta.json", {"audit": audit, "created_at": now_iso()})
    print(
        json.dumps(
            {
                "name": "e21a",
                "branch": BRANCH,
                "train_dataset": train_name,
                "dev_dataset": dev_name,
                "eval_datasets": eval_names,
                "config": config.as_posix(),
                "note": note.as_posix(),
                "audit": audit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
