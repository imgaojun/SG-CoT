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
BUDGETS = ["none", "light", "standard", "deep"]

SPECS = {
    "e26a": {
        "branch": "eventmentions_budget_e26a_mention_schema",
        "title": "E26A Mention + Schema Planning",
        "objective": "Isolate whether compact schema/role planning is the useful step behind E21A.",
        "description": "standard uses <SCHEMA_PLAN> with event_type, trigger text, and likely role names only; no argument texts or offsets.",
        "target_style": "mention_schema",
    },
    "e26b": {
        "branch": "eventmentions_budget_e26b_mention_schema_groundlite",
        "title": "E26B Mention + Schema + Lightweight Grounding",
        "objective": "Test whether short role-to-text grounding improves Argument without E23-style copy burden.",
        "description": "standard adds <GROUND_HINT> with compact role/text pairs and no offsets.",
        "target_style": "mention_schema_groundlite",
    },
    "e26c": {
        "branch": "eventmentions_budget_e26c_mention_schema_verify",
        "title": "E26C Mention + Schema + Verification",
        "objective": "Test whether compact consistency verification protects Event full-frame quality.",
        "description": "standard adds <VERIFY> with event count, role coverage, and compact consistency checks.",
        "target_style": "mention_schema_verify",
    },
}


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def compact_events(row):
    events = []
    for event in e21.gold_json(row).get("events", []):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        roles = []
        seen = set()
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            if role and role not in seen:
                roles.append(role)
                seen.add(role)
        events.append({"event_type": event.get("event_type"), "trigger": trigger.get("text"), "roles": roles})
    return events


def schema_plan(row):
    return {"events": compact_events(row)}


def ground_hint(row):
    events = []
    for event in e21.gold_json(row).get("events", []):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        seen = set()
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            text = arg.get("text")
            key = (role, text)
            if not role or not text or key in seen:
                continue
            seen.add(key)
            args.append({"role": role, "text": text})
        events.append({"event_type": event.get("event_type"), "trigger": trigger.get("text"), "arguments": args})
    return {"events": events}


def verify_hint(row):
    stats = e21.event_stats(row)
    events = []
    for event in e21.gold_json(row).get("events", []):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        roles = []
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict) and arg.get("role") and arg.get("role") not in roles:
                roles.append(arg.get("role"))
        checks = []
        if event.get("event_type"):
            checks.append("type_present")
        if trigger.get("text"):
            checks.append("trigger_present")
        if roles:
            checks.append("roles_filled")
        events.append({"event_type": event.get("event_type"), "trigger": trigger.get("text"), "filled_roles": roles, "checks": checks})
    return {"event_count": stats["event_count"], "argument_count": stats["argument_count"], "events": events}


def hint_for_budget(row, budget, variant):
    if budget == "none":
        return None, None
    if budget == "light":
        return "SCHEMA_PLAN", {"event_types": e21.schema_check(row).get("event_types", [])}
    if budget == "deep":
        return "VERIFY", verify_hint(row)
    if variant == "e26a":
        return "SCHEMA_PLAN", schema_plan(row)
    if variant == "e26b":
        return "GROUND_HINT", ground_hint(row)
    if variant == "e26c":
        return "VERIFY", verify_hint(row)
    raise ValueError(f"unknown variant: {variant}")


def budget_instruction(budget, variant):
    if budget == "none":
        block = "Do not output an intermediate reasoning block."
    elif budget == "light":
        block = "After the budget tag, output `<SCHEMA_PLAN>{...}</SCHEMA_PLAN>` with event types only."
    elif budget == "deep":
        block = "After the budget tag, output `<VERIFY>{...}</VERIFY>` with compact event-count and consistency checks."
    elif variant == "e26a":
        block = "After the budget tag, output `<SCHEMA_PLAN>{...}</SCHEMA_PLAN>` with event_type, trigger text, and likely role names only."
    elif variant == "e26b":
        block = "After the budget tag, output `<GROUND_HINT>{...}</GROUND_HINT>` with compact role/text pairs only, without offsets."
    elif variant == "e26c":
        block = "After the budget tag, output `<VERIFY>{...}</VERIFY>` with event count, filled roles, and consistency checks."
    else:
        raise ValueError(f"unknown variant: {variant}")
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
        f"Then output `<REASONING_BUDGET>{budget}</REASONING_BUDGET>`. {block} "
        "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        "Intermediate steps are compact planning hints, not complete duplicate annotations. "
        "Do not output text outside the requested tags. "
        "If no valid event is expressed by the candidate set, all JSON objects must use empty event/type lists as appropriate."
    )


def output_text(row, budget, variant):
    mentions = json.dumps(e21.event_mentions_from_gold(row), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(e21.gold_json(row), ensure_ascii=False, separators=(",", ":"))
    chunks = [f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>", f"<REASONING_BUDGET>{budget}</REASONING_BUDGET>"]
    tag, hint = hint_for_budget(row, budget, variant)
    if tag and hint is not None:
        chunks.append(f"<{tag}>{json.dumps(hint, ensure_ascii=False, separators=(',', ':'))}</{tag}>")
    chunks.append(f"<FINAL>{final}</FINAL>")
    return "\n".join(chunks)


def clone(row, budget, role, variant):
    spec = SPECS[variant]
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["instruction"] = budget_instruction(budget, variant)
    out["output"] = output_text(row, budget, variant)
    out["gold_output"] = json.dumps(e21.gold_json(row), ensure_ascii=False)
    stats = e21.event_stats(row)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "step_decomposed_reasoning_e26",
            "adaptive_target_style": spec["target_style"],
            "adaptive_dataset_role": role,
            "adaptive_reasoning_budget": budget,
            "adaptive_budget_label": budget,
            "e26_variant": variant,
            "e26_branch": spec["branch"],
            "e26_event_count": stats["event_count"],
            "e26_argument_count": stats["argument_count"],
            "e26_role_count": stats["role_count"],
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
  - forced-reasoning
  - step-decomposition
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
python3 scripts/prepare_1_7b_step_decomposed_reasoning_e26_20260527.py
bash scripts/launch_1_7b_step_decomposed_reasoning_e26_20260527.sh train {variant} 2
bash scripts/launch_1_7b_step_decomposed_reasoning_e26_20260527.sh devpick {variant} 2
bash scripts/launch_1_7b_step_decomposed_reasoning_e26_20260527.sh formal {variant} 2 3 4 7
```

## Run Log

### {log_stamp}

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


def prepare_variant(variant):
    spec = SPECS[variant]
    branch = spec["branch"]
    train_rows = e21.e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    rng = random.Random(20260527)
    train = []
    for row in train_rows:
        for budget in BUDGETS:
            train.append(clone(row, budget, "train", variant))
    rng.shuffle(train)
    train_name = f"{ADAPTIVE_PREFIX}_{branch}_train_pos"
    write_dataset(train_name, train)

    dev_rows = e21.e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_name = f"{ADAPTIVE_PREFIX}_{branch}_forced_standard_dev_seen_pos"
    write_dataset(dev_name, [clone(row, "standard", "dev_seen", variant) for row in dev_rows])

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        rows = e21.e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_{split}_pos.jsonl")
        for budget in BUDGETS:
            name = f"{ADAPTIVE_PREFIX}_{branch}_forced_{budget}_{split}_pos"
            write_dataset(name, [clone(row, budget, split, variant) for row in rows])
            eval_names.append(name)

    audit = {
        "recipe": spec["description"],
        "variant": variant,
        "branch": branch,
        "formal_train_count": len(train_rows),
        "budgets": BUDGETS,
        "rows_per_budget": len(train_rows),
        "total_train_rows": len(train),
        "train_budget_counts": dict(Counter(row["meta"]["adaptive_reasoning_budget"] for row in train)),
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
    payload = [prepare_variant(variant) for variant in ["e26a", "e26b", "e26c"]]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
