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
    "e24a": {
        "branch": "eventmentions_budget_e24a_lite_standard",
        "title": "E24A Lightweight Standard Reason",
        "objective": "Test whether a shorter E21-style standard scaffold keeps Argument/Event gains while reducing Trigger interference.",
        "description": "standard uses a compact <REASON_HINT> block with only event types, trigger texts, and role names.",
        "target_style": "lite_standard",
    },
    "e25a": {
        "branch": "eventmentions_budget_e25a_final_first_weakreason",
        "title": "E25A Final-First Weak Reason",
        "objective": "Approximate final-loss dominance by putting FINAL before a weak reasoning hint for non-none budgets.",
        "description": "non-none budgets emit FINAL before a minimal <REASON_HINT> block, reducing pressure to copy from the reasoning block into FINAL.",
        "target_style": "final_first_weakreason",
    },
}


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def role_names(row):
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
        events.append(
            {
                "event_type": event.get("event_type"),
                "trigger": trigger.get("text"),
                "roles": roles,
            }
        )
    return {"events": events}


def event_type_hint(row):
    event_types = []
    for event in e21.gold_json(row).get("events", []):
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        if event_type and event_type not in event_types:
            event_types.append(event_type)
    return {"event_types": event_types}


def hint_for_budget(row, budget):
    if budget == "light":
        return e21.schema_check(row)
    if budget == "standard":
        return role_names(row)
    if budget == "deep":
        return event_type_hint(row)
    return None


def hint_tag(budget):
    if budget == "none":
        return None
    return "REASON_HINT"


def budget_instruction(budget, variant):
    spec = SPECS[variant]
    if variant == "e24a":
        block = {
            "none": "Do not output an intermediate reasoning block.",
            "light": "After the budget tag, output `<REASON_HINT>{...}</REASON_HINT>` with event_types and trigger_count.",
            "standard": "After the budget tag, output `<REASON_HINT>{...}</REASON_HINT>` with compact event_type, trigger text, and role names only.",
            "deep": "After the budget tag, output `<REASON_HINT>{...}</REASON_HINT>` with event_types only.",
        }[budget]
        return (
            "You are doing event extraction. Use only the provided candidate event types and schema cards. "
            "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
            f"Then output `<REASONING_BUDGET>{budget}</REASONING_BUDGET>`. {block} "
            "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
            "Do not copy offsets from the reasoning hint; FINAL must use the document offsets. "
            "Do not output text outside the requested tags. "
            "If no valid event is expressed by the candidate set, all JSON objects must use empty event/type lists as appropriate."
        )
    block = (
        "Do not output a reasoning hint."
        if budget == "none"
        else "After FINAL, output `<REASON_HINT>{...}</REASON_HINT>` as a minimal audit hint only."
    )
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
        f"Then output `<REASONING_BUDGET>{budget}</REASONING_BUDGET>`. "
        "Next output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        f"{block} "
        "Do not output text outside the requested tags. "
        "If no valid event is expressed by the candidate set, all JSON objects must use empty event/type lists as appropriate."
    )


def output_text(row, budget, variant):
    mentions = json.dumps(e21.event_mentions_from_gold(row), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(e21.gold_json(row), ensure_ascii=False, separators=(",", ":"))
    chunks = [f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>", f"<REASONING_BUDGET>{budget}</REASONING_BUDGET>"]
    tag = hint_tag(budget)
    hint = hint_for_budget(row, budget)
    if variant == "e24a":
        if tag and hint is not None:
            chunks.append(f"<{tag}>{json.dumps(hint, ensure_ascii=False, separators=(',', ':'))}</{tag}>")
        chunks.append(f"<FINAL>{final}</FINAL>")
    else:
        chunks.append(f"<FINAL>{final}</FINAL>")
        if tag and hint is not None:
            chunks.append(f"<{tag}>{json.dumps(hint, ensure_ascii=False, separators=(',', ':'))}</{tag}>")
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
            "adaptive_source": "reason_lightweight_e24_e25",
            "adaptive_target_style": spec["target_style"],
            "adaptive_dataset_role": role,
            "adaptive_reasoning_budget": budget,
            "adaptive_budget_label": budget,
            "e24_e25_variant": variant,
            "e24_e25_branch": spec["branch"],
            "e24_e25_event_count": stats["event_count"],
            "e24_e25_argument_count": stats["argument_count"],
            "e24_e25_role_count": stats["role_count"],
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
    exp_id = f"2026-05-26_stage2_1_7b_{branch}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    out_dir = REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full"
    config_path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
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
  - event-mentions
  - reasoning-budget
  - lightweight-reasoning
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
python3 scripts/prepare_1_7b_reason_lightweight_e24_e25_20260526.py
bash scripts/launch_1_7b_reason_lightweight_e24_e25_20260526.sh train {variant} 0
bash scripts/launch_1_7b_reason_lightweight_e24_e25_20260526.sh devpick {variant} 0
bash scripts/launch_1_7b_reason_lightweight_e24_e25_20260526.sh formal {variant} 0 1 2 3
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


def prepare_variant(variant):
    spec = SPECS[variant]
    branch = spec["branch"]
    train_rows = e21.e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    rng = random.Random(20260526)
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
        "middle_tag": "REASON_HINT",
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
    payload = [prepare_variant(variant) for variant in ["e24a", "e25a"]]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
