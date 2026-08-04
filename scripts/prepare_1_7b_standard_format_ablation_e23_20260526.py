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

import prepare_1_7b_trigger_preserving_forms_e22_20260526 as e22  # noqa: E402


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
    "e23a": {
        "branch": "eventmentions_budget_e23a_argument_table",
        "title": "E23A Argument Table",
        "standard_tag": "ARGUMENT_TABLE",
        "objective": "Test whether standard reasoning should expose trigger-preserving argument slots directly.",
        "description": "standard uses argument table with event_type, trigger text/start/end, and argument role/text/start/end.",
    },
    "e23b": {
        "branch": "eventmentions_budget_e23b_schema_grounded_role_table",
        "title": "E23B Schema-Grounded Role Table",
        "standard_tag": "ROLE_TABLE",
        "objective": "Test whether schema-grounded role tables reduce role confusion.",
        "description": "standard uses trigger-preserving role table with allowed_roles and filled_roles.",
    },
    "e23c": {
        "branch": "eventmentions_budget_e23c_event_by_event_role_table",
        "title": "E23C Event-by-Event Role Table",
        "standard_tag": "EVENT_ROLE_TABLE",
        "objective": "Test whether event-by-event decomposition improves multi-event extraction stability.",
        "description": "standard uses event-indexed trigger-preserving role table blocks.",
    },
}

ROLE_SCHEMA = {
    "Business:Declare-Bankruptcy": ["Org"],
    "Business:End-Org": ["Org", "Place"],
    "Business:Merge-Org": ["Org"],
    "Business:Start-Org": ["Agent", "Org", "Place"],
    "Conflict:Attack": ["Attacker", "Target", "Instrument", "Place"],
    "Conflict:Demonstrate": ["Entity", "Place"],
    "Contact:Meet": ["Entity", "Place"],
    "Contact:Phone-Write": ["Entity", "Place"],
    "Justice:Acquit": ["Defendant", "Adjudicator", "Place"],
    "Justice:Appeal": ["Plaintiff", "Adjudicator", "Place"],
    "Justice:Arrest-Jail": ["Person", "Agent", "Place"],
    "Justice:Charge-Indict": ["Defendant", "Prosecutor", "Adjudicator", "Place"],
    "Justice:Convict": ["Defendant", "Adjudicator", "Place"],
    "Justice:Execute": ["Person", "Agent", "Place"],
    "Justice:Extradite": ["Person", "Agent", "Origin", "Destination"],
    "Justice:Fine": ["Entity", "Adjudicator", "Place"],
    "Justice:Pardon": ["Defendant", "Adjudicator", "Place"],
    "Justice:Release-Parole": ["Person", "Entity", "Place"],
    "Justice:Sentence": ["Defendant", "Adjudicator", "Place"],
    "Justice:Sue": ["Plaintiff", "Defendant", "Adjudicator", "Place"],
    "Justice:Trial-Hearing": ["Defendant", "Prosecutor", "Adjudicator", "Place"],
    "Life:Be-Born": ["Person", "Place"],
    "Life:Die": ["Victim", "Agent", "Instrument", "Place"],
    "Life:Divorce": ["Person", "Place"],
    "Life:Injure": ["Victim", "Agent", "Instrument", "Place"],
    "Life:Marry": ["Person", "Place"],
    "Movement:Transport-Artifact": ["Artifact", "Agent", "Origin", "Destination", "Vehicle", "Place"],
    "Movement:Transport-Person": ["Person", "Agent", "Origin", "Destination", "Vehicle", "Place"],
    "Personnel:Elect": ["Person", "Entity", "Place"],
    "Personnel:End-Position": ["Person", "Entity", "Place"],
    "Personnel:Nominate": ["Person", "Agent", "Place"],
    "Personnel:Start-Position": ["Person", "Entity", "Place"],
    "Transaction:Transfer-Money": ["Giver", "Recipient", "Beneficiary", "Money", "Place"],
    "Transaction:Transfer-Ownership": ["Buyer", "Seller", "Beneficiary", "Artifact", "Place"],
}


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def trigger_obj(event):
    trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
    return {"text": trigger.get("text"), "start": trigger.get("start"), "end": trigger.get("end")}


def argument_table(row):
    events = []
    for event_idx, event in enumerate(e22.gold_json(row).get("events", [])):
        if not isinstance(event, dict):
            continue
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
                    "status": "kept",
                }
            )
        events.append(
            {
                "event_id": event_idx,
                "event_type": event.get("event_type"),
                "trigger": trigger_obj(event),
                "arguments": args,
            }
        )
    return {"events": events}


def schema_grounded_role_table(row):
    events = []
    for event in e22.gold_json(row).get("events", []):
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        filled = []
        seen = set()
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            if not role or role in seen:
                continue
            seen.add(role)
            filled.append({"role": role, "filled": True})
        events.append(
            {
                "event_type": event_type,
                "trigger": trigger_obj(event),
                "allowed_roles": ROLE_SCHEMA.get(event_type, []),
                "filled_roles": filled,
            }
        )
    return {"events": events}


def event_by_event_role_table(row):
    events = []
    for event_idx, event in enumerate(e22.gold_json(row).get("events", [])):
        if not isinstance(event, dict):
            continue
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
        events.append(
            {
                "event_id": event_idx,
                "event_type": event.get("event_type"),
                "trigger": trigger_obj(event),
                "roles": roles,
            }
        )
    return {"events": events}


def standard_block(row, variant):
    if variant == "e23a":
        return "ARGUMENT_TABLE", argument_table(row)
    if variant == "e23b":
        return "ROLE_TABLE", schema_grounded_role_table(row)
    if variant == "e23c":
        return "EVENT_ROLE_TABLE", event_by_event_role_table(row)
    raise KeyError(variant)


def budget_instruction(budget, variant):
    spec = SPECS[variant]
    standard_text = {
        "e23a": "After the budget tag, output `<ARGUMENT_TABLE>{...}</ARGUMENT_TABLE>` with event_type, trigger text/start/end, and argument role/text/start/end.",
        "e23b": "After the budget tag, output `<ROLE_TABLE>{...}</ROLE_TABLE>` with event_type, trigger text/start/end, allowed_roles, and filled_roles.",
        "e23c": "After the budget tag, output `<EVENT_ROLE_TABLE>{...}</EVENT_ROLE_TABLE>` with event_id, event_type, trigger text/start/end, and filled roles.",
    }[variant]
    block = {
        "none": "Do not output an intermediate reasoning block.",
        "light": "After the budget tag, output `<SCHEMA_CHECK>{...}</SCHEMA_CHECK>` with event_types and trigger_count.",
        "standard": standard_text,
        "deep": "After the budget tag, output `<ARGUMENT_VERIFY>{...}</ARGUMENT_VERIFY>` with trigger text/start/end, kept arguments, and compact schema checks.",
    }[budget]
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
        f"Then output `<REASONING_BUDGET>{budget}</REASONING_BUDGET>`. {block} "
        "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        "Do not output text outside the requested tags. "
        "If no valid event is expressed by the candidate set, all JSON objects must use empty event/type lists as appropriate."
    )


def output_text(row, budget, variant):
    mentions = json.dumps(e22.event_mentions_from_gold(row), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(e22.gold_json(row), ensure_ascii=False, separators=(",", ":"))
    chunks = [f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>", f"<REASONING_BUDGET>{budget}</REASONING_BUDGET>"]
    if budget == "light":
        chunks.append(f"<SCHEMA_CHECK>{json.dumps(e22.schema_check(row), ensure_ascii=False, separators=(',', ':'))}</SCHEMA_CHECK>")
    elif budget == "standard":
        tag, block = standard_block(row, variant)
        chunks.append(f"<{tag}>{json.dumps(block, ensure_ascii=False, separators=(',', ':'))}</{tag}>")
    elif budget == "deep":
        chunks.append(f"<ARGUMENT_VERIFY>{json.dumps(e22.argument_verify(row), ensure_ascii=False, separators=(',', ':'))}</ARGUMENT_VERIFY>")
    chunks.append(f"<FINAL>{final}</FINAL>")
    return "\n".join(chunks)


def clone(row, budget, role, variant):
    spec = SPECS[variant]
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["instruction"] = budget_instruction(budget, variant)
    out["output"] = output_text(row, budget, variant)
    out["gold_output"] = json.dumps(e22.gold_json(row), ensure_ascii=False)
    stats = e22.event_stats(row)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "standard_format_ablation_e23",
            "adaptive_target_style": "standard_reason_format_ablation",
            "adaptive_dataset_role": role,
            "adaptive_reasoning_budget": budget,
            "adaptive_budget_label": budget,
            "e23_variant": variant,
            "e23_branch": spec["branch"],
            "e23_standard_tag": spec["standard_tag"],
            "e23_event_count": stats["event_count"],
            "e23_argument_count": stats["argument_count"],
            "e23_role_count": stats["role_count"],
        }
    )
    return out


def write_dataset(name, rows):
    file_name = f"{name}.jsonl"
    e22.e15.write_jsonl(DATA_DIR / file_name, rows)
    e22.e15.update_dataset_info(name, file_name)


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
  - reason-format-ablation
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

Run standard reason-format ablation: {spec['description']}

## Setup

- train dataset: `{train_name}`
- dev dataset: `{dev_name}`
- audit: `{json.dumps(audit, ensure_ascii=False, sort_keys=True)}`

## Commands

```bash
cd {REPO}
python3 scripts/prepare_1_7b_standard_format_ablation_e23_20260526.py
bash scripts/launch_1_7b_standard_format_ablation_e23_20260526.sh train {variant} 1
bash scripts/launch_1_7b_standard_format_ablation_e23_20260526.sh devpick {variant} 1
bash scripts/launch_1_7b_standard_format_ablation_e23_20260526.sh formal {variant} 0 1 2 3
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
    train_rows = e22.e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    rng = random.Random(20260526)
    train = []
    for row in train_rows:
        for budget in BUDGETS:
            train.append(clone(row, budget, "train", variant))
    rng.shuffle(train)
    train_name = f"{ADAPTIVE_PREFIX}_{branch}_train_pos"
    write_dataset(train_name, train)

    dev_rows = e22.e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_name = f"{ADAPTIVE_PREFIX}_{branch}_forced_standard_dev_seen_pos"
    write_dataset(dev_name, [clone(row, "standard", "dev_seen", variant) for row in dev_rows])

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        rows = e22.e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_{split}_pos.jsonl")
        for budget in BUDGETS:
            name = f"{ADAPTIVE_PREFIX}_{branch}_forced_{budget}_{split}_pos"
            write_dataset(name, [clone(row, budget, split, variant) for row in rows])
            eval_names.append(name)

    audit = {
        "recipe": f"E23 standard format ablation: {spec['description']}",
        "variant": variant,
        "branch": branch,
        "formal_train_count": len(train_rows),
        "budgets": BUDGETS,
        "rows_per_budget": len(train_rows),
        "total_train_rows": len(train),
        "standard_tag": spec["standard_tag"],
        "train_budget_counts": dict(Counter(row["meta"]["adaptive_reasoning_budget"] for row in train)),
        "training_recipe": {"learning_rate": 3.0e-6, "num_train_epochs": 3.0},
    }
    config = write_config(variant, train_name, dev_name)
    note = write_note(variant, train_name, dev_name, audit)
    e22.e15.write_json(DATA_DIR / f"{train_name}.meta.json", {"audit": audit, "created_at": now_iso()})
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
    payload = [prepare_variant(variant) for variant in ["e23a", "e23b", "e23c"]]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
