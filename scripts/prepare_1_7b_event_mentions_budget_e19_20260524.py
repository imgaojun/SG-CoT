import json
import random
import sys
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
SOURCE_BRANCH = "confrare10_heur10_typeonlylite"
OUTCOME_ROOT = REPO / "outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942"
TZ = timezone(timedelta(hours=8))

REASON_REPEAT = 5
LEARNING_RATE = 3.0e-6
NUM_EPOCHS = 4.0

VARIANTS = {
    "e19a": {
        "title": "E19A Event Mentions Mixed Budget",
        "branch": "eventmentions_budget_e19a_mixed_eventpos_r5",
        "standard_only": False,
        "objective": "Train a single-pass model that emits EVENT_MENTIONS, REASONING_BUDGET none/standard, then FINAL.",
    },
    "e19b": {
        "title": "E19B Event Mentions Standard Only",
        "branch": "eventmentions_budget_e19b_standardonly",
        "standard_only": True,
        "objective": "Ablate whether EVENT_MENTIONS before a standard reasoning budget improves reason-all generation.",
    },
}


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def key(row):
    return (row.get("meta") or {}).get("wnd_id")


def pred_key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("sample_key") or meta.get("doc_id")


def load_prediction_map(path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[pred_key(row)] = row
    return rows


def event_positive_keys():
    direct = load_prediction_map(OUTCOME_ROOT / "forced_direct/train/predictions.jsonl")
    reason = load_prediction_map(OUTCOME_ROOT / "forced_reason/train/predictions.jsonl")
    selected = {}
    for k in sorted(set(direct) & set(reason)):
        d = direct[k]
        r = reason[k]
        gains = {
            "argument_f1": r.get("argument_f1", 0.0) - d.get("argument_f1", 0.0),
            "event_f1": r.get("event_f1", 0.0) - d.get("event_f1", 0.0),
            "trigger_f1": r.get("trigger_f1", 0.0) - d.get("trigger_f1", 0.0),
        }
        gains["sum"] = gains["argument_f1"] + gains["event_f1"] + gains["trigger_f1"]
        if gains["event_f1"] > 0 and gains["sum"] > 0:
            selected[k] = gains
    return selected


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


def instruction(mode):
    budget_text = (
        "Use `<REASONING_BUDGET>none</REASONING_BUDGET>` when the event arguments can be filled directly. "
        "Use `<REASONING_BUDGET>standard</REASONING_BUDGET>` when internal schema-grounded reasoning is useful for roles or arguments. "
        if mode == "free"
        else f"Use `<REASONING_BUDGET>{mode}</REASONING_BUDGET>`."
    )
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans for all event mentions. "
        "Then output a reasoning budget. "
        f"{budget_text} "
        "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        "Do not output analysis or text outside the requested tags. "
        "If no valid event is expressed by the candidate set, both JSON objects must be {\"events\": []}."
    )


def output_text(row, budget):
    mentions = json.dumps(event_mentions_from_gold(row), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(gold_json(row), ensure_ascii=False, separators=(",", ":"))
    return f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>\n<REASONING_BUDGET>{budget}</REASONING_BUDGET>\n<FINAL>{final}</FINAL>"


def clone(row, branch, source, duplicate_idx=0):
    out = json.loads(json.dumps(row, ensure_ascii=False))
    meta = out.setdefault("meta", {})
    meta["adaptive_source"] = "event_mentions_budget_e19"
    meta["adaptive_target_style"] = "event_mentions_budget_final"
    meta["e19_branch"] = branch
    meta["e19_source"] = source
    meta["e19_duplicate_index"] = duplicate_idx
    return out


def train_row(row, branch, budget, source, duplicate_idx=0, gains=None):
    out = clone(row, branch, source, duplicate_idx)
    out["instruction"] = instruction(budget)
    out["output"] = output_text(row, budget)
    out["gold_output"] = json.dumps(gold_json(row), ensure_ascii=False)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_dataset_role": "train",
            "adaptive_reasoning_budget": budget,
            "adaptive_budget_label": budget,
        }
    )
    if gains:
        for metric, value in gains.items():
            meta[f"e19_outcome_gain_{metric}"] = value
    return out


def eval_row(row, branch, budget, role):
    out = clone(row, branch, f"eval_{budget}", 0)
    out["instruction"] = instruction(budget)
    out["output"] = output_text(row, budget)
    out["gold_output"] = json.dumps(gold_json(row), ensure_ascii=False)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_dataset_role": role,
            "adaptive_reasoning_budget": budget,
            "adaptive_budget_label": budget,
        }
    )
    return out


def build_train_rows(variant):
    rng = random.Random(20260524)
    branch = variant["branch"]
    standard_only = variant["standard_only"]
    formal_train = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    formal_by_key = {key(row): row for row in formal_train}
    positives = event_positive_keys()

    rows = []
    if standard_only:
        rows = [train_row(row, branch, "standard", "all_train_standard", 0) for row in formal_train]
    else:
        rows.extend(train_row(row, branch, "none", "all_train_none_anchor", 0) for row in formal_train)
        for k, gains in positives.items():
            if k not in formal_by_key:
                continue
            for dup in range(REASON_REPEAT):
                rows.append(train_row(formal_by_key[k], branch, "standard", "event_positive_standard", dup, gains))
    rng.shuffle(rows)
    audit = {
        "source_branch": SOURCE_BRANCH,
        "outcome_root": OUTCOME_ROOT.as_posix(),
        "selection_rule": "standard iff reason_gain event_f1 > 0 and summed A/E/T gain > 0; none otherwise",
        "standard_only": standard_only,
        "formal_train_count": len(formal_train),
        "event_positive_count": len(positives),
        "reason_repeat": 0 if standard_only else REASON_REPEAT,
        "none_rows": 0 if standard_only else len(formal_train),
        "standard_rows": len(rows) if standard_only else len(positives) * REASON_REPEAT,
        "total_count": len(rows),
        "training_recipe": {"learning_rate": LEARNING_RATE, "num_train_epochs": NUM_EPOCHS},
    }
    return rows, audit


def write_dataset(name, rows):
    file_name = f"{name}.jsonl"
    e15.write_jsonl(DATA_DIR / file_name, rows)
    e15.update_dataset_info(name, file_name)
    return file_name


def write_config(variant, train_name, dev_name):
    branch = variant["branch"]
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
        "max_samples": 10000,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "packing": False,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": 0.05,
        "bf16": True,
        "val_size": 0.0,
        "eval_steps": 10,
        "do_eval": True,
        "save_only_model": True,
        "num_train_epochs": NUM_EPOCHS,
        "load_best_model_at_end": False,
        "deepspeed": "/workspace/project/configs/deepspeed/zero2_optimizer_offload_cpu.json",
    }
    write_yaml(out_config, config)
    return out_config.as_posix()


def write_note(name, variant, train_name, dev_name, audit):
    timestamp = now_iso()
    branch = variant["branch"]
    exp_id = f"2026-05-24_stage2_1_7b_{branch}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    out_dir = REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full"
    config_path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    body = f"""---
id: {exp_id}
title: Stage2 1.7B {variant['title']}
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
objective: {variant['objective']}
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
---

# Stage2 1.7B {variant['title']}

## Goal

Train a single-pass model that emits event mentions first, then a reasoning budget, then the final full event list.

## Setup

- train dataset: `{train_name}`
- dev dataset: `{dev_name}`
- audit: `{json.dumps(audit, ensure_ascii=False, sort_keys=True)}`

## Commands

```bash
cd {REPO}
python3 scripts/prepare_1_7b_event_mentions_budget_e19_20260524.py
bash scripts/launch_1_7b_event_mentions_budget_e19_20260524.sh train {name} 0
bash scripts/launch_1_7b_event_mentions_budget_e19_20260524.sh devpick {name} 0
bash scripts/launch_1_7b_event_mentions_budget_e19_20260524.sh formal {name} 0 1 2 3
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
- run checkpoint selection and formal forced-budget evaluation.
"""
    note.write_text(body, encoding="utf-8")
    return note


def build_variant(name, variant):
    branch = variant["branch"]
    train_rows, audit = build_train_rows(variant)
    train_name = f"{ADAPTIVE_PREFIX}_{branch}_train_pos"
    dev_name = f"{ADAPTIVE_PREFIX}_{branch}_dev_seen_pos"
    write_dataset(train_name, train_rows)

    formal_dev = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_rows = [eval_row(row, branch, "none" if not variant["standard_only"] else "standard", "dev_seen") for row in formal_dev]
    write_dataset(dev_name, dev_rows)

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        formal_rows = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_{split}_pos.jsonl")
        budgets = ["none", "standard"] if not variant["standard_only"] else ["standard"]
        for budget in budgets:
            target_name = f"{ADAPTIVE_PREFIX}_{branch}_forced_{budget}_{split}_pos"
            rows = [eval_row(row, branch, budget, split) for row in formal_rows]
            write_dataset(target_name, rows)
            eval_names.append(target_name)

    config = write_config(variant, train_name, dev_name)
    note = write_note(name, variant, train_name, dev_name, audit)
    e15.write_json(DATA_DIR / f"{train_name}.meta.json", {"audit": audit, "created_at": now_iso()})
    return {
        "name": name,
        "branch": branch,
        "train_dataset": train_name,
        "dev_dataset": dev_name,
        "eval_datasets": eval_names,
        "config": config,
        "note": note.as_posix(),
        "audit": audit,
    }


def main():
    results = [build_variant(name, variant) for name, variant in VARIANTS.items()]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
