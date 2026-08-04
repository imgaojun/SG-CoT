import json
import random
import sys
from collections import Counter, defaultdict
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
OUTCOME_ROOT = REPO / "outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942"
TZ = timezone(timedelta(hours=8))

E20A_BRANCH = "eventmentions_budget_e20a_weighted_standard_cxop"
E20B_BRANCH = "eventmentions_budget_e20b_multibudget_profile"
BUDGETS = ["none", "light", "standard", "deep"]


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


def outcome_gains():
    direct = load_prediction_map(OUTCOME_ROOT / "forced_direct/train/predictions.jsonl")
    reason = load_prediction_map(OUTCOME_ROOT / "forced_reason/train/predictions.jsonl")
    gains = {}
    for k in sorted(set(direct) & set(reason)):
        d = direct[k]
        r = reason[k]
        item = {
            "argument_f1": r.get("argument_f1", 0.0) - d.get("argument_f1", 0.0),
            "event_f1": r.get("event_f1", 0.0) - d.get("event_f1", 0.0),
            "trigger_f1": r.get("trigger_f1", 0.0) - d.get("trigger_f1", 0.0),
            "direct_argument_f1": d.get("argument_f1", 0.0),
            "direct_event_f1": d.get("event_f1", 0.0),
            "direct_trigger_f1": d.get("trigger_f1", 0.0),
        }
        item["sum"] = item["argument_f1"] + item["event_f1"] + item["trigger_f1"]
        gains[k] = item
    return gains


def gold_json(row):
    if "gold_output" in row:
        return json.loads(row["gold_output"])
    return json.loads(row["output"])


def event_stats(row):
    payload = gold_json(row)
    events = [event for event in payload.get("events", []) if isinstance(event, dict)]
    event_types = []
    arg_count = 0
    role_count = 0
    for event in events:
        event_type = event.get("event_type")
        if event_type:
            event_types.append(event_type)
        args = event.get("arguments") or []
        arg_count += len(args)
        role_count += len({arg.get("role") for arg in args if isinstance(arg, dict) and arg.get("role")})
    return {
        "event_count": len(events),
        "argument_count": arg_count,
        "role_count": role_count,
        "event_types": event_types,
    }


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


def budget_instruction(mode):
    descriptions = {
        "none": "Use `<REASONING_BUDGET>none</REASONING_BUDGET>` and fill the final event list directly from the trigger evidence.",
        "light": "Use `<REASONING_BUDGET>light</REASONING_BUDGET>` for a compact schema check before filling roles.",
        "standard": "Use `<REASONING_BUDGET>standard</REASONING_BUDGET>` for schema-grounded reasoning over triggers, roles, and argument spans.",
        "deep": "Use `<REASONING_BUDGET>deep</REASONING_BUDGET>` for the hardest cases: multiple events, dense arguments, rare types, or role conflicts.",
    }
    if mode == "free":
        budget_text = "Choose exactly one reasoning budget from `none`, `light`, `standard`, and `deep`."
    else:
        budget_text = descriptions[mode]
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans for all event mentions. "
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
    meta["adaptive_source"] = "reason_budget_e20"
    meta["adaptive_target_style"] = "event_mentions_budget_final"
    meta["e20_branch"] = branch
    meta["e20_source"] = source
    meta["e20_duplicate_index"] = duplicate_idx
    return out


def make_row(row, branch, budget, source, role, duplicate_idx=0, extra_meta=None):
    out = clone(row, branch, source, duplicate_idx)
    out["instruction"] = budget_instruction(budget)
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
    if extra_meta:
        meta.update(extra_meta)
    return out


def rare_types(train_rows, max_count=20):
    counts = Counter()
    for row in train_rows:
        counts.update(event_stats(row)["event_types"])
    return {event_type for event_type, count in counts.items() if count <= max_count}


def e20a_repeat_plan(row, gains, rare_type_set):
    stats = event_stats(row)
    k = key(row)
    g = gains.get(k, {})
    reasons = ["base_standard"]
    repeat = 1
    if g and g.get("sum", 0.0) > 0 and g.get("event_f1", 0.0) >= 0:
        repeat += 4
        reasons.append("outcome_positive_event_safe")
    if stats["event_count"] >= 2 or stats["argument_count"] >= 4 or stats["role_count"] >= 3:
        repeat += 2
        reasons.append("complex_output")
    if set(stats["event_types"]) & rare_type_set:
        repeat += 2
        reasons.append("rare_type")
    if g and (g.get("direct_argument_f1", 1.0) < 1.0 or g.get("direct_event_f1", 1.0) < 1.0):
        repeat += 1
        reasons.append("direct_error")
    repeat = min(repeat, 8)
    return repeat, reasons, stats, g


def build_e20a_train(train_rows):
    rng = random.Random(20260524)
    gains = outcome_gains()
    rare_type_set = rare_types(train_rows)
    rows = []
    repeat_hist = Counter()
    reason_hist = Counter()
    selected_counts = defaultdict(int)
    for row in train_rows:
        repeat, reasons, stats, g = e20a_repeat_plan(row, gains, rare_type_set)
        repeat_hist[repeat] += 1
        for reason in reasons:
            reason_hist[reason] += 1
        for dup in range(repeat):
            rows.append(
                make_row(
                    row,
                    E20A_BRANCH,
                    "standard",
                    "weighted_standard",
                    "train",
                    dup,
                    {
                        "e20_weight_repeat": repeat,
                        "e20_weight_reasons": reasons,
                        "e20_event_count": stats["event_count"],
                        "e20_argument_count": stats["argument_count"],
                        "e20_role_count": stats["role_count"],
                        **{f"e20_outcome_gain_{name}": value for name, value in g.items()},
                    },
                )
            )
        for reason in reasons:
            selected_counts[reason] += repeat
    rng.shuffle(rows)
    audit = {
        "recipe": "weighted standard-only reason expert via repeated rows",
        "formal_train_count": len(train_rows),
        "total_train_rows": len(rows),
        "repeat_histogram": dict(sorted(repeat_hist.items())),
        "row_reason_counts": dict(sorted(reason_hist.items())),
        "weighted_reason_counts": dict(sorted(selected_counts.items())),
        "rare_type_threshold": 20,
        "rare_types": sorted(rare_type_set),
        "outcome_root": OUTCOME_ROOT.as_posix(),
        "training_recipe": {"learning_rate": 3.0e-6, "num_train_epochs": 4.0},
    }
    return rows, audit


def build_e20b_train(train_rows):
    rng = random.Random(20260524)
    rows = []
    for row in train_rows:
        stats = event_stats(row)
        for budget in BUDGETS:
            rows.append(
                make_row(
                    row,
                    E20B_BRANCH,
                    budget,
                    "multibudget_equal",
                    "train",
                    0,
                    {
                        "e20_event_count": stats["event_count"],
                        "e20_argument_count": stats["argument_count"],
                        "e20_role_count": stats["role_count"],
                    },
                )
            )
    rng.shuffle(rows)
    audit = {
        "recipe": "equal multi-budget forced profiling dataset",
        "formal_train_count": len(train_rows),
        "budgets": BUDGETS,
        "rows_per_budget": len(train_rows),
        "total_train_rows": len(rows),
        "training_recipe": {"learning_rate": 3.0e-6, "num_train_epochs": 3.0},
    }
    return rows, audit


def write_dataset(name, rows):
    file_name = f"{name}.jsonl"
    e15.write_jsonl(DATA_DIR / file_name, rows)
    e15.update_dataset_info(name, file_name)
    return file_name


def write_config(branch, train_name, dev_name, epochs):
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
        "num_train_epochs": epochs,
        "load_best_model_at_end": False,
        "deepspeed": "/workspace/project/configs/deepspeed/zero2_optimizer_offload_cpu.json",
    }
    write_yaml(out_config, config)
    return out_config


def write_note(exp_id, title, branch, train_name, dev_name, audit, commands):
    timestamp = now_iso()
    out_dir = REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full"
    config_path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    command_block = "\n".join(commands)
    body = f"""---
id: {exp_id}
title: {title}
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
  - reason-expert
objective: Improve or profile forced reasoning budgets before training a selector.
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

# {title}

## Goal

Train and evaluate event-mentions + reasoning-budget generation without adding an external pipeline.

## Setup

- train dataset: `{train_name}`
- dev dataset: `{dev_name}`
- audit: `{json.dumps(audit, ensure_ascii=False, sort_keys=True)}`

## Commands

```bash
cd {REPO}
python3 scripts/prepare_1_7b_reason_budget_e20_20260524.py
{command_block}
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


def build_variant(name, branch, train_rows, budgets, train_builder, epochs, title):
    e20_train, audit = train_builder(train_rows)
    train_name = f"{ADAPTIVE_PREFIX}_{branch}_train_pos"
    dev_name = f"{ADAPTIVE_PREFIX}_{branch}_forced_standard_dev_seen_pos"
    write_dataset(train_name, e20_train)

    formal_dev = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_dev_seen_pos.jsonl")
    write_dataset(dev_name, [make_row(row, branch, "standard", "eval_standard", "dev_seen") for row in formal_dev])

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        formal_rows = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_{split}_pos.jsonl")
        for budget in budgets:
            target_name = f"{ADAPTIVE_PREFIX}_{branch}_forced_{budget}_{split}_pos"
            write_dataset(target_name, [make_row(row, branch, budget, f"eval_{budget}", split) for row in formal_rows])
            eval_names.append(target_name)

    config = write_config(branch, train_name, dev_name, epochs)
    e15.write_json(DATA_DIR / f"{train_name}.meta.json", {"audit": audit, "created_at": now_iso()})
    exp_id = f"2026-05-24_stage2_1_7b_{branch}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    commands = [
        f"bash scripts/launch_1_7b_reason_budget_e20_20260524.sh train {name} 0",
        f"bash scripts/launch_1_7b_reason_budget_e20_20260524.sh devpick {name} 0",
        f"bash scripts/launch_1_7b_reason_budget_e20_20260524.sh formal {name} 0 1 2 3",
    ]
    note = write_note(exp_id, title, branch, train_name, dev_name, audit, commands)
    return {
        "name": name,
        "branch": branch,
        "train_dataset": train_name,
        "dev_dataset": dev_name,
        "eval_datasets": eval_names,
        "config": config.as_posix(),
        "note": note.as_posix(),
        "audit": audit,
    }


def main():
    train_rows = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    results = [
        build_variant(
            "e20a",
            E20A_BRANCH,
            train_rows,
            ["none", "standard"],
            build_e20a_train,
            4.0,
            "Stage2 1.7B E20A Weighted Standard Reason Expert",
        ),
        build_variant(
            "e20b",
            E20B_BRANCH,
            train_rows,
            BUDGETS,
            build_e20b_train,
            3.0,
            "Stage2 1.7B E20B Multi-Budget Forced Profiling",
        ),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
