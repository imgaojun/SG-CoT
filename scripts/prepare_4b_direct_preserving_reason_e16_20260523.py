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
SOURCE_BRANCH = "confrare10_heur10_typeonlylite"
RUN_PREFIX = "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
WARM_START = (
    "/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/"
    "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/checkpoint-2064"
)
TZ = timezone(timedelta(hours=8))

REASON_REPEAT = 12
RETENTION_REPEAT = 6
LEARNING_RATE = 1.0e-6
NUM_EPOCHS = 3.0

VARIANTS = {
    "e16a_noreasonblock_directpreserve": {
        "title": "E16A No Reason Block Direct Preserve",
        "branch": "confrare10_typeonlylite_reasonfmt_e16a_noreasonblock_directpreserve",
        "style": "no_reason_block",
        "source_e15": "e15a_noreasonblock",
    },
    "e16c_finalfirst_directpreserve": {
        "title": "E16C Final First Direct Preserve",
        "branch": "confrare10_typeonlylite_reasonfmt_e16c_finalfirst_directpreserve",
        "style": "final_first_reason",
        "source_e15": "e15c_finalfirst",
    },
}


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def route_label(row):
    return (row.get("meta") or {}).get("adaptive_route_label", "direct")


def key(row):
    return (row.get("meta") or {}).get("wnd_id")


def mark_e16(row, branch, source):
    meta = row.setdefault("meta", {})
    meta["e16_source"] = source
    meta["e16_branch"] = branch
    meta["adaptive_source"] = "direct_preserving_reason_e16"
    return row


def build_train_rows(branch, style):
    rng = random.Random(20260523)
    adaptive = e15.load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_train_pos.jsonl")
    formal_direct = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    direct_by_key = {key(row): row for row in formal_direct}
    reason_rows = [row for row in adaptive if route_label(row) == "reason"]
    direct_rows = [row for row in adaptive if route_label(row) != "reason"]

    reason_part = []
    for row in reason_rows:
        for dup in range(REASON_REPEAT):
            reason_part.append(mark_e16(e15.reason_row(row, branch, style, "reason_oversample", dup), branch, "reason_oversample"))

    anchor_part = []
    for row in direct_rows:
        anchor_part.append(mark_e16(e15.direct_anchor_row(direct_by_key[key(row)], branch, "direct_anchor", 0), branch, "direct_anchor"))

    retention_part = []
    for dup in range(RETENTION_REPEAT):
        for row in reason_rows:
            retention_part.append(
                mark_e16(
                    e15.direct_anchor_row(direct_by_key[key(row)], branch, "reason_window_direct_retention", dup),
                    branch,
                    "reason_window_direct_retention",
                )
            )

    rows = reason_part + anchor_part + retention_part
    rng.shuffle(rows)
    audit = {
        "source_branch": SOURCE_BRANCH,
        "style": style,
        "adaptive_train_count": len(adaptive),
        "source_reason_count": len(reason_rows),
        "source_direct_count": len(direct_rows),
        "reason_repeat": REASON_REPEAT,
        "reason_rows_after_repeat": len(reason_part),
        "direct_anchor_rows": len(anchor_part),
        "retention_repeat": RETENTION_REPEAT,
        "retention_rows": len(retention_part),
        "total_count": len(rows),
        "route_label_counts": {"reason": len(reason_part), "direct": len(anchor_part) + len(retention_part)},
        "training_recipe": {
            "learning_rate": LEARNING_RATE,
            "num_train_epochs": NUM_EPOCHS,
            "intent": "preserve direct backbone while keeping E15 low-interference reason format",
        },
    }
    return rows, audit


def write_dataset(name, rows):
    file_name = f"{name}.jsonl"
    e15.write_jsonl(DATA_DIR / file_name, rows)
    e15.update_dataset_info(name, file_name)
    return file_name


def write_config(branch, train_name, dev_name):
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


def write_note(variant_id, spec, train_name, dev_name, audit):
    timestamp = now_iso()
    exp_id = f"2026-05-23_stage2_4b_direct_preserving_reason_e16_{variant_id}_richere_split1_oracle_mixed_noise_qwen3_4b"
    out_dir = REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{spec['branch']}_full"
    config_path = CONFIG_DIR / f"{RUN_PREFIX}_{spec['branch']}_full_stepmatch.yaml"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    body = f"""---
id: {exp_id}
title: Stage2 4B Direct Preserving Reason {spec['title']}
kind: experiment
status: planned
created_at: {timestamp}
updated_at: {timestamp}
owners:
  - codex
tags:
  - stage2
  - qwen3-4b
  - reason-expert
  - direct-preservation
objective: Preserve the direct extraction backbone while keeping the low-interference E15 reason format.
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
  base_model: Qwen3-4B-Instruct
  warm_start: {WARM_START}
  branch: {spec['branch']}
  style: {spec['style']}
  source_e15: {spec['source_e15']}
---

# Stage2 4B Direct Preserving Reason {spec['title']}

## Goal

Test whether a lower-pressure, direct-preserving recipe can keep E15's low reason interference while recovering direct extraction strength.

## Setup

- train dataset: `{train_name}`
- dev dataset: `{dev_name}`
- audit: `{json.dumps(audit, ensure_ascii=False, sort_keys=True)}`

## Commands

```bash
cd {REPO}
python3 scripts/prepare_4b_direct_preserving_reason_e16_20260523.py
bash scripts/launch_4b_direct_preserving_reason_e16_20260523.sh train {variant_id} <gpu>
bash scripts/launch_4b_direct_preserving_reason_e16_20260523.sh devpick {variant_id} <gpu>
bash scripts/launch_4b_direct_preserving_reason_e16_20260523.sh formal {variant_id} <gpu0> <gpu1> <gpu2> <gpu3>
python3 scripts/summarize_4b_direct_preserving_reason_e16_20260523.py
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
- run devpick and formal forced-direct/forced-reason evaluation.
"""
    note.write_text(body, encoding="utf-8")
    return note


def build_variant(variant_id, spec):
    branch = spec["branch"]
    style = spec["style"]
    train_rows, audit = build_train_rows(branch, style)

    dev_source = e15.load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_dev_seen_pos.jsonl")
    dev_rows = [
        e15.reason_row(row, branch, style, "dev_seen_source", 0, "dev_seen", "free_route")
        if route_label(row) == "reason"
        else e15.clone(row, branch, "dev_seen_source", 0)
        for row in dev_source
    ]
    for row in dev_rows:
        row["instruction"] = e15.free_route_instruction(style)
        mark_e16(row, branch, "dev_seen_source")
        row.setdefault("meta", {})["adaptive_dataset_role"] = "dev_seen"
        row.setdefault("meta", {})["adaptive_target_style"] = style

    train_name = f"{ADAPTIVE_PREFIX}_{branch}_train_pos"
    dev_name = f"{ADAPTIVE_PREFIX}_{branch}_dev_seen_pos"
    write_dataset(train_name, train_rows)
    write_dataset(dev_name, dev_rows)
    e15.write_json(DATA_DIR / f"{train_name}.meta.json", {"audit": audit, "created_at": now_iso()})
    e15.write_json(DATA_DIR / f"{dev_name}.meta.json", {"num_examples": len(dev_rows), "created_at": now_iso()})

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        direct_source = e15.load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_forced_direct_{split}_pos.jsonl")
        reason_source = e15.load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_forced_reason_{split}_pos.jsonl")
        direct_name = f"{ADAPTIVE_PREFIX}_{branch}_forced_direct_{split}_pos"
        reason_name = f"{ADAPTIVE_PREFIX}_{branch}_forced_reason_{split}_pos"
        direct_rows = [
            mark_e16(e15.direct_eval_row(row, branch, f"forced_direct_{split}_source", split), branch, f"forced_direct_{split}_source")
            for row in direct_source
        ]
        reason_rows = [
            mark_e16(e15.reason_row(row, branch, style, f"forced_reason_{split}_source", 0, split, "forced_reason"), branch, f"forced_reason_{split}_source")
            for row in reason_source
        ]
        write_dataset(direct_name, direct_rows)
        write_dataset(reason_name, reason_rows)
        eval_names.extend([direct_name, reason_name])

    config = write_config(branch, train_name, dev_name)
    note = write_note(variant_id, spec, train_name, dev_name, audit)
    return {
        "variant": variant_id,
        "branch": branch,
        "train_dataset": train_name,
        "dev_dataset": dev_name,
        "eval_datasets": eval_names,
        "config": config,
        "note": note.as_posix(),
        "audit": audit,
    }


def main():
    results = {variant_id: build_variant(variant_id, spec) for variant_id, spec in VARIANTS.items()}
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
