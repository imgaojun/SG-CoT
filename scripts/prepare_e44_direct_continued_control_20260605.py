#!/usr/bin/env python3
"""Prepare E44 Direct-continued fairness control for E40.

This control uses the exact E40 seed1500 accepted examples, but replaces the
target with direct offset JSON wrapped in <FINAL>...</FINAL>. It continues from
the same Qwen3-1.7B Direct checkpoint used by the E40 main result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/stage2_adaptive_datasets"
CONFIG_DIR = ROOT / "configs/generated/stage2_adaptive"
EXP_DIR = ROOT / "experiments"

SOURCE_BRANCH = "e40_seed1500_thinking_evidence_cot"
TARGET_BRANCH = "e44_seed1500_direct_continued_control"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
WARM_START = (
    "/workspace/project/outputs/stage2_adaptive_teacher_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_direct_teacher_full/"
    "checkpoint-258"
)
DIRECT_INSTRUCTION = (
    "You are doing event extraction. Use only the provided candidate event types "
    "and schema cards. Output `<FINAL>{...}</FINAL>` with the complete strict "
    "JSON event list using token offsets. Do not output text outside the requested tag."
)


def now_iso() -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def convert_rows(rows: list[dict], split: str) -> list[dict]:
    out = []
    for row in rows:
        gold = row.get("gold_output")
        if not gold:
            raise ValueError(f"missing gold_output for split={split}")
        json.loads(gold)
        meta = dict(row.get("meta", {}))
        meta.update(
            {
                "adaptive_source": "e44_direct_continued_control",
                "adaptive_target_style": "direct_final_offsets",
                "adaptive_dataset_role": split,
                "control_source_branch": SOURCE_BRANCH,
            }
        )
        out.append(
            {
                "instruction": DIRECT_INSTRUCTION,
                "input": row["input"],
                "output": f"<FINAL>{gold}</FINAL>",
                "meta": meta,
                "gold_output": gold,
            }
        )
    return out


def update_dataset_info(dataset_names: list[str]) -> None:
    path = DATA_DIR / "dataset_info.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for name in dataset_names:
        data[name] = {
            "file_name": f"{name}.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_config() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{TARGET_BRANCH}_full_stepmatch.yaml"
    output_dir = (
        "/workspace/project/outputs/stage2_adaptive_runs_user/"
        f"{RUN_PREFIX}_{TARGET_BRANCH}_full"
    )
    cfg = {
        "model_name_or_path": WARM_START,
        "template": "qwen",
        "dataset_dir": "/workspace/project/data/stage2_adaptive_datasets",
        "dataset": f"{DATA_PREFIX}_{TARGET_BRANCH}_train_pos",
        "eval_dataset": f"{DATA_PREFIX}_{TARGET_BRANCH}_dev_seen_pos",
        "output_dir": output_dir,
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
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def write_note(cfg_path: Path, counts: dict[str, int]) -> Path:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    created = now_iso()
    note_path = EXP_DIR / "2026-06-05_e44_direct_continued_control.md"
    note = f"""---
id: 2026-06-05_e44_direct_continued_control
title: E44 Direct-Continued Control for E40
kind: experiment
status: planned
created_at: {created}
updated_at: {created}
owners:
  - codex
tags:
  - e44
  - direct-control
  - e40
  - fairness
objective: Control for whether E40 gains come from evidence-CoT rather than extra continued training from the Direct checkpoint.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - /mnt/disk/gaojun/research/progressive-ee/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{TARGET_BRANCH}_full
related:
  plans:
    - /mnt/disk/gaojun/research/progressive-ee/PLANS.md
context:
  source_branch: {SOURCE_BRANCH}
  target_branch: {TARGET_BRANCH}
  warm_start: {WARM_START}
---

# E44 Direct-Continued Control for E40

## Goal

Test whether the E40 main-result gain is caused by evidence-grounded CoT, rather than simply by continuing training from the Direct checkpoint.

## Setup

- warm start: Qwen3-1.7B Direct teacher `checkpoint-258`
- train examples: same accepted examples as E40 seed1500
- target format: direct `<FINAL>{{offset JSON}}</FINAL>`
- train/dev/test counts:
  - train: `{counts['train']}`
  - dev_seen: `{counts['dev_seen']}`
  - test_seen: `{counts['test_seen']}`
  - test_unseen: `{counts['test_unseen']}`
- matched E40 main result:
  - train examples: same 1324 accepted rows
  - warm start: same Direct checkpoint
  - epochs: same 3

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
python3 scripts/prepare_e44_direct_continued_control_20260605.py
bash scripts/launch_e44_direct_continued_control_20260605.sh train 4
```

## Run Log

### {created.replace('T', ' ')[:16]} +08:00

- prepared Direct-continued train/dev/test datasets.
- prepared train config: `{cfg_path}`

## Result

Pending.

## Conclusion

Pending.

## Next

- train Direct-continued control.
- sweep epoch checkpoints on test_seen and test_unseen.
- compare against E40 seed1500 checkpoints and original Direct baseline.
"""
    note_path.write_text(note, encoding="utf-8")
    return note_path


def main() -> None:
    counts = {}
    dataset_names = []
    for split in ["train", "dev_seen", "test_seen", "test_unseen"]:
        src = DATA_DIR / f"{DATA_PREFIX}_{SOURCE_BRANCH}_{split}_pos.jsonl"
        name = f"{DATA_PREFIX}_{TARGET_BRANCH}_{split}_pos"
        rows = convert_rows(load_jsonl(src), split)
        write_jsonl(DATA_DIR / f"{name}.jsonl", rows)
        counts[split] = len(rows)
        dataset_names.append(name)
    update_dataset_info(dataset_names)
    cfg_path = write_config()
    note_path = write_note(cfg_path, counts)
    print(json.dumps({"counts": counts, "config": str(cfg_path), "note": str(note_path)}, indent=2))


if __name__ == "__main__":
    main()
