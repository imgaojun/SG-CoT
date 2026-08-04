#!/usr/bin/env python3
"""Prepare E77 E76 direct-control datasets and config.

E77 uses the exact E76 examples and splits, but replaces the CoT target with
the gold direct JSON output. This isolates whether E76 gains come from the
contrastive CoT itself or from the E76 sample distribution.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXP_DIR = REPO / "experiments"

DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX = "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
SOURCE_BRANCH = "e76_contrastive_exactness_glm51_full1500_thinking_evidence_cot"
TARGET_BRANCH = "e77_e76_direct_control"
WARM_START = (
    "/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/"
    "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/"
    "checkpoint-2064"
)
DIRECT_INSTRUCTION = (
    "You are doing event extraction. Use only the provided candidate event types "
    "and their schema cards. Extract all event mentions supported by the text "
    'and output strict JSON with token offsets. If no valid event is expressed '
    'by the candidate set, output {"events": []}.'
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def convert_rows(rows: list[dict], split: str) -> list[dict]:
    converted = []
    for idx, row in enumerate(rows):
        gold = row.get("gold_output")
        if not gold:
            raise ValueError(f"missing gold_output at split={split} idx={idx}")
        json.loads(gold)
        meta = dict(row.get("meta") or {})
        meta.update(
            {
                "adaptive_source": "e77_e76_direct_control",
                "adaptive_target_style": "direct_json_offsets_no_cot",
                "adaptive_dataset_role": split,
                "control_source_branch": SOURCE_BRANCH,
                "control_changed_variable": "remove_thinking_use_gold_output",
            }
        )
        converted.append(
            {
                "instruction": DIRECT_INSTRUCTION,
                "input": row["input"],
                "output": gold,
                "meta": meta,
                "gold_output": gold,
            }
        )
    return converted


def update_dataset_info(names: list[str]) -> None:
    info_path = DATA_DIR / "dataset_info.json"
    data = json.loads(info_path.read_text(encoding="utf-8"))
    for name in names:
        data[name] = {
            "file_name": f"{name}.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    info_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_config(train_name: str, dev_name: str) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{TARGET_BRANCH}_full_stepmatch.yaml"
    output_dir = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{TARGET_BRANCH}_full"
    cfg = {
        "model_name_or_path": WARM_START,
        "template": "qwen",
        "dataset_dir": "/workspace/project/data/stage2_adaptive_datasets",
        "dataset": train_name,
        "eval_dataset": dev_name,
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
        "learning_rate": 2.0e-6,
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
    ts = now_iso()
    note_path = EXP_DIR / "2026-06-15_e77_e76_direct_control.md"
    note = f"""---
id: 2026-06-15_e77_e76_direct_control
title: E77 E76 Direct-Control
kind: experiment
status: planned
created_at: {ts}
updated_at: {ts}
owners:
  - gaojun
tags:
  - e77
  - e76
  - direct-control
  - qwen3-4b
  - fairness
objective: Test whether E76 gains come from contrastive CoT rather than the E76 sample distribution alone.
artifacts:
  - /mnt/disk/gaojun/research/progressive-ee/{cfg_path}
  - /mnt/disk/gaojun/research/progressive-ee/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{TARGET_BRANCH}_full
related:
  - /mnt/disk/gaojun/research/progressive-ee/experiments/2026-06-14_e76_contrastive_exactness_cot.md
  - /mnt/disk/gaojun/research/progressive-ee/experiments/2026-06-15_e76_repeat1_2ep_stability.md
context:
  source_branch: {SOURCE_BRANCH}
  target_branch: {TARGET_BRANCH}
  warm_start: {WARM_START}
  train_rows: {counts["train"]}
---

# E77 E76 Direct-Control

## Goal

Use the exact E76 examples and split files, but remove `<thinking>` supervision and train direct JSON output. This isolates whether the E76 main-result gain is caused by contrastive CoT rather than by the E76 data distribution alone.

## Setup

- Source branch: `{SOURCE_BRANCH}`
- Target branch: `{TARGET_BRANCH}`
- Output format: bare strict JSON with token offsets, no `<thinking>` and no `<final>` tag.
- Warm start: Qwen3-4B Direct `checkpoint-2064`.
- Hyperparameters: matched to E76 original, including `3` epochs, `2e-6`, batch/grad accumulation, cutoff `1536`, and Zero2 offload.
- Counts:
  - train: `{counts["train"]}`
  - dev_seen: `{counts["dev_seen"]}`
  - test_seen: `{counts["test_seen"]}`
  - test_unseen: `{counts["test_unseen"]}`

## Commands

```bash
python3 scripts/prepare_e77_e76_direct_control_20260615.py
bash scripts/launch_e65_e57_cross_model_20260608.sh train-qwen4-e77 2
bash scripts/launch_e65_e57_cross_model_20260608.sh dev-qwen4-e77 <checkpoint> <gpu>
bash scripts/launch_e65_e57_cross_model_20260608.sh eval-qwen4-e77 <checkpoint> <gpu>
```

## Run Log

### {ts.replace("T", " ")[:16]} +08:00

- Prepared E77 direct-control datasets and config.
- Config: `{cfg_path}`
- Training not launched yet.

## Result

Pending.

## Conclusion

Pending.

## Next

- Train Qwen3-4B E77.
- Run dev checkpoint selection by `Argument F1 -> Event F1 -> Trigger F1`.
- Evaluate the dev-selected checkpoint on `test_seen` and `test_unseen`.
- Compare E77 against E76 original ck186 and E76 repeat1 ck93.
"""
    note_path.write_text(note, encoding="utf-8")
    return note_path


def main() -> None:
    counts = {}
    names = []
    for split in ["train", "dev_seen", "test_seen", "test_unseen"]:
        src = DATA_DIR / f"{DATA_PREFIX}_{SOURCE_BRANCH}_{split}_pos.jsonl"
        target_name = f"{DATA_PREFIX}_{TARGET_BRANCH}_{split}_pos"
        rows = convert_rows(load_jsonl(src), split)
        write_jsonl(DATA_DIR / f"{target_name}.jsonl", rows)
        counts[split] = len(rows)
        names.append(target_name)
    update_dataset_info(names)
    cfg_path = write_config(names[0], names[1])
    note_path = write_note(cfg_path, counts)
    print(json.dumps({"counts": counts, "config": str(cfg_path), "note": str(note_path)}, indent=2))


if __name__ == "__main__":
    main()
