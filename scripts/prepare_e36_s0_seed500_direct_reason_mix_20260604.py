#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"

PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DIRECT_BRANCH = "e36_s0_seed500_direct_final_only"
REASON_BRANCH = "e36_s0_seed500_llm_checklist_reason"
MIX_BRANCH = "e36_s0_seed500_mix_dr1to1"

SHA_TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(SHA_TZ).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def register_dataset(name: str, file_name: str) -> None:
    info_path = DATA_DIR / "dataset_info.json"
    data = json.loads(info_path.read_text(encoding="utf-8"))
    data[name] = {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    info_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_path(branch: str, split: str) -> Path:
    return DATA_DIR / f"{PREFIX}_{branch}_{split}_pos.jsonl"


def mixed_name(split: str) -> str:
    return f"{PREFIX}_{MIX_BRANCH}_{split}_pos"


def clone_with_mix_meta(row: dict, style: str, pair_index: int) -> dict:
    out = dict(row)
    meta = dict(out.get("meta", {}))
    meta["adaptive_target_style"] = style
    meta["adaptive_mix_branch"] = MIX_BRANCH
    meta["adaptive_mix_ratio"] = "direct:reason=1:1"
    meta["adaptive_mix_pair_index"] = pair_index
    out["meta"] = meta
    return out


def build_train_rows() -> list[dict]:
    direct_rows = load_jsonl(source_path(DIRECT_BRANCH, "train"))
    reason_rows = load_jsonl(source_path(REASON_BRANCH, "train"))
    reason_by_id = {r["meta"]["e36_sample_id"]: r for r in reason_rows}
    mixed = []
    for pair_index, direct in enumerate(direct_rows):
        sample_id = direct["meta"]["e36_sample_id"]
        reason = reason_by_id[sample_id]
        mixed.append(clone_with_mix_meta(direct, "direct_final_only", pair_index))
        mixed.append(clone_with_mix_meta(reason, "llm_checklist_reasoning", pair_index))
    return mixed


def copy_eval_split(source_branch: str, target_suffix: str, split: str) -> str:
    rows = load_jsonl(source_path(source_branch, split))
    out_name = f"{PREFIX}_{MIX_BRANCH}_{target_suffix}_{split}_pos"
    out_path = DATA_DIR / f"{out_name}.jsonl"
    copied = []
    for idx, row in enumerate(rows):
        style = "direct_final_only" if source_branch == DIRECT_BRANCH else "llm_checklist_reasoning"
        copied.append(clone_with_mix_meta(row, style, idx))
    write_jsonl(out_path, copied)
    register_dataset(out_name, out_path.name)
    return out_name


def write_config(train_dataset: str, eval_dataset: str) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = (
        "/workspace/project/outputs/stage2_adaptive_runs_user/"
        f"{RUN_PREFIX}_{MIX_BRANCH}_full"
    )
    config = f"""model_name_or_path: /workspace/project/outputs/stage2_adaptive_teacher_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_direct_teacher_full/checkpoint-258
template: qwen
dataset_dir: /workspace/project/data/stage2_adaptive_datasets
dataset: {train_dataset}
eval_dataset: {eval_dataset}
output_dir: {out_dir}
stage: sft
do_train: true
overwrite_cache: true
preprocessing_num_workers: 8
save_strategy: epoch
eval_strategy: epoch
logging_steps: 1
report_to: none
finetuning_type: full
cutoff_len: 1024
max_samples: 20000
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
packing: false
learning_rate: 3.0e-06
warmup_ratio: 0.05
bf16: true
val_size: 0.0
eval_steps: 10
do_eval: true
save_only_model: true
num_train_epochs: 3.0
load_best_model_at_end: false
deepspeed: /workspace/project/configs/deepspeed/zero2_optimizer_offload_cpu.json
"""
    path = CONFIG_DIR / f"{RUN_PREFIX}_{MIX_BRANCH}_full_stepmatch.yaml"
    path.write_text(config, encoding="utf-8")
    return path


def write_experiment_note(config_path: Path, train_dataset: str, eval_dataset: str, train_rows: int) -> Path:
    ts = now_iso()
    path = EXPERIMENT_DIR / "2026-06-04_e36_s0_seed500_direct_reason_mix_train.md"
    if path.exists():
        return path
    body = f"""---
id: 2026-06-04_e36_s0_seed500_direct_reason_mix_train
title: E36 S0 Seed500 Direct+Reason 1:1 Mixed Train
kind: experiment
status: planned
created_at: {ts}
updated_at: {ts}
owners:
  - codex
tags:
  - e36
  - qwen3-1.7b
  - direct
  - llm-reasoning
  - mixed-training
objective: Train a 1:1 mixture of Direct final-only and LLM checklist reasoning supervision to preserve seen stability while retaining unseen Argument/Event gains.
artifacts:
  configs:
    - {config_path}
  outputs:
    - /mnt/disk/gaojun/research/progressive-ee/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{MIX_BRANCH}_full
related:
  formal_eval:
    - /mnt/disk/gaojun/research/progressive-ee/experiments/2026-06-04_e36_s0_seed500_no_budget_formal_eval.md
  plans:
    - /mnt/disk/gaojun/research/progressive-ee/PLANS.md
context:
  dataset: {train_dataset}
  eval_dataset: {eval_dataset}
  train_rows: {train_rows}
  mix_ratio: direct:reason=1:1
---

# E36 S0 Seed500 Direct+Reason 1:1 Mixed Train

## Goal

Train a single model on both Direct final-only and LLM checklist reasoning outputs for the same accepted E36 S0 seed500 examples.

## Setup

- base/warm start: Qwen3-1.7B direct teacher checkpoint-258.
- Direct rows: `499`.
- Reason rows: `499`.
- Total train rows: `{train_rows}`.
- Primary question: can mixed supervision recover Direct's seen stability while keeping Reason's unseen Argument/Event gain?

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
python3 scripts/prepare_e36_s0_seed500_direct_reason_mix_20260604.py
docker run ... llamafactory-cli train {config_path.relative_to(REPO)}
```

## Run Log

Pending launch.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- evaluate the trained mixed model under Direct-style and Reason-style prompts on test_seen/test_unseen.
"""
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-note", action="store_true")
    args = parser.parse_args()

    train_rows = build_train_rows()
    train_name = mixed_name("train")
    train_path = DATA_DIR / f"{train_name}.jsonl"
    write_jsonl(train_path, train_rows)
    register_dataset(train_name, train_path.name)

    reason_dev = copy_eval_split(REASON_BRANCH, "reason", "dev_seen")
    copy_eval_split(DIRECT_BRANCH, "direct", "test_seen")
    copy_eval_split(DIRECT_BRANCH, "direct", "test_unseen")
    copy_eval_split(REASON_BRANCH, "reason", "test_seen")
    copy_eval_split(REASON_BRANCH, "reason", "test_unseen")

    config_path = write_config(train_name, reason_dev)
    note_path = None if args.skip_note else write_experiment_note(config_path, train_name, reason_dev, len(train_rows))

    summary = {
        "mix_branch": MIX_BRANCH,
        "train_dataset": train_name,
        "train_rows": len(train_rows),
        "config_path": config_path.as_posix(),
        "experiment_note": note_path.as_posix() if note_path else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
