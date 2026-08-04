#!/usr/bin/env python3
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPTIVE_PREFIX = f"{DATA_PREFIX}_adaptive"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
WARM_START = (
    "/workspace/project/outputs/stage2_adaptive_teacher_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_direct_teacher_full/checkpoint-258"
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def register_dataset(name: str, file_name: str) -> None:
    info_path = DATA_DIR / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info[name] = {"file_name": file_name, "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def train_config(branch: str, train_name: str, dev_name: str, epochs: float = 3.0, lr: float = 3.0e-6) -> dict:
    return {
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
        "logging_steps": 1,
        "report_to": "none",
        "finetuning_type": "full",
        "cutoff_len": 1536,
        "max_samples": 20000,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "packing": False,
        "learning_rate": lr,
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


def main() -> None:
    source_name = f"{ADAPTIVE_PREFIX}_e40_seed1500_thinking_evidence_cot_train_pos"
    source_path = DATA_DIR / f"{source_name}.jsonl"
    rows = load_jsonl(source_path)
    if len(rows) < 500:
        raise SystemExit(f"Need at least 500 rows, got {len(rows)} from {source_path}")
    branch = "e40_seed500_nested_thinking_evidence_cot"
    train_name = f"{ADAPTIVE_PREFIX}_{branch}_train_pos"
    train_path = DATA_DIR / f"{train_name}.jsonl"
    picked = rows[:500]
    for idx, row in enumerate(picked):
        row.setdefault("meta", {})["e40_scaling_source"] = "nested_first500_from_e40_seed1500"
        row["meta"]["e40_scaling_index"] = idx
    write_jsonl(train_path, picked)
    register_dataset(train_name, train_path.name)

    # Reuse the exact same eval prompt/data protocol as seed1500 for fair comparison.
    for split in ["dev_seen", "test_seen", "test_unseen"]:
        src = DATA_DIR / f"{ADAPTIVE_PREFIX}_e40_seed1500_thinking_evidence_cot_{split}_pos.jsonl"
        name = f"{ADAPTIVE_PREFIX}_{branch}_{split}_pos"
        dst = DATA_DIR / f"{name}.jsonl"
        if not dst.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        register_dataset(name, dst.name)

    config_path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    write_yaml(
        config_path,
        train_config(
            branch,
            train_name,
            f"{ADAPTIVE_PREFIX}_{branch}_dev_seen_pos",
        ),
    )

    exp_path = EXPERIMENT_DIR / "2026-06-04_e40_data_scaling.md"
    exp_path.write_text(
        f"""---
id: 2026-06-04_e40_data_scaling
title: E40 Evidence-CoT Data Scaling
kind: experiment
status: running
created_at: 2026-06-04T14:10:00+08:00
updated_at: 2026-06-04T14:10:00+08:00
owners:
  - codex
tags:
  - e40
  - data-scaling
  - evidence-cot
objective: Test whether E40 evidence-CoT improves with supervision data scale.
artifacts:
  configs:
    - {config_path.as_posix()}
  outputs:
    - /mnt/disk/gaojun/research/progressive-ee/outputs/stage2_strategy_cot_e40/e40_seed3000_20260604
    - /mnt/disk/gaojun/research/progressive-ee/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full
related:
  plans:
    - /mnt/disk/gaojun/research/progressive-ee/PLANS.md
context:
  scale_points:
    - 500
    - 1324
    - target_3000
---

# E40 Evidence-CoT Data Scaling

## Goal

Build a data-scaling curve for E40 evidence-CoT after the seed1500 run showed clear gains over Direct.

## Setup

- scale 500: nested first 500 accepted rows from E40 seed1500, same eval prompts.
- scale 1324: existing E40 seed1500 accepted rows.
- scale 3000: generate a new larger DeepSeek V4 Pro evidence-CoT set.

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
python3 scripts/prepare_e40_data_scaling_20260604.py
OPENAI_API_KEY=<virtual-key> python3 scripts/generate_evidence_cot_e40_20260604.py --run_name e40_seed3000 --limit 3000 --seed 4042 --workers 16 --base_url ${LLM_BASE_URL} --model deepseek-v4-pro --verifier_model deepseek-v4-pro --timeout 240
```

## Run Log

### 2026-06-04 14:10 +08:00

- prepared nested 500-row E40 training dataset
- prepared train config: `{config_path.as_posix()}`

## Result

Pending.

## Conclusion

Pending.

## Next

- train nested 500-row E40 model
- generate E40 seed3000 data
- train and evaluate seed3000 after generation completes
""",
        encoding="utf-8",
    )

    print(json.dumps({"train_dataset": train_name, "train_rows": len(picked), "config": config_path.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
