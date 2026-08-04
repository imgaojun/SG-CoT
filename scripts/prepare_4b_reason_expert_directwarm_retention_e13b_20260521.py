import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
FORMAL_DATA_DIR = REPO / "data/stage2_formal_datasets"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPTIVE_PREFIX = f"{DATA_PREFIX}_adaptive"
SOURCE_BRANCH = "confrare10_heur10_typeonlylite"
BRANCH = "confrare10_typeonlylite_directwarm_retention_e13b"
RUN_PREFIX = "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
WARM_START = (
    "/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/"
    "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/checkpoint-2064"
)
OUT_CONFIG = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
OUT_DIR = REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_full"
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def update_dataset_info(dataset_name: str, file_name: str):
    info_path = DATA_DIR / "dataset_info.json"
    info = load_json(info_path) if info_path.exists() else {}
    info[dataset_name] = {
        "file_name": file_name,
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
    write_json(info_path, info)


def key(row):
    return (row.get("meta") or {}).get("wnd_id")


def route_label(row):
    return (row.get("meta") or {}).get("adaptive_route_label", "direct")


def clone(row, source: str, duplicate_idx: int = 0):
    out = json.loads(json.dumps(row, ensure_ascii=False))
    meta = out.setdefault("meta", {})
    meta["e13b_source"] = source
    meta["e13b_duplicate_index"] = duplicate_idx
    meta["e13b_branch"] = BRANCH
    return out


def direct_anchor_row(direct_row, source: str, duplicate_idx: int = 0):
    out = clone(direct_row, source, duplicate_idx)
    out["output"] = "<ROUTE>direct</ROUTE>\n<FINAL>" + direct_row["output"] + "</FINAL>"
    out["gold_output"] = direct_row["output"]
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "e13b_directwarm_retention",
            "adaptive_dataset_role": "train",
            "adaptive_route_mode": "forced_direct_anchor",
            "adaptive_route_label": "direct",
            "adaptive_target_style": "direct_retention",
        }
    )
    return out


def build_train_rows():
    rng = random.Random(20260521)
    adaptive = load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_train_pos.jsonl")
    direct = load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    direct_by_key = {key(row): row for row in direct}
    if set(key(row) for row in adaptive) - set(direct_by_key):
        raise ValueError("adaptive/direct train wnd_id mismatch")

    reason_rows = [row for row in adaptive if route_label(row) == "reason"]
    direct_rows = [row for row in adaptive if route_label(row) != "reason"]

    # Approximate 60/30/10 by oversampling sparse reason rows.
    reason_repeat = 18
    reason_part = [
        clone(row, "reason_oversample", dup)
        for row in reason_rows
        for dup in range(reason_repeat)
    ]
    anchor_part = [
        direct_anchor_row(direct_by_key[key(row)], "direct_anchor", 0)
        for row in direct_rows
    ]
    # Retention hard anchors are deterministic duplicates of reason-labeled windows,
    # using direct final targets to counteract extraction drift on contentious cases.
    retention_part = []
    for dup in range(3):
        for row in reason_rows:
            retention_part.append(
                direct_anchor_row(direct_by_key[key(row)], "reason_window_direct_retention", dup)
            )

    rows = reason_part + anchor_part + retention_part
    rng.shuffle(rows)
    audit = {
        "source_branch": SOURCE_BRANCH,
        "adaptive_train_count": len(adaptive),
        "source_reason_count": len(reason_rows),
        "source_direct_count": len(direct_rows),
        "reason_repeat": reason_repeat,
        "reason_rows_after_repeat": len(reason_part),
        "direct_anchor_rows": len(anchor_part),
        "retention_rows": len(retention_part),
        "total_count": len(rows),
        "route_label_counts": {
            "reason": len(reason_part),
            "direct": len(anchor_part) + len(retention_part),
        },
        "mix_ratio": {
            "reason": len(reason_part) / len(rows),
            "direct_anchor": len(anchor_part) / len(rows),
            "retention": len(retention_part) / len(rows),
        },
    }
    return rows, audit


def build_dev_rows():
    rows = load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_dev_seen_pos.jsonl")
    return [clone(row, "dev_seen_source", 0) for row in rows]


def copy_eval_rows():
    made = []
    for split in ["test_seen", "test_unseen"]:
        for mode in ["forced_direct", "forced_reason"]:
            source_name = f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_{mode}_{split}_pos"
            target_name = f"{ADAPTIVE_PREFIX}_{BRANCH}_{mode}_{split}_pos"
            source_path = DATA_DIR / f"{source_name}.jsonl"
            rows = load_jsonl(source_path)
            out_rows = []
            for row in rows:
                out = clone(row, f"{mode}_{split}_eval_source", 0)
                meta = out.setdefault("meta", {})
                meta["adaptive_source"] = "e13b_directwarm_retention"
                meta["adaptive_dataset_role"] = split
                meta["adaptive_route_mode"] = mode
                out_rows.append(out)
            file_name = f"{target_name}.jsonl"
            write_jsonl(DATA_DIR / file_name, out_rows)
            update_dataset_info(target_name, file_name)
            write_json(
                DATA_DIR / f"{target_name}.meta.json",
                {
                    "dataset": target_name,
                    "source_dataset": source_name,
                    "num_examples": len(out_rows),
                    "created_at": now_iso(),
                },
            )
            made.append(target_name)
    return made


def write_datasets():
    train_name = f"{ADAPTIVE_PREFIX}_{BRANCH}_train_pos"
    dev_name = f"{ADAPTIVE_PREFIX}_{BRANCH}_dev_seen_pos"
    train_rows, audit = build_train_rows()
    dev_rows = build_dev_rows()
    train_file = f"{train_name}.jsonl"
    dev_file = f"{dev_name}.jsonl"
    write_jsonl(DATA_DIR / train_file, train_rows)
    write_jsonl(DATA_DIR / dev_file, dev_rows)
    update_dataset_info(train_name, train_file)
    update_dataset_info(dev_name, dev_file)
    write_json(
        DATA_DIR / f"{train_name}.meta.json",
        {"dataset": train_name, "audit": audit, "created_at": now_iso()},
    )
    write_json(
        DATA_DIR / f"{dev_name}.meta.json",
        {"dataset": dev_name, "source_branch": SOURCE_BRANCH, "num_examples": len(dev_rows), "created_at": now_iso()},
    )
    eval_names = copy_eval_rows()
    return train_name, dev_name, eval_names, audit


def write_config(train_name: str, dev_name: str):
    config = {
        "model_name_or_path": WARM_START,
        "template": "qwen",
        "dataset_dir": "/workspace/project/data/stage2_adaptive_datasets",
        "dataset": train_name,
        "eval_dataset": dev_name,
        "output_dir": f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_full",
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
        "learning_rate": 2.0e-6,
        "warmup_ratio": 0.05,
        "bf16": True,
        "val_size": 0.0,
        "eval_steps": 10,
        "do_eval": True,
        "save_only_model": True,
        "num_train_epochs": 4.0,
        "load_best_model_at_end": False,
        "deepspeed": "/workspace/project/configs/deepspeed/zero2_optimizer_offload_cpu.json",
    }
    write_yaml(OUT_CONFIG, config)
    return config


def write_note(train_name: str, dev_name: str, audit):
    timestamp = now_iso()
    exp_id = "2026-05-21_stage2_4b_reason_expert_directwarm_retention_e13b_richere_split1_oracle_mixed_noise_qwen3_4b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    body = f"""---
id: {exp_id}
title: Stage2 4B Direct-Warmup Retention Reason Expert E13B
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
  - direct-warmup
  - retention
objective: Improve the 4B reason expert by warm-starting from the direct extractor and mixing reason supervision with direct-retention anchors.
artifacts:
  configs:
    - {OUT_CONFIG}
  outputs:
    - {OUT_DIR}
related:
  plans:
    - {REPO / 'PLANS.md'}
  reports:
    - {REPO / 'reports/2026-05-21_stage2_4b_route_nll_selector_s12.md'}
context:
  dataset: RichERE split1 oracle_mixed_noise_top10_shuffle
  base_model: Qwen3-4B-Instruct
  warm_start: {WARM_START}
  branch: {BRANCH}
  source_branch: {SOURCE_BRANCH}
---

# Stage2 4B Direct-Warmup Retention Reason Expert E13B

## Goal

Train a stronger 4B reason expert that preserves direct extraction competence while learning reason-mode behavior.

## Setup

- train dataset: `{train_name}`
- dev dataset: `{dev_name}`
- source adaptive branch: `{SOURCE_BRANCH}`
- warm start: `{WARM_START}`
- audit: `{json.dumps(audit, sort_keys=True)}`

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
python3 scripts/prepare_4b_reason_expert_directwarm_retention_e13b_20260521.py
bash scripts/launch_4b_reason_expert_e13b_20260521.sh train 0
bash scripts/launch_4b_reason_expert_e13b_20260521.sh devpick 0
bash scripts/launch_4b_reason_expert_e13b_20260521.sh formal 0 1 2 3
python3 scripts/summarize_4b_reason_expert_e13b_20260521.py
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- prepared E13B mixed training data, config, and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- run devpick and forced-direct/forced-reason evals.
- run formal replay and compare against old 4B typeonlylite expert, M06 transfer, S12, and oracle.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    train_name, dev_name, eval_names, audit = write_datasets()
    config = write_config(train_name, dev_name)
    note = write_note(train_name, dev_name, audit)
    print(
        json.dumps(
            {
                "train_dataset": train_name,
                "dev_dataset": dev_name,
                "eval_datasets": eval_names,
                "audit": audit,
                "config": OUT_CONFIG.as_posix(),
                "output_dir": OUT_DIR.as_posix(),
                "note": note.as_posix(),
                "model_name_or_path": config["model_name_or_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
