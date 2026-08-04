#!/usr/bin/env python3
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
LABEL_DIR = DATA_DIR / "labels"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DIRECT_PREFIX = "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
BRANCH = "modular_d1930_r2058_utility_m02_routecls_noauxwarm_lr2e6_save50"
LABEL_SOURCE = "modular_d1930_r2058_utility_m02"
MARGIN = 0.02
ROUTE_REASON_OVERSAMPLE = 4
NOAUX_CKPT = (
    "/workspace/project/outputs/stage2_adaptive_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full/checkpoint-1184"
)
TEMPLATE_CONFIG = CONFIG_DIR / (
    f"{RUN_PREFIX}_outcome15_l15bal30_routecls_noauxwarm_lr2e6_save50_probe_full_stepmatch.yaml"
)
SCHEMA = REPO / "data/schema/richere-en.event_schema.json"
TZ = timezone(timedelta(hours=8))


PREDICTIONS = {
    "train": {
        "direct": REPO / "outputs/stage2_modular_dualexpert/train_teacher_outputs_d1930_r2058_20260517/direct_expert_forced_direct_train/predictions.jsonl",
        "reason": REPO / "outputs/stage2_modular_dualexpert/train_teacher_outputs_d1930_r2058_20260517/reason_expert_forced_reason_train/predictions.jsonl",
    },
    "dev_seen": {
        "direct": REPO / "outputs/stage2_adaptive_runs_user_devpick_frontier/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux_full_forced_direct_dev_seen_max512/checkpoint-1930/predictions.jsonl",
        "reason": REPO / "outputs/stage2_adaptive_runs_user_devpick_frontier/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux_full_forced_reason_dev_seen_max512/checkpoint-2058/predictions.jsonl",
    },
}
FORMAL_DIRECT_ROOT = REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_balrouteaux_20260516/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux/checkpoint-1930/forced_direct"
FORMAL_REASON_ROOT = REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux/checkpoint-2058/forced_reason"


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def run(cmd):
    subprocess.run(cmd, cwd=REPO, check=True)


def label_path(split: str):
    return LABEL_DIR / f"{DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.jsonl"


def label_summary_path(split: str):
    return LABEL_DIR / f"{DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.summary.json"


def build_labels(split: str, direct_path: Path, reason_path: Path):
    if not direct_path.exists() or not reason_path.exists():
        raise FileNotFoundError(f"missing paired predictions for {split}: {direct_path}, {reason_path}")
    run(
        [
            "python3",
            "src/stage2_analysis/build_adaptive_outcome_route_labels.py",
            "--forced_direct_predictions",
            direct_path.as_posix(),
            "--forced_reason_predictions",
            reason_path.as_posix(),
            "--output_jsonl",
            label_path(split).as_posix(),
            "--summary_json",
            label_summary_path(split).as_posix(),
            "--reason_rate_cap",
            "1.0",
            "--margin",
            str(MARGIN),
            "--label_source",
            LABEL_SOURCE,
            "--miner_checkpoint",
            "D1930_direct_R2058_reason",
        ]
    )


def build_all_labels():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for split, paths in PREDICTIONS.items():
        build_labels(split, paths["direct"], paths["reason"])
    for split in ["test", "test_seen", "test_unseen"]:
        build_labels(
            split,
            FORMAL_DIRECT_ROOT / split / "predictions.jsonl",
            FORMAL_REASON_ROOT / split / "predictions.jsonl",
        )


def build_datasets():
    run(
        [
            "python3",
            "src/stage2_cot/build_adaptive_route_reasoning_dataset.py",
            "--schema_path",
            SCHEMA.as_posix(),
            "--direct_train_jsonl",
            f"{DIRECT_PREFIX}_train_pos.jsonl",
            "--direct_dev_jsonl",
            f"{DIRECT_PREFIX}_dev_seen_pos.jsonl",
            "--direct_test_jsonl",
            f"{DIRECT_PREFIX}_test_pos.jsonl",
            "--direct_test_seen_jsonl",
            f"{DIRECT_PREFIX}_test_seen_pos.jsonl",
            "--direct_test_unseen_jsonl",
            f"{DIRECT_PREFIX}_test_unseen_pos.jsonl",
            "--train_label_jsonl",
            label_path("train").as_posix(),
            "--dev_label_jsonl",
            label_path("dev_seen").as_posix(),
            "--test_label_jsonl",
            label_path("test").as_posix(),
            "--test_seen_label_jsonl",
            label_path("test_seen").as_posix(),
            "--test_unseen_label_jsonl",
            label_path("test_unseen").as_posix(),
            "--dataset_dir",
            DATA_DIR.as_posix(),
            "--train_dataset_name",
            f"{DATA_PREFIX}_{BRANCH}_train_pos",
            "--dev_dataset_name",
            f"{DATA_PREFIX}_{BRANCH}_dev_seen_pos",
            "--test_dataset_name",
            f"{DATA_PREFIX}_{BRANCH}_test_pos",
            "--test_seen_dataset_name",
            f"{DATA_PREFIX}_{BRANCH}_test_seen_pos",
            "--test_unseen_dataset_name",
            f"{DATA_PREFIX}_{BRANCH}_test_unseen_pos",
            "--target_style",
            "type_role_hint_plan_lite",
            "--max_role_checks_per_sample",
            "6",
            "--seed",
            "17",
            "--route_only_train",
            "--route_only_eval",
            "--route_classifier_prompt",
            "--route_reason_oversample",
            str(ROUTE_REASON_OVERSAMPLE),
        ]
    )


def make_config():
    config = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    config["model_name_or_path"] = NOAUX_CKPT
    config["dataset"] = f"{DATA_PREFIX}_{BRANCH}_train_pos"
    config["eval_dataset"] = f"{DATA_PREFIX}_{BRANCH}_dev_seen_pos"
    config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_full"
    config["learning_rate"] = 2.0e-6
    config["num_train_epochs"] = 1.5
    config["logging_steps"] = 5
    config["save_strategy"] = "steps"
    config["save_steps"] = 50
    config["eval_strategy"] = "steps"
    config["eval_steps"] = 50
    config["load_best_model_at_end"] = False
    cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
    write_yaml(cfg_path, config)
    return cfg_path, REPO / config["output_dir"].replace("/workspace/project/", "")


def dataset_audit(split: str):
    return load_json(DATA_DIR / f"{DATA_PREFIX}_{BRANCH}_{split}_pos.meta.json")["audit"]


def label_summary(split: str):
    return load_json(label_summary_path(split))


def make_note(cfg_path: Path, output_dir: Path, timestamp: str):
    exp_id = "2026-05-20_stage2_modular_dualexpert_utility_router_m02_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    title = "Stage2 Modular Dual-Expert Utility Router M02 D1930/R2058"
    audits = {split: dataset_audit(split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]}
    summaries = {split: label_summary(split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]}
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
  - adaptive-routing
  - modular-dual-expert
  - binary-router
  - utility-teacher
  - route-classification
objective: Prepare a non-leaking train/dev binary router from D1930/R2058 paired utility labels with reason_gain >= {MARGIN}.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  experiments:
    - {REPO / 'experiments/2026-05-20_stage2_oracle_best_binary_teacher_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
    - {REPO / 'experiments/2026-05-17_stage2_modular_dualexpert_utility_router_margin05_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  direct_expert: D1930_forced_direct
  reason_expert: R2058_forced_reason
  label_source: {LABEL_SOURCE}
  route_label_rule: reason_gain >= {MARGIN}
  route_reason_oversample: {ROUTE_REASON_OVERSAMPLE}
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 1.5
  save_steps: 50
---

# {title}

## Goal

Create a trainable, non-leaking binary router dataset that mirrors the margin-0.02 direct/reason-like teacher idea using train/dev D1930/R2058 paired utility scores.

## Setup

- branch: `{BRANCH}`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- model start: `{NOAUX_CKPT}`
- margin: `{MARGIN}`
- train/dev labels come from train/dev paired outputs, not formal oracle-best labels.

Dataset audits:

```json
{json.dumps(audits, ensure_ascii=False, indent=2)}
```

Label summaries:

```json
{json.dumps(summaries, ensure_ascii=False, indent=2)}
```

## Commands

```bash
cd {REPO}
python3 scripts/prepare_modular_dualexpert_utility_router_m02_20260520.py
bash scripts/launch_modular_dualexpert_utility_router_train_20260517.sh {BRANCH}=<gpu>
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- built D1930/R2058 margin-0.02 labels for train/dev/formal splits.
- built route-only classifier datasets.
- created training config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch exactly one binary router training run.
- score dev route-choice NLL over saved checkpoints.
- only then perform frozen formal replay.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = now_iso()
    build_all_labels()
    build_datasets()
    cfg_path, output_dir = make_config()
    note = make_note(cfg_path, output_dir, timestamp)
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "note": note.as_posix(),
                "audits": {split: dataset_audit(split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]},
                "label_summaries": {split: label_summary(split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
