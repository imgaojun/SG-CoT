import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH = "modular_d1930_r2058_utility_margin05_routecls_noauxwarm_lr2e6_save50"
LABEL_SOURCE = "modular_d1930_r2058_utility_margin05"
MARGIN = 0.5
NOAUX_CKPT = (
    "/workspace/project/outputs/stage2_adaptive_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full/checkpoint-1184"
)
TEMPLATE_CONFIG = CONFIG_DIR / (
    f"{RUN_PREFIX}_outcome15_l15bal30_routecls_noauxwarm_lr2e6_save50_probe_full_stepmatch.yaml"
)
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def label_summary(split: str):
    return load_json(
        REPO
        / "data/stage2_adaptive_datasets/labels"
        / f"{DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.summary.json"
    )


def dataset_audit(dataset_name: str):
    return load_json(REPO / f"data/stage2_adaptive_datasets/{dataset_name}.meta.json")["audit"]


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


def make_note(cfg_path: Path, output_dir: Path, timestamp: str):
    exp_id = "2026-05-17_stage2_modular_dualexpert_utility_router_margin05_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    if note.exists():
        return note
    title = "Stage2 Modular Dual-Expert Utility Router Margin05 D1930/R2058"
    train_audit = dataset_audit(f"{DATA_PREFIX}_{BRANCH}_train_pos")
    dev_audit = dataset_audit(f"{DATA_PREFIX}_{BRANCH}_dev_seen_pos")
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
  - utility-router
  - margin-labels
  - route-classification
  - route-nll
objective: Train an independent route-only selector on high-confidence D1930/R2058 utility labels where reason is positive iff reason_gain > {MARGIN}.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  plans:
    - {REPO / 'PLANS.md'}
  experiments:
    - {REPO / 'experiments/2026-05-17_stage2_modular_dualexpert_utility_router_gainpos_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
    - {REPO / 'experiments/2026-05-17_stage2_modular_dualexpert_train_teacher_outputs_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  direct_expert: D1930_forced_direct
  reason_expert: R2058_forced_reason
  label_source: {LABEL_SOURCE}
  route_label_rule: reason_gain > {MARGIN}
  route_reason_oversample: 4
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 1.5
  save_steps: 50
---

# {title}

## Goal

Test whether removing low-margin/tie-like utility labels makes the separated utility router more stable than the `reason_gain > 0` branch.

## Setup

- branch: `{BRANCH}`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- model start: `{NOAUX_CKPT}`
- train dataset: `{DATA_PREFIX}_{BRANCH}_train_pos`
- dev dataset: `{DATA_PREFIX}_{BRANCH}_dev_seen_pos`
- train audit: `{json.dumps(train_audit, sort_keys=True)}`
- dev audit: `{json.dumps(dev_audit, sort_keys=True)}`
- label summaries: `{json.dumps(summaries, sort_keys=True)}`

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
bash scripts/build_modular_dualexpert_utility_router_margin05_20260517.sh
bash scripts/launch_modular_dualexpert_utility_router_train_20260517.sh {BRANCH}=<gpu>
bash scripts/run_modular_dualexpert_utility_router_margin05_after_train_20260517.sh
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- built margin-positive utility labels from paired D1930 direct and R2058 reason outputs.
- built route-only classifier datasets with utility labels and reason oversampling.
- created config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch utility-router training.
- sweep generated route and route-choice NLL on dev.
- compare against the `reason_gain > 0` branch before any formal run.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = now_iso()
    cfg_path, output_dir = make_config()
    note = make_note(cfg_path, output_dir, timestamp)
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "note": note.as_posix(),
                "label_summaries": {
                    split: label_summary(split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
