import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH = "sampled_reason_expert_forcedreason_from_noaux_20260517"
NOAUX_CKPT = (
    "/workspace/project/outputs/stage2_adaptive_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full/checkpoint-1184"
)
TEMPLATE_CONFIG = CONFIG_DIR / (
    f"{RUN_PREFIX}_outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full_stepmatch.yaml"
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


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_config():
    config = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    config["model_name_or_path"] = NOAUX_CKPT
    config["dataset"] = f"{DATA_PREFIX}_{BRANCH}_forced_reason_train_pos"
    config["eval_dataset"] = f"{DATA_PREFIX}_{BRANCH}_forced_reason_dev_seen_pos"
    config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_full"
    config["learning_rate"] = 2.0e-6
    config["num_train_epochs"] = 3.0
    config["logging_steps"] = 5
    config["save_strategy"] = "epoch"
    config["eval_strategy"] = "epoch"
    config["load_best_model_at_end"] = False
    cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
    write_yaml(cfg_path, config)
    return cfg_path, REPO / config["output_dir"].replace("/workspace/project/", "")


def audit(dataset_name: str):
    return load_json(REPO / f"data/stage2_adaptive_datasets/{dataset_name}.meta.json")["audit"]


def make_note(cfg_path: Path, output_dir: Path, timestamp: str):
    exp_id = "2026-05-17_stage2_sampled_reason_expert_forcedreason_from_noaux_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    if note.exists():
        return note
    title = "Stage2 Sampled Supervision Reason Expert Forced-Reason From Noaux"
    train_dataset = f"{DATA_PREFIX}_{BRANCH}_forced_reason_train_pos"
    dev_dataset = f"{DATA_PREFIX}_{BRANCH}_forced_reason_dev_seen_pos"
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
  - sampled-supervision
  - reason-expert
  - richere
  - qwen3-1.7b
objective: Retrain a front reason expert on forced-reason adaptive targets before K=8 sampled counterfactual utility labeling.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  plans:
    - {REPO / 'PLANS.md'}
  experiments:
    - {REPO / 'experiments/2026-05-17_stage2_sampled_counterfactual_utility_k8_label_discovery_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  candidate_source: oracle_mixed_noise_top10_shuffle
  model: Qwen3-1.7B
  warm_start_from_noaux: true
  route_mode: forced_reason
  target_style: type_role_hint_plan_lite
  learning_rate: 2.0e-6
  num_train_epochs: 3.0
---

# {title}

## Goal

Train a dedicated forced-reason expert that will be used only for train/dev sampled counterfactual utility estimation. This intentionally keeps the reason generator separate from the direct expert before any shared-base merge attempt.

## Setup

- branch: `{BRANCH}`
- train dataset: `{train_dataset}`
- dev dataset: `{dev_dataset}`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- model start: `{NOAUX_CKPT}`
- train audit: `{json.dumps(audit(train_dataset), sort_keys=True)}`
- dev audit: `{json.dumps(audit(dev_dataset), sort_keys=True)}`

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
bash scripts/build_sampled_reason_expert_20260517.sh
bash scripts/launch_sampled_reason_expert_train_20260517.sh {BRANCH}=<gpu>
bash scripts/launch_sampled_reason_expert_forced_reason_devpick_20260517.sh {BRANCH}=<gpu,gpu>
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- created forced-reason train/dev datasets and paired forced-direct train/dev datasets for later sampling.
- created the reason-expert training config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- train the reason expert.
- select the best forced-reason checkpoint on dev_seen.
- use that checkpoint for K=8 sampled counterfactual utility labeling on train and dev_seen.
"""
    write_text(note, body)
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
