import json
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
TEMPLATE_CONFIG = CONFIG_DIR / f"{RUN_PREFIX}_likelihood15_goldplan_type_role_hint_plan_lite_bal30_full_stepmatch.yaml"
NOW = "2026-05-13T12:00:00+08:00"


BRANCHES = {
    "outcome10_l15bal30_routeonly_probe": {"epochs": 3.0, "objective": "Probe whether outcome10 route labels are learnable using route-only targets."},
    "outcome15_l15bal30_routeonly_probe": {"epochs": 3.0, "objective": "Probe whether outcome15 route labels are learnable using route-only targets."},
    "outcome10_l15bal30_type_role_hint_plan_lite_raw": {"epochs": 16.0, "objective": "Train full adaptive router with outcome10 labels and no route auxiliary rows."},
    "outcome10_l15bal30_type_role_hint_plan_lite_routeaux1x": {"epochs": 16.0, "objective": "Train full adaptive router with outcome10 labels and one route-only auxiliary row per sample."},
    "outcome15_l15bal30_type_role_hint_plan_lite_routeaux1x": {"epochs": 16.0, "objective": "Train full adaptive router with outcome15 labels and one route-only auxiliary row per sample."},
    "outcome10_l15bal30_type_role_hint_plan_lite_routeaux2x": {"epochs": 16.0, "objective": "Train full adaptive router with outcome10 labels and two route-only auxiliary rows per sample."},
}


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def title(branch: str):
    return "Stage2 Adaptive " + branch.replace("_", " ").title() + " RichERE Split1 Oracle Mixed Noise Qwen3-1.7B"


def make_note(branch: str, cfg_path: Path, output_dir: Path, objective: str):
    exp_id = f"2026-05-13_stage2_adaptive_{branch}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    body = f"""---
id: {exp_id}
title: {title(branch)}
kind: experiment
status: planned
created_at: {NOW}
updated_at: {NOW}
owners:
  - codex
tags:
  - stage2
  - adaptive-reasoning
  - outcome-routing
  - richere
  - qwen3-1.7b
objective: {objective}
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  plans:
    - {REPO / 'PLANS.md'}
context:
  dataset: RichERE
  split: split1
  candidate_source: oracle_mixed_noise_top10_shuffle
  model: Qwen3-1.7B
  route_label_source: outcome_l15bal30
  target_style: type_role_hint_plan_lite
---

# {title(branch)}

## Goal

{objective}

## Setup

- train dataset: `{DATA_PREFIX}_{branch}_train_pos`
- dev dataset: `{DATA_PREFIX}_{branch}_dev_seen_pos`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`

## Commands

```bash
bash scripts/launch_adaptive_outcome_train_20260513.sh {branch}=<gpu>
```

## Run Log

### 2026-05-13 12:00 +08:00

- created experiment note.
- planned dataset/config generation via `scripts/build_adaptive_outcome_wave_20260513.sh`.

## Result

Pending.

## Conclusion

Pending.

## Next

- generate outcome labels and datasets.
- train this branch.
- run route-only/dev frontier and selected formal eval if gate passes.
"""
    write_text(note, body)
    return note


def main():
    with open(TEMPLATE_CONFIG, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)
    made = []
    for branch, meta in BRANCHES.items():
        config = dict(template)
        config["dataset"] = f"{DATA_PREFIX}_{branch}_train_pos"
        config["eval_dataset"] = f"{DATA_PREFIX}_{branch}_dev_seen_pos"
        config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full"
        config["num_train_epochs"] = meta["epochs"]
        config["logging_steps"] = 5
        cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
        write_yaml(cfg_path, config)
        note = make_note(branch, cfg_path, REPO / config["output_dir"].replace("/workspace/project/", ""), meta["objective"])
        made.append({"branch": branch, "config": str(cfg_path), "note": str(note)})
    print(json.dumps(made, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
