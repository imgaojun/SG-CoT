import json
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
TEMPLATE_CONFIG = CONFIG_DIR / f"{RUN_PREFIX}_outcome10_l15bal30_routeonly_probe_full_stepmatch.yaml"
NOW = "2026-05-13T15:00:00+08:00"


BRANCHES = {
    "outcome10_l15bal30_routecls_balanced_probe": {
        "objective": "Probe outcome10 route labels with an explicit route-classification prompt and reason oversampling.",
    },
    "outcome15_l15bal30_routecls_balanced_probe": {
        "objective": "Probe outcome15 route labels with an explicit route-classification prompt and reason oversampling.",
    },
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
  - route-classification
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
  target_style: route_classifier
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
bash scripts/build_adaptive_outcome_routecls_wave_20260513.sh
bash scripts/launch_adaptive_outcome_train_20260513.sh {branch}=<gpu>
bash scripts/launch_adaptive_outcome_route_probe_devpick_20260513.sh {branch}=<gpu>
```

## Run Log

### 2026-05-13 15:00 +08:00

- planned route-classification probe after route-only outcome probes collapsed to all-direct predictions.

## Result

Pending.

## Conclusion

Pending.

## Next

- build route-classification dataset.
- train 3-epoch probe.
- evaluate route precision/recall on `dev_seen`.
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
        config["num_train_epochs"] = 3.0
        config["logging_steps"] = 5
        cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
        write_yaml(cfg_path, config)
        output_dir = REPO / config["output_dir"].replace("/workspace/project/", "")
        note = make_note(branch, cfg_path, output_dir, meta["objective"])
        made.append({"branch": branch, "config": str(cfg_path), "note": str(note)})
    print(json.dumps(made, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
