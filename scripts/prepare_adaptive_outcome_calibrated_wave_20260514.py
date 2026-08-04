import json
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
TEMPLATE_CONFIG = CONFIG_DIR / f"{RUN_PREFIX}_outcome15_l15bal30_type_role_hint_plan_lite_routeaux1x_full_stepmatch.yaml"
NOW = "2026-05-14T11:00:00+08:00"


BRANCHES = {
    "outcome15cal_nlltop10_type_role_hint_plan_lite_routeaux2x_reasonos2": {
        "label_source": "outcome15cal_nlltop10",
        "objective": "Train full adaptive routing from outcome15 route-NLL top10 calibrated labels with route auxiliary rows and reason oversampling.",
    },
    "outcome15cal_nlltop15_type_role_hint_plan_lite_routeaux2x_reasonos2": {
        "label_source": "outcome15cal_nlltop15",
        "objective": "Train full adaptive routing from outcome15 route-NLL top15 calibrated labels with route auxiliary rows and reason oversampling.",
    },
}


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def title(branch: str):
    return "Stage2 Adaptive " + branch.replace("_", " ").title() + " RichERE Split1 Oracle Mixed Noise Qwen3-1.7B"


def make_note(branch: str, cfg_path: Path, output_dir: Path, objective: str, label_source: str):
    exp_id = f"2026-05-14_stage2_adaptive_{branch}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
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
  - calibrated-router
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
  reports:
    - {REPO / 'reports/2026-05-13_stage2_adaptive_outcome15_calibrated_router_formal_summary.md'}
context:
  dataset: RichERE
  split: split1
  candidate_source: oracle_mixed_noise_top10_shuffle
  model: Qwen3-1.7B
  route_label_source: {label_source}
  target_style: type_role_hint_plan_lite
  route_aux_repeat: 2
  route_reason_oversample: 2
---

# {title(branch)}

## Goal

{objective}

## Setup

- train dataset: `{DATA_PREFIX}_{branch}_train_pos`
- dev dataset: `{DATA_PREFIX}_{branch}_dev_seen_pos`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- router label source: `{label_source}` from `outcome15_l15bal30_routecls_balanced_probe/checkpoint-221` route-NLL ranking.

## Commands

```bash
bash scripts/build_adaptive_outcome_calibrated_scorebase_20260514.sh
bash scripts/launch_adaptive_outcome_calibrated_route_score_20260514.sh train=<gpu> dev_seen=<gpu>
bash scripts/build_adaptive_outcome_calibrated_wave_20260514.sh
python3 scripts/prepare_adaptive_outcome_calibrated_wave_20260514.py
bash scripts/launch_adaptive_outcome_calibrated_train_20260514.sh {branch}=<gpu>
bash scripts/launch_adaptive_outcome_calibrated_devpick_20260514.sh {branch}=<gpu>
python3 src/stage2_analysis/select_adaptive_execution_gate.py --branch {branch}
```

## Run Log

### 2026-05-14 11:00 +08:00

- planned outcome-calibrated router distillation wave.
- objective is to preserve the separate-router signal inside a full adaptive model using route auxiliary rows and reason oversampling.

## Result

Pending.

## Conclusion

Pending.

## Next

- train branch.
- run dev gate with free-route plus forced-direct/forced-reason execution simulation.
- run formal eval only if dev gate passes.
"""
    write_text(note, body)
    return note


def main():
    with TEMPLATE_CONFIG.open("r", encoding="utf-8") as f:
        template = yaml.safe_load(f)
    made = []
    for branch, meta in BRANCHES.items():
        config = dict(template)
        config["dataset"] = f"{DATA_PREFIX}_{branch}_train_pos"
        config["eval_dataset"] = f"{DATA_PREFIX}_{branch}_dev_seen_pos"
        config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full"
        config["num_train_epochs"] = 8.0
        config["logging_steps"] = 5
        config["save_strategy"] = "epoch"
        config["eval_strategy"] = "epoch"
        config["load_best_model_at_end"] = False
        cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
        write_yaml(cfg_path, config)
        output_dir = REPO / config["output_dir"].replace("/workspace/project/", "")
        note = make_note(branch, cfg_path, output_dir, meta["objective"], meta["label_source"])
        made.append({"branch": branch, "config": str(cfg_path), "note": str(note)})
    print(json.dumps(made, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
