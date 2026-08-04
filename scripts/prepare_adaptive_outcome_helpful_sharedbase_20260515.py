import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BASE_MODEL = "/workspace/models/LLM-Research/Qwen3-1.7B"
NOAUX_CKPT = (
    "/workspace/project/outputs/stage2_adaptive_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full/checkpoint-1184"
)
TEMPLATE_CONFIG = CONFIG_DIR / f"{RUN_PREFIX}_outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full_stepmatch.yaml"
TZ = timezone(timedelta(hours=8))


BRANCHES = {
    "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2": {
        "objective": "Outcome-helpful shared-base adaptive run using fixed classifier-prompt route aux and selected direct anchors.",
        "model_name_or_path": BASE_MODEL,
        "learning_rate": 1.0e-5,
        "warm_start": False,
    },
    "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux": {
        "objective": "Outcome-helpful shared-base adaptive run warm-started from clean noaux checkpoint to preserve direct extraction.",
        "model_name_or_path": NOAUX_CKPT,
        "learning_rate": 5.0e-6,
        "warm_start": True,
    },
}

EXPECTED_AUDIT = {
    "total_count": 4704,
    "full_with_final_count": 2648,
    "route_only_count": 2056,
    "route_only_classifier_prompt_count": 2056,
    "route_only_full_extraction_prompt_count": 0,
}


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


def title(branch: str):
    return "Stage2 Adaptive " + branch.replace("_", " ").title() + " RichERE Split1 Oracle Mixed Noise Qwen3-1.7B"


def audit_for(branch: str):
    meta = REPO / f"data/stage2_adaptive_datasets/{DATA_PREFIX}_{branch}_train_pos.meta.json"
    payload = load_json(meta)
    return payload["audit"]


def make_note(branch: str, cfg_path: Path, output_dir: Path, meta: dict, timestamp: str):
    exp_id = f"2026-05-15_stage2_adaptive_{branch}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    audit = audit_for(branch)
    warm = str(meta["warm_start"]).lower()
    body = f"""---
id: {exp_id}
title: {title(branch)}
kind: experiment
status: planned
created_at: {timestamp}
updated_at: {timestamp}
owners:
  - codex
tags:
  - stage2
  - adaptive-reasoning
  - outcome-routing
  - shared-base
  - route-nll
  - richere
  - qwen3-1.7b
objective: {meta['objective']}
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  plans:
    - {REPO / 'PLANS.md'}
  reports:
    - {REPO / 'reports/2026-05-15_stage2_adaptive_sharedbase_optimization_plan.md'}
context:
  dataset: RichERE
  split: split1
  candidate_source: oracle_mixed_noise_top10_shuffle
  model: Qwen3-1.7B
  warm_start_from_noaux: {warm}
  route_label_source: outcome_l15bal30_15
  target_style: type_role_hint_plan_lite
  route_aux_repeat: 1
  route_aux_classifier_prompt: true
  route_reason_oversample: 2
  pair_selected_direct: true
---

# {title(branch)}

## Goal

{meta['objective']}

## Setup

- train dataset: `{DATA_PREFIX}_{branch}_train_pos`
- dev dataset: `{DATA_PREFIX}_{branch}_dev_seen_pos`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- model start: `{meta['model_name_or_path']}`
- route labels: `outcome_l15bal30_15`, mined from forced-direct vs forced-reason execution.
- expected audit: `{json.dumps(EXPECTED_AUDIT, sort_keys=True)}`
- actual audit: `{json.dumps(audit, sort_keys=True)}`

## Commands

```bash
bash scripts/build_adaptive_outcome_helpful_sharedbase_20260515.sh
bash scripts/launch_adaptive_outcome_helpful_sharedbase_train_20260515.sh {branch}=<gpu>
bash scripts/launch_adaptive_outcome_helpful_sharedbase_devpick_20260515.sh {branch}=<gpu,gpu>
bash scripts/launch_adaptive_outcome_helpful_sharedbase_route_nll_dev_20260515.sh {branch}=<gpu>
python3 src/stage2_analysis/select_adaptive_sharedbase_nll_execution_gate.py --branch {branch}
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- created outcome-helpful shared-base config and experiment note.
- dataset audit passed:
  - total: `{audit['total_count']}`
  - full with final: `{audit['full_with_final_count']}`
  - route-only: `{audit['route_only_count']}`
  - route-only classifier prompt: `{audit['route_only_classifier_prompt_count']}`
  - route-only full extraction prompt: `{audit['route_only_full_extraction_prompt_count']}`

## Result

Pending.

## Conclusion

Pending.

## Next

- train this branch.
- run devpick and route-NLL execution gate.
- run formal only if dev route-NLL execution, format validity, and direct-retention gates pass.
"""
    write_text(note, body)
    return note


def main():
    timestamp = now_iso()
    with TEMPLATE_CONFIG.open("r", encoding="utf-8") as f:
        template = yaml.safe_load(f)
    made = []
    for branch, meta in BRANCHES.items():
        config = dict(template)
        config["model_name_or_path"] = meta["model_name_or_path"]
        config["dataset"] = f"{DATA_PREFIX}_{branch}_train_pos"
        config["eval_dataset"] = f"{DATA_PREFIX}_{branch}_dev_seen_pos"
        config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full"
        config["learning_rate"] = meta["learning_rate"]
        config["num_train_epochs"] = 8.0
        config["logging_steps"] = 5
        config["save_strategy"] = "epoch"
        config["eval_strategy"] = "epoch"
        config["load_best_model_at_end"] = False
        cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
        write_yaml(cfg_path, config)
        output_dir = REPO / config["output_dir"].replace("/workspace/project/", "")
        note = make_note(branch, cfg_path, output_dir, meta, timestamp)
        made.append({"branch": branch, "config": cfg_path.as_posix(), "note": note.as_posix()})
    print(json.dumps(made, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
