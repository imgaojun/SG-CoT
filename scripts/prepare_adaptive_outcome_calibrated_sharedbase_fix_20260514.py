import json
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
TEMPLATE_CONFIG = CONFIG_DIR / f"{RUN_PREFIX}_outcome15cal_nlltop15_type_role_hint_plan_lite_routeaux2x_reasonos2_full_stepmatch.yaml"
NOW = "2026-05-14T18:10:02+08:00"


BRANCHES = {
    "outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2": {
        "objective": "No-aux sanity run for shared-base outcome15cal top15 labels; verify full adaptive extraction format recovers without route-only auxiliary rows.",
        "route_aux_repeat": 0,
        "route_aux_classifier_prompt": False,
        "expected_audit": {
            "total_count": 2364,
            "full_with_final_count": 2364,
            "route_only_count": 0,
            "route_only_classifier_prompt_count": 0,
            "route_only_full_extraction_prompt_count": 0,
        },
    },
    "outcome15cal_nlltop15_type_role_hint_plan_lite_routeauxclf1x_reasonos2": {
        "objective": "Shared-base outcome15cal top15 run with one route-only auxiliary row per sample under route-classifier prompt.",
        "route_aux_repeat": 1,
        "route_aux_classifier_prompt": True,
        "expected_audit": {
            "total_count": 4420,
            "full_with_final_count": 2364,
            "route_only_count": 2056,
            "route_only_classifier_prompt_count": 2056,
            "route_only_full_extraction_prompt_count": 0,
        },
    },
}


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


def make_note(branch: str, cfg_path: Path, output_dir: Path, meta: dict):
    exp_id = f"2026-05-14_stage2_adaptive_{branch}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    audit = audit_for(branch)
    expected = meta["expected_audit"]
    route_aux_classifier_prompt = str(meta["route_aux_classifier_prompt"]).lower()
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
  - shared-base
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
    - {REPO / 'reports/2026-05-14_stage2_adaptive_outcome_calibrated_failure_analysis_and_optimization_plan.md'}
context:
  dataset: RichERE
  split: split1
  candidate_source: oracle_mixed_noise_top10_shuffle
  model: Qwen3-1.7B
  route_label_source: outcome15cal_nlltop15
  target_style: type_role_hint_plan_lite
  route_aux_repeat: {meta['route_aux_repeat']}
  route_aux_classifier_prompt: {route_aux_classifier_prompt}
  route_reason_oversample: 2
---

# {title(branch)}

## Goal

{meta['objective']}

## Setup

- train dataset: `{DATA_PREFIX}_{branch}_train_pos`
- dev dataset: `{DATA_PREFIX}_{branch}_dev_seen_pos`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- route labels: `outcome15cal_nlltop15` from `outcome15_l15bal30_routecls_balanced_probe/checkpoint-221` route-NLL ranking.
- expected audit: `{json.dumps(expected, sort_keys=True)}`
- actual audit: `{json.dumps(audit, sort_keys=True)}`

## Commands

```bash
bash scripts/build_adaptive_outcome_calibrated_sharedbase_fix_20260514.sh
python3 scripts/prepare_adaptive_outcome_calibrated_sharedbase_fix_20260514.py
bash scripts/launch_adaptive_outcome_calibrated_sharedbase_fix_train_20260514.sh {branch}=<gpu>
bash scripts/launch_adaptive_outcome_calibrated_sharedbase_fix_devpick_20260514.sh {branch}=<gpu,gpu>
python3 src/stage2_analysis/select_adaptive_sharedbase_fix_execution_gate.py --branch {branch}
python3 scripts/build_adaptive_outcome_calibrated_sharedbase_fix_formal_manifest_20260514.py
python3 scripts/summarize_adaptive_outcome_calibrated_sharedbase_fix_formal_20260514.py
```

## Run Log

### 2026-05-14 18:10 +08:00

- created shared-base fix config and experiment note.
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
- run devpick and shared-base fix execution gate.
- run formal only if dev gate passes.
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
        note = make_note(branch, cfg_path, output_dir, meta)
        made.append({"branch": branch, "config": cfg_path.as_posix(), "note": note.as_posix()})
    print(json.dumps(made, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
