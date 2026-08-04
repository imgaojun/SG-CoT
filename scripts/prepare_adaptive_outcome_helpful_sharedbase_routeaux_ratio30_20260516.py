import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH = "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_routeauxr30_reasonos2_from_noaux"
NOAUX_CKPT = (
    "/workspace/project/outputs/stage2_adaptive_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full/checkpoint-1184"
)
TEMPLATE_CONFIG = CONFIG_DIR / (
    f"{RUN_PREFIX}_outcome_l15bal30_15_type_role_hint_plan_lite_"
    "routeauxclf1x_pairdirect_reasonos2_from_noaux_full_stepmatch.yaml"
)
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit():
    meta = REPO / f"data/stage2_adaptive_datasets/{DATA_PREFIX}_{BRANCH}_train_pos.meta.json"
    return load_json(meta)["audit"]


def make_config():
    with TEMPLATE_CONFIG.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["model_name_or_path"] = NOAUX_CKPT
    config["dataset"] = f"{DATA_PREFIX}_{BRANCH}_train_pos"
    config["eval_dataset"] = f"{DATA_PREFIX}_{BRANCH}_dev_seen_pos"
    config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_full"
    config["learning_rate"] = 5.0e-6
    config["num_train_epochs"] = 8.0
    config["logging_steps"] = 5
    config["save_strategy"] = "epoch"
    config["eval_strategy"] = "epoch"
    config["load_best_model_at_end"] = False
    cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
    write_yaml(cfg_path, config)
    return cfg_path, REPO / config["output_dir"].replace("/workspace/project/", "")


def make_note(cfg_path: Path, output_dir: Path, audit_payload: dict, timestamp: str):
    exp_id = f"2026-05-16_stage2_adaptive_{BRANCH}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    title = "Stage2 Adaptive Outcome-Helpful Shared-Base Route-Aux R30 RichERE Split1 Qwen3-1.7B"
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
  - adaptive-reasoning
  - outcome-routing
  - shared-base
  - route-nll
  - richere
  - qwen3-1.7b
objective: Test a moderate route-only classifier aux reason ratio to preserve direct-mode gains without suppressing reason-mode headroom.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  plans:
    - {REPO / 'PLANS.md'}
  reports:
    - {REPO / 'reports/2026-05-16_stage2_adaptive_outcome_helpful_sharedbase_balrouteaux_nll_formal_summary.md'}
    - {REPO / 'reports/2026-05-15_stage2_adaptive_outcome_helpful_sharedbase_nll_selector_calibration.md'}
context:
  dataset: RichERE
  split: split1
  candidate_source: oracle_mixed_noise_top10_shuffle
  model: Qwen3-1.7B
  warm_start_from_noaux: true
  route_label_source: outcome_l15bal30_15
  target_style: type_role_hint_plan_lite
  route_aux_repeat: 1
  route_aux_classifier_prompt: true
  route_aux_reason_target_rate: 0.30
  route_reason_oversample: 2
  pair_selected_direct: true
---

# {title}

## Goal

Test one narrow hypothesis: the fully balanced route-only aux branch improved the direct extractor and unseen routing, but it weakened the reason mode on seen/overall formal splits. This branch uses a moderate route-only reason target rate of `0.30` to reduce direct-label dominance without pushing route-only aux to a `1:1` direct/reason balance.

## Setup

- branch: `{BRANCH}`
- train dataset: `{DATA_PREFIX}_{BRANCH}_train_pos`
- dev dataset: `{DATA_PREFIX}_{BRANCH}_dev_seen_pos`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- model start: `{NOAUX_CKPT}`
- actual audit: `{json.dumps(audit_payload, sort_keys=True)}`

Expected shape:

- full rows with `<FINAL>` remain `2648`.
- route-only rows use classifier prompt only.
- route-only full-extraction-prompt rows must be `0`.
- route-only direct/reason rows are `1760 / 755`, or about `30.0%` reason among route-only aux rows.

## Commands

```bash
bash scripts/build_adaptive_outcome_helpful_sharedbase_routeaux_ratio30_20260516.sh
bash scripts/launch_adaptive_outcome_helpful_sharedbase_train_20260515.sh {BRANCH}=<gpu>
bash scripts/launch_adaptive_outcome_helpful_sharedbase_devpick_20260515.sh {BRANCH}=<gpu,gpu>
bash scripts/launch_adaptive_outcome_helpful_sharedbase_route_nll_dev_20260515.sh {BRANCH}=<gpu>
python3 src/stage2_analysis/select_adaptive_sharedbase_nll_execution_gate.py --branch {BRANCH}
bash scripts/run_adaptive_outcome_helpful_sharedbase_routeaux_ratio30_after_train_20260516.sh
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- created route-aux ratio30 dataset/config/note.
- dataset audit:
  - total: `{audit_payload['total_count']}`
  - full with final: `{audit_payload['full_with_final_count']}`
  - route-only: `{audit_payload['route_only_count']}`
  - route-only direct/reason: `{audit_payload['route_only_direct_rows']} / {audit_payload['route_only_reason_rows']}`
  - route-only classifier prompt: `{audit_payload['route_only_classifier_prompt_count']}`
  - route-only full extraction prompt: `{audit_payload['route_only_full_extraction_prompt_count']}`

## Result

Pending.

## Conclusion

Pending.

## Next

- train this branch.
- run devpick and route-NLL execution gate.
- run formal only if JSON validity, direct retention, reason headroom, and NLL execution gate pass.
"""
    write_text(note, body)
    return note


def main():
    timestamp = now_iso()
    audit_payload = audit()
    cfg_path, output_dir = make_config()
    note = make_note(cfg_path, output_dir, audit_payload, timestamp)
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "note": note.as_posix(),
                "audit": audit_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
