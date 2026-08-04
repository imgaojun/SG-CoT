import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH = "outcome15_l15bal30_routecls_noauxwarm_lr2e6_save50_probe"
NOAUX_CKPT = (
    "/workspace/project/outputs/stage2_adaptive_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full/checkpoint-1184"
)
TEMPLATE_CONFIG = CONFIG_DIR / (
    f"{RUN_PREFIX}_outcome15_l15bal30_routecls_balanced_probe_full_stepmatch.yaml"
)
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(dataset_name: str):
    meta = REPO / f"data/stage2_adaptive_datasets/{dataset_name}.meta.json"
    return load_json(meta)["audit"]


def make_config():
    with TEMPLATE_CONFIG.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
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


def make_note(cfg_path: Path, output_dir: Path, train_audit: dict, dev_audit: dict, timestamp: str):
    exp_id = f"2026-05-17_stage2_adaptive_{BRANCH}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    if note.exists():
        return note
    title = "Stage2 Adaptive Outcome15 Route-Classifier No-Aux Warm Start Probe RichERE Split1 Qwen3-1.7B"
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
  - route-classification
  - route-nll
  - richere
  - qwen3-1.7b
objective: Test whether route-choice supervision becomes more stable when the route classifier is warm-started from the clean no-aux extractor and swept with finer checkpoints.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  plans:
    - {REPO / 'PLANS.md'}
  reports:
    - {REPO / 'reports/2026-05-13_stage2_adaptive_outcome_router_execution_analysis.md'}
context:
  dataset: RichERE
  split: split1
  candidate_source: oracle_mixed_noise_top10_shuffle
  model: Qwen3-1.7B
  warm_start_from_noaux: true
  route_label_source: outcome_l15bal30_15
  target_style: route_classifier
  route_reason_oversample: 6
  learning_rate: 2.0e-6
  num_train_epochs: 1.5
  save_steps: 50
---

# {title}

## Goal

Answer one narrow question: can route-choice scoring avoid the previous routecls collapse if we start from the clean no-aux adaptive extractor, use a lower LR, and keep dense checkpoint observations?

## Setup

- branch: `{BRANCH}`
- train dataset: `{DATA_PREFIX}_{BRANCH}_train_pos`
- dev dataset: `{DATA_PREFIX}_{BRANCH}_dev_seen_pos`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- model start: `{NOAUX_CKPT}`
- train audit: `{json.dumps(train_audit, sort_keys=True)}`
- dev audit: `{json.dumps(dev_audit, sort_keys=True)}`

Expected shape:

- route-only classifier-prompt train rows: `3536`
- train direct/reason rows after oversampling: `1760 / 1776`
- route-only classifier-prompt dev rows: `197`
- dev direct/reason rows: `167 / 30`
- full extraction rows in this selector dataset: `0`

## Commands

```bash
bash scripts/build_adaptive_outcome_routecls_noauxwarm_20260517.sh
bash scripts/launch_adaptive_outcome_routecls_noauxwarm_train_20260517.sh {BRANCH}=<gpu>
bash scripts/run_adaptive_outcome_routecls_noauxwarm_after_train_20260517.sh
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- created the no-aux warm-start route-classifier dataset/config/note.
- planned route generation and route-NLL scoring over all saved checkpoints.
- planned dev-only execution simulation against the existing `l15bal30_ckpt942` forced-direct/forced-reason mining outputs for comparability with the 2026-05-13 routecls probe.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- sweep route generation and route-NLL over saved checkpoints.
- keep this as a dev diagnostic unless top-budget route-NLL is stable enough to justify a separate formal selector test.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = now_iso()
    cfg_path, output_dir = make_config()
    train_audit = audit(f"{DATA_PREFIX}_{BRANCH}_train_pos")
    dev_audit = audit(f"{DATA_PREFIX}_{BRANCH}_dev_seen_pos")
    note = make_note(cfg_path, output_dir, train_audit, dev_audit, timestamp)
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "note": note.as_posix(),
                "train_audit": train_audit,
                "dev_audit": dev_audit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
