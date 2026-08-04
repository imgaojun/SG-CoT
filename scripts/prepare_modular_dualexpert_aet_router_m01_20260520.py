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
SCHEMA = REPO / "data/schema/richere-en.event_schema.json"
TZ = timezone(timedelta(hours=8))
ROUTE_REASON_OVERSAMPLE = 6
NOAUX_CKPT = (
    "/workspace/project/outputs/stage2_adaptive_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full/checkpoint-1184"
)
TEMPLATE_CONFIG = CONFIG_DIR / (
    f"{RUN_PREFIX}_outcome15_l15bal30_routecls_noauxwarm_lr2e6_save50_probe_full_stepmatch.yaml"
)


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


BRANCHES = {
    "aet_safe_router_m01_routecls_noauxwarm_lr2e6_save50": {
        "label_source": "modular_d1930_r2058_aet_safe_m01",
        "title": "Stage2 A/E/T Safe Router M01 D1930/R2058",
        "objective": "Train a route-only selector using A/E/T-safe labels where reason must not hurt argument/event and may only slightly hurt trigger.",
        "rule": "reason iff reason_valid_json and argument_gain >= 0 and event_gain >= 0 and trigger_gain >= -0.002 and max(A,E,T gain) >= 0.005",
        "predicate": lambda g, valid: (
            valid
            and g["argument_gain"] >= 0.0
            and g["event_gain"] >= 0.0
            and g["trigger_gain"] >= -0.002
            and max(g["argument_gain"], g["event_gain"], g["trigger_gain"]) >= 0.005
        ),
    },
    "aet_event_router_m01_routecls_noauxwarm_lr2e6_save50": {
        "label_source": "modular_d1930_r2058_aet_event_m01",
        "title": "Stage2 A/E/T Event Router M01 D1930/R2058",
        "objective": "Train a route-only selector using event-positive labels with argument/trigger near-nonnegative constraints.",
        "rule": "reason iff reason_valid_json and event_gain >= 0.005 and argument_gain >= -0.002 and trigger_gain >= -0.002",
        "predicate": lambda g, valid: (
            valid
            and g["event_gain"] >= 0.005
            and g["argument_gain"] >= -0.002
            and g["trigger_gain"] >= -0.002
        ),
    },
}


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def run(cmd):
    subprocess.run(cmd, cwd=REPO, check=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prediction_key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id") or row.get("id")


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def valid_json(row):
    return bool(row.get("valid_final_json", row.get("valid_json", False)))


def score(row):
    return (
        float(row.get("argument_f1", 0.0) or 0.0)
        + float(row.get("event_f1", 0.0) or 0.0)
        + 0.25 * float(row.get("trigger_f1", 0.0) or 0.0)
    )


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def label_path(label_source: str, split: str):
    return LABEL_DIR / f"{DATA_PREFIX}_{label_source}_{split}_labels.jsonl"


def label_summary_path(label_source: str, split: str):
    return LABEL_DIR / f"{DATA_PREFIX}_{label_source}_{split}_labels.summary.json"


def build_labels_for_branch(branch: str, spec: dict, split: str, direct_path: Path, reason_path: Path):
    if not direct_path.exists() or not reason_path.exists():
        raise FileNotFoundError(f"missing paired predictions for {split}: {direct_path}, {reason_path}")
    direct_rows = load_prediction_map(direct_path)
    reason_rows = load_prediction_map(reason_path)
    common_keys = sorted(set(direct_rows) & set(reason_rows))
    labels = []
    for idx, key in enumerate(common_keys):
        direct = direct_rows[key]
        reason = reason_rows[key]
        gains = {
            "argument_gain": float(reason.get("argument_f1", 0.0) or 0.0) - float(direct.get("argument_f1", 0.0) or 0.0),
            "event_gain": float(reason.get("event_f1", 0.0) or 0.0) - float(direct.get("event_f1", 0.0) or 0.0),
            "trigger_gain": float(reason.get("trigger_f1", 0.0) or 0.0) - float(direct.get("trigger_f1", 0.0) or 0.0),
        }
        reason_is_valid = valid_json(reason)
        route_label = "reason" if spec["predicate"](gains, reason_is_valid) else "direct"
        labels.append(
            {
                "idx": idx,
                "wnd_id": key,
                "route_label": route_label,
                "label_source": spec["label_source"],
                "source_split": split,
                "label_rule": spec["rule"],
                "reason_gain": score(reason) - score(direct),
                "direct_score": score(direct),
                "reason_score": score(reason),
                "direct_trigger_f1": float(direct.get("trigger_f1", 0.0) or 0.0),
                "direct_argument_f1": float(direct.get("argument_f1", 0.0) or 0.0),
                "direct_event_f1": float(direct.get("event_f1", 0.0) or 0.0),
                "reason_trigger_f1": float(reason.get("trigger_f1", 0.0) or 0.0),
                "reason_argument_f1": float(reason.get("argument_f1", 0.0) or 0.0),
                "reason_event_f1": float(reason.get("event_f1", 0.0) or 0.0),
                **gains,
                "direct_valid_json": valid_json(direct),
                "reason_valid_json": reason_is_valid,
            }
        )
    out = label_path(spec["label_source"], split)
    summary_out = label_summary_path(spec["label_source"], split)
    write_jsonl(out, labels)
    write_json(summary_out, summarize_labels(labels, direct_path, reason_path, out, spec, split))


def mean(rows, key):
    vals = [row[key] for row in rows]
    return sum(vals) / len(vals) if vals else None


def summarize_labels(labels, direct_path, reason_path, out, spec, split):
    reason_rows = [row for row in labels if row["route_label"] == "reason"]
    total = len(labels)
    return {
        "forced_direct_predictions": direct_path.as_posix(),
        "forced_reason_predictions": reason_path.as_posix(),
        "output_jsonl": out.as_posix(),
        "label_source": spec["label_source"],
        "source_split": split,
        "label_rule": spec["rule"],
        "num_examples": total,
        "direct_count": total - len(reason_rows),
        "reason_count": len(reason_rows),
        "reason_rate": len(reason_rows) / total if total else 0.0,
        "selected_mean_argument_gain": mean(reason_rows, "argument_gain"),
        "selected_mean_event_gain": mean(reason_rows, "event_gain"),
        "selected_mean_trigger_gain": mean(reason_rows, "trigger_gain"),
        "selected_mean_reason_gain": mean(reason_rows, "reason_gain"),
        "direct_valid_json_rate": sum(1 for row in labels if row["direct_valid_json"]) / total if total else 0.0,
        "reason_valid_json_rate": sum(1 for row in labels if row["reason_valid_json"]) / total if total else 0.0,
    }


def build_all_labels(branch: str, spec: dict):
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for split, paths in PREDICTIONS.items():
        build_labels_for_branch(branch, spec, split, paths["direct"], paths["reason"])
    for split in ["test", "test_seen", "test_unseen"]:
        build_labels_for_branch(
            branch,
            spec,
            split,
            FORMAL_DIRECT_ROOT / split / "predictions.jsonl",
            FORMAL_REASON_ROOT / split / "predictions.jsonl",
        )


def build_datasets(branch: str, spec: dict):
    label_source = spec["label_source"]
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
            label_path(label_source, "train").as_posix(),
            "--dev_label_jsonl",
            label_path(label_source, "dev_seen").as_posix(),
            "--test_label_jsonl",
            label_path(label_source, "test").as_posix(),
            "--test_seen_label_jsonl",
            label_path(label_source, "test_seen").as_posix(),
            "--test_unseen_label_jsonl",
            label_path(label_source, "test_unseen").as_posix(),
            "--dataset_dir",
            DATA_DIR.as_posix(),
            "--train_dataset_name",
            f"{DATA_PREFIX}_{branch}_train_pos",
            "--dev_dataset_name",
            f"{DATA_PREFIX}_{branch}_dev_seen_pos",
            "--test_dataset_name",
            f"{DATA_PREFIX}_{branch}_test_pos",
            "--test_seen_dataset_name",
            f"{DATA_PREFIX}_{branch}_test_seen_pos",
            "--test_unseen_dataset_name",
            f"{DATA_PREFIX}_{branch}_test_unseen_pos",
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


def make_config(branch: str):
    config = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    config["model_name_or_path"] = NOAUX_CKPT
    config["dataset"] = f"{DATA_PREFIX}_{branch}_train_pos"
    config["eval_dataset"] = f"{DATA_PREFIX}_{branch}_dev_seen_pos"
    config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full"
    config["learning_rate"] = 2.0e-6
    config["num_train_epochs"] = 1.5
    config["logging_steps"] = 5
    config["save_strategy"] = "steps"
    config["save_steps"] = 50
    config["eval_strategy"] = "steps"
    config["eval_steps"] = 50
    config["load_best_model_at_end"] = False
    cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    write_yaml(cfg_path, config)
    return cfg_path, REPO / config["output_dir"].replace("/workspace/project/", "")


def dataset_audit(branch: str, split: str):
    return load_json(DATA_DIR / f"{DATA_PREFIX}_{branch}_{split}_pos.meta.json")["audit"]


def label_summary(spec: dict, split: str):
    return load_json(label_summary_path(spec["label_source"], split))


def make_note(branch: str, spec: dict, cfg_path: Path, output_dir: Path, timestamp: str):
    exp_id = f"2026-05-20_stage2_{branch}_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    audits = {split: dataset_audit(branch, split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]}
    summaries = {split: label_summary(spec, split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]}
    body = f"""---
id: {exp_id}
title: {spec['title']}
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
  - aet-router
  - route-classification
objective: {spec['objective']}
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  experiments:
    - {REPO / 'experiments/2026-05-20_stage2_modular_dualexpert_utility_router_m02_rank_window_formal_replay_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  direct_expert: D1930_forced_direct
  reason_expert: R2058_forced_reason
  label_source: {spec['label_source']}
  route_label_rule: {spec['rule']}
  route_reason_oversample: {ROUTE_REASON_OVERSAMPLE}
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 1.5
  save_steps: 50
---

# {spec['title']}

## Goal

Train a non-leaking A/E/T-aligned route-only selector and evaluate whether it improves raw Argument/Event/Trigger deltas over forced-direct.

## Setup

- branch: `{branch}`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- model start: `{NOAUX_CKPT}`
- label rule: `{spec['rule']}`
- train/dev labels come from train/dev paired outputs, not formal labels.

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
python3 scripts/prepare_modular_dualexpert_aet_router_m01_20260520.py
bash scripts/launch_modular_dualexpert_aet_router_m01_train_20260520.sh {branch}=<gpu>
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- built D1930/R2058 A/E/T labels for train/dev/formal splits.
- built route-only classifier datasets.
- created training config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch exactly one training run.
- score dev route-choice NLL over saved checkpoints.
- select A/E/T-constrained dev policies before formal replay.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = now_iso()
    out = {}
    for branch, spec in BRANCHES.items():
        build_all_labels(branch, spec)
        build_datasets(branch, spec)
        cfg_path, output_dir = make_config(branch)
        note = make_note(branch, spec, cfg_path, output_dir, timestamp)
        out[branch] = {
            "config": cfg_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "note": note.as_posix(),
            "audits": {split: dataset_audit(branch, split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]},
            "label_summaries": {split: label_summary(spec, split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]},
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
