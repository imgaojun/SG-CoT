#!/usr/bin/env python3
import copy
import json
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.prepare_modular_dualexpert_aet_stable_router_m02_20260520 as stable
from src.stage2_cot.build_adaptive_route_reasoning_dataset import audit_rows
from src.stage2_data.build_formal_stage2_dataset import update_dataset_info


BRANCH = "aet_rankstable_router_m04a_routecls_noauxwarm_lr2e6_save50"
LABEL_SOURCE = "modular_d1930_r2058_aet_rankstable_m04a"
METHOD_ID = "m04a"
TITLE = "Stage2 A/E/T Rank-Stable Router M04A D1930/R2058"
OBJECTIVE = "Train a route-only selector with m02 stable labels plus duplicated hard negatives to reduce rank-region drift."
GOAL = (
    "Test whether train-time hard-negative weighting by duplication makes route-choice NLL rank windows "
    "transfer more stably from dev to formal."
)
RULE = (
    "reason iff reason_valid_json and A/E/T-safe gain, with train/dev bucket stability; "
    "train rows are duplicated by rank-stability class"
)
WEIGHTS = {
    "stable_reason_positive": 5,
    "safe_unstable_hard_negative": 3,
    "harmful_reason_looking_hard_negative": 4,
    "ordinary_direct": 1,
}


def wnd_id(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id") or row.get("id")


def is_harmful_reason_looking(label_row):
    if label_row.get("route_label") != "direct":
        return False
    if label_row.get("hard_negative"):
        return False
    if not label_row.get("reason_valid_json"):
        return False
    gains = [
        float(label_row.get("argument_gain", 0.0) or 0.0),
        float(label_row.get("event_gain", 0.0) or 0.0),
        float(label_row.get("trigger_gain", 0.0) or 0.0),
    ]
    has_meaningful_upside = max(gains) >= 0.005
    has_aet_harm = (
        gains[0] < 0.0
        or gains[1] < 0.0
        or gains[2] < -0.002
    )
    return has_meaningful_upside and has_aet_harm


def weight_class(label_row):
    if label_row.get("route_label") == "reason":
        return "stable_reason_positive"
    if label_row.get("hard_negative"):
        return "safe_unstable_hard_negative"
    if is_harmful_reason_looking(label_row):
        return "harmful_reason_looking_hard_negative"
    return "ordinary_direct"


def load_label_map():
    rows = stable.base.load_jsonl(stable.base.label_path(LABEL_SOURCE, "train"))
    return {row["wnd_id"]: row for row in rows}


def duplicate_train_dataset():
    dataset_path = stable.base.DATA_DIR / f"{stable.base.DATA_PREFIX}_{BRANCH}_train_pos.jsonl"
    meta_path = stable.base.DATA_DIR / f"{stable.base.DATA_PREFIX}_{BRANCH}_train_pos.meta.json"
    label_map = load_label_map()
    source_rows = stable.base.load_jsonl(dataset_path)
    duplicated = []
    class_counts = Counter()
    weighted_class_counts = Counter()
    missing_labels = []
    for row in source_rows:
        key = wnd_id(row)
        label_row = label_map.get(key)
        if label_row is None:
            missing_labels.append(key)
            continue
        cls = weight_class(label_row)
        repeat = WEIGHTS[cls]
        class_counts[cls] += 1
        weighted_class_counts[cls] += repeat
        for dup_idx in range(repeat):
            item = copy.deepcopy(row)
            meta = dict(item.get("meta") or {})
            meta[f"{METHOD_ID}_weight_class"] = cls
            meta[f"{METHOD_ID}_weight_repeat"] = repeat
            meta[f"{METHOD_ID}_duplicate_index"] = dup_idx
            if dup_idx > 0:
                meta["adaptive_pair_source"] = f"{METHOD_ID}_{cls}_duplicate_{dup_idx}"
            item["meta"] = meta
            duplicated.append(item)
    if missing_labels:
        raise RuntimeError(f"missing labels for {len(missing_labels)} train rows; first={missing_labels[0]}")
    stable.base.write_jsonl(dataset_path, duplicated)
    meta = stable.base.load_json(meta_path)
    meta[f"num_examples_before_{METHOD_ID}_duplication"] = len(source_rows)
    meta["num_examples"] = len(duplicated)
    meta["audit"] = audit_rows(duplicated)
    meta[f"{METHOD_ID}_weighting"] = {
        "strategy": "weighted_by_row_duplication",
        "weights": WEIGHTS,
        "unweighted_class_counts": dict(class_counts),
        "weighted_class_counts": dict(weighted_class_counts),
    }
    stable.base.write_json(meta_path, meta)
    update_dataset_info(stable.base.DATA_DIR, f"{stable.base.DATA_PREFIX}_{BRANCH}_train_pos", dataset_path.name)
    return meta[f"{METHOD_ID}_weighting"]


def make_note(cfg_path, output_dir, timestamp, bucket_stats, weighting):
    exp_id = f"2026-05-20_stage2_{BRANCH}_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = stable.base.EXPERIMENT_DIR / f"{exp_id}.md"
    audits = {
        split: stable.base.dataset_audit(BRANCH, split)
        for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]
    }
    summaries = {
        split: stable.base.label_summary({"label_source": LABEL_SOURCE}, split)
        for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]
    }
    stable_buckets = {k: v for k, v in bucket_stats.items() if v["stable_reason_bucket"]}
    body = f"""---
id: {exp_id}
title: {TITLE}
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
  - aet-router
  - route-classification
  - stable-router
  - rank-stability
objective: {OBJECTIVE}
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  experiments:
    - {stable.base.REPO / 'experiments/2026-05-20_stage2_aet_stable_router_m02_routecls_noauxwarm_lr2e6_save50_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  direct_expert: D1930_forced_direct
  reason_expert: R2058_forced_reason
  label_source: {LABEL_SOURCE}
  route_label_rule: {RULE}
  weighting_strategy: weighted_by_row_duplication
  weighting_method_id: {METHOD_ID}
  stable_reason_positive_weight: {WEIGHTS['stable_reason_positive']}
  safe_unstable_hard_negative_weight: {WEIGHTS['safe_unstable_hard_negative']}
  harmful_reason_looking_hard_negative_weight: {WEIGHTS['harmful_reason_looking_hard_negative']}
  ordinary_direct_weight: {WEIGHTS['ordinary_direct']}
  stable_bucket_min_count: {stable.BUCKET_MIN_COUNT}
  stable_bucket_max_harm_rate: {stable.BUCKET_MAX_HARM_RATE}
  stable_bucket_min_mean_gain: {stable.BUCKET_MIN_MEAN_GAIN}
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 1.5
  save_steps: 50
---

# {TITLE}

## Goal

{GOAL}

## Setup

- branch: `{BRANCH}`
- config: `{cfg_path.relative_to(stable.base.REPO)}`
- output: `{output_dir.relative_to(stable.base.REPO)}`
- model start: `{stable.base.NOAUX_CKPT}`
- label rule: `{RULE}`
- stable buckets: `{len(stable_buckets)}` / `{len(bucket_stats)}`
- {METHOD_ID} weighting: `{json.dumps(weighting, ensure_ascii=False)}`
- train/dev labels come from train/dev paired outputs. Formal labels are probe labels only and must not be used for training or selection.
- train rows are duplicated after one-row route-only dataset construction; dev/test/formal datasets remain one-row-per-sample.

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
cd {stable.base.REPO}
PYTHONDONTWRITEBYTECODE=1 python3 scripts/prepare_modular_dualexpert_aet_rankstable_router_{METHOD_ID}_20260520.py
bash scripts/launch_modular_dualexpert_aet_rankstable_router_{METHOD_ID}_train_20260520.sh {BRANCH}=<gpu>
bash scripts/launch_modular_dualexpert_aet_rankstable_router_{METHOD_ID}_route_nll_dev_20260520.sh {BRANCH}=<gpu>
PYTHONDONTWRITEBYTECODE=1 python3 scripts/calibrate_modular_dualexpert_aet_rankstable_router_{METHOD_ID}_dev_20260520.py
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- built m02-style train/dev bucket-stable A/E/T labels plus formal probe labels.
- built route-only classifier datasets.
- duplicated train rows by {METHOD_ID} weight class; dev/test/formal remain one-row-per-sample.
- created training config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- score dev route-choice NLL over saved checkpoints.
- select low-budget fold-stable windows before formal replay.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = stable.base.now_iso()
    stable.BRANCH = BRANCH
    stable.LABEL_SOURCE = LABEL_SOURCE
    stable.TITLE = TITLE
    stable.OBJECTIVE = OBJECTIVE
    stable.GOAL = GOAL
    stable.RULE = RULE
    stable.REASON_OVERSAMPLE = 1
    bucket_stats = stable.collect_bucket_stats()
    stable.base.write_json(
        stable.base.LABEL_DIR / f"{stable.base.DATA_PREFIX}_{LABEL_SOURCE}_bucket_stats.json",
        bucket_stats,
    )
    stable.build_all_labels(bucket_stats)
    old_oversample = stable.base.ROUTE_REASON_OVERSAMPLE
    try:
        stable.base.ROUTE_REASON_OVERSAMPLE = 1
        stable.base.build_datasets(BRANCH, {"label_source": LABEL_SOURCE})
    finally:
        stable.base.ROUTE_REASON_OVERSAMPLE = old_oversample
    weighting = duplicate_train_dataset()
    cfg_path, output_dir = stable.base.make_config(BRANCH)
    note = make_note(cfg_path, output_dir, timestamp, bucket_stats, weighting)
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "label_source": LABEL_SOURCE,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "note": note.as_posix(),
                "stable_buckets": sum(1 for row in bucket_stats.values() if row["stable_reason_bucket"]),
                "total_buckets": len(bucket_stats),
                "weighting_method_id": METHOD_ID,
                f"{METHOD_ID}_weighting": weighting,
                "dataset_audits": {
                    split: stable.base.dataset_audit(BRANCH, split)
                    for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]
                },
                "label_summaries": {
                    split: stable.base.label_summary({"label_source": LABEL_SOURCE}, split)
                    for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
