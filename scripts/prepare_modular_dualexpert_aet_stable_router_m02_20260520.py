#!/usr/bin/env python3
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.prepare_modular_dualexpert_aet_router_m01_20260520 as base


BRANCH = "aet_stable_router_m02_routecls_noauxwarm_lr2e6_save50"
LABEL_SOURCE = "modular_d1930_r2058_aet_stable_m02"
TITLE = "Stage2 A/E/T Stable Router M02 D1930/R2058"
OBJECTIVE = "Train a route-only selector with train/dev bucket-stable A/E/T-safe labels and explicit hard negatives."
GOAL = "Train a non-leaking router whose positive reason labels are restricted to train/dev-stable A/E/T-safe buckets, reducing rank-region drift seen in m01."
RULE = (
    "reason iff reason_valid_json and A/E/T-safe gain, with train/dev bucket stability; "
    "hard negatives are safe-looking examples from unstable buckets"
)
REASON_OVERSAMPLE = 4
BUCKET_MIN_COUNT = 2
BUCKET_MAX_HARM_RATE = 0.35
BUCKET_MIN_MEAN_GAIN = 0.02


def metric(row, name):
    return float(row.get(name, 0.0) or 0.0)


def gains(direct, reason):
    return {
        "argument_gain": metric(reason, "argument_f1") - metric(direct, "argument_f1"),
        "event_gain": metric(reason, "event_f1") - metric(direct, "event_f1"),
        "trigger_gain": metric(reason, "trigger_f1") - metric(direct, "trigger_f1"),
    }


def is_aet_safe(g, valid):
    return (
        valid
        and g["argument_gain"] >= 0.0
        and g["event_gain"] >= 0.0
        and g["trigger_gain"] >= -0.002
        and max(g["argument_gain"], g["event_gain"], g["trigger_gain"]) >= 0.005
    )


def event_count(row):
    pred = row.get("predicted") or row.get("final_predicted") or {}
    events = pred.get("events") if isinstance(pred, dict) else []
    return len(events) if isinstance(events, list) else 0


def argument_count(row):
    pred = row.get("predicted") or row.get("final_predicted") or {}
    events = pred.get("events") if isinstance(pred, dict) else []
    if not isinstance(events, list):
        return 0
    total = 0
    for event in events:
        args = event.get("arguments") if isinstance(event, dict) else []
        if isinstance(args, list):
            total += len(args)
    return total


def bucket_key(direct, reason):
    meta = direct.get("meta") or {}
    candidate_types = meta.get("candidate_types") or []
    families = sorted({str(t).split(":", 1)[0] for t in candidate_types})
    family_sig = "+".join(families[:3]) if families else "unknown"
    gold_count = len(meta.get("gold_event_types") or [])
    event_delta = max(-2, min(2, event_count(reason) - event_count(direct)))
    arg_delta = max(-4, min(4, argument_count(reason) - argument_count(direct)))
    return f"fam={family_sig}|goldn={gold_count}|ed={event_delta}|ad={arg_delta}"


def collect_bucket_stats():
    stats = defaultdict(lambda: {"count": 0, "safe": 0, "harm": 0, "gain_sum": 0.0})
    for split in ["train", "dev_seen"]:
        paths = base.PREDICTIONS[split]
        direct_rows = base.load_prediction_map(paths["direct"])
        reason_rows = base.load_prediction_map(paths["reason"])
        for key in sorted(set(direct_rows) & set(reason_rows)):
            direct = direct_rows[key]
            reason = reason_rows[key]
            g = gains(direct, reason)
            reason_gain = base.score(reason) - base.score(direct)
            bucket = bucket_key(direct, reason)
            stats[bucket]["count"] += 1
            stats[bucket]["gain_sum"] += reason_gain
            if is_aet_safe(g, base.valid_json(reason)):
                stats[bucket]["safe"] += 1
            if g["argument_gain"] < 0 or g["event_gain"] < 0 or g["trigger_gain"] < -0.002:
                stats[bucket]["harm"] += 1
    out = {}
    for bucket, row in stats.items():
        count = row["count"]
        out[bucket] = {
            **row,
            "mean_gain": row["gain_sum"] / count if count else 0.0,
            "harm_rate": row["harm"] / count if count else 0.0,
            "stable_reason_bucket": (
                count >= BUCKET_MIN_COUNT
                and row["safe"] > 0
                and row["gain_sum"] / count >= BUCKET_MIN_MEAN_GAIN
                and row["harm"] / count <= BUCKET_MAX_HARM_RATE
            ),
        }
    return out


def build_labels_for_split(split, direct_path: Path, reason_path: Path, bucket_stats):
    direct_rows = base.load_prediction_map(direct_path)
    reason_rows = base.load_prediction_map(reason_path)
    common_keys = sorted(set(direct_rows) & set(reason_rows))
    labels = []
    hard_negative_count = 0
    bucket_counts = Counter()
    reason_bucket_counts = Counter()
    for idx, key in enumerate(common_keys):
        direct = direct_rows[key]
        reason = reason_rows[key]
        g = gains(direct, reason)
        reason_is_valid = base.valid_json(reason)
        bucket = bucket_key(direct, reason)
        bucket_info = bucket_stats.get(bucket) or {}
        safe = is_aet_safe(g, reason_is_valid)
        stable_bucket = bool(bucket_info.get("stable_reason_bucket"))
        route_label = "reason" if safe and stable_bucket else "direct"
        hard_negative = safe and not stable_bucket
        hard_negative_count += int(hard_negative)
        bucket_counts[bucket] += 1
        if route_label == "reason":
            reason_bucket_counts[bucket] += 1
        labels.append(
            {
                "idx": idx,
                "wnd_id": key,
                "route_label": route_label,
                "label_source": LABEL_SOURCE,
                "source_split": split,
                "label_rule": RULE,
                "bucket": bucket,
                "bucket_count": bucket_info.get("count", 0),
                "bucket_mean_gain": bucket_info.get("mean_gain", 0.0),
                "bucket_harm_rate": bucket_info.get("harm_rate", 1.0),
                "stable_reason_bucket": stable_bucket,
                "hard_negative": hard_negative,
                "reason_gain": base.score(reason) - base.score(direct),
                "direct_score": base.score(direct),
                "reason_score": base.score(reason),
                "direct_trigger_f1": metric(direct, "trigger_f1"),
                "direct_argument_f1": metric(direct, "argument_f1"),
                "direct_event_f1": metric(direct, "event_f1"),
                "reason_trigger_f1": metric(reason, "trigger_f1"),
                "reason_argument_f1": metric(reason, "argument_f1"),
                "reason_event_f1": metric(reason, "event_f1"),
                **g,
                "direct_valid_json": base.valid_json(direct),
                "reason_valid_json": reason_is_valid,
            }
        )
    out = base.label_path(LABEL_SOURCE, split)
    summary_out = base.label_summary_path(LABEL_SOURCE, split)
    base.write_jsonl(out, labels)
    summary = base.summarize_labels(
        labels,
        direct_path,
        reason_path,
        out,
        {"label_source": LABEL_SOURCE, "rule": RULE},
        split,
    )
    summary["hard_negative_count"] = hard_negative_count
    summary["num_buckets"] = len(bucket_counts)
    summary["num_reason_buckets"] = len(reason_bucket_counts)
    summary["top_reason_buckets"] = reason_bucket_counts.most_common(20)
    base.write_json(summary_out, summary)


def build_all_labels(bucket_stats):
    for split, paths in base.PREDICTIONS.items():
        build_labels_for_split(split, paths["direct"], paths["reason"], bucket_stats)
    for split in ["test", "test_seen", "test_unseen"]:
        build_labels_for_split(
            split,
            base.FORMAL_DIRECT_ROOT / split / "predictions.jsonl",
            base.FORMAL_REASON_ROOT / split / "predictions.jsonl",
            bucket_stats,
        )


def make_note(cfg_path, output_dir, timestamp, bucket_stats):
    exp_id = f"2026-05-20_stage2_{BRANCH}_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = base.EXPERIMENT_DIR / f"{exp_id}.md"
    audits = {split: base.dataset_audit(BRANCH, split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]}
    summaries = {split: base.label_summary({"label_source": LABEL_SOURCE}, split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]}
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
objective: {OBJECTIVE}
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  experiments:
    - {base.REPO / 'experiments/2026-05-20_stage2_aet_router_m01_goldfree_guard_drift_diagnosis_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  direct_expert: D1930_forced_direct
  reason_expert: R2058_forced_reason
  label_source: {LABEL_SOURCE}
  route_label_rule: {RULE}
  route_reason_oversample: {REASON_OVERSAMPLE}
  stable_bucket_min_count: {BUCKET_MIN_COUNT}
  stable_bucket_max_harm_rate: {BUCKET_MAX_HARM_RATE}
  stable_bucket_min_mean_gain: {BUCKET_MIN_MEAN_GAIN}
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
- config: `{cfg_path.relative_to(base.REPO)}`
- output: `{output_dir.relative_to(base.REPO)}`
- model start: `{base.NOAUX_CKPT}`
- label rule: `{RULE}`
- stable buckets: `{len(stable_buckets)}` / `{len(bucket_stats)}`
- train/dev labels come from train/dev paired outputs. Formal labels are probe labels only and must not be used for training or selection.

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
cd {base.REPO}
python3 scripts/prepare_modular_dualexpert_aet_stable_router_m02_20260520.py
bash scripts/launch_modular_dualexpert_aet_stable_router_m02_train_20260520.sh {BRANCH}=<gpu>
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- built train/dev bucket-stable A/E/T labels plus formal probe labels.
- built route-only classifier datasets.
- created training config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- score dev route-choice NLL over saved checkpoints.
- select early/stable dev windows before formal replay.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = base.now_iso()
    bucket_stats = collect_bucket_stats()
    base.write_json(
        base.LABEL_DIR / f"{base.DATA_PREFIX}_{LABEL_SOURCE}_bucket_stats.json",
        bucket_stats,
    )
    build_all_labels(bucket_stats)
    old_oversample = base.ROUTE_REASON_OVERSAMPLE
    try:
        base.ROUTE_REASON_OVERSAMPLE = REASON_OVERSAMPLE
        base.build_datasets(BRANCH, {"label_source": LABEL_SOURCE})
    finally:
        base.ROUTE_REASON_OVERSAMPLE = old_oversample
    cfg_path, output_dir = base.make_config(BRANCH)
    note = make_note(cfg_path, output_dir, timestamp, bucket_stats)
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
                "label_summaries": {
                    split: base.label_summary({"label_source": LABEL_SOURCE}, split)
                    for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
