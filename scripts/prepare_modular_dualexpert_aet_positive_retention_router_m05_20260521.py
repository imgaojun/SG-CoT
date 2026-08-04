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


BRANCH = "aet_positive_retention_router_m05_routecls_noauxwarm_lr2e6_save50"
LABEL_SOURCE = "modular_d1930_r2058_aet_positive_retention_m05"
SOURCE_LABEL = "modular_d1930_r2058_aet_stable_m02"
METHOD_ID = "m05"
TITLE = "Stage2 A/E/T Positive-Retention Router M05 D1930/R2058"
OBJECTIVE = "Prepare a route-only selector dataset that distills the successful m02 positive-retention window into train-time weights."
GOAL = (
    "Construct and audit m05 labels/datasets before training: retained stable positives get higher weight, "
    "while failed-neighbor-style reason-looking direct cases become hard negatives."
)
RULE = (
    "reason iff m02 stable A/E/T-safe positive; train rows are weighted for positive retention, "
    "with reason-looking failed-neighbor hard negatives and safe-unstable hard negatives"
)
WEIGHTS = {
    "retained_positive": 6,
    "safe_unstable_hard_negative": 2,
    "failed_neighbor_hard_negative": 4,
    "ordinary_direct": 1,
}


def now_label():
    return stable.base.now_iso()


def wnd_id(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id") or row.get("id")


def source_label_path(split):
    return stable.base.label_path(SOURCE_LABEL, split)


def m05_label_path(split):
    return stable.base.label_path(LABEL_SOURCE, split)


def is_failed_neighbor_style(label_row):
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
    return max(gains) >= 0.005 and (gains[0] < 0.0 or gains[1] < 0.0 or gains[2] < 0.0)


def teacher_class(label_row):
    if label_row.get("route_label") == "reason":
        return "retained_positive"
    if label_row.get("hard_negative"):
        return "safe_unstable_hard_negative"
    if is_failed_neighbor_style(label_row):
        return "failed_neighbor_hard_negative"
    return "ordinary_direct"


def transform_labels_for_split(split):
    source_rows = stable.base.load_jsonl(source_label_path(split))
    rows = []
    counts = Counter()
    for row in source_rows:
        item = dict(row)
        cls = teacher_class(row)
        counts[cls] += 1
        item["label_source"] = LABEL_SOURCE
        item["label_rule"] = RULE
        item["m05_source_label_source"] = SOURCE_LABEL
        item["m05_teacher_class"] = cls
        item["m05_weight_repeat"] = WEIGHTS[cls] if split == "train" else 1
        item["m05_retention_positive"] = cls == "retained_positive"
        item["m05_failed_neighbor_hard_negative"] = cls == "failed_neighbor_hard_negative"
        item["m05_safe_unstable_hard_negative"] = cls == "safe_unstable_hard_negative"
        rows.append(item)
    out = m05_label_path(split)
    stable.base.write_jsonl(out, rows)
    summary_path = stable.base.label_summary_path(LABEL_SOURCE, split)
    summary = stable.base.summarize_labels(
        rows,
        stable.base.PREDICTIONS[split]["direct"] if split in stable.base.PREDICTIONS else stable.base.FORMAL_DIRECT_ROOT / split / "predictions.jsonl",
        stable.base.PREDICTIONS[split]["reason"] if split in stable.base.PREDICTIONS else stable.base.FORMAL_REASON_ROOT / split / "predictions.jsonl",
        out,
        {"label_source": LABEL_SOURCE, "rule": RULE},
        split,
    )
    summary["m05_teacher_class_counts"] = dict(counts)
    stable.base.write_json(summary_path, summary)
    return dict(counts)


def build_all_labels():
    return {split: transform_labels_for_split(split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]}


def load_m05_train_label_map():
    return {row["wnd_id"]: row for row in stable.base.load_jsonl(m05_label_path("train"))}


def duplicate_train_dataset():
    dataset_path = stable.base.DATA_DIR / f"{stable.base.DATA_PREFIX}_{BRANCH}_train_pos.jsonl"
    meta_path = stable.base.DATA_DIR / f"{stable.base.DATA_PREFIX}_{BRANCH}_train_pos.meta.json"
    label_map = load_m05_train_label_map()
    source_rows = stable.base.load_jsonl(dataset_path)
    duplicated = []
    class_counts = Counter()
    weighted_class_counts = Counter()
    missing = []
    for row in source_rows:
        key = wnd_id(row)
        label_row = label_map.get(key)
        if label_row is None:
            missing.append(key)
            continue
        cls = label_row["m05_teacher_class"]
        repeat = WEIGHTS[cls]
        class_counts[cls] += 1
        weighted_class_counts[cls] += repeat
        for dup_idx in range(repeat):
            item = copy.deepcopy(row)
            meta = dict(item.get("meta") or {})
            meta["m05_weight_class"] = cls
            meta["m05_weight_repeat"] = repeat
            meta["m05_duplicate_index"] = dup_idx
            meta["m05_teacher_source"] = (
                "positive_retention_rank425_500"
                if cls == "retained_positive"
                else "failed_neighbor_rank300_375_or_275_375"
                if cls == "failed_neighbor_hard_negative"
                else cls
            )
            if dup_idx > 0:
                meta["adaptive_pair_source"] = f"m05_{cls}_duplicate_{dup_idx}"
            item["meta"] = meta
            duplicated.append(item)
    if missing:
        raise RuntimeError(f"missing m05 labels for {len(missing)} train rows; first={missing[0]}")
    stable.base.write_jsonl(dataset_path, duplicated)
    meta = stable.base.load_json(meta_path)
    meta["num_examples_before_m05_duplication"] = len(source_rows)
    meta["num_examples"] = len(duplicated)
    meta["audit"] = audit_rows(duplicated)
    meta["m05_weighting"] = {
        "strategy": "weighted_by_row_duplication",
        "weights": WEIGHTS,
        "unweighted_class_counts": dict(class_counts),
        "weighted_class_counts": dict(weighted_class_counts),
    }
    stable.base.write_json(meta_path, meta)
    update_dataset_info(stable.base.DATA_DIR, f"{stable.base.DATA_PREFIX}_{BRANCH}_train_pos", dataset_path.name)
    return meta["m05_weighting"]


def quantiles(values):
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    vals = sorted(values)
    def pick(frac):
        return vals[round((len(vals) - 1) * frac)]
    return {
        "min": vals[0],
        "p25": pick(0.25),
        "median": pick(0.5),
        "p75": pick(0.75),
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
    }


def class_gain_summary(labels, cls):
    rows = [row for row in labels if row["m05_teacher_class"] == cls]
    return {
        "count": len(rows),
        "argument_gain": quantiles([float(row.get("argument_gain", 0.0) or 0.0) for row in rows]),
        "event_gain": quantiles([float(row.get("event_gain", 0.0) or 0.0) for row in rows]),
        "trigger_gain": quantiles([float(row.get("trigger_gain", 0.0) or 0.0) for row in rows]),
        "reason_gain": quantiles([float(row.get("reason_gain", 0.0) or 0.0) for row in rows]),
        "stable_bucket_rate": (
            sum(1 for row in rows if row.get("stable_reason_bucket")) / len(rows)
            if rows else 0.0
        ),
        "hard_negative_rate": (
            sum(1 for row in rows if row.get("hard_negative")) / len(rows)
            if rows else 0.0
        ),
    }


def write_audit_report(label_class_counts, weighting):
    labels_by_split = {
        split: stable.base.load_jsonl(m05_label_path(split))
        for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]
    }
    dataset_audits = {
        split: stable.base.dataset_audit(BRANCH, split)
        for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]
    }
    label_summaries = {
        split: stable.base.label_summary({"label_source": LABEL_SOURCE}, split)
        for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]
    }
    class_summaries = {
        split: {
            cls: class_gain_summary(labels_by_split[split], cls)
            for cls in WEIGHTS
        }
        for split in ["train", "dev_seen"]
    }
    payload = {
        "branch": BRANCH,
        "label_source": LABEL_SOURCE,
        "source_label": SOURCE_LABEL,
        "rule": RULE,
        "weights": WEIGHTS,
        "label_class_counts": label_class_counts,
        "weighting": weighting,
        "dataset_audits": dataset_audits,
        "label_summaries": label_summaries,
        "class_summaries": class_summaries,
        "training_recommendation": (
            "train_m05_candidate"
            if class_summaries["train"]["retained_positive"]["count"] >= 50
            and class_summaries["train"]["failed_neighbor_hard_negative"]["count"] >= 30
            else "do_not_train_yet"
        ),
    }
    out_json = stable.base.REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_m05_dataset_audit.json"
    out_md = stable.base.REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_m05_dataset_audit.md"
    stable.base.write_json(out_json, payload)
    lines = [
        "# M05 Positive-Retention Dataset Audit",
        "",
        "This audit checks whether the m05 positive-retention route-selector dataset is ready for training.",
        "",
        "## Class Counts",
        "",
        "| split | retained positive | safe-unstable hard neg | failed-neighbor hard neg | ordinary direct |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]:
        counts = label_class_counts[split]
        lines.append(
            "| {split} | {rp} | {safe} | {failed} | {ordinary} |".format(
                split=split,
                rp=counts.get("retained_positive", 0),
                safe=counts.get("safe_unstable_hard_negative", 0),
                failed=counts.get("failed_neighbor_hard_negative", 0),
                ordinary=counts.get("ordinary_direct", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Train Weighting",
            "",
            f"`{json.dumps(weighting, ensure_ascii=False)}`",
            "",
            "## Recommendation",
            "",
            f"- `{payload['training_recommendation']}`.",
            "- Retained positives and failed-neighbor hard negatives are both present in train, so the first m05 training run is reasonable if GPU budget is available.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_json, out_md, payload


def make_note(cfg_path, output_dir, timestamp, audit_md, audit_payload):
    exp_id = "2026-05-21_stage2_aet_positive_retention_router_m05_routecls_noauxwarm_lr2e6_save50_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = stable.base.EXPERIMENT_DIR / f"{exp_id}.md"
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
  - positive-retention
  - m05
objective: {OBJECTIVE}
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
  reports:
    - {audit_md}
related:
  experiments:
    - {stable.base.REPO / 'experiments/2026-05-21_stage2_aet_m05_teacher_target_diagnosis_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  direct_expert: D1930_forced_direct
  reason_expert: R2058_forced_reason
  label_source: {LABEL_SOURCE}
  source_label_source: {SOURCE_LABEL}
  route_label_rule: {RULE}
  weighting_strategy: weighted_by_row_duplication
  retained_positive_weight: {WEIGHTS['retained_positive']}
  safe_unstable_hard_negative_weight: {WEIGHTS['safe_unstable_hard_negative']}
  failed_neighbor_hard_negative_weight: {WEIGHTS['failed_neighbor_hard_negative']}
  ordinary_direct_weight: {WEIGHTS['ordinary_direct']}
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
- audit report: `{audit_md.relative_to(stable.base.REPO)}`
- training recommendation from audit: `{audit_payload['training_recommendation']}`

## Commands

```bash
cd {stable.base.REPO}
PYTHONDONTWRITEBYTECODE=1 python3 scripts/prepare_modular_dualexpert_aet_positive_retention_router_m05_20260521.py
bash scripts/launch_modular_dualexpert_aet_positive_retention_router_m05_train_20260521.sh {BRANCH}=<gpu>
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- built m05 labels from m02 stable labels.
- built route-only classifier datasets.
- duplicated train rows by m05 teacher class.
- wrote config, dataset audit, and experiment note.

## Result

Pending training.

## Conclusion

Pending.

## Next

- review dataset audit.
- launch training only if retained positives and failed-neighbor hard negatives look sufficient.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = now_label()
    label_class_counts = build_all_labels()
    old_oversample = stable.base.ROUTE_REASON_OVERSAMPLE
    try:
        stable.base.ROUTE_REASON_OVERSAMPLE = 1
        stable.base.build_datasets(BRANCH, {"label_source": LABEL_SOURCE})
    finally:
        stable.base.ROUTE_REASON_OVERSAMPLE = old_oversample
    weighting = duplicate_train_dataset()
    cfg_path, output_dir = stable.base.make_config(BRANCH)
    audit_json, audit_md, audit_payload = write_audit_report(label_class_counts, weighting)
    note = make_note(cfg_path, output_dir, timestamp, audit_md, audit_payload)
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "label_source": LABEL_SOURCE,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "audit_json": audit_json.as_posix(),
                "audit_md": audit_md.as_posix(),
                "note": note.as_posix(),
                "training_recommendation": audit_payload["training_recommendation"],
                "weighting": weighting,
                "label_class_counts": label_class_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
