#!/usr/bin/env python3
import copy
import json
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.prepare_modular_dualexpert_aet_stable_router_m02_20260520 as stable  # noqa: E402
from src.stage2_cot.build_adaptive_route_reasoning_dataset import audit_rows  # noqa: E402
from src.stage2_data.build_formal_stage2_dataset import update_dataset_info  # noqa: E402


BRANCH = "aet_union_distill_router_m07_routecls_noauxwarm_lr2e6_save50"
LABEL_SOURCE = "modular_d1930_r2058_aet_union_distill_m07"
SOURCE_LABEL = "modular_d1930_r2058_aet_stable_m02"
TITLE = "Stage2 A/E/T Union-Distill Router M07 D1930/R2058"
OBJECTIVE = "Train a full route-only selector that distills m02 Event retention plus m05-style Argument/Trigger add-ons with trigger-harm hard negatives."
GOAL = "Build and train a full selector instead of composing post-hoc rules, targeting m06 union-level aggregate gains with better trigger-risk control."
RULE = (
    "reason iff m02 stable Event-retention positive or A/T add-on positive; direct for trigger-harm, "
    "failed-neighbor, and ordinary negatives"
)
POSITIVE_CLASSES = {"event_retention_positive", "trigger_argument_addon_positive"}
WEIGHTS = {
    "event_retention_positive": 6,
    "trigger_argument_addon_positive": 5,
    "trigger_harm_hard_negative": 5,
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


def m07_label_path(split):
    return stable.base.label_path(LABEL_SOURCE, split)


def f(row, key):
    return float(row.get(key, 0.0) or 0.0)


def is_failed_neighbor_style(label_row):
    if label_row.get("route_label") != "direct":
        return False
    if label_row.get("hard_negative"):
        return False
    if not label_row.get("reason_valid_json"):
        return False
    gains = [f(label_row, "argument_gain"), f(label_row, "event_gain"), f(label_row, "trigger_gain")]
    return max(gains) >= 0.005 and (gains[0] < 0.0 or gains[1] < 0.0 or gains[2] < 0.0)


def teacher_class(label_row):
    arg_gain = f(label_row, "argument_gain")
    event_gain = f(label_row, "event_gain")
    trigger_gain = f(label_row, "trigger_gain")
    valid = bool(label_row.get("reason_valid_json"))
    if label_row.get("route_label") == "reason":
        return "event_retention_positive"
    if valid and arg_gain >= 0.005 and event_gain >= 0.0 and trigger_gain >= 0.005:
        return "trigger_argument_addon_positive"
    if valid and max(arg_gain, event_gain, trigger_gain) >= 0.005 and (
        trigger_gain < -0.002 or min(arg_gain, event_gain, trigger_gain) < -0.01
    ):
        return "trigger_harm_hard_negative"
    if label_row.get("hard_negative") or is_failed_neighbor_style(label_row):
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
        item["route_label"] = "reason" if cls in POSITIVE_CLASSES else "direct"
        item["label_source"] = LABEL_SOURCE
        item["label_rule"] = RULE
        item["m07_source_label_source"] = SOURCE_LABEL
        item["m07_teacher_class"] = cls
        item["m07_weight_repeat"] = WEIGHTS[cls] if split == "train" else 1
        item["m07_event_retention_positive"] = cls == "event_retention_positive"
        item["m07_trigger_argument_addon_positive"] = cls == "trigger_argument_addon_positive"
        item["m07_trigger_harm_hard_negative"] = cls == "trigger_harm_hard_negative"
        item["m07_failed_neighbor_hard_negative"] = cls == "failed_neighbor_hard_negative"
        rows.append(item)
    out = m07_label_path(split)
    stable.base.write_jsonl(out, rows)
    summary_path = stable.base.label_summary_path(LABEL_SOURCE, split)
    direct_path = (
        stable.base.PREDICTIONS[split]["direct"]
        if split in stable.base.PREDICTIONS
        else stable.base.FORMAL_DIRECT_ROOT / split / "predictions.jsonl"
    )
    reason_path = (
        stable.base.PREDICTIONS[split]["reason"]
        if split in stable.base.PREDICTIONS
        else stable.base.FORMAL_REASON_ROOT / split / "predictions.jsonl"
    )
    summary = stable.base.summarize_labels(
        rows,
        direct_path,
        reason_path,
        out,
        {"label_source": LABEL_SOURCE, "rule": RULE},
        split,
    )
    summary["m07_teacher_class_counts"] = dict(counts)
    stable.base.write_json(summary_path, summary)
    return dict(counts)


def build_all_labels():
    return {split: transform_labels_for_split(split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]}


def load_m07_train_label_map():
    return {row["wnd_id"]: row for row in stable.base.load_jsonl(m07_label_path("train"))}


def duplicate_train_dataset():
    dataset_path = stable.base.DATA_DIR / f"{stable.base.DATA_PREFIX}_{BRANCH}_train_pos.jsonl"
    meta_path = stable.base.DATA_DIR / f"{stable.base.DATA_PREFIX}_{BRANCH}_train_pos.meta.json"
    label_map = load_m07_train_label_map()
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
        cls = label_row["m07_teacher_class"]
        repeat = WEIGHTS[cls]
        class_counts[cls] += 1
        weighted_class_counts[cls] += repeat
        for dup_idx in range(repeat):
            item = copy.deepcopy(row)
            meta = dict(item.get("meta") or {})
            meta["m07_weight_class"] = cls
            meta["m07_weight_repeat"] = repeat
            meta["m07_duplicate_index"] = dup_idx
            if dup_idx > 0:
                meta["adaptive_pair_source"] = f"m07_{cls}_duplicate_{dup_idx}"
            item["meta"] = meta
            duplicated.append(item)
    if missing:
        raise RuntimeError(f"missing m07 labels for {len(missing)} train rows; first={missing[0]}")
    stable.base.write_jsonl(dataset_path, duplicated)
    meta = stable.base.load_json(meta_path)
    meta["num_examples_before_m07_duplication"] = len(source_rows)
    meta["num_examples"] = len(duplicated)
    meta["audit"] = audit_rows(duplicated)
    meta["m07_weighting"] = {
        "strategy": "weighted_by_row_duplication",
        "weights": WEIGHTS,
        "unweighted_class_counts": dict(class_counts),
        "weighted_class_counts": dict(weighted_class_counts),
    }
    stable.base.write_json(meta_path, meta)
    update_dataset_info(stable.base.DATA_DIR, f"{stable.base.DATA_PREFIX}_{BRANCH}_train_pos", dataset_path.name)
    return meta["m07_weighting"]


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
    rows = [row for row in labels if row["m07_teacher_class"] == cls]
    return {
        "count": len(rows),
        "argument_gain": quantiles([f(row, "argument_gain") for row in rows]),
        "event_gain": quantiles([f(row, "event_gain") for row in rows]),
        "trigger_gain": quantiles([f(row, "trigger_gain") for row in rows]),
        "reason_gain": quantiles([f(row, "reason_gain") for row in rows]),
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
    splits = ["train", "dev_seen", "test", "test_seen", "test_unseen"]
    labels_by_split = {split: stable.base.load_jsonl(m07_label_path(split)) for split in splits}
    dataset_audits = {split: stable.base.dataset_audit(BRANCH, split) for split in splits}
    label_summaries = {split: stable.base.label_summary({"label_source": LABEL_SOURCE}, split) for split in splits}
    class_summaries = {
        split: {cls: class_gain_summary(labels_by_split[split], cls) for cls in WEIGHTS}
        for split in ["train", "dev_seen"]
    }
    payload = {
        "branch": BRANCH,
        "label_source": LABEL_SOURCE,
        "source_label": SOURCE_LABEL,
        "rule": RULE,
        "weights": WEIGHTS,
        "positive_classes": sorted(POSITIVE_CLASSES),
        "label_class_counts": label_class_counts,
        "weighting": weighting,
        "dataset_audits": dataset_audits,
        "label_summaries": label_summaries,
        "class_summaries": class_summaries,
        "training_recommendation": (
            "train_m07_candidate"
            if class_summaries["train"]["event_retention_positive"]["count"] >= 50
            and class_summaries["train"]["trigger_argument_addon_positive"]["count"] >= 30
            and class_summaries["train"]["trigger_harm_hard_negative"]["count"] >= 30
            else "do_not_train_yet"
        ),
    }
    out_json = stable.base.REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_m07_dataset_audit.json"
    out_md = stable.base.REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_m07_dataset_audit.md"
    stable.base.write_json(out_json, payload)
    lines = [
        "# M07 Union-Distill Dataset Audit",
        "",
        "This audit checks whether the m07 full route-selector dataset is ready for training.",
        "",
        "## Class Counts",
        "",
        "| split | event-retention pos | A/T add-on pos | trigger-harm hard neg | failed-neighbor hard neg | ordinary direct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in splits:
        counts = label_class_counts[split]
        lines.append(
            "| {split} | {event} | {addon} | {trigger} | {failed} | {ordinary} |".format(
                split=split,
                event=counts.get("event_retention_positive", 0),
                addon=counts.get("trigger_argument_addon_positive", 0),
                trigger=counts.get("trigger_harm_hard_negative", 0),
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
            "- The first m07 run should test whether one full selector can learn the m06 union-like boundary.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_json, out_md, payload


def make_note(cfg_path, output_dir, timestamp, audit_md, audit_payload):
    exp_id = "2026-05-21_stage2_aet_union_distill_router_m07_routecls_noauxwarm_lr2e6_save50_d1930_r2058_richere_split1_oracle_mixed_noise_qwen3_1_7b"
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
  - union-distill
  - m07
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
    - {stable.base.REPO / 'experiments/2026-05-21_stage2_aet_m06_combo_selectors_m02_pr_m05_lowbudget_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  direct_expert: D1930_forced_direct
  reason_expert: R2058_forced_reason
  label_source: {LABEL_SOURCE}
  source_label_source: {SOURCE_LABEL}
  route_label_rule: {RULE}
  weighting_strategy: weighted_by_row_duplication
  event_retention_positive_weight: {WEIGHTS['event_retention_positive']}
  trigger_argument_addon_positive_weight: {WEIGHTS['trigger_argument_addon_positive']}
  trigger_harm_hard_negative_weight: {WEIGHTS['trigger_harm_hard_negative']}
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
PYTHONDONTWRITEBYTECODE=1 python3 scripts/prepare_modular_dualexpert_aet_union_distill_router_m07_20260521.py
bash scripts/launch_modular_dualexpert_aet_union_distill_router_m07_train_20260521.sh {BRANCH}=<gpu>
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- built m07 labels from m02 stable labels with A/T add-on positives and trigger-harm hard negatives.
- built route-only classifier datasets.
- duplicated train rows by m07 teacher class.
- wrote config, dataset audit, and experiment note.

## Result

Pending training.

## Conclusion

Pending.

## Next

- review dataset audit.
- launch training if m07 has enough event-retention positives, A/T add-on positives, and trigger-harm hard negatives.
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
