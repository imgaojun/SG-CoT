#!/usr/bin/env python3
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.prepare_multibudget_route_selectors_m08_m09_20260521 as base  # noqa: E402
from src.stage2_cot.build_adaptive_route_reasoning_dataset import audit_rows  # noqa: E402
from src.stage2_data.build_formal_stage2_dataset import update_dataset_info  # noqa: E402


TZ = timezone(timedelta(hours=8))
BRANCH = "multibudget_retention_router_m11_routecls_noauxwarm_lr2e6_save50"
LABEL_SOURCE = "multibudget_retention_m11_direct_light_mid_full"
TITLE = "Stage2 Multibudget Retention Router M11"
OBJECTIVE = "Train a retention-aware multibudget route-only selector using safe budget positives and failed-window hard negatives."
ROUTES = ["direct", "reason_light", "reason_mid", "reason_full"]
WEIGHTS = {
    "direct": 1,
    "reason_light": 5,
    "reason_mid": 5,
    "reason_full": 6,
    "safe_unstable_hard_negative": 2,
    "failed_window_hard_negative": 4,
}

M09_DEV = REPO / "reports/artifacts/2026-05-21_stage2_multibudget_fourclass_router_m09_dev.json"
M10_DEV = REPO / "reports/artifacts/2026-05-21_stage2_multibudget_m10_m06_union_plus_m09_addon_dev.json"


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id") or row.get("id")


def metric(row, name):
    return float(row.get(name, 0.0) or 0.0)


def utility(row):
    return metric(row, "argument_f1") + metric(row, "event_f1") + 0.25 * metric(row, "trigger_f1")


def valid(row):
    return bool(row.get("valid_final_json", row.get("valid_json", False)))


def gain_dict(route, row, direct):
    return {
        "route": route,
        "argument_gain": metric(row, "argument_f1") - metric(direct, "argument_f1"),
        "event_gain": metric(row, "event_f1") - metric(direct, "event_f1"),
        "trigger_gain": metric(row, "trigger_f1") - metric(direct, "trigger_f1"),
        "utility_gain": utility(row) - utility(direct),
        "valid_json": valid(row),
    }


def is_safe_positive(g):
    return (
        g["valid_json"]
        and g["argument_gain"] >= 0.0
        and g["event_gain"] >= 0.0
        and g["trigger_gain"] >= -0.002
        and max(g["argument_gain"], g["event_gain"], g["trigger_gain"]) >= 0.005
        and g["utility_gain"] >= 0.005
    )


def is_failed_window_style(g):
    return (
        g["valid_json"]
        and max(g["argument_gain"], g["event_gain"], g["trigger_gain"]) >= 0.005
        and (g["argument_gain"] < 0.0 or g["event_gain"] < 0.0 or g["trigger_gain"] < -0.002)
    )


def choose_lowest_safe(candidates, wnd_id):
    direct = candidates["direct"][wnd_id]
    gains = {route: gain_dict(route, candidates[route][wnd_id], direct) for route in ROUTES}
    safe = [route for route in ROUTES if route != "direct" and is_safe_positive(gains[route])]
    if safe:
        safe.sort(key=lambda route: (base.BUDGET_ORDER[route], -gains[route]["utility_gain"]))
        route = safe[0]
        return route, "retained_safe_positive", gains
    failed = [route for route in ROUTES if route != "direct" and is_failed_window_style(gains[route])]
    if failed:
        return "direct", "failed_window_hard_negative", gains
    safe_looking = [
        route
        for route in ROUTES
        if route != "direct"
        and gains[route]["valid_json"]
        and gains[route]["argument_gain"] >= -0.002
        and gains[route]["event_gain"] >= -0.002
        and gains[route]["trigger_gain"] >= -0.002
        and gains[route]["utility_gain"] > 0.0
    ]
    if safe_looking:
        return "direct", "safe_unstable_hard_negative", gains
    return "direct", "ordinary_direct", gains


def source_rows(split):
    return base.read_jsonl(base.DATA_DIR / f"{base.DATA_PREFIX}_{base.ROUTE_ROW_BRANCH}_{split}_pos.jsonl")


def build_split(split):
    paths = {"direct": base.DIRECT, "reason_light": base.LIGHT, "reason_mid": base.MID, "reason_full": base.FULL}
    missing = [paths[route][split] for route in ROUTES if not paths[route][split].exists()]
    if missing:
        raise FileNotFoundError("\n".join(p.as_posix() for p in missing))
    candidates = {route: base.prediction_map(paths[route][split]) for route in ROUTES}
    common = set.intersection(*(set(candidates[route]) for route in ROUTES))
    rows = []
    labels = []
    label_counts = Counter()
    class_counts = Counter()
    weighted_class_counts = Counter()
    for source in source_rows(split):
        wnd_id = key(source)
        if wnd_id not in common:
            continue
        route, cls, gains = choose_lowest_safe(candidates, wnd_id)
        label_counts[route] += 1
        class_counts[cls] += 1
        labels.append(
            {
                "wnd_id": wnd_id,
                "route_label": route,
                "teacher_class": cls,
                "source_split": split,
                "label_source": LABEL_SOURCE,
                "label_rule": "lowest-budget A/E/T-safe positive; failed safe-looking budgets become direct hard negatives",
                "route_diagnostics": gains,
            }
        )
        item = copy.deepcopy(source)
        item["instruction"] = base.route_instruction(ROUTES)
        item["output"] = f"<ROUTE>{route}</ROUTE>"
        meta = dict(item.get("meta") or {})
        meta["adaptive_route_label"] = route
        meta["multibudget_label_source"] = LABEL_SOURCE
        meta["m11_teacher_class"] = cls
        item["meta"] = meta
        repeat = WEIGHTS.get(route, 1) if cls == "retained_safe_positive" else WEIGHTS.get(cls, 1)
        if split != "train":
            repeat = 1
        weighted_class_counts[cls] += repeat
        for dup_idx in range(repeat):
            dup = copy.deepcopy(item)
            dup_meta = dict(dup.get("meta") or {})
            dup_meta["m11_weight_repeat"] = repeat
            dup_meta["m11_duplicate_index"] = dup_idx
            dup["meta"] = dup_meta
            rows.append(dup)
    dataset_name = f"{base.DATA_PREFIX}_{BRANCH}_{split}_pos"
    dataset_path = base.DATA_DIR / f"{dataset_name}.jsonl"
    meta_path = base.DATA_DIR / f"{dataset_name}.meta.json"
    label_path = base.LABEL_DIR / f"{base.DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.jsonl"
    summary_path = base.LABEL_DIR / f"{base.DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.summary.json"
    write_jsonl(dataset_path, rows)
    write_jsonl(label_path, labels)
    update_dataset_info(base.DATA_DIR, dataset_name, dataset_path.name)
    summary = {
        "label_source": LABEL_SOURCE,
        "source_split": split,
        "num_examples": len(labels),
        "weighted_num_examples": len(rows),
        "label_counts": dict(label_counts),
        "label_rates": {k: v / len(labels) for k, v in label_counts.items()} if labels else {},
        "teacher_class_counts": dict(class_counts),
        "weighted_teacher_class_counts": dict(weighted_class_counts),
        "dataset_jsonl": dataset_path.as_posix(),
        "labels_jsonl": label_path.as_posix(),
    }
    write_json(summary_path, summary)
    write_json(
        meta_path,
        {
            "dataset_name": dataset_name,
            "num_examples": len(rows),
            "num_unique_examples": len(labels),
            "routes": ROUTES,
            "weights": WEIGHTS,
            "audit": audit_rows(rows),
            "label_counts": dict(label_counts),
            "teacher_class_counts": dict(class_counts),
        },
    )
    return summary


def make_config():
    config = yaml.safe_load(base.TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    config["model_name_or_path"] = base.WARM_START
    config["dataset"] = f"{base.DATA_PREFIX}_{BRANCH}_train_pos"
    config["eval_dataset"] = f"{base.DATA_PREFIX}_{BRANCH}_dev_seen_pos"
    config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{base.RUN_PREFIX}_{BRANCH}_full"
    config["learning_rate"] = 2.0e-6
    config["num_train_epochs"] = 1.5
    config["save_steps"] = 50
    config["eval_steps"] = 50
    config["logging_steps"] = 5
    config["save_strategy"] = "steps"
    config["eval_strategy"] = "steps"
    config["load_best_model_at_end"] = False
    out = base.CONFIG_DIR / f"{base.RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return out, REPO / config["output_dir"].replace("/workspace/project/", "")


def make_note(timestamp, summaries, cfg_path, output_dir):
    exp_id = "2026-05-21_stage2_multibudget_retention_router_m11_routecls_noauxwarm_lr2e6_save50_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = base.EXPERIMENT_DIR / f"{exp_id}.md"
    note.write_text(
        f"""---
id: {exp_id}
title: {TITLE}
kind: experiment
status: running
created_at: {timestamp}
updated_at: {timestamp}
owners:
  - codex
tags:
  - stage2
  - adaptive-routing
  - multibudget-router
  - positive-retention
objective: {OBJECTIVE}
artifacts:
  configs:
    - {cfg_path.as_posix()}
  outputs:
    - {output_dir.as_posix()}
related:
  plans:
    - {REPO / "PLANS.md"}
context:
  dataset: RichERE
  split: split1
  label_source: {LABEL_SOURCE}
  routes: {json.dumps(ROUTES)}
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 1.5
---

# {TITLE}

## Goal

Train a retention-aware multibudget selector that prefers the lowest safe reason budget and learns hard negatives from failed budget-looking cases.

## Setup

- branch: `{BRANCH}`
- routes: `{", ".join(ROUTES)}`
- label rule: lowest-budget A/E/T-safe positive; failed safe-looking budgets become direct hard negatives.
- train weights: `{json.dumps(WEIGHTS)}`

Label summaries:

```json
{json.dumps(summaries, ensure_ascii=False, indent=2)}
```

## Commands

```bash
cd {REPO}
python3 scripts/prepare_multibudget_retention_router_m11_20260521.py
bash scripts/launch_modular_dualexpert_utility_router_train_20260517.sh {BRANCH}=<gpu>
```

## Run Log

### {timestamp.replace("T", " ")[:16]} +08:00

- prepared labels, datasets, config, and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- train selector.
- score route-choice NLL on dev checkpoints.
- calibrate dev windows and formal replay if dev has all-positive candidates.
""",
        encoding="utf-8",
    )
    return note


def main():
    timestamp = now_iso()
    summaries = {split: build_split(split) for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]}
    cfg_path, output_dir = make_config()
    note = make_note(timestamp, summaries, cfg_path, output_dir)
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "label_source": LABEL_SOURCE,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "note": note.as_posix(),
                "summaries": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
