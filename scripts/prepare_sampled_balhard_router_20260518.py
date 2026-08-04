import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.prepare_sampled_confident_router_20260518 import (
    CONFIG_DIR,
    DATA_DIR,
    DATA_PREFIX,
    EXPERIMENT_DIR,
    LABEL_SOURCE,
    NOAUX_CKPT,
    REPO,
    RUN_PREFIX,
    TEMPLATE_CONFIG,
    base_row_id,
    build_route_row,
    label_path,
    label_summary_path,
    load_json,
    load_jsonl,
    source_path,
    update_dataset_info,
    write_json,
    write_jsonl,
    write_yaml,
)


BRANCH = "sampled_k8_ckpt258_balhard_routecls_noauxwarm_lr2e6_save25"
POSITIVE_REPEAT = 8
NEGATIVE_TARGET_MULTIPLIER = 8
SEED = 18
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def rank_hard_negative(row):
    trigger_harm = 1.0 - float(row.get("p_trigger_noharm", 0.0))
    mean_gain = float(row.get("mean_gain", 0.0))
    p_win = float(row.get("p_win", 0.0))
    low_direct = 2.25 - float(row.get("direct_mean_score", 0.0))
    return (trigger_harm >= 0.25, mean_gain > 0.0, p_win > 0.25, trigger_harm, mean_gain, p_win, low_direct)


def select_balanced_train_labels(labels):
    positives = [row for row in labels if row.get("utility_label") == "stable_reason"]
    stable_direct = [row for row in labels if row.get("utility_label") == "stable_direct"]
    hard_pool = [
        row
        for row in stable_direct
        if row.get("mean_gain", 0.0) > 0.0
        or row.get("p_win", 0.0) > 0.25
        or row.get("p_trigger_noharm", 1.0) < 0.75
        or row.get("direct_mean_score", 2.25) < 1.0
    ]
    hard_pool = sorted(hard_pool, key=rank_hard_negative, reverse=True)
    target_negatives = len(positives) * NEGATIVE_TARGET_MULTIPLIER
    selected = []
    seen = set()
    for row in hard_pool:
        if row["wnd_id"] not in seen:
            selected.append(row)
            seen.add(row["wnd_id"])
        if len(selected) >= target_negatives:
            break
    if len(selected) < target_negatives:
        rest = [row for row in stable_direct if row["wnd_id"] not in seen]
        random.Random(SEED).shuffle(rest)
        selected.extend(rest[: target_negatives - len(selected)])
    return positives, selected[:target_negatives], {
        "positive_count": len(positives),
        "stable_direct_count": len(stable_direct),
        "hard_pool_count": len(hard_pool),
        "selected_negative_count": len(selected[:target_negatives]),
        "target_negative_count": target_negatives,
    }


def audit_rows(rows, source_count: int, confident_count: int, skipped_count: int, selection_summary: dict):
    direct_count = sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "direct")
    reason_count = sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "reason")
    return {
        "source_count": source_count,
        "confident_label_count": confident_count,
        "skipped_ambiguous_or_missing_count": skipped_count,
        "total_count": len(rows),
        "route_only_count": len(rows),
        "route_only_classifier_prompt_count": len(rows),
        "route_only_full_extraction_prompt_count": 0,
        "route_only_rows_with_final": sum(1 for row in rows if "<FINAL>" in row.get("output", "")),
        "route_only_direct_rows": direct_count,
        "route_only_reason_rows": reason_count,
        "direct_count": direct_count,
        "reason_count": reason_count,
        "reason_rate": reason_count / len(rows) if rows else 0.0,
        **selection_summary,
    }


def build_train_split(dataset_name: str):
    labels = load_jsonl(label_path("train"))
    confident = [row for row in labels if row.get("utility_label") in {"stable_reason", "stable_direct"}]
    positives, negatives, selection_summary = select_balanced_train_labels(labels)
    selected_by_id = {row["wnd_id"]: row for row in positives + negatives}
    source_by_id = {base_row_id(row): row for row in load_jsonl(source_path("train"))}
    rows = []
    for label_row in positives:
        source_row = source_by_id[label_row["wnd_id"]]
        for idx in range(POSITIVE_REPEAT):
            rows.append(build_route_row(source_row, label_row, "train", idx))
    for label_row in negatives:
        source_row = source_by_id[label_row["wnd_id"]]
        rows.append(build_route_row(source_row, label_row, "train", 0))
    random.Random(SEED).shuffle(rows)
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(DATA_DIR / file_name, rows)
    update_dataset_info(dataset_name, file_name)
    meta = {
        "dataset_name": dataset_name,
        "file_name": file_name,
        "split": "train",
        "label_source": LABEL_SOURCE,
        "source_jsonl": source_path("train").as_posix(),
        "label_jsonl": label_path("train").as_posix(),
        "label_summary_json": label_summary_path("train").as_posix(),
        "positive_repeat": POSITIVE_REPEAT,
        "negative_strategy": "hard_direct_balanced",
        "excluded_utility_labels": ["ambiguous"],
        "selected_label_count": len(selected_by_id),
        "audit": audit_rows(
            rows,
            len(source_by_id),
            len(confident),
            len(labels) - len(confident),
            selection_summary,
        ),
    }
    write_json(DATA_DIR / f"{dataset_name}.meta.json", meta)
    return meta


def build_dev_split(dataset_name: str):
    labels = load_jsonl(label_path("dev_seen"))
    confident = [row for row in labels if row.get("utility_label") in {"stable_reason", "stable_direct"}]
    label_by_id = {row["wnd_id"]: row for row in confident}
    source_rows = load_jsonl(source_path("dev_seen"))
    rows = []
    for source_row in source_rows:
        label_row = label_by_id.get(base_row_id(source_row))
        if label_row is None:
            continue
        rows.append(build_route_row(source_row, label_row, "dev_seen", 0))
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(DATA_DIR / file_name, rows)
    update_dataset_info(dataset_name, file_name)
    selection_summary = {
        "positive_count": sum(1 for row in confident if row.get("utility_label") == "stable_reason"),
        "stable_direct_count": sum(1 for row in confident if row.get("utility_label") == "stable_direct"),
        "hard_pool_count": None,
        "selected_negative_count": sum(1 for row in confident if row.get("utility_label") == "stable_direct"),
        "target_negative_count": None,
    }
    meta = {
        "dataset_name": dataset_name,
        "file_name": file_name,
        "split": "dev_seen",
        "label_source": LABEL_SOURCE,
        "source_jsonl": source_path("dev_seen").as_posix(),
        "label_jsonl": label_path("dev_seen").as_posix(),
        "label_summary_json": label_summary_path("dev_seen").as_posix(),
        "positive_repeat": 1,
        "negative_strategy": "all_confident_direct",
        "excluded_utility_labels": ["ambiguous"],
        "audit": audit_rows(
            rows,
            len(source_rows),
            len(confident),
            len(labels) - len(confident),
            selection_summary,
        ),
    }
    write_json(DATA_DIR / f"{dataset_name}.meta.json", meta)
    return meta


def make_config():
    config = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    config["model_name_or_path"] = NOAUX_CKPT
    config["dataset"] = f"{DATA_PREFIX}_{BRANCH}_train_pos"
    config["eval_dataset"] = f"{DATA_PREFIX}_{BRANCH}_dev_seen_pos"
    config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_full"
    config["learning_rate"] = 2.0e-6
    config["num_train_epochs"] = 2.0
    config["logging_steps"] = 5
    config["save_strategy"] = "steps"
    config["save_steps"] = 25
    config["eval_strategy"] = "steps"
    config["eval_steps"] = 25
    config["load_best_model_at_end"] = False
    cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
    write_yaml(cfg_path, config)
    return cfg_path, REPO / config["output_dir"].replace("/workspace/project/", "")


def make_note(cfg_path: Path, output_dir: Path, train_meta: dict, dev_meta: dict, timestamp: str):
    exp_id = "2026-05-18_stage2_sampled_k8_balhard_routecls_checkpoint258_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    title = "Stage2 Sampled K8 Balanced-Hard Route Classifier Checkpoint-258"
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
  - adaptive-routing
  - sampled-supervision
  - hard-negative
  - route-classification
  - route-nll
objective: Test whether stable_reason examples become learnable when the route classifier is trained against balanced hard-direct negatives rather than all stable_direct examples.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
  reports:
    - {REPO / 'reports/2026-05-18_stage2_sampled_k8_balhard_routecls_checkpoint258_dev_probe.md'}
related:
  plans:
    - {REPO / 'PLANS.md'}
  experiments:
    - {REPO / 'experiments/2026-05-18_stage2_sampled_k8_confident_routecls_checkpoint258_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  label_source: {LABEL_SOURCE}
  reason_checkpoint: checkpoint-258
  positive_repeat: {POSITIVE_REPEAT}
  negative_strategy: hard_direct_balanced
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 2.0
  save_steps: 25
---

# {title}

## Goal

Diagnose whether the failed first confident-only router was caused by direct-class dominance or by an intrinsically weak prompt-level route signal. This run keeps stable_reason positives, balances them against hard stable_direct negatives, and evaluates on the same real dev confident distribution.

## Setup

- branch: `{BRANCH}`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- train dataset: `{DATA_PREFIX}_{BRANCH}_train_pos`
- dev dataset: `{DATA_PREFIX}_{BRANCH}_dev_seen_pos`
- train audit: `{json.dumps(train_meta['audit'], sort_keys=True)}`
- dev audit: `{json.dumps(dev_meta['audit'], sort_keys=True)}`

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
python3 scripts/prepare_sampled_balhard_router_20260518.py
bash scripts/launch_sampled_confident_router_train_20260518.sh {BRANCH}=<gpu>
bash scripts/run_sampled_balhard_router_after_train_20260518.sh
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- built balanced-hard routecls train data from K=8 sampled labels.
- created config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- run generated-route and route-NLL dev probes.
- compare against the first confident-only router.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = now_iso()
    train_name = f"{DATA_PREFIX}_{BRANCH}_train_pos"
    dev_name = f"{DATA_PREFIX}_{BRANCH}_dev_seen_pos"
    train_meta = build_train_split(train_name)
    dev_meta = build_dev_split(dev_name)
    cfg_path, output_dir = make_config()
    note = make_note(cfg_path, output_dir, train_meta, dev_meta, timestamp)
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "note": note.as_posix(),
                "train_meta": train_meta,
                "dev_meta": dev_meta,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
