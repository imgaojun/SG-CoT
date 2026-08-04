#!/usr/bin/env python3
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.prepare_sampled_confident_router_20260518 import (  # noqa: E402
    CONFIG_DIR,
    DATA_DIR,
    DATA_PREFIX,
    EXPERIMENT_DIR,
    NOAUX_CKPT,
    RUN_PREFIX,
    TEMPLATE_CONFIG,
    adapt_input,
    load_json,
    load_jsonl,
    route_classifier_instruction,
    update_dataset_info,
    write_json,
    write_jsonl,
    write_yaml,
)


TEACHER_JSONL = REPO / "outputs/stage2_oracle_best_binary_teacher_20260520/binary_teacher_cases.jsonl"
FORMAL_PREFIX = REPO / "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
BRANCH = "oraclebest_binary_teacher_m02_route_probe_noauxwarm_lr2e6_save50"
MARGIN = 0.02
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def source_path(split: str):
    return FORMAL_PREFIX.parent / f"{FORMAL_PREFIX.name}_{split}_pos.jsonl"


def base_row_id(row):
    return row.get("meta", {}).get("wnd_id")


def label_field():
    return f"label_margin_{MARGIN:.2f}"


def load_teacher_labels():
    labels = {}
    for row in load_jsonl(TEACHER_JSONL):
        labels[(row["split"], row["key"])] = row
    return labels


def build_route_row(source_row, label_row, split: str):
    route_label = label_row[label_field()]
    item = {
        "instruction": route_classifier_instruction(),
        "input": adapt_input(source_row["input"]),
        "output": f"<ROUTE>{route_label}</ROUTE>",
        "meta": dict(source_row.get("meta", {})),
    }
    meta = item["meta"]
    meta.update(
        {
            "adaptive_source": "oracle_best_binary_teacher_route_probe",
            "adaptive_dataset_role": split,
            "adaptive_route_mode": "free_route",
            "adaptive_route_label": route_label,
            "adaptive_target_style": "route_classifier_only",
            "adaptive_label_source": "strong_system_v0_oracle_best_binary_teacher",
            "adaptive_route_only": True,
            "adaptive_route_classifier_prompt": True,
            "teacher_margin": MARGIN,
            "teacher_gain_best_non_direct": label_row["gain_best_non_direct"],
            "teacher_best_non_direct_candidate": label_row["best_non_direct_candidate"],
            "teacher_direct_score": label_row["direct_score"],
            "teacher_best_non_direct_score": label_row["best_non_direct_score"],
            "teacher_oracle_best_candidate": label_row["oracle_best_candidate"],
            "teacher_oracle_best_score": label_row["oracle_best_score"],
            "leakage_risk": "formal_teacher_do_not_train",
        }
    )
    return item


def audit_rows(rows, source_count):
    direct_count = sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "direct")
    reason_count = sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "reason")
    best_counts = Counter(row["meta"].get("teacher_best_non_direct_candidate") for row in rows)
    return {
        "source_count": source_count,
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
        "teacher_margin": MARGIN,
        "teacher_best_non_direct_candidate_counts": dict(best_counts),
        "leakage_risk": "formal_teacher_do_not_train",
    }


def build_split(split: str, labels):
    source_rows = load_jsonl(source_path(split))
    rows = []
    missing = []
    for source_row in source_rows:
        wnd_id = base_row_id(source_row)
        label = labels.get((split, wnd_id))
        if label is None:
            missing.append(wnd_id)
            continue
        rows.append(build_route_row(source_row, label, split))
    if missing:
        raise ValueError(f"{split} has missing teacher labels: {missing[:10]} (n={len(missing)})")
    dataset_name = f"{DATA_PREFIX}_{BRANCH}_{split}_pos"
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(DATA_DIR / file_name, rows)
    update_dataset_info(dataset_name, file_name)
    meta = {
        "dataset_name": dataset_name,
        "file_name": file_name,
        "split": split,
        "source_jsonl": source_path(split).as_posix(),
        "teacher_jsonl": TEACHER_JSONL.as_posix(),
        "label_field": label_field(),
        "audit": audit_rows(rows, len(source_rows)),
    }
    write_json(DATA_DIR / f"{dataset_name}.meta.json", meta)
    return meta


def make_config():
    config = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    config["model_name_or_path"] = NOAUX_CKPT
    config["dataset"] = f"{DATA_PREFIX}_{BRANCH}_test_seen_pos"
    config["eval_dataset"] = f"{DATA_PREFIX}_{BRANCH}_test_unseen_pos"
    config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_formal_probe_full"
    config["learning_rate"] = 2.0e-6
    config["num_train_epochs"] = 1.0
    config["logging_steps"] = 5
    config["save_strategy"] = "steps"
    config["save_steps"] = 50
    config["eval_strategy"] = "steps"
    config["eval_steps"] = 50
    config["load_best_model_at_end"] = False
    config["do_train"] = False
    cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_formal_probe_stepmatch.yaml"
    write_yaml(cfg_path, config)
    return cfg_path, REPO / config["output_dir"].replace("/workspace/project/", "")


def make_note(cfg_path: Path, output_dir: Path, metas: dict, timestamp: str):
    exp_id = "2026-05-20_stage2_oracle_best_binary_teacher_route_probe_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    title = "Stage2 Oracle-Best Binary Teacher Route Probe"
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
  - oracle-teacher
  - binary-router
  - route-probe
objective: Prepare route-classifier formatted formal probe datasets from the margin-0.02 oracle-best binary teacher without using them for training.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
related:
  experiments:
    - {REPO / 'experiments/2026-05-20_stage2_oracle_best_binary_teacher_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  candidate_source: oracle_mixed_noise_top10_shuffle
  teacher_margin: {MARGIN}
  leakage_risk: formal_teacher_do_not_train
---

# {title}

## Goal

Prepare binary route-classifier formatted datasets from the oracle-best teacher so the label format and distribution are explicit before training a non-leaking router.

## Setup

- branch: `{BRANCH}`
- config: `{cfg_path.relative_to(REPO)}`
- output placeholder: `{output_dir.relative_to(REPO)}`
- teacher labels: `{TEACHER_JSONL.relative_to(REPO)}`
- margin: `{MARGIN}`
- leakage note: these formal teacher labels are for probe/scoring only and must not be used as router training data.

Audits:

```json
{json.dumps(metas, ensure_ascii=False, indent=2)}
```

## Commands

```bash
cd {REPO}
python3 scripts/prepare_oracle_best_binary_teacher_route_probe_20260520.py
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- created formal route-probe datasets from margin-0.02 binary teacher labels.
- wrote a no-train config only as an interface placeholder.

## Result

Pending.

## Conclusion

Pending.

## Next

- construct non-leaking train/dev teacher labels before launching any router training.
- use these formal probe datasets only for route-choice scoring diagnostics or replay.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = now_iso()
    labels = load_teacher_labels()
    metas = {split: build_split(split, labels) for split in ["test_seen", "test_unseen"]}
    cfg_path, output_dir = make_config()
    note = make_note(cfg_path, output_dir, metas, timestamp)
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "note": note.as_posix(),
                "metas": metas,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
