import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
FORMAL_DATA_DIR = REPO / "data/stage2_formal_datasets"
LABEL_DIR = DATA_DIR / "labels"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DIRECT_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
BRANCH = "sampled_k8_ckpt258_confident_routecls_noauxwarm_lr2e6_save50"
LABEL_SOURCE = "sampled_counterfactual_utility_k8_checkpoint-258"
NOAUX_CKPT = (
    "/workspace/project/outputs/stage2_adaptive_runs_user/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full/checkpoint-1184"
)
TEMPLATE_CONFIG = CONFIG_DIR / (
    f"{RUN_PREFIX}_outcome15_l15bal30_routecls_noauxwarm_lr2e6_save50_probe_full_stepmatch.yaml"
)
ROUTE_REASON_OVERSAMPLE = 8
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def update_dataset_info(dataset_name: str, file_name: str):
    path = DATA_DIR / "dataset_info.json"
    info = load_json(path) if path.exists() else {}
    info[dataset_name] = {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    write_json(path, info)


def route_classifier_instruction():
    return (
        "You are doing route selection for schema-conditioned event extraction. "
        "Use only the provided text, candidate event types, and schema cards. "
        "Choose whether a downstream extractor should use direct extraction or compact reasoning. "
        "Output exactly one tag and nothing else: `<ROUTE>direct</ROUTE>` or `<ROUTE>reason</ROUTE>`. "
        "Use `<ROUTE>reason</ROUTE>` when repeated sampled counterfactual evaluation shows that reasoning "
        "is likely to improve event type, trigger, or role grounding without harming trigger extraction. "
        "Use `<ROUTE>direct</ROUTE>` when direct extraction is expected to be sufficient or more stable."
    )


def adapt_input(input_text: str):
    return input_text.replace("\n\nReturn JSON only.", "\n\nReturn the route tag only.")


def label_path(split: str):
    return LABEL_DIR / f"{DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.jsonl"


def label_summary_path(split: str):
    return LABEL_DIR / f"{DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.summary.json"


def source_path(split: str):
    return FORMAL_DATA_DIR / f"{DIRECT_PREFIX}_{split}_pos.jsonl"


def load_confident_labels(split: str):
    labels = {}
    skipped = {}
    for row in load_jsonl(label_path(split)):
        utility_label = row.get("utility_label")
        if utility_label == "stable_reason":
            labels[row["wnd_id"]] = row
        elif utility_label == "stable_direct":
            labels[row["wnd_id"]] = row
        else:
            skipped[row["wnd_id"]] = row
    return labels, skipped


def base_row_id(row):
    return row.get("meta", {}).get("wnd_id")


def build_route_row(source_row, label_row, split: str, duplicate_index: int = 0):
    item = {
        "instruction": route_classifier_instruction(),
        "input": adapt_input(source_row["input"]),
        "output": f"<ROUTE>{label_row['route_label']}</ROUTE>",
        "meta": dict(source_row.get("meta", {})),
    }
    meta = item["meta"]
    meta.update(
        {
            "adaptive_source": "sampled_confident_routecls",
            "adaptive_dataset_role": split,
            "adaptive_route_mode": "free_route",
            "adaptive_route_label": label_row["route_label"],
            "adaptive_target_style": "route_classifier_only",
            "adaptive_label_source": label_row.get("label_source"),
            "adaptive_utility_label": label_row.get("utility_label"),
            "adaptive_route_only": True,
            "adaptive_route_classifier_prompt": True,
            "sampled_mean_gain": label_row.get("mean_gain"),
            "sampled_p_win": label_row.get("p_win"),
            "sampled_p_trigger_noharm": label_row.get("p_trigger_noharm"),
            "sampled_reason_valid_rate": label_row.get("reason_valid_rate"),
            "sampled_direct_mean_score": label_row.get("direct_mean_score"),
            "sampled_reason_mean_score": label_row.get("reason_mean_score"),
            "sampled_expected_samples_per_route": label_row.get("expected_samples_per_route"),
            "route_reason_oversample_duplicate_index": duplicate_index,
        }
    )
    return item


def audit_rows(rows, source_count: int, confident_count: int, skipped_count: int):
    direct_count = sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "direct")
    reason_count = sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "reason")
    with_final = sum(1 for row in rows if "<FINAL>" in row.get("output", ""))
    return {
        "source_count": source_count,
        "confident_label_count": confident_count,
        "skipped_ambiguous_or_missing_count": skipped_count,
        "total_count": len(rows),
        "route_only_count": len(rows),
        "route_only_classifier_prompt_count": len(rows),
        "route_only_full_extraction_prompt_count": 0,
        "route_only_rows_with_final": with_final,
        "route_only_direct_rows": direct_count,
        "route_only_reason_rows": reason_count,
        "direct_count": direct_count,
        "reason_count": reason_count,
        "reason_rate": reason_count / len(rows) if rows else 0.0,
    }


def build_split(split: str, dataset_name: str, oversample_reason: bool):
    labels, skipped = load_confident_labels(split)
    source_rows = load_jsonl(source_path(split))
    rows = []
    missing = []
    for source_row in source_rows:
        wnd_id = base_row_id(source_row)
        label_row = labels.get(wnd_id)
        if label_row is None:
            if wnd_id not in skipped:
                missing.append(wnd_id)
            continue
        repeat = ROUTE_REASON_OVERSAMPLE if oversample_reason and label_row.get("route_label") == "reason" else 1
        for dup_idx in range(repeat):
            rows.append(build_route_row(source_row, label_row, split, dup_idx))
    if missing:
        raise ValueError(f"{split} has source rows with no sampled label: {missing[:10]} (n={len(missing)})")
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(DATA_DIR / file_name, rows)
    update_dataset_info(dataset_name, file_name)
    meta = {
        "dataset_name": dataset_name,
        "file_name": file_name,
        "split": split,
        "label_source": LABEL_SOURCE,
        "source_jsonl": source_path(split).as_posix(),
        "label_jsonl": label_path(split).as_posix(),
        "label_summary_json": label_summary_path(split).as_posix(),
        "route_reason_oversample": ROUTE_REASON_OVERSAMPLE if oversample_reason else 1,
        "excluded_utility_labels": ["ambiguous"],
        "audit": audit_rows(rows, len(source_rows), len(labels), len(skipped)),
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
    config["save_steps"] = 50
    config["eval_strategy"] = "steps"
    config["eval_steps"] = 50
    config["load_best_model_at_end"] = False
    cfg_path = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
    write_yaml(cfg_path, config)
    return cfg_path, REPO / config["output_dir"].replace("/workspace/project/", "")


def make_note(cfg_path: Path, output_dir: Path, train_meta: dict, dev_meta: dict, timestamp: str):
    exp_id = "2026-05-18_stage2_sampled_k8_confident_routecls_checkpoint258_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    title = "Stage2 Sampled K8 Confident Route Classifier Checkpoint-258"
    train_summary = load_json(label_summary_path("train"))
    dev_summary = load_json(label_summary_path("dev_seen"))
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
  - confident-only
  - route-classification
  - route-nll
  - richere
  - qwen3-1.7b
objective: Train a route-only classifier on K=8 sampled counterfactual stable_reason/stable_direct labels, excluding ambiguous examples.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
  reports:
    - {REPO / 'reports/2026-05-18_stage2_sampled_k8_confident_routecls_checkpoint258_dev_probe.md'}
related:
  plans:
    - {REPO / 'PLANS.md'}
  experiments:
    - {REPO / 'experiments/2026-05-17_stage2_sampled_counterfactual_utility_k8_label_discovery_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  label_source: {LABEL_SOURCE}
  reason_checkpoint: checkpoint-258
  direct_expert: D1930_forced_direct
  reason_expert: sampled_reason_expert_forcedreason_from_noaux_checkpoint-258
  confident_labels_only: true
  excluded_label: ambiguous
  route_reason_oversample: {ROUTE_REASON_OVERSAMPLE}
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 2.0
  save_steps: 50
---

# {title}

## Goal

Train the first router from sampled counterfactual utility supervision. This run tests whether the stable Reason signal discovered by K=8 sampling can be learned by a route-only classifier without mixing in ambiguous examples.

## Setup

- branch: `{BRANCH}`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- train dataset: `{DATA_PREFIX}_{BRANCH}_train_pos`
- dev dataset: `{DATA_PREFIX}_{BRANCH}_dev_seen_pos`
- model start: `{NOAUX_CKPT}`
- train sampled label summary: `{json.dumps(train_summary, sort_keys=True)}`
- dev sampled label summary: `{json.dumps(dev_summary, sort_keys=True)}`
- train dataset audit: `{json.dumps(train_meta['audit'], sort_keys=True)}`
- dev dataset audit: `{json.dumps(dev_meta['audit'], sort_keys=True)}`

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
python3 scripts/prepare_sampled_confident_router_20260518.py
bash scripts/launch_sampled_confident_router_train_20260518.sh {BRANCH}=<gpu>
bash scripts/run_sampled_confident_router_after_train_20260518.sh
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- created confident-only route classifier datasets from K=8 sampled labels.
- excluded ambiguous examples.
- oversampled stable_reason train rows by `{ROUTE_REASON_OVERSAMPLE}`.
- created config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch routecls training.
- run generated-route and route-NLL dev probes.
- evaluate with threshold sweeps and sampled expected routed-minus-direct metrics.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = now_iso()
    train_name = f"{DATA_PREFIX}_{BRANCH}_train_pos"
    dev_name = f"{DATA_PREFIX}_{BRANCH}_dev_seen_pos"
    train_meta = build_split("train", train_name, oversample_reason=True)
    dev_meta = build_split("dev_seen", dev_name, oversample_reason=False)
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
