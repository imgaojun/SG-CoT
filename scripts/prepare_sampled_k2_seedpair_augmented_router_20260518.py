#!/usr/bin/env python3
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.analyze_sampled_k2_seedpair_robustness_20260518 import (  # noqa: E402
    SEED_PAIRS,
    build_feature_rows,
)
from scripts.prepare_sampled_balhard_router_20260518 import (  # noqa: E402
    SEED,
    rank_hard_negative,
)
from scripts.prepare_sampled_confident_router_20260518 import (  # noqa: E402
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
    label_path,
    label_summary_path,
    load_jsonl,
    source_path,
    update_dataset_info,
    write_json,
    write_jsonl,
    write_yaml,
)
from scripts.prepare_sampled_k2_compact_evidence_balhard_router_20260518 import (  # noqa: E402
    SAMPLE_COUNT,
    adapt_input,
    render_compact_evidence,
    route_classifier_instruction,
)


BRANCH = "sampled_k2pairaug_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
TRANSFER_DATASET_ID = "sampled_k2pairaug_transfer_20260518"
POSITIVE_REPEAT_PER_PAIR = 4
NEGATIVE_UNIQUE_MULTIPLIER = 4
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def pair_slug(pair_name: str):
    return f"seedpair{pair_name}"


def build_route_row(source_row, label_row, features, split: str, pair_name: str, seeds, duplicate_index: int = 0):
    item = {
        "instruction": route_classifier_instruction(),
        "input": adapt_input(source_row["input"], render_compact_evidence(features)),
        "output": f"<ROUTE>{label_row['route_label']}</ROUTE>",
        "meta": dict(source_row.get("meta", {})),
    }
    meta = item["meta"]
    meta.update(
        {
            "adaptive_source": "sampled_k2_seedpair_augmented_compact_evidence_balhard_routecls",
            "adaptive_dataset_role": split,
            "adaptive_route_mode": "free_route",
            "adaptive_route_label": label_row["route_label"],
            "adaptive_target_style": "route_classifier_only_with_seedpair_augmented_k2_compact_output_consistency_evidence",
            "adaptive_label_source": label_row.get("label_source"),
            "adaptive_utility_label": label_row.get("utility_label"),
            "adaptive_route_only": True,
            "adaptive_route_classifier_prompt": True,
            "sampled_evidence_source": "k2_direct_reason_output_consistency_gold_free",
            "sampled_evidence_style": f"compact_v1_k2_{pair_slug(pair_name)}",
            "sampled_evidence_samples_per_route": SAMPLE_COUNT,
            "sampled_evidence_seed_pair": pair_name,
            "sampled_evidence_seeds": seeds,
            "sampled_supervision_label_samples_per_route": 8,
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


def select_augmented_train_labels(labels):
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
    target_negatives = len(positives) * NEGATIVE_UNIQUE_MULTIPLIER
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
        "positive_unique_count": len(positives),
        "stable_direct_count": len(stable_direct),
        "hard_pool_count": len(hard_pool),
        "selected_negative_unique_count": len(selected[:target_negatives]),
        "target_negative_unique_count": target_negatives,
        "positive_repeat_per_pair": POSITIVE_REPEAT_PER_PAIR,
        "negative_unique_multiplier": NEGATIVE_UNIQUE_MULTIPLIER,
    }


def audit_rows(rows, source_count: int, confident_count: int, skipped_count: int, selection_summary: dict):
    direct_count = sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "direct")
    reason_count = sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "reason")
    pair_counts = {}
    for row in rows:
        pair = row["meta"].get("sampled_evidence_seed_pair")
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
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
        "sampled_evidence_samples_per_route": SAMPLE_COUNT,
        "seed_pair_row_counts": pair_counts,
        **selection_summary,
    }


def feature_maps(split: str, labels):
    return {
        pair_name: {row["key"]: row["features"] for row in build_feature_rows(split, seeds, labels)}
        for pair_name, seeds in SEED_PAIRS
    }


def build_train_split(dataset_name: str):
    labels = load_jsonl(label_path("train"))
    confident = [row for row in labels if row.get("utility_label") in {"stable_reason", "stable_direct"}]
    positives, negatives, selection_summary = select_augmented_train_labels(labels)
    source_by_id = {base_row_id(row): row for row in load_jsonl(source_path("train"))}
    features_by_pair = feature_maps("train", labels)
    rows = []
    for pair_name, seeds in SEED_PAIRS:
        features_by_id = features_by_pair[pair_name]
        for label_row in positives:
            source_row = source_by_id[label_row["wnd_id"]]
            for idx in range(POSITIVE_REPEAT_PER_PAIR):
                rows.append(build_route_row(source_row, label_row, features_by_id[label_row["wnd_id"]], "train", pair_name, seeds, idx))
        for label_row in negatives:
            source_row = source_by_id[label_row["wnd_id"]]
            rows.append(build_route_row(source_row, label_row, features_by_id[label_row["wnd_id"]], "train", pair_name, seeds, 0))
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
        "seed_pairs": [{"name": pair_name, "seeds": seeds} for pair_name, seeds in SEED_PAIRS],
        "negative_strategy": "hard_direct_balanced_seedpair_augmented",
        "excluded_utility_labels": ["ambiguous"],
        "audit": audit_rows(rows, len(source_by_id), len(confident), len(labels) - len(confident), selection_summary),
    }
    write_json(DATA_DIR / f"{dataset_name}.meta.json", meta)
    return meta


def build_dev_pair_split(dataset_name: str, pair_name: str, seeds, labels, features_by_id):
    confident = [row for row in labels if row.get("utility_label") in {"stable_reason", "stable_direct"}]
    label_by_id = {row["wnd_id"]: row for row in confident}
    source_rows = load_jsonl(source_path("dev_seen"))
    rows = []
    for source_row in source_rows:
        label_row = label_by_id.get(base_row_id(source_row))
        if label_row is None:
            continue
        rows.append(build_route_row(source_row, label_row, features_by_id[label_row["wnd_id"]], "dev_seen", pair_name, seeds, 0))
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(DATA_DIR / file_name, rows)
    update_dataset_info(dataset_name, file_name)
    selection_summary = {
        "positive_unique_count": sum(1 for row in confident if row.get("utility_label") == "stable_reason"),
        "stable_direct_count": sum(1 for row in confident if row.get("utility_label") == "stable_direct"),
        "hard_pool_count": None,
        "selected_negative_unique_count": sum(1 for row in confident if row.get("utility_label") == "stable_direct"),
        "target_negative_unique_count": None,
        "positive_repeat_per_pair": 1,
        "negative_unique_multiplier": None,
    }
    meta = {
        "dataset_name": dataset_name,
        "file_name": file_name,
        "split": "dev_seen",
        "label_source": LABEL_SOURCE,
        "source_jsonl": source_path("dev_seen").as_posix(),
        "label_jsonl": label_path("dev_seen").as_posix(),
        "label_summary_json": label_summary_path("dev_seen").as_posix(),
        "seed_pair": pair_name,
        "seeds": seeds,
        "excluded_utility_labels": ["ambiguous"],
        "audit": audit_rows(rows, len(source_rows), len(confident), len(labels) - len(confident), selection_summary),
    }
    write_json(DATA_DIR / f"{dataset_name}.meta.json", meta)
    return meta, rows


def build_dev_splits():
    labels = load_jsonl(label_path("dev_seen"))
    features_by_pair = feature_maps("dev_seen", labels)
    pair_metas = {}
    aggregate_rows = []
    for pair_name, seeds in SEED_PAIRS:
        name = f"{DATA_PREFIX}_{TRANSFER_DATASET_ID}_{pair_slug(pair_name)}_dev_seen_pos"
        meta, rows = build_dev_pair_split(name, pair_name, seeds, labels, features_by_pair[pair_name])
        pair_metas[pair_name] = meta
        aggregate_rows.extend(rows)
    aggregate_name = f"{DATA_PREFIX}_{BRANCH}_dev_seen_seedpairs_pos"
    aggregate_file = f"{aggregate_name}.jsonl"
    write_jsonl(DATA_DIR / aggregate_file, aggregate_rows)
    update_dataset_info(aggregate_name, aggregate_file)
    aggregate_meta = {
        "dataset_name": aggregate_name,
        "file_name": aggregate_file,
        "split": "dev_seen_seedpairs",
        "label_source": LABEL_SOURCE,
        "seed_pairs": [{"name": pair_name, "seeds": seeds} for pair_name, seeds in SEED_PAIRS],
        "audit": {
            "total_count": len(aggregate_rows),
            "route_only_count": len(aggregate_rows),
            "direct_count": sum(1 for row in aggregate_rows if row["meta"].get("adaptive_route_label") == "direct"),
            "reason_count": sum(1 for row in aggregate_rows if row["meta"].get("adaptive_route_label") == "reason"),
            "seed_pair_row_counts": {
                pair_name: pair_metas[pair_name]["audit"]["total_count"]
                for pair_name, _seeds in SEED_PAIRS
            },
        },
    }
    write_json(DATA_DIR / f"{aggregate_name}.meta.json", aggregate_meta)
    return pair_metas, aggregate_meta


def make_config():
    config = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    config["model_name_or_path"] = NOAUX_CKPT
    config["dataset"] = f"{DATA_PREFIX}_{BRANCH}_train_pos"
    config["eval_dataset"] = f"{DATA_PREFIX}_{BRANCH}_dev_seen_seedpairs_pos"
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


def make_note(cfg_path: Path, output_dir: Path, train_meta: dict, dev_pair_metas: dict, dev_aggregate_meta: dict, timestamp: str):
    exp_id = "2026-05-18_stage2_sampled_k2_seedpair_augmented_compact_evidence_balhard_routecls_checkpoint258_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    title = "Stage2 Sampled K2 Seed-Pair-Augmented Compact-Evidence Balanced-Hard Route Classifier Checkpoint-258"
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
  - output-consistency
  - hard-negative
  - route-classification
  - compact-evidence
  - seedpair-augmentation
objective: Train a K2 compact-evidence route classifier that sees multiple K2 seed-pair evidence prompts during training, reducing seed-pair distribution sensitivity.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
    - {REPO / f'outputs/stage2_adaptive_route_seedpair_transfer_augmented_20260518/{BRANCH}'}
  reports:
    - {REPO / 'reports/2026-05-18_stage2_sampled_k2_seedpair_augmented_compact_evidence_router_dev_transfer.md'}
related:
  plans:
    - {REPO / 'PLANS.md'}
  experiments:
    - {REPO / 'experiments/2026-05-18_stage2_sampled_k2_seedpair_transfer_router_checkpoint_sweep_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  label_source: {LABEL_SOURCE}
  reason_checkpoint: checkpoint-258
  sampled_evidence_samples_per_route: {SAMPLE_COUNT}
  sampled_supervision_label_samples_per_route: 8
  seed_pairs: {[seeds for _pair_name, seeds in SEED_PAIRS]}
  positive_repeat_per_pair: {POSITIVE_REPEAT_PER_PAIR}
  negative_unique_multiplier: {NEGATIVE_UNIQUE_MULTIPLIER}
  negative_strategy: hard_direct_balanced_seedpair_augmented
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 2.0
  save_steps: 25
---

# {title}

## Goal

The previous K2 compact router was score-positive across seed pairs, but trigger delta was slightly unstable. This experiment keeps the K8 stable_reason/stable_direct supervision and K2 compact evidence format, but trains on all four K2 seed-pair evidence variants so the route model is not specialized to `17/18`.

## Setup

- branch: `{BRANCH}`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- train dataset: `{DATA_PREFIX}_{BRANCH}_train_pos`
- eval dataset: `{DATA_PREFIX}_{BRANCH}_dev_seen_seedpairs_pos`
- transfer dataset id: `{TRANSFER_DATASET_ID}`
- train audit: `{json.dumps(train_meta['audit'], sort_keys=True)}`
- dev aggregate audit: `{json.dumps(dev_aggregate_meta['audit'], sort_keys=True)}`
- dev pair audits: `{json.dumps({k: v['audit'] for k, v in dev_pair_metas.items()}, sort_keys=True)}`

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
python3 scripts/prepare_sampled_k2_seedpair_augmented_router_20260518.py
bash scripts/launch_sampled_confident_router_train_20260518.sh {BRANCH}=<gpu>
bash scripts/run_sampled_k2_seedpair_augmented_router_after_train_20260518.sh
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- created seed-pair-augmented train/dev datasets.
- created config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- evaluate every generated checkpoint across the four K2 dev seed pairs.
- promote to formal only if score remains positive and trigger harm is controlled across seed pairs.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = now_iso()
    train_name = f"{DATA_PREFIX}_{BRANCH}_train_pos"
    train_meta = build_train_split(train_name)
    dev_pair_metas, dev_aggregate_meta = build_dev_splits()
    cfg_path, output_dir = make_config()
    note = make_note(cfg_path, output_dir, train_meta, dev_pair_metas, dev_aggregate_meta, timestamp)
    config_manifest = CONFIG_DIR / "sampledk2_seedpair_augmented_router_20260518.json"
    write_json(
        config_manifest,
        {
            "id": "sampledk2_seedpair_augmented_router_20260518",
            "kind": "analysis_config",
            "created_at": timestamp,
            "branch": BRANCH,
            "transfer_dataset_id": TRANSFER_DATASET_ID,
            "config": cfg_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "train_meta": train_meta,
            "dev_pair_metas": dev_pair_metas,
            "dev_aggregate_meta": dev_aggregate_meta,
        },
    )
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "transfer_dataset_id": TRANSFER_DATASET_ID,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "note": note.as_posix(),
                "train_audit": train_meta["audit"],
                "dev_aggregate_audit": dev_aggregate_meta["audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
