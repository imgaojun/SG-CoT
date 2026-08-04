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


BRANCH = "sampled_k2_structproxy_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
POSITIVE_REPEAT_PER_PAIR = 4
NEGATIVE_UNIQUE_MULTIPLIER = 4
ARG_TEXT_JACCARD_MIN = 0.25
EVENT_COUNT_DELTA_MAX = 0.5
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def pair_slug(pair_name: str):
    return f"seedpair{pair_name}"


def structural_pass(features: dict):
    return (
        features["route_argument_disagreement"] <= 1.0 - ARG_TEXT_JACCARD_MIN
        and features["reason_minus_direct_event_count_mean"] <= EVENT_COUNT_DELTA_MAX
    )


def clear_structural_direct(features: dict):
    return (
        features["reason_minus_direct_event_count_mean"] > 0.0
        and features["route_argument_disagreement"] > 0.60
    )


def build_route_row(source_row, label_row, features, split: str, pair_name: str, seeds, route_label: str, duplicate_index: int = 0):
    item = {
        "instruction": route_classifier_instruction(),
        "input": adapt_input(source_row["input"], render_compact_evidence(features)),
        "output": f"<ROUTE>{route_label}</ROUTE>",
        "meta": dict(source_row.get("meta", {})),
    }
    meta = item["meta"]
    meta.update(
        {
            "adaptive_source": "sampled_k2_structural_proxy_compact_evidence_routecls",
            "adaptive_dataset_role": split,
            "adaptive_route_mode": "free_route",
            "adaptive_route_label": route_label,
            "adaptive_target_style": "route_classifier_only_with_k2_structural_proxy_evidence",
            "adaptive_label_source": label_row.get("label_source"),
            "adaptive_utility_label": label_row.get("utility_label"),
            "adaptive_route_only": True,
            "adaptive_route_classifier_prompt": True,
            "sampled_evidence_source": "k2_direct_reason_output_consistency_gold_free",
            "sampled_evidence_style": f"compact_v1_k2_structproxy_{pair_slug(pair_name)}",
            "sampled_evidence_samples_per_route": SAMPLE_COUNT,
            "sampled_evidence_seed_pair": pair_name,
            "sampled_evidence_seeds": seeds,
            "sampled_supervision_label_samples_per_route": 8,
            "structural_proxy_arg_text_jaccard_min": ARG_TEXT_JACCARD_MIN,
            "structural_proxy_event_count_delta_max": EVENT_COUNT_DELTA_MAX,
            "structural_proxy_arg_text_jaccard_proxy": 1.0 - features["route_argument_disagreement"],
            "structural_proxy_event_count_delta": features["reason_minus_direct_event_count_mean"],
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


def feature_maps(split: str, labels):
    return {
        pair_name: {row["key"]: row["features"] for row in build_feature_rows(split, seeds, labels)}
        for pair_name, seeds in SEED_PAIRS
    }


def select_structural_train_examples(labels, features_by_pair):
    positives = []
    direct_candidates = []
    for label in labels:
        key = label["wnd_id"]
        utility_label = label.get("utility_label")
        pair_pass = {
            pair_name: structural_pass(features_by_pair[pair_name][key])
            for pair_name, _seeds in SEED_PAIRS
        }
        pass_count = sum(pair_pass.values())
        if utility_label == "stable_reason" and pass_count >= 1:
            positives.append((label, pair_pass))
        elif utility_label == "stable_direct":
            clear_count = sum(
                1 for pair_name, _seeds in SEED_PAIRS
                if clear_structural_direct(features_by_pair[pair_name][key])
            )
            direct_candidates.append((label, clear_count))

    hard_pool = [
        (label, clear_count)
        for label, clear_count in direct_candidates
        if clear_count > 0
        or label.get("mean_gain", 0.0) > 0.0
        or label.get("p_win", 0.0) > 0.25
        or label.get("p_trigger_noharm", 1.0) < 0.75
        or label.get("direct_mean_score", 2.25) < 1.0
    ]
    hard_pool = sorted(
        hard_pool,
        key=lambda item: (item[1], rank_hard_negative(item[0])),
        reverse=True,
    )
    target_negatives = len(positives) * NEGATIVE_UNIQUE_MULTIPLIER
    negatives = []
    seen = set()
    for label, _clear_count in hard_pool:
        if label["wnd_id"] not in seen:
            negatives.append(label)
            seen.add(label["wnd_id"])
        if len(negatives) >= target_negatives:
            break
    if len(negatives) < target_negatives:
        rest = [label for label, _clear_count in direct_candidates if label["wnd_id"] not in seen]
        random.Random(SEED).shuffle(rest)
        negatives.extend(rest[: target_negatives - len(negatives)])

    return positives, negatives[:target_negatives], {
        "positive_unique_count": len(positives),
        "stable_direct_count": len(direct_candidates),
        "hard_pool_count": len(hard_pool),
        "selected_negative_unique_count": len(negatives[:target_negatives]),
        "target_negative_unique_count": target_negatives,
        "positive_repeat_per_pair": POSITIVE_REPEAT_PER_PAIR,
        "negative_unique_multiplier": NEGATIVE_UNIQUE_MULTIPLIER,
        "positive_rule": "stable_reason and structural pass on at least 1/4 K2 seed pairs",
        "negative_rule": "stable_direct hard/direct pool with structural-drift priority",
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


def build_train_split(dataset_name: str):
    labels = load_jsonl(label_path("train"))
    confident = [row for row in labels if row.get("utility_label") in {"stable_reason", "stable_direct"}]
    features_by_pair = feature_maps("train", labels)
    positives, negatives, selection_summary = select_structural_train_examples(labels, features_by_pair)
    source_by_id = {base_row_id(row): row for row in load_jsonl(source_path("train"))}
    rows = []
    for pair_name, seeds in SEED_PAIRS:
        features_by_id = features_by_pair[pair_name]
        for label_row, pair_pass in positives:
            if not pair_pass[pair_name]:
                continue
            source_row = source_by_id[label_row["wnd_id"]]
            for idx in range(POSITIVE_REPEAT_PER_PAIR):
                rows.append(build_route_row(source_row, label_row, features_by_id[label_row["wnd_id"]], "train", pair_name, seeds, "reason", idx))
        for label_row in negatives:
            source_row = source_by_id[label_row["wnd_id"]]
            rows.append(build_route_row(source_row, label_row, features_by_id[label_row["wnd_id"]], "train", pair_name, seeds, "direct", 0))
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
        "negative_strategy": "hard_direct_structural_drift_priority",
        "excluded_utility_labels": ["ambiguous", "stable_reason_without_any_structural_support"],
        "audit": audit_rows(rows, len(source_by_id), len(confident), len(labels) - len(confident), selection_summary),
    }
    write_json(DATA_DIR / f"{dataset_name}.meta.json", meta)
    return meta


def build_dev_split(dataset_name: str):
    labels = load_jsonl(label_path("dev_seen"))
    confident = [row for row in labels if row.get("utility_label") in {"stable_reason", "stable_direct"}]
    features_by_pair = feature_maps("dev_seen", labels)
    label_by_id = {row["wnd_id"]: row for row in confident}
    source_rows = load_jsonl(source_path("dev_seen"))
    rows = []
    for pair_name, seeds in SEED_PAIRS:
        features_by_id = features_by_pair[pair_name]
        for source_row in source_rows:
            label_row = label_by_id.get(base_row_id(source_row))
            if label_row is None:
                continue
            key = label_row["wnd_id"]
            route_label = "reason" if label_row.get("utility_label") == "stable_reason" and structural_pass(features_by_id[key]) else "direct"
            rows.append(build_route_row(source_row, label_row, features_by_id[key], "dev_seen", pair_name, seeds, route_label, 0))
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(DATA_DIR / file_name, rows)
    update_dataset_info(dataset_name, file_name)
    selection_summary = {
        "positive_unique_count": sum(1 for row in confident if row.get("utility_label") == "stable_reason"),
        "stable_direct_count": sum(1 for row in confident if row.get("utility_label") == "stable_direct"),
        "hard_pool_count": None,
        "selected_negative_unique_count": sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "direct"),
        "target_negative_unique_count": None,
        "positive_repeat_per_pair": 1,
        "negative_unique_multiplier": None,
        "positive_rule": "stable_reason and structural pass for this K2 seed pair",
        "negative_rule": "all non-positive confident examples for this K2 seed pair",
    }
    meta = {
        "dataset_name": dataset_name,
        "file_name": file_name,
        "split": "dev_seen_seedpairs",
        "label_source": LABEL_SOURCE,
        "source_jsonl": source_path("dev_seen").as_posix(),
        "label_jsonl": label_path("dev_seen").as_posix(),
        "label_summary_json": label_summary_path("dev_seen").as_posix(),
        "seed_pairs": [{"name": pair_name, "seeds": seeds} for pair_name, seeds in SEED_PAIRS],
        "excluded_utility_labels": ["ambiguous"],
        "audit": audit_rows(rows, len(source_rows), len(confident), len(labels) - len(confident), selection_summary),
    }
    write_json(DATA_DIR / f"{dataset_name}.meta.json", meta)
    return meta


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


def make_note(cfg_path: Path, output_dir: Path, train_meta: dict, dev_meta: dict, timestamp: str):
    exp_id = "2026-05-19_stage2_sampled_k2_structural_proxy_supervised_router_checkpoint258_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    title = "Stage2 Sampled K2 Structural-Proxy Supervised Route Classifier Checkpoint-258"
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
  - gold-free-proxy
  - structural-overlap
  - route-classification
  - compact-evidence
objective: Train a route-choice selector from the current structural gold-free proxy signal instead of using the rule only at runtime.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
    - {REPO / f'outputs/stage2_modular_dualexpert/sampled_confident_router_20260518/route_likelihood/{BRANCH}'}
  reports:
    - {REPO / 'reports/2026-05-19_stage2_sampled_k2_structural_proxy_supervised_router_dev_probe.md'}
related:
  plans:
    - {REPO / 'PLANS.md'}
  experiments:
    - {REPO / 'experiments/2026-05-19_stage2_sampled_k2_structural_proxy_locked_seedpair23_24_validation_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
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
  structural_proxy_arg_text_jaccard_min: {ARG_TEXT_JACCARD_MIN}
  structural_proxy_event_count_delta_max: {EVENT_COUNT_DELTA_MAX}
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 2.0
  save_steps: 25
---

# {title}

## Goal

Use the current structural gold-free proxy as actual route-choice supervision, rather than only as a runtime rule. This is intentionally less conservative than stopping at the locked rule: the selector is allowed to learn a broader decision boundary from positive structural examples and hard direct negatives.

## Setup

- branch: `{BRANCH}`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- train dataset: `{DATA_PREFIX}_{BRANCH}_train_pos`
- eval dataset: `{DATA_PREFIX}_{BRANCH}_dev_seen_seedpairs_pos`
- train audit: `{json.dumps(train_meta['audit'], sort_keys=True)}`
- dev audit: `{json.dumps(dev_meta['audit'], sort_keys=True)}`

    Positive Reason supervision:

```text
stable_reason under K8 sampled utility labels
and structural pass on at least 1/4 K2 seed pairs
```

Structural pass uses train/dev available gold-free evidence:

```text
route_argument_disagreement <= 0.75
and reason_minus_direct_event_count_mean <= 0.5
```

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
python3 scripts/prepare_sampled_k2_structural_proxy_router_20260519.py
bash scripts/launch_sampled_confident_router_train_20260518.sh {BRANCH}=<gpu>
bash scripts/run_sampled_k2_structural_proxy_router_after_train_20260519.sh
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- created structural-proxy supervised route-classifier datasets.
- created config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- run dev route-generation and route-NLL probes.
- if dev probe is viable, run formal routed execution against the locked structural proxy reference.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    timestamp = now_iso()
    train_name = f"{DATA_PREFIX}_{BRANCH}_train_pos"
    dev_name = f"{DATA_PREFIX}_{BRANCH}_dev_seen_seedpairs_pos"
    train_meta = build_train_split(train_name)
    dev_meta = build_dev_split(dev_name)
    cfg_path, output_dir = make_config()
    note = make_note(cfg_path, output_dir, train_meta, dev_meta, timestamp)
    manifest = CONFIG_DIR / "sampledk2_structural_proxy_router_20260519.json"
    write_json(
        manifest,
        {
            "id": "sampledk2_structural_proxy_router_20260519",
            "kind": "analysis_config",
            "created_at": timestamp,
            "branch": BRANCH,
            "config": cfg_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "train_meta": train_meta,
            "dev_meta": dev_meta,
        },
    )
    print(
        json.dumps(
            {
                "branch": BRANCH,
                "config": cfg_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "note": note.as_posix(),
                "train_audit": train_meta["audit"],
                "dev_audit": dev_meta["audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
