import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.analyze_sampled_k8_output_consistency_selector_20260518 import build_rows as build_consistency_rows  # noqa: E402
from scripts.prepare_sampled_balhard_router_20260518 import (  # noqa: E402
    NEGATIVE_TARGET_MULTIPLIER,
    POSITIVE_REPEAT,
    SEED,
    select_balanced_train_labels,
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
    load_json,
    load_jsonl,
    source_path,
    update_dataset_info,
    write_json,
    write_jsonl,
    write_yaml,
)


BRANCH = "sampled_k8_ckpt258_evidbalhard_routecls_noauxwarm_lr2e6_save25"
SAMPLE_COUNT = 8
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def route_classifier_instruction():
    return (
        "You are doing route selection for schema-conditioned event extraction. "
        "Use the provided text, candidate event types, schema cards, and repeated-output consistency evidence. "
        "Choose whether a downstream extractor should use direct extraction or compact reasoning. "
        "The consistency evidence is computed from repeated model outputs and does not use gold labels. "
        "Output exactly one tag and nothing else: `<ROUTE>direct</ROUTE>` or `<ROUTE>reason</ROUTE>`."
    )


def adapt_input(input_text: str, evidence_text: str):
    base = input_text.replace("\n\nReturn JSON only.", "").rstrip()
    return f"{base}\n\n{evidence_text}\n\nReturn the route tag only."


def fmt(value):
    return f"{float(value):.3f}"


def render_evidence(features: dict):
    lines = [
        "Repeated-output consistency evidence (K=8 per route; no gold metrics):",
        "- Direct output consistency:",
        (
            f"  valid={fmt(features['direct_valid_rate'])}; "
            f"full_consensus={fmt(features['direct_full_consensus'])}; "
            f"trigger_consensus={fmt(features['direct_trigger_consensus'])}; "
            f"argument_consensus={fmt(features['direct_argument_consensus'])}; "
            f"event_type_consensus={fmt(features['direct_event_type_consensus'])}"
        ),
        (
            f"  event_count_mean/std={fmt(features['direct_event_count_mean'])}/{fmt(features['direct_event_count_std'])}; "
            f"argument_count_mean/std={fmt(features['direct_argument_count_mean'])}/{fmt(features['direct_argument_count_std'])}; "
            f"empty_rate={fmt(features['direct_empty_mean'])}"
        ),
        "- Reason output consistency:",
        (
            f"  valid={fmt(features['reason_valid_rate'])}; "
            f"full_consensus={fmt(features['reason_full_consensus'])}; "
            f"trigger_consensus={fmt(features['reason_trigger_consensus'])}; "
            f"argument_consensus={fmt(features['reason_argument_consensus'])}; "
            f"event_type_consensus={fmt(features['reason_event_type_consensus'])}"
        ),
        (
            f"  event_count_mean/std={fmt(features['reason_event_count_mean'])}/{fmt(features['reason_event_count_std'])}; "
            f"argument_count_mean/std={fmt(features['reason_argument_count_mean'])}/{fmt(features['reason_argument_count_std'])}; "
            f"empty_rate={fmt(features['reason_empty_mean'])}"
        ),
        (
            f"  plan_lines={fmt(features['reason_plan_line_count_mean'])}; "
            f"plan_contrasts={fmt(features['reason_plan_contrast_count_mean'])}; "
            f"roles_present={fmt(features['reason_plan_role_present_count_mean'])}; "
            f"roles_absent={fmt(features['reason_plan_role_absent_count_mean'])}"
        ),
        "- Cross-route disagreement:",
        (
            f"  full={fmt(features['route_full_disagreement'])}; "
            f"trigger={fmt(features['route_trigger_disagreement'])}; "
            f"argument={fmt(features['route_argument_disagreement'])}; "
            f"event_type={fmt(features['route_event_type_disagreement'])}"
        ),
        "- Derived consistency cues:",
        (
            f"  direct_instability={fmt(features['direct_instability'])}; "
            f"reason_stability={fmt(features['reason_stability'])}; "
            f"reason_consistency_advantage={fmt(features['reason_consistency_advantage'])}; "
            f"direct_sparse_reason_rich={fmt(features['direct_sparse_reason_rich'])}"
        ),
    ]
    return "\n".join(lines)


def consistency_feature_map(split: str):
    labels = load_jsonl(label_path(split))
    rows = build_consistency_rows(split, SAMPLE_COUNT, labels)
    return {row["key"]: row["features"] for row in rows}


def build_route_row(source_row, label_row, features, split: str, duplicate_index: int = 0):
    item = {
        "instruction": route_classifier_instruction(),
        "input": adapt_input(source_row["input"], render_evidence(features)),
        "output": f"<ROUTE>{label_row['route_label']}</ROUTE>",
        "meta": dict(source_row.get("meta", {})),
    }
    meta = item["meta"]
    meta.update(
        {
            "adaptive_source": "sampled_evidence_balhard_routecls",
            "adaptive_dataset_role": split,
            "adaptive_route_mode": "free_route",
            "adaptive_route_label": label_row["route_label"],
            "adaptive_target_style": "route_classifier_only_with_output_consistency_evidence",
            "adaptive_label_source": label_row.get("label_source"),
            "adaptive_utility_label": label_row.get("utility_label"),
            "adaptive_route_only": True,
            "adaptive_route_classifier_prompt": True,
            "sampled_evidence_source": "k8_direct_reason_output_consistency_gold_free",
            "sampled_evidence_samples_per_route": SAMPLE_COUNT,
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
        "sampled_evidence_samples_per_route": SAMPLE_COUNT,
        **selection_summary,
    }


def build_train_split(dataset_name: str):
    labels = load_jsonl(label_path("train"))
    confident = [row for row in labels if row.get("utility_label") in {"stable_reason", "stable_direct"}]
    positives, negatives, selection_summary = select_balanced_train_labels(labels)
    selected_by_id = {row["wnd_id"]: row for row in positives + negatives}
    source_by_id = {base_row_id(row): row for row in load_jsonl(source_path("train"))}
    features_by_id = consistency_feature_map("train")
    rows = []
    for label_row in positives:
        source_row = source_by_id[label_row["wnd_id"]]
        features = features_by_id[label_row["wnd_id"]]
        for idx in range(POSITIVE_REPEAT):
            rows.append(build_route_row(source_row, label_row, features, "train", idx))
    for label_row in negatives:
        source_row = source_by_id[label_row["wnd_id"]]
        features = features_by_id[label_row["wnd_id"]]
        rows.append(build_route_row(source_row, label_row, features, "train", 0))
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
        "sampled_evidence_samples_per_route": SAMPLE_COUNT,
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
    features_by_id = consistency_feature_map("dev_seen")
    rows = []
    for source_row in source_rows:
        label_row = label_by_id.get(base_row_id(source_row))
        if label_row is None:
            continue
        rows.append(build_route_row(source_row, label_row, features_by_id[label_row["wnd_id"]], "dev_seen", 0))
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
        "sampled_evidence_samples_per_route": SAMPLE_COUNT,
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
    exp_id = "2026-05-18_stage2_sampled_k8_evidence_balhard_routecls_checkpoint258_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    title = "Stage2 Sampled K8 Evidence Balanced-Hard Route Classifier Checkpoint-258"
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
objective: Test whether explicit K=8 gold-free output-consistency evidence makes stable_reason routing learnable by a route classifier.
artifacts:
  configs:
    - {cfg_path}
  outputs:
    - {output_dir}
  reports:
    - {REPO / 'reports/2026-05-18_stage2_sampled_k8_evidence_balhard_routecls_checkpoint258_dev_probe.md'}
related:
  plans:
    - {REPO / 'PLANS.md'}
  experiments:
    - {REPO / 'experiments/2026-05-18_stage2_sampled_k8_output_consistency_linear_selector_checkpoint258_richere_split1_oracle_mixed_noise_qwen3_1_7b.md'}
context:
  dataset: RichERE
  split: split1
  label_source: {LABEL_SOURCE}
  reason_checkpoint: checkpoint-258
  sampled_evidence_samples_per_route: {SAMPLE_COUNT}
  positive_repeat: {POSITIVE_REPEAT}
  negative_strategy: hard_direct_balanced
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 2.0
  save_steps: 25
---

# {title}

## Goal

Train a route-only classifier with explicit repeated-output consistency evidence. The previous routecls runs only saw the raw prompt and failed to recover stable_reason reliably; output-consistency diagnostics showed stronger signal. This experiment tests whether exposing that evidence inside the router input closes the gap.

## Setup

- branch: `{BRANCH}`
- config: `{cfg_path.relative_to(REPO)}`
- output: `{output_dir.relative_to(REPO)}`
- train dataset: `{DATA_PREFIX}_{BRANCH}_train_pos`
- dev dataset: `{DATA_PREFIX}_{BRANCH}_dev_seen_pos`
- evidence: K=`{SAMPLE_COUNT}` direct/reason output consistency, validity, consensus, route disagreement, and Reason plan statistics.
- train audit: `{json.dumps(train_meta['audit'], sort_keys=True)}`
- dev audit: `{json.dumps(dev_meta['audit'], sort_keys=True)}`

## Commands

```bash
cd /mnt/disk/gaojun/research/progressive-ee
python3 scripts/prepare_sampled_evidence_balhard_router_20260518.py
bash scripts/launch_sampled_confident_router_train_20260518.sh {BRANCH}=<gpu>
bash scripts/run_sampled_evidence_balhard_router_after_train_20260518.sh
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- created evidence-augmented balanced-hard route classifier datasets.
- added K=8 gold-free output-consistency evidence to each route prompt.
- created config and experiment note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- run generated-route and route-NLL dev probes.
- compare against plain balanced-hard routecls and output-consistency linear selector.
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
