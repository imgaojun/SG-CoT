#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.analyze_sampled_k8_output_consistency_selector_20260518 import (  # noqa: E402
    group_features,
    mean,
    pair_disagreement,
)
from scripts.prepare_sampled_confident_router_20260518 import (  # noqa: E402
    DATA_DIR,
    DATA_PREFIX,
    RUN_PREFIX,
    base_row_id,
    load_jsonl,
    update_dataset_info,
    write_json,
    write_jsonl,
)
from scripts.prepare_sampled_k2_compact_evidence_balhard_router_20260518 import (  # noqa: E402
    BRANCH as TRAINED_BRANCH,
    SAMPLE_COUNT,
    adapt_input,
    render_compact_evidence,
    route_classifier_instruction,
)


TZ = timezone(timedelta(hours=8))
REASON_BRANCH = "sampled_reason_expert_forcedreason_from_noaux_20260517"
REASON_CKPT = "checkpoint-258"
RUN_ID = f"{REASON_BRANCH}_{REASON_CKPT}"
FORMAL_ID = os.environ.get("FORMAL_ID", "sampled_k2_formal_route_nll_probe_20260518")
SPLITS = ["test_seen", "test_unseen"]
SEEDS = [int(seed) for seed in os.environ.get("SAMPLED_K2_FORMAL_SEEDS", "17 18").split()]
ROUTES = ["direct", "reason"]
FORMAL_SOURCE_PREFIX = REPO / "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
SAMPLE_ROOT = (
    REPO / os.environ.get(
        "SAMPLE_ROOT",
        f"outputs/stage2_modular_dualexpert/formal_k2_counterfactual_utility_20260518/{RUN_ID}",
    )
)
CONFIG_PATH = REPO / os.environ.get(
    "CONFIG_PATH",
    "configs/generated/stage2_adaptive/sampledk2_formal_route_nll_probe_20260518.json",
)
OUTPUT_ROOT = REPO / os.environ.get(
    "OUTPUT_ROOT",
    f"outputs/stage2_adaptive_route_formal_nll_20260518/{TRAINED_BRANCH}",
)
REPORT_PATH = REPO / os.environ.get(
    "REPORT_PATH",
    "reports/2026-05-18_stage2_sampled_k2_formal_route_nll_probe.md",
)


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def key_for(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or meta.get("doc_id")


def source_path(split: str):
    return Path(f"{FORMAL_SOURCE_PREFIX}_{split}_pos.jsonl")


def load_samples(split: str, route: str):
    grouped = {}
    for seed in SEEDS:
        path = SAMPLE_ROOT / split / route / f"seed-{seed}" / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in load_jsonl(path):
            grouped.setdefault(key_for(row), []).append(row)
    return grouped


def build_features(split: str):
    samples = {route: load_samples(split, route) for route in ROUTES}
    keys = sorted(set(samples["direct"]) & set(samples["reason"]))
    features_by_id = {}
    for key in keys:
        direct_rows = samples["direct"][key]
        reason_rows = samples["reason"][key]
        if len(direct_rows) != SAMPLE_COUNT or len(reason_rows) != SAMPLE_COUNT:
            raise ValueError(
                f"expected K={SAMPLE_COUNT} rows for {split}/{key}, "
                f"got direct={len(direct_rows)} reason={len(reason_rows)}"
            )
        direct_feat = group_features(direct_rows)
        reason_feat = group_features(reason_rows)
        features = {}
        for prefix, feat in [("direct", direct_feat), ("reason", reason_feat)]:
            for name, value in feat.items():
                features[f"{prefix}_{name}"] = value
        for name in sorted(set(direct_feat) & set(reason_feat)):
            features[f"reason_minus_direct_{name}"] = reason_feat[name] - direct_feat[name]
            features[f"abs_reason_minus_direct_{name}"] = abs(reason_feat[name] - direct_feat[name])

        direct_consensus = mean(
            [
                direct_feat["full_consensus"],
                direct_feat["trigger_consensus"],
                direct_feat["argument_consensus"],
                direct_feat["event_type_consensus"],
            ]
        )
        reason_consensus = mean(
            [
                reason_feat["full_consensus"],
                reason_feat["trigger_consensus"],
                reason_feat["argument_consensus"],
                reason_feat["event_type_consensus"],
            ]
        )
        direct_count_instability = mean(
            [
                direct_feat["event_count_std"],
                direct_feat["trigger_count_std"],
                direct_feat["argument_count_std"],
                direct_feat["type_count_std"],
            ]
        )
        reason_count_instability = mean(
            [
                reason_feat["event_count_std"],
                reason_feat["trigger_count_std"],
                reason_feat["argument_count_std"],
                reason_feat["type_count_std"],
            ]
        )
        features.update(
            {
                "direct_consensus_avg": direct_consensus,
                "reason_consensus_avg": reason_consensus,
                "direct_instability": (1.0 - direct_consensus) + direct_count_instability,
                "reason_instability": (1.0 - reason_consensus) + reason_count_instability,
                "reason_stability": reason_feat["valid_rate"] + reason_consensus - reason_count_instability,
                "direct_unstable_reason_stable": (1.0 - direct_consensus) + direct_count_instability + reason_consensus,
                "reason_consistency_advantage": reason_consensus - direct_consensus - reason_count_instability,
                "direct_sparse_reason_rich": (
                    reason_feat["event_count_mean"]
                    + 0.35 * reason_feat["argument_count_mean"]
                    - direct_feat["event_count_mean"]
                    - 0.35 * direct_feat["argument_count_mean"]
                ),
                "reason_plan_signal": (
                    reason_feat["plan_contrast_count_mean"]
                    + reason_feat["plan_role_present_count_mean"]
                    - 0.2 * reason_feat["plan_role_absent_count_mean"]
                ),
                "route_full_disagreement": pair_disagreement(direct_rows, reason_rows, "full"),
                "route_trigger_disagreement": pair_disagreement(direct_rows, reason_rows, "trigger"),
                "route_argument_disagreement": pair_disagreement(direct_rows, reason_rows, "argument"),
                "route_event_type_disagreement": pair_disagreement(direct_rows, reason_rows, "event_type"),
            }
        )
        features_by_id[key] = features
    return features_by_id


def build_route_row(source_row, features, split: str):
    item = {
        "instruction": route_classifier_instruction(),
        "input": adapt_input(source_row["input"], render_compact_evidence(features)),
        "output": "<ROUTE>direct</ROUTE>",
        "meta": dict(source_row.get("meta", {})),
    }
    item["meta"].update(
        {
            "adaptive_source": FORMAL_ID,
            "adaptive_dataset_role": split,
            "adaptive_route_mode": "free_route",
            "adaptive_route_label": "direct",
            "adaptive_target_style": "route_classifier_only_with_formal_k2_compact_output_consistency_evidence",
            "adaptive_route_only": True,
            "adaptive_route_classifier_prompt": True,
            "sampled_evidence_source": "formal_k2_direct_reason_output_consistency_gold_free",
            "sampled_evidence_style": "compact_v1_k2_formal",
            "sampled_evidence_samples_per_route": SAMPLE_COUNT,
            "sampled_evidence_seeds": SEEDS,
        }
    )
    return item


def build_split(split: str):
    source_rows = load_jsonl(source_path(split))
    source_by_id = {base_row_id(row): row for row in source_rows}
    features_by_id = build_features(split)
    missing = sorted(set(source_by_id) - set(features_by_id))
    extra = sorted(set(features_by_id) - set(source_by_id))
    if missing or extra:
        raise ValueError(f"{split} source/evidence mismatch: missing={missing[:5]} extra={extra[:5]}")
    rows = [
        build_route_row(source_by_id[key], features_by_id[key], split)
        for key in sorted(source_by_id)
    ]
    dataset_name = f"{DATA_PREFIX}_{FORMAL_ID}_{split}_pos"
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(DATA_DIR / file_name, rows)
    update_dataset_info(dataset_name, file_name)
    meta = {
        "dataset_name": dataset_name,
        "file_name": file_name,
        "split": split,
        "source_jsonl": source_path(split).as_posix(),
        "sample_root": SAMPLE_ROOT.as_posix(),
        "trained_branch": TRAINED_BRANCH,
        "reason_checkpoint": REASON_CKPT,
        "sampled_evidence_samples_per_route": SAMPLE_COUNT,
        "sampled_evidence_seeds": SEEDS,
        "audit": {
            "source_count": len(source_rows),
            "total_count": len(rows),
            "route_only_count": len(rows),
            "route_only_classifier_prompt_count": len(rows),
            "route_only_rows_with_final": sum(1 for row in rows if "<FINAL>" in row.get("output", "")),
        },
    }
    write_json(DATA_DIR / f"{dataset_name}.meta.json", meta)
    return meta


def write_config(split_metas, timestamp: str):
    payload = {
        "id": FORMAL_ID,
        "kind": "formal_probe_config",
        "created_at": timestamp,
        "updated_at": timestamp,
        "trained_branch": TRAINED_BRANCH,
        "adapter_paths": {
            "checkpoint-50": f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{TRAINED_BRANCH}_full/checkpoint-50",
            "checkpoint-75": f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{TRAINED_BRANCH}_full/checkpoint-75",
        },
        "reason_checkpoint": REASON_CKPT,
        "sample_root": SAMPLE_ROOT.as_posix(),
        "sampled_evidence_samples_per_route": SAMPLE_COUNT,
        "sampled_evidence_seeds": SEEDS,
        "splits": {
            split: {
                "dataset_name": f"{DATA_PREFIX}_{FORMAL_ID}_{split}_pos",
                "source_jsonl": source_path(split).as_posix(),
                "count": split_metas[split]["audit"]["total_count"],
            }
            for split in SPLITS
        },
        "aggregate_test_from": SPLITS,
        "pre_registered_policies": [
            {"checkpoint": "checkpoint-50", "policy": "margin_ge_0p25", "threshold": 0.25},
            {"checkpoint": "checkpoint-75", "policy": "margin_ge_0p05", "threshold": 0.05},
        ],
        "output_root": OUTPUT_ROOT.as_posix(),
        "report": REPORT_PATH.as_posix(),
    }
    write_json(CONFIG_PATH, payload)
    return payload


def main():
    timestamp = now_iso()
    split_metas = {split: build_split(split) for split in SPLITS}
    config = write_config(split_metas, timestamp)
    print(
        json.dumps(
            {
                "config": CONFIG_PATH.as_posix(),
                "datasets": {split: meta["dataset_name"] for split, meta in split_metas.items()},
                "output_root": config["output_root"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
