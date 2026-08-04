#!/usr/bin/env python3
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.analyze_sampled_k2_seedpair_robustness_20260518 import (  # noqa: E402
    LABEL_SOURCE,
    SEED_PAIRS,
    build_feature_rows,
    label_path,
)
from scripts.prepare_sampled_confident_router_20260518 import (  # noqa: E402
    DATA_DIR,
    DATA_PREFIX,
    RUN_PREFIX,
    base_row_id,
    load_jsonl,
    source_path,
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
CHECKPOINT = "checkpoint-50"
TRANSFER_ID = "sampled_k2_seedpair_transfer_ckpt50_20260518"
CONFIG_PATH = REPO / "configs/generated/stage2_adaptive/sampledk2_seedpair_transfer_checkpoint50_20260518.json"
OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_seedpair_transfer_20260518"
REPORT_PATH = REPO / "reports/2026-05-18_stage2_sampled_k2_seedpair_transfer_router_checkpoint50.md"


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def pair_slug(pair_name: str):
    return f"seedpair{pair_name}"


def dataset_name(pair_name: str):
    return f"{DATA_PREFIX}_{TRANSFER_ID}_{pair_slug(pair_name)}_dev_seen_pos"


def build_route_row(source_row, label_row, features, pair_name: str, seeds):
    item = {
        "instruction": route_classifier_instruction(),
        "input": adapt_input(source_row["input"], render_compact_evidence(features)),
        "output": f"<ROUTE>{label_row['route_label']}</ROUTE>",
        "meta": dict(source_row.get("meta", {})),
    }
    item["meta"].update(
        {
            "adaptive_source": "sampled_k2_seedpair_transfer_router",
            "adaptive_dataset_role": "dev_seen",
            "adaptive_route_mode": "free_route",
            "adaptive_route_label": label_row["route_label"],
            "adaptive_target_style": "route_classifier_only_with_seedpair_k2_compact_output_consistency_evidence",
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
        }
    )
    return item


def audit_rows(rows, source_count: int, label_count: int, skipped_count: int, pair_name: str, seeds):
    direct_count = sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "direct")
    reason_count = sum(1 for row in rows if row["meta"].get("adaptive_route_label") == "reason")
    return {
        "seed_pair": pair_name,
        "seeds": seeds,
        "source_count": source_count,
        "confident_label_count": label_count,
        "skipped_ambiguous_or_missing_count": skipped_count,
        "total_count": len(rows),
        "route_only_count": len(rows),
        "route_only_classifier_prompt_count": len(rows),
        "route_only_rows_with_final": sum(1 for row in rows if "<FINAL>" in row.get("output", "")),
        "direct_count": direct_count,
        "reason_count": reason_count,
        "reason_rate": reason_count / len(rows) if rows else 0.0,
        "sampled_evidence_samples_per_route": SAMPLE_COUNT,
    }


def build_pair_dataset(pair_name: str, seeds, labels):
    source_rows = load_jsonl(source_path("dev_seen"))
    source_by_id = {base_row_id(row): row for row in source_rows}
    feature_rows = build_feature_rows("dev_seen", seeds, labels)
    features_by_id = {row["key"]: row["features"] for row in feature_rows}
    confident = [row for row in labels if row.get("utility_label") in {"stable_reason", "stable_direct"}]
    rows = []
    missing = []
    for label_row in confident:
        key = label_row["wnd_id"]
        source_row = source_by_id.get(key)
        features = features_by_id.get(key)
        if source_row is None or features is None:
            missing.append(key)
            continue
        rows.append(build_route_row(source_row, label_row, features, pair_name, seeds))
    if missing:
        raise ValueError(f"missing source/features for {pair_name}: {missing[:10]} (n={len(missing)})")

    name = dataset_name(pair_name)
    file_name = f"{name}.jsonl"
    write_jsonl(DATA_DIR / file_name, rows)
    update_dataset_info(name, file_name)
    meta = {
        "dataset_name": name,
        "file_name": file_name,
        "split": "dev_seen",
        "label_source": LABEL_SOURCE,
        "source_jsonl": source_path("dev_seen").as_posix(),
        "label_jsonl": label_path("dev_seen").as_posix(),
        "trained_branch": TRAINED_BRANCH,
        "checkpoint": CHECKPOINT,
        "audit": audit_rows(rows, len(source_rows), len(confident), len(labels) - len(confident), pair_name, seeds),
    }
    write_json(DATA_DIR / f"{name}.meta.json", meta)
    return meta


def write_config(pair_metas, timestamp: str):
    payload = {
        "id": TRANSFER_ID,
        "kind": "analysis_config",
        "created_at": timestamp,
        "updated_at": timestamp,
        "trained_branch": TRAINED_BRANCH,
        "checkpoint": CHECKPOINT,
        "adapter_path": (
            f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{TRAINED_BRANCH}_full/{CHECKPOINT}"
        ),
        "label_source": LABEL_SOURCE,
        "split": "dev_seen",
        "seed_pairs": [
            {"name": pair_name, "seeds": seeds, "dataset_name": dataset_name(pair_name)}
            for pair_name, seeds in SEED_PAIRS
        ],
        "datasets": pair_metas,
        "output_root": OUTPUT_ROOT.as_posix(),
        "report": REPORT_PATH.as_posix(),
    }
    write_json(CONFIG_PATH, payload)
    return payload


def main():
    timestamp = now_iso()
    labels = load_jsonl(label_path("dev_seen"))
    pair_metas = {
        pair_name: build_pair_dataset(pair_name, seeds, labels)
        for pair_name, seeds in SEED_PAIRS
    }
    config = write_config(pair_metas, timestamp)
    print(json.dumps({"config": CONFIG_PATH.as_posix(), "datasets": list(pair_metas), "output_root": config["output_root"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
