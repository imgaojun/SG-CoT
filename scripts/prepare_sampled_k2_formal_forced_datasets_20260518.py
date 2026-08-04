#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from src.stage2_cot.build_adaptive_route_reasoning_dataset import (  # noqa: E402
    ROUTE_FORCED_DIRECT,
    ROUTE_FORCED_REASON,
    audit_rows,
    build_rows,
    register_dataset,
    variant_dataset_name,
)
from src.stage2_data.build_formal_stage2_dataset import load_schema_map  # noqa: E402


DATA_DIR = REPO / "data/stage2_adaptive_datasets"
FORMAL_PREFIX = REPO / "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPT_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH = "sampled_reason_expert_forcedreason_from_noaux_20260517"
SCHEMA = REPO / "data/schema/richere-en.event_schema.json"
TARGET_STYLE = "type_role_hint_plan_lite"
MAX_ROLE_CHECKS = 6
SPLITS = ["test_seen", "test_unseen"]


def build_variant(schema_by_type, split: str, route_mode: str):
    source_jsonl = Path(f"{FORMAL_PREFIX}_{split}_pos.jsonl")
    base_dataset_name = f"{ADAPT_PREFIX}_{BRANCH}_{split}_pos"
    dataset_name = variant_dataset_name(base_dataset_name, route_mode, split)
    rows, labels = build_rows(
        source_jsonl.as_posix(),
        schema_by_type,
        None,
        route_mode,
        MAX_ROLE_CHECKS,
        TARGET_STYLE,
        split,
        False,
        False,
        False,
        0,
        False,
        False,
        False,
        False,
        1,
    )
    audit = audit_rows(rows)
    if audit["full_rows_without_final"]:
        raise ValueError(f"{dataset_name} has full rows without <FINAL>: {audit}")
    register_dataset(
        DATA_DIR,
        dataset_name,
        rows,
        {
            "schema_path": SCHEMA.as_posix(),
            "target_style": TARGET_STYLE,
            "max_role_checks_per_sample": MAX_ROLE_CHECKS,
            "source_jsonl": source_jsonl.as_posix(),
            "label_jsonl": None,
            "dataset_role": split,
            "route_mode": route_mode,
            "num_examples": len(rows),
            "audit": audit,
            "label_count": len(labels),
            "purpose": "sampled_k2_formal_counterfactual_evidence",
            "branch": BRANCH,
        },
    )
    return {
        "dataset_name": dataset_name,
        "jsonl": (DATA_DIR / f"{dataset_name}.jsonl").as_posix(),
        "meta": (DATA_DIR / f"{dataset_name}.meta.json").as_posix(),
        "route_mode": route_mode,
        "split": split,
        "num_examples": len(rows),
        "audit": audit,
    }


def main():
    for split in SPLITS:
        source = Path(f"{FORMAL_PREFIX}_{split}_pos.jsonl")
        if not source.is_file():
            raise SystemExit(f"missing formal source: {source}")
    if not SCHEMA.is_file():
        raise SystemExit(f"missing schema: {SCHEMA}")
    schema_by_type = load_schema_map(SCHEMA)
    made = []
    for split in SPLITS:
        for route_mode in [ROUTE_FORCED_DIRECT, ROUTE_FORCED_REASON]:
            made.append(build_variant(schema_by_type, split, route_mode))
    print(json.dumps({"made": made}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
