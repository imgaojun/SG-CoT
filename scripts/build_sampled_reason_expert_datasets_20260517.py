import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_cot.build_adaptive_route_reasoning_dataset import (  # noqa: E402
    ROUTE_FORCED_DIRECT,
    ROUTE_FORCED_REASON,
    audit_rows,
    build_rows,
    register_dataset,
    variant_dataset_name,
)
from src.stage2_data.build_formal_stage2_dataset import load_schema_map  # noqa: E402


DATA_DIR = REPO_ROOT / "data/stage2_adaptive_datasets"
DIRECT_PREFIX = REPO_ROOT / "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPT_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
LABEL_SOURCE = "outcome_l15bal30_15"
LABEL_DIR = DATA_DIR / "labels"
TRAIN_LABEL = LABEL_DIR / f"{ADAPT_PREFIX}_{LABEL_SOURCE}_train_labels.jsonl"
DEV_LABEL = LABEL_DIR / f"{ADAPT_PREFIX}_{LABEL_SOURCE}_dev_seen_labels.jsonl"
SCHEMA = REPO_ROOT / "data/schema/richere-en.event_schema.json"
TARGET_STYLE = "type_role_hint_plan_lite"
MAX_ROLE_CHECKS = 6
BRANCH = "sampled_reason_expert_forcedreason_from_noaux_20260517"


def build_variant(schema_by_type, source_jsonl: Path, label_jsonl: Path, route_mode: str, role: str):
    base_dataset_name = f"{ADAPT_PREFIX}_{BRANCH}_{role}_pos"
    dataset_name = variant_dataset_name(base_dataset_name, route_mode, role)
    rows, labels = build_rows(
        source_jsonl.as_posix(),
        schema_by_type,
        label_jsonl.as_posix(),
        route_mode,
        MAX_ROLE_CHECKS,
        TARGET_STYLE,
        role,
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
            "label_jsonl": label_jsonl.as_posix(),
            "dataset_role": role,
            "route_mode": route_mode,
            "num_examples": len(rows),
            "audit": audit,
            "label_count": len(labels),
            "purpose": "sampled_counterfactual_utility_train_dev_only",
            "branch": BRANCH,
        },
    )
    return {
        "dataset_name": dataset_name,
        "jsonl": (DATA_DIR / f"{dataset_name}.jsonl").as_posix(),
        "meta": (DATA_DIR / f"{dataset_name}.meta.json").as_posix(),
        "route_mode": route_mode,
        "role": role,
        "num_examples": len(rows),
        "audit": audit,
    }


def main():
    for path in [TRAIN_LABEL, DEV_LABEL, SCHEMA]:
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")
    schema_by_type = load_schema_map(SCHEMA)
    train_source = Path(DIRECT_PREFIX.as_posix() + "_train_pos.jsonl")
    dev_source = Path(DIRECT_PREFIX.as_posix() + "_dev_seen_pos.jsonl")
    made = []
    for route_mode in [ROUTE_FORCED_DIRECT, ROUTE_FORCED_REASON]:
        made.append(
            build_variant(
                schema_by_type,
                train_source,
                TRAIN_LABEL,
                route_mode,
                "train",
            )
        )
        made.append(
            build_variant(
                schema_by_type,
                dev_source,
                DEV_LABEL,
                route_mode,
                "dev_seen",
            )
        )
    print(json.dumps({"made": made}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
