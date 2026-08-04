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
TRAIN_LABEL = DATA_DIR / "labels" / f"{ADAPT_PREFIX}_{LABEL_SOURCE}_train_labels.jsonl"
SCHEMA = REPO_ROOT / "data/schema/richere-en.event_schema.json"
TARGET_STYLE = "type_role_hint_plan_lite"
MAX_ROLE_CHECKS = 6

DIRECT_BRANCH = "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux"
REASON_BRANCH = "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux"


def build_variant(schema_by_type, branch, route_mode):
    train_dataset_name = f"{ADAPT_PREFIX}_{branch}_train_pos"
    variant_name = variant_dataset_name(train_dataset_name, route_mode, "train")
    rows, labels = build_rows(
        (DIRECT_PREFIX.as_posix() + "_train_pos.jsonl"),
        schema_by_type,
        TRAIN_LABEL.as_posix(),
        route_mode,
        MAX_ROLE_CHECKS,
        TARGET_STYLE,
        "train",
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
    register_dataset(
        DATA_DIR,
        variant_name,
        rows,
        {
            "schema_path": SCHEMA.as_posix(),
            "target_style": TARGET_STYLE,
            "max_role_checks_per_sample": MAX_ROLE_CHECKS,
            "source_jsonl": DIRECT_PREFIX.as_posix() + "_train_pos.jsonl",
            "label_jsonl": TRAIN_LABEL.as_posix(),
            "dataset_role": "train",
            "route_mode": route_mode,
            "num_examples": len(rows),
            "audit": audit,
            "label_count": len(labels),
            "purpose": "modular_dualexpert_train_teacher_eval",
            "teacher_branch": branch,
        },
    )
    return {
        "branch": branch,
        "route_mode": route_mode,
        "dataset_name": variant_name,
        "jsonl": (DATA_DIR / f"{variant_name}.jsonl").as_posix(),
        "meta": (DATA_DIR / f"{variant_name}.meta.json").as_posix(),
        "num_examples": len(rows),
        "audit": audit,
    }


def main():
    if not TRAIN_LABEL.is_file():
        raise SystemExit(f"missing train labels: {TRAIN_LABEL}")
    schema_by_type = load_schema_map(SCHEMA)
    made = [
        build_variant(schema_by_type, DIRECT_BRANCH, ROUTE_FORCED_DIRECT),
        build_variant(schema_by_type, REASON_BRANCH, ROUTE_FORCED_REASON),
    ]
    print(json.dumps({"made": made}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
