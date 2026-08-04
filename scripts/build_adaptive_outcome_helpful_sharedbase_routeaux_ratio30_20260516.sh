#!/usr/bin/env bash
set -euo pipefail

DIRECT_PREFIX="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_DIR="data/stage2_adaptive_datasets"
LABEL_DIR="${DATA_DIR}/labels"
SCHEMA="data/schema/richere-en.event_schema.json"
LABEL_SOURCE="outcome_l15bal30_15"

BRANCH="outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_routeauxr30_reasonos2_from_noaux"

TRAIN_LABEL="${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_train_labels.jsonl"
DEV_LABEL="${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_dev_seen_labels.jsonl"

if [[ ! -s "${TRAIN_LABEL}" || ! -s "${DEV_LABEL}" ]]; then
  echo "Missing labels for ${LABEL_SOURCE}." >&2
  echo "Run scripts/build_adaptive_outcome_wave_20260513.sh or restore the label files." >&2
  exit 1
fi

python3 src/stage2_cot/build_adaptive_route_reasoning_dataset.py \
  --schema_path "${SCHEMA}" \
  --direct_train_jsonl "${DIRECT_PREFIX}_train_pos.jsonl" \
  --direct_dev_jsonl "${DIRECT_PREFIX}_dev_seen_pos.jsonl" \
  --direct_test_jsonl "${DIRECT_PREFIX}_test_pos.jsonl" \
  --direct_test_seen_jsonl "${DIRECT_PREFIX}_test_seen_pos.jsonl" \
  --direct_test_unseen_jsonl "${DIRECT_PREFIX}_test_unseen_pos.jsonl" \
  --train_label_jsonl "${TRAIN_LABEL}" \
  --dev_label_jsonl "${DEV_LABEL}" \
  --dataset_dir "${DATA_DIR}" \
  --train_dataset_name "${ADAPT_PREFIX}_${BRANCH}_train_pos" \
  --dev_dataset_name "${ADAPT_PREFIX}_${BRANCH}_dev_seen_pos" \
  --test_dataset_name "${ADAPT_PREFIX}_${BRANCH}_test_pos" \
  --test_seen_dataset_name "${ADAPT_PREFIX}_${BRANCH}_test_seen_pos" \
  --test_unseen_dataset_name "${ADAPT_PREFIX}_${BRANCH}_test_unseen_pos" \
  --target_style type_role_hint_plan_lite \
  --max_role_checks_per_sample 6 \
  --seed 15 \
  --write_forced_eval_variants \
  --pair_selected_direct \
  --route_aux_repeat 1 \
  --route_aux_classifier_prompt \
  --route_aux_reason_target_rate 0.30 \
  --route_reason_oversample 2

python3 - "${DATA_DIR}/${ADAPT_PREFIX}_${BRANCH}_train_pos.meta.json" <<'PY'
import json
import sys
from pathlib import Path

meta_path = Path(sys.argv[1])
audit = json.loads(meta_path.read_text(encoding="utf-8"))["audit"]
expected = {
    "total_count": 5163,
    "full_with_final_count": 2648,
    "route_only_count": 2515,
    "route_only_classifier_prompt_count": 2515,
    "route_only_full_extraction_prompt_count": 0,
    "route_only_direct_rows": 1760,
    "route_only_reason_rows": 755,
}
bad = {key: {"expected": value, "actual": audit.get(key)} for key, value in expected.items() if audit.get(key) != value}
if bad:
    raise SystemExit(json.dumps({"meta": meta_path.as_posix(), "bad": bad, "audit": audit}, indent=2))
print(json.dumps({"meta": meta_path.as_posix(), "audit": audit}, indent=2))
PY

python3 scripts/prepare_adaptive_outcome_helpful_sharedbase_routeaux_ratio30_20260516.py
