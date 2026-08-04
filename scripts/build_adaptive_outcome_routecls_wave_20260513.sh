#!/usr/bin/env bash
set -euo pipefail

DIRECT_PREFIX="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_DIR="data/stage2_adaptive_datasets"
LABEL_DIR="${DATA_DIR}/labels"
SCHEMA="data/schema/richere-en.event_schema.json"

build_routecls_dataset() {
  local branch="$1"
  local label_source="$2"
  local oversample="$3"

  python3 src/stage2_cot/build_adaptive_route_reasoning_dataset.py \
    --schema_path "${SCHEMA}" \
    --direct_train_jsonl "${DIRECT_PREFIX}_train_pos.jsonl" \
    --direct_dev_jsonl "${DIRECT_PREFIX}_dev_seen_pos.jsonl" \
    --direct_test_jsonl "${DIRECT_PREFIX}_test_pos.jsonl" \
    --direct_test_seen_jsonl "${DIRECT_PREFIX}_test_seen_pos.jsonl" \
    --direct_test_unseen_jsonl "${DIRECT_PREFIX}_test_unseen_pos.jsonl" \
    --train_label_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${label_source}_train_labels.jsonl" \
    --dev_label_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${label_source}_dev_seen_labels.jsonl" \
    --dataset_dir "${DATA_DIR}" \
    --train_dataset_name "${ADAPT_PREFIX}_${branch}_train_pos" \
    --dev_dataset_name "${ADAPT_PREFIX}_${branch}_dev_seen_pos" \
    --test_dataset_name "${ADAPT_PREFIX}_${branch}_test_pos" \
    --test_seen_dataset_name "${ADAPT_PREFIX}_${branch}_test_seen_pos" \
    --test_unseen_dataset_name "${ADAPT_PREFIX}_${branch}_test_unseen_pos" \
    --target_style type_role_hint_plan_lite \
    --max_role_checks_per_sample 6 \
    --seed 13 \
    --route_only_train \
    --route_only_eval \
    --route_classifier_prompt \
    --route_reason_oversample "${oversample}"
}

build_routecls_dataset "outcome10_l15bal30_routecls_balanced_probe" "outcome_l15bal30_10" 9
build_routecls_dataset "outcome15_l15bal30_routecls_balanced_probe" "outcome_l15bal30_15" 6
