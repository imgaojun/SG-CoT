#!/usr/bin/env bash
set -euo pipefail

DIRECT_PREFIX="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_DIR="data/stage2_adaptive_datasets"
LABEL_DIR="${DATA_DIR}/labels"
SCHEMA="data/schema/richere-en.event_schema.json"
BRANCH="outcome15_l15bal30_routecls_scorebase"

python3 src/stage2_cot/build_adaptive_route_reasoning_dataset.py \
  --schema_path "${SCHEMA}" \
  --direct_train_jsonl "${DIRECT_PREFIX}_train_pos.jsonl" \
  --direct_dev_jsonl "${DIRECT_PREFIX}_dev_seen_pos.jsonl" \
  --direct_test_jsonl "${DIRECT_PREFIX}_test_pos.jsonl" \
  --direct_test_seen_jsonl "${DIRECT_PREFIX}_test_seen_pos.jsonl" \
  --direct_test_unseen_jsonl "${DIRECT_PREFIX}_test_unseen_pos.jsonl" \
  --train_label_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_outcome_l15bal30_15_train_labels.jsonl" \
  --dev_label_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_outcome_l15bal30_15_dev_seen_labels.jsonl" \
  --dataset_dir "${DATA_DIR}" \
  --train_dataset_name "${ADAPT_PREFIX}_${BRANCH}_train_pos" \
  --dev_dataset_name "${ADAPT_PREFIX}_${BRANCH}_dev_seen_pos" \
  --test_dataset_name "${ADAPT_PREFIX}_${BRANCH}_test_pos" \
  --test_seen_dataset_name "${ADAPT_PREFIX}_${BRANCH}_test_seen_pos" \
  --test_unseen_dataset_name "${ADAPT_PREFIX}_${BRANCH}_test_unseen_pos" \
  --target_style type_role_hint_plan_lite \
  --max_role_checks_per_sample 6 \
  --seed 14 \
  --route_only_train \
  --route_only_eval \
  --route_classifier_prompt \
  --route_reason_oversample 1
