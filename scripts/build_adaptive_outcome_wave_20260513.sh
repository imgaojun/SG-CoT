#!/usr/bin/env bash
set -euo pipefail

DIRECT_PREFIX="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_DIR="data/stage2_adaptive_datasets"
LABEL_DIR="${DATA_DIR}/labels"
SCHEMA="data/schema/richere-en.event_schema.json"

MINER_BRANCH="likelihood15_goldplan_type_role_hint_plan_lite_bal30"
MINER_DATASET="${ADAPT_PREFIX}_${MINER_BRANCH}"
PLACEHOLDER_TRAIN_LABEL="${LABEL_DIR}/${ADAPT_PREFIX}_likelihood_goldplan15_train_labels.jsonl"
PLACEHOLDER_DEV_LABEL="${LABEL_DIR}/${ADAPT_PREFIX}_likelihood_goldplan15_dev_seen_labels.jsonl"
MINER_CKPT="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_${MINER_BRANCH}_full/checkpoint-942"

python src/stage2_cot/build_adaptive_route_reasoning_dataset.py \
  --schema_path "${SCHEMA}" \
  --direct_train_jsonl "${DIRECT_PREFIX}_train_pos.jsonl" \
  --direct_dev_jsonl "${DIRECT_PREFIX}_dev_seen_pos.jsonl" \
  --direct_test_jsonl "${DIRECT_PREFIX}_test_pos.jsonl" \
  --direct_test_seen_jsonl "${DIRECT_PREFIX}_test_seen_pos.jsonl" \
  --direct_test_unseen_jsonl "${DIRECT_PREFIX}_test_unseen_pos.jsonl" \
  --train_label_jsonl "${PLACEHOLDER_TRAIN_LABEL}" \
  --dev_label_jsonl "${PLACEHOLDER_DEV_LABEL}" \
  --dataset_dir "${DATA_DIR}" \
  --train_dataset_name "${MINER_DATASET}_train_pos" \
  --dev_dataset_name "${MINER_DATASET}_dev_seen_pos" \
  --test_dataset_name "${MINER_DATASET}_test_pos" \
  --test_seen_dataset_name "${MINER_DATASET}_test_seen_pos" \
  --test_unseen_dataset_name "${MINER_DATASET}_test_unseen_pos" \
  --target_style type_role_hint_plan_lite \
  --max_role_checks_per_sample 6 \
  --seed 13 \
  --write_forced_eval_variants \
  --write_forced_train_variants

mkdir -p "${LABEL_DIR}"

for cap in 10 15; do
  rate="0.${cap}"
  label_source="outcome_l15bal30_${cap}"
  python src/stage2_analysis/build_adaptive_outcome_route_labels.py \
    --forced_direct_predictions "outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942/forced_direct/train/predictions.jsonl" \
    --forced_reason_predictions "outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942/forced_reason/train/predictions.jsonl" \
    --output_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${label_source}_train_labels.jsonl" \
    --summary_json "${LABEL_DIR}/${ADAPT_PREFIX}_${label_source}_train_labels.summary.json" \
    --reason_rate_cap "${rate}" \
    --margin 0.0 \
    --label_source "${label_source}" \
    --miner_checkpoint "${MINER_CKPT}"
  python src/stage2_analysis/build_adaptive_outcome_route_labels.py \
    --forced_direct_predictions "outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942/forced_direct/dev_seen/predictions.jsonl" \
    --forced_reason_predictions "outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942/forced_reason/dev_seen/predictions.jsonl" \
    --output_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${label_source}_dev_seen_labels.jsonl" \
    --summary_json "${LABEL_DIR}/${ADAPT_PREFIX}_${label_source}_dev_seen_labels.summary.json" \
    --reason_rate_cap "${rate}" \
    --margin 0.0 \
    --label_source "${label_source}" \
    --miner_checkpoint "${MINER_CKPT}"
done

build_dataset() {
  local branch="$1"
  local label_source="$2"
  local route_aux_repeat="$3"
  local route_only_train="$4"

  local route_only_flag=()
  if [[ "${route_only_train}" == "1" ]]; then
    route_only_flag=(--route_only_train)
  fi

  python src/stage2_cot/build_adaptive_route_reasoning_dataset.py \
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
    --write_forced_eval_variants \
    --route_aux_repeat "${route_aux_repeat}" \
    "${route_only_flag[@]}"
}

build_dataset "outcome10_l15bal30_routeonly_probe" "outcome_l15bal30_10" 0 1
build_dataset "outcome15_l15bal30_routeonly_probe" "outcome_l15bal30_15" 0 1
build_dataset "outcome10_l15bal30_type_role_hint_plan_lite_raw" "outcome_l15bal30_10" 0 0
build_dataset "outcome10_l15bal30_type_role_hint_plan_lite_routeaux1x" "outcome_l15bal30_10" 1 0
build_dataset "outcome15_l15bal30_type_role_hint_plan_lite_routeaux1x" "outcome_l15bal30_15" 1 0
build_dataset "outcome10_l15bal30_type_role_hint_plan_lite_routeaux2x" "outcome_l15bal30_10" 2 0
