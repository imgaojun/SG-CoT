#!/usr/bin/env bash
set -euo pipefail

DIRECT_PREFIX="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_DIR="data/stage2_adaptive_datasets"
LABEL_DIR="${DATA_DIR}/labels"
SCHEMA="data/schema/richere-en.event_schema.json"

ROUTER_CKPT="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome15_l15bal30_routecls_balanced_probe_full/checkpoint-221"
SCORE_ROOT="outputs/stage2_adaptive_route_likelihood_probe/outcome15cal_20260514"

mkdir -p "${LABEL_DIR}" "${SCORE_ROOT}"

build_labels() {
  local cap="$1"
  local label_source="outcome15cal_nlltop${cap}"
  local rate
  rate="$(python3 - <<PY
cap = ${cap}
print(f"{cap / 100:.2f}")
PY
)"

  python3 src/stage2_analysis/build_adaptive_nll_budget_route_labels.py \
    --score_jsonl "${SCORE_ROOT}/train_scores.jsonl" \
    --output_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${label_source}_train_labels.jsonl" \
    --summary_json "${LABEL_DIR}/${ADAPT_PREFIX}_${label_source}_train_labels.summary.json" \
    --reason_rate_cap "${rate}" \
    --label_source "${label_source}" \
    --router_checkpoint "${ROUTER_CKPT}"

  python3 src/stage2_analysis/build_adaptive_nll_budget_route_labels.py \
    --score_jsonl "${SCORE_ROOT}/dev_seen_scores.jsonl" \
    --output_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${label_source}_dev_seen_labels.jsonl" \
    --summary_json "${LABEL_DIR}/${ADAPT_PREFIX}_${label_source}_dev_seen_labels.summary.json" \
    --reason_rate_cap "${rate}" \
    --label_source "${label_source}" \
    --router_checkpoint "${ROUTER_CKPT}"
}

build_dataset() {
  local branch="$1"
  local label_source="$2"

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
    --seed 14 \
    --write_forced_eval_variants \
    --route_aux_repeat 2 \
    --route_reason_oversample 2
}

if [[ ! -s "${SCORE_ROOT}/train_scores.jsonl" || ! -s "${SCORE_ROOT}/dev_seen_scores.jsonl" ]]; then
  echo "Missing route-NLL scores under ${SCORE_ROOT}." >&2
  echo "Run scripts/launch_adaptive_outcome_calibrated_route_score_20260514.sh first." >&2
  exit 1
fi

build_labels 10
build_labels 15

build_dataset "outcome15cal_nlltop10_type_role_hint_plan_lite_routeaux2x_reasonos2" "outcome15cal_nlltop10"
build_dataset "outcome15cal_nlltop15_type_role_hint_plan_lite_routeaux2x_reasonos2" "outcome15cal_nlltop15"
