#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
DIRECT_PREFIX="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH="outcome15cal_nlltop15_type_role_hint_plan_lite_routeauxclf1x_reasonos2"
SCOREBASE_BRANCH="outcome15cal_nlltop15_routecls_scorebase"
EVAL_JSONL="${DATA_PREFIX}_${SCOREBASE_BRANCH}_dev_seen_pos.jsonl"
SCORE_ROOT="outputs/stage2_adaptive_route_likelihood_probe/sharedbase_routeauxclf1x_20260515"
DEV_PICK_ROOT="outputs/stage2_adaptive_runs_user_devpick_frontier"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"
DATA_DIR="data/stage2_adaptive_datasets"
LABEL_DIR="${DATA_DIR}/labels"
SCHEMA="data/schema/richere-en.event_schema.json"

ensure_scorebase_dataset() {
  if [[ -s "${DATA_DIR}/${EVAL_JSONL}" ]]; then
    return 0
  fi

  python3 src/stage2_cot/build_adaptive_route_reasoning_dataset.py \
    --schema_path "${SCHEMA}" \
    --direct_train_jsonl "${DIRECT_PREFIX}_train_pos.jsonl" \
    --direct_dev_jsonl "${DIRECT_PREFIX}_dev_seen_pos.jsonl" \
    --direct_test_jsonl "${DIRECT_PREFIX}_test_pos.jsonl" \
    --direct_test_seen_jsonl "${DIRECT_PREFIX}_test_seen_pos.jsonl" \
    --direct_test_unseen_jsonl "${DIRECT_PREFIX}_test_unseen_pos.jsonl" \
    --train_label_jsonl "${LABEL_DIR}/${DATA_PREFIX}_outcome15cal_nlltop15_train_labels.jsonl" \
    --dev_label_jsonl "${LABEL_DIR}/${DATA_PREFIX}_outcome15cal_nlltop15_dev_seen_labels.jsonl" \
    --dataset_dir "${DATA_DIR}" \
    --train_dataset_name "${DATA_PREFIX}_${SCOREBASE_BRANCH}_train_pos" \
    --dev_dataset_name "${DATA_PREFIX}_${SCOREBASE_BRANCH}_dev_seen_pos" \
    --test_dataset_name "${DATA_PREFIX}_${SCOREBASE_BRANCH}_test_pos" \
    --test_seen_dataset_name "${DATA_PREFIX}_${SCOREBASE_BRANCH}_test_seen_pos" \
    --test_unseen_dataset_name "${DATA_PREFIX}_${SCOREBASE_BRANCH}_test_unseen_pos" \
    --target_style type_role_hint_plan_lite \
    --max_role_checks_per_sample 6 \
    --seed 15 \
    --route_only_train \
    --route_only_eval \
    --route_classifier_prompt
}

launch_score() {
  local ckpt="$1"
  local host_gpu="$2"
  local name="adaptive_sharedbase_route_nll_${ckpt}_20260515"
  local adapter="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full/${ckpt}"
  local output_jsonl="${SCORE_ROOT}/${ckpt}/dev_seen_scores.jsonl"
  local summary_json="${SCORE_ROOT}/${ckpt}/dev_seen_summary.json"
  local log="${LOG_DIR}/adaptive_sharedbase_route_nll_${ckpt}_20260515.log"

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists, skipping launch: ${name}" >&2
    return 0
  fi

  docker run -d \
    --name "${name}" \
    --user root \
    --gpus "\"device=${host_gpu}\"" \
    --ipc host \
    --shm-size 16g \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -v "${LF_ROOT}/cache/torch_extensions:/workspace/.cache/torch_extensions" \
    -v "${LF_ROOT}/logs:/workspace/logs" \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    -e HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    -e TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    -e WANDB_DIR=/workspace/logs/wandb \
    -w /workspace/project \
    "${IMAGE}" \
    bash -lc "
      set -euo pipefail
      mkdir -p ${SCORE_ROOT}/${ckpt} ${LOG_DIR}
      python src/stage2_quality_validation/score_adaptive_route_choice_likelihood.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${adapter} \
        --eval_jsonl data/stage2_adaptive_datasets/${EVAL_JSONL} \
        --output_jsonl ${output_jsonl} \
        --summary_json ${summary_json} \
        --max_length 1024 2>&1 | tee ${log}
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${SCORE_ROOT}/${ckpt} ${log}
    "
}

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 checkpoint-1939=<gpu> checkpoint-2216=<gpu> [--summarize]" >&2
  exit 2
fi

summarize=0
ensure_scorebase_dataset
for item in "$@"; do
  if [[ "${item}" == "--summarize" ]]; then
    summarize=1
    continue
  fi
  ckpt="${item%%=*}"
  gpu="${item#*=}"
  launch_score "${ckpt}" "${gpu}"
done

if [[ "${summarize}" == "1" ]]; then
  for ckpt in checkpoint-1939 checkpoint-2216; do
    if [[ ! -s "${SCORE_ROOT}/${ckpt}/dev_seen_scores.jsonl" ]]; then
      echo "missing score output for ${ckpt}; wait for scoring containers before summarizing" >&2
      exit 1
    fi
  done
  python3 src/stage2_analysis/analyze_adaptive_outcome_router_execution.py \
    --forced_direct_predictions "${DEV_PICK_ROOT}/${RUN_PREFIX}_${BRANCH}_full_forced_direct_dev_seen_max512/checkpoint-1939/predictions.jsonl" \
    --forced_reason_predictions "${DEV_PICK_ROOT}/${RUN_PREFIX}_${BRANCH}_full_forced_reason_dev_seen_max512/checkpoint-1939/predictions.jsonl" \
    --router "generated_ckpt1939=outputs/stage2_adaptive_runs_user_devpick_route/${RUN_PREFIX}_${BRANCH}_full_route_dev_seen_max16/checkpoint-1939/predictions.jsonl" \
    --score_router "nll_ckpt1939_argmin=${SCORE_ROOT}/checkpoint-1939/dev_seen_scores.jsonl" \
    --score_router "nll_ckpt1939_top05=${SCORE_ROOT}/checkpoint-1939/dev_seen_scores.jsonl:0.05" \
    --score_router "nll_ckpt1939_top10=${SCORE_ROOT}/checkpoint-1939/dev_seen_scores.jsonl:0.10" \
    --score_router "nll_ckpt1939_top15=${SCORE_ROOT}/checkpoint-1939/dev_seen_scores.jsonl:0.15" \
    --output_json reports/artifacts/2026-05-15_stage2_adaptive_sharedbase_route_nll_probe_ckpt1939_dev_seen.json \
    --output_md reports/2026-05-15_stage2_adaptive_sharedbase_route_nll_probe_ckpt1939_dev_seen.md

  python3 src/stage2_analysis/analyze_adaptive_outcome_router_execution.py \
    --forced_direct_predictions "${DEV_PICK_ROOT}/${RUN_PREFIX}_${BRANCH}_full_forced_direct_dev_seen_max512/checkpoint-2216/predictions.jsonl" \
    --forced_reason_predictions "${DEV_PICK_ROOT}/${RUN_PREFIX}_${BRANCH}_full_forced_reason_dev_seen_max512/checkpoint-2216/predictions.jsonl" \
    --router "generated_ckpt2216=outputs/stage2_adaptive_runs_user_devpick_route/${RUN_PREFIX}_${BRANCH}_full_route_dev_seen_max16/checkpoint-2216/predictions.jsonl" \
    --score_router "nll_ckpt2216_argmin=${SCORE_ROOT}/checkpoint-2216/dev_seen_scores.jsonl" \
    --score_router "nll_ckpt2216_top05=${SCORE_ROOT}/checkpoint-2216/dev_seen_scores.jsonl:0.05" \
    --score_router "nll_ckpt2216_top10=${SCORE_ROOT}/checkpoint-2216/dev_seen_scores.jsonl:0.10" \
    --score_router "nll_ckpt2216_top15=${SCORE_ROOT}/checkpoint-2216/dev_seen_scores.jsonl:0.15" \
    --output_json reports/artifacts/2026-05-15_stage2_adaptive_sharedbase_route_nll_probe_ckpt2216_dev_seen.json \
    --output_md reports/2026-05-15_stage2_adaptive_sharedbase_route_nll_probe_ckpt2216_dev_seen.md
fi
