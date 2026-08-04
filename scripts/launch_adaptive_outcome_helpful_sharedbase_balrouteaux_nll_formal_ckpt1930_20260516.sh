#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH="outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux"
LABEL_SOURCE="outcome_l15bal30_15"
CKPT="checkpoint-1930"
ADAPTER="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full/${CKPT}"
FORMAL_ROOT="outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_balrouteaux_20260516/richere_split1_qwen3_1_7b_adaptive_${BRANCH}/${CKPT}"
ROUTE_SCORE_ROOT="outputs/stage2_adaptive_route_likelihood_probe/outcome_helpful_sharedbase_balrouteaux_formal_20260516/${BRANCH}/${CKPT}"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"

launch_extraction() {
  local split="$1"
  local mode="$2"
  local host_gpu="$3"
  local name="adaptive_outcome_helpful_sharedbase_balrouteaux_nllformal_${CKPT}_${mode}_${split}_20260516"
  local eval_jsonl="data/stage2_adaptive_datasets/${DATA_PREFIX}_${BRANCH}_${mode}_${split}_pos.jsonl"
  local output_dir="${FORMAL_ROOT}/${mode}/${split}"
  local log="${LOG_DIR}/adaptive_outcome_helpful_sharedbase_balrouteaux_nllformal_${CKPT}_${mode}_${split}_20260516.log"

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
      mkdir -p ${output_dir} ${LOG_DIR}
      python src/stage2_quality_validation/eval_adaptive_route_generation.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${ADAPTER} \
        --eval_jsonl ${eval_jsonl} \
        --output_dir ${output_dir} \
        --batch_size 2 \
        --max_new_tokens 512 \
        --temperature 0.0 2>&1 | tee ${log}
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${output_dir} ${log}
    "
}

launch_score() {
  local split="$1"
  local host_gpu="$2"
  local name="adaptive_outcome_helpful_sharedbase_balrouteaux_nllformal_${CKPT}_route_nll_${split}_20260516"
  local eval_jsonl="data/stage2_adaptive_datasets/${DATA_PREFIX}_${LABEL_SOURCE}_routecls_scorebase_${split}_pos.jsonl"
  local output_dir="${ROUTE_SCORE_ROOT}/${split}"
  local log="${LOG_DIR}/adaptive_outcome_helpful_sharedbase_balrouteaux_nllformal_${CKPT}_route_nll_${split}_20260516.log"

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
      mkdir -p ${output_dir} ${LOG_DIR}
      python src/stage2_quality_validation/score_adaptive_route_choice_likelihood.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${ADAPTER} \
        --eval_jsonl ${eval_jsonl} \
        --output_jsonl ${output_dir}/scores.jsonl \
        --summary_json ${output_dir}/summary.json \
        --max_length 1024 2>&1 | tee ${log}
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${output_dir} ${log}
    "
}

summarize_split() {
  local split="$1"
  python3 src/stage2_analysis/analyze_adaptive_outcome_router_execution.py \
    --forced_direct_predictions "${FORMAL_ROOT}/forced_direct/${split}/predictions.jsonl" \
    --forced_reason_predictions "${FORMAL_ROOT}/forced_reason/${split}/predictions.jsonl" \
    --score_router "nll_${CKPT}_argmin=${ROUTE_SCORE_ROOT}/${split}/scores.jsonl" \
    --score_router "nll_${CKPT}_top05=${ROUTE_SCORE_ROOT}/${split}/scores.jsonl:0.05" \
    --score_router "nll_${CKPT}_top10=${ROUTE_SCORE_ROOT}/${split}/scores.jsonl:0.10" \
    --score_router "nll_${CKPT}_top15=${ROUTE_SCORE_ROOT}/${split}/scores.jsonl:0.15" \
    --score_router "nll_${CKPT}_top20=${ROUTE_SCORE_ROOT}/${split}/scores.jsonl:0.20" \
    --score_router "nll_${CKPT}_top30=${ROUTE_SCORE_ROOT}/${split}/scores.jsonl:0.30" \
    --output_json "reports/artifacts/2026-05-16_stage2_adaptive_outcome_helpful_sharedbase_balrouteaux_nll_formal_${CKPT}_${split}.json" \
    --output_md "reports/2026-05-16_stage2_adaptive_outcome_helpful_sharedbase_balrouteaux_nll_formal_${CKPT}_${split}.md"
}

case "${1:-}" in
  launch)
    launch_extraction test forced_direct 0
    launch_extraction test_seen forced_direct 1
    launch_extraction test_unseen forced_direct 2
    launch_extraction test forced_reason 3
    launch_extraction test_seen forced_reason 4
    launch_extraction test_unseen forced_reason 7
    launch_score test 0
    launch_score test_seen 1
    launch_score test_unseen 2
    ;;
  summarize)
    for split in test test_seen test_unseen; do
      for path in \
        "${FORMAL_ROOT}/forced_direct/${split}/predictions.jsonl" \
        "${FORMAL_ROOT}/forced_reason/${split}/predictions.jsonl" \
        "${ROUTE_SCORE_ROOT}/${split}/scores.jsonl"; do
        if [[ ! -s "${path}" ]]; then
          echo "missing required file: ${path}" >&2
          exit 1
        fi
      done
      summarize_split "${split}"
    done
    ;;
  *)
    echo "usage: $0 {launch|summarize}" >&2
    exit 2
    ;;
esac
