#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_QWEN4="/workspace/models/LLM-Research/Qwen3-4B"
RUN_PREFIX="richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py"

case "${2:-cot}" in
  cot)
    BRANCH="e60b_glm51_factor_balanced_600_w16_thinking_evidence_cot"
    LABEL="e61_cot_factor_balanced"
    ;;
  direct)
    BRANCH="e60b_glm51_factor_balanced_600_w16_direct_surface_evidence_direct"
    LABEL="e61_direct_factor_balanced"
    ;;
  *)
    echo "unknown variant: ${2:-}" >&2
    exit 2
    ;;
esac

CONFIG="configs/generated/stage2_adaptive/${RUN_PREFIX}_${BRANCH}_full_stepmatch.yaml"
RUN_DIR="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${BRANCH}"
OUT_BASE="outputs/stage2_strategy_cot_e60/${LABEL}_qwen4_20260607"
LOG_BASE="${OUT_BASE}/logs"

docker_common() {
  docker run -d --user root --ipc host --shm-size 16g \
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
    -w /workspace/project "$@"
}

train() {
  local gpu="${1:-1}"
  local name="stage2_${LABEL}_train_20260607"
  local log_path="${LOG_BASE}/${LABEL}_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_DIR}' '${log_path}' '${OUT_BASE}' || true
  "
}

eval_pair() {
  local ckpt="$1"
  local gpu="${2:-1}"
  local adapter="${RUN_DIR}/checkpoint-${ckpt}"
  local name="stage2_${LABEL}_ckpt${ckpt}_eval_pair_20260607"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT}' \
      --base_model '${BASE_QWEN4}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT}' \
      --base_model '${BASE_QWEN4}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}'
  "
}

case "${1:-}" in
  train)
    mkdir -p "${LOG_BASE}"
    train "${3:-1}"
    ;;
  eval)
    mkdir -p "${LOG_BASE}"
    eval_pair "${3:?checkpoint}" "${4:-1}"
    ;;
  *)
    echo "usage: $0 train <cot|direct> [gpu] | eval <cot|direct> <ckpt> [gpu]" >&2
    exit 2
    ;;
esac
