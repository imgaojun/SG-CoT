#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_BASE="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py"
OUT_BASE="outputs/stage2_strategy_cot_e40/formal_e40_data_scaling_20260604"
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

train_branch() {
  local branch="$1"
  local gpu="$2"
  local config="configs/generated/stage2_adaptive/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_${branch}_full_stepmatch.yaml"
  local name="stage2_1_7b_${branch}_train_20260604"
  local log_path="${LOG_BASE}/${branch}_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${config}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_BASE}_${branch}_full' '${log_path}' '${OUT_BASE}' || true
  "
}

eval_branch_ckpt_pair() {
  local branch="$1"
  local ckpt="$2"
  local gpu="$3"
  local adapter="${RUN_BASE}_${branch}_full/checkpoint-${ckpt}"
  local name="stage2_1_7b_${branch}_ckpt${ckpt}_eval_pair_20260604"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${branch}/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/${branch}/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT}' \
      --base_model '${BASE_MODEL}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_${branch}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${branch}/checkpoint-${ckpt}/test_seen' \
      --batch_size 8 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${branch}_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT}' \
      --base_model '${BASE_MODEL}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_${branch}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${branch}/checkpoint-${ckpt}/test_unseen' \
      --batch_size 8 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${branch}_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}'
  "
}

case "${1:-}" in
  train500)
    mkdir -p "${LOG_BASE}"
    train_branch e40_seed500_nested_thinking_evidence_cot "${2:-1}"
    ;;
  eval500)
    mkdir -p "${LOG_BASE}"
    eval_branch_ckpt_pair e40_seed500_nested_thinking_evidence_cot "${2:-96}" "${3:-1}"
    ;;
  train3000)
    mkdir -p "${LOG_BASE}"
    train_branch e40_seed3000_thinking_evidence_cot "${2:-1}"
    ;;
  eval3000)
    mkdir -p "${LOG_BASE}"
    eval_branch_ckpt_pair e40_seed3000_thinking_evidence_cot "${2:-345}" "${3:-1}"
    ;;
  *)
    echo "usage: $0 train500 [gpu] | eval500 [ckpt] [gpu] | train3000 [gpu] | eval3000 [ckpt] [gpu]" >&2
    exit 2
    ;;
esac
