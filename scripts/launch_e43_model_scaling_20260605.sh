#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_QWEN4="/workspace/models/LLM-Research/Qwen3-4B"
BASE_LLAMA3="/workspace/models/LLM-Research/Llama-3.2-3B-Instruct"
BRANCH_QWEN4_E40="e40_seed1500_thinking_evidence_cot"
BRANCH_LLAMA3_DIRECT="direct_e43"
BRANCH_LLAMA3_E40="e40_seed1500_thinking_evidence_cot_basewarm"
RUN_PREFIX_QWEN4="richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX_LLAMA3_FORMAL="richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle"
RUN_PREFIX_LLAMA3_ADAPTIVE="richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
FORMAL_DATA_BASE="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
CONFIG_QWEN4_E40="configs/generated/stage2_adaptive/${RUN_PREFIX_QWEN4}_${BRANCH_QWEN4_E40}_full_stepmatch.yaml"
CONFIG_LLAMA3_DIRECT="configs/generated/stage2_formal/${RUN_PREFIX_LLAMA3_FORMAL}_${BRANCH_LLAMA3_DIRECT}_full_stepmatch.yaml"
CONFIG_LLAMA3_E40="configs/generated/stage2_adaptive/${RUN_PREFIX_LLAMA3_ADAPTIVE}_${BRANCH_LLAMA3_E40}_full_stepmatch.yaml"
RUN_QWEN4_E40="outputs/stage2_adaptive_runs_user/${RUN_PREFIX_QWEN4}_${BRANCH_QWEN4_E40}_full"
RUN_LLAMA3_DIRECT="outputs/stage2_full_sft_runs_stepmatch_user/${RUN_PREFIX_LLAMA3_FORMAL}_${BRANCH_LLAMA3_DIRECT}_full"
RUN_LLAMA3_E40="outputs/stage2_adaptive_runs_user/${RUN_PREFIX_LLAMA3_ADAPTIVE}_${BRANCH_LLAMA3_E40}_full"
EVAL_SCRIPT_E40="src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py"
EVAL_SCRIPT_DIRECT="src/stage2_quality_validation/eval_adaptive_route_generation.py"
OUT_BASE="outputs/stage2_strategy_cot_e43/model_scaling_20260605"
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

train_qwen4_e40() {
  local gpu="${1:-1}"
  local name="stage2_e43_qwen4_e40_seed1500_train_20260605"
  local log_path="${LOG_BASE}/qwen4_e40_seed1500_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E40}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E40}' '${log_path}' '${OUT_BASE}' || true
  "
}

eval_qwen4_e40_pair() {
  local ckpt="$1"
  local gpu="$2"
  local adapter="${RUN_QWEN4_E40}/checkpoint-${ckpt}"
  local name="stage2_e43_qwen4_e40_seed1500_ckpt${ckpt}_eval_pair_20260605"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/qwen4_e40_seed1500/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/qwen4_e40_seed1500/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_E40}' \
      --base_model '${BASE_QWEN4}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_${BRANCH_QWEN4_E40}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/qwen4_e40_seed1500/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/qwen4_e40_seed1500_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_E40}' \
      --base_model '${BASE_QWEN4}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_${BRANCH_QWEN4_E40}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/qwen4_e40_seed1500/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/qwen4_e40_seed1500_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}'
  "
}

train_llama3_direct() {
  local gpu="${1:-2}"
  local name="stage2_e43_llama3_direct_train_20260605"
  local log_path="${LOG_BASE}/llama3_direct_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_LLAMA3_DIRECT}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_LLAMA3_DIRECT}' '${log_path}' '${OUT_BASE}' || true
  "
}

train_llama3_e40() {
  local gpu="${1:-3}"
  local name="stage2_e43_llama3_e40_basewarm_train_20260605"
  local log_path="${LOG_BASE}/llama3_e40_basewarm_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_LLAMA3_E40}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_LLAMA3_E40}' '${log_path}' '${OUT_BASE}' || true
  "
}

eval_llama3_direct_pair() {
  local ckpt="$1"
  local gpu="$2"
  local adapter="${RUN_LLAMA3_DIRECT}/checkpoint-${ckpt}"
  local name="stage2_e43_llama3_direct_ckpt${ckpt}_eval_pair_20260605"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/llama3_direct/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/llama3_direct/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_DIRECT}' \
      --base_model '${BASE_LLAMA3}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${FORMAL_DATA_BASE}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/llama3_direct/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 512 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/llama3_direct_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_DIRECT}' \
      --base_model '${BASE_LLAMA3}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${FORMAL_DATA_BASE}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/llama3_direct/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 512 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/llama3_direct_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}'
  "
}

eval_llama3_e40_pair() {
  local ckpt="$1"
  local gpu="$2"
  local adapter="${RUN_LLAMA3_E40}/checkpoint-${ckpt}"
  local name="stage2_e43_llama3_e40_basewarm_ckpt${ckpt}_eval_pair_20260605"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/llama3_e40_basewarm/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/llama3_e40_basewarm/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_E40}' \
      --base_model '${BASE_LLAMA3}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_e40_seed1500_thinking_evidence_cot_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/llama3_e40_basewarm/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/llama3_e40_basewarm_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_E40}' \
      --base_model '${BASE_LLAMA3}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_e40_seed1500_thinking_evidence_cot_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/llama3_e40_basewarm/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/llama3_e40_basewarm_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}'
  "
}

case "${1:-}" in
  train-qwen4-e40)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e40 "${2:-1}"
    ;;
  eval-qwen4-e40)
    mkdir -p "${LOG_BASE}"
    eval_qwen4_e40_pair "${2:?checkpoint}" "${3:-1}"
    ;;
  train-llama3-direct)
    mkdir -p "${LOG_BASE}"
    train_llama3_direct "${2:-2}"
    ;;
  train-llama3-e40)
    mkdir -p "${LOG_BASE}"
    train_llama3_e40 "${2:-3}"
    ;;
  eval-llama3-direct)
    mkdir -p "${LOG_BASE}"
    eval_llama3_direct_pair "${2:?checkpoint}" "${3:-2}"
    ;;
  eval-llama3-e40)
    mkdir -p "${LOG_BASE}"
    eval_llama3_e40_pair "${2:?checkpoint}" "${3:-3}"
    ;;
  *)
    echo "usage: $0 train-qwen4-e40 [gpu] | eval-qwen4-e40 <ckpt> [gpu] | train-llama3-direct [gpu] | train-llama3-e40 [gpu] | eval-llama3-direct <ckpt> [gpu] | eval-llama3-e40 <ckpt> [gpu]" >&2
    exit 2
    ;;
esac
