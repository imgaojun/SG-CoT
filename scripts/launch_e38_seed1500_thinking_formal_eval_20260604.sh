#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation.py"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_BASE="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
OUT_BASE="outputs/stage2_strategy_cot_e37/formal_e38_seed1500_20260604"
LOG_BASE="${OUT_BASE}/logs"
ADAPTER="${RUN_BASE}_e38_seed1500_thinking_natural_cot_full/checkpoint-219"

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

launch_eval() {
  local name="$1"
  local gpu="$2"
  local split="$3"
  local output_dir="${OUT_BASE}/${split}"
  local log_path="${LOG_BASE}/${split}.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${output_dir}' '${LOG_BASE}'
    python '${EVAL_SCRIPT}' \
      --base_model '${BASE_MODEL}' \
      --adapter_path '${ADAPTER}' \
      --eval_jsonl '${DATA_BASE}_e38_seed1500_thinking_natural_cot_${split}_pos.jsonl' \
      --output_dir '${output_dir}' \
      --batch_size 8 \
      --max_new_tokens 768 \
      --temperature 0.0 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${output_dir}' '${log_path}'
  "
}

case "${1:-all}" in
  all)
    mkdir -p "${LOG_BASE}"
    launch_eval stage2_1_7b_e38_seed1500_thinking_eval_seen_20260604 1 test_seen
    launch_eval stage2_1_7b_e38_seed1500_thinking_eval_unseen_20260604 2 test_unseen
    ;;
  *)
    echo "usage: $0 [all]" >&2
    exit 2
    ;;
esac
