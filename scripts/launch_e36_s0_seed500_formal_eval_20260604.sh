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
OUT_BASE="outputs/stage2_llm_reasoning_e36/formal_20260604"
LOG_BASE="outputs/stage2_llm_reasoning_e36/formal_20260604/logs"

DIRECT_ADAPTER="${RUN_BASE}_e36_s0_seed500_direct_final_only_full/checkpoint-96"
REASON_ADAPTER="${RUN_BASE}_e36_s0_seed500_llm_checklist_reason_full/checkpoint-96"

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
  local adapter="$3"
  local eval_jsonl="$4"
  local output_dir="$5"
  local max_new="$6"
  local log_path="$7"

  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${output_dir}' '$(dirname "${log_path}")'
    python '${EVAL_SCRIPT}' \
      --base_model '${BASE_MODEL}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${eval_jsonl}' \
      --output_dir '${output_dir}' \
      --batch_size 8 \
      --max_new_tokens '${max_new}' \
      --temperature 0.0 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${output_dir}' '${log_path}'
  "
}

case "${1:-all}" in
  all)
    mkdir -p "${LOG_BASE}"
    launch_eval stage2_1_7b_e36_s0_seed500_direct_eval_seen_20260604 1 \
      "${DIRECT_ADAPTER}" \
      "${DATA_BASE}_e36_s0_seed500_direct_final_only_test_seen_pos.jsonl" \
      "${OUT_BASE}/e36_s0_seed500_direct/test_seen" 512 \
      "${LOG_BASE}/direct_test_seen.log"
    launch_eval stage2_1_7b_e36_s0_seed500_direct_eval_unseen_20260604 2 \
      "${DIRECT_ADAPTER}" \
      "${DATA_BASE}_e36_s0_seed500_direct_final_only_test_unseen_pos.jsonl" \
      "${OUT_BASE}/e36_s0_seed500_direct/test_unseen" 512 \
      "${LOG_BASE}/direct_test_unseen.log"
    launch_eval stage2_1_7b_e36_s0_seed500_reason_eval_seen_20260604 4 \
      "${REASON_ADAPTER}" \
      "${DATA_BASE}_e36_s0_seed500_llm_checklist_reason_test_seen_pos.jsonl" \
      "${OUT_BASE}/e36_s0_seed500_reason/test_seen" 768 \
      "${LOG_BASE}/reason_test_seen.log"
    launch_eval stage2_1_7b_e36_s0_seed500_reason_eval_unseen_20260604 5 \
      "${REASON_ADAPTER}" \
      "${DATA_BASE}_e36_s0_seed500_llm_checklist_reason_test_unseen_pos.jsonl" \
      "${OUT_BASE}/e36_s0_seed500_reason/test_unseen" 768 \
      "${LOG_BASE}/reason_test_unseen.log"
    ;;
  *)
    echo "usage: $0 [all]" >&2
    exit 2
    ;;
esac
