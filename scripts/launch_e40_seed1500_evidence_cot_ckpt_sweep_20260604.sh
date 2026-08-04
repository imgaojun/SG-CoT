#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
BRANCH="e40_seed1500_thinking_evidence_cot"
RUN_BASE="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${BRANCH}"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py"
OUT_BASE="outputs/stage2_strategy_cot_e40/formal_e40_seed1500_ckpt_sweep_20260604"
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

launch_eval() {
  local ckpt="$1"
  local split="$2"
  local gpu="$3"
  local name="stage2_1_7b_e40_evidence_${ckpt}_${split}_20260604"
  local adapter="${RUN_BASE}_${BRANCH}_full/checkpoint-${ckpt}"
  local output_dir="${OUT_BASE}/checkpoint-${ckpt}/${split}"
  local log_path="${LOG_BASE}/checkpoint-${ckpt}_${split}.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${output_dir}' '${LOG_BASE}'
    python '${EVAL_SCRIPT}' \
      --base_model '${BASE_MODEL}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_${split}_pos.jsonl' \
      --output_dir '${output_dir}' \
      --batch_size 8 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${output_dir}' '${log_path}'
  "
}

case "${1:-all}" in
  all)
    mkdir -p "${LOG_BASE}"
    launch_eval 83 test_seen 1
    launch_eval 83 test_unseen 2
    launch_eval 166 test_seen 3
    launch_eval 166 test_unseen 4
    launch_eval 249 test_seen 5
    launch_eval 249 test_unseen 6
    ;;
  *)
    echo "usage: $0 [all]" >&2
    exit 2
    ;;
esac
