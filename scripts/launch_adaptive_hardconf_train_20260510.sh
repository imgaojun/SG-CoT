#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
CONFIG_PREFIX="configs/generated/stage2_adaptive/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"

launch_branch() {
  local branch="$1"
  local host_gpu="$2"
  local name="richere_qwen3_adaptive_${branch}_train_20260510"
  local config="${CONFIG_PREFIX}_${branch}_full_stepmatch.yaml"
  local log="${LOG_DIR}/${RUN_PREFIX}_${branch}_full_train.log"

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 1
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
    -e FORCE_TORCHRUN=1 \
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
      mkdir -p ${LOG_DIR}
      llamafactory-cli train ${config} 2>&1 | tee ${log}
      chown -R 1000:1000 \
        ${log} \
        outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${branch}_full
    "
}

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 branch=gpu [branch=gpu ...]" >&2
  echo "branches:" >&2
  echo "  hardconf10_heur10_type_role_hint_plan_lite" >&2
  echo "  hardconf15_heur15_type_role_hint_plan_lite" >&2
  echo "  hardconf10_calibrated_type_role_hint_plan_lite" >&2
  echo "  hardconf10_directdup" >&2
  exit 2
fi

for item in "$@"; do
  branch="${item%%=*}"
  gpu="${item#*=}"
  launch_branch "${branch}" "${gpu}"
done
