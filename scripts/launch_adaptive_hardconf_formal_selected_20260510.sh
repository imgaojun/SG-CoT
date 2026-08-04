#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
CONTAINER_NAME="richere_qwen3_adaptive_hardconf_formal_selected_20260510"
MANIFEST="configs/generated/stage2_adaptive/richere_qwen3_1_7b_adaptive_hardconf_checkpoint_frontier_formal_selected_manifest.json"
STATUS="outputs/stage2_adaptive_runs_user_formal_clean/richere_qwen3_1_7b_adaptive_hardconf_checkpoint_frontier_formal_selected_status.json"
LOG="outputs/stage2_adaptive_runs_user_logs/richere_qwen3_1_7b_adaptive_hardconf_checkpoint_frontier_formal_selected.log"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "container already exists: ${CONTAINER_NAME}" >&2
  exit 1
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --user root \
  --gpus '"device=0,1,2,3"' \
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
    mkdir -p outputs/stage2_adaptive_runs_user_logs outputs/stage2_adaptive_runs_user_formal_clean
    python src/stage2_formal/orchestrate_best_eval_from_selection.py \
      --manifest_json ${MANIFEST} \
      --gpu_ids 0 1 2 3 \
      --batch_size 2 \
      --max_new_tokens 512 \
      --temperature 0.0 \
      --eval_script src/stage2_quality_validation/eval_adaptive_route_generation.py \
      --log_path ${LOG} \
      --status_json ${STATUS} \
      --reuse_existing
    chown -R 1000:1000 \
      outputs/stage2_adaptive_runs_user_formal_clean \
      outputs/stage2_adaptive_runs_user_logs
  "
