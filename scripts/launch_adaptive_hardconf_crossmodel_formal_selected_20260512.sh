#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
LOG_ROOT="outputs/stage2_adaptive_runs_user_logs"
FORMAL_ROOT="outputs/stage2_adaptive_runs_user_formal_clean"

launch_formal() {
  local name="$1"
  local manifest="$2"
  local status="$3"
  local log="$4"
  local host_gpus="$5"

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists, skipping launch: ${name}" >&2
    return 0
  fi

  docker run -d \
    --name "${name}" \
    --user root \
    --gpus "\"device=${host_gpus}\"" \
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
      mkdir -p ${LOG_ROOT} ${FORMAL_ROOT}
      python src/stage2_formal/orchestrate_best_eval_from_selection.py \
        --manifest_json ${manifest} \
        --gpu_ids 0 1 \
        --batch_size 1 \
        --max_new_tokens 512 \
        --temperature 0.0 \
        --eval_script src/stage2_quality_validation/eval_adaptive_route_generation.py \
        --log_path ${log} \
        --status_json ${status} \
        --reuse_existing
      chown -R 1000:1000 ${FORMAL_ROOT} ${LOG_ROOT}
    "
}

launch_formal \
  "richere_qwen3_4b_adaptive_hardconf_formal_selected_20260512" \
  "configs/generated/stage2_adaptive/richere_adaptive_hardconf_crossmodel_checkpoint_frontier_formal_selected_manifest.json" \
  "outputs/stage2_adaptive_runs_user_formal_clean/richere_qwen3_4b_adaptive_hardconf_checkpoint_frontier_formal_selected_status.json" \
  "outputs/stage2_adaptive_runs_user_logs/richere_qwen3_4b_adaptive_hardconf_checkpoint_frontier_formal_selected.log" \
  "4,6"

launch_formal \
  "richere_llama3_2_3b_adaptive_hardconf_formal_selected_20260512" \
  "configs/generated/stage2_adaptive/richere_llama3_2_3b_adaptive_hardconf_checkpoint_frontier_formal_selected_manifest.json" \
  "outputs/stage2_adaptive_runs_user_formal_clean/richere_llama3_2_3b_adaptive_hardconf_checkpoint_frontier_formal_selected_status.json" \
  "outputs/stage2_adaptive_runs_user_logs/richere_llama3_2_3b_adaptive_hardconf_checkpoint_frontier_formal_selected.log" \
  "2,3"
