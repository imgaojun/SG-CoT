#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
ADAPTER="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_plan_lite_full"
EVAL_JSONL="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_plan_lite_forced_reason_train_pos.jsonl"
OUTPUT_DIR="outputs/stage2_multibudget/light_type_plan_lite_20260521/forced_reason/train"
LOG="outputs/stage2_adaptive_runs_user_logs/multibudget_light_type_plan_lite_forced_reason_train_20260521.log"
NAME="multibudget_light_type_plan_lite_forced_reason_train_20260521"
GPU="${1:-0}"

if [[ ! -s "${PROJECT_ROOT}/${EVAL_JSONL}" ]]; then
  echo "missing eval jsonl: ${PROJECT_ROOT}/${EVAL_JSONL}" >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${NAME}"; then
  echo "container already exists: ${NAME}" >&2
  exit 1
fi

docker run -d \
  --name "${NAME}" \
  --user root \
  --gpus "\"device=${GPU}\"" \
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
    mkdir -p ${OUTPUT_DIR} outputs/stage2_adaptive_runs_user_logs
    python src/stage2_quality_validation/eval_adaptive_route_generation.py \
      --base_model ${BASE_MODEL} \
      --adapter_path ${ADAPTER} \
      --eval_jsonl ${EVAL_JSONL} \
      --output_dir ${OUTPUT_DIR} \
      --max_new_tokens 512 \
      --temperature 0.0 \
      --batch_size 2 2>&1 | tee ${LOG}
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${OUTPUT_DIR} ${LOG}
  "
