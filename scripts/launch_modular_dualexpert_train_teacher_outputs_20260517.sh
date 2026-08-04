#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DIRECT_BRANCH="outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux"
REASON_BRANCH="outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux"
DIRECT_CKPT="checkpoint-1930"
REASON_CKPT="checkpoint-2058"
DIRECT_ADAPTER="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${DIRECT_BRANCH}_full/${DIRECT_CKPT}"
REASON_ADAPTER="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${REASON_BRANCH}_full/${REASON_CKPT}"
DIRECT_EVAL_JSONL="data/stage2_adaptive_datasets/${DATA_PREFIX}_${DIRECT_BRANCH}_forced_direct_train_pos.jsonl"
REASON_EVAL_JSONL="data/stage2_adaptive_datasets/${DATA_PREFIX}_${REASON_BRANCH}_forced_reason_train_pos.jsonl"
OUT_ROOT="outputs/stage2_modular_dualexpert/train_teacher_outputs_d1930_r2058_20260517"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"

DIRECT_GPU="${1:-1}"
REASON_GPU="${2:-2}"

launch_eval() {
  local name="$1"
  local host_gpu="$2"
  local adapter="$3"
  local eval_jsonl="$4"
  local output_dir="$5"
  local log="$6"

  if [[ ! -s "${eval_jsonl}" ]]; then
    echo "missing eval jsonl: ${eval_jsonl}" >&2
    exit 1
  fi
  if [[ ! -d "${adapter}" ]]; then
    echo "missing adapter: ${adapter}" >&2
    exit 1
  fi
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    exit 1
  fi
  if [[ -d "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -maxdepth 1 | grep -q .; then
    echo "output dir already has content: ${output_dir}" >&2
    exit 1
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
      mkdir -p ${output_dir} ${LOG_DIR}
      python src/stage2_quality_validation/eval_adaptive_route_generation.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${adapter} \
        --eval_jsonl ${eval_jsonl} \
        --output_dir ${output_dir} \
        --batch_size 8 \
        --max_new_tokens 512 \
        --temperature 0.0 2>&1 | tee ${log}
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${output_dir} ${log}
    "
}

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
python3 scripts/build_modular_dualexpert_train_teacher_inputs_20260517.py

launch_eval \
  "modular_dualexpert_teacher_d1930_direct_train_20260517" \
  "${DIRECT_GPU}" \
  "${DIRECT_ADAPTER}" \
  "${DIRECT_EVAL_JSONL}" \
  "${OUT_ROOT}/direct_expert_forced_direct_train" \
  "${LOG_DIR}/modular_dualexpert_teacher_d1930_direct_train_20260517.log"

launch_eval \
  "modular_dualexpert_teacher_r2058_reason_train_20260517" \
  "${REASON_GPU}" \
  "${REASON_ADAPTER}" \
  "${REASON_EVAL_JSONL}" \
  "${OUT_ROOT}/reason_expert_forced_reason_train" \
  "${LOG_DIR}/modular_dualexpert_teacher_r2058_reason_train_20260517.log"
