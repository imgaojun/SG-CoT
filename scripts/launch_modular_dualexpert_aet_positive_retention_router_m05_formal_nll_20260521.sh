#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH="aet_positive_retention_router_m05_routecls_noauxwarm_lr2e6_save50"
RUN_DIR="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full"
SCORE_ROOT="outputs/stage2_modular_dualexpert/aet_positive_retention_router_m05_20260521/formal_route_likelihood/${BRANCH}"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"

launch_score() {
  local ckpt="$1"
  local split="$2"
  local host_gpu="$3"
  local name="modular_dualexpert_aet_m05_formal_nll_${ckpt}_${split}_20260521"
  local eval_jsonl="data/stage2_adaptive_datasets/${DATA_PREFIX}_${BRANCH}_${split}_pos.jsonl"
  local output_dir="${SCORE_ROOT}/${ckpt}/${split}"
  local log="${LOG_DIR}/modular_dualexpert_aet_m05_formal_nll_${ckpt}_${split}_20260521.log"

  if [[ ! -s "${eval_jsonl}" ]]; then
    echo "missing eval jsonl: ${eval_jsonl}" >&2
    return 1
  fi
  if [[ ! -d "${RUN_DIR}/${ckpt}" ]]; then
    echo "missing checkpoint: ${RUN_DIR}/${ckpt}" >&2
    return 1
  fi
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists, skipping launch: ${name}" >&2
    return 0
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
      python src/stage2_quality_validation/score_adaptive_route_choice_likelihood.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${RUN_DIR}/${ckpt} \
        --eval_jsonl ${eval_jsonl} \
        --output_jsonl ${output_dir}/scores.jsonl \
        --summary_json ${output_dir}/summary.json \
        --max_length 1024 2>&1 | tee ${log}
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${output_dir} ${log}
    "
}

case "${1:-}" in
  launch)
    ckpt="${2:?checkpoint required, e.g. checkpoint-150}"
    launch_score "${ckpt}" test_seen "${3:-0}"
    launch_score "${ckpt}" test_unseen "${4:-1}"
    ;;
  *)
    echo "usage: $0 launch <checkpoint> [gpu_seen] [gpu_unseen]" >&2
    exit 2
    ;;
esac
