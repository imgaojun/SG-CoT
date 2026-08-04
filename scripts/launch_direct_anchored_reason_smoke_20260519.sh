#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
REASON_ADAPTER="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_sampled_reason_expert_forcedreason_from_noaux_20260517_full/checkpoint-258"
POLICY="fm0p25_mr0p75_n1_aj0p40_ed0p00_am0p00"
PROMPT_VARIANT="${PROMPT_VARIANT:-anchor_conservative}"
EVAL_JSONL="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_directanchored_reason_20260519_${POLICY}_${PROMPT_VARIANT}_test_pos.jsonl"
OUTPUT_DIR="outputs/stage2_adaptive_direct_anchored_reason_generation_20260519/${POLICY}_${PROMPT_VARIANT}_tagged"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"
LOG="${LOG_DIR}/direct_anchored_reason_smoke_${POLICY}_${PROMPT_VARIANT}_tagged_20260519.log"
NAME="direct_anchored_reason_smoke_${POLICY}_${PROMPT_VARIANT}_tagged_20260519"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 <gpu> [anchor_conservative|anchor_revise|anchor_verify]" >&2
  exit 2
fi

HOST_GPU="$1"
if [[ "$#" -eq 2 ]]; then
  PROMPT_VARIANT="$2"
  EVAL_JSONL="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_directanchored_reason_20260519_${POLICY}_${PROMPT_VARIANT}_test_pos.jsonl"
  OUTPUT_DIR="outputs/stage2_adaptive_direct_anchored_reason_generation_20260519/${POLICY}_${PROMPT_VARIANT}_tagged"
  LOG="${LOG_DIR}/direct_anchored_reason_smoke_${POLICY}_${PROMPT_VARIANT}_tagged_20260519.log"
  NAME="direct_anchored_reason_smoke_${POLICY}_${PROMPT_VARIANT}_tagged_20260519"
fi

if [[ "${PROMPT_VARIANT}" != "anchor_conservative" && "${PROMPT_VARIANT}" != "anchor_revise" && "${PROMPT_VARIANT}" != "anchor_verify" ]]; then
  echo "unsupported prompt variant: ${PROMPT_VARIANT}" >&2
  exit 2
fi

if [[ ! -d "${REASON_ADAPTER}" ]]; then
  echo "missing reason adapter: ${REASON_ADAPTER}" >&2
  exit 1
fi
if [[ ! -s "${EVAL_JSONL}" ]]; then
  echo "missing eval jsonl: ${EVAL_JSONL}" >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${NAME}"; then
  echo "container already exists, skipping launch: ${NAME}" >&2
  exit 0
fi

docker run -d \
  --name "${NAME}" \
  --user root \
  --gpus "\"device=${HOST_GPU}\"" \
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
    mkdir -p ${OUTPUT_DIR} ${LOG_DIR}
    python src/stage2_quality_validation/eval_adaptive_route_generation.py \
      --base_model ${BASE_MODEL} \
      --adapter_path ${REASON_ADAPTER} \
      --eval_jsonl ${EVAL_JSONL} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size ${BATCH_SIZE} \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --temperature ${TEMPERATURE} 2>&1 | tee ${LOG}
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${OUTPUT_DIR} ${LOG}
  "
