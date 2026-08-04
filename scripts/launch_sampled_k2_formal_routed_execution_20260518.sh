#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DIRECT_BRANCH="outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux"
REASON_BRANCH="sampled_reason_expert_forcedreason_from_noaux_20260517"
DIRECT_ADAPTER="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${DIRECT_BRANCH}_full/checkpoint-1930"
REASON_ADAPTER="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${REASON_BRANCH}_full/checkpoint-258"
OUTPUT_ROOT="outputs/stage2_adaptive_route_formal_execution_20260518/sampledk2_ckpt50_margin025"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 <direct|reason> <test_seen|test_unseen> <gpu>" >&2
  exit 2
fi

ROUTE="$1"
SPLIT="$2"
HOST_GPU="$3"

if [[ "${ROUTE}" != "direct" && "${ROUTE}" != "reason" ]]; then
  echo "unsupported route: ${ROUTE}" >&2
  exit 2
fi
if [[ "${SPLIT}" != "test_seen" && "${SPLIT}" != "test_unseen" ]]; then
  echo "unsupported split: ${SPLIT}" >&2
  exit 2
fi

if [[ "${ROUTE}" == "direct" ]]; then
  ADAPTER="${DIRECT_ADAPTER}"
  MODE="forced_direct"
else
  ADAPTER="${REASON_ADAPTER}"
  MODE="forced_reason"
fi

EVAL_JSONL="data/stage2_adaptive_datasets/${ADAPT_PREFIX}_${REASON_BRANCH}_${MODE}_${SPLIT}_pos.jsonl"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODE}/${SPLIT}"
LOG="${LOG_DIR}/sampled_k2_formal_routed_execution_${MODE}_${SPLIT}_20260518.log"
NAME="sampled_k2_formal_routed_exec_${MODE}_${SPLIT}_20260518"

if [[ ! -d "${ADAPTER}" ]]; then
  echo "missing adapter: ${ADAPTER}" >&2
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
      --adapter_path ${ADAPTER} \
      --eval_jsonl ${EVAL_JSONL} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size ${BATCH_SIZE} \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --temperature ${TEMPERATURE} 2>&1 | tee ${LOG}
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${OUTPUT_DIR} ${LOG}
  "
