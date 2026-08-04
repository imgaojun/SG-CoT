#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="data/stage2_adaptive_datasets/${ADAPT_PREFIX}"
OUTPUT_ROOT="outputs/stage2_modular_dualexpert/sampled_counterfactual_utility_20260517"
LOG_DIR="outputs/stage2_modular_dualexpert/sampled_counterfactual_utility_20260517/logs"
DIRECT_BRANCH="outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux"
REASON_BRANCH="sampled_reason_expert_forcedreason_from_noaux_20260517"
DIRECT_ADAPTER="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${DIRECT_BRANCH}_full/checkpoint-1930"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 <reason_checkpoint_tag> <direct|reason> <train|dev_seen> <gpu> [seed ...]" >&2
  echo "example: $0 checkpoint-258 direct dev_seen 2 17 18 19 20 21 22 23 24" >&2
  exit 2
fi

REASON_CKPT="$1"
ROUTE="$2"
SPLIT="$3"
HOST_GPU="$4"
shift 4

if [[ "${ROUTE}" != "direct" && "${ROUTE}" != "reason" ]]; then
  echo "unsupported route: ${ROUTE}" >&2
  exit 2
fi
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "dev_seen" ]]; then
  echo "unsupported split: ${SPLIT}" >&2
  exit 2
fi

if [[ "$#" -eq 0 ]]; then
  SEEDS=(17 18 19 20 21 22 23 24)
else
  SEEDS=("$@")
fi

RUN_ID="${REASON_BRANCH}_${REASON_CKPT}"
REASON_ADAPTER="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${REASON_BRANCH}_full/${REASON_CKPT}"
if [[ "${ROUTE}" == "direct" ]]; then
  ADAPTER="${DIRECT_ADAPTER}"
  EVAL_JSONL="${DATA_PREFIX}_${REASON_BRANCH}_forced_direct_${SPLIT}_pos.jsonl"
else
  ADAPTER="${REASON_ADAPTER}"
  EVAL_JSONL="${DATA_PREFIX}_${REASON_BRANCH}_forced_reason_${SPLIT}_pos.jsonl"
fi

if [[ ! -d "${ADAPTER}" ]]; then
  echo "missing adapter: ${ADAPTER}" >&2
  exit 1
fi
if [[ ! -s "${EVAL_JSONL}" ]]; then
  echo "missing eval jsonl: ${EVAL_JSONL}" >&2
  exit 1
fi

NAME="sampled_counterfactual_k8_${SPLIT}_${ROUTE}_${REASON_CKPT}_20260517"
LOG="${LOG_DIR}/${SPLIT}_${ROUTE}_${REASON_CKPT}.log"

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
    mkdir -p ${LOG_DIR}
    exec > >(tee -a ${LOG}) 2>&1
    for seed in ${SEEDS[*]}; do
      out_dir=\"${OUTPUT_ROOT}/${RUN_ID}/${SPLIT}/${ROUTE}/seed-\${seed}\"
      if [[ -s \"\${out_dir}/predictions.jsonl\" ]]; then
        echo \"skip existing ${ROUTE} ${SPLIT} seed-\${seed}\"
        continue
      fi
      python src/stage2_quality_validation/eval_adaptive_route_generation_samples.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${ADAPTER} \
        --eval_jsonl ${EVAL_JSONL} \
        --output_dir \"\${out_dir}\" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --top_k ${TOP_K} \
        --batch_size ${BATCH_SIZE} \
        --seed \"\${seed}\" \
        --sample_id \"seed-\${seed}\" \
        --route_mode ${ROUTE}
    done
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${OUTPUT_ROOT}/${RUN_ID}/${SPLIT}/${ROUTE} ${LOG}
  "
