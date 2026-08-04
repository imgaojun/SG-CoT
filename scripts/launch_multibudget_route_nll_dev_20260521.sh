#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
DATA_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"
SCORE_BASE="outputs/stage2_multibudget/route_likelihood_20260521"

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 branch routes gpu[,gpu...] [split]" >&2
  echo "example: $0 multibudget_ternary_router_m08_routecls_noauxwarm_lr2e6_save50 direct,reason_mid,reason_full 0 dev_seen" >&2
  exit 2
fi

BRANCH="$1"
ROUTES="$2"
HOST_GPUS="$3"
SPLIT="${4:-dev_seen}"
NAME="multibudget_route_nll_${BRANCH}_${SPLIT}_20260521"
RUN_DIR="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full"
EVAL_JSONL="data/stage2_adaptive_datasets/${DATA_PREFIX}_${BRANCH}_${SPLIT}_pos.jsonl"
SCORE_ROOT="${SCORE_BASE}/${BRANCH}"
LOG="${LOG_DIR}/${NAME}.log"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${NAME}"; then
  echo "container already exists: ${NAME}" >&2
  exit 1
fi

GPU_COUNT="$(printf '%s\n' "${HOST_GPUS}" | awk -F, '{print NF}')"
CONTAINER_GPUS="$(seq 0 "$((GPU_COUNT - 1))" | tr '\n' ' ' | sed 's/[[:space:]]*$//')"

docker run -d \
  --name "${NAME}" \
  --user root \
  --gpus "\"device=${HOST_GPUS}\"" \
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
    mkdir -p ${SCORE_ROOT} ${LOG_DIR}
    CKPTS=\$(find ${RUN_DIR} -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
    if [[ -z \"\${CKPTS}\" ]]; then
      echo \"no checkpoints found under ${RUN_DIR}\" >&2
      exit 1
    fi
    idx=0
    pids=()
    for ckpt in \${CKPTS}; do
      gpu_list=(${CONTAINER_GPUS})
      gpu=\${gpu_list[\$((idx % ${GPU_COUNT}))]}
      out_dir=${SCORE_ROOT}/\${ckpt}
      mkdir -p \${out_dir}
      CUDA_VISIBLE_DEVICES=\${gpu} python src/stage2_quality_validation/score_adaptive_multiroute_choice_likelihood.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${RUN_DIR}/\${ckpt} \
        --eval_jsonl ${EVAL_JSONL} \
        --routes ${ROUTES} \
        --output_jsonl \${out_dir}/${SPLIT}_scores.jsonl \
        --summary_json \${out_dir}/${SPLIT}_summary.json \
        --max_length 1024 >> ${LOG} 2>&1 &
      pids+=(\"\$!\")
      idx=\$((idx + 1))
      if [[ \${#pids[@]} -ge ${GPU_COUNT} ]]; then
        wait \"\${pids[0]}\"
        pids=(\"\${pids[@]:1}\")
      fi
    done
    for pid in \"\${pids[@]}\"; do wait \"\${pid}\"; done
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${SCORE_ROOT} ${LOG}
  "
