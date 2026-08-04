#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH="sampled_k2_structproxy_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
SCORE_ROOT="${SCORE_ROOT:-outputs/stage2_modular_dualexpert/sampled_confident_router_20260518/route_likelihood}"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"

host_gpu="${1:-0}"
name="sampled_k2_structproxy_router_route_nll_20260519"
run_dir="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full"
eval_jsonl="data/stage2_adaptive_datasets/${DATA_PREFIX}_${BRANCH}_dev_seen_seedpairs_pos.jsonl"
branch_score_root="${SCORE_ROOT}/${BRANCH}"
log="${LOG_DIR}/sampled_k2_structproxy_router_route_nll_20260519.log"

if [[ ! -s "${eval_jsonl}" ]]; then
  echo "missing eval jsonl: ${eval_jsonl}" >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
  echo "container already exists: ${name}" >&2
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
    mkdir -p ${branch_score_root} ${LOG_DIR}
    CKPTS=\$(find ${run_dir} -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
    if [[ -z \"\${CKPTS}\" ]]; then
      echo \"no checkpoints found under ${run_dir}\" >&2
      exit 1
    fi
    : > ${log}
    for ckpt in \${CKPTS}; do
      mkdir -p ${branch_score_root}/\${ckpt}
      echo \"[route-nll] ${BRANCH} \${ckpt}\" | tee -a ${log}
      python src/stage2_quality_validation/score_adaptive_route_choice_likelihood.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${run_dir}/\${ckpt} \
        --eval_jsonl ${eval_jsonl} \
        --output_jsonl ${branch_score_root}/\${ckpt}/dev_seen_scores.jsonl \
        --summary_json ${branch_score_root}/\${ckpt}/dev_seen_summary.json \
        --max_length 1024 2>&1 | tee -a ${log}
    done
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${branch_score_root} ${log}
  "
