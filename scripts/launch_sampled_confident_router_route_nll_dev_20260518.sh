#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
SCORE_ROOT="${SCORE_ROOT:-outputs/stage2_modular_dualexpert/sampled_confident_router_20260518/route_likelihood}"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"

launch_branch() {
  local branch="$1"
  local host_gpu="$2"
  local name="sampled_confident_router_route_nll_${branch}_20260518"
  local run_dir="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${branch}_full"
  local eval_jsonl="data/stage2_adaptive_datasets/${DATA_PREFIX}_${branch}_dev_seen_pos.jsonl"
  local branch_score_root="${SCORE_ROOT}/${branch}"
  local log="${LOG_DIR}/sampled_confident_router_route_nll_${branch}_20260518.log"

  if [[ ! -s "${eval_jsonl}" ]]; then
    echo "missing eval jsonl: ${eval_jsonl}" >&2
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
      mkdir -p ${branch_score_root} ${LOG_DIR}
      CKPTS=\$(find ${run_dir} -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
      if [[ -z \"\${CKPTS}\" ]]; then
        echo \"no checkpoints found under ${run_dir}\" >&2
        exit 1
      fi
      : > ${log}
      for ckpt in \${CKPTS}; do
        mkdir -p ${branch_score_root}/\${ckpt}
        echo \"[route-nll] ${branch} \${ckpt}\" | tee -a ${log}
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
}

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 branch=gpu [branch=gpu ...]" >&2
  exit 2
fi

for item in "$@"; do
  branch="${item%%=*}"
  gpu="${item#*=}"
  launch_branch "${branch}" "${gpu}"
done
