#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
ROUTER_ADAPTER="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome15_l15bal30_routecls_balanced_probe_full/checkpoint-221"
DATA_PREFIX="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_outcome15_l15bal30_routecls_scorebase"
SCORE_ROOT="outputs/stage2_adaptive_route_likelihood_probe/outcome15cal_20260514"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"

launch_split() {
  local split="$1"
  local host_gpu="$2"
  local name="adaptive_outcome15cal_route_score_${split}_20260514"
  local eval_jsonl="${DATA_PREFIX}_${split}_pos.jsonl"
  local output_jsonl="${SCORE_ROOT}/${split}_scores.jsonl"
  local summary_json="${SCORE_ROOT}/${split}_summary.json"
  local log="${LOG_DIR}/adaptive_outcome15cal_route_score_${split}_20260514.log"

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
      mkdir -p ${SCORE_ROOT} ${LOG_DIR}
      python src/stage2_quality_validation/score_adaptive_route_choice_likelihood.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${ROUTER_ADAPTER} \
        --eval_jsonl ${eval_jsonl} \
        --output_jsonl ${output_jsonl} \
        --summary_json ${summary_json} \
        --max_length 1024 2>&1 | tee ${log}
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${SCORE_ROOT} ${log}
    "
}

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 train=<gpu> dev_seen=<gpu>" >&2
  exit 2
fi

for item in "$@"; do
  split="${item%%=*}"
  gpu="${item#*=}"
  launch_split "${split}" "${gpu}"
done
