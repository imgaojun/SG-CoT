#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
ADAPTER="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_typeonlylite_directwarm_retention_e13b_full/checkpoint-1544"
DATA_PREFIX="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_typeonlylite_directwarm_retention_e13b_s14"
SCORE_ROOT="outputs/stage2_4b_reason_expert/e13b_route_nll_s14_20260522"
DEV_EXEC_ROOT="outputs/stage2_4b_reason_expert/e13b_dev_forced_execution_s14_20260522"
LOG_ROOT="outputs/stage2_4b_reason_expert/logs_s14_20260522"

docker_common() {
  docker run -d \
    --user root \
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
    "$@"
}

launch_score() {
  local split="$1"
  local gpu="$2"
  local eval_jsonl="${DATA_PREFIX}_${split}_pos.jsonl"
  local output_dir="${SCORE_ROOT}/${split}"
  local log="${LOG_ROOT}/route_nll_${split}.log"
  local name="stage2_4b_e13b_s14_route_nll_${split}_20260522"
  if [[ -s "${output_dir}/scores.jsonl" ]]; then
    echo "scores already exist: ${output_dir}/scores.jsonl"
    return 0
  fi
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${output_dir} ${LOG_ROOT}
    python src/stage2_quality_validation/score_adaptive_route_choice_likelihood.py \
      --base_model ${BASE_MODEL} \
      --adapter_path ${ADAPTER} \
      --eval_jsonl ${eval_jsonl} \
      --output_jsonl ${output_dir}/scores.jsonl \
      --summary_json ${output_dir}/summary.json \
      --max_length 1024 2>&1 | tee ${log}
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${output_dir} ${log}
  "
}

launch_dev_exec() {
  local mode="$1"
  local gpu="$2"
  local eval_jsonl="${DATA_PREFIX}_${mode}_dev_seen_pos.jsonl"
  local output_dir="${DEV_EXEC_ROOT}/${mode}"
  local log="${LOG_ROOT}/dev_exec_${mode}.log"
  local name="stage2_4b_e13b_s14_dev_exec_${mode}_20260522"
  if [[ -s "${output_dir}/predictions.jsonl" ]]; then
    echo "predictions already exist: ${output_dir}/predictions.jsonl"
    return 0
  fi
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${output_dir} ${LOG_ROOT}
    python src/stage2_quality_validation/eval_adaptive_route_generation.py \
      --base_model ${BASE_MODEL} \
      --adapter_path ${ADAPTER} \
      --eval_jsonl ${eval_jsonl} \
      --output_dir ${output_dir} \
      --batch_size 8 \
      --max_new_tokens 512 \
      --temperature 0.0 2>&1 | tee ${log}
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${output_dir} ${log}
  "
}

case "${1:-}" in
  launch)
    launch_score dev_seen "${2:-0}"
    launch_score test_seen "${3:-1}"
    launch_score test_unseen "${4:-2}"
    launch_dev_exec forced_direct "${5:-3}"
    launch_dev_exec forced_reason "${6:-4}"
    ;;
  launch-score)
    launch_score dev_seen "${2:-0}"
    launch_score test_seen "${3:-1}"
    launch_score test_unseen "${4:-2}"
    ;;
  launch-dev-exec)
    launch_dev_exec forced_direct "${2:-0}"
    launch_dev_exec forced_reason "${3:-1}"
    ;;
  *)
    echo "usage: $0 {launch|launch-score|launch-dev-exec} [gpu...]" >&2
    exit 2
    ;;
esac
