#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
DATA_PREFIX="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_ROOT="outputs/stage2_adaptive_runs_user"
SUMMARY_ROOT="outputs/stage2_adaptive_runs_user_devpick"
SCORE_ROOT="outputs/stage2_4b_selector/route_likelihood_s12_20260521"
DEV_EXEC_ROOT="outputs/stage2_4b_selector/dev_forced_execution_s12_20260521"
LOG_ROOT="outputs/stage2_4b_selector/logs_s12_20260521"

system_branch() {
  case "$1" in
    typeonlylite) echo "confrare10_heur10_typeonlylite" ;;
    typerolelite) echo "confrare10_heur10_typerolelite" ;;
    *) echo "unknown system: $1" >&2; return 2 ;;
  esac
}

summary_path() {
  local branch="$1"
  echo "${SUMMARY_ROOT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_${branch}_full_free_dev_seen_max512/selection_summary.json"
}

run_dir() {
  local branch="$1"
  echo "${RUN_ROOT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_${branch}_full"
}

adapter_path() {
  local branch="$1"
  local summary
  summary="$(summary_path "${branch}")"
  python3 - "$summary" "$(run_dir "${branch}")" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
fallback = Path(sys.argv[2])
data = json.loads(summary.read_text())
raw = data["best"]["checkpoint_path"].replace("/workspace/project/", "")
candidate = Path(raw)
if candidate.exists():
    print(candidate.as_posix())
else:
    print(fallback.as_posix())
PY
}

eval_jsonl() {
  local branch="$1"
  local split="$2"
  echo "${DATA_PREFIX}_${branch}_${split}_pos.jsonl"
}

forced_eval_jsonl() {
  local branch="$1"
  local mode="$2"
  echo "${DATA_PREFIX}_${branch}_${mode}_dev_seen_pos.jsonl"
}

launch_one() {
  local system="$1"
  local split="$2"
  local host_gpu="$3"
  local branch adapter eval output_dir log container
  branch="$(system_branch "${system}")"
  adapter="$(adapter_path "${branch}")"
  eval="$(eval_jsonl "${branch}" "${split}")"
  output_dir="${SCORE_ROOT}/${system}/${split}"
  log="${LOG_ROOT}/${system}_${split}.log"
  container="stage2_4b_s12_route_nll_${system}_${split}_20260521"

  if [[ ! -s "${eval}" ]]; then
    echo "missing eval jsonl: ${eval}" >&2
    return 1
  fi
  if [[ ! -e "${adapter}/config.json" && ! -e "${adapter}/adapter_config.json" ]]; then
    echo "missing adapter/model path: ${adapter}" >&2
    return 1
  fi
  if [[ -s "${output_dir}/scores.jsonl" ]]; then
    echo "scores already exist, skipping: ${output_dir}/scores.jsonl"
    return 0
  fi
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${container}"; then
    echo "container already exists, skipping launch: ${container}" >&2
    return 0
  fi

  docker run -d \
    --name "${container}" \
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
      mkdir -p ${output_dir} ${LOG_ROOT}
      echo adapter_path=${adapter}
      python src/stage2_quality_validation/score_adaptive_route_choice_likelihood.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${adapter} \
        --eval_jsonl ${eval} \
        --output_jsonl ${output_dir}/scores.jsonl \
        --summary_json ${output_dir}/summary.json \
        --max_length 1024 2>&1 | tee ${log}
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${output_dir} ${log}
    "
}

launch_dev_exec_one() {
  local system="$1"
  local mode="$2"
  local host_gpu="$3"
  local branch adapter eval output_dir log container
  branch="$(system_branch "${system}")"
  adapter="$(adapter_path "${branch}")"
  eval="$(forced_eval_jsonl "${branch}" "${mode}")"
  output_dir="${DEV_EXEC_ROOT}/${system}/${mode}"
  log="${LOG_ROOT}/${system}_${mode}_dev_seen_execution.log"
  container="stage2_4b_s12_dev_exec_${system}_${mode}_20260521"

  if [[ ! -s "${eval}" ]]; then
    echo "missing eval jsonl: ${eval}" >&2
    return 1
  fi
  if [[ ! -e "${adapter}/config.json" && ! -e "${adapter}/adapter_config.json" ]]; then
    echo "missing adapter/model path: ${adapter}" >&2
    return 1
  fi
  if [[ -s "${output_dir}/predictions.jsonl" ]]; then
    echo "predictions already exist, skipping: ${output_dir}/predictions.jsonl"
    return 0
  fi
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${container}"; then
    echo "container already exists, skipping launch: ${container}" >&2
    return 0
  fi

  docker run -d \
    --name "${container}" \
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
      mkdir -p ${output_dir} ${LOG_ROOT}
      echo adapter_path=${adapter}
      python src/stage2_quality_validation/eval_adaptive_route_generation.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${adapter} \
        --eval_jsonl ${eval} \
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
    launch_one typeonlylite dev_seen "${2:-0}"
    launch_one typerolelite dev_seen "${3:-1}"
    launch_one typeonlylite test_seen "${4:-2}"
    launch_one typerolelite test_seen "${5:-3}"
    launch_one typeonlylite test_unseen "${6:-4}"
    launch_one typerolelite test_unseen "${7:-5}"
    ;;
  launch-dev)
    launch_one typeonlylite dev_seen "${2:-0}"
    launch_one typerolelite dev_seen "${3:-1}"
    ;;
  launch-formal)
    launch_one typeonlylite test_seen "${2:-0}"
    launch_one typerolelite test_seen "${3:-1}"
    launch_one typeonlylite test_unseen "${4:-2}"
    launch_one typerolelite test_unseen "${5:-3}"
    ;;
  launch-dev-exec)
    launch_dev_exec_one typeonlylite forced_direct "${2:-0}"
    launch_dev_exec_one typeonlylite forced_reason "${3:-1}"
    launch_dev_exec_one typerolelite forced_direct "${4:-2}"
    launch_dev_exec_one typerolelite forced_reason "${5:-3}"
    ;;
  *)
    echo "usage: $0 {launch|launch-dev|launch-formal|launch-dev-exec} [gpu...]" >&2
    exit 2
    ;;
esac
