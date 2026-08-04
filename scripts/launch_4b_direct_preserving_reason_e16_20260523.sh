#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
RUN_PREFIX="richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
LOG_ROOT="outputs/stage2_4b_direct_preserving_reason/e16_logs_20260523"
DEVPICK_ROOT="outputs/stage2_4b_direct_preserving_reason/e16_devpick_20260523"
FORMAL_ROOT="outputs/stage2_4b_direct_preserving_reason/e16_formal_20260523"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation.py"
SHORTLIST_SCRIPT="src/stage2_formal/parallel_shortlist_dev_select.py"

branch_for() {
  case "$1" in
    e16a_noreasonblock_directpreserve) echo "confrare10_typeonlylite_reasonfmt_e16a_noreasonblock_directpreserve" ;;
    e16c_finalfirst_directpreserve) echo "confrare10_typeonlylite_reasonfmt_e16c_finalfirst_directpreserve" ;;
    *) echo "unknown variant: $1" >&2; return 2 ;;
  esac
}

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

run_dir_for() {
  local branch="$1"
  echo "outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${branch}_full"
}

config_for() {
  local branch="$1"
  echo "configs/generated/stage2_adaptive/${RUN_PREFIX}_${branch}_full_stepmatch.yaml"
}

data_prefix_for() {
  local branch="$1"
  echo "${DATA_BASE}_${branch}"
}

selected_adapter() {
  local variant="$1"
  local branch run_dir summary
  branch="$(branch_for "${variant}")"
  run_dir="$(run_dir_for "${branch}")"
  summary="${DEVPICK_ROOT}/${variant}/free_route/selection_summary.json"
  if [[ -s "${summary}" ]]; then
    python3 - "$summary" "$run_dir" <<'PY'
import json
import sys
from pathlib import Path
summary = Path(sys.argv[1])
fallback = Path(sys.argv[2])
data = json.loads(summary.read_text())
raw = data["best"]["checkpoint_path"].replace("/workspace/project/", "")
candidate = Path(raw)
print(candidate.as_posix() if candidate.exists() else fallback.as_posix())
PY
  else
    echo "${run_dir}"
  fi
}

train_one() {
  local variant="$1"
  local gpu="$2"
  local branch run_dir config log name
  branch="$(branch_for "${variant}")"
  run_dir="$(run_dir_for "${branch}")"
  config="$(config_for "${branch}")"
  log="${LOG_ROOT}/${variant}_train.log"
  name="stage2_4b_e16_${variant}_train_20260523"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${LOG_ROOT}
    FORCE_TORCHRUN=1 llamafactory-cli train ${config} 2>&1 | tee ${log}
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${run_dir} ${log}
  "
}

devpick_one() {
  local variant="$1"
  local gpu="$2"
  local branch run_dir data_prefix output_root log name
  branch="$(branch_for "${variant}")"
  run_dir="$(run_dir_for "${branch}")"
  data_prefix="$(data_prefix_for "${branch}")"
  output_root="${DEVPICK_ROOT}/${variant}/free_route"
  log="${LOG_ROOT}/${variant}_devpick.log"
  name="stage2_4b_e16_${variant}_devpick_20260523"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${LOG_ROOT} ${output_root}
    ckpts=\$(find ${run_dir} -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
    python ${SHORTLIST_SCRIPT} \
      --base_model ${BASE_MODEL} \
      --run_dir ${run_dir} \
      --eval_jsonl ${data_prefix}_dev_seen_pos.jsonl \
      --output_root ${output_root} \
      --checkpoint_tags \${ckpts} \
      --gpu_ids 0 \
      --metric_keys argument_f1 event_f1 trigger_f1 \
      --greater_is_better \
      --batch_size 1 \
      --max_new_tokens 512 \
      --temperature 0.0 \
      --eval_script ${EVAL_SCRIPT} \
      --log_path ${log} \
      --status_json ${output_root}/status.json \
      --reuse_existing
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${DEVPICK_ROOT}/${variant} ${log}
  "
}

eval_one() {
  local variant="$1"
  local mode="$2"
  local split="$3"
  local gpu="$4"
  local branch adapter data_prefix eval_jsonl output_dir log name
  branch="$(branch_for "${variant}")"
  adapter="$(selected_adapter "${variant}")"
  data_prefix="$(data_prefix_for "${branch}")"
  eval_jsonl="${data_prefix}_${mode}_${split}_pos.jsonl"
  output_dir="${FORMAL_ROOT}/${variant}/${mode}/${split}"
  log="${LOG_ROOT}/${variant}_${mode}_${split}.log"
  name="stage2_4b_e16_${variant}_${mode}_${split}_20260523"
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
    echo adapter_path=${adapter}
    python ${EVAL_SCRIPT} \
      --base_model ${BASE_MODEL} \
      --adapter_path ${adapter} \
      --eval_jsonl ${eval_jsonl} \
      --output_dir ${output_dir} \
      --batch_size 8 \
      --max_new_tokens 512 \
      --temperature 0.0 2>&1 | tee ${log}
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${output_dir} ${log}
  "
}

formal_one() {
  local variant="$1"
  eval_one "${variant}" forced_direct test_seen "${2:-0}"
  eval_one "${variant}" forced_reason test_seen "${3:-1}"
  eval_one "${variant}" forced_direct test_unseen "${4:-2}"
  eval_one "${variant}" forced_reason test_unseen "${5:-3}"
}

case "${1:-}" in
  train)
    train_one "${2:?variant}" "${3:-0}"
    ;;
  train-all)
    train_one e16a_noreasonblock_directpreserve "${2:-0}"
    train_one e16c_finalfirst_directpreserve "${3:-1}"
    ;;
  devpick)
    devpick_one "${2:?variant}" "${3:-0}"
    ;;
  devpick-all)
    devpick_one e16a_noreasonblock_directpreserve "${2:-0}"
    devpick_one e16c_finalfirst_directpreserve "${3:-1}"
    ;;
  formal)
    formal_one "${2:?variant}" "${3:-0}" "${4:-1}" "${5:-2}" "${6:-3}"
    ;;
  *)
    echo "usage: $0 {train|train-all|devpick|devpick-all|formal} ..." >&2
    exit 2
    ;;
esac
