#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
LOG_ROOT="outputs/stage2_1_7b_event_mentions_budget/e19_logs_20260524"
DEVPICK_ROOT="outputs/stage2_1_7b_event_mentions_budget/e19_devpick_20260524"
FORMAL_ROOT="outputs/stage2_1_7b_event_mentions_budget/e19_formal_20260524"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation.py"
SHORTLIST_SCRIPT="src/stage2_formal/parallel_shortlist_dev_select.py"

branch_for() {
  case "$1" in
    e19a) echo "eventmentions_budget_e19a_mixed_eventpos_r5" ;;
    e19b) echo "eventmentions_budget_e19b_standardonly" ;;
    *) echo "unknown variant: $1" >&2; exit 2 ;;
  esac
}

budgets_for() {
  case "$1" in
    e19a) echo "none standard" ;;
    e19b) echo "standard" ;;
    *) echo "unknown variant: $1" >&2; exit 2 ;;
  esac
}

docker_common() {
  docker run -d --user root --ipc host --shm-size 16g \
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
    -w /workspace/project "$@"
}

run_dir() {
  local branch
  branch="$(branch_for "$1")"
  echo "outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${branch}_full"
}

config_path() {
  local branch
  branch="$(branch_for "$1")"
  echo "configs/generated/stage2_adaptive/${RUN_PREFIX}_${branch}_full_stepmatch.yaml"
}

data_prefix() {
  local branch
  branch="$(branch_for "$1")"
  echo "${DATA_BASE}_${branch}"
}

selected_adapter() {
  local variant="$1"
  local summary="${DEVPICK_ROOT}/${variant}/free_budget/selection_summary.json"
  local fallback
  fallback="$(run_dir "${variant}")"
  if [[ -s "${summary}" ]]; then
    python3 - "$summary" "$fallback" <<'PY'
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
    echo "${fallback}"
  fi
}

train_one() {
  local variant="$1"
  local gpu="$2"
  local name="stage2_1_7b_${variant}_eventmentions_budget_train_20260524"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${LOG_ROOT}/${variant}
    FORCE_TORCHRUN=1 llamafactory-cli train $(config_path "${variant}") 2>&1 | tee ${LOG_ROOT}/${variant}/train.log
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} $(run_dir "${variant}") ${LOG_ROOT}/${variant}/train.log
  "
}

devpick_one() {
  local variant="$1"
  local gpu="$2"
  local name="stage2_1_7b_${variant}_eventmentions_budget_devpick_20260524"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${LOG_ROOT}/${variant} ${DEVPICK_ROOT}/${variant}/free_budget
    ckpts=\$(find $(run_dir "${variant}") -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
    python ${SHORTLIST_SCRIPT} \
      --base_model ${BASE_MODEL} \
      --run_dir $(run_dir "${variant}") \
      --eval_jsonl $(data_prefix "${variant}")_dev_seen_pos.jsonl \
      --output_root ${DEVPICK_ROOT}/${variant}/free_budget \
      --checkpoint_tags \${ckpts} \
      --gpu_ids 0 \
      --metric_keys event_f1 argument_f1 trigger_f1 \
      --greater_is_better \
      --batch_size 1 \
      --max_new_tokens 768 \
      --temperature 0.0 \
      --eval_script ${EVAL_SCRIPT} \
      --log_path ${LOG_ROOT}/${variant}/devpick.log \
      --status_json ${DEVPICK_ROOT}/${variant}/free_budget/status.json \
      --reuse_existing
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${DEVPICK_ROOT}/${variant} ${LOG_ROOT}/${variant}/devpick.log
  "
}

eval_one() {
  local variant="$1"
  local budget="$2"
  local split="$3"
  local gpu="$4"
  local output_dir="${FORMAL_ROOT}/${variant}/forced_${budget}/${split}"
  local name="stage2_1_7b_${variant}_eventmentions_budget_forced_${budget}_${split}_20260524"
  if [[ -s "${output_dir}/predictions.jsonl" ]]; then
    echo "predictions already exist: ${output_dir}/predictions.jsonl"
    return 0
  fi
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  local adapter
  adapter="$(selected_adapter "${variant}")"
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${output_dir} ${LOG_ROOT}/${variant}
    echo adapter_path=${adapter}
    python ${EVAL_SCRIPT} \
      --base_model ${BASE_MODEL} \
      --adapter_path ${adapter} \
      --eval_jsonl $(data_prefix "${variant}")_forced_${budget}_${split}_pos.jsonl \
      --output_dir ${output_dir} \
      --batch_size 8 \
      --max_new_tokens 768 \
      --temperature 0.0 2>&1 | tee ${LOG_ROOT}/${variant}/forced_${budget}_${split}.log
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${output_dir} ${LOG_ROOT}/${variant}/forced_${budget}_${split}.log
  "
}

formal_one() {
  local variant="$1"
  local gpu0="$2"
  local gpu1="$3"
  local gpu2="$4"
  local gpu3="$5"
  if [[ "${variant}" == "e19a" ]]; then
    eval_one "${variant}" none test_seen "${gpu0}"
    eval_one "${variant}" standard test_seen "${gpu1}"
    eval_one "${variant}" none test_unseen "${gpu2}"
    eval_one "${variant}" standard test_unseen "${gpu3}"
  else
    eval_one "${variant}" standard test_seen "${gpu0}"
    eval_one "${variant}" standard test_unseen "${gpu1}"
  fi
}

case "${1:-}" in
  train) train_one "${2:-}" "${3:-0}" ;;
  devpick) devpick_one "${2:-}" "${3:-0}" ;;
  formal) formal_one "${2:-}" "${3:-0}" "${4:-1}" "${5:-2}" "${6:-3}" ;;
  *)
    echo "usage: $0 {train VARIANT [gpu]|devpick VARIANT [gpu]|formal VARIANT [gpu0 gpu1 gpu2 gpu3]}" >&2
    exit 2
    ;;
esac
