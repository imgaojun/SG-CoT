#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
LOG_ROOT="outputs/stage2_1_7b_paired_augmentation/e27_logs_20260527"
DEVPICK_ROOT="outputs/stage2_1_7b_paired_augmentation/e27_devpick_20260527"
FORMAL_ROOT="outputs/stage2_1_7b_paired_augmentation/e27_formal_20260527"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation.py"
SHORTLIST_SCRIPT="src/stage2_formal/parallel_shortlist_dev_select.py"

branch_for() {
  case "$1" in
    e27a) echo "eventmentions_budget_e27a_none_aug" ;;
    e27b) echo "eventmentions_budget_e27b_span_reason_aug" ;;
    e27c) echo "eventmentions_budget_e27c_paired_none_standard_aug" ;;
    e27d) echo "eventmentions_budget_e27d_balanced_none_aug" ;;
    e27e) echo "eventmentions_budget_e27e_hardneg_none_aug" ;;
    e28a) echo "eventmentions_budget_e28a_balanced_natural_step_reason" ;;
    e29a) echo "eventmentions_budget_e29a_hardneg_natural_step_reason" ;;
    e29b) echo "eventmentions_budget_e29b_balanced_compact_step_reason" ;;
    e29c) echo "eventmentions_budget_e29c_hardneg_compact_step_reason" ;;
    e30a) echo "eventmentions_budget_e30a_tail_type_balanced_none_aug" ;;
    e30b) echo "eventmentions_budget_e30b_tail_type_balanced_natural_step" ;;
    e30c) echo "eventmentions_budget_e30c_tail_type_balanced_minimal_type_step" ;;
    e31a) echo "eventmentions_budget_e31a_type_complexity_none_aug" ;;
    e31b) echo "eventmentions_budget_e31b_type_complexity_natural_step" ;;
    e32a) echo "eventmentions_budget_e32a_trigger_preserving_tail_natural_step" ;;
    e32b) echo "eventmentions_budget_e32b_trigger_role_ground_natural_step" ;;
    e32c) echo "eventmentions_budget_e32c_trigger_role_ground_direct_anchor" ;;
    e35a) echo "eventmentions_budget_e35a_boundary_contrast_direct" ;;
    e35b) echo "eventmentions_budget_e35b_boundary_check_reason" ;;
    e35c) echo "eventmentions_budget_e35c_boundary_check_direct_anchor" ;;
    *) echo "unknown variant: $1" >&2; exit 2 ;;
  esac
}

dev_budget_for() {
  case "$1" in
    e27a|e27d|e27e|e30a|e31a|e35a) echo "none" ;;
    e27b|e27c|e28a|e29a|e29b|e29c|e30b|e30c|e31b|e32a|e32b|e32c|e35b|e35c) echo "standard" ;;
    *) echo "unknown variant: $1" >&2; exit 2 ;;
  esac
}

budgets_for() {
  case "$1" in
    e27a|e27b|e27c|e27d|e27e|e28a|e29a|e29b|e29c|e30a|e30b|e30c|e31a|e31b|e32a|e32b|e32c|e35a|e35b|e35c) echo "none standard" ;;
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
  local budget
  budget="$(dev_budget_for "${variant}")"
  local summary="${DEVPICK_ROOT}/${variant}/forced_${budget}_dev/selection_summary.json"
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
  local name="stage2_1_7b_${variant}_paired_aug_train_20260527"
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
  local budget
  budget="$(dev_budget_for "${variant}")"
  local name="stage2_1_7b_${variant}_paired_aug_devpick_20260527"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${LOG_ROOT}/${variant} ${DEVPICK_ROOT}/${variant}/forced_${budget}_dev
    ckpts=\$(find $(run_dir "${variant}") -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
    python ${SHORTLIST_SCRIPT} \
      --base_model ${BASE_MODEL} \
      --run_dir $(run_dir "${variant}") \
      --eval_jsonl $(data_prefix "${variant}")_forced_${budget}_dev_seen_pos.jsonl \
      --output_root ${DEVPICK_ROOT}/${variant}/forced_${budget}_dev \
      --checkpoint_tags \${ckpts} \
      --gpu_ids 0 \
      --metric_keys event_f1 argument_f1 trigger_f1 \
      --greater_is_better \
      --batch_size 1 \
      --max_new_tokens 768 \
      --temperature 0.0 \
      --eval_script ${EVAL_SCRIPT} \
      --log_path ${LOG_ROOT}/${variant}/devpick.log \
      --status_json ${DEVPICK_ROOT}/${variant}/forced_${budget}_dev/status.json \
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
  local name="stage2_1_7b_${variant}_paired_aug_forced_${budget}_${split}_20260527"
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
  shift
  local gpus=("$@")
  if [[ "${#gpus[@]}" -eq 0 ]]; then
    gpus=(1 2 3 4 7)
  fi
  local i=0
  for budget in $(budgets_for "${variant}"); do
    for split in test_seen test_unseen; do
      eval_one "${variant}" "${budget}" "${split}" "${gpus[$((i % ${#gpus[@]}))]}"
      i=$((i + 1))
    done
  done
}

case "${1:-}" in
  train) train_one "${2:-}" "${3:-1}" ;;
  devpick) devpick_one "${2:-}" "${3:-1}" ;;
  formal) shift; variant="${1:-}"; shift || true; formal_one "${variant}" "$@" ;;
  *)
    echo "usage: $0 {train VARIANT [gpu]|devpick VARIANT [gpu]|formal VARIANT [gpu...]}" >&2
    exit 2
    ;;
esac
