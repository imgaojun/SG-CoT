#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
RUN_PREFIX="richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH="confrare10_typeonlylite_reasonfmt_e18b_latentreason_eventpos_sumpos_r5"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
LOG_ROOT="outputs/stage2_4b_event_preserving_reason/e18b_logs_20260523"
DEVPICK_ROOT="outputs/stage2_4b_event_preserving_reason/e18b_devpick_20260523"
FORMAL_ROOT="outputs/stage2_4b_event_preserving_reason/e18b_formal_20260523"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation.py"
SHORTLIST_SCRIPT="src/stage2_formal/parallel_shortlist_dev_select.py"

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
  echo "outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full"
}

config_path() {
  echo "configs/generated/stage2_adaptive/${RUN_PREFIX}_${BRANCH}_full_stepmatch.yaml"
}

data_prefix() {
  echo "${DATA_BASE}_${BRANCH}"
}

selected_adapter() {
  local summary="${DEVPICK_ROOT}/free_route/selection_summary.json"
  local fallback
  fallback="$(run_dir)"
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
  local gpu="$1"
  local name="stage2_4b_e18b_latentreason_eventpos_r5_train_20260523"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${LOG_ROOT}
    FORCE_TORCHRUN=1 llamafactory-cli train $(config_path) 2>&1 | tee ${LOG_ROOT}/train.log
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} $(run_dir) ${LOG_ROOT}/train.log
  "
}

devpick_one() {
  local gpu="$1"
  local name="stage2_4b_e18b_latentreason_eventpos_r5_devpick_20260523"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${LOG_ROOT} ${DEVPICK_ROOT}/free_route
    ckpts=\$(find $(run_dir) -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
    python ${SHORTLIST_SCRIPT} \
      --base_model ${BASE_MODEL} \
      --run_dir $(run_dir) \
      --eval_jsonl $(data_prefix)_dev_seen_pos.jsonl \
      --output_root ${DEVPICK_ROOT}/free_route \
      --checkpoint_tags \${ckpts} \
      --gpu_ids 0 \
      --metric_keys event_f1 argument_f1 trigger_f1 \
      --greater_is_better \
      --batch_size 1 \
      --max_new_tokens 512 \
      --temperature 0.0 \
      --eval_script ${EVAL_SCRIPT} \
      --log_path ${LOG_ROOT}/devpick.log \
      --status_json ${DEVPICK_ROOT}/free_route/status.json \
      --reuse_existing
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${DEVPICK_ROOT} ${LOG_ROOT}/devpick.log
  "
}

eval_one() {
  local mode="$1"
  local split="$2"
  local gpu="$3"
  local output_dir="${FORMAL_ROOT}/${mode}/${split}"
  local name="stage2_4b_e18b_latentreason_eventpos_r5_${mode}_${split}_20260523"
  if [[ -s "${output_dir}/predictions.jsonl" ]]; then
    echo "predictions already exist: ${output_dir}/predictions.jsonl"
    return 0
  fi
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  local adapter
  adapter="$(selected_adapter)"
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${output_dir} ${LOG_ROOT}
    echo adapter_path=${adapter}
    python ${EVAL_SCRIPT} \
      --base_model ${BASE_MODEL} \
      --adapter_path ${adapter} \
      --eval_jsonl $(data_prefix)_${mode}_${split}_pos.jsonl \
      --output_dir ${output_dir} \
      --batch_size 8 \
      --max_new_tokens 512 \
      --temperature 0.0 2>&1 | tee ${LOG_ROOT}/${mode}_${split}.log
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${output_dir} ${LOG_ROOT}/${mode}_${split}.log
  "
}

case "${1:-}" in
  train) train_one "${2:-1}" ;;
  devpick) devpick_one "${2:-1}" ;;
  formal)
    eval_one forced_direct test_seen "${2:-1}"
    eval_one forced_reason test_seen "${3:-2}"
    eval_one forced_direct test_unseen "${4:-3}"
    eval_one forced_reason test_unseen "${5:-4}"
    ;;
  *)
    echo "usage: $0 {train [gpu]|devpick [gpu]|formal [gpu0 gpu1 gpu2 gpu3]}" >&2
    exit 2
    ;;
esac
