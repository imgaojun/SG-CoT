#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
BRANCH="confrare10_typeonlylite_directwarm_retention_e13b"
RUN_KEY="richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_${BRANCH}"
RUN_DIR="outputs/stage2_adaptive_runs_user/${RUN_KEY}_full"
CONFIG="configs/generated/stage2_adaptive/${RUN_KEY}_full_stepmatch.yaml"
DATA_PREFIX="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${BRANCH}"
FORMAL_ROOT="outputs/stage2_4b_reason_expert/e13b_formal_20260521"
DEVPICK_ROOT="outputs/stage2_4b_reason_expert/e13b_devpick_20260521"
LOG_ROOT="outputs/stage2_4b_reason_expert/logs_e13b_20260521"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation.py"
SHORTLIST_SCRIPT="src/stage2_formal/parallel_shortlist_dev_select.py"

selected_adapter() {
  local summary="${DEVPICK_ROOT}/free_route/selection_summary.json"
  if [[ -s "${summary}" ]]; then
    python3 - "$summary" "${RUN_DIR}" <<'PY'
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
    echo "${RUN_DIR}"
  fi
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

train() {
  local gpu="$1"
  local name="stage2_4b_reason_e13b_train_20260521"
  local log="${LOG_ROOT}/train.log"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${LOG_ROOT}
    FORCE_TORCHRUN=1 llamafactory-cli train ${CONFIG} 2>&1 | tee ${log}
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${RUN_DIR} ${log}
  "
}

devpick() {
  local gpu="$1"
  local name="stage2_4b_reason_e13b_devpick_20260521"
  local log="${LOG_ROOT}/devpick.log"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${LOG_ROOT} ${DEVPICK_ROOT}
    ckpts=\$(find ${RUN_DIR} -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
    python ${SHORTLIST_SCRIPT} \
      --base_model ${BASE_MODEL} \
      --run_dir ${RUN_DIR} \
      --eval_jsonl ${DATA_PREFIX}_dev_seen_pos.jsonl \
      --output_root ${DEVPICK_ROOT}/free_route \
      --checkpoint_tags \${ckpts} \
      --gpu_ids 0 \
      --metric_keys argument_f1 event_f1 trigger_f1 \
      --greater_is_better \
      --batch_size 1 \
      --max_new_tokens 512 \
      --temperature 0.0 \
      --eval_script ${EVAL_SCRIPT} \
      --log_path ${log} \
      --status_json ${DEVPICK_ROOT}/free_route/status.json \
      --reuse_existing
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${DEVPICK_ROOT} ${log}
  "
}

eval_one() {
  local mode="$1"
  local split="$2"
  local gpu="$3"
  local adapter
  adapter="$(selected_adapter)"
  local eval_jsonl="${DATA_PREFIX}_${mode}_${split}_pos.jsonl"
  local output_dir="${FORMAL_ROOT}/${mode}/${split}"
  local name="stage2_4b_reason_e13b_${mode}_${split}_20260521"
  local log="${LOG_ROOT}/${mode}_${split}.log"
  if [[ ! -s "${eval_jsonl}" ]]; then
    echo "missing eval jsonl: ${eval_jsonl}" >&2
    return 1
  fi
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

formal() {
  eval_one forced_direct test_seen "${1:-0}"
  eval_one forced_reason test_seen "${2:-1}"
  eval_one forced_direct test_unseen "${3:-2}"
  eval_one forced_reason test_unseen "${4:-3}"
}

case "${1:-}" in
  train) train "${2:-0}" ;;
  devpick) devpick "${2:-0}" ;;
  formal) formal "${2:-0}" "${3:-1}" "${4:-2}" "${5:-3}" ;;
  all)
    train "${2:-0}"
    ;;
  *)
    echo "usage: $0 {train|devpick|formal|all} [gpu...]" >&2
    exit 2
    ;;
esac
