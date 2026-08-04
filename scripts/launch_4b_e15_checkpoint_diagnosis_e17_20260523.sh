#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
RUN_PREFIX="richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
LOG_ROOT="outputs/stage2_4b_reason_checkpoint_diagnosis/e17_logs_20260523"
FORMAL_ROOT="outputs/stage2_4b_reason_checkpoint_diagnosis/e17_formal_20260523"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation.py"

branch_for() {
  case "$1" in
    e15a_noreasonblock) echo "confrare10_typeonlylite_reasonfmt_e15a_noreasonblock" ;;
    e15c_finalfirst) echo "confrare10_typeonlylite_reasonfmt_e15c_finalfirst" ;;
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

data_prefix_for() {
  local branch="$1"
  echo "${DATA_BASE}_${branch}"
}

eval_one_inline() {
  cat <<'SH'
eval_one() {
  local variant="$1"
  local ckpt="$2"
  local mode="$3"
  local split="$4"
  local branch run_dir adapter data_prefix eval_jsonl output_dir log
  branch="$(branch_for "${variant}")"
  run_dir="$(run_dir_for "${branch}")"
  adapter="${run_dir}/checkpoint-${ckpt}"
  data_prefix="$(data_prefix_for "${branch}")"
  eval_jsonl="${data_prefix}_${mode}_${split}_pos.jsonl"
  output_dir="${FORMAL_ROOT}/${variant}/checkpoint-${ckpt}/${mode}/${split}"
  log="${LOG_ROOT}/${variant}_checkpoint-${ckpt}_${mode}_${split}.log"
  if [[ -s "${output_dir}/predictions.jsonl" ]]; then
    echo "[skip] ${output_dir}/predictions.jsonl exists"
    return 0
  fi
  mkdir -p "${output_dir}" "${LOG_ROOT}"
  echo "[run] variant=${variant} checkpoint=${ckpt} mode=${mode} split=${split} adapter=${adapter}"
  python "${EVAL_SCRIPT}" \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${adapter}" \
    --eval_jsonl "${eval_jsonl}" \
    --output_dir "${output_dir}" \
    --batch_size 8 \
    --max_new_tokens 512 \
    --temperature 0.0 2>&1 | tee "${log}"
}
SH
}

queue_one() {
  local queue_id="$1"
  local gpu="$2"
  shift 2
  local name="stage2_4b_e17_queue${queue_id}_formal_20260523"
  local queue_log="${LOG_ROOT}/queue${queue_id}.log"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists: ${name}" >&2
    return 0
  fi
  local task_payload="$*"
  docker_common --name "${name}" --gpus "\"device=${gpu}\"" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p ${LOG_ROOT}
    $(declare -f branch_for)
    $(declare -f run_dir_for)
    $(declare -f data_prefix_for)
    RUN_PREFIX='${RUN_PREFIX}'
    DATA_BASE='${DATA_BASE}'
    BASE_MODEL='${BASE_MODEL}'
    FORMAL_ROOT='${FORMAL_ROOT}'
    LOG_ROOT='${LOG_ROOT}'
    EVAL_SCRIPT='${EVAL_SCRIPT}'
    $(eval_one_inline)
    echo '${task_payload}' | tr ' ' '\n' | while IFS=',' read -r variant ckpt mode split; do
      [[ -z \"\${variant}\" ]] && continue
      eval_one \"\${variant}\" \"\${ckpt}\" \"\${mode}\" \"\${split}\"
    done 2>&1 | tee ${queue_log}
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${FORMAL_ROOT} ${LOG_ROOT}
  "
}

case "${1:-}" in
  queue)
    queue_one "${2:?queue_id}" "${3:?gpu}" "${@:4}"
    ;;
  queue-all)
    queue_one 0 "${2:-0}" \
      e15a_noreasonblock,386,forced_direct,test_seen \
      e15a_noreasonblock,386,forced_reason,test_seen \
      e15a_noreasonblock,386,forced_direct,test_unseen \
      e15a_noreasonblock,386,forced_reason,test_unseen \
      e15a_noreasonblock,1158,forced_direct,test_seen \
      e15a_noreasonblock,1158,forced_reason,test_seen \
      e15a_noreasonblock,1158,forced_direct,test_unseen \
      e15a_noreasonblock,1158,forced_reason,test_unseen \
      e15c_finalfirst,386,forced_direct,test_seen \
      e15c_finalfirst,386,forced_reason,test_seen \
      e15c_finalfirst,386,forced_direct,test_unseen \
      e15c_finalfirst,386,forced_reason,test_unseen \
      e15c_finalfirst,1158,forced_direct,test_seen \
      e15c_finalfirst,1158,forced_reason,test_seen \
      e15c_finalfirst,1158,forced_direct,test_unseen \
      e15c_finalfirst,1158,forced_reason,test_unseen
    queue_one 1 "${3:-2}" \
      e15a_noreasonblock,772,forced_direct,test_seen \
      e15a_noreasonblock,772,forced_reason,test_seen \
      e15a_noreasonblock,772,forced_direct,test_unseen \
      e15a_noreasonblock,772,forced_reason,test_unseen \
      e15a_noreasonblock,1544,forced_direct,test_seen \
      e15a_noreasonblock,1544,forced_reason,test_seen \
      e15a_noreasonblock,1544,forced_direct,test_unseen \
      e15a_noreasonblock,1544,forced_reason,test_unseen \
      e15c_finalfirst,772,forced_direct,test_seen \
      e15c_finalfirst,772,forced_reason,test_seen \
      e15c_finalfirst,772,forced_direct,test_unseen \
      e15c_finalfirst,772,forced_reason,test_unseen \
      e15c_finalfirst,1544,forced_direct,test_seen \
      e15c_finalfirst,1544,forced_reason,test_seen \
      e15c_finalfirst,1544,forced_direct,test_unseen \
      e15c_finalfirst,1544,forced_reason,test_unseen
    ;;
  *)
    echo "usage: $0 {queue <id> <gpu> task...|queue-all [gpu0 gpu1]}" >&2
    echo "task format: variant,ckpt,mode,split" >&2
    exit 2
    ;;
esac
