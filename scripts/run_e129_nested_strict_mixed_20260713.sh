#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
MODEL_ROOT="/mnt/disk/gaojun/models"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
IMAGE="llamafactory-lab:0.9.4-py3.12"
CONFIG_DIR="configs/generated/stage2_development"
DATA_DIR="data/stage2_development_e129"
PROCESSED_ROOT="data/processed/type_holdout"
PROTOCOL="e129-nested-jtrial-v1"
SPLIT="split1"
PREFIX="e129_jtrial"
OUTPUT_ROOT="outputs/stage2_development_e129"
BACKUP_ROOT="/mnt/manhattan/manhadun_backup/gaojun/progressive-ee/checkpoints"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
HELDOUT="${PROCESSED_ROOT}/richere-en/${PROTOCOL}/${SPLIT}/unseen_types.json"
cd "${PROJECT_ROOT}"

docker_common() {
  docker run --rm --user root --ipc host --shm-size 16g \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -v "${LF_ROOT}/cache/torch_extensions:/workspace/.cache/torch_extensions" \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    -w /workspace/project "$@"
}

assert_disk_headroom() {
  local minimum_gib="${1:-500}"
  local available_kib
  available_kib="$(df -Pk /mnt/disk | awk 'NR==2 {print $4}')"
  if (( available_kib < minimum_gib * 1024 * 1024 )); then
    echo "insufficient /mnt/disk headroom: require ${minimum_gib} GiB" >&2
    return 10
  fi
  [[ -d /mnt/manhattan/manhadun_backup/gaojun && -w /mnt/manhattan/manhadun_backup/gaojun ]] || {
    echo "manhadun backup destination is not writable" >&2
    return 11
  }
}

release_owned_label_service() {
  local gpu="$1" state_dir="/mnt/disk/gaojun/tmp/gpu-label-service"
  local metadata="${state_dir}/gpu${gpu}.json" pid_file="${state_dir}/gpu${gpu}.pid"
  [[ -f "${metadata}" && -f "${pid_file}" ]] || return 0
  local owner pid recorded_pid cmdline
  owner="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("owner", ""))' "${metadata}")"
  pid="$(tr -d '[:space:]' < "${pid_file}")"
  recorded_pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid", ""))' "${metadata}")"
  [[ "${owner}" == "gaojun" && "${pid}" == "${recorded_pid}" && "${pid}" =~ ^[0-9]+$ ]] || return 0
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "${cmdline}" == *gpu-label-service* && "${cmdline}" == *"gpu${gpu}"* ]]; then
    docker rm -f "gpu-label-service-gaojun-gpu${gpu}" >/dev/null 2>&1 || kill "${pid}"
  fi
}

assert_gpu_idle() {
  local gpu="$1" used util processes
  release_owned_label_service "${gpu}"
  IFS=',' read -r used util < <(
    nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' '
  )
  processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "${processes}" || "${used}" -gt 1024 || "${util}" -gt 5 ]]; then
    echo "GPU ${gpu} is not idle: memory=${used} MiB utilization=${util}% processes=${processes:-none}" >&2
    return 12
  fi
}

assert_gpu_set_idle() {
  local gpu_spec="$1" expected_count="$2"
  local -a gpu_ids
  local gpu
  declare -A seen=()
  IFS=',' read -r -a gpu_ids <<< "${gpu_spec}"
  if (( ${#gpu_ids[@]} != expected_count )); then
    echo "expected ${expected_count} GPUs, got ${#gpu_ids[@]} in ${gpu_spec}" >&2
    return 14
  fi
  for gpu in "${gpu_ids[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ && -z "${seen[${gpu}]:-}" ]] || {
      echo "invalid or duplicate GPU set: ${gpu_spec}" >&2
      return 14
    }
    seen["${gpu}"]=1
    assert_gpu_idle "${gpu}"
  done
}

assert_data_ready() {
  python3 scripts/build_e129_nested_strict_development.py audit >/dev/null
}

latest_recoverable_checkpoint() {
  local output="$1"
  local checkpoint
  checkpoint="$(find "${output}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -1)"
  [[ "${checkpoint}" =~ ^checkpoint-[0-9]+$ ]] || {
    echo "no recoverable checkpoint under ${output}" >&2
    return 15
  }
  echo "${output}/${checkpoint}"
}

assert_resume_ready() {
  local method="$1" output checkpoint
  output="$(run_dir "${method}")"
  [[ -d "${output}" ]] || {
    echo "missing interrupted run directory: ${output}" >&2
    return 15
  }
  [[ ! -e "${output}/config.json" ]] || {
    echo "refusing to resume already-completed run: ${output}" >&2
    return 15
  }
  checkpoint="$(latest_recoverable_checkpoint "${output}")"
  python3 scripts/validate_training_artifact_20260712.py \
    --model_dir "${checkpoint}" \
    --trainer_state "${checkpoint}/trainer_state.json" \
    --require_finite_step_log >/dev/null
  python3 - "${checkpoint}/trainer_state.json" <<'PY'
import json, sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
step = int(state.get("global_step", 0))
max_steps = int(state.get("max_steps", 0))
if not (0 < step < max_steps):
    raise SystemExit(f"checkpoint is not an interrupted finite run: step={step}, max_steps={max_steps}")
print(f"resume checkpoint verified: step={step}/{max_steps}")
PY
}

run_dir() {
  case "$1" in
    direct) echo "${OUTPUT_ROOT}/runs/e129a_jtrial_direct_seed42" ;;
    pure) echo "${OUTPUT_ROOT}/runs/e129b_jtrial_pure_sgcot_seed42" ;;
    mixed) echo "${OUTPUT_ROOT}/runs/e129c_jtrial_dualmode_mixed_seed42" ;;
    *) return 2 ;;
  esac
}

config_path() {
  case "$1" in
    direct) echo "${CONFIG_DIR}/e129a_jtrial_direct_seed42.yaml" ;;
    pure) echo "${CONFIG_DIR}/e129b_jtrial_pure_sgcot_seed42.yaml" ;;
    mixed) echo "${CONFIG_DIR}/e129c_jtrial_dualmode_mixed_seed42.yaml" ;;
    *) return 2 ;;
  esac
}

selection_path() {
  case "$1" in
    pure|mixed) echo "${OUTPUT_ROOT}/selection/$1_seen_loss.json" ;;
    *) return 2 ;;
  esac
}

selected_variant() {
  local method="$1" selection
  selection="$(selection_path "${method}")"
  python3 - "${selection}" <<'PY'
import json, sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["id"] == "e129_seen_loss_checkpoint_selection_v1"
assert value["selection_split"] == "dev_seen"
print(value["selected_checkpoint"])
PY
}

assert_training_config_compatible() {
  local config="$1"
  python3 - "${config}" <<'PY'
import sys

import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
if config.get("deepspeed") and config.get("save_only_model") and config.get("load_best_model_at_end"):
    raise SystemExit(
        "DeepSpeed is incompatible with save_only_model=true and load_best_model_at_end=true"
    )
PY
}

build_part() {
  local part="$1"
  local scope="all"
  local numeric="${PREFIX}_numeric_${part}"
  [[ "${part}" == "train" ]] && scope="seen_only"
  python3 src/stage2_data/build_formal_stage2_dataset.py \
    --data_root "${PROCESSED_ROOT}" --dataset richere-en --protocol "${PROTOCOL}" \
    --split "${SPLIT}" --part "${part}" --schema_path data/schema/richere-en.event_schema.json \
    --candidate_source oracle_mixed_noise --candidate_scope "${scope}" --top_k 10 \
    --candidate_order_mode shuffle --selection_mode positive_only --seed 13 \
    --dataset_dir "${DATA_DIR}" --dataset_name "${numeric}"
  for mode in direct sgcot; do
    local name="${PREFIX}_${mode}_${part}" extra=()
    [[ "${part}" == "train" ]] && extra+=(--require_no_heldout_leak)
    python3 scripts/build_surface_evidence_dataset_20260712.py \
      --input_jsonl "${DATA_DIR}/${numeric}.jsonl" \
      --output_jsonl "${DATA_DIR}/${name}.jsonl" --mode "${mode}" \
      --dataset_name "${name}" --dataset_info "${DATA_DIR}/dataset_info.json" \
      --heldout_types_json "${HELDOUT}" "${extra[@]}"
  done
}

case "${ACTION}" in
  build-data)
    protocol_stats="${PROCESSED_ROOT}/richere-en/${PROTOCOL}/${SPLIT}/stats.json"
    if [[ ! -e "${PROCESSED_ROOT}/richere-en/${PROTOCOL}" ]]; then
      python3 scripts/build_e129_nested_strict_development.py protocol
    else
      python3 - "${protocol_stats}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
assert value["id"] == "e129a_nested_jtrial_protocol_v1"
assert value["train_dev_unseen_overlap"] == 0
assert value["pseudo_unseen_rows"] >= 75
PY
    fi
    if [[ -e "${DATA_DIR}/e129a_dataset_audit.json" ]]; then
      echo "refusing to overwrite completed E129 data assets" >&2
      exit 13
    fi
    if [[ -e "${DATA_DIR}/e129_jtrial_dualmode_mixed_train.jsonl" ]]; then
      python3 scripts/build_e129_nested_strict_development.py audit
      exit 0
    fi
    mkdir -p "${DATA_DIR}"
    build_part train
    build_part dev_seen
    build_part dev_unseen
    python3 scripts/build_e129_nested_strict_development.py filter-traces
    python3 scripts/build_e129_nested_strict_development.py mix
    python3 scripts/build_e129_nested_strict_development.py audit
    ;;
  train)
    method="${2:?direct, pure, or mixed required}" gpus="${3:?comma-separated GPUs required}"
    config="$(config_path "${method}")"
    expected_gpus=2
    [[ "${method}" == "direct" ]] && expected_gpus=4
    assert_training_config_compatible "${config}"
    assert_disk_headroom 500
    assert_data_ready
    assert_gpu_set_idle "${gpus}" "${expected_gpus}"
    output="$(run_dir "${method}")"
    [[ ! -e "${output}" ]] || { echo "refusing to reuse ${output}" >&2; exit 13; }
    if [[ "${method}" != "direct" ]]; then
      python3 scripts/validate_training_artifact_20260712.py \
        --model_dir "$(run_dir direct)" >/dev/null
    fi
    mkdir -p "${OUTPUT_ROOT}/logs"
    gpu_request="\"device=${gpus}\""
    docker_common --gpus "${gpu_request}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${config}' 2>&1 | tee '${OUTPUT_ROOT}/logs/train_${method}_seed42.log'"
    python3 scripts/validate_training_artifact_20260712.py --model_dir "${output}"
    ;;
  resume)
    method="${2:?direct, pure, or mixed required}" gpus="${3:?comma-separated GPUs required}"
    config="$(config_path "${method}")"
    expected_gpus=2
    [[ "${method}" == "direct" ]] && expected_gpus=4
    assert_training_config_compatible "${config}"
    assert_disk_headroom 500
    assert_data_ready
    assert_resume_ready "${method}"
    assert_gpu_set_idle "${gpus}" "${expected_gpus}"
    mkdir -p "${OUTPUT_ROOT}/logs"
    gpu_request="\"device=${gpus}\""
    docker_common --gpus "${gpu_request}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${config}' 2>&1 | tee -a '${OUTPUT_ROOT}/logs/train_${method}_seed42.log'"
    python3 scripts/validate_training_artifact_20260712.py --model_dir "$(run_dir "${method}")"
    ;;
  select)
    method="${2:?pure or mixed required}"
    [[ "${method}" == "pure" || "${method}" == "mixed" ]] || exit 2
    python3 scripts/validate_training_artifact_20260712.py --model_dir "$(run_dir "${method}")" >/dev/null
    python3 scripts/select_e129_seen_loss_checkpoint.py \
      --run_dir "$(run_dir "${method}")" \
      --output_json "$(selection_path "${method}")" \
      --expected_candidates 3
    ;;
  eval)
    method="${2:?direct, pure, or mixed required}" variant="${3:?root or checkpoint name required}"
    split="${4:?seen or unseen required}" gpu="${5:?GPU required}"
    assert_data_ready
    assert_gpu_idle "${gpu}"
    if [[ "${variant}" == "selected" ]]; then
      variant="$(selected_variant "${method}")"
    fi
    model="$(run_dir "${method}")"
    [[ "${variant}" == "root" ]] || model="${model}/${variant}"
    python3 scripts/validate_training_artifact_20260712.py --model_dir "${model}" >/dev/null
    eval_mode="direct"
    [[ "${method}" == "pure" ]] && eval_mode="sgcot"
    data="/workspace/project/${DATA_DIR}/e129_jtrial_${eval_mode}_dev_${split}.jsonl"
    output="${OUTPUT_ROOT}/eval/${method}_${variant}/dev_${split}"
    [[ ! -e "${output}" ]] || { echo "refusing to reuse ${output}" >&2; exit 13; }
    max_tokens=512
    [[ "${method}" == "pure" ]] && max_tokens=1024
    docker_common --gpus "device=${gpu}" "${IMAGE}" python \
      src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
      --base_model "${BASE_MODEL}" --adapter_path "/workspace/project/${model}" \
      --eval_jsonl "${data}" --output_dir "/workspace/project/${output}" \
      --batch_size 4 --max_new_tokens "${max_tokens}" --temperature 0
    ;;
  compare)
    pure_variant="${2:?pure checkpoint variant required}"
    mixed_variant="${3:?mixed checkpoint variant required}"
    [[ "${pure_variant}" == "selected" ]] && pure_variant="$(selected_variant pure)"
    [[ "${mixed_variant}" == "selected" ]] && mixed_variant="$(selected_variant mixed)"
    python3 -m scripts.compare_e129_nested_development \
      --direct_seen "${OUTPUT_ROOT}/eval/direct_root/dev_seen" \
      --direct_unseen "${OUTPUT_ROOT}/eval/direct_root/dev_unseen" \
      --pure_seen "${OUTPUT_ROOT}/eval/pure_${pure_variant}/dev_seen" \
      --pure_unseen "${OUTPUT_ROOT}/eval/pure_${pure_variant}/dev_unseen" \
      --mixed_seen "${OUTPUT_ROOT}/eval/mixed_${mixed_variant}/dev_seen" \
      --mixed_unseen "${OUTPUT_ROOT}/eval/mixed_${mixed_variant}/dev_unseen" \
      --output_dir "${OUTPUT_ROOT}/analysis/compare_pure_${pure_variant}_mixed_${mixed_variant}"
    ;;
  archive)
    method="${2:?direct, pure, or mixed required}"
    ARCHIVE_DELETE_LOCAL="${ARCHIVE_DELETE_LOCAL:-0}" \
      bash scripts/archive_training_checkpoints_20260713.sh "$(run_dir "${method}")" "${BACKUP_ROOT}"
    ;;
  *)
    echo "usage: $0 build-data|train <direct|pure|mixed> <comma-separated-gpus>|resume <direct|pure|mixed> <comma-separated-gpus>|select <pure|mixed>|eval <method> <root|selected|checkpoint-N> <seen|unseen> <gpu>|compare <pure-variant> <mixed-variant>|archive <method>" >&2
    exit 2
    ;;
esac
