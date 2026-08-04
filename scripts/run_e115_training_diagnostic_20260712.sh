#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
STATE_DIR="/mnt/disk/gaojun/tmp/gpu-label-service"

SOURCE_PREF="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_e114a_e81_atomic_counterfactual_orpo_seed1140.jsonl"
SOURCE_HASH="0b919c65b4e6f9484b6b20506508e5d4de831c10b162e980d60b8097f77ff8b4"
SAMPLES="/workspace/project/outputs/stage2_preference_mining/e110a_e81_k4_seed1104/samples.shard-*.jsonl"
START_MODEL="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full"
BUILD_OUTPUT="/workspace/project/outputs/stage2_preference_mining/e115a_e81_training_diagnostic_seed1150"
HOST_BUILD_OUTPUT="${PROJECT_ROOT}/outputs/stage2_preference_mining/e115a_e81_training_diagnostic_seed1150"
DATASET_DIR="/workspace/project/data/stage2_adaptive_datasets"
SMOKE_NAME="richere_balanced_split1_e115a_e81_atomic_counterfactual_orpo_smoke40_docdiverse_seed1150"
SMOKE_JSONL="${DATASET_DIR}/${SMOKE_NAME}.jsonl"
STYLE_JSONL="${BUILD_OUTPUT}/canonical_native_matched200.jsonl"
SMOKE_CONFIG="configs/generated/stage2_preference/e115b_e81_atomic_orpo_smoke40_fullpass_seed42.yaml"
SMOKE_MODEL="/workspace/project/outputs/stage2_preference_runs/e115b_e81_atomic_orpo_smoke40_fullpass_seed42"
HOST_SMOKE_MODEL="${PROJECT_ROOT}/outputs/stage2_preference_runs/e115b_e81_atomic_orpo_smoke40_fullpass_seed42"

docker_common() {
  docker run --rm --user root --ipc host --shm-size 16g \
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
    -w /workspace/project "$@"
}

release_owned_label_service() {
  local gpu="$1"
  local metadata="${STATE_DIR}/gpu${gpu}.json"
  [[ -f "${metadata}" ]] || return 0
  local owner metadata_gpu pid cmdline
  owner="$(jq -r '.owner // empty' "${metadata}")"
  metadata_gpu="$(jq -r '.gpu // .gpu_index // empty' "${metadata}")"
  pid="$(jq -r '.pid // empty' "${metadata}")"
  if [[ "${owner}" != "gaojun" || "${metadata_gpu}" != "${gpu}" || ! "${pid}" =~ ^[0-9]+$ ]]; then
    echo "refusing label-service release: invalid metadata ${metadata}" >&2
    return 2
  fi
  [[ -r "/proc/${pid}/cmdline" ]] || return 0
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  if [[ "${cmdline}" != *gpu-label-service* || "${cmdline}" != *gaojun* || "${cmdline}" != *"gpu${gpu}"* ]]; then
    echo "refusing label-service release: PID ${pid} does not match metadata" >&2
    return 2
  fi
  kill "${pid}"
  for _ in $(seq 1 20); do
    [[ -d "/proc/${pid}" ]] || return 0
    sleep 0.25
  done
  echo "label-service PID ${pid} did not exit" >&2
  return 2
}

require_frozen() {
  jq -e '.frozen == true and .selection_seed == 1150' \
    "${HOST_BUILD_OUTPUT}/frozen_artifacts.json" >/dev/null
  [[ "$(sha256sum "${PROJECT_ROOT}/data/stage2_adaptive_datasets/${SMOKE_NAME}.jsonl" | cut -d' ' -f1)" == \
    "$(jq -r '.artifacts.smoke40.sha256' "${HOST_BUILD_OUTPUT}/frozen_artifacts.json")" ]]
}

run_gpu() {
  local gpu="$1"
  shift
  release_owned_label_service "${gpu}"
  docker_common --gpus "device=${gpu}" "${IMAGE}" "$@"
}

case "${1:-}" in
  build)
    docker_common "${IMAGE}" python scripts/build_e115_training_diagnostics_20260712.py \
      --preference_jsonl "${SOURCE_PREF}" \
      --expected_preference_sha256 "${SOURCE_HASH}" \
      --samples_glob "${SAMPLES}" --output_dir "${BUILD_OUTPUT}" \
      --dataset_dir "${DATASET_DIR}" --dataset_info "${DATASET_DIR}/dataset_info.json" \
      --smoke_name "${SMOKE_NAME}" --smoke_per_category 8 \
      --style_per_category 40 --seed 1150
    ;;
  score-pre)
    require_frozen
    gpu="${2:-0}"
    run_gpu "${gpu}" bash -lc \
      "python scripts/score_e115_training_diagnostics_20260712.py \
        --model_path '${START_MODEL}' --preference_jsonl '${SMOKE_JSONL}' \
        --margin_output '${BUILD_OUTPUT}/pretrain_margin_smoke40.json' \
        --style_jsonl '${STYLE_JSONL}' \
        --style_output '${BUILD_OUTPUT}/pretrain_canonical_native_nll200.json' \
        --cutoff_len 1536 --beta 0.1 2>&1 | tee '${BUILD_OUTPUT}/pretrain_scoring.log'"
    ;;
  train-smoke)
    require_frozen
    [[ -f "${HOST_BUILD_OUTPUT}/pretrain_margin_smoke40.json" ]]
    [[ -f "${HOST_BUILD_OUTPUT}/pretrain_canonical_native_nll200.json" ]]
    if [[ -f "${HOST_SMOKE_MODEL}/trainer_state.json" ]]; then
      echo "E115B already ran; refusing to reuse its output directory" >&2
      exit 2
    fi
    gpu="${2:-0}"
    run_gpu "${gpu}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${SMOKE_CONFIG}' 2>&1 | tee '${BUILD_OUTPUT}/orpo_smoke40_train.log'"
    ;;
  score-post)
    require_frozen
    [[ -f "${HOST_SMOKE_MODEL}/trainer_state.json" ]]
    gpu="${2:-0}"
    run_gpu "${gpu}" bash -lc \
      "python scripts/score_e115_training_diagnostics_20260712.py \
        --model_path '${SMOKE_MODEL}' --preference_jsonl '${SMOKE_JSONL}' \
        --margin_output '${SMOKE_MODEL}/posttrain_margin_smoke40.json' \
        --cutoff_len 1536 --beta 0.1 2>&1 | tee '${SMOKE_MODEL}/posttrain_scoring.log'"
    ;;
  gate)
    require_frozen
    docker_common "${IMAGE}" python scripts/compare_e115_margin_gate_20260712.py \
      --pre_margin_json "${BUILD_OUTPUT}/pretrain_margin_smoke40.json" \
      --post_margin_json "${SMOKE_MODEL}/posttrain_margin_smoke40.json" \
      --style_json "${BUILD_OUTPUT}/pretrain_canonical_native_nll200.json" \
      --trainer_state "${SMOKE_MODEL}/trainer_state.json" \
      --model_dir "${SMOKE_MODEL}" \
      --output_json "${SMOKE_MODEL}/e115_training_gate.json" \
      --minimum_improved_categories 4 \
      --required_improved_categories extra_frame trigger_drift \
      --maximum_nll_gap 0.3 --expected_steps 5
    ;;
  *)
    echo "usage: $0 build|score-pre [gpu]|train-smoke [gpu]|score-post [gpu]|gate" >&2
    exit 2
    ;;
esac
