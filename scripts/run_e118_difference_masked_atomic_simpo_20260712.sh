#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
STATE_DIR="/mnt/disk/gaojun/tmp/gpu-label-service"

SOURCE_NAME="richere_balanced_split1_e115a_e81_atomic_counterfactual_orpo_smoke40_docdiverse_seed1150"
SOURCE_JSONL="/workspace/project/data/stage2_adaptive_datasets/${SOURCE_NAME}.jsonl"
SOURCE_SHA256="de1f034e027706e52885d915cf5440fe228469e157313fcb5640a27ddcc7785b"
START_MODEL="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full"
AUDIT_OUTPUT="/workspace/project/outputs/stage2_preference_mining/e118a_e81_difference_mask_audit_seed1180"
HOST_AUDIT_OUTPUT="${PROJECT_ROOT}/outputs/stage2_preference_mining/e118a_e81_difference_mask_audit_seed1180"
TRAIN_CONFIG="configs/generated/stage2_preference/e118b_e81_difference_masked_atomic_simpo_smoke40_seed42.yaml"
TRAIN_OUTPUT="/workspace/project/outputs/stage2_preference_runs/e118b_e81_difference_masked_atomic_simpo_smoke40_seed42"
HOST_TRAIN_OUTPUT="${PROJECT_ROOT}/outputs/stage2_preference_runs/e118b_e81_difference_masked_atomic_simpo_smoke40_seed42"

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
    -e PYTHONPATH=/workspace/project \
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

require_source() {
  [[ "$(sha256sum "${PROJECT_ROOT}/data/stage2_adaptive_datasets/${SOURCE_NAME}.jsonl" | cut -d' ' -f1)" == "${SOURCE_SHA256}" ]]
}

require_frozen_mask() {
  require_source
  jq -e '.frozen == true and .context_tokens == 1 and .test_data_access == false' \
    "${HOST_AUDIT_OUTPUT}/frozen_artifacts.json" >/dev/null
  local registered manifest actual
  registered="$(jq -r '.mask_manifest_sha256' "${HOST_AUDIT_OUTPUT}/frozen_artifacts.json")"
  manifest="${HOST_AUDIT_OUTPUT}/difference_mask_manifest.jsonl"
  actual="$(sha256sum "${manifest}" | cut -d' ' -f1)"
  [[ "${registered}" == "${actual}" ]]
}

run_gpu() {
  local gpu="$1"
  shift
  release_owned_label_service "${gpu}"
  docker_common --gpus "device=${gpu}" "${IMAGE}" "$@"
}

case "${1:-}" in
  audit)
    require_source
    docker_common "${IMAGE}" python scripts/audit_e118_difference_masking_20260712.py \
      --preference_jsonl "${SOURCE_JSONL}" --expected_sha256 "${SOURCE_SHA256}" \
      --model_path "${START_MODEL}" --output_dir "${AUDIT_OUTPUT}" \
      --context_tokens 1 --expected_pairs 40 --expected_per_category 8
    ;;
  score-pre)
    require_frozen_mask
    gpu="${2:-0}"
    run_gpu "${gpu}" bash -lc \
      "set -o pipefail; python scripts/score_e118_difference_masked_margins_20260712.py \
        --model_path '${START_MODEL}' --preference_jsonl '${SOURCE_JSONL}' \
        --mask_manifest '${AUDIT_OUTPUT}/difference_mask_manifest.jsonl' \
        --output_json '${AUDIT_OUTPUT}/pretrain_difference_margins.json' \
        --cutoff_len 1536 --context_tokens 1 2>&1 | tee '${AUDIT_OUTPUT}/pretrain_scoring.log'"
    ;;
  train-smoke)
    require_frozen_mask
    [[ -f "${HOST_AUDIT_OUTPUT}/pretrain_difference_margins.json" ]]
    if [[ -f "${HOST_TRAIN_OUTPUT}/trainer_state.json" ]]; then
      echo "E118B already ran; refusing to reuse its output directory" >&2
      exit 2
    fi
    gpu="${2:-0}"
    release_owned_label_service "${gpu}"
    docker_common --gpus "device=${gpu}" \
      -e E118_DIFF_CONTEXT_TOKENS=1 \
      -e PYTHONPATH=/workspace/project/scripts/compat/e118_difference_mask:/workspace/project \
      "${IMAGE}" bash -lc \
      "set -o pipefail; mkdir -p '${TRAIN_OUTPUT}'; FORCE_TORCHRUN=1 llamafactory-cli train '${TRAIN_CONFIG}' 2>&1 | tee '${TRAIN_OUTPUT}/train.log'"
    ;;
  score-post)
    require_frozen_mask
    [[ -f "${HOST_TRAIN_OUTPUT}/trainer_state.json" ]]
    gpu="${2:-0}"
    run_gpu "${gpu}" bash -lc \
      "set -o pipefail; python scripts/score_e118_difference_masked_margins_20260712.py \
        --model_path '${TRAIN_OUTPUT}' --preference_jsonl '${SOURCE_JSONL}' \
        --mask_manifest '${AUDIT_OUTPUT}/difference_mask_manifest.jsonl' \
        --output_json '${TRAIN_OUTPUT}/posttrain_difference_margins.json' \
        --cutoff_len 1536 --context_tokens 1 2>&1 | tee '${TRAIN_OUTPUT}/posttrain_scoring.log'"
    ;;
  gate)
    require_frozen_mask
    docker_common "${IMAGE}" python scripts/compare_e118_difference_masked_gate_20260712.py \
      --pre_score_json "${AUDIT_OUTPUT}/pretrain_difference_margins.json" \
      --post_score_json "${TRAIN_OUTPUT}/posttrain_difference_margins.json" \
      --trainer_state "${TRAIN_OUTPUT}/trainer_state.json" \
      --training_log "${TRAIN_OUTPUT}/train.log" \
      --model_dir "${TRAIN_OUTPUT}" \
      --output_json "${TRAIN_OUTPUT}/e118_difference_masked_gate.json" \
      --minimum_masked_delta 0.005 --maximum_chosen_full_logp_drop 0.02 \
      --expected_steps 5
    ;;
  *)
    echo "usage: $0 audit|score-pre [gpu]|train-smoke [gpu]|score-post [gpu]|gate" >&2
    exit 2
    ;;
esac
