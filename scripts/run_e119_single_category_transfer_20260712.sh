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
DATA_OUTPUT="/workspace/project/outputs/stage2_preference_mining/e119a_e81_single_category_transfer_diagnostic"
HOST_DATA_OUTPUT="${PROJECT_ROOT}/outputs/stage2_preference_mining/e119a_e81_single_category_transfer_diagnostic"
MASK_MANIFEST="/workspace/project/outputs/stage2_preference_mining/e118a_e81_difference_mask_audit_seed1180/difference_mask_manifest.jsonl"
PRE_SCORE="/workspace/project/outputs/stage2_preference_mining/e118a_e81_difference_mask_audit_seed1180/pretrain_difference_margins.json"

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
  return 2
}

experiment_id() {
  case "$1" in
    extra_frame) echo "e119b1_e81_extra_frame_only_masked_simpo_seed42" ;;
    wrong_type) echo "e119b2_e81_wrong_type_only_masked_simpo_seed42" ;;
    argument_omission) echo "e119b3_e81_argument_omission_only_masked_simpo_seed42" ;;
    *) return 2 ;;
  esac
}

require_frozen() {
  [[ "$(sha256sum "${PROJECT_ROOT}/data/stage2_adaptive_datasets/${SOURCE_NAME}.jsonl" | cut -d' ' -f1)" == "${SOURCE_SHA256}" ]]
  jq -e '.frozen == true and .pairs_per_category == 8 and .test_data_access == false' \
    "${HOST_DATA_OUTPUT}/frozen_artifacts.json" >/dev/null
  local category expected actual
  for category in extra_frame wrong_type argument_omission; do
    expected="$(jq -r --arg category "${category}" '.artifact_sha256[$category]' "${HOST_DATA_OUTPUT}/frozen_artifacts.json")"
    actual="$(sha256sum "${HOST_DATA_OUTPUT}/${category}.jsonl" | cut -d' ' -f1)"
    [[ "${expected}" == "${actual}" ]]
  done
  [[ -f "${PROJECT_ROOT}/outputs/stage2_preference_mining/e118a_e81_difference_mask_audit_seed1180/pretrain_difference_margins.json" ]]
}

case "${1:-}" in
  build)
    docker_common "${IMAGE}" python scripts/build_e119_single_category_diagnostics_20260712.py \
      --source_jsonl "${SOURCE_JSONL}" --expected_sha256 "${SOURCE_SHA256}" \
      --output_dir "${DATA_OUTPUT}" --expected_per_category 8
    ;;
  train)
    category="${2:?category required}"
    gpu="${3:?GPU required}"
    id="$(experiment_id "${category}")"
    require_frozen
    if [[ -f "${PROJECT_ROOT}/outputs/stage2_preference_runs/${id}/trainer_state.json" ]]; then
      echo "${id} already ran; refusing output reuse" >&2
      exit 2
    fi
    release_owned_label_service "${gpu}"
    docker_common --gpus "device=${gpu}" \
      -e E118_DIFF_CONTEXT_TOKENS=1 \
      -e PYTHONPATH=/workspace/project/scripts/compat/e118_difference_mask:/workspace/project \
      "${IMAGE}" bash -lc \
      "set -o pipefail; mkdir -p '/workspace/project/outputs/stage2_preference_runs/${id}'; \
       FORCE_TORCHRUN=1 llamafactory-cli train 'configs/generated/stage2_preference/${id}.yaml' \
       2>&1 | tee '/workspace/project/outputs/stage2_preference_runs/${id}/train.log'"
    ;;
  score)
    category="${2:?category required}"
    gpu="${3:?GPU required}"
    id="$(experiment_id "${category}")"
    require_frozen
    release_owned_label_service "${gpu}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "set -o pipefail; python scripts/score_e118_difference_masked_margins_20260712.py \
        --model_path '/workspace/project/outputs/stage2_preference_runs/${id}' \
        --preference_jsonl '${SOURCE_JSONL}' --mask_manifest '${MASK_MANIFEST}' \
        --output_json '/workspace/project/outputs/stage2_preference_runs/${id}/posttrain_difference_margins.json' \
        --cutoff_len 1536 --context_tokens 1 2>&1 | tee \
        '/workspace/project/outputs/stage2_preference_runs/${id}/posttrain_scoring.log'"
    ;;
  gate)
    category="${2:?category required}"
    id="$(experiment_id "${category}")"
    require_frozen
    docker_common "${IMAGE}" python scripts/compare_e119_single_category_transfer_20260712.py \
      --target_category "${category}" --pre_score_json "${PRE_SCORE}" \
      --post_score_json "/workspace/project/outputs/stage2_preference_runs/${id}/posttrain_difference_margins.json" \
      --trainer_state "/workspace/project/outputs/stage2_preference_runs/${id}/trainer_state.json" \
      --training_log "/workspace/project/outputs/stage2_preference_runs/${id}/train.log" \
      --model_dir "/workspace/project/outputs/stage2_preference_runs/${id}" \
      --output_json "/workspace/project/outputs/stage2_preference_runs/${id}/single_category_transfer.json"
    ;;
  *)
    echo "usage: $0 build|train <category> <gpu>|score <category> <gpu>|gate <category>" >&2
    exit 2
    ;;
esac
