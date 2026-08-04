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
DATA_OUTPUT="/workspace/project/outputs/stage2_preference_mining/e120a_e81_remaining_single_category_transfer"
HOST_DATA_OUTPUT="${PROJECT_ROOT}/outputs/stage2_preference_mining/e120a_e81_remaining_single_category_transfer"
BASE_MODEL="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full"
MASK_MANIFEST="/workspace/project/outputs/stage2_preference_mining/e118a_e81_difference_mask_audit_seed1180/difference_mask_manifest.jsonl"
PRE_SCORE="/workspace/project/outputs/stage2_preference_mining/e118a_e81_difference_mask_audit_seed1180/pretrain_difference_margins.json"
WEIGHT_OUTPUT="/workspace/project/outputs/stage2_preference_mining/e120c_e81_atomic_transfer_balance_seed42/transfer_balanced_weights.json"
HOST_WEIGHT_OUTPUT="${PROJECT_ROOT}/outputs/stage2_preference_mining/e120c_e81_atomic_transfer_balance_seed42/transfer_balanced_weights.json"
COMPOSED_ID="e120d_e81_transfer_balanced_delta_composition_seed42"

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

require_gpu_idle() {
  local gpu="$1"
  release_owned_label_service "${gpu}"
  local used utilization
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  utilization="$(nvidia-smi --id="${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  if (( used > 1024 || utilization > 5 )); then
    echo "GPU ${gpu} is not idle: memory=${used} MiB utilization=${utilization}%" >&2
    return 3
  fi
}

expert_id() {
  case "$1" in
    event_omission) echo "e120b1_e81_event_omission_only_masked_simpo_seed42" ;;
    trigger_drift) echo "e120b2_e81_trigger_drift_only_masked_simpo_seed42" ;;
    *) return 2 ;;
  esac
}

require_data() {
  [[ "$(sha256sum "${PROJECT_ROOT}/data/stage2_adaptive_datasets/${SOURCE_NAME}.jsonl" | cut -d' ' -f1)" == "${SOURCE_SHA256}" ]]
  jq -e '.frozen == true and .pairs_per_category == 8 and .test_data_access == false' \
    "${HOST_DATA_OUTPUT}/frozen_artifacts.json" >/dev/null
}

score_model() {
  local model_path="$1"
  local output_path="$2"
  local log_path="$3"
  docker_common --gpus "device=${4}" "${IMAGE}" bash -lc \
    "set -o pipefail; python scripts/score_e118_difference_masked_margins_20260712.py \
      --model_path '${model_path}' --preference_jsonl '${SOURCE_JSONL}' \
      --mask_manifest '${MASK_MANIFEST}' --output_json '${output_path}' \
      --cutoff_len 1536 --context_tokens 1 2>&1 | tee '${log_path}'"
}

case "${1:-}" in
  build)
    docker_common "${IMAGE}" python scripts/build_e120_remaining_category_slices_20260712.py \
      --source_jsonl "${SOURCE_JSONL}" --expected_sha256 "${SOURCE_SHA256}" \
      --output_dir "${DATA_OUTPUT}" --expected_per_category 8
    ;;
  train)
    category="${2:?category required}"
    gpu="${3:?GPU required}"
    id="$(expert_id "${category}")"
    require_data
    [[ ! -e "${PROJECT_ROOT}/outputs/stage2_preference_runs/${id}" ]] || {
      echo "${id} output already exists; refusing reuse" >&2
      exit 2
    }
    require_gpu_idle "${gpu}"
    docker_common --gpus "device=${gpu}" \
      -e E118_DIFF_CONTEXT_TOKENS=1 \
      -e PYTHONPATH=/workspace/project/scripts/compat/e118_difference_mask:/workspace/project \
      "${IMAGE}" bash -lc \
      "set -o pipefail; mkdir -p '/workspace/project/outputs/stage2_preference_runs/${id}'; \
       FORCE_TORCHRUN=1 llamafactory-cli train 'configs/generated/stage2_preference/${id}.yaml' \
       2>&1 | tee '/workspace/project/outputs/stage2_preference_runs/${id}/train.log'"
    ;;
  score-expert)
    category="${2:?category required}"
    gpu="${3:?GPU required}"
    id="$(expert_id "${category}")"
    require_data
    require_gpu_idle "${gpu}"
    score_model \
      "/workspace/project/outputs/stage2_preference_runs/${id}" \
      "/workspace/project/outputs/stage2_preference_runs/${id}/posttrain_difference_margins.json" \
      "/workspace/project/outputs/stage2_preference_runs/${id}/posttrain_scoring.log" \
      "${gpu}"
    ;;
  gate-expert)
    category="${2:?category required}"
    id="$(expert_id "${category}")"
    require_data
    docker_common "${IMAGE}" python scripts/compare_e119_single_category_transfer_20260712.py \
      --target_category "${category}" --pre_score_json "${PRE_SCORE}" \
      --post_score_json "/workspace/project/outputs/stage2_preference_runs/${id}/posttrain_difference_margins.json" \
      --trainer_state "/workspace/project/outputs/stage2_preference_runs/${id}/trainer_state.json" \
      --training_log "/workspace/project/outputs/stage2_preference_runs/${id}/train.log" \
      --model_dir "/workspace/project/outputs/stage2_preference_runs/${id}" \
      --output_json "/workspace/project/outputs/stage2_preference_runs/${id}/single_category_transfer.json"
    ;;
  solve)
    [[ ! -e "$(dirname "${HOST_WEIGHT_OUTPUT}")" ]] || {
      echo "E120C output already exists; refusing reuse" >&2
      exit 2
    }
    docker_common "${IMAGE}" python scripts/solve_e120_transfer_balanced_weights_20260712.py \
      --report argument_omission=/workspace/project/outputs/stage2_preference_runs/e119b3_e81_argument_omission_only_masked_simpo_seed42/single_category_transfer.json \
      --report event_omission=/workspace/project/outputs/stage2_preference_runs/e120b1_e81_event_omission_only_masked_simpo_seed42/single_category_transfer.json \
      --report extra_frame=/workspace/project/outputs/stage2_preference_runs/e119b1_e81_extra_frame_only_masked_simpo_seed42/single_category_transfer.json \
      --report trigger_drift=/workspace/project/outputs/stage2_preference_runs/e120b2_e81_trigger_drift_only_masked_simpo_seed42/single_category_transfer.json \
      --report wrong_type=/workspace/project/outputs/stage2_preference_runs/e119b2_e81_wrong_type_only_masked_simpo_seed42/single_category_transfer.json \
      --output_json "${WEIGHT_OUTPUT}" --composition_scale 5.0 --full_floor 0.0
    ;;
  merge)
    [[ -f "${HOST_WEIGHT_OUTPUT}" ]]
    docker_common "${IMAGE}" python scripts/merge_e120_transfer_balanced_deltas_20260712.py \
      --base_model "${BASE_MODEL}" \
      --expert argument_omission=/workspace/project/outputs/stage2_preference_runs/e119b3_e81_argument_omission_only_masked_simpo_seed42 \
      --expert event_omission=/workspace/project/outputs/stage2_preference_runs/e120b1_e81_event_omission_only_masked_simpo_seed42 \
      --expert extra_frame=/workspace/project/outputs/stage2_preference_runs/e119b1_e81_extra_frame_only_masked_simpo_seed42 \
      --expert trigger_drift=/workspace/project/outputs/stage2_preference_runs/e120b2_e81_trigger_drift_only_masked_simpo_seed42 \
      --expert wrong_type=/workspace/project/outputs/stage2_preference_runs/e119b2_e81_wrong_type_only_masked_simpo_seed42 \
      --weights_json "${WEIGHT_OUTPUT}" --composition_scale 5.0 \
      --output_dir "/workspace/project/outputs/stage2_preference_runs/${COMPOSED_ID}"
    ;;
  score-composed)
    gpu="${2:?GPU required}"
    require_gpu_idle "${gpu}"
    score_model \
      "/workspace/project/outputs/stage2_preference_runs/${COMPOSED_ID}" \
      "/workspace/project/outputs/stage2_preference_runs/${COMPOSED_ID}/posttrain_difference_margins.json" \
      "/workspace/project/outputs/stage2_preference_runs/${COMPOSED_ID}/posttrain_scoring.log" \
      "${gpu}"
    ;;
  gate-composed)
    docker_common "${IMAGE}" python scripts/compare_e120_composition_gate_20260712.py \
      --pre_score_json "${PRE_SCORE}" \
      --post_score_json "/workspace/project/outputs/stage2_preference_runs/${COMPOSED_ID}/posttrain_difference_margins.json" \
      --weights_json "${WEIGHT_OUTPUT}" \
      --composition_manifest "/workspace/project/outputs/stage2_preference_runs/${COMPOSED_ID}/composition_manifest.json" \
      --output_json "/workspace/project/outputs/stage2_preference_runs/${COMPOSED_ID}/composition_gate.json" \
      --min_masked_mean_delta 0.005 --min_chosen_full_delta -0.02
    ;;
  test)
    docker_common "${IMAGE}" python -m unittest tests.test_e120_transfer_balanced_composition
    ;;
  *)
    echo "usage: $0 build|train <category> <gpu>|score-expert <category> <gpu>|gate-expert <category>|solve|merge|score-composed <gpu>|gate-composed|test" >&2
    exit 2
    ;;
esac
