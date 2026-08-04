#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
MODEL_ROOT="/mnt/disk/gaojun/models"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
SOURCE_DATA_DIR="/workspace/project/data/stage2_confirmation_e121"
TARGET_DATA_DIR="/workspace/project/data/stage2_confirmation_e124"
PREFIX="richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle"
RUN_NAME="e124b_glm51_selfverifier4096_full1500"
OUTPUT_DIR="/workspace/project/outputs/stage2_confirmation_e124/${RUN_NAME}"
HOST_OUTPUT_DIR="${PROJECT_ROOT}/outputs/stage2_confirmation_e124/${RUN_NAME}"
WORKERS="${E124B_WORKERS:-8}"
RESUME="${E124B_RESUME:-0}"
SOURCE="${SOURCE_DATA_DIR}/${PREFIX}_sgcot_target_train_pos.jsonl"
GENERATED_TRAIN="${TARGET_DATA_DIR}/${PREFIX}_${RUN_NAME}_thinking_evidence_cot_train_pos.jsonl"
NORMALIZED_NAME="${PREFIX}_sgcot_autocluster_e124b_train_pos"
NORMALIZED_TRAIN="${TARGET_DATA_DIR}/${NORMALIZED_NAME}.jsonl"
HELDOUT="/workspace/project/data/processed/type_holdout/richere-en/balanced-subtype-v2-confirmation/split1/unseen_types.json"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"

[[ -n "${LITELLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}" ]] || {
  echo "a LiteLLM key is required via LITELLM_API_KEY, LLM_API_KEY, or OPENAI_API_KEY" >&2
  exit 10
}
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "E124B_WORKERS must be a positive integer" >&2
  exit 12
}
[[ "${RESUME}" == "0" || "${RESUME}" == "1" ]] || {
  echo "E124B_RESUME must be 0 or 1" >&2
  exit 13
}
[[ ! -e "${HOST_OUTPUT_DIR}" || "${RESUME}" == "1" ]] || {
  echo "refusing to reuse E124B output directory: ${HOST_OUTPUT_DIR}" >&2
  exit 11
}
[[ -e "${HOST_OUTPUT_DIR}" || "${RESUME}" == "0" ]] || {
  echo "cannot resume missing E124B output directory: ${HOST_OUTPUT_DIR}" >&2
  exit 14
}

extra=()
if [[ -n "${GEN_BASE_URL:-${LITELLM_BASE_URL:-}}" ]]; then
  extra+=(--base_url "${GEN_BASE_URL:-${LITELLM_BASE_URL}}")
fi

docker_common() {
  docker run --rm --user root --ipc host --shm-size 16g \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -v "${LF_ROOT}/cache/torch_extensions:/workspace/.cache/torch_extensions" \
    -e LITELLM_API_KEY -e LLM_API_KEY -e OPENAI_API_KEY \
    -e PYTHONUNBUFFERED=1 \
    -w /workspace/project "$@"
}

docker_common "${IMAGE}" python scripts/generate_strategy_variants_cot_e47_20260606.py \
  --run_name "${RUN_NAME}" \
  --limit 1500 --seed 1111 --workers "${WORKERS}" \
  --model glm-5.1 --gen_max_tokens 8192 \
  --verifier_model glm-5.1 --verify_max_tokens 4096 \
  --max_attempts 3 \
  --prompt_profile e95_trigger_locked_autocluster \
  --repair_profile strict_full --output_protocol xml_tags \
  --sampled_rows_path "${SOURCE}" --sampled_rows_mode priority_sample \
  --output_dir "${OUTPUT_DIR}" \
  --formal_data_dir "${SOURCE_DATA_DIR}" \
  --data_prefix "${PREFIX}_sgcot_target" \
  --adaptive_prefix "${PREFIX}" \
  --adaptive_data_dir "${TARGET_DATA_DIR}" \
  --config_dir /workspace/project/configs/generated/stage2_confirmation/e124b_generated \
  --train_dataset_dir "${TARGET_DATA_DIR}" \
  --run_prefix e124b_glm51_selfverifier4096 \
  --warm_start /workspace/project/outputs/stage2_confirmation_e121/runs/e121b_direct_surface_seed42 \
  --auto_cluster_map_path /workspace/project/data/schema/richere-en.auto_cluster_map.json \
  "${extra[@]}"

docker_common "${IMAGE}" python scripts/audit_e124b_full_generation_20260712.py \
  --raw_jsonl "${OUTPUT_DIR}/e40_raw.jsonl" \
  --summary_json "${OUTPUT_DIR}/e47_summary.json" \
  --output_json "${OUTPUT_DIR}/e124b_gate.json" \
  --expected_rows 1500 --min_accepted 1400 --max_attempts 3 \
  --max_verifier_failure_rate 0.01 \
  --verify_max_tokens 4096 --min_p99_headroom_tokens 256 --require_pass

docker_common "${IMAGE}" python scripts/normalize_strict_sgcot_dataset_20260712.py \
  --generated_jsonl "${GENERATED_TRAIN}" \
  --reference_surface_jsonl "${SOURCE}" \
  --output_jsonl "${NORMALIZED_TRAIN}" \
  --dataset_name "${NORMALIZED_NAME}" \
  --dataset_info "${TARGET_DATA_DIR}/dataset_info.json" \
  --heldout_types_json "${HELDOUT}" \
  --require_zero_leaks --min_rows 1400 \
  --model_path "${BASE_MODEL}" --cutoff_len 2048

docker_common "${IMAGE}" python scripts/validate_strict_unseen_dataset_20260712.py \
  --input_jsonl "${NORMALIZED_TRAIN}" \
  --heldout_types_json "${HELDOUT}" \
  --output_json "${OUTPUT_DIR}/normalized_trace_audit.json" \
  --require_zero_leaks --require_exact_surface_recovery
