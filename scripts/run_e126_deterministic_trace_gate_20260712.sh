#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
MODEL_ROOT="/mnt/disk/gaojun/models"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
E124B_DIR="/workspace/project/outputs/stage2_confirmation_e124/e124b_glm51_selfverifier4096_full1500"
RAW="${E124B_DIR}/e40_raw.jsonl"
SAMPLED="${E124B_DIR}/sampled_rows.jsonl"
RAW_SHA="073dcd5685dc3b04b4dd045e9b3cfbb495fa7d96fb745090b04323eae2bec66c"
SAMPLED_SHA="c55d53d8179faf2619cfd93dcfea598f378464b4573466bfb85b0ec27cfb90d4"
E126_OUTPUT_ROOT="/workspace/project/outputs/stage2_confirmation_e126"
E126A_OUTPUT="${E126_OUTPUT_ROOT}/e126a_deterministic_hard_valid_full1500"
E126B_OUTPUT="${E126_OUTPUT_ROOT}/e126b_deepseek_alias_audit100"
E126_DATA_DIR="/workspace/project/data/stage2_confirmation_e126"
SOURCE_DATA_DIR="/workspace/project/data/stage2_confirmation_e121"
PREFIX="richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle"
SOURCE="${SOURCE_DATA_DIR}/${PREFIX}_sgcot_target_train_pos.jsonl"
NORMALIZED_NAME="${PREFIX}_sgcot_autocluster_e126a_train_pos"
NORMALIZED_TRAIN="${E126_DATA_DIR}/${NORMALIZED_NAME}.jsonl"
HELDOUT="/workspace/project/data/processed/type_holdout/richere-en/balanced-subtype-v2-confirmation/split1/unseen_types.json"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"

[[ "${ACTION}" == "build" || "${ACTION}" == "audit" ]] || {
  echo "usage: $0 build|audit" >&2
  exit 2
}

verify_frozen_inputs() {
  local host_dir="${PROJECT_ROOT}/outputs/stage2_confirmation_e124/e124b_glm51_selfverifier4096_full1500"
  [[ "$(sha256sum "${host_dir}/e40_raw.jsonl" | cut -d' ' -f1)" == "${RAW_SHA}" ]] || {
    echo "frozen E124B raw SHA mismatch" >&2
    exit 11
  }
  [[ "$(sha256sum "${host_dir}/sampled_rows.jsonl" | cut -d' ' -f1)" == "${SAMPLED_SHA}" ]] || {
    echo "frozen E124B sampled-row SHA mismatch" >&2
    exit 12
  }
}

docker_common() {
  docker run --rm --user root --ipc host --shm-size 16g \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -e LITELLM_API_KEY -e LLM_API_KEY -e OPENAI_API_KEY \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    -w /workspace/project "$@"
}

verify_frozen_inputs

if [[ "${ACTION}" == "build" ]]; then
  docker_common "${IMAGE}" python scripts/build_e126_deterministic_trace_dataset_20260712.py \
    --raw_jsonl "${RAW}" \
    --sampled_rows_jsonl "${SAMPLED}" \
    --raw_sha256 "${RAW_SHA}" \
    --output_dir "${E126A_OUTPUT}" \
    --run_name e126a_deterministic_hard_valid_full1500 \
    --require_pass

  docker_common "${IMAGE}" python scripts/normalize_strict_sgcot_dataset_20260712.py \
    --generated_jsonl "${E126A_OUTPUT}/hard_valid_evidence_cot.jsonl" \
    --reference_surface_jsonl "${SOURCE}" \
    --output_jsonl "${NORMALIZED_TRAIN}" \
    --dataset_name "${NORMALIZED_NAME}" \
    --dataset_info "${E126_DATA_DIR}/dataset_info.json" \
    --heldout_types_json "${HELDOUT}" \
    --require_zero_leaks \
    --min_rows 1400 \
    --model_path "${BASE_MODEL}" \
    --cutoff_len 2048

  docker_common "${IMAGE}" python scripts/validate_strict_unseen_dataset_20260712.py \
    --input_jsonl "${NORMALIZED_TRAIN}" \
    --heldout_types_json "${HELDOUT}" \
    --output_json "${E126A_OUTPUT}/normalized_trace_audit.json" \
    --require_zero_leaks \
    --require_exact_surface_recovery
  exit 0
fi

[[ -f "${PROJECT_ROOT}/data/stage2_confirmation_e126/${NORMALIZED_NAME}.jsonl" ]] || {
  echo "E126A normalized dataset is required before audit" >&2
  exit 13
}
[[ -n "${LITELLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}" ]] || {
  echo "a LiteLLM key is required via LITELLM_API_KEY, LLM_API_KEY, or OPENAI_API_KEY" >&2
  exit 10
}
extra=()
if [[ -n "${GEN_BASE_URL:-${LITELLM_BASE_URL:-}}" ]]; then
  extra+=(--base_url "${GEN_BASE_URL:-${LITELLM_BASE_URL}}")
fi
docker_common "${IMAGE}" python scripts/run_e124c_independent_deepseek_audit_20260712.py \
  --input_jsonl "${NORMALIZED_TRAIN}" \
  --output_dir "${E126B_OUTPUT}" \
  --protocol e126b-independent-deepseek-alias-audit100-v1 \
  --sample_size 100 \
  --seed 1260 \
  --model deepseek-v4-pro \
  --reasoning_effort high \
  --verifier_profile target_role_alias_v1 \
  --max_tokens 8192 \
  --max_attempts 3 \
  --workers 16 \
  --min_semantic_pass 85 \
  --timeout 600 \
  --require_pass \
  "${extra[@]}"
