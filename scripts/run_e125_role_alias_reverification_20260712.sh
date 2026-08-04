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
E125_DATA_DIR="/workspace/project/data/stage2_confirmation_e125"
SOURCE_DATA_DIR="/workspace/project/data/stage2_confirmation_e121"
PREFIX="richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle"
SOURCE="${SOURCE_DATA_DIR}/${PREFIX}_sgcot_target_train_pos.jsonl"
HELDOUT="/workspace/project/data/processed/type_holdout/richere-en/balanced-subtype-v2-confirmation/split1/unseen_types.json"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
RAW_SHA="073dcd5685dc3b04b4dd045e9b3cfbb495fa7d96fb745090b04323eae2bec66c"

[[ "${ACTION}" == "smoke" || "${ACTION}" == "full" ]] || {
  echo "usage: $0 smoke|full" >&2
  exit 2
}
[[ -n "${LITELLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}" ]] || {
  echo "a LiteLLM key is required via LITELLM_API_KEY, LLM_API_KEY, or OPENAI_API_KEY" >&2
  exit 10
}
[[ "$(sha256sum "${PROJECT_ROOT}/outputs/stage2_confirmation_e124/e124b_glm51_selfverifier4096_full1500/e40_raw.jsonl" | cut -d' ' -f1)" == "${RAW_SHA}" ]] || {
  echo "frozen E124B raw SHA mismatch" >&2
  exit 11
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

if [[ "${ACTION}" == "smoke" ]]; then
  OUTPUT="/workspace/project/outputs/stage2_confirmation_e125/e125a_role_alias_reverify_smoke40"
  docker_common "${IMAGE}" python scripts/reverify_e125_role_alias_20260712.py \
    --raw_jsonl "${RAW}" --sampled_rows_jsonl "${SAMPLED}" \
    --output_dir "${OUTPUT}" --run_name e125a_role_alias_reverify_smoke40 \
    --sample_size 40 --seed 1250 \
    --model glm-5.1 --verifier_profile target_role_alias_v1 \
    --max_tokens 6144 --max_attempts 3 --workers 16 \
    --min_hard_valid 40 --min_valid_judgments 40 --min_semantic_pass 37 \
    --max_failure_rate 0.0 --min_p99_headroom_tokens 512 \
    --require_pass "${extra[@]}"
  exit 0
fi

OUTPUT="/workspace/project/outputs/stage2_confirmation_e125/e125b_role_alias_reverify_full1500"
NORMALIZED_NAME="${PREFIX}_sgcot_autocluster_e125b_train_pos"
NORMALIZED_TRAIN="${E125_DATA_DIR}/${NORMALIZED_NAME}.jsonl"
docker_common "${IMAGE}" python scripts/reverify_e125_role_alias_20260712.py \
  --raw_jsonl "${RAW}" --sampled_rows_jsonl "${SAMPLED}" \
  --output_dir "${OUTPUT}" --run_name e125b_role_alias_reverify_full1500 \
  --sample_size 0 --seed 1250 \
  --model glm-5.1 --verifier_profile target_role_alias_v1 \
  --max_tokens 6144 --max_attempts 3 --workers 16 \
  --min_hard_valid 1499 --min_valid_judgments 1490 --min_semantic_pass 1400 \
  --max_failure_rate 0.01 --min_p99_headroom_tokens 512 \
  --require_pass "${extra[@]}"

docker_common "${IMAGE}" python scripts/normalize_strict_sgcot_dataset_20260712.py \
  --generated_jsonl "${OUTPUT}/accepted_evidence_cot.jsonl" \
  --reference_surface_jsonl "${SOURCE}" \
  --output_jsonl "${NORMALIZED_TRAIN}" \
  --dataset_name "${NORMALIZED_NAME}" \
  --dataset_info "${E125_DATA_DIR}/dataset_info.json" \
  --heldout_types_json "${HELDOUT}" \
  --require_zero_leaks --min_rows 1400 \
  --model_path "${BASE_MODEL}" --cutoff_len 2048

docker_common "${IMAGE}" python scripts/validate_strict_unseen_dataset_20260712.py \
  --input_jsonl "${NORMALIZED_TRAIN}" \
  --heldout_types_json "${HELDOUT}" \
  --output_json "${OUTPUT}/normalized_trace_audit.json" \
  --require_zero_leaks --require_exact_surface_recovery
