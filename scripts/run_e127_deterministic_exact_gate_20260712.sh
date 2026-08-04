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
E127_OUTPUT_ROOT="/workspace/project/outputs/stage2_confirmation_e127"
E127A_OUTPUT="${E127_OUTPUT_ROOT}/e127a_deterministic_exact_zero_leak_full1500"
E127B_OUTPUT="${E127_OUTPUT_ROOT}/e127b_deepseek_alias_audit100"
E127B1_OUTPUT="${E127_OUTPUT_ROOT}/e127b1_deepseek_alias_audit100_interfacefix"
E127_DATA_DIR="/workspace/project/data/stage2_confirmation_e127"
SOURCE_DATA_DIR="/workspace/project/data/stage2_confirmation_e121"
PREFIX="richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle"
SOURCE="${SOURCE_DATA_DIR}/${PREFIX}_sgcot_target_train_pos.jsonl"
NORMALIZED_NAME="${PREFIX}_sgcot_autocluster_e127a_train_pos"
NORMALIZED_TRAIN="${E127_DATA_DIR}/${NORMALIZED_NAME}.jsonl"
NORMALIZED_SHA="2da251e5a0f56e7b7c783299f2c24ab1160513eab0dd73ffc90f869d2c502d8d"
AUDIT_SAMPLE_SHA="c2c9c9baa38bfcbf28a1f67ecdf8c48e9806613e036df3894ab8c92a0d5d215a"
HELDOUT="/workspace/project/data/processed/type_holdout/richere-en/balanced-subtype-v2-confirmation/split1/unseen_types.json"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"

[[ "${ACTION}" == "build" || "${ACTION}" == "audit" || "${ACTION}" == "audit-interfacefix" ]] || {
  echo "usage: $0 build|audit|audit-interfacefix" >&2
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
  docker_common "${IMAGE}" python scripts/build_e127_deterministic_exact_trace_dataset_20260712.py \
    --raw_jsonl "${RAW}" \
    --sampled_rows_jsonl "${SAMPLED}" \
    --raw_sha256 "${RAW_SHA}" \
    --sampled_rows_sha256 "${SAMPLED_SHA}" \
    --heldout_types_json "${HELDOUT}" \
    --output_dir "${E127A_OUTPUT}" \
    --run_name e127a_deterministic_exact_zero_leak_full1500 \
    --minimum_rows 1400 \
    --require_pass

  docker_common "${IMAGE}" python scripts/normalize_strict_sgcot_dataset_20260712.py \
    --generated_jsonl "${E127A_OUTPUT}/deterministic_exact_evidence_cot.jsonl" \
    --reference_surface_jsonl "${SOURCE}" \
    --output_jsonl "${NORMALIZED_TRAIN}" \
    --dataset_name "${NORMALIZED_NAME}" \
    --dataset_info "${E127_DATA_DIR}/dataset_info.json" \
    --heldout_types_json "${HELDOUT}" \
    --require_zero_leaks \
    --min_rows 1400 \
    --model_path "${BASE_MODEL}" \
    --cutoff_len 2048

  docker_common "${IMAGE}" python scripts/validate_strict_unseen_dataset_20260712.py \
    --input_jsonl "${NORMALIZED_TRAIN}" \
    --heldout_types_json "${HELDOUT}" \
    --output_json "${E127A_OUTPUT}/normalized_trace_audit.json" \
    --require_zero_leaks \
    --require_exact_surface_recovery
  exit 0
fi

[[ -f "${PROJECT_ROOT}/data/stage2_confirmation_e127/${NORMALIZED_NAME}.jsonl" ]] || {
  echo "E127A normalized dataset is required before audit" >&2
  exit 13
}
HOST_NORMALIZED="${PROJECT_ROOT}/data/stage2_confirmation_e127/${NORMALIZED_NAME}.jsonl"
[[ "$(sha256sum "${HOST_NORMALIZED}" | cut -d' ' -f1)" == "${NORMALIZED_SHA}" ]] || {
  echo "E127B input SHA mismatch" >&2
  exit 14
}
OBSERVED_SAMPLE_SHA="$(python3 -c '
import hashlib, sys
from pathlib import Path
from scripts.run_e124c_independent_deepseek_audit_20260712 import load_jsonl, select_audit_rows, wnd_id
rows = select_audit_rows(load_jsonl(Path(sys.argv[1])), 100, 1270)
print(hashlib.sha256("\n".join(wnd_id(row) for row in rows).encode()).hexdigest())
' "${HOST_NORMALIZED}")"
[[ "${OBSERVED_SAMPLE_SHA}" == "${AUDIT_SAMPLE_SHA}" ]] || {
  echo "E127B audit sample SHA mismatch" >&2
  exit 15
}
[[ -n "${LITELLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}" ]] || {
  echo "a LiteLLM key is required via LITELLM_API_KEY, LLM_API_KEY, or OPENAI_API_KEY" >&2
  exit 10
}
extra=()
if [[ -n "${GEN_BASE_URL:-${LITELLM_BASE_URL:-}}" ]]; then
  extra+=(--base_url "${GEN_BASE_URL:-${LITELLM_BASE_URL}}")
fi
AUDIT_OUTPUT="${E127B_OUTPUT}"
AUDIT_PROTOCOL="e127b-independent-deepseek-alias-audit100-v1"
HARD_PROFILE="generator_hard_verify"
if [[ "${ACTION}" == "audit-interfacefix" ]]; then
  AUDIT_OUTPUT="${E127B1_OUTPUT}"
  AUDIT_PROTOCOL="e127b1-independent-deepseek-alias-audit100-interfacefix-v1"
  HARD_PROFILE="normalized_surface_exact"
fi
docker_common "${IMAGE}" python scripts/run_e124c_independent_deepseek_audit_20260712.py \
  --input_jsonl "${NORMALIZED_TRAIN}" \
  --output_dir "${AUDIT_OUTPUT}" \
  --protocol "${AUDIT_PROTOCOL}" \
  --sample_size 100 \
  --seed 1270 \
  --model deepseek-v4-pro \
  --reasoning_effort high \
  --verifier_profile target_role_alias_v1 \
  --hard_profile "${HARD_PROFILE}" \
  --max_tokens 8192 \
  --max_attempts 3 \
  --workers 16 \
  --min_semantic_pass 85 \
  --timeout 600 \
  --require_pass \
  "${extra[@]}"
