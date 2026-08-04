#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
MODEL_ROOT="/mnt/disk/gaojun/models"
SOURCE_REL="data/stage2_confirmation_e127/richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle_sgcot_autocluster_e127a_train_pos.jsonl"
EXCLUDE_REL="outputs/stage2_confirmation_e127/e127b1_deepseek_alias_audit100_interfacefix/sampled_rows.jsonl"
SOURCE="/workspace/project/${SOURCE_REL}"
EXCLUDE="/workspace/project/${EXCLUDE_REL}"
OUTPUT="/workspace/project/outputs/stage2_confirmation_e128/e128a_disjoint_core_reasoning_audit100"
SOURCE_SHA="2da251e5a0f56e7b7c783299f2c24ab1160513eab0dd73ffc90f869d2c502d8d"
EXCLUDE_SHA="5eeb59d3b28a73f783ca82c314aa8b448ee626769be2861097333c6ec0f8148c"
SAMPLE_SHA="e3cd83ab5aba8de43b0af07214bb86a56cbe58aeb95525b12b66bd02c5255e5f"

[[ "$(sha256sum "${PROJECT_ROOT}/${SOURCE_REL}" | cut -d' ' -f1)" == "${SOURCE_SHA}" ]] || {
  echo "E128 input SHA mismatch" >&2
  exit 11
}
[[ "$(sha256sum "${PROJECT_ROOT}/${EXCLUDE_REL}" | cut -d' ' -f1)" == "${EXCLUDE_SHA}" ]] || {
  echo "E128 exclusion SHA mismatch" >&2
  exit 12
}
OBSERVED_SAMPLE_SHA="$(python3 -c '
import hashlib, sys
from pathlib import Path
from scripts.run_e124c_independent_deepseek_audit_20260712 import load_jsonl, select_audit_rows, wnd_id
source, excluded = Path(sys.argv[1]), Path(sys.argv[2])
excluded_ids = {wnd_id(row) for row in load_jsonl(excluded)}
rows = select_audit_rows(load_jsonl(source), 100, 1280, excluded_ids)
print(hashlib.sha256("\n".join(wnd_id(row) for row in rows).encode()).hexdigest())
' "${PROJECT_ROOT}/${SOURCE_REL}" "${PROJECT_ROOT}/${EXCLUDE_REL}")"
[[ "${OBSERVED_SAMPLE_SHA}" == "${SAMPLE_SHA}" ]] || {
  echo "E128 sample SHA mismatch" >&2
  exit 13
}
[[ ! -e "${PROJECT_ROOT}/outputs/stage2_confirmation_e128/e128a_disjoint_core_reasoning_audit100" ]] || {
  echo "refusing to reuse E128 output directory" >&2
  exit 14
}
[[ -n "${LITELLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}" ]] || {
  echo "a LiteLLM key is required via LITELLM_API_KEY, LLM_API_KEY, or OPENAI_API_KEY" >&2
  exit 10
}

extra=()
if [[ -n "${GEN_BASE_URL:-${LITELLM_BASE_URL:-}}" ]]; then
  extra+=(--base_url "${GEN_BASE_URL:-${LITELLM_BASE_URL}}")
fi

docker run --rm --user root --ipc host --shm-size 16g \
  -v "${PROJECT_ROOT}:/workspace/project" \
  -v "${MODEL_ROOT}:/workspace/models" \
  -e LITELLM_API_KEY -e LLM_API_KEY -e OPENAI_API_KEY \
  -e PYTHONUNBUFFERED=1 \
  -w /workspace/project "${IMAGE}" \
  python scripts/run_e124c_independent_deepseek_audit_20260712.py \
  --input_jsonl "${SOURCE}" \
  --exclude_jsonl "${EXCLUDE}" \
  --output_dir "${OUTPUT}" \
  --protocol e128a-disjoint-core-reasoning-audit100-v1 \
  --sample_size 100 \
  --seed 1280 \
  --model deepseek-v4-pro \
  --reasoning_effort high \
  --verifier_profile target_role_alias_core_reasoning_v1 \
  --hard_profile normalized_surface_exact \
  --semantic_profile core_reasoning_v1 \
  --max_tokens 8192 \
  --max_attempts 3 \
  --workers 16 \
  --min_semantic_pass 85 \
  --timeout 600 \
  --require_pass \
  "${extra[@]}"
