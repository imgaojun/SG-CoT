#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
SOURCE="/workspace/project/data/stage2_confirmation_e124/richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle_sgcot_autocluster_e124b_train_pos.jsonl"
OUTPUT="/workspace/project/outputs/stage2_confirmation_e124/e124c_deepseek_audit100"
HOST_OUTPUT="${PROJECT_ROOT}/outputs/stage2_confirmation_e124/e124c_deepseek_audit100"

[[ -n "${LITELLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}" ]] || {
  echo "a LiteLLM key is required via LITELLM_API_KEY, LLM_API_KEY, or OPENAI_API_KEY" >&2
  exit 10
}
[[ ! -e "${HOST_OUTPUT}" ]] || {
  echo "refusing to reuse E124C output directory: ${HOST_OUTPUT}" >&2
  exit 11
}

extra=()
if [[ -n "${GEN_BASE_URL:-${LITELLM_BASE_URL:-}}" ]]; then
  extra+=(--base_url "${GEN_BASE_URL:-${LITELLM_BASE_URL}}")
fi

docker run --rm --user root --ipc host --shm-size 16g \
  -v "${PROJECT_ROOT}:/workspace/project" \
  -e LITELLM_API_KEY -e LLM_API_KEY -e OPENAI_API_KEY \
  -e PYTHONUNBUFFERED=1 \
  -w /workspace/project "${IMAGE}" \
  python scripts/run_e124c_independent_deepseek_audit_20260712.py \
    --input_jsonl "${SOURCE}" --output_dir "${OUTPUT}" \
    --sample_size 100 --seed 1242 \
    --model deepseek-v4-pro --reasoning_effort high --max_tokens 4096 \
    --max_attempts 3 --workers 8 --min_semantic_pass 95 --timeout 360 --require_pass \
    "${extra[@]}"
