#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
MODEL_ROOT="/mnt/disk/gaojun/models"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
DATA_DIR="/workspace/project/data/stage2_confirmation_e121"
PREFIX="richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle"
RUN_NAME="e123a_glm51_verifierhigh4096_smoke40"
PROTOCOL="e123a-glm51-verifierhigh4096-trainonly-smoke40"
OUTPUT_DIR="/workspace/project/outputs/stage2_confirmation_e123/${RUN_NAME}"
HOST_OUTPUT_DIR="${PROJECT_ROOT}/outputs/stage2_confirmation_e123/${RUN_NAME}"
SOURCE="${DATA_DIR}/${PREFIX}_sgcot_target_train_pos.jsonl"

[[ -n "${LITELLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}" ]] || {
  echo "a LiteLLM key is required via LITELLM_API_KEY, LLM_API_KEY, or OPENAI_API_KEY" >&2
  exit 10
}
[[ ! -e "${HOST_OUTPUT_DIR}" ]] || {
  echo "refusing to reuse E123 output directory: ${HOST_OUTPUT_DIR}" >&2
  exit 11
}

extra=()
if [[ -n "${GEN_BASE_URL:-${LITELLM_BASE_URL:-}}" ]]; then
  extra+=(--base_url "${GEN_BASE_URL:-${LITELLM_BASE_URL}}")
fi

docker run --rm --user root --ipc host --shm-size 16g \
  -v "${PROJECT_ROOT}:/workspace/project" \
  -v "${MODEL_ROOT}:/workspace/models" \
  -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
  -v "${LF_ROOT}/cache/torch_extensions:/workspace/.cache/torch_extensions" \
  -e LITELLM_API_KEY -e LLM_API_KEY -e OPENAI_API_KEY \
  -e PYTHONUNBUFFERED=1 \
  -w /workspace/project "${IMAGE}" \
  python scripts/generate_strategy_variants_cot_e47_20260606.py \
    --run_name "${RUN_NAME}" \
    --limit 40 \
    --seed 1111 \
    --workers 8 \
    --model glm-5.1 \
    --gen_max_tokens 8192 \
    --verifier_model deepseek-v4-pro \
    --verifier_reasoning_effort high \
    --verify_max_tokens 4096 \
    --max_attempts 3 \
    --prompt_profile e95_trigger_locked_autocluster \
    --repair_profile strict_full \
    --output_protocol xml_tags \
    --sampled_rows_path "${SOURCE}" \
    --sampled_rows_mode priority_sample \
    --output_dir "${OUTPUT_DIR}" \
    --formal_data_dir "${DATA_DIR}" \
    --data_prefix "${PREFIX}_sgcot_target" \
    --adaptive_prefix "${PREFIX}" \
    --adaptive_data_dir "${OUTPUT_DIR}/generated_data" \
    --config_dir "${OUTPUT_DIR}/generated_configs" \
    --train_dataset_dir "${OUTPUT_DIR}/generated_data" \
    --run_prefix e123a_glm51_verifierhigh4096 \
    --warm_start /workspace/project/outputs/stage2_confirmation_e121/runs/e121b_direct_surface_seed42 \
    --auto_cluster_map_path /workspace/project/data/schema/richere-en.auto_cluster_map.json \
    "${extra[@]}"

docker run --rm --user root \
  -v "${PROJECT_ROOT}:/workspace/project" \
  -w /workspace/project "${IMAGE}" \
  python scripts/audit_e122_verifier_budget_smoke_20260712.py \
  --raw_jsonl "${OUTPUT_DIR}/e40_raw.jsonl" \
  --summary_json "${OUTPUT_DIR}/e47_summary.json" \
  --output_json "${OUTPUT_DIR}/e123a_gate.json" \
  --protocol "${PROTOCOL}" \
  --expected_rows 40 \
  --min_accepted 38 \
  --max_attempts 3 \
  --verify_max_tokens 4096 \
  --min_headroom_tokens 256 \
  --require_pass
