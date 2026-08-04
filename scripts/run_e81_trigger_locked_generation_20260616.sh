#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk/gaojun/research/progressive-ee

# Optional first arg: sample limit (default 1500). Use a small value for a smoke test.
LIMIT="${1:-1500}"

PROMPT_PROFILE="e81_trigger_locked_arbitration"
SEED=8101

if [ "${LIMIT}" -lt 50 ]; then
  RUN_NAME="e81_trigger_locked_arbitration_glm51_smoke${LIMIT}"
  OUT_DIR="outputs/stage2_strategy_cot_e81/e81_trigger_locked_arbitration_glm51_smoke${LIMIT}_20260616"
else
  RUN_NAME="e81_trigger_locked_arbitration_glm51_full1500"
  OUT_DIR="outputs/stage2_strategy_cot_e81/e81_trigger_locked_arbitration_glm51_full1500_20260616"
fi

MANIFEST="${OUT_DIR}/e76_manifest_rows.jsonl"

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

python3 scripts/build_e76_contrastive_manifest_20260614.py \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}"

python3 scripts/generate_strategy_variants_cot_e47_20260606.py \
  --run_name "${RUN_NAME}" \
  --limit "${LIMIT}" \
  --seed "${SEED}" \
  --workers 24 \
  --base_url ${LLM_BASE_URL} \
  --model glm-5.1 \
  --verifier_model deepseek-v4-pro \
  --verifier_reasoning_effort max \
  --prompt_profile "${PROMPT_PROFILE}" \
  --output_protocol xml_tags \
  --gen_max_tokens 8192 \
  --verify_max_tokens 1800 \
  --timeout 420 \
  --sampled_rows_path "${MANIFEST}" \
  --output_dir "${OUT_DIR}"
