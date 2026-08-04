#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk/gaojun/research/progressive-ee

VARIANT="${1:?variant must be e80a or e80b}"

case "${VARIANT}" in
  e80a)
    RUN_NAME="e80a_no_type_arbitration_glm51_full1500"
    PROMPT_PROFILE="e80a_no_type_arbitration"
    OUT_DIR="outputs/stage2_strategy_cot_e80/e80a_no_type_arbitration_glm51_full1500_20260615"
    SEED=8001
    ;;
  e80b)
    RUN_NAME="e80b_no_argument_pruning_glm51_full1500"
    PROMPT_PROFILE="e80b_no_argument_pruning"
    OUT_DIR="outputs/stage2_strategy_cot_e80/e80b_no_argument_pruning_glm51_full1500_20260615"
    SEED=8002
    ;;
  *)
    echo "usage: $0 e80a|e80b" >&2
    exit 2
    ;;
esac

MANIFEST="${OUT_DIR}/e76_manifest_rows.jsonl"

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

python3 scripts/build_e76_contrastive_manifest_20260614.py \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}"

python3 scripts/generate_strategy_variants_cot_e47_20260606.py \
  --run_name "${RUN_NAME}" \
  --limit 1500 \
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
