#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk/gaojun/research/progressive-ee

OUT_DIR="outputs/stage2_strategy_cot_e76/e76_contrastive_exactness_glm51_full1500_20260614"
MANIFEST="${OUT_DIR}/e76_manifest_rows.jsonl"

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

python3 scripts/build_e76_contrastive_manifest_20260614.py \
  --output_dir "${OUT_DIR}" \
  --seed 7601

python3 scripts/generate_strategy_variants_cot_e47_20260606.py \
  --run_name e76_contrastive_exactness_glm51_full1500 \
  --limit 1500 \
  --seed 7601 \
  --workers 24 \
  --base_url ${LLM_BASE_URL} \
  --model glm-5.1 \
  --verifier_model deepseek-v4-pro \
  --verifier_reasoning_effort max \
  --prompt_profile e76_contrastive_exactness \
  --output_protocol xml_tags \
  --gen_max_tokens 8192 \
  --verify_max_tokens 1800 \
  --timeout 420 \
  --sampled_rows_path "${MANIFEST}" \
  --output_dir "${OUT_DIR}"
