#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk/gaojun/research/progressive-ee

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

python3 scripts/generate_strategy_variants_cot_e47_20260606.py \
  --run_name e73_e57_recall_first_exactness_last_glm51_3k \
  --limit 3000 \
  --seed 7303 \
  --workers 24 \
  --base_url ${LLM_BASE_URL} \
  --model glm-5.1 \
  --verifier_model deepseek-v4-pro \
  --verifier_reasoning_effort max \
  --prompt_profile e73_e57_recall_first_exactness_last \
  --output_protocol xml_tags \
  --gen_max_tokens 8192 \
  --verify_max_tokens 1800 \
  --timeout 420 \
  --output_dir outputs/stage2_strategy_cot_e73/e73_e57_recall_first_exactness_last_glm51_3k_20260614
