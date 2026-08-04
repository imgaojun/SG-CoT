#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk/gaojun/research/progressive-ee

# Optional first arg: sample limit (default 260 = full WikiEvents train pool). <50 -> smoke.
LIMIT="${1:-260}"

PROMPT_PROFILE="e83_trigger_locked_schema_driven"
SEED=8301

# WikiEvents (KAIROS) dataset-family overrides
RUN_PREFIX="wikievents_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
ADAPTIVE_PREFIX="wikievents_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="wikievents_split1_oracle_mixed_noise_top10_shuffle"
WARM_START="/workspace/project/outputs/stage2_full_sft_runs_wikievents/wikievents_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full"
POOL="data/stage2_formal_datasets/${DATA_PREFIX}_train_pos.jsonl"

if [ "${LIMIT}" -lt 50 ]; then
  RUN_NAME="e83_wikievents_trigger_locked_schema_driven_glm51_smoke${LIMIT}"
  OUT_DIR="outputs/stage2_strategy_cot_e83/wikievents_e83_trigger_locked_schema_driven_glm51_smoke${LIMIT}_20260709"
else
  RUN_NAME="e83_wikievents_trigger_locked_schema_driven_glm51_full"
  OUT_DIR="outputs/stage2_strategy_cot_e83/wikievents_e83_trigger_locked_schema_driven_glm51_full_20260709"
fi
MANIFEST="${OUT_DIR}/e76_manifest_rows.jsonl"
mkdir -p "${OUT_DIR}"

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

# Build the contrastive manifest from the WikiEvents train pool (RichERE-tuned buckets degrade gracefully -> selects all).
python3 scripts/build_e76_contrastive_manifest_20260614.py \
  --input "${POOL}" \
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
  --gen_max_tokens 16384 \
  --verify_max_tokens 1800 \
  --timeout 600 \
  --sampled_rows_path "${MANIFEST}" \
  --output_dir "${OUT_DIR}" \
  --run_prefix "${RUN_PREFIX}" \
  --adaptive_prefix "${ADAPTIVE_PREFIX}" \
  --data_prefix "${DATA_PREFIX}" \
  --warm_start "${WARM_START}" \
  ${RETRY_FLAG:-}
