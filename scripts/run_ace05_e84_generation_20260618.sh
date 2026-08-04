#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk/gaojun/research/progressive-ee

LIMIT="${1:-1500}"
PROMPT_PROFILE="e84_trigger_locked_no_arbitration"
SEED=8401

RUN_PREFIX="ace05_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
ADAPTIVE_PREFIX="ace05_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="ace05_balanced_split1_oracle_mixed_noise_top10_shuffle"
WARM_START="/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/ace05_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/checkpoint-2704"
ACE05_POOL="data/stage2_formal_datasets/${DATA_PREFIX}_train_pos.jsonl"

if [ "${LIMIT}" -lt 50 ]; then
  RUN_NAME="e84_ace05_no_arbitration_glm51_smoke${LIMIT}"
  OUT_DIR="outputs/stage2_strategy_cot_e84/ace05_e84_no_arbitration_glm51_smoke${LIMIT}_20260618"
else
  RUN_NAME="e84_ace05_no_arbitration_glm51_full1500"
  OUT_DIR="outputs/stage2_strategy_cot_e84/ace05_e84_no_arbitration_glm51_full1500_20260618"
fi
MANIFEST="${OUT_DIR}/e76_manifest_rows.jsonl"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

python3 scripts/build_e76_contrastive_manifest_20260614.py --input "${ACE05_POOL}" --output_dir "${OUT_DIR}" --seed "${SEED}"

python3 scripts/generate_strategy_variants_cot_e47_20260606.py \
  --run_name "${RUN_NAME}" --limit "${LIMIT}" --seed "${SEED}" --workers 16 \
  --base_url ${LLM_BASE_URL} --model glm-5.1 \
  --verifier_model deepseek-v4-pro --verifier_reasoning_effort max \
  --prompt_profile "${PROMPT_PROFILE}" --output_protocol xml_tags \
  --gen_max_tokens 8192 --verify_max_tokens 1800 --timeout 420 \
  --sampled_rows_path "${MANIFEST}" --output_dir "${OUT_DIR}" \
  --run_prefix "${RUN_PREFIX}" --adaptive_prefix "${ADAPTIVE_PREFIX}" \
  --data_prefix "${DATA_PREFIX}" --warm_start "${WARM_START}"
