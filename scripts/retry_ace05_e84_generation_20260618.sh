#!/usr/bin/env bash
# E84 ACE05 no-arbitration generation: rejected-only retry loop.
# Re-runs the python generator with --retry_rejected (reuse_existing keeps accepted rows,
# only failed/rejected rows are re-attempted). Loops until the accept gate or max rounds,
# tolerating the GLM endpoint's intermittent 502s.
set -uo pipefail
cd /mnt/disk/gaojun/research/progressive-ee

LIMIT=1500
GATE="${1:-1100}"          # ACE05 accept gate
MAX_ROUNDS="${2:-40}"
SLEEP_BETWEEN="${3:-90}"    # seconds between rounds to let the endpoint breathe

PROMPT_PROFILE="e84_trigger_locked_no_arbitration"
SEED=8401
RUN_NAME="e84_ace05_no_arbitration_glm51_full1500"
RUN_PREFIX="ace05_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
ADAPTIVE_PREFIX="ace05_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="ace05_balanced_split1_oracle_mixed_noise_top10_shuffle"
WARM_START="/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/ace05_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/checkpoint-2704"
OUT_DIR="outputs/stage2_strategy_cot_e84/ace05_e84_no_arbitration_glm51_full1500_20260618"
MANIFEST="${OUT_DIR}/e76_manifest_rows.jsonl"
SUMMARY="${OUT_DIR}/e47_summary.json"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

acc() { python3 -c "import json,sys; print(json.load(open('${SUMMARY}'))['accepted'])" 2>/dev/null || echo 0; }

for r in $(seq 1 "${MAX_ROUNDS}"); do
  cur=$(acc)
  echo "=== [retry round ${r}] start $(date '+%m-%d %H:%M')  accepted=${cur}/${GATE} ==="
  if [ "${cur}" -ge "${GATE}" ]; then
    echo "=== GATE reached (${cur} >= ${GATE}); stop. ==="
    break
  fi
  python3 scripts/generate_strategy_variants_cot_e47_20260606.py \
    --run_name "${RUN_NAME}" --limit "${LIMIT}" --seed "${SEED}" --workers 12 \
    --base_url ${LLM_BASE_URL} --model glm-5.1 \
    --verifier_model deepseek-v4-pro --verifier_reasoning_effort max \
    --prompt_profile "${PROMPT_PROFILE}" --output_protocol xml_tags \
    --gen_max_tokens 8192 --verify_max_tokens 1800 --timeout 420 \
    --sampled_rows_path "${MANIFEST}" --output_dir "${OUT_DIR}" \
    --run_prefix "${RUN_PREFIX}" --adaptive_prefix "${ADAPTIVE_PREFIX}" \
    --data_prefix "${DATA_PREFIX}" --warm_start "${WARM_START}" \
    --retry_rejected 2>&1 | tail -3
  new=$(acc)
  echo "=== [retry round ${r}] end $(date '+%m-%d %H:%M')  accepted=${cur} -> ${new} ==="
  if [ "${new}" -ge "${GATE}" ]; then
    echo "=== GATE reached (${new} >= ${GATE}); stop. ==="
    break
  fi
  sleep "${SLEEP_BETWEEN}"
done
echo "=== retry loop done; final accepted=$(acc) ==="
