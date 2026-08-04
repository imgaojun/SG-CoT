#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <reason_checkpoint_tag>" >&2
  echo "example: $0 checkpoint-771" >&2
  exit 2
fi

REASON_CKPT="$1"
REASON_BRANCH="sampled_reason_expert_forcedreason_from_noaux_20260517"
RUN_ID="${REASON_BRANCH}_${REASON_CKPT}"
ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
LABEL_SOURCE="sampled_counterfactual_utility_k8_${REASON_CKPT}"
SAMPLE_ROOT="outputs/stage2_modular_dualexpert/sampled_counterfactual_utility_20260517/${RUN_ID}"
LABEL_DIR="data/stage2_adaptive_datasets/labels"
REPORT_JSON="reports/artifacts/2026-05-17_stage2_sampled_counterfactual_utility_k8_label_diagnostic_${REASON_CKPT}.json"
REPORT_MD="reports/2026-05-17_stage2_sampled_counterfactual_utility_k8_label_diagnostic_${REASON_CKPT}.md"

mkdir -p "${LABEL_DIR}" reports/artifacts

for split in train dev_seen; do
  direct_root="${SAMPLE_ROOT}/${split}/direct"
  reason_root="${SAMPLE_ROOT}/${split}/reason"
  output_jsonl="${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_${split}_labels.jsonl"
  summary_json="${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_${split}_labels.summary.json"
  python3 src/stage2_analysis/build_sampled_counterfactual_utility_labels.py \
    --direct_root "${direct_root}" \
    --reason_root "${reason_root}" \
    --output_jsonl "${output_jsonl}" \
    --summary_json "${summary_json}" \
    --label_source "${LABEL_SOURCE}"
done

python3 scripts/summarize_sampled_counterfactual_utility_labels_20260517.py \
  --train_summary "${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_train_labels.summary.json" \
  --dev_summary "${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_dev_seen_labels.summary.json" \
  --output_json "${REPORT_JSON}" \
  --output_md "${REPORT_MD}"
