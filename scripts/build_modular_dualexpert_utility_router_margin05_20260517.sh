#!/usr/bin/env bash
set -euo pipefail

DIRECT_PREFIX="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_DIR="data/stage2_adaptive_datasets"
LABEL_DIR="${DATA_DIR}/labels"
SCHEMA="data/schema/richere-en.event_schema.json"

BRANCH="modular_d1930_r2058_utility_margin05_routecls_noauxwarm_lr2e6_save50"
LABEL_SOURCE="modular_d1930_r2058_utility_margin05"
MARGIN="0.5"

TEACHER_ROOT="outputs/stage2_modular_dualexpert/train_teacher_outputs_d1930_r2058_20260517"
TRAIN_DIRECT="${TEACHER_ROOT}/direct_expert_forced_direct_train/predictions.jsonl"
TRAIN_REASON="${TEACHER_ROOT}/reason_expert_forced_reason_train/predictions.jsonl"

DEV_DIRECT="outputs/stage2_adaptive_runs_user_devpick_frontier/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux_full_forced_direct_dev_seen_max512/checkpoint-1930/predictions.jsonl"
DEV_REASON="outputs/stage2_adaptive_runs_user_devpick_frontier/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux_full_forced_reason_dev_seen_max512/checkpoint-2058/predictions.jsonl"

FORMAL_DIRECT_ROOT="outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_balrouteaux_20260516/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux/checkpoint-1930/forced_direct"
FORMAL_REASON_ROOT="outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux/checkpoint-2058/forced_reason"

build_labels() {
  local split="$1"
  local direct_predictions="$2"
  local reason_predictions="$3"
  local output_jsonl="${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_${split}_labels.jsonl"
  local summary_json="${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_${split}_labels.summary.json"

  if [[ ! -s "${direct_predictions}" || ! -s "${reason_predictions}" ]]; then
    echo "missing paired predictions for ${split}" >&2
    echo "direct: ${direct_predictions}" >&2
    echo "reason: ${reason_predictions}" >&2
    exit 1
  fi

  python3 src/stage2_analysis/build_adaptive_outcome_route_labels.py \
    --forced_direct_predictions "${direct_predictions}" \
    --forced_reason_predictions "${reason_predictions}" \
    --output_jsonl "${output_jsonl}" \
    --summary_json "${summary_json}" \
    --reason_rate_cap 1.0 \
    --margin "${MARGIN}" \
    --label_source "${LABEL_SOURCE}" \
    --miner_checkpoint "D1930_direct_R2058_reason"
}

mkdir -p "${LABEL_DIR}"

build_labels "train" "${TRAIN_DIRECT}" "${TRAIN_REASON}"
build_labels "dev_seen" "${DEV_DIRECT}" "${DEV_REASON}"
build_labels "test" "${FORMAL_DIRECT_ROOT}/test/predictions.jsonl" "${FORMAL_REASON_ROOT}/test/predictions.jsonl"
build_labels "test_seen" "${FORMAL_DIRECT_ROOT}/test_seen/predictions.jsonl" "${FORMAL_REASON_ROOT}/test_seen/predictions.jsonl"
build_labels "test_unseen" "${FORMAL_DIRECT_ROOT}/test_unseen/predictions.jsonl" "${FORMAL_REASON_ROOT}/test_unseen/predictions.jsonl"

python3 src/stage2_cot/build_adaptive_route_reasoning_dataset.py \
  --schema_path "${SCHEMA}" \
  --direct_train_jsonl "${DIRECT_PREFIX}_train_pos.jsonl" \
  --direct_dev_jsonl "${DIRECT_PREFIX}_dev_seen_pos.jsonl" \
  --direct_test_jsonl "${DIRECT_PREFIX}_test_pos.jsonl" \
  --direct_test_seen_jsonl "${DIRECT_PREFIX}_test_seen_pos.jsonl" \
  --direct_test_unseen_jsonl "${DIRECT_PREFIX}_test_unseen_pos.jsonl" \
  --train_label_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_train_labels.jsonl" \
  --dev_label_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_dev_seen_labels.jsonl" \
  --test_label_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_test_labels.jsonl" \
  --test_seen_label_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_test_seen_labels.jsonl" \
  --test_unseen_label_jsonl "${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_test_unseen_labels.jsonl" \
  --dataset_dir "${DATA_DIR}" \
  --train_dataset_name "${ADAPT_PREFIX}_${BRANCH}_train_pos" \
  --dev_dataset_name "${ADAPT_PREFIX}_${BRANCH}_dev_seen_pos" \
  --test_dataset_name "${ADAPT_PREFIX}_${BRANCH}_test_pos" \
  --test_seen_dataset_name "${ADAPT_PREFIX}_${BRANCH}_test_seen_pos" \
  --test_unseen_dataset_name "${ADAPT_PREFIX}_${BRANCH}_test_unseen_pos" \
  --target_style type_role_hint_plan_lite \
  --max_role_checks_per_sample 6 \
  --seed 17 \
  --route_only_train \
  --route_only_eval \
  --route_classifier_prompt \
  --route_reason_oversample 4

python3 - "${DATA_DIR}/${ADAPT_PREFIX}_${BRANCH}_train_pos.meta.json" "${DATA_DIR}/${ADAPT_PREFIX}_${BRANCH}_dev_seen_pos.meta.json" <<'PY'
import json
import sys
from pathlib import Path

payload = []
for raw in sys.argv[1:]:
    path = Path(raw)
    meta = json.loads(path.read_text(encoding="utf-8"))
    audit = meta["audit"]
    bad = {}
    for key, expected in {
        "full_with_final_count": 0,
        "route_only_full_extraction_prompt_count": 0,
        "route_only_rows_with_final": 0,
    }.items():
        if audit.get(key) != expected:
            bad[key] = {"expected": expected, "actual": audit.get(key)}
    if audit.get("route_only_classifier_prompt_count") != audit.get("total_count"):
        bad["route_only_classifier_prompt_count"] = {
            "expected": audit.get("total_count"),
            "actual": audit.get("route_only_classifier_prompt_count"),
        }
    if bad:
        raise SystemExit(json.dumps({"meta": path.as_posix(), "bad": bad, "audit": audit}, indent=2))
    payload.append({"meta": path.as_posix(), "audit": audit})
print(json.dumps(payload, indent=2))
PY

python3 scripts/prepare_modular_dualexpert_utility_router_margin05_20260517.py
