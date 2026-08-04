#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/disk/gaojun/research/progressive-ee}"
cd "${ROOT}"

DATA_DIR="data/stage2_strict_datasets"
PREFIX="richere_balanced_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle"
HELDOUT="data/processed/type_holdout/richere-en/balanced-subtype-v1/split1/unseen_types.json"

for part in train dev_seen test_seen test_unseen; do
  candidate_scope="all"
  if [[ "${part}" == "train" ]]; then
    candidate_scope="seen_only"
  fi
  numeric_name="${PREFIX}_numeric_${part}_pos"
  python3 src/stage2_data/build_formal_stage2_dataset.py \
    --data_root data/processed/type_holdout \
    --dataset richere-en \
    --protocol balanced-subtype-v1 \
    --split split1 \
    --part "${part}" \
    --schema_path data/schema/richere-en.event_schema.json \
    --candidate_source oracle_mixed_noise \
    --candidate_scope "${candidate_scope}" \
    --top_k 10 \
    --candidate_order_mode shuffle \
    --selection_mode positive_only \
    --seed 13 \
    --dataset_dir "${DATA_DIR}" \
    --dataset_name "${numeric_name}"

  direct_name="${PREFIX}_direct_surface_${part}_pos"
  direct_extra=()
  if [[ "${part}" == "train" ]]; then
    direct_extra+=(--require_no_heldout_leak)
  fi
  python3 scripts/build_surface_evidence_dataset_20260712.py \
    --input_jsonl "${DATA_DIR}/${numeric_name}.jsonl" \
    --output_jsonl "${DATA_DIR}/${direct_name}.jsonl" \
    --mode direct \
    --dataset_name "${direct_name}" \
    --dataset_info "${DATA_DIR}/dataset_info.json" \
    --heldout_types_json "${HELDOUT}" \
    "${direct_extra[@]}"

  sgcot_name="${PREFIX}_sgcot_target_${part}_pos"
  sgcot_extra=()
  if [[ "${part}" == "train" ]]; then
    sgcot_extra+=(--require_no_heldout_leak)
  fi
  python3 scripts/build_surface_evidence_dataset_20260712.py \
    --input_jsonl "${DATA_DIR}/${numeric_name}.jsonl" \
    --output_jsonl "${DATA_DIR}/${sgcot_name}.jsonl" \
    --mode sgcot \
    --dataset_name "${sgcot_name}" \
    --dataset_info "${DATA_DIR}/dataset_info.json" \
    --heldout_types_json "${HELDOUT}" \
    "${sgcot_extra[@]}"
done

python3 scripts/validate_strict_unseen_dataset_20260712.py \
  --input_jsonl \
    "${DATA_DIR}/${PREFIX}_numeric_train_pos.jsonl" \
    "${DATA_DIR}/${PREFIX}_direct_surface_train_pos.jsonl" \
    "${DATA_DIR}/${PREFIX}_sgcot_target_train_pos.jsonl" \
  --heldout_types_json "${HELDOUT}" \
  --output_json "${DATA_DIR}/e111a_strict_train_audit.json" \
  --require_zero_leaks \
  --require_exact_surface_recovery
