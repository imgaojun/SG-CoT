#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/disk/gaojun/research/progressive-ee}"
cd "${ROOT}"

DATA_DIR="data/stage2_confirmation_e121"
PROTOCOL="balanced-subtype-v2-confirmation"
HOLDOUT_ROOT="data/processed/type_holdout/richere-en/${PROTOCOL}"
HELDOUT="${HOLDOUT_ROOT}/split1/unseen_types.json"
SEEN="${HOLDOUT_ROOT}/split1/seen_types.json"
SCHEMA="data/schema/richere-en.event_schema.json"

python3 scripts/build_e121_type_holdout_20260712.py \
  --audit_output "${DATA_DIR}/e121a_type_selection_audit.json"

build_part() {
  local split="$1"
  local part="$2"
  local candidate_scope="all"
  if [[ "${split}" == "split1" && "${part}" == "train" ]]; then
    candidate_scope="seen_only"
  fi
  local prefix="richere_v2confirm_${split}_strict_seenonly_oracle_mixed_noise_top10_shuffle"
  local numeric_name="${prefix}_numeric_${part}_pos"

  python3 src/stage2_data/build_formal_stage2_dataset.py \
    --data_root data/processed/type_holdout \
    --dataset richere-en \
    --protocol "${PROTOCOL}" \
    --split "${split}" \
    --part "${part}" \
    --schema_path "${SCHEMA}" \
    --candidate_source oracle_mixed_noise \
    --candidate_scope "${candidate_scope}" \
    --top_k 10 \
    --candidate_order_mode shuffle \
    --selection_mode positive_only \
    --seed 13 \
    --dataset_dir "${DATA_DIR}" \
    --dataset_name "${numeric_name}"

  for mode in direct sgcot; do
    local suffix="direct_surface"
    if [[ "${mode}" == "sgcot" ]]; then
      suffix="sgcot_target"
    fi
    local output_name="${prefix}_${suffix}_${part}_pos"
    local extra=()
    if [[ "${split}" == "split1" && "${part}" == "train" ]]; then
      extra+=(--require_no_heldout_leak)
    fi
    python3 scripts/build_surface_evidence_dataset_20260712.py \
      --input_jsonl "${DATA_DIR}/${numeric_name}.jsonl" \
      --output_jsonl "${DATA_DIR}/${output_name}.jsonl" \
      --mode "${mode}" \
      --dataset_name "${output_name}" \
      --dataset_info "${DATA_DIR}/dataset_info.json" \
      --heldout_types_json "${HELDOUT}" \
      "${extra[@]}"
  done
}

for part in train dev_seen test_seen test_unseen; do
  build_part split1 "${part}"
done
for split in split2 split3 split4 split5; do
  build_part "${split}" test_unseen
done

DIRECT_POOLED="richere_v2confirm_pooled15_strict_oracle_mixed_noise_top10_shuffle_direct_surface_test_unseen_pos"
SGCOT_POOLED="richere_v2confirm_pooled15_strict_oracle_mixed_noise_top10_shuffle_sgcot_target_test_unseen_pos"
direct_inputs=()
sgcot_inputs=()
for split in split1 split2 split3 split4 split5; do
  prefix="richere_v2confirm_${split}_strict_seenonly_oracle_mixed_noise_top10_shuffle"
  direct_inputs+=("${DATA_DIR}/${prefix}_direct_surface_test_unseen_pos.jsonl")
  sgcot_inputs+=("${DATA_DIR}/${prefix}_sgcot_target_test_unseen_pos.jsonl")
done

python3 scripts/build_e121_pooled_dataset_20260712.py \
  --input_jsonl "${direct_inputs[@]}" \
  --output_jsonl "${DATA_DIR}/${DIRECT_POOLED}.jsonl" \
  --dataset_name "${DIRECT_POOLED}" \
  --dataset_info "${DATA_DIR}/dataset_info.json" \
  --heldout_types_json "${HELDOUT}" \
  --expected_raw_rows 332 \
  --expected_unique_rows 272 \
  --expected_event_mentions 317

python3 scripts/build_e121_pooled_dataset_20260712.py \
  --input_jsonl "${sgcot_inputs[@]}" \
  --output_jsonl "${DATA_DIR}/${SGCOT_POOLED}.jsonl" \
  --dataset_name "${SGCOT_POOLED}" \
  --dataset_info "${DATA_DIR}/dataset_info.json" \
  --heldout_types_json "${HELDOUT}" \
  --expected_raw_rows 332 \
  --expected_unique_rows 272 \
  --expected_event_mentions 317

SPLIT1_PREFIX="richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle"
python3 scripts/validate_strict_unseen_dataset_20260712.py \
  --input_jsonl \
    "${DATA_DIR}/${SPLIT1_PREFIX}_numeric_train_pos.jsonl" \
    "${DATA_DIR}/${SPLIT1_PREFIX}_direct_surface_train_pos.jsonl" \
    "${DATA_DIR}/${SPLIT1_PREFIX}_sgcot_target_train_pos.jsonl" \
  --heldout_types_json "${HELDOUT}" \
  --output_json "${DATA_DIR}/e121a_strict_train_leakage_audit.json" \
  --require_zero_leaks \
  --require_exact_surface_recovery

direct_pairs=(
  "${DATA_DIR}/${SPLIT1_PREFIX}_direct_surface_train_pos.jsonl"
  "${DATA_DIR}/${SPLIT1_PREFIX}_direct_surface_dev_seen_pos.jsonl"
  "${DATA_DIR}/${SPLIT1_PREFIX}_direct_surface_test_seen_pos.jsonl"
  "${DATA_DIR}/${SPLIT1_PREFIX}_direct_surface_test_unseen_pos.jsonl"
  "${DATA_DIR}/${DIRECT_POOLED}.jsonl"
)
sgcot_pairs=(
  "${DATA_DIR}/${SPLIT1_PREFIX}_sgcot_target_train_pos.jsonl"
  "${DATA_DIR}/${SPLIT1_PREFIX}_sgcot_target_dev_seen_pos.jsonl"
  "${DATA_DIR}/${SPLIT1_PREFIX}_sgcot_target_test_seen_pos.jsonl"
  "${DATA_DIR}/${SPLIT1_PREFIX}_sgcot_target_test_unseen_pos.jsonl"
  "${DATA_DIR}/${SGCOT_POOLED}.jsonl"
)
python3 scripts/audit_e121_confirmation_data_20260712.py \
  --direct_jsonl "${direct_pairs[@]}" \
  --sgcot_jsonl "${sgcot_pairs[@]}" \
  --heldout_types_json "${HELDOUT}" \
  --seen_types_json "${SEEN}" \
  --output_json "${DATA_DIR}/e121a_dataset_audit.json"

python3 scripts/preflight_e121_generation_20260712.py \
  --input_jsonl "${DATA_DIR}/${SPLIT1_PREFIX}_sgcot_target_train_pos.jsonl" \
  --heldout_types_json "${HELDOUT}" \
  --seen_types_json "${SEEN}" \
  --auto_cluster_map data/schema/richere-en.auto_cluster_map.json \
  --output_json "${DATA_DIR}/e121c_generation_preflight.json" \
  --limit 1500 \
  --seed 1111 \
  --run_name e121_autocluster_confirmation
