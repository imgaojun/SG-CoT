#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
CONFIG_DIR="configs/generated/stage2_confirmation"
DATA_DIR="/workspace/project/data/stage2_confirmation_e121"
HOST_DATA_DIR="${PROJECT_ROOT}/data/stage2_confirmation_e121"
OUTPUT_ROOT="/workspace/project/outputs/stage2_confirmation_e121"
HOST_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/stage2_confirmation_e121"
PREFIX="richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle"
POOLED_PREFIX="richere_v2confirm_pooled15_strict_oracle_mixed_noise_top10_shuffle"
PROTOCOL="balanced-subtype-v2-confirmation"
HELDOUT="/workspace/project/data/processed/type_holdout/richere-en/${PROTOCOL}/split1/unseen_types.json"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
GEN_RUN="e121_autocluster_confirmation"
GEN_OUTPUT="${OUTPUT_ROOT}/generation/e121c_autocluster"
GEN_TRAIN="${DATA_DIR}/${PREFIX}_${GEN_RUN}_thinking_evidence_cot_train_pos.jsonl"
SGCOT_REFERENCE="${DATA_DIR}/${PREFIX}_sgcot_target_train_pos.jsonl"
SGCOT_TRAIN_NAME="${PREFIX}_sgcot_autocluster_train_pos"
SGCOT_TRAIN="${DATA_DIR}/${SGCOT_TRAIN_NAME}.jsonl"
FREEZE_MANIFEST="${OUTPUT_ROOT}/freeze/e121_frozen_inputs.json"
HOST_FREEZE_MANIFEST="${HOST_OUTPUT_ROOT}/freeze/e121_frozen_inputs.json"
LOG_DIR="${OUTPUT_ROOT}/logs"

docker_common() {
  docker run --rm --user root --ipc host --shm-size 16g \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -v "${LF_ROOT}/cache/torch_extensions:/workspace/.cache/torch_extensions" \
    -v "${LF_ROOT}/logs:/workspace/logs" \
    -e LITELLM_API_KEY \
    -e LLM_API_KEY \
    -e OPENAI_API_KEY \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    -e HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    -e TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    -w /workspace/project "$@"
}

release_owned_label_service() {
  local gpu="$1"
  local state_dir="/mnt/disk/gaojun/tmp/gpu-label-service"
  local metadata="${state_dir}/gpu${gpu}.json"
  local pid_file="${state_dir}/gpu${gpu}.pid"
  [[ -f "${metadata}" && -f "${pid_file}" ]] || return 0
  local owner pid recorded_pid cmdline
  owner="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("owner", ""))' "${metadata}")"
  pid="$(tr -d '[:space:]' < "${pid_file}")"
  recorded_pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid", ""))' "${metadata}")"
  [[ "${owner}" == "gaojun" && "${pid}" == "${recorded_pid}" && "${pid}" =~ ^[0-9]+$ ]] || return 0
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "${cmdline}" == *gpu-label-service* && "${cmdline}" == *"gpu${gpu}"* ]]; then
    docker rm -f "gpu-label-service-gaojun-gpu${gpu}" >/dev/null 2>&1 || kill "${pid}"
  fi
}

assert_gpu_idle() {
  local gpu="$1"
  release_owned_label_service "${gpu}"
  local used util processes
  IFS=',' read -r used util < <(
    nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' '
  )
  processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "${processes}" || "${used}" -gt 1024 || "${util}" -gt 5 ]]; then
    echo "GPU ${gpu} is not idle: memory=${used} MiB utilization=${util}% processes=${processes:-none}" >&2
    return 8
  fi
}

verify_freeze() {
  [[ -f "${HOST_FREEZE_MANIFEST}" ]] || {
    echo "missing E121 freeze manifest: ${HOST_FREEZE_MANIFEST}" >&2
    return 9
  }
  python3 scripts/e121_freeze_manifest_20260712.py \
    --mode verify \
    --root "${PROJECT_ROOT}" \
    --manifest "${HOST_FREEZE_MANIFEST}"
}

config_path() {
  local method="$1" seed="$2"
  case "${method}" in
    direct) echo "${CONFIG_DIR}/e121b_direct_surface_seed${seed}.yaml" ;;
    sgcot) echo "${CONFIG_DIR}/e121d_auto_sgcot_seed${seed}.yaml" ;;
    *) return 2 ;;
  esac
}

model_path() {
  local method="$1" seed="$2"
  case "${method}" in
    direct) echo "${OUTPUT_ROOT}/runs/e121b_direct_surface_seed${seed}" ;;
    sgcot) echo "${OUTPUT_ROOT}/runs/e121d_auto_sgcot_seed${seed}" ;;
    *) return 2 ;;
  esac
}

eval_data() {
  local method="$1" split="$2" suffix
  suffix="direct_surface"
  [[ "${method}" == "sgcot" ]] && suffix="sgcot_target"
  case "${split}" in
    test_seen|test_unseen) echo "${DATA_DIR}/${PREFIX}_${suffix}_${split}_pos.jsonl" ;;
    pooled_unseen) echo "${DATA_DIR}/${POOLED_PREFIX}_${suffix}_test_unseen_pos.jsonl" ;;
    *) return 2 ;;
  esac
}

eval_output() {
  local method="$1" seed="$2" split="$3" run
  run="e121b_direct_surface_seed${seed}"
  [[ "${method}" == "sgcot" ]] && run="e121d_auto_sgcot_seed${seed}"
  echo "${OUTPUT_ROOT}/eval/${run}/${split}"
}

run_generation() {
  local limit="$1" output="$2" run_name="$3" adaptive_dir="$4" config_dir="$5"
  local extra=()
  local generation_base_url="${GEN_BASE_URL:-${LITELLM_BASE_URL:-}}"
  if [[ -n "${generation_base_url}" ]]; then
    extra+=(--base_url "${generation_base_url}")
  fi
  if [[ "${GEN_RETRY_REJECTED:-0}" == "1" ]]; then
    extra+=(--retry_rejected)
  fi
  docker_common "${IMAGE}" python scripts/generate_strategy_variants_cot_e47_20260606.py \
    --run_name "${run_name}" \
    --limit "${limit}" \
    --seed 1111 \
    --workers "${GEN_WORKERS:-8}" \
    --model deepseek-v4-pro \
    --verifier_model deepseek-v4-pro \
    --verifier_reasoning_effort max \
    --prompt_profile e95_trigger_locked_autocluster \
    --repair_profile strict_full \
    --output_protocol xml_tags \
    --sampled_rows_path "${SGCOT_REFERENCE}" \
    --sampled_rows_mode priority_sample \
    --output_dir "${output}" \
    --formal_data_dir "${DATA_DIR}" \
    --data_prefix "${PREFIX}_sgcot_target" \
    --adaptive_prefix "${PREFIX}" \
    --adaptive_data_dir "${adaptive_dir}" \
    --config_dir "${config_dir}" \
    --train_dataset_dir "${DATA_DIR}" \
    --run_prefix e121_richere_v2confirm_qwen3_4b \
    --warm_start "${OUTPUT_ROOT}/runs/e121b_direct_surface_seed42" \
    --auto_cluster_map_path /workspace/project/data/schema/richere-en.auto_cluster_map.json \
    --max_attempts "${GEN_MAX_ATTEMPTS:-3}" \
    "${extra[@]}"
}

case "${1:-}" in
  build-data)
    docker_common "${IMAGE}" bash -lc \
      "PROJECT_ROOT=/workspace/project bash scripts/build_e121_confirmation_data_20260712.sh"
    docker_common "${IMAGE}" python scripts/audit_sft_dataset_lengths_20260712.py \
      --input_jsonl "${DATA_DIR}/${PREFIX}_direct_surface_train_pos.jsonl" \
      --model_path "${BASE_MODEL}" \
      --cutoff_len 1536 \
      --output_json "${DATA_DIR}/e121a_direct_surface_length_audit.json" \
      --require_all_fit
    ;;
  generate-smoke)
    [[ -n "${LITELLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}" ]] || {
      echo "a LiteLLM key is required via LITELLM_API_KEY, LLM_API_KEY, or OPENAI_API_KEY" >&2
      exit 10
    }
    run_generation 2 "${GEN_OUTPUT}_smoke2" "${GEN_RUN}_smoke2" \
      "${OUTPUT_ROOT}/generation/smoke_data" "${OUTPUT_ROOT}/generation/smoke_configs"
    python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["accepted"] == 2, s' \
      "${HOST_OUTPUT_ROOT}/generation/e121c_autocluster_smoke2/e47_summary.json"
    ;;
  generate)
    [[ -n "${LITELLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}" ]] || {
      echo "a LiteLLM key is required via LITELLM_API_KEY, LLM_API_KEY, or OPENAI_API_KEY" >&2
      exit 10
    }
    run_generation 1500 "${GEN_OUTPUT}" "${GEN_RUN}" "${DATA_DIR}" "/workspace/project/${CONFIG_DIR}/generated"
    docker_common "${IMAGE}" python scripts/normalize_strict_sgcot_dataset_20260712.py \
      --generated_jsonl "${GEN_TRAIN}" \
      --reference_surface_jsonl "${SGCOT_REFERENCE}" \
      --output_jsonl "${SGCOT_TRAIN}" \
      --dataset_name "${SGCOT_TRAIN_NAME}" \
      --dataset_info "${DATA_DIR}/dataset_info.json" \
      --heldout_types_json "${HELDOUT}" \
      --require_zero_leaks \
      --min_rows 1400 \
      --model_path "${BASE_MODEL}" \
      --cutoff_len 2048
    docker_common "${IMAGE}" python scripts/validate_strict_unseen_dataset_20260712.py \
      --input_jsonl "${SGCOT_TRAIN}" \
      --heldout_types_json "${HELDOUT}" \
      --output_json "${GEN_OUTPUT}/normalized_trace_audit.json" \
      --require_zero_leaks \
      --require_exact_surface_recovery
    ;;
  freeze)
    [[ -f "${HOST_DATA_DIR}/${SGCOT_TRAIN_NAME}.jsonl" ]] || {
      echo "normalized SG-CoT data is required before freezing" >&2
      exit 11
    }
    python3 scripts/e121_freeze_manifest_20260712.py \
      --mode build \
      --root "${PROJECT_ROOT}" \
      --manifest "${HOST_FREEZE_MANIFEST}" \
      --path configs/seen_unseen_type_holdout_protocols.json \
      --path data/schema/richere-en.auto_cluster_map.json \
      --glob 'configs/generated/stage2_confirmation/*' \
      --glob 'data/processed/type_holdout/richere-en/balanced-subtype-v2-confirmation/split*/*.json' \
      --glob 'data/stage2_confirmation_e121/*.json' \
      --glob 'data/stage2_confirmation_e121/*.jsonl' \
      --glob 'outputs/stage2_confirmation_e121/generation/e121c_autocluster/*' \
      --path scripts/build_e121_type_holdout_20260712.py \
      --path scripts/build_e121_confirmation_data_20260712.sh \
      --path scripts/build_e121_pooled_dataset_20260712.py \
      --path scripts/audit_e121_confirmation_data_20260712.py \
      --path scripts/preflight_e121_generation_20260712.py \
      --path scripts/e121_freeze_manifest_20260712.py \
      --path scripts/compare_e121_confirmation_n3_20260712.py \
      --path scripts/compare_strict_n3_gate_20260712.py \
      --path scripts/compare_preference_run_gate_20260712.py \
      --path scripts/run_e121_strict_confirmation_20260712.sh \
      --path scripts/build_surface_evidence_dataset_20260712.py \
      --path scripts/audit_sft_dataset_lengths_20260712.py \
      --path scripts/normalize_strict_sgcot_dataset_20260712.py \
      --path scripts/validate_strict_unseen_dataset_20260712.py \
      --path scripts/generate_strategy_variants_cot_e47_20260606.py \
      --path scripts/generate_evidence_cot_e40_20260604.py \
      --path scripts/generate_strategy_natural_cot_e37_20260604.py \
      --path scripts/build_auto_cluster_map_20260702.py \
      --path scripts/rescore_surface_predictions_20260712.py \
      --path src/data_preprocessing/type_holdout/generate_type_holdout.py \
      --path src/stage2_data/build_formal_stage2_dataset.py \
      --path src/stage2_preference/reasoning_preference.py \
      --path src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py
    verify_freeze
    ;;
  train-direct|train-sgcot)
    method="${1#train-}"
    seed="${2:?seed 42, 8322, or 8333 required}"
    gpu="${3:?GPU index required}"
    [[ "${seed}" == "42" || "${seed}" == "8322" || "${seed}" == "8333" ]] || exit 2
    assert_gpu_idle "${gpu}"
    config="$(config_path "${method}" "${seed}")"
    output="$(model_path "${method}" "${seed}")"
    [[ ! -e "${PROJECT_ROOT}${output#/workspace/project}" ]] || {
      echo "refusing to reuse output directory: ${output}" >&2
      exit 12
    }
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "mkdir -p '${LOG_DIR}'; FORCE_TORCHRUN=1 llamafactory-cli train '${config}' 2>&1 | tee '${LOG_DIR}/${method}_seed${seed}.log'"
    ;;
  train-direct-smoke|train-sgcot-smoke)
    method="${1#train-}"
    method="${method%-smoke}"
    gpu="${2:?GPU index required}"
    assert_gpu_idle "${gpu}"
    config="${CONFIG_DIR}/e121b0_direct_surface_smoke16.yaml"
    [[ "${method}" == "sgcot" ]] && config="${CONFIG_DIR}/e121d0_auto_sgcot_smoke16.yaml"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "mkdir -p '${LOG_DIR}'; FORCE_TORCHRUN=1 llamafactory-cli train '${config}' 2>&1 | tee '${LOG_DIR}/${method}_smoke16.log'"
    ;;
  eval)
    method="${2:?direct or sgcot required}"
    seed="${3:?seed required}"
    split="${4:?test_seen, test_unseen, or pooled_unseen required}"
    gpu="${5:?GPU index required}"
    verify_freeze
    assert_gpu_idle "${gpu}"
    model="$(model_path "${method}" "${seed}")"
    data="$(eval_data "${method}" "${split}")"
    output="$(eval_output "${method}" "${seed}" "${split}")"
    [[ ! -e "${PROJECT_ROOT}${output#/workspace/project}" ]] || {
      echo "refusing to reuse evaluation directory: ${output}" >&2
      exit 12
    }
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "mkdir -p '${LOG_DIR}'; python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' --adapter_path '${model}' --eval_jsonl '${data}' \
        --output_dir '${output}' --batch_size 4 --max_new_tokens 1024 --temperature 0.0 \
        2>&1 | tee '${LOG_DIR}/eval_${method}_seed${seed}_${split}.log'"
    ;;
  compare-n3)
    verify_freeze
    python3 scripts/compare_e121_confirmation_n3_20260712.py \
      --baseline_seen \
        "${HOST_OUTPUT_ROOT}/eval/e121b_direct_surface_seed42/test_seen" \
        "${HOST_OUTPUT_ROOT}/eval/e121b_direct_surface_seed8322/test_seen" \
        "${HOST_OUTPUT_ROOT}/eval/e121b_direct_surface_seed8333/test_seen" \
      --baseline_unseen \
        "${HOST_OUTPUT_ROOT}/eval/e121b_direct_surface_seed42/test_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e121b_direct_surface_seed8322/test_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e121b_direct_surface_seed8333/test_unseen" \
      --baseline_pooled \
        "${HOST_OUTPUT_ROOT}/eval/e121b_direct_surface_seed42/pooled_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e121b_direct_surface_seed8322/pooled_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e121b_direct_surface_seed8333/pooled_unseen" \
      --candidate_seen \
        "${HOST_OUTPUT_ROOT}/eval/e121d_auto_sgcot_seed42/test_seen" \
        "${HOST_OUTPUT_ROOT}/eval/e121d_auto_sgcot_seed8322/test_seen" \
        "${HOST_OUTPUT_ROOT}/eval/e121d_auto_sgcot_seed8333/test_seen" \
      --candidate_unseen \
        "${HOST_OUTPUT_ROOT}/eval/e121d_auto_sgcot_seed42/test_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e121d_auto_sgcot_seed8322/test_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e121d_auto_sgcot_seed8333/test_unseen" \
      --candidate_pooled \
        "${HOST_OUTPUT_ROOT}/eval/e121d_auto_sgcot_seed42/pooled_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e121d_auto_sgcot_seed8322/pooled_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e121d_auto_sgcot_seed8333/pooled_unseen" \
      --heldout_types_json "${PROJECT_ROOT}/data/processed/type_holdout/richere-en/${PROTOCOL}/split1/unseen_types.json" \
      --gate_config "${PROJECT_ROOT}/${CONFIG_DIR}/e121e_confirmation_gate.json" \
      --output_dir "${HOST_OUTPUT_ROOT}/analysis/e121e_n3_confirmation" \
      --bootstrap_samples 10000 \
      --bootstrap_seed 20260712
    ;;
  *)
    echo "usage: $0 build-data | generate-smoke | generate | freeze | train-{direct|sgcot} <seed> <gpu> | train-{direct|sgcot}-smoke <gpu> | eval <direct|sgcot> <seed> <test_seen|test_unseen|pooled_unseen> <gpu> | compare-n3" >&2
    exit 2
    ;;
esac
