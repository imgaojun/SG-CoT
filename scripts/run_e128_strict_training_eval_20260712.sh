#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
MODEL_ROOT="/mnt/disk/gaojun/models"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
CONFIG_DIR="configs/generated/stage2_confirmation"
OUTPUT_ROOT="/workspace/project/outputs/stage2_confirmation_e128"
HOST_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/stage2_confirmation_e128"
LOG_DIR="${OUTPUT_ROOT}/logs"
FREEZE_MANIFEST="${OUTPUT_ROOT}/freeze/e128_frozen_inputs.json"
HOST_FREEZE_MANIFEST="${HOST_OUTPUT_ROOT}/freeze/e128_frozen_inputs.json"
HOST_SMOKE_MODEL="${HOST_OUTPUT_ROOT}/smoke/e128b0_auto_sgcot_smoke16"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
PREFIX="richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle"
POOLED_PREFIX="richere_v2confirm_pooled15_strict_oracle_mixed_noise_top10_shuffle"
EVAL_DATA_DIR="/workspace/project/data/stage2_confirmation_e121"
PROTOCOL="balanced-subtype-v2-confirmation"
EVAL_CONFIG_REL="${CONFIG_DIR}/e128e_strict_confirmation_eval.json"
EVAL_CONFIG_HOST="${PROJECT_ROOT}/${EVAL_CONFIG_REL}"

eval_setting() {
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); [value := value[key] for key in sys.argv[2:]]; print(value)' \
    "${EVAL_CONFIG_HOST}" "$@"
}

EVAL_BATCH_SIZE="$(eval_setting decode batch_size)"
EVAL_TEMPERATURE="$(eval_setting decode temperature)"
EVAL_MAX_NEW_TOKENS="$(eval_setting decode max_new_tokens)"
BOOTSTRAP_SAMPLES="$(eval_setting bootstrap samples)"
BOOTSTRAP_SEED="$(eval_setting bootstrap seed)"

docker_common() {
  docker run --rm --user root --ipc host --shm-size 16g \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -v "${LF_ROOT}/cache/torch_extensions:/workspace/.cache/torch_extensions" \
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
    echo "missing E128 freeze manifest: ${HOST_FREEZE_MANIFEST}" >&2
    return 9
  }
  python3 scripts/e121_freeze_manifest_20260712.py \
    --mode verify \
    --root "${PROJECT_ROOT}" \
    --manifest "${HOST_FREEZE_MANIFEST}"
}

assert_completed_model() {
  local model_dir="$1"
  python3 scripts/validate_training_artifact_20260712.py \
    --model_dir "${model_dir}"
}

assert_smoke_passed() {
  assert_completed_model "${HOST_SMOKE_MODEL}"
  python3 scripts/validate_training_artifact_20260712.py \
    --model_dir "${HOST_SMOKE_MODEL}" \
    --trainer_state "${HOST_SMOKE_MODEL}/trainer_state.json" \
    --min_global_step 1 \
    --require_finite_step_log
}

sgcot_config() {
  local seed="$1"
  echo "${CONFIG_DIR}/e128c_auto_sgcot_seed${seed}.yaml"
}

direct_model() {
  local seed="$1"
  echo "/workspace/project/outputs/stage2_confirmation_e121/runs/e121b_direct_surface_seed${seed}"
}

sgcot_model() {
  local seed="$1"
  echo "${OUTPUT_ROOT}/runs/e128c_auto_sgcot_seed${seed}"
}

eval_data() {
  local method="$1" split="$2" suffix="direct_surface"
  [[ "${method}" == "sgcot" ]] && suffix="sgcot_target"
  case "${split}" in
    test_seen|test_unseen) echo "${EVAL_DATA_DIR}/${PREFIX}_${suffix}_${split}_pos.jsonl" ;;
    pooled_unseen) echo "${EVAL_DATA_DIR}/${POOLED_PREFIX}_${suffix}_test_unseen_pos.jsonl" ;;
    *) return 2 ;;
  esac
}

eval_output() {
  local method="$1" seed="$2" split="$3"
  echo "${OUTPUT_ROOT}/eval/e128_${method}_seed${seed}/${split}"
}

case "${ACTION}" in
  freeze)
    assert_smoke_passed
    docker_common "${IMAGE}" python scripts/e121_freeze_manifest_20260712.py \
      --mode build \
      --manifest_id e128_frozen_inputs_v1 \
      --root /workspace/project \
      --manifest "${FREEZE_MANIFEST}" \
      --path configs/seen_unseen_type_holdout_protocols.json \
      --path configs/generated/stage2_confirmation/e121a_confirmation_data.json \
      --glob 'configs/generated/stage2_confirmation/e121b_direct_surface_seed*.yaml' \
      --path configs/generated/stage2_confirmation/e121e_confirmation_gate.json \
      --path configs/deepspeed/zero2_optimizer_offload_cpu.json \
      --glob 'configs/generated/stage2_confirmation/e128*.json' \
      --glob 'configs/generated/stage2_confirmation/e128*.yaml' \
      --path data/schema/richere-en.auto_cluster_map.json \
      --path data/stage2_confirmation_e127/dataset_info.json \
      --path data/stage2_confirmation_e127/richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle_sgcot_autocluster_e127a_train_pos.jsonl \
      --path data/stage2_confirmation_e127/richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle_sgcot_autocluster_e127a_train_pos.summary.json \
      --glob 'data/stage2_confirmation_e121/*.json' \
      --glob 'data/stage2_confirmation_e121/*.jsonl' \
      --glob 'data/processed/type_holdout/richere-en/balanced-subtype-v2-confirmation/split*/*.json' \
      --path outputs/stage2_confirmation_e127/e127a_deterministic_exact_zero_leak_full1500/summary.json \
      --path outputs/stage2_confirmation_e127/e127a_deterministic_exact_zero_leak_full1500/selection_manifest.jsonl \
      --path outputs/stage2_confirmation_e127/e127a_deterministic_exact_zero_leak_full1500/excluded_manifest.jsonl \
      --path outputs/stage2_confirmation_e127/e127a_deterministic_exact_zero_leak_full1500/normalized_trace_audit.json \
      --path outputs/stage2_confirmation_e128/e128a_disjoint_core_reasoning_audit100/summary.json \
      --path outputs/stage2_confirmation_e128/e128a_disjoint_core_reasoning_audit100/sampled_rows.jsonl \
      --path reports/2026-07-12_e128_strict_trace_quality.md \
      --path scripts/build_e121_type_holdout_20260712.py \
      --path scripts/build_e121_confirmation_data_20260712.sh \
      --path scripts/build_e121_pooled_dataset_20260712.py \
      --path scripts/audit_e121_confirmation_data_20260712.py \
      --path scripts/build_surface_evidence_dataset_20260712.py \
      --path scripts/audit_sft_dataset_lengths_20260712.py \
      --path scripts/build_e127_deterministic_exact_trace_dataset_20260712.py \
      --path scripts/run_e124c_independent_deepseek_audit_20260712.py \
      --path scripts/run_e128_core_reasoning_audit_20260712.sh \
      --path scripts/run_e128_strict_training_eval_20260712.sh \
      --path scripts/run_e121_strict_confirmation_20260712.sh \
      --path scripts/validate_training_artifact_20260712.py \
      --path scripts/e121_freeze_manifest_20260712.py \
      --path scripts/compare_e121_confirmation_n3_20260712.py \
      --path scripts/compare_strict_n3_gate_20260712.py \
      --path scripts/compare_preference_run_gate_20260712.py \
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
      --path src/stage2_quality_validation/eval_adapter_generation.py \
      --path src/stage2_quality_validation/eval_adaptive_route_generation.py \
      --path src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
      --path src/stage2_quality_validation/generation_diagnostics.py
    verify_freeze
    ;;
  verify-freeze)
    verify_freeze
    ;;
  train-smoke)
    gpu="${2:?GPU index required}"
    assert_gpu_idle "${gpu}"
    assert_completed_model "${PROJECT_ROOT}/outputs/stage2_confirmation_e121/runs/e121b_direct_surface_seed42"
    [[ ! -e "${HOST_OUTPUT_ROOT}/smoke/e128b0_auto_sgcot_smoke16" ]] || {
      echo "refusing to reuse E128 smoke output" >&2
      exit 12
    }
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "mkdir -p '${LOG_DIR}'; FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_DIR}/e128b0_auto_sgcot_smoke16.yaml' 2>&1 | tee '${LOG_DIR}/e128b0_auto_sgcot_smoke16.log'"
    assert_smoke_passed
    ;;
  train)
    seed="${2:?seed 42, 8322, or 8333 required}"
    gpu="${3:?GPU index required}"
    [[ "${seed}" == "42" || "${seed}" == "8322" || "${seed}" == "8333" ]] || exit 2
    verify_freeze
    assert_gpu_idle "${gpu}"
    host_output="${HOST_OUTPUT_ROOT}/runs/e128c_auto_sgcot_seed${seed}"
    [[ ! -e "${host_output}" ]] || {
      echo "refusing to reuse E128 seed${seed} output" >&2
      exit 12
    }
    direct_path="$(direct_model "${seed}")"
    assert_completed_model "${PROJECT_ROOT}${direct_path#/workspace/project}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "mkdir -p '${LOG_DIR}'; FORCE_TORCHRUN=1 llamafactory-cli train '$(sgcot_config "${seed}")' 2>&1 | tee '${LOG_DIR}/e128c_auto_sgcot_seed${seed}.log'"
    ;;
  eval)
    method="${2:?direct or sgcot required}"
    seed="${3:?seed required}"
    split="${4:?test_seen, test_unseen, or pooled_unseen required}"
    gpu="${5:?GPU index required}"
    [[ "${method}" == "direct" || "${method}" == "sgcot" ]] || exit 2
    [[ "${seed}" == "42" || "${seed}" == "8322" || "${seed}" == "8333" ]] || exit 2
    verify_freeze
    assert_gpu_idle "${gpu}"
    model="$(direct_model "${seed}")"
    [[ "${method}" == "sgcot" ]] && model="$(sgcot_model "${seed}")"
    data="$(eval_data "${method}" "${split}")"
    output="$(eval_output "${method}" "${seed}" "${split}")"
    assert_completed_model "${PROJECT_ROOT}${model#/workspace/project}"
    [[ -f "${PROJECT_ROOT}${data#/workspace/project}" ]] || {
      echo "missing evaluation data: ${data}" >&2
      exit 11
    }
    [[ ! -e "${PROJECT_ROOT}${output#/workspace/project}" ]] || {
      echo "refusing to reuse E128 evaluation output: ${output}" >&2
      exit 12
    }
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "mkdir -p '${LOG_DIR}'; python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' --adapter_path '${model}' --eval_jsonl '${data}' \
        --output_dir '${output}' --batch_size '${EVAL_BATCH_SIZE}' \
        --max_new_tokens '${EVAL_MAX_NEW_TOKENS}' --temperature '${EVAL_TEMPERATURE}' \
        2>&1 | tee '${LOG_DIR}/eval_${method}_seed${seed}_${split}.log'"
    ;;
  compare-n3)
    verify_freeze
    python3 scripts/compare_e121_confirmation_n3_20260712.py \
      --baseline_seen \
        "${HOST_OUTPUT_ROOT}/eval/e128_direct_seed42/test_seen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_direct_seed8322/test_seen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_direct_seed8333/test_seen" \
      --baseline_unseen \
        "${HOST_OUTPUT_ROOT}/eval/e128_direct_seed42/test_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_direct_seed8322/test_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_direct_seed8333/test_unseen" \
      --baseline_pooled \
        "${HOST_OUTPUT_ROOT}/eval/e128_direct_seed42/pooled_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_direct_seed8322/pooled_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_direct_seed8333/pooled_unseen" \
      --candidate_seen \
        "${HOST_OUTPUT_ROOT}/eval/e128_sgcot_seed42/test_seen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_sgcot_seed8322/test_seen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_sgcot_seed8333/test_seen" \
      --candidate_unseen \
        "${HOST_OUTPUT_ROOT}/eval/e128_sgcot_seed42/test_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_sgcot_seed8322/test_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_sgcot_seed8333/test_unseen" \
      --candidate_pooled \
        "${HOST_OUTPUT_ROOT}/eval/e128_sgcot_seed42/pooled_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_sgcot_seed8322/pooled_unseen" \
        "${HOST_OUTPUT_ROOT}/eval/e128_sgcot_seed8333/pooled_unseen" \
      --heldout_types_json "${PROJECT_ROOT}/data/processed/type_holdout/richere-en/${PROTOCOL}/split1/unseen_types.json" \
      --gate_config "${PROJECT_ROOT}/${CONFIG_DIR}/e121e_confirmation_gate.json" \
      --eval_config "${EVAL_CONFIG_HOST}" \
      --output_dir "${HOST_OUTPUT_ROOT}/analysis/e128_n3_confirmation" \
      --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
      --bootstrap_seed "${BOOTSTRAP_SEED}"
    ;;
  *)
    echo "usage: $0 freeze|verify-freeze|train-smoke <gpu>|train <seed> <gpu>|eval <direct|sgcot> <seed> <test_seen|test_unseen|pooled_unseen> <gpu>|compare-n3" >&2
    exit 2
    ;;
esac
