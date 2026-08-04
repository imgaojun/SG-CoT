#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
STATE_DIR="/mnt/disk/gaojun/tmp/gpu-label-service"

INPUT="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_train_pos.jsonl"
FINALONLY_INPUT="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e77b_e81_rowmatched_control_train_pos.jsonl"
SAMPLES="/workspace/project/outputs/stage2_preference_mining/e110a_e81_k4_seed1104/samples.shard-*.jsonl"
START_MODEL="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
EVAL_DATA="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot"
FINALONLY_EVAL_DATA="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e77b_e81_rowmatched_control"

BUILD_OUTPUT="/workspace/project/outputs/stage2_preference_mining/e114a_e81_atomic_counterfactual_seed1140"
HOST_BUILD_OUTPUT="${PROJECT_ROOT}/outputs/stage2_preference_mining/e114a_e81_atomic_counterfactual_seed1140"
PREFERENCE_NAME="richere_balanced_split1_e114a_e81_atomic_counterfactual_orpo_seed1140"
DETERMINISTIC_NAME="richere_balanced_split1_e114c3_e81_atomic_counterfactual_deterministic_only_seed1140"
SMOKE_NAME="richere_balanced_split1_e114b0_e81_atomic_counterfactual_orpo_smoke16"
DATASET_DIR="/workspace/project/data/stage2_adaptive_datasets"
PREFERENCE_JSONL="${DATASET_DIR}/${PREFERENCE_NAME}.jsonl"
DETERMINISTIC_JSONL="${DATASET_DIR}/${DETERMINISTIC_NAME}.jsonl"
SMOKE_JSONL="${DATASET_DIR}/${SMOKE_NAME}.jsonl"
FREEZE_JSON="${HOST_BUILD_OUTPUT}/frozen_artifacts.json"

SMOKE_CONFIG="configs/generated/stage2_preference/e114b0_e81_atomic_orpo_smoke16.yaml"
MAIN_CONFIG="configs/generated/stage2_preference/e114b1_e81_atomic_orpo_seed42.yaml"
RSFT_CONFIG="configs/generated/stage2_preference/e114c1_e81_atomic_chosen_rsft_seed42.yaml"
FINALONLY_CONFIG="configs/generated/stage2_preference/e114c2_e77b_atomic_finalonly_orpo_seed42.yaml"
DETERMINISTIC_CONFIG="configs/generated/stage2_preference/e114c3_e81_atomic_deterministic_orpo_seed42.yaml"

SMOKE_MODEL="/workspace/project/outputs/stage2_preference_runs/e114b0_e81_atomic_orpo_smoke16"
MAIN_MODEL="/workspace/project/outputs/stage2_preference_runs/e114b1_e81_atomic_orpo_seed42"
MAIN_EVAL="/workspace/project/outputs/stage2_preference_eval/e114b1_e81_atomic_orpo_seed42"
RSFT_MODEL="/workspace/project/outputs/stage2_preference_runs/e114c1_e81_atomic_chosen_rsft_seed42"
RSFT_EVAL="/workspace/project/outputs/stage2_preference_eval/e114c1_e81_atomic_chosen_rsft_seed42"
FINALONLY_MODEL="/workspace/project/outputs/stage2_preference_runs/e114c2_e77b_atomic_finalonly_orpo_seed42"
FINALONLY_EVAL="/workspace/project/outputs/stage2_preference_eval/e114c2_e77b_atomic_finalonly_orpo_seed42"
DETERMINISTIC_MODEL="/workspace/project/outputs/stage2_preference_runs/e114c3_e81_atomic_deterministic_orpo_seed42"
DETERMINISTIC_EVAL="/workspace/project/outputs/stage2_preference_eval/e114c3_e81_atomic_deterministic_orpo_seed42"

docker_common() {
  docker run --rm --user root --ipc host --shm-size 16g \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -v "${LF_ROOT}/cache/torch_extensions:/workspace/.cache/torch_extensions" \
    -v "${LF_ROOT}/logs:/workspace/logs" \
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
  local metadata="${STATE_DIR}/gpu${gpu}.json"
  [[ -f "${metadata}" ]] || return 0
  local owner metadata_gpu pid cmdline
  owner="$(jq -r '.owner // empty' "${metadata}")"
  metadata_gpu="$(jq -r '.gpu // empty' "${metadata}")"
  pid="$(jq -r '.pid // empty' "${metadata}")"
  if [[ "${owner}" != "gaojun" || "${metadata_gpu}" != "${gpu}" || ! "${pid}" =~ ^[0-9]+$ ]]; then
    echo "refusing label-service release: invalid metadata ${metadata}" >&2
    return 2
  fi
  [[ -r "/proc/${pid}/cmdline" ]] || return 0
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  if [[ "${cmdline}" != *gpu-label-service* || "${cmdline}" != *gaojun* || "${cmdline}" != *"gpu${gpu}"* ]]; then
    echo "refusing label-service release: PID ${pid} does not match metadata" >&2
    return 2
  fi
  kill "${pid}"
  for _ in $(seq 1 20); do
    [[ -d "/proc/${pid}" ]] || return 0
    sleep 0.25
  done
  echo "label-service PID ${pid} did not exit" >&2
  return 2
}

require_frozen() {
  [[ -f "${FREEZE_JSON}" ]] && jq -e '.frozen == true' "${FREEZE_JSON}" >/dev/null
}

require_smoke_passed() {
  local margin="${PROJECT_ROOT}/outputs/stage2_preference_runs/e114b0_e81_atomic_orpo_smoke16/post_step_margin.json"
  [[ -f "${margin}" ]] && jq -e '.positive_mean_reward_margin == true' "${margin}" >/dev/null
}

run_train() {
  local config="$1"
  local gpu="$2"
  local log="$3"
  require_frozen
  release_owned_label_service "${gpu}"
  mkdir -p "$(dirname "${PROJECT_ROOT}/${log}")"
  docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
    "FORCE_TORCHRUN=1 llamafactory-cli train '${config}' 2>&1 | tee '/workspace/project/${log}'"
}

run_eval() {
  local model="$1"
  local eval_prefix="$2"
  local output="$3"
  local split="$4"
  local gpu="$5"
  local log="$6"
  require_frozen
  release_owned_label_service "${gpu}"
  docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
    "python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
      --base_model '${BASE_MODEL}' --adapter_path '${model}' \
      --eval_jsonl '${eval_prefix}_${split}_pos.jsonl' \
      --output_dir '${output}/${split}' --batch_size 4 --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${BUILD_OUTPUT}/${log}'"
}

seed_config() {
  case "$1" in
    42) echo "${MAIN_CONFIG}" ;;
    8322) echo "configs/generated/stage2_preference/e114b2_e81_atomic_orpo_seed8322.yaml" ;;
    8333) echo "configs/generated/stage2_preference/e114b3_e81_atomic_orpo_seed8333.yaml" ;;
    *) return 2 ;;
  esac
}

seed_model() {
  case "$1" in
    42) echo "${MAIN_MODEL}" ;;
    8322|8333) echo "/workspace/project/outputs/stage2_preference_runs/e114b${1: -1}_e81_atomic_orpo_seed${1}" ;;
    *) return 2 ;;
  esac
}

seed_eval() {
  case "$1" in
    42) echo "${MAIN_EVAL}" ;;
    8322) echo "/workspace/project/outputs/stage2_preference_eval/e114b2_e81_atomic_orpo_seed8322" ;;
    8333) echo "/workspace/project/outputs/stage2_preference_eval/e114b3_e81_atomic_orpo_seed8333" ;;
    *) return 2 ;;
  esac
}

case "${1:-}" in
  build)
    if [[ -f "${FREEZE_JSON}" ]]; then
      echo "E114A is frozen; refusing to rebuild without a new experiment id" >&2
      exit 2
    fi
    docker_common "${IMAGE}" python scripts/build_atomic_reasoning_preferences_e114_20260712.py \
      --input_jsonl "${INPUT}" --samples_glob "${SAMPLES}" --model_path "${START_MODEL}" \
      --output_dir "${BUILD_OUTPUT}" --dataset_dir "${DATASET_DIR}" \
      --dataset_info "${DATASET_DIR}/dataset_info.json" --preference_name "${PREFERENCE_NAME}" \
      --deterministic_preference_name "${DETERMINISTIC_NAME}" --smoke_name "${SMOKE_NAME}" \
      --finalonly_prompt_jsonl "${FINALONLY_INPUT}" --target_pairs 900 --profile e81 \
      --proposal_mode observed_first --renderer_version ac_rpo_v1 --cutoff_len 1536 \
      --min_length_ratio 0.9 --max_length_ratio 1.1 --seed 1140
    ;;
  audit)
    docker_common "${IMAGE}" python scripts/validate_atomic_preferences_e114_20260712.py \
      --preference_jsonl "${PREFERENCE_JSONL}" \
      --deterministic_preference_jsonl "${DETERMINISTIC_JSONL}" \
      --assignment_manifest "${BUILD_OUTPUT}/assignment_manifest.jsonl" \
      --build_summary "${BUILD_OUTPUT}/build_summary.json" --input_jsonl "${INPUT}" \
      --samples_glob "${SAMPLES}" \
      --model_path "${START_MODEL}" --profile e81 --target_pairs 900 \
      --renderer_version ac_rpo_v1 --cutoff_len 1536 --min_length_ratio 0.9 \
      --max_length_ratio 1.1 --output_json "${BUILD_OUTPUT}/pair_audit.json" \
      --freeze_json "${BUILD_OUTPUT}/frozen_artifacts.json"
    ;;
  train-smoke)
    run_train "${SMOKE_CONFIG}" "${2:-0}" \
      "outputs/stage2_preference_mining/e114a_e81_atomic_counterfactual_seed1140/orpo_smoke16_train.log"
    ;;
  score-smoke)
    require_frozen
    gpu="${2:-0}"
    release_owned_label_service "${gpu}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" python scripts/score_preference_margin_20260712.py \
      --model_path "${SMOKE_MODEL}" --preference_jsonl "${SMOKE_JSONL}" \
      --output_json "${SMOKE_MODEL}/post_step_margin.json" --cutoff_len 1536 --beta 0.1
    ;;
  train-main)
    require_smoke_passed
    run_train "${MAIN_CONFIG}" "${2:-0}" \
      "outputs/stage2_preference_mining/e114a_e81_atomic_counterfactual_seed1140/orpo_seed42_train.log"
    ;;
  eval-main)
    run_eval "${MAIN_MODEL}" "${EVAL_DATA}" "${MAIN_EVAL}" \
      "${2:?test_seen or test_unseen required}" "${3:-0}" "orpo_seed42_${2}_eval.log"
    ;;
  compare-main)
    python scripts/compare_preference_run_gate_20260712.py \
      --baseline_seen outputs/stage2_preference_eval/e110_shared_recovery_baseline/test_seen \
      --baseline_unseen outputs/stage2_preference_eval/e110_shared_recovery_baseline/test_unseen \
      --candidate_seen outputs/stage2_preference_eval/e114b1_e81_atomic_orpo_seed42/test_seen \
      --candidate_unseen outputs/stage2_preference_eval/e114b1_e81_atomic_orpo_seed42/test_unseen \
      --output_dir outputs/stage2_preference_eval/e114b1_e81_atomic_orpo_seed42/gate \
      --gate e81 --bootstrap_samples 10000 --seed 20260712
    ;;
  train-main-seed)
    seed="${2:?seed required}"
    run_train "$(seed_config "${seed}")" "${3:?GPU required}" \
      "outputs/stage2_preference_mining/e114a_e81_atomic_counterfactual_seed1140/orpo_seed${seed}_train.log"
    ;;
  eval-main-seed)
    seed="${2:?seed required}"
    split="${3:?split required}"
    run_eval "$(seed_model "${seed}")" "${EVAL_DATA}" "$(seed_eval "${seed}")" \
      "${split}" "${4:?GPU required}" "orpo_seed${seed}_${split}_eval.log"
    ;;
  train-rsft)
    run_train "${RSFT_CONFIG}" "${2:-0}" \
      "outputs/stage2_preference_mining/e114a_e81_atomic_counterfactual_seed1140/chosen_rsft_train.log"
    ;;
  train-finalonly)
    run_train "${FINALONLY_CONFIG}" "${2:-0}" \
      "outputs/stage2_preference_mining/e114a_e81_atomic_counterfactual_seed1140/finalonly_orpo_train.log"
    ;;
  train-deterministic)
    run_train "${DETERMINISTIC_CONFIG}" "${2:-0}" \
      "outputs/stage2_preference_mining/e114a_e81_atomic_counterfactual_seed1140/deterministic_orpo_train.log"
    ;;
  eval-rsft)
    run_eval "${RSFT_MODEL}" "${EVAL_DATA}" "${RSFT_EVAL}" "${2:?split required}" \
      "${3:-0}" "chosen_rsft_${2}_eval.log"
    ;;
  eval-finalonly)
    run_eval "${FINALONLY_MODEL}" "${FINALONLY_EVAL_DATA}" "${FINALONLY_EVAL}" \
      "${2:?split required}" "${3:-0}" "finalonly_${2}_eval.log"
    ;;
  eval-deterministic)
    run_eval "${DETERMINISTIC_MODEL}" "${EVAL_DATA}" "${DETERMINISTIC_EVAL}" \
      "${2:?split required}" "${3:-0}" "deterministic_${2}_eval.log"
    ;;
  *)
    echo "usage: $0 build|audit|train-smoke [gpu]|score-smoke [gpu]|train-main [gpu]|eval-main <split> [gpu]|compare-main|train-main-seed <seed> <gpu>|eval-main-seed <seed> <split> <gpu>|train-rsft [gpu]|train-finalonly [gpu]|train-deterministic [gpu]|eval-rsft|eval-finalonly|eval-deterministic <split> [gpu]" >&2
    exit 2
    ;;
esac
