#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
PREFIX="richere_balanced_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle"
DATA_DIR="/workspace/project/data/stage2_strict_datasets"
HELDOUT="/workspace/project/data/processed/type_holdout/richere-en/balanced-subtype-v1/split1/unseen_types.json"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
GEN_RUN="e111c_strict_sgcot_autocluster"
GEN_OUTPUT="/workspace/project/outputs/stage2_strict_generation/${GEN_RUN}"
GEN_TRAIN="/workspace/project/data/stage2_adaptive_datasets/${PREFIX}_${GEN_RUN}_thinking_evidence_cot_train_pos.jsonl"
SGCOT_TRAIN="${DATA_DIR}/${PREFIX}_sgcot_autocluster_train_pos.jsonl"
SGCOT_REFERENCE="${DATA_DIR}/${PREFIX}_sgcot_target_train_pos.jsonl"
PREF_SCRIPT="scripts/mine_reasoning_preferences_e110_20260711.py"
PREF_OUTPUT="/workspace/project/outputs/stage2_preference_mining/e112a_strict_sgcot_k4_seed1104"
PREF_NAME="richere_balanced_split1_e112a_strict_sgcot_reasoning_path_orpo_k4_seed1104"
PREF_SMOKE_NAME="richere_balanced_split1_e112a_strict_sgcot_reasoning_path_orpo_smoke16"
LOG_DIR="/workspace/project/outputs/stage2_strict_logs"

docker_common() {
  docker run --rm --user root --ipc host --shm-size 16g \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -v "${LF_ROOT}/cache/torch_extensions:/workspace/.cache/torch_extensions" \
    -v "${LF_ROOT}/logs:/workspace/logs" \
    -e OPENAI_API_KEY \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    -e HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    -e TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    -w /workspace/project "$@"
}

config_path() {
  local method="$1"
  local seed="$2"
  case "${method}" in
    direct) echo "configs/generated/stage2_strict/e111b_direct_surface_seed${seed}.yaml" ;;
    sgcot) echo "configs/generated/stage2_strict/e111d_sgcot_seed${seed}.yaml" ;;
    rpo) echo "configs/generated/stage2_strict/e112b_sgcot_rpo_seed${seed}.yaml" ;;
    *) return 2 ;;
  esac
}

model_path() {
  local method="$1"
  local seed="$2"
  case "${method}" in
    direct) echo "/workspace/project/outputs/stage2_strict_runs/e111b_direct_surface_seed${seed}" ;;
    sgcot) echo "/workspace/project/outputs/stage2_strict_runs/e111d_sgcot_seed${seed}" ;;
    rpo) echo "/workspace/project/outputs/stage2_strict_runs/e112b_sgcot_rpo_seed${seed}" ;;
    *) return 2 ;;
  esac
}

eval_prefix() {
  local method="$1"
  case "${method}" in
    direct) echo "${DATA_DIR}/${PREFIX}_direct_surface" ;;
    sgcot|rpo) echo "${DATA_DIR}/${PREFIX}_sgcot_target" ;;
    *) return 2 ;;
  esac
}

eval_output() {
  local method="$1"
  local seed="$2"
  case "${method}" in
    direct) echo "/workspace/project/outputs/stage2_strict_eval/e111b_direct_surface_seed${seed}" ;;
    sgcot) echo "/workspace/project/outputs/stage2_strict_eval/e111d_sgcot_seed${seed}" ;;
    rpo) echo "/workspace/project/outputs/stage2_strict_eval/e112b_sgcot_rpo_seed${seed}" ;;
    *) return 2 ;;
  esac
}

sample_preferences() {
  local output="$1"
  shift
  python "${PREF_SCRIPT}" \
    --mode sample \
    --model_path /workspace/project/outputs/stage2_strict_runs/e111d_sgcot_seed42 \
    --input_jsonl "${SGCOT_TRAIN}" \
    --output_dir "${output}" \
    --num_samples 4 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_new_tokens 1024 \
    --seed 1104 \
    --profile e81 \
    "$@"
}

run_generation() {
  local limit="$1"
  local output="$2"
  local retry_flag="$3"
  local run_name="$4"
  local extra=()
  if [[ -n "${GEN_BASE_URL:-}" ]]; then
    extra+=(--base_url "${GEN_BASE_URL}")
  fi
  if [[ "${retry_flag}" == "retry" ]]; then
    extra+=(--retry_rejected)
  fi
  docker_common "${IMAGE}" python scripts/generate_strategy_variants_cot_e47_20260606.py \
    --run_name "${run_name}" \
    --limit "${limit}" \
    --seed 1111 \
    --workers "${GEN_WORKERS:-8}" \
    --model "${GEN_MODEL:-deepseek-v4-pro}" \
    --verifier_model "${VERIFIER_MODEL:-deepseek-v4-pro}" \
    --prompt_profile e95_trigger_locked_autocluster \
    --repair_profile strict_full \
    --output_protocol xml_tags \
    --sampled_rows_path "${SGCOT_REFERENCE}" \
    --output_dir "${output}" \
    --formal_data_dir "${DATA_DIR}" \
    --data_prefix "${PREFIX}_sgcot_target" \
    --adaptive_prefix "${PREFIX}" \
    --run_prefix e111_strict_seenonly_qwen3_4b \
    --warm_start /workspace/project/outputs/stage2_strict_runs/e111b_direct_surface_seed42 \
    --auto_cluster_map_path /workspace/project/data/schema/richere-en.auto_cluster_map.json \
    --max_attempts 3 \
    "${extra[@]}"
}

case "${1:-}" in
  build-data)
    docker_common "${IMAGE}" bash -lc \
      "PROJECT_ROOT=/workspace/project bash scripts/build_e111_strict_seenonly_surface_data_20260712.sh"
    docker_common "${IMAGE}" python scripts/audit_sft_dataset_lengths_20260712.py \
      --input_jsonl "${DATA_DIR}/${PREFIX}_direct_surface_train_pos.jsonl" \
      --model_path "${BASE_MODEL}" \
      --cutoff_len 1536 \
      --output_json "${DATA_DIR}/e111a_direct_surface_train_length_audit.json" \
      --require_all_fit
    ;;
  generate-smoke)
    run_generation 2 "${GEN_OUTPUT}_smoke2" fresh "${GEN_RUN}_smoke2"
    ;;
  generate-sgcot)
    run_generation 1500 "${GEN_OUTPUT}" retry "${GEN_RUN}"
    ;;
  normalize-sgcot)
    docker_common "${IMAGE}" python scripts/normalize_strict_sgcot_dataset_20260712.py \
      --generated_jsonl "${GEN_TRAIN}" \
      --reference_surface_jsonl "${SGCOT_REFERENCE}" \
      --output_jsonl "${SGCOT_TRAIN}" \
      --dataset_name "${PREFIX}_sgcot_autocluster_train_pos" \
      --dataset_info "${DATA_DIR}/dataset_info.json" \
      --heldout_types_json "${HELDOUT}" \
      --require_zero_leaks \
      --min_rows 1400 \
      --model_path "${BASE_MODEL}" \
      --cutoff_len 2048
    ;;
  train-direct|train-sgcot|train-rpo)
    method="${1#train-}"
    seed="${2:?seed 42, 8322, or 8333 required}"
    gpu="${3:?GPU index required}"
    config="$(config_path "${method}" "${seed}")"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "mkdir -p '${LOG_DIR}'; FORCE_TORCHRUN=1 llamafactory-cli train '${config}' 2>&1 | tee '${LOG_DIR}/${method}_seed${seed}.log'"
    ;;
  eval-direct|eval-sgcot|eval-rpo)
    method="${1#eval-}"
    seed="${2:?seed 42, 8322, or 8333 required}"
    split="${3:?test_seen or test_unseen required}"
    gpu="${4:?GPU index required}"
    model="$(model_path "${method}" "${seed}")"
    data="$(eval_prefix "${method}")_${split}_pos.jsonl"
    output="$(eval_output "${method}" "${seed}")/${split}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "mkdir -p '${LOG_DIR}'; python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' --adapter_path '${model}' --eval_jsonl '${data}' \
        --output_dir '${output}' --batch_size 4 --max_new_tokens 1024 --temperature 0.0 \
        2>&1 | tee '${LOG_DIR}/${method}_seed${seed}_${split}.log'"
    ;;
  sample-smoke)
    gpu="${2:?GPU index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "$(declare -f sample_preferences); PREF_SCRIPT='${PREF_SCRIPT}'; SGCOT_TRAIN='${SGCOT_TRAIN}'; sample_preferences '${PREF_OUTPUT}/smoke2' --max_examples 2 --overwrite --log_every 1"
    ;;
  sample-shard)
    shard="${2:?shard 0..3 required}"
    gpu="${3:?GPU index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "$(declare -f sample_preferences); PREF_SCRIPT='${PREF_SCRIPT}'; SGCOT_TRAIN='${SGCOT_TRAIN}'; sample_preferences '${PREF_OUTPUT}' --num_shards 4 --shard_index '${shard}' --log_every 10"
    ;;
  sample-topup-shard)
    shard="${2:?shard 0..3 required}"
    gpu="${3:?GPU index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "$(declare -f sample_preferences); PREF_SCRIPT='${PREF_SCRIPT}'; SGCOT_TRAIN='${SGCOT_TRAIN}'; sample_preferences '${PREF_OUTPUT}' --num_shards 4 --shard_index '${shard}' --sample_round 1 --wnd_ids_json '${PREF_OUTPUT}/topup_wnd_ids.json' --log_every 10"
    ;;
  build-preferences|build-preference-smoke)
    name="${PREF_NAME}"
    output="${PREF_OUTPUT}"
    min_pairs=900
    max_args=()
    if [[ "${1}" == "build-preference-smoke" ]]; then
      name="${PREF_SMOKE_NAME}"
      output="${PREF_OUTPUT}/orpo_smoke16"
      min_pairs=16
      max_args=(--max_pairs 16)
    fi
    docker_common "${IMAGE}" python "${PREF_SCRIPT}" \
      --mode build \
      --model_path /workspace/project/outputs/stage2_strict_runs/e111d_sgcot_seed42 \
      --input_jsonl "${SGCOT_TRAIN}" \
      --output_dir "${output}" \
      --samples_glob "${PREF_OUTPUT}/samples.shard-*.jsonl" \
      --profile e81 \
      --cutoff_len 1536 \
      --min_pairs "${min_pairs}" \
      --dataset_dir "${DATA_DIR}" \
      --dataset_info "${DATA_DIR}/dataset_info.json" \
      --preference_name "${name}" \
      "${max_args[@]}"
    ;;
  audit-preferences)
    docker_common "${IMAGE}" python scripts/validate_reasoning_preferences_e110_20260712.py \
      --preference_jsonl "${DATA_DIR}/${PREF_NAME}.jsonl" \
      --input_jsonl "${SGCOT_TRAIN}" \
      --model_path /workspace/project/outputs/stage2_strict_runs/e111d_sgcot_seed42 \
      --profile e81 \
      --cutoff_len 1536 \
      --min_pairs 900 \
      --output_json "${PREF_OUTPUT}/pair_audit.json"
    ;;
  train-preference-smoke)
    gpu="${2:?GPU index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train configs/generated/stage2_strict/e112b0_sgcot_rpo_smoke16.yaml 2>&1 | tee '${PREF_OUTPUT}/orpo_smoke16/train.log'"
    ;;
  score-preference-smoke)
    gpu="${2:?GPU index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" python scripts/score_preference_margin_20260712.py \
      --model_path /workspace/project/outputs/stage2_strict_runs/e112b0_sgcot_rpo_smoke16 \
      --preference_jsonl "${DATA_DIR}/${PREF_SMOKE_NAME}.jsonl" \
      --output_json "${PREF_OUTPUT}/orpo_smoke16/reward_margin.json" \
      --cutoff_len 1536 \
      --beta 0.1
    ;;
  compare-sgcot|compare-rpo)
    method="${1#compare-}"
    seed="${2:?seed 42, 8322, or 8333 required}"
    candidate="$(eval_output "${method}" "${seed}")"
    baseline="$(eval_output direct "${seed}")"
    gate=none
    if [[ "${method}" == "rpo" ]]; then
      gate=strict
    fi
    docker_common "${IMAGE}" python scripts/compare_preference_run_gate_20260712.py \
      --baseline_seen "${baseline}/test_seen" \
      --baseline_unseen "${baseline}/test_unseen" \
      --candidate_seen "${candidate}/test_seen" \
      --candidate_unseen "${candidate}/test_unseen" \
      --output_dir "${candidate}/comparison_vs_direct" \
      --gate "${gate}" \
      --bootstrap_samples 10000 \
      --seed 20260712
    ;;
  compare-rpo-n3)
    docker_common "${IMAGE}" python scripts/compare_strict_n3_gate_20260712.py \
      --baseline_seen \
        /workspace/project/outputs/stage2_strict_eval/e111b_direct_surface_seed42/test_seen \
        /workspace/project/outputs/stage2_strict_eval/e111b_direct_surface_seed8322/test_seen \
        /workspace/project/outputs/stage2_strict_eval/e111b_direct_surface_seed8333/test_seen \
      --baseline_unseen \
        /workspace/project/outputs/stage2_strict_eval/e111b_direct_surface_seed42/test_unseen \
        /workspace/project/outputs/stage2_strict_eval/e111b_direct_surface_seed8322/test_unseen \
        /workspace/project/outputs/stage2_strict_eval/e111b_direct_surface_seed8333/test_unseen \
      --candidate_seen \
        /workspace/project/outputs/stage2_strict_eval/e112b_sgcot_rpo_seed42/test_seen \
        /workspace/project/outputs/stage2_strict_eval/e112b_sgcot_rpo_seed8322/test_seen \
        /workspace/project/outputs/stage2_strict_eval/e112b_sgcot_rpo_seed8333/test_seen \
      --candidate_unseen \
        /workspace/project/outputs/stage2_strict_eval/e112b_sgcot_rpo_seed42/test_unseen \
        /workspace/project/outputs/stage2_strict_eval/e112b_sgcot_rpo_seed8322/test_unseen \
        /workspace/project/outputs/stage2_strict_eval/e112b_sgcot_rpo_seed8333/test_unseen \
      --output_dir /workspace/project/outputs/stage2_strict_eval/e112b_sgcot_rpo_n3_gate \
      --bootstrap_samples 10000 \
      --seed 20260712
    ;;
  *)
    echo "usage: $0 build-data | generate-smoke | generate-sgcot | normalize-sgcot | train-{direct|sgcot|rpo} <seed> <gpu> | eval-{direct|sgcot|rpo} <seed> <test_seen|test_unseen> <gpu> | sample-smoke <gpu> | sample-shard <0..3> <gpu> | sample-topup-shard <0..3> <gpu> | build-preferences | build-preference-smoke | audit-preferences | train-preference-smoke <gpu> | score-preference-smoke <gpu> | compare-{sgcot|rpo} <seed> | compare-rpo-n3" >&2
    exit 2
    ;;
esac
