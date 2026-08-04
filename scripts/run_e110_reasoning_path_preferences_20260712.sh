#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
SCRIPT="scripts/mine_reasoning_preferences_e110_20260711.py"
MODEL="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full"
INPUT="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_train_pos.jsonl"
FINALONLY_INPUT="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e77b_e81_rowmatched_control_train_pos.jsonl"
OUTPUT="/workspace/project/outputs/stage2_preference_mining/e110a_e81_k4_seed1104"
NAME="richere_balanced_split1_e110a_e81_reasoning_path_orpo_k4_seed1104"
SMOKE_NAME="richere_balanced_split1_e110b0_e81_reasoning_path_orpo_smoke16"
SMOKE_CONFIG="configs/generated/stage2_preference/e110b0_e81_orpo_smoke16.yaml"
MAIN_CONFIG="configs/generated/stage2_preference/e110b1_e81_orpo_seed42.yaml"
RSFT_CONFIG="configs/generated/stage2_preference/e110c1_e81_chosen_rsft_seed42.yaml"
FINALONLY_CONFIG="configs/generated/stage2_preference/e110c2_e77b_finalonly_orpo_seed42.yaml"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
MAIN_MODEL="/workspace/project/outputs/stage2_preference_runs/e110b1_e81_orpo_seed42"
EVAL_DATA="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot"
EVAL_OUTPUT="/workspace/project/outputs/stage2_preference_eval/e110b1_e81_orpo_seed42"
RSFT_MODEL="/workspace/project/outputs/stage2_preference_runs/e110c1_e81_chosen_rsft_seed42"
RSFT_EVAL_OUTPUT="/workspace/project/outputs/stage2_preference_eval/e110c1_e81_chosen_rsft_seed42"
FINALONLY_MODEL="/workspace/project/outputs/stage2_preference_runs/e110c2_e77b_finalonly_orpo_seed42"
FINALONLY_EVAL_DATA="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e77b_e81_rowmatched_control"
FINALONLY_EVAL_OUTPUT="/workspace/project/outputs/stage2_preference_eval/e110c2_e77b_finalonly_orpo_seed42"

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

sample_args() {
  local output="$1"
  shift
  python "${SCRIPT}" \
    --mode sample \
    --model_path "${MODEL}" \
    --input_jsonl "${INPUT}" \
    --output_dir "${output}" \
    --num_samples 4 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_new_tokens 1024 \
    --seed 1104 \
    --profile e81 \
    "$@"
}

e81_start_config() {
  local seed="$1"
  echo "configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_s${seed}_full_stepmatch.yaml"
}

e81_start_model() {
  local seed="$1"
  if [[ "${seed}" == "42" ]]; then
    echo "${MODEL}"
  else
    echo "/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_s${seed}_full"
  fi
}

orpo_config() {
  local seed="$1"
  case "${seed}" in
    42) echo "${MAIN_CONFIG}" ;;
    8322) echo "configs/generated/stage2_preference/e110b2_e81_orpo_seed8322.yaml" ;;
    8333) echo "configs/generated/stage2_preference/e110b3_e81_orpo_seed8333.yaml" ;;
    *) return 2 ;;
  esac
}

orpo_model() {
  local seed="$1"
  case "${seed}" in
    42) echo "${MAIN_MODEL}" ;;
    8322) echo "/workspace/project/outputs/stage2_preference_runs/e110b2_e81_orpo_seed8322" ;;
    8333) echo "/workspace/project/outputs/stage2_preference_runs/e110b3_e81_orpo_seed8333" ;;
    *) return 2 ;;
  esac
}

orpo_eval_output() {
  local seed="$1"
  case "${seed}" in
    42) echo "${EVAL_OUTPUT}" ;;
    8322) echo "/workspace/project/outputs/stage2_preference_eval/e110b2_e81_orpo_seed8322" ;;
    8333) echo "/workspace/project/outputs/stage2_preference_eval/e110b3_e81_orpo_seed8333" ;;
    *) return 2 ;;
  esac
}

case "${1:-}" in
  smoke)
    docker_common --gpus "device=${2:-1}" "${IMAGE}" bash -lc \
      "$(declare -f sample_args); SCRIPT='${SCRIPT}'; MODEL='${MODEL}'; INPUT='${INPUT}'; sample_args '${OUTPUT}/smoke' --max_examples 2 --overwrite --log_every 1"
    ;;
  sample-shard)
    shard="${2:?shard index required}"
    gpu="${3:?gpu index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "$(declare -f sample_args); SCRIPT='${SCRIPT}'; MODEL='${MODEL}'; INPUT='${INPUT}'; sample_args '${OUTPUT}' --num_shards 4 --shard_index '${shard}' --log_every 10"
    ;;
  sample-resume6-shard)
    shard="${2:?shard index required}"
    gpu="${3:?gpu index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "$(declare -f sample_args); SCRIPT='${SCRIPT}'; MODEL='${MODEL}'; INPUT='${INPUT}'; sample_args '${OUTPUT}' --num_shards 6 --shard_index '${shard}' --completed_samples_glob '${OUTPUT}/samples.shard-*.jsonl' --log_every 10"
    ;;
  sample-topup-shard)
    shard="${2:?shard index required}"
    gpu="${3:?gpu index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "$(declare -f sample_args); SCRIPT='${SCRIPT}'; MODEL='${MODEL}'; INPUT='${INPUT}'; sample_args '${OUTPUT}' --num_shards 4 --shard_index '${shard}' --sample_round 1 --wnd_ids_json '${OUTPUT}/topup_wnd_ids.json' --log_every 10"
    ;;
  sample-topup6-shard)
    shard="${2:?shard index required}"
    gpu="${3:?gpu index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "$(declare -f sample_args); SCRIPT='${SCRIPT}'; MODEL='${MODEL}'; INPUT='${INPUT}'; sample_args '${OUTPUT}' --num_shards 6 --shard_index '${shard}' --sample_round 1 --wnd_ids_json '${OUTPUT}/topup_wnd_ids.json' --completed_samples_glob '${OUTPUT}/samples.shard-*.jsonl' --log_every 10"
    ;;
  build)
    docker_common "${IMAGE}" python "${SCRIPT}" \
      --mode build \
      --model_path "${MODEL}" \
      --input_jsonl "${INPUT}" \
      --output_dir "${OUTPUT}" \
      --samples_glob "${OUTPUT}/samples.shard-*.jsonl" \
      --profile e81 \
      --cutoff_len 1536 \
      --min_pairs 900 \
      --dataset_dir /workspace/project/data/stage2_adaptive_datasets \
      --dataset_info /workspace/project/data/stage2_adaptive_datasets/dataset_info.json \
      --finalonly_prompt_jsonl "${FINALONLY_INPUT}" \
      --preference_name "${NAME}"
    ;;
  audit)
    docker_common "${IMAGE}" python scripts/validate_reasoning_preferences_e110_20260712.py \
      --preference_jsonl "/workspace/project/data/stage2_adaptive_datasets/${NAME}.jsonl" \
      --input_jsonl "${INPUT}" \
      --model_path "${MODEL}" \
      --profile e81 \
      --cutoff_len 1536 \
      --min_pairs 900 \
      --output_json "${OUTPUT}/pair_audit.json"
    ;;
  build-smoke)
    docker_common "${IMAGE}" python "${SCRIPT}" \
      --mode build \
      --model_path "${MODEL}" \
      --input_jsonl "${INPUT}" \
      --output_dir "${OUTPUT}/orpo_smoke16" \
      --samples_glob "${OUTPUT}/samples.shard-*.jsonl" \
      --profile e81 \
      --cutoff_len 1536 \
      --min_pairs 16 \
      --max_pairs 16 \
      --dataset_dir /workspace/project/data/stage2_adaptive_datasets \
      --dataset_info /workspace/project/data/stage2_adaptive_datasets/dataset_info.json \
      --finalonly_prompt_jsonl "${FINALONLY_INPUT}" \
      --preference_name "${SMOKE_NAME}"
    ;;
  train-smoke)
    docker_common --gpus "device=${2:-1}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${SMOKE_CONFIG}' 2>&1 | tee '${OUTPUT}/orpo_smoke16/train.log'"
    ;;
  train-main)
    docker_common --gpus "device=${2:-1}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${MAIN_CONFIG}' 2>&1 | tee '${OUTPUT}/orpo_seed42_train.log'"
    ;;
  train-e81-start)
    seed="${2:?seed 8322 or 8333 required}"
    gpu="${3:?GPU index required}"
    config="$(e81_start_config "${seed}")"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${config}' 2>&1 | tee '${OUTPUT}/e81_start_seed${seed}_train.log'"
    ;;
  eval-e81-start)
    seed="${2:?seed 8322 or 8333 required}"
    split="${3:?test_seen or test_unseen required}"
    gpu="${4:?GPU index required}"
    start_model="$(e81_start_model "${seed}")"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' --adapter_path '${start_model}' \
        --eval_jsonl '${EVAL_DATA}_${split}_pos.jsonl' \
        --output_dir '/workspace/project/outputs/stage2_preference_eval/e110_e81_start_seed${seed}/${split}' \
        --batch_size 4 --max_new_tokens 1024 --temperature 0.0 \
        2>&1 | tee '${OUTPUT}/e81_start_seed${seed}_${split}_eval.log'"
    ;;
  train-main-seed)
    seed="${2:?seed 42, 8322, or 8333 required}"
    gpu="${3:?GPU index required}"
    config="$(orpo_config "${seed}")"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${config}' 2>&1 | tee '${OUTPUT}/orpo_seed${seed}_train.log'"
    ;;
  train-rsft)
    docker_common --gpus "device=${2:-1}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${RSFT_CONFIG}' 2>&1 | tee '${OUTPUT}/rsft_seed42_train.log'"
    ;;
  train-finalonly)
    docker_common --gpus "device=${2:-1}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${FINALONLY_CONFIG}' 2>&1 | tee '${OUTPUT}/finalonly_seed42_train.log'"
    ;;
  eval-main)
    split="${2:?test_seen or test_unseen required}"
    gpu="${3:-1}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' \
        --adapter_path '${MAIN_MODEL}' \
        --eval_jsonl '${EVAL_DATA}_${split}_pos.jsonl' \
        --output_dir '${EVAL_OUTPUT}/${split}' \
        --batch_size 4 \
        --max_new_tokens 1024 \
        --temperature 0.0 2>&1 | tee '${OUTPUT}/orpo_seed42_${split}_eval.log'"
    ;;
  eval-main-seed)
    seed="${2:?seed 42, 8322, or 8333 required}"
    split="${3:?test_seen or test_unseen required}"
    gpu="${4:?GPU index required}"
    model="$(orpo_model "${seed}")"
    eval_output="$(orpo_eval_output "${seed}")"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' --adapter_path '${model}' \
        --eval_jsonl '${EVAL_DATA}_${split}_pos.jsonl' \
        --output_dir '${eval_output}/${split}' \
        --batch_size 4 --max_new_tokens 1024 --temperature 0.0 \
        2>&1 | tee '${OUTPUT}/orpo_seed${seed}_${split}_eval.log'"
    ;;
  eval-rsft)
    split="${2:?test_seen or test_unseen required}"
    gpu="${3:-1}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' \
        --adapter_path '${RSFT_MODEL}' \
        --eval_jsonl '${EVAL_DATA}_${split}_pos.jsonl' \
        --output_dir '${RSFT_EVAL_OUTPUT}/${split}' \
        --batch_size 4 --max_new_tokens 1024 --temperature 0.0 \
        2>&1 | tee '${OUTPUT}/rsft_seed42_${split}_eval.log'"
    ;;
  eval-finalonly)
    split="${2:?test_seen or test_unseen required}"
    gpu="${3:-1}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' \
        --adapter_path '${FINALONLY_MODEL}' \
        --eval_jsonl '${FINALONLY_EVAL_DATA}_${split}_pos.jsonl' \
        --output_dir '${FINALONLY_EVAL_OUTPUT}/${split}' \
        --batch_size 4 --max_new_tokens 1024 --temperature 0.0 \
        2>&1 | tee '${OUTPUT}/finalonly_seed42_${split}_eval.log'"
    ;;
  compare-main)
    python scripts/compare_preference_run_gate_20260712.py \
      --baseline_seen outputs/stage2_preference_eval/e110_shared_recovery_baseline/test_seen \
      --baseline_unseen outputs/stage2_preference_eval/e110_shared_recovery_baseline/test_unseen \
      --candidate_seen outputs/stage2_preference_eval/e110b1_e81_orpo_seed42/test_seen \
      --candidate_unseen outputs/stage2_preference_eval/e110b1_e81_orpo_seed42/test_unseen \
      --output_dir outputs/stage2_preference_eval/e110b1_e81_orpo_seed42/gate \
      --gate e81 \
      --bootstrap_samples 10000 \
      --seed 20260712
    ;;
  compare-main-seed)
    seed="${2:?seed 42, 8322, or 8333 required}"
    candidate="$(orpo_eval_output "${seed}")"
    if [[ "${seed}" == "42" ]]; then
      baseline="outputs/stage2_preference_eval/e110_shared_recovery_baseline"
    else
      baseline="outputs/stage2_preference_eval/e110_e81_start_seed${seed}"
    fi
    python scripts/compare_preference_run_gate_20260712.py \
      --baseline_seen "${baseline}/test_seen" \
      --baseline_unseen "${baseline}/test_unseen" \
      --candidate_seen "${candidate#/workspace/project/}/test_seen" \
      --candidate_unseen "${candidate#/workspace/project/}/test_unseen" \
      --output_dir "${candidate#/workspace/project/}/gate" \
      --gate e81 \
      --bootstrap_samples 10000 \
      --seed 20260712
    ;;
  compare-rsft)
    python scripts/compare_preference_run_gate_20260712.py \
      --baseline_seen outputs/stage2_preference_eval/e110_shared_recovery_baseline/test_seen \
      --baseline_unseen outputs/stage2_preference_eval/e110_shared_recovery_baseline/test_unseen \
      --candidate_seen outputs/stage2_preference_eval/e110c1_e81_chosen_rsft_seed42/test_seen \
      --candidate_unseen outputs/stage2_preference_eval/e110c1_e81_chosen_rsft_seed42/test_unseen \
      --output_dir outputs/stage2_preference_eval/e110c1_e81_chosen_rsft_seed42/comparison \
      --bootstrap_samples 10000 --seed 20260712
    ;;
  compare-finalonly)
    python scripts/compare_preference_run_gate_20260712.py \
      --baseline_seen outputs/stage2_preference_eval/e110_shared_recovery_e77b_baseline/test_seen \
      --baseline_unseen outputs/stage2_preference_eval/e110_shared_recovery_e77b_baseline/test_unseen \
      --candidate_seen outputs/stage2_preference_eval/e110c2_e77b_finalonly_orpo_seed42/test_seen \
      --candidate_unseen outputs/stage2_preference_eval/e110c2_e77b_finalonly_orpo_seed42/test_unseen \
      --output_dir outputs/stage2_preference_eval/e110c2_e77b_finalonly_orpo_seed42/comparison \
      --bootstrap_samples 10000 --seed 20260712
    ;;
  *)
    echo "usage: $0 smoke [gpu] | sample-shard <0..3> <gpu> | sample-resume6-shard <0..5> <gpu> | sample-topup-shard <0..3> <gpu> | sample-topup6-shard <0..5> <gpu> | build | audit | build-smoke | train-smoke [gpu] | train-main [gpu] | train-e81-start <8322|8333> <gpu> | eval-e81-start <seed> <split> <gpu> | train-main-seed <seed> <gpu> | eval-main-seed <seed> <split> <gpu> | train-rsft [gpu] | train-finalonly [gpu] | eval-main|eval-rsft|eval-finalonly <test_seen|test_unseen> [gpu] | compare-main|compare-main-seed <seed>|compare-rsft|compare-finalonly" >&2
    exit 2
    ;;
esac
