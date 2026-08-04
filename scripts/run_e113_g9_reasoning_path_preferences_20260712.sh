#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
SCRIPT="scripts/mine_reasoning_preferences_e110_20260711.py"
INPUT="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_gollie_style_top10_shuffle_adaptive_hybrid_sgcot_thinking_evidence_cot_train_pos.jsonl"
EVAL_DATA="/workspace/project/data/stage2_adaptive_datasets/richere_balanced_split1_gollie_style_top10_shuffle_adaptive_hybrid_sgcot_thinking_evidence_cot"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
BASE_CONFIG="configs/generated/stage2_preference/e113b1_g9_orpo_base_seed42.yaml"
BASE_TRAINED_MODEL="/workspace/project/outputs/stage2_preference_runs/e113b1_g9_orpo_base_seed42"
BASE_EVAL_OUTPUT="/workspace/project/outputs/stage2_preference_eval/e113b1_g9_orpo_base_seed42"

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

resolve_tag() {
  local tag="$1"
  case "${tag}" in
    base)
      MODEL="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_gollie_style_dualmode_g9_cotcalib_full/checkpoint-91"
      CONFIG="${BASE_CONFIG}"
      TRAINED_MODEL="${BASE_TRAINED_MODEL}"
      EVAL_OUTPUT="${BASE_EVAL_OUTPUT}"
      BASELINE_OUTPUT="/workspace/project/outputs/stage2_preference_eval/e113_shared_recovery_g9_base_baseline"
      ;;
    s8322)
      MODEL="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_gollie_style_dualmode_g9_s8322_cotcalib_full/checkpoint-91"
      CONFIG="configs/generated/stage2_preference/e113b2_g9_orpo_s8322_seed8322.yaml"
      TRAINED_MODEL="/workspace/project/outputs/stage2_preference_runs/e113b2_g9_orpo_s8322_seed8322"
      EVAL_OUTPUT="/workspace/project/outputs/stage2_preference_eval/e113b2_g9_orpo_s8322_seed8322"
      BASELINE_OUTPUT="/workspace/project/outputs/stage2_preference_eval/e113_shared_recovery_g9_s8322_baseline"
      ;;
    s8333)
      MODEL="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_gollie_style_dualmode_g9_s8333_cotcalib_full/checkpoint-91"
      CONFIG="configs/generated/stage2_preference/e113b3_g9_orpo_s8333_seed8333.yaml"
      TRAINED_MODEL="/workspace/project/outputs/stage2_preference_runs/e113b3_g9_orpo_s8333_seed8333"
      EVAL_OUTPUT="/workspace/project/outputs/stage2_preference_eval/e113b3_g9_orpo_s8333_seed8333"
      BASELINE_OUTPUT="/workspace/project/outputs/stage2_preference_eval/e113_shared_recovery_g9_s8333_baseline"
      ;;
    *)
      echo "unknown G9 tag: ${tag}" >&2
      exit 2
      ;;
  esac
  OUTPUT="/workspace/project/outputs/stage2_preference_mining/e113a_g9/${tag}_k4_seed1104"
  NAME="richere_balanced_split1_e113a_g9_${tag}_reasoning_path_orpo_k4_seed1104"
}

sample_args() {
  python "${SCRIPT}" \
    --mode sample \
    --model_path "${MODEL}" \
    --input_jsonl "${INPUT}" \
    --output_dir "${OUTPUT}" \
    --num_samples 4 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_new_tokens 1024 \
    --seed 1104 \
    --profile g9 \
    "$@"
}

gollie_predictions() {
  local tag="$1"
  local split="$2"
  case "${tag}" in
    base) echo "outputs/stage2_strategy_cot_e65/e57_cross_model_20260608/qwen4_gollie_style/checkpoint-2064/${split}/predictions.jsonl" ;;
    s8322) echo "outputs/strengthen_20260709/new/gollie_s8322/${split}/predictions.jsonl" ;;
    s8333) echo "outputs/strengthen_20260709/new/gollie_s8333/${split}/predictions.jsonl" ;;
    *) return 2 ;;
  esac
}

case "${1:-}" in
  smoke)
    resolve_tag "${2:-base}"
    docker_common --gpus "device=${3:-1}" "${IMAGE}" bash -lc \
      "$(declare -f sample_args); SCRIPT='${SCRIPT}'; MODEL='${MODEL}'; INPUT='${INPUT}'; OUTPUT='${OUTPUT}'; sample_args --max_examples 2 --overwrite --log_every 1"
    ;;
  sample-shard)
    resolve_tag "${2:?base, s8322, or s8333 required}"
    shard="${3:?shard index required}"
    gpu="${4:?gpu index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "$(declare -f sample_args); SCRIPT='${SCRIPT}'; MODEL='${MODEL}'; INPUT='${INPUT}'; OUTPUT='${OUTPUT}'; sample_args --num_shards 4 --shard_index '${shard}' --log_every 10"
    ;;
  sample-topup-shard)
    resolve_tag "${2:?base, s8322, or s8333 required}"
    shard="${3:?shard index required}"
    gpu="${4:?gpu index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "$(declare -f sample_args); SCRIPT='${SCRIPT}'; MODEL='${MODEL}'; INPUT='${INPUT}'; OUTPUT='${OUTPUT}'; sample_args --num_shards 4 --shard_index '${shard}' --sample_round 1 --wnd_ids_json '${OUTPUT}/topup_wnd_ids.json' --log_every 10"
    ;;
  build)
    resolve_tag "${2:?base, s8322, or s8333 required}"
    docker_common "${IMAGE}" python "${SCRIPT}" \
      --mode build \
      --model_path "${MODEL}" \
      --input_jsonl "${INPUT}" \
      --output_dir "${OUTPUT}" \
      --samples_glob "${OUTPUT}/samples.shard-*.jsonl" \
      --profile g9 \
      --cutoff_len 2048 \
      --min_pairs 900 \
      --dataset_dir /workspace/project/data/stage2_adaptive_datasets \
      --dataset_info /workspace/project/data/stage2_adaptive_datasets/dataset_info.json \
      --preference_name "${NAME}"
    ;;
  audit)
    resolve_tag "${2:?base, s8322, or s8333 required}"
    docker_common "${IMAGE}" python scripts/validate_reasoning_preferences_e110_20260712.py \
      --preference_jsonl "/workspace/project/data/stage2_adaptive_datasets/${NAME}.jsonl" \
      --input_jsonl "${INPUT}" \
      --model_path "${MODEL}" \
      --profile g9 \
      --cutoff_len 2048 \
      --min_pairs 900 \
      --output_json "${OUTPUT}/pair_audit.json"
    ;;
  train-base)
    docker_common --gpus "device=${2:-1}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${BASE_CONFIG}' 2>&1 | tee /workspace/project/outputs/stage2_preference_mining/e113a_g9/base_k4_seed1104/orpo_train.log"
    ;;
  eval-base)
    split="${2:?test_seen or test_unseen required}"
    gpu="${3:-1}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' \
        --adapter_path '${BASE_TRAINED_MODEL}' \
        --eval_jsonl '${EVAL_DATA}_${split}_pos.jsonl' \
        --output_dir '${BASE_EVAL_OUTPUT}/${split}' \
        --batch_size 4 --max_new_tokens 1024 --temperature 0.0 \
        2>&1 | tee /workspace/project/outputs/stage2_preference_mining/e113a_g9/base_k4_seed1104/orpo_${split}_eval.log"
    ;;
  compare-base)
    python scripts/compare_preference_run_gate_20260712.py \
      --baseline_seen outputs/stage2_preference_eval/e113_shared_recovery_g9_base_baseline/test_seen \
      --baseline_unseen outputs/stage2_preference_eval/e113_shared_recovery_g9_base_baseline/test_unseen \
      --candidate_seen outputs/stage2_preference_eval/e113b1_g9_orpo_base_seed42/test_seen \
      --candidate_unseen outputs/stage2_preference_eval/e113b1_g9_orpo_base_seed42/test_unseen \
      --output_dir outputs/stage2_preference_eval/e113b1_g9_orpo_base_seed42/gate \
      --gate g9 \
      --reference_sc_seen_event_gain 0.053858529897310936 \
      --bootstrap_samples 10000 --seed 20260712
    ;;
  eval-start)
    resolve_tag "${2:?base, s8322, or s8333 required}"
    split="${3:?test_seen or test_unseen required}"
    gpu="${4:?GPU index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' --adapter_path '${MODEL}' \
        --eval_jsonl '${EVAL_DATA}_${split}_pos.jsonl' \
        --output_dir '${BASELINE_OUTPUT}/${split}' \
        --batch_size 4 --max_new_tokens 1024 --temperature 0.0 \
        2>&1 | tee '${OUTPUT}/start_${split}_eval.log'"
    ;;
  train-tag)
    resolve_tag "${2:?base, s8322, or s8333 required}"
    gpu="${3:?GPU index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG}' 2>&1 | tee '${OUTPUT}/orpo_train.log'"
    ;;
  eval-tag)
    resolve_tag "${2:?base, s8322, or s8333 required}"
    split="${3:?test_seen or test_unseen required}"
    gpu="${4:?GPU index required}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
        --base_model '${BASE_MODEL}' --adapter_path '${TRAINED_MODEL}' \
        --eval_jsonl '${EVAL_DATA}_${split}_pos.jsonl' \
        --output_dir '${EVAL_OUTPUT}/${split}' \
        --batch_size 4 --max_new_tokens 1024 --temperature 0.0 \
        2>&1 | tee '${OUTPUT}/orpo_${split}_eval.log'"
    ;;
  compare-tag)
    resolve_tag "${2:?base, s8322, or s8333 required}"
    python scripts/compare_preference_run_gate_20260712.py \
      --baseline_seen "${BASELINE_OUTPUT#/workspace/project/}/test_seen" \
      --baseline_unseen "${BASELINE_OUTPUT#/workspace/project/}/test_unseen" \
      --candidate_seen "${EVAL_OUTPUT#/workspace/project/}/test_seen" \
      --candidate_unseen "${EVAL_OUTPUT#/workspace/project/}/test_unseen" \
      --output_dir "${EVAL_OUTPUT#/workspace/project/}/gate" \
      --gate g9 \
      --reference_sc_seen_event_gain 0.053858529897310936 \
      --bootstrap_samples 10000 --seed 20260712
    ;;
  sc-tag)
    resolve_tag "${2:?base, s8322, or s8333 required}"
    split="${3:?test_seen or test_unseen required}"
    gpu="${4:?GPU index required}"
    raw_output="/workspace/project/outputs/stage2_analysis/e113_g9_orpo_sc/${2}/${split}"
    docker_common --gpus "device=${gpu}" "${IMAGE}" bash -lc \
      "python scripts/self_consistency_eval_20260703.py \
        --model_path '${TRAINED_MODEL}' \
        --eval_jsonl '${EVAL_DATA}_${split}_pos.jsonl' \
        --output_dir '${raw_output}' \
        --n_samples 8 --temperature 0.8 --top_p 0.95 --max_new_tokens 1024 --vote_k 3 \
        --label 'e113_${2}_${split}' 2>&1 | tee '${OUTPUT}/orpo_sc_${split}.log'"
    ;;
  rescore-sc-tag)
    resolve_tag "${2:?base, s8322, or s8333 required}"
    split="${3:?test_seen or test_unseen required}"
    python scripts/rescore_self_consistency_samples_20260712.py \
      --samples_jsonl "outputs/stage2_analysis/e113_g9_orpo_sc/${2}/${split}/samples.jsonl" \
      --eval_jsonl "${EVAL_DATA#/workspace/project/}_${split}_pos.jsonl" \
      --output_dir "outputs/stage2_preference_eval/e113_g9_orpo_sc_k3/${2}/${split}" \
      --vote_k 3
    ;;
  rescore-gollie)
    tag="${2:?base, s8322, or s8333 required}"
    split="${3:?test_seen or test_unseen required}"
    input="$(gollie_predictions "${tag}" "${split}")"
    python scripts/rescore_surface_predictions_20260712.py \
      --input_predictions "${input}" \
      --output_dir "outputs/stage2_preference_eval/e113_shared_recovery_gollie_${tag}/${split}"
    ;;
  compare-n3)
    python scripts/compare_g9_n3_gate_20260712.py \
      --g9_start_seen \
        outputs/stage2_preference_eval/e113_shared_recovery_g9_base_baseline/test_seen \
        outputs/stage2_preference_eval/e113_shared_recovery_g9_s8322_baseline/test_seen \
        outputs/stage2_preference_eval/e113_shared_recovery_g9_s8333_baseline/test_seen \
      --g9_start_unseen \
        outputs/stage2_preference_eval/e113_shared_recovery_g9_base_baseline/test_unseen \
        outputs/stage2_preference_eval/e113_shared_recovery_g9_s8322_baseline/test_unseen \
        outputs/stage2_preference_eval/e113_shared_recovery_g9_s8333_baseline/test_unseen \
      --candidate_seen \
        outputs/stage2_preference_eval/e113b1_g9_orpo_base_seed42/test_seen \
        outputs/stage2_preference_eval/e113b2_g9_orpo_s8322_seed8322/test_seen \
        outputs/stage2_preference_eval/e113b3_g9_orpo_s8333_seed8333/test_seen \
      --candidate_unseen \
        outputs/stage2_preference_eval/e113b1_g9_orpo_base_seed42/test_unseen \
        outputs/stage2_preference_eval/e113b2_g9_orpo_s8322_seed8322/test_unseen \
        outputs/stage2_preference_eval/e113b3_g9_orpo_s8333_seed8333/test_unseen \
      --gollie_seen \
        outputs/stage2_preference_eval/e113_shared_recovery_gollie_base/test_seen \
        outputs/stage2_preference_eval/e113_shared_recovery_gollie_s8322/test_seen \
        outputs/stage2_preference_eval/e113_shared_recovery_gollie_s8333/test_seen \
      --output_dir outputs/stage2_preference_eval/e113_g9_orpo_n3_gate \
      --bootstrap_samples 10000 --seed 20260712
    ;;
  *)
    echo "usage: $0 smoke [tag] [gpu] | sample-shard <tag> <0..3> <gpu> | sample-topup-shard <tag> <0..3> <gpu> | build <tag> | audit <tag> | eval-start <tag> <split> <gpu> | train-tag <tag> <gpu> | eval-tag <tag> <split> <gpu> | compare-tag <tag> | rescore-gollie <tag> <split> | compare-n3 | sc-tag <tag> <split> <gpu> | rescore-sc-tag <tag> <split> | train-base [gpu] | eval-base <split> [gpu] | compare-base" >&2
    exit 2
    ;;
esac
