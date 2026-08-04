#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
BRANCH="pairall_type_role_hint_plan_lite_scorer"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_DIR="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full"
DATA_PREFIX="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_pairall_type_role_hint_plan_lite"
SCORE_DIR="outputs/stage2_adaptive_likelihood_scores/pairall_type_role_hint_plan_lite_scorer"
LABEL_DIR="data/stage2_adaptive_datasets/labels"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"
CONTAINER="adaptive_likelihood_pairall_scorer_score_labels_20260512"
GPU="${1:-3}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER}"; then
  echo "container already exists: ${CONTAINER}" >&2
  exit 1
fi

docker run -d \
  --name "${CONTAINER}" \
  --user root \
  --gpus "\"device=${GPU}\"" \
  --ipc host \
  --shm-size 16g \
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
  -e WANDB_DIR=/workspace/logs/wandb \
  -w /workspace/project \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    mkdir -p ${SCORE_DIR} ${LABEL_DIR} ${LOG_DIR}
    CKPT=\$(ls -d ${RUN_DIR}/checkpoint-* | sort -V | tail -n 1)
    echo \"using checkpoint: \${CKPT}\" | tee ${LOG_DIR}/${RUN_PREFIX}_${BRANCH}_likelihood_scoring.log
    python src/stage2_quality_validation/score_adaptive_paired_likelihood.py \
      --base_model ${BASE_MODEL} \
      --adapter_path \${CKPT} \
      --paired_jsonl ${DATA_PREFIX}_train_pos.jsonl \
      --output_jsonl ${SCORE_DIR}/train.jsonl \
      --summary_json ${SCORE_DIR}/train.summary.json \
      --split train \
      --max_length 1024 2>&1 | tee -a ${LOG_DIR}/${RUN_PREFIX}_${BRANCH}_likelihood_scoring.log
    python src/stage2_quality_validation/score_adaptive_paired_likelihood.py \
      --base_model ${BASE_MODEL} \
      --adapter_path \${CKPT} \
      --direct_jsonl ${DATA_PREFIX}_forced_direct_dev_seen_pos.jsonl \
      --reason_jsonl ${DATA_PREFIX}_forced_reason_dev_seen_pos.jsonl \
      --output_jsonl ${SCORE_DIR}/dev_seen.jsonl \
      --summary_json ${SCORE_DIR}/dev_seen.summary.json \
      --split dev_seen \
      --max_length 1024 2>&1 | tee -a ${LOG_DIR}/${RUN_PREFIX}_${BRANCH}_likelihood_scoring.log
    python src/stage2_analysis/build_adaptive_likelihood_route_labels.py \
      --scores_jsonl ${SCORE_DIR}/train.jsonl \
      --output_jsonl ${LABEL_DIR}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood_goldplan10_train_labels.jsonl \
      --summary_json ${LABEL_DIR}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood_goldplan10_train_labels.summary.json \
      --reason_rate_cap 0.10 \
      --margin 0.0 \
      --label_source likelihood_goldplan10
    python src/stage2_analysis/build_adaptive_likelihood_route_labels.py \
      --scores_jsonl ${SCORE_DIR}/train.jsonl \
      --output_jsonl ${LABEL_DIR}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood_goldplan15_train_labels.jsonl \
      --summary_json ${LABEL_DIR}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood_goldplan15_train_labels.summary.json \
      --reason_rate_cap 0.15 \
      --margin 0.0 \
      --label_source likelihood_goldplan15
    chown -R 1000:1000 ${SCORE_DIR} ${LABEL_DIR}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood_goldplan* ${LOG_DIR}/${RUN_PREFIX}_${BRANCH}_likelihood_scoring.log
  "
