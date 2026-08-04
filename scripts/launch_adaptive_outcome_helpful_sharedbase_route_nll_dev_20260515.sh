#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
DIRECT_PREFIX="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
SCOREBASE_BRANCH="outcome_l15bal30_15_routecls_scorebase"
EVAL_JSONL="${DATA_PREFIX}_${SCOREBASE_BRANCH}_dev_seen_pos.jsonl"
SCORE_ROOT="outputs/stage2_adaptive_route_likelihood_probe/outcome_helpful_sharedbase_20260515"
LOG_DIR="outputs/stage2_adaptive_runs_user_logs"
DATA_DIR="data/stage2_adaptive_datasets"
LABEL_DIR="${DATA_DIR}/labels"
SCHEMA="data/schema/richere-en.event_schema.json"
LABEL_SOURCE="outcome_l15bal30_15"

ensure_scorebase_dataset() {
  if [[ -s "${DATA_DIR}/${EVAL_JSONL}" ]]; then
    return 0
  fi

  python3 src/stage2_cot/build_adaptive_route_reasoning_dataset.py \
    --schema_path "${SCHEMA}" \
    --direct_train_jsonl "${DIRECT_PREFIX}_train_pos.jsonl" \
    --direct_dev_jsonl "${DIRECT_PREFIX}_dev_seen_pos.jsonl" \
    --direct_test_jsonl "${DIRECT_PREFIX}_test_pos.jsonl" \
    --direct_test_seen_jsonl "${DIRECT_PREFIX}_test_seen_pos.jsonl" \
    --direct_test_unseen_jsonl "${DIRECT_PREFIX}_test_unseen_pos.jsonl" \
    --train_label_jsonl "${LABEL_DIR}/${DATA_PREFIX}_${LABEL_SOURCE}_train_labels.jsonl" \
    --dev_label_jsonl "${LABEL_DIR}/${DATA_PREFIX}_${LABEL_SOURCE}_dev_seen_labels.jsonl" \
    --dataset_dir "${DATA_DIR}" \
    --train_dataset_name "${DATA_PREFIX}_${SCOREBASE_BRANCH}_train_pos" \
    --dev_dataset_name "${DATA_PREFIX}_${SCOREBASE_BRANCH}_dev_seen_pos" \
    --test_dataset_name "${DATA_PREFIX}_${SCOREBASE_BRANCH}_test_pos" \
    --test_seen_dataset_name "${DATA_PREFIX}_${SCOREBASE_BRANCH}_test_seen_pos" \
    --test_unseen_dataset_name "${DATA_PREFIX}_${SCOREBASE_BRANCH}_test_unseen_pos" \
    --target_style type_role_hint_plan_lite \
    --max_role_checks_per_sample 6 \
    --seed 15 \
    --route_only_train \
    --route_only_eval \
    --route_classifier_prompt
}

launch_branch() {
  local branch="$1"
  local host_gpu="$2"
  local name="adaptive_outcome_helpful_sharedbase_route_nll_${branch}_20260515"
  local run_dir="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${branch}_full"
  local branch_score_root="${SCORE_ROOT}/${branch}"
  local log="${LOG_DIR}/adaptive_outcome_helpful_sharedbase_route_nll_${branch}_20260515.log"

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists, skipping launch: ${name}" >&2
    return 0
  fi

  docker run -d \
    --name "${name}" \
    --user root \
    --gpus "\"device=${host_gpu}\"" \
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
      mkdir -p ${branch_score_root} ${LOG_DIR}
      CKPTS=\$(find ${run_dir} -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
      if [[ -z \"\${CKPTS}\" ]]; then
        echo \"no checkpoints found under ${run_dir}\" >&2
        exit 1
      fi
      : > ${log}
      for ckpt in \${CKPTS}; do
        mkdir -p ${branch_score_root}/\${ckpt}
        echo \"[route-nll] ${branch} \${ckpt}\" | tee -a ${log}
        python src/stage2_quality_validation/score_adaptive_route_choice_likelihood.py \
          --base_model ${BASE_MODEL} \
          --adapter_path ${run_dir}/\${ckpt} \
          --eval_jsonl data/stage2_adaptive_datasets/${EVAL_JSONL} \
          --output_jsonl ${branch_score_root}/\${ckpt}/dev_seen_scores.jsonl \
          --summary_json ${branch_score_root}/\${ckpt}/dev_seen_summary.json \
          --max_length 1024 2>&1 | tee -a ${log}
      done
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${branch_score_root} ${log}
    "
}

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 branch=gpu [branch=gpu ...]" >&2
  exit 2
fi

ensure_scorebase_dataset
for item in "$@"; do
  branch="${item%%=*}"
  gpu="${item#*=}"
  launch_branch "${branch}" "${gpu}"
done
