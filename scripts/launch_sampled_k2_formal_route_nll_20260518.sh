#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH="${BRANCH:-sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25}"
FORMAL_DATASET_ID="${FORMAL_DATASET_ID:-sampled_k2_formal_route_nll_probe_20260518}"
EXPERIMENT_ID="${EXPERIMENT_ID:-sampled_k2_formal_route_nll_probe_20260518}"
CHECKPOINTS="${CHECKPOINTS:-checkpoint-50 checkpoint-75}"
SPLITS="${SPLITS:-test_seen test_unseen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage2_adaptive_route_formal_nll_20260518/${BRANCH}}"
LOG="${LOG:-outputs/stage2_adaptive_runs_user_logs/${EXPERIMENT_ID}.log}"
HOST_GPU="${1:-0}"
NAME="${EXPERIMENT_ID}_score"

for ckpt in ${CHECKPOINTS}; do
  ADAPTER="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full/${ckpt}"
  if [[ ! -d "${ADAPTER}" ]]; then
    echo "missing adapter: ${ADAPTER}" >&2
    exit 1
  fi
done

if docker ps -a --format '{{.Names}}' | grep -Fxq "${NAME}"; then
  echo "container already exists, skipping launch: ${NAME}" >&2
  exit 0
fi

docker run -d \
  --name "${NAME}" \
  --user root \
  --gpus "\"device=${HOST_GPU}\"" \
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
    mkdir -p ${OUTPUT_ROOT} \$(dirname ${LOG})
    : > ${LOG}
    for ckpt in ${CHECKPOINTS}; do
      adapter=\"outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full/\${ckpt}\"
      for split in ${SPLITS}; do
        eval_jsonl=\"data/stage2_adaptive_datasets/${DATA_PREFIX}_${FORMAL_DATASET_ID}_\${split}_pos.jsonl\"
        out_dir=\"${OUTPUT_ROOT}/\${ckpt}/\${split}\"
        if [[ ! -s \"\${eval_jsonl}\" ]]; then
          echo \"missing eval jsonl: \${eval_jsonl}\" >&2
          exit 1
        fi
        if [[ -s \"\${out_dir}/scores.jsonl\" && -s \"\${out_dir}/summary.json\" ]]; then
          echo \"skip existing \${ckpt} \${split}\" | tee -a ${LOG}
          continue
        fi
        mkdir -p \"\${out_dir}\"
        echo \"[formal-route-nll] \${ckpt} \${split}\" | tee -a ${LOG}
        python src/stage2_quality_validation/score_adaptive_route_choice_likelihood.py \
          --base_model ${BASE_MODEL} \
          --adapter_path \"\${adapter}\" \
          --eval_jsonl \"\${eval_jsonl}\" \
          --output_jsonl \"\${out_dir}/scores.jsonl\" \
          --summary_json \"\${out_dir}/summary.json\" \
          --max_length 1024 2>&1 | tee -a ${LOG}
      done
    done
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${OUTPUT_ROOT} ${LOG}
  "
