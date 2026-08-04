#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX="data/stage2_adaptive_datasets/${ADAPT_PREFIX}"
OUTPUT_ROOT="outputs/stage2_modular_dualexpert/sampled_counterfactual_utility_20260517"
LOG_DIR="outputs/stage2_modular_dualexpert/sampled_counterfactual_utility_20260517/logs"
DIRECT_BRANCH="outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux"
REASON_BRANCH="sampled_reason_expert_forcedreason_from_noaux_20260517"
DIRECT_ADAPTER="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${DIRECT_BRANCH}_full/checkpoint-1930"
SEEDS=(17 18 19 20 21 22 23 24)
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <reason_checkpoint_tag> <direct_gpu> <reason_gpu>" >&2
  echo "example: $0 checkpoint-771 0 1" >&2
  exit 2
fi

REASON_CKPT="$1"
DIRECT_GPU="$2"
REASON_GPU="$3"
RUN_ID="${REASON_BRANCH}_${REASON_CKPT}"
REASON_ADAPTER="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${REASON_BRANCH}_full/${REASON_CKPT}"

launch_route() {
  local route="$1"
  local host_gpu="$2"
  local adapter="$3"
  local name="sampled_counterfactual_k8_${route}_${REASON_CKPT}_20260517"
  local log="${LOG_DIR}/${route}_${REASON_CKPT}.log"

  if [[ ! -d "${adapter}" ]]; then
    echo "missing adapter: ${adapter}" >&2
    return 1
  fi

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
      mkdir -p ${LOG_DIR}
      exec > >(tee -a ${log}) 2>&1
      for split in train dev_seen; do
        if [[ \"${route}\" == \"direct\" ]]; then
          eval_jsonl=\"${DATA_PREFIX}_${REASON_BRANCH}_forced_direct_\${split}_pos.jsonl\"
        else
          eval_jsonl=\"${DATA_PREFIX}_${REASON_BRANCH}_forced_reason_\${split}_pos.jsonl\"
        fi
        for seed in ${SEEDS[*]}; do
          out_dir=\"${OUTPUT_ROOT}/${RUN_ID}/\${split}/${route}/seed-\${seed}\"
          if [[ -s \"\${out_dir}/predictions.jsonl\" ]]; then
            echo \"skip existing ${route} \${split} seed-\${seed}\"
            continue
          fi
          python src/stage2_quality_validation/eval_adaptive_route_generation_samples.py \
            --base_model ${BASE_MODEL} \
            --adapter_path ${adapter} \
            --eval_jsonl \"\${eval_jsonl}\" \
            --output_dir \"\${out_dir}\" \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature ${TEMPERATURE} \
            --top_p ${TOP_P} \
            --top_k ${TOP_K} \
            --batch_size ${BATCH_SIZE} \
            --seed \"\${seed}\" \
            --sample_id \"seed-\${seed}\" \
            --route_mode ${route}
        done
      done
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${OUTPUT_ROOT}/${RUN_ID} ${log}
    "
}

launch_route direct "${DIRECT_GPU}" "${DIRECT_ADAPTER}"
launch_route reason "${REASON_GPU}" "${REASON_ADAPTER}"
