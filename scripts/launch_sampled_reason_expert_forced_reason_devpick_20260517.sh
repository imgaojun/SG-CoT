#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
DATA_PREFIX="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
PICK_PREFIX="outputs/stage2_sampled_supervision/reason_expert_devpick_20260517/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
LOG_PREFIX="outputs/stage2_adaptive_runs_user_logs/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
GEN_EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation.py"
SHORTLIST_SCRIPT="src/stage2_formal/parallel_shortlist_dev_select.py"

launch_branch() {
  local branch="$1"
  local host_gpus="$2"
  local name="sampled_reason_expert_${branch}_forced_reason_devpick_20260517"
  local run_dir="${RUN_PREFIX}_${branch}_full"
  local eval_jsonl="${DATA_PREFIX}_${branch}_forced_reason_dev_seen_pos.jsonl"
  local output_root="${PICK_PREFIX}_${branch}_forced_reason_dev_seen_max512"
  local log="${LOG_PREFIX}_${branch}_forced_reason_devpick_max512.log"
  local gpu_count
  local container_gpus

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists, skipping launch: ${name}" >&2
    return 0
  fi

  gpu_count="$(printf '%s\n' "${host_gpus}" | awk -F, '{print NF}')"
  container_gpus="$(seq 0 "$((gpu_count - 1))" | tr '\n' ' ' | sed 's/[[:space:]]*$//')"

  docker run -d \
    --name "${name}" \
    --user root \
    --gpus "\"device=${host_gpus}\"" \
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
      mkdir -p outputs/stage2_adaptive_runs_user_logs outputs/stage2_sampled_supervision/reason_expert_devpick_20260517
      CKPTS=\$(find ${run_dir} -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
      if [[ -z \"\${CKPTS}\" ]]; then
        echo \"no checkpoints found under ${run_dir}\" >&2
        exit 1
      fi
      python ${SHORTLIST_SCRIPT} --base_model ${BASE_MODEL} --run_dir ${run_dir} --eval_jsonl ${eval_jsonl} --output_root ${output_root} --checkpoint_tags \${CKPTS} --gpu_ids ${container_gpus} --metric_keys argument_f1 event_f1 trigger_f1 json_valid_rate --greater_is_better --batch_size 2 --max_new_tokens 512 --temperature 0.0 --eval_script ${GEN_EVAL_SCRIPT} --log_path ${log} --status_json ${output_root}/status.json --reuse_existing
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${output_root} ${log}
    "
}

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 branch=gpu[,gpu...] [branch=gpu[,gpu...] ...]" >&2
  exit 2
fi

for item in "$@"; do
  branch="${item%%=*}"
  gpu="${item#*=}"
  launch_branch "${branch}" "${gpu}"
done
