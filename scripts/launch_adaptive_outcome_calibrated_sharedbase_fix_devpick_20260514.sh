#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
DATA_PREFIX="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
FREE_PICK_PREFIX="outputs/stage2_adaptive_runs_user_devpick/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
FRONTIER_PICK_PREFIX="outputs/stage2_adaptive_runs_user_devpick_frontier/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
ROUTE_PICK_PREFIX="outputs/stage2_adaptive_runs_user_devpick_route/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
LOG_PREFIX="outputs/stage2_adaptive_runs_user_logs/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
GEN_EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation.py"
ROUTE_EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_choice.py"
SHORTLIST_SCRIPT="src/stage2_formal/parallel_shortlist_dev_select.py"

launch_branch() {
  local branch="$1"
  local host_gpus="$2"
  local name="adaptive_outcome_calibrated_sharedbase_fix_${branch}_devpick_20260514"
  local run_dir="${RUN_PREFIX}_${branch}_full"
  local free_jsonl="${DATA_PREFIX}_${branch}_dev_seen_pos.jsonl"
  local forced_direct_jsonl="${DATA_PREFIX}_${branch}_forced_direct_dev_seen_pos.jsonl"
  local forced_reason_jsonl="${DATA_PREFIX}_${branch}_forced_reason_dev_seen_pos.jsonl"
  local free_root="${FREE_PICK_PREFIX}_${branch}_full_free_dev_seen_max512"
  local direct_root="${FRONTIER_PICK_PREFIX}_${branch}_full_forced_direct_dev_seen_max512"
  local reason_root="${FRONTIER_PICK_PREFIX}_${branch}_full_forced_reason_dev_seen_max512"
  local route_root="${ROUTE_PICK_PREFIX}_${branch}_full_route_dev_seen_max16"
  local free_log="${LOG_PREFIX}_${branch}_full_devpick_free_max512.log"
  local direct_log="${LOG_PREFIX}_${branch}_full_devpick_forced_direct_max512.log"
  local reason_log="${LOG_PREFIX}_${branch}_full_devpick_forced_reason_max512.log"
  local route_log="${LOG_PREFIX}_${branch}_full_route_devpick_max16.log"
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
      mkdir -p outputs/stage2_adaptive_runs_user_logs outputs/stage2_adaptive_runs_user_devpick outputs/stage2_adaptive_runs_user_devpick_frontier outputs/stage2_adaptive_runs_user_devpick_route
      CKPTS=\$(find ${run_dir} -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
      if [[ -z \"\${CKPTS}\" ]]; then
        echo \"no checkpoints found under ${run_dir}\" >&2
        exit 1
      fi
      python ${SHORTLIST_SCRIPT} --base_model ${BASE_MODEL} --run_dir ${run_dir} --eval_jsonl ${free_jsonl} --output_root ${free_root} --checkpoint_tags \${CKPTS} --gpu_ids ${container_gpus} --metric_keys argument_f1 event_f1 trigger_f1 json_valid_rate --greater_is_better --batch_size 2 --max_new_tokens 512 --temperature 0.0 --eval_script ${GEN_EVAL_SCRIPT} --log_path ${free_log} --status_json ${free_root}/status.json --reuse_existing
      python ${SHORTLIST_SCRIPT} --base_model ${BASE_MODEL} --run_dir ${run_dir} --eval_jsonl ${forced_direct_jsonl} --output_root ${direct_root} --checkpoint_tags \${CKPTS} --gpu_ids ${container_gpus} --metric_keys argument_f1 event_f1 trigger_f1 json_valid_rate --greater_is_better --batch_size 2 --max_new_tokens 512 --temperature 0.0 --eval_script ${GEN_EVAL_SCRIPT} --log_path ${direct_log} --status_json ${direct_root}/status.json --reuse_existing
      python ${SHORTLIST_SCRIPT} --base_model ${BASE_MODEL} --run_dir ${run_dir} --eval_jsonl ${forced_reason_jsonl} --output_root ${reason_root} --checkpoint_tags \${CKPTS} --gpu_ids ${container_gpus} --metric_keys argument_f1 event_f1 trigger_f1 json_valid_rate --greater_is_better --batch_size 2 --max_new_tokens 512 --temperature 0.0 --eval_script ${GEN_EVAL_SCRIPT} --log_path ${reason_log} --status_json ${reason_root}/status.json --reuse_existing
      python ${SHORTLIST_SCRIPT} --base_model ${BASE_MODEL} --run_dir ${run_dir} --eval_jsonl ${free_jsonl} --output_root ${route_root} --checkpoint_tags \${CKPTS} --gpu_ids ${container_gpus} --metric_keys reason_f1 reason_recall reason_precision route_accuracy --greater_is_better --batch_size 8 --max_new_tokens 16 --temperature 0.0 --eval_script ${ROUTE_EVAL_SCRIPT} --log_path ${route_log} --status_json ${route_root}/status.json --reuse_existing
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} ${free_root} ${direct_root} ${reason_root} ${route_root} ${free_log} ${direct_log} ${reason_log} ${route_log}
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
