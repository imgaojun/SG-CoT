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
LOG_PREFIX="outputs/stage2_adaptive_runs_user_logs/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_generation.py"
SHORTLIST_SCRIPT="src/stage2_formal/parallel_shortlist_dev_select.py"
COMMON_CKPTS="checkpoint-129 checkpoint-258 checkpoint-387 checkpoint-516 checkpoint-645 checkpoint-774 checkpoint-903 checkpoint-1032 checkpoint-1161 checkpoint-1290 checkpoint-1419 checkpoint-1548 checkpoint-1677 checkpoint-1806 checkpoint-1935 checkpoint-2064"
DUP_CKPTS="checkpoint-142 checkpoint-284 checkpoint-426 checkpoint-568 checkpoint-710 checkpoint-852 checkpoint-994 checkpoint-1136 checkpoint-1278 checkpoint-1420 checkpoint-1562 checkpoint-1704 checkpoint-1846 checkpoint-1988 checkpoint-2130 checkpoint-2272"

launch_branch() {
  local branch="$1"
  local host_gpus="$2"
  local ckpts="$3"
  local name="adaptive_hardconf_${branch}_devpick_20260510"
  local run_dir="${RUN_PREFIX}_${branch}_full"
  local free_jsonl="${DATA_PREFIX}_${branch}_dev_seen_pos.jsonl"
  local forced_direct_jsonl="${DATA_PREFIX}_${branch}_forced_direct_dev_seen_pos.jsonl"
  local forced_reason_jsonl="${DATA_PREFIX}_${branch}_forced_reason_dev_seen_pos.jsonl"
  local free_root="${FREE_PICK_PREFIX}_${branch}_full_free_dev_seen_max512"
  local direct_root="${FRONTIER_PICK_PREFIX}_${branch}_full_forced_direct_dev_seen_max512"
  local reason_root="${FRONTIER_PICK_PREFIX}_${branch}_full_forced_reason_dev_seen_max512"
  local free_log="${LOG_PREFIX}_${branch}_full_devpick_free_max512.log"
  local direct_log="${LOG_PREFIX}_${branch}_full_devpick_forced_direct_max512.log"
  local reason_log="${LOG_PREFIX}_${branch}_full_devpick_forced_reason_max512.log"

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "container already exists, skipping launch: ${name}" >&2
    return 0
  fi

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
      mkdir -p outputs/stage2_adaptive_runs_user_logs outputs/stage2_adaptive_runs_user_devpick outputs/stage2_adaptive_runs_user_devpick_frontier
      python ${SHORTLIST_SCRIPT} \
        --base_model ${BASE_MODEL} \
        --run_dir ${run_dir} \
        --eval_jsonl ${free_jsonl} \
        --output_root ${free_root} \
        --checkpoint_tags ${ckpts} \
        --gpu_ids 0 \
        --metric_keys argument_f1 event_f1 trigger_f1 \
        --greater_is_better \
        --batch_size 2 \
        --max_new_tokens 512 \
        --temperature 0.0 \
        --eval_script ${EVAL_SCRIPT} \
        --log_path ${free_log} \
        --status_json ${free_root}/status.json \
        --reuse_existing
      python ${SHORTLIST_SCRIPT} \
        --base_model ${BASE_MODEL} \
        --run_dir ${run_dir} \
        --eval_jsonl ${forced_direct_jsonl} \
        --output_root ${direct_root} \
        --checkpoint_tags ${ckpts} \
        --gpu_ids 0 \
        --metric_keys argument_f1 event_f1 trigger_f1 \
        --greater_is_better \
        --batch_size 2 \
        --max_new_tokens 512 \
        --temperature 0.0 \
        --eval_script ${EVAL_SCRIPT} \
        --log_path ${direct_log} \
        --status_json ${direct_root}/status.json \
        --reuse_existing
      python ${SHORTLIST_SCRIPT} \
        --base_model ${BASE_MODEL} \
        --run_dir ${run_dir} \
        --eval_jsonl ${forced_reason_jsonl} \
        --output_root ${reason_root} \
        --checkpoint_tags ${ckpts} \
        --gpu_ids 0 \
        --metric_keys argument_f1 event_f1 trigger_f1 \
        --greater_is_better \
        --batch_size 2 \
        --max_new_tokens 512 \
        --temperature 0.0 \
        --eval_script ${EVAL_SCRIPT} \
        --log_path ${reason_log} \
        --status_json ${reason_root}/status.json \
        --reuse_existing
      chown -R 1000:1000 \
        ${free_root} \
        ${direct_root} \
        ${reason_root} \
        ${free_log} \
        ${direct_log} \
        ${reason_log}
    "
}

launch_branch "hardconf10_heur10_type_role_hint_plan_lite" "0" "${COMMON_CKPTS}"
launch_branch "hardconf15_heur15_type_role_hint_plan_lite" "1" "${COMMON_CKPTS}"
launch_branch "hardconf10_calibrated_type_role_hint_plan_lite" "0" "${COMMON_CKPTS}"
launch_branch "hardconf10_directdup" "3" "${DUP_CKPTS}"
