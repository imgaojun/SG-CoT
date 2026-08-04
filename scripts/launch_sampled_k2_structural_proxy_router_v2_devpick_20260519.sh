#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
BRANCH="sampled_k2_structproxy_strictv2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
DATA_PREFIX="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
PICK_PREFIX="outputs/stage2_adaptive_runs_user_devpick_route/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
LOG_PREFIX="outputs/stage2_adaptive_runs_user_logs/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
EVAL_SCRIPT="src/stage2_quality_validation/eval_adaptive_route_choice.py"
SHORTLIST_SCRIPT="src/stage2_formal/parallel_shortlist_dev_select.py"

host_gpu="${1:-0}"
name="sampled_k2_structproxy_v2_router_route_devpick_20260519"
run_dir="${RUN_PREFIX}_${BRANCH}_full"
eval_jsonl="${DATA_PREFIX}_${BRANCH}_dev_seen_seedpairs_pos.jsonl"
output_root="${PICK_PREFIX}_${BRANCH}_full_route_dev_seen_seedpairs_max16"
log="${LOG_PREFIX}_${BRANCH}_full_route_devpick_seedpairs_max16.log"

if [[ ! -s "${eval_jsonl}" ]]; then
  echo "missing eval jsonl: ${eval_jsonl}" >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
  echo "container already exists: ${name}" >&2
  exit 1
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
    mkdir -p outputs/stage2_adaptive_runs_user_logs outputs/stage2_adaptive_runs_user_devpick_route
    CKPTS=\$(find ${run_dir} -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')
    if [[ -z \"\${CKPTS}\" ]]; then
      echo \"no checkpoints found under ${run_dir}\" >&2
      exit 1
    fi
    python ${SHORTLIST_SCRIPT} \
      --base_model ${BASE_MODEL} \
      --run_dir ${run_dir} \
      --eval_jsonl ${eval_jsonl} \
      --output_root ${output_root} \
      --checkpoint_tags \${CKPTS} \
      --gpu_ids 0 \
      --metric_keys reason_f1 reason_recall reason_precision route_accuracy \
      --greater_is_better \
      --batch_size 8 \
      --max_new_tokens 16 \
      --temperature 0.0 \
      --eval_script ${EVAL_SCRIPT} \
      --log_path ${log} \
      --status_json ${output_root}/status.json \
      --reuse_existing
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${output_root} ${log}
  "
