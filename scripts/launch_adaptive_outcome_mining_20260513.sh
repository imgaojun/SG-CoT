#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
CONTAINER_NAME="richere_qwen3_adaptive_outcome_mining_l15bal30_20260513"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-1.7B"
ADAPTER="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30_full/checkpoint-942"
DATA_PREFIX="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30"
OUT_ROOT="outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942"
LOG="outputs/stage2_adaptive_runs_user_logs/richere_qwen3_adaptive_outcome_mining_l15bal30_20260513.log"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "container already exists: ${CONTAINER_NAME}" >&2
  exit 1
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --user root \
  --gpus '"device=0,1,2,3"' \
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
    mkdir -p outputs/stage2_adaptive_runs_user_logs ${OUT_ROOT}
    {
      python src/stage2_cot/build_adaptive_route_reasoning_dataset.py \
        --schema_path data/schema/richere-en.event_schema.json \
        --direct_train_jsonl data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_train_pos.jsonl \
        --direct_dev_jsonl data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_dev_seen_pos.jsonl \
        --direct_test_jsonl data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_test_pos.jsonl \
        --direct_test_seen_jsonl data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_test_seen_pos.jsonl \
        --direct_test_unseen_jsonl data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_test_unseen_pos.jsonl \
        --train_label_jsonl data/stage2_adaptive_datasets/labels/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood_goldplan15_train_labels.jsonl \
        --dev_label_jsonl data/stage2_adaptive_datasets/labels/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood_goldplan15_dev_seen_labels.jsonl \
        --dataset_dir data/stage2_adaptive_datasets \
        --train_dataset_name richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30_train_pos \
        --dev_dataset_name richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30_dev_seen_pos \
        --test_dataset_name richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30_test_pos \
        --test_seen_dataset_name richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30_test_seen_pos \
        --test_unseen_dataset_name richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30_test_unseen_pos \
        --target_style type_role_hint_plan_lite \
        --max_role_checks_per_sample 6 \
        --seed 13 \
        --write_forced_eval_variants \
        --write_forced_train_variants
      CUDA_VISIBLE_DEVICES=0 python src/stage2_quality_validation/eval_adaptive_route_generation.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${ADAPTER} \
        --eval_jsonl ${DATA_PREFIX}_forced_direct_train_pos.jsonl \
        --output_dir ${OUT_ROOT}/forced_direct/train \
        --batch_size 8 \
        --max_new_tokens 512 \
        --temperature 0.0 &
      p0=\$!
      CUDA_VISIBLE_DEVICES=1 python src/stage2_quality_validation/eval_adaptive_route_generation.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${ADAPTER} \
        --eval_jsonl ${DATA_PREFIX}_forced_reason_train_pos.jsonl \
        --output_dir ${OUT_ROOT}/forced_reason/train \
        --batch_size 8 \
        --max_new_tokens 512 \
        --temperature 0.0 &
      p1=\$!
      CUDA_VISIBLE_DEVICES=2 python src/stage2_quality_validation/eval_adaptive_route_generation.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${ADAPTER} \
        --eval_jsonl ${DATA_PREFIX}_forced_direct_dev_seen_pos.jsonl \
        --output_dir ${OUT_ROOT}/forced_direct/dev_seen \
        --batch_size 8 \
        --max_new_tokens 512 \
        --temperature 0.0 &
      p2=\$!
      CUDA_VISIBLE_DEVICES=3 python src/stage2_quality_validation/eval_adaptive_route_generation.py \
        --base_model ${BASE_MODEL} \
        --adapter_path ${ADAPTER} \
        --eval_jsonl ${DATA_PREFIX}_forced_reason_dev_seen_pos.jsonl \
        --output_dir ${OUT_ROOT}/forced_reason/dev_seen \
        --batch_size 8 \
        --max_new_tokens 512 \
        --temperature 0.0 &
      p3=\$!
      wait \$p0
      wait \$p1
      wait \$p2
      wait \$p3
      bash scripts/build_adaptive_outcome_wave_20260513.sh
    } 2>&1 | tee ${LOG}
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} ${OUT_ROOT} data/stage2_adaptive_datasets configs/generated/stage2_adaptive outputs/stage2_adaptive_runs_user_logs
  "
