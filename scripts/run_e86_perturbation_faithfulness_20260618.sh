#!/usr/bin/env bash
# E86 perturbation faithfulness: docker wrapper. Arg1=GPU index, arg2=mode (smoke|full).
set -euo pipefail
cd /mnt/disk/gaojun/research/progressive-ee

GPU="${1:-0}"
MODE="${2:-full}"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
MODEL_ROOT="/mnt/disk/gaojun/models"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
IMAGE="llamafactory-lab:0.9.4-py3.12"

BASE_QWEN4="/workspace/models/LLM-Research/Qwen3-4B"
CKPT="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full/checkpoint-273"
PRED="/workspace/project/outputs/stage2_strategy_cot_e65/e57_cross_model_20260608/qwen4_e81_trigger_locked_arbitration/checkpoint-273"

if [ "${MODE}" = "smoke" ]; then
  # tiny copies for a fast correctness check
  mkdir -p outputs/stage2_e86_perturbation_faithfulness_20260618/_smoke
  head -6 "${PROJECT_ROOT}/outputs/stage2_strategy_cot_e65/e57_cross_model_20260608/qwen4_e81_trigger_locked_arbitration/checkpoint-273/test_unseen/predictions.jsonl" \
    > outputs/stage2_e86_perturbation_faithfulness_20260618/_smoke/seen.jsonl
  head -6 "${PROJECT_ROOT}/outputs/stage2_strategy_cot_e65/e57_cross_model_20260608/qwen4_e81_trigger_locked_arbitration/checkpoint-273/test_unseen/predictions.jsonl" \
    > outputs/stage2_e86_perturbation_faithfulness_20260618/_smoke/unseen.jsonl
  PRED_SEEN="/workspace/project/outputs/stage2_e86_perturbation_faithfulness_20260618/_smoke/seen.jsonl"
  PRED_UNSEEN="/workspace/project/outputs/stage2_e86_perturbation_faithfulness_20260618/_smoke/unseen.jsonl"
  OUT="/workspace/project/outputs/stage2_e86_perturbation_faithfulness_20260618/_smoke/out"
  NAME="e86_perturb_smoke_20260618"
else
  PRED_SEEN="${PRED}/test_seen/predictions.jsonl"
  PRED_UNSEEN="${PRED}/test_unseen/predictions.jsonl"
  OUT="/workspace/project/outputs/stage2_e86_perturbation_faithfulness_20260618/e81_ck273"
  NAME="e86_perturb_full_20260618"
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run --rm --user root --ipc host --shm-size 16g \
  -v "${PROJECT_ROOT}:/workspace/project" \
  -v "${MODEL_ROOT}:/workspace/models" \
  -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
  -e PYTHONUNBUFFERED=1 -e HF_HOME=/workspace/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
  --gpus "device=${GPU}" --name "${NAME}" -w /workspace/project "${IMAGE}" bash -lc "
    set -euo pipefail
    python scripts/eval_e86_perturbation_faithfulness_20260618.py \
      --base_model '${BASE_QWEN4}' --model_path '${CKPT}' \
      --pred_seen '${PRED_SEEN}' --pred_unseen '${PRED_UNSEEN}' \
      --output_dir '${OUT}' --batch_size 8 --max_new_tokens 640
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '/workspace/project/outputs/stage2_e86_perturbation_faithfulness_20260618' || true
  "
