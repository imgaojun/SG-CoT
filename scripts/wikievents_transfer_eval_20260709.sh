#!/usr/bin/env bash
# WikiEvents (KAIROS) 第三本体零样本迁移评测 — Tier 1
# RichERE 训好的 e83(schema-driven SG-CoT)+ Direct 对照,直接评 WikiEvents test_seen/test_unseen。
# Direct: 输出 offset JSON -> eval_adapter_generation.py(offset-exact)
# e83   : 输出 surface-only <final> JSON -> eval_adaptive_route_generation_evidence.py(surface->offset 恢复)
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"

E83="/workspace/project/outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83_richere_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_full"
DIRECT="/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_repeat1_full"
DATA="/workspace/project/data/wikievents_transfer/wikievents_split1_oracle_mixed_noise_top10_shuffle"
OUT="/workspace/project/outputs/wikievents_transfer_eval"
EVAL_OFFSET="src/stage2_quality_validation/eval_adapter_generation.py"
EVAL_EVID="src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py"

mkdir -p "${PROJECT_ROOT}/outputs/wikievents_transfer_eval/logs"

run_eval() {
  local name="$1" gpu="$2" evalpy="$3" adapter="$4" split="$5" maxtok="$6"
  local outdir="${OUT}/${name}"
  local log="${OUT}/logs/${name}.log"
  docker run -d --user root --ipc host --shm-size 16g \
    --name "wikitransfer_${name}" --gpus "\"device=${gpu}\"" \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    -w /workspace/project "${IMAGE}" bash -lc "
      python ${evalpy} \
        --base_model '${BASE_MODEL}' \
        --adapter_path '${adapter}' \
        --eval_jsonl '${DATA}_${split}_pos.jsonl' \
        --output_dir '${outdir}' \
        --batch_size 4 \
        --max_new_tokens ${maxtok} 2>&1 | tee '${log}'
      HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
      chown -R \${HOST_UGID} '${outdir}' '${log}' || true
    "
  echo "launched wikitransfer_${name} on GPU ${gpu} -> ${outdir}"
}

# Direct(offset 评测器)                       name              gpu eval          adapter    split         maxtok
run_eval direct_test_seen     2 "${EVAL_OFFSET}" "${DIRECT}" test_seen   3072
run_eval direct_test_unseen   3 "${EVAL_OFFSET}" "${DIRECT}" test_unseen 3072
# e83 SG-CoT(evidence 恢复评测器)
run_eval e83_test_seen        5 "${EVAL_EVID}"   "${E83}"    test_seen   3072
run_eval e83_test_unseen      7 "${EVAL_EVID}"   "${E83}"    test_unseen 3072

echo "all 4 launched."
