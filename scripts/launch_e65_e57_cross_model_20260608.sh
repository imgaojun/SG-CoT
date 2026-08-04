#!/usr/bin/env bash
set -euo pipefail

IMAGE="llamafactory-lab:0.9.4-py3.12"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
MODEL_ROOT="/mnt/disk/gaojun/models"

BASE_QWEN17="/workspace/models/LLM-Research/Qwen3-1.7B"
BASE_QWEN4="/workspace/models/LLM-Research/Qwen3-4B"
BASE_LLAMA3="/workspace/models/LLM-Research/Llama-3.2-3B-Instruct"
DATA_BRANCH="e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot"
DATA_BRANCH_E70="e70_candidate_audit_v2_glm51_full1500_thinking_evidence_cot"
DATA_BRANCH_E72="e72_e57_backbone_subtype_minarg_glm51_full1500_thinking_evidence_cot"
DATA_BRANCH_E73="e73_e57_recall_first_exactness_last_glm51_full1500_thinking_evidence_cot"
DATA_BRANCH_E75="e73_e57_recall_first_exactness_last_glm51_3k_thinking_evidence_cot"
DATA_BRANCH_E76="e76_contrastive_exactness_glm51_full1500_thinking_evidence_cot"
DATA_BRANCH_E77="e77_e76_direct_control"
DATA_BRANCH_E80A="e80a_no_type_arbitration_glm51_full1500_thinking_evidence_cot"
DATA_BRANCH_E80B="e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot"
DATA_BRANCH_E81="e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot"
DATA_BASE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${DATA_BRANCH}"
DATA_BASE_E70="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${DATA_BRANCH_E70}"
DATA_BASE_E72="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${DATA_BRANCH_E72}"
DATA_BASE_E73="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${DATA_BRANCH_E73}"
DATA_BASE_E75="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${DATA_BRANCH_E75}"
DATA_BASE_E76="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${DATA_BRANCH_E76}"
DATA_BASE_E77="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${DATA_BRANCH_E77}"
DATA_BASE_E80A="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${DATA_BRANCH_E80A}"
DATA_BASE_E80B="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${DATA_BRANCH_E80B}"
DATA_BASE_E81="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_${DATA_BRANCH_E81}"
EVAL_SCRIPT_EVIDENCE="src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py"
EVAL_SCRIPT_DIRECT="src/stage2_quality_validation/eval_adapter_generation.py"

CONFIG_QWEN4_E70="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e70_candidate_audit_v2_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_QWEN4_E72="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e72_e57_backbone_subtype_minarg_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_QWEN4_E73="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e73_e57_recall_first_exactness_last_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_QWEN4_E73_REPEAT1="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e73_e57_recall_first_exactness_last_glm51_full1500_thinking_evidence_cot_repeat1_full_stepmatch.yaml"
CONFIG_QWEN4_E73_REPEAT2="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e73_e57_recall_first_exactness_last_glm51_full1500_thinking_evidence_cot_repeat2_full_stepmatch.yaml"
CONFIG_QWEN4_E75="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e73_e57_recall_first_exactness_last_glm51_3k_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_QWEN4_E76="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e76_contrastive_exactness_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_QWEN4_E76_REPEAT1_2EP="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e76_contrastive_exactness_glm51_full1500_thinking_evidence_cot_repeat1_2ep_full_stepmatch.yaml"
CONFIG_QWEN4_E76_REPEAT2_2EP="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e76_contrastive_exactness_glm51_full1500_thinking_evidence_cot_repeat2_2ep_full_stepmatch.yaml"
CONFIG_QWEN4_E77="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e77_e76_direct_control_full_stepmatch.yaml"
CONFIG_QWEN4_E80A="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80a_no_type_arbitration_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_QWEN4_E80B="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_QWEN4_E80B_REPEAT1="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot_repeat1_full_stepmatch.yaml"
CONFIG_QWEN4_E80B_REPEAT2="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot_repeat2_full_stepmatch.yaml"
CONFIG_QWEN4_E81="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_QWEN4_E81_REPEAT1="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat1_full_stepmatch.yaml"
CONFIG_QWEN4_E81_REPEAT2="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat2_full_stepmatch.yaml"
CONFIG_QWEN4_E81_REPEAT3="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat3_full_stepmatch.yaml"
RUN_QWEN4_E81_REPEAT3="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat3_full"
CONFIG_QWEN4_E81_REPEAT4="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat4_full_stepmatch.yaml"
RUN_QWEN4_E81_REPEAT4="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat4_full"
CONFIG_QWEN4_E57_REPEAT3="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat3_full_stepmatch.yaml"
RUN_QWEN4_E57_REPEAT3="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat3_full"
CONFIG_QWEN4_E57_REPEAT4="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat4_full_stepmatch.yaml"
RUN_QWEN4_E57_REPEAT4="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat4_full"
CONFIG_QWEN4_ACE05_DIRECT="configs/generated/stage2_formal/ace05_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full_stepmatch.yaml"
CONFIG_QWEN4_ACE05_E83="configs/generated/stage2_adaptive/ace05_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83_ace05_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
DATA_BASE_ACE05_E83="data/stage2_adaptive_datasets/ace05_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e83_ace05_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot"
CONFIG_QWEN4_ACE05_E84="configs/generated/stage2_adaptive/ace05_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e84_ace05_no_arbitration_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
RUN_QWEN4_ACE05_E84="outputs/stage2_adaptive_runs_user/ace05_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e84_ace05_no_arbitration_glm51_full1500_thinking_evidence_cot_full"
DATA_BASE_ACE05_E84="data/stage2_adaptive_datasets/ace05_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e84_ace05_no_arbitration_glm51_full1500_thinking_evidence_cot"
CONFIG_QWEN4_E85="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e85_secondteacher_trigger_locked_arbitration_deepseek_full1500_thinking_evidence_cot_full_stepmatch.yaml"
RUN_QWEN4_E85="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e85_secondteacher_trigger_locked_arbitration_deepseek_full1500_thinking_evidence_cot_full"
DATA_BASE_E85="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e85_secondteacher_trigger_locked_arbitration_deepseek_full1500_thinking_evidence_cot"
DATA_BASE_E56="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot"
CONFIG_QWEN4_E83_RICHERE="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83_richere_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
RUN_QWEN4_E83_RICHERE="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83_richere_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_full"
DATA_BASE_E83_RICHERE="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e83_richere_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot"
DATA_BASE_RICHERE_DIRECT="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
CONFIG_QWEN4_E57_REPEAT1="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat1_full_stepmatch.yaml"
CONFIG_QWEN4_E57_REPEAT2="configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat2_full_stepmatch.yaml"
RUN_QWEN4_E57_REPEAT1="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat1_full"
RUN_QWEN4_E57_REPEAT2="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat2_full"
CONFIG_QWEN4_DIRECT_REPEAT1="configs/generated/stage2_cot/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_repeat1_full_stepmatch.yaml"
CONFIG_QWEN4_DIRECT_REPEAT2="configs/generated/stage2_cot/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_repeat2_full_stepmatch.yaml"
RUN_QWEN4_DIRECT_REPEAT1="outputs/stage2_full_sft_runs_stepmatch_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_repeat1_full"
RUN_QWEN4_DIRECT_REPEAT2="outputs/stage2_full_sft_runs_stepmatch_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_repeat2_full"
CONFIG_QWEN17="configs/generated/stage2_adaptive/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_e65_e57cross_full_stepmatch.yaml"
CONFIG_QWEN17_E67="configs/generated/stage2_adaptive/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_e67_e57cross_from_e66_ckpt2451_full_stepmatch.yaml"
CONFIG_QWEN17_E70="configs/generated/stage2_adaptive/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e70_candidate_audit_v2_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_LLAMA3="configs/generated/stage2_adaptive/richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_e65_e57cross_full_stepmatch.yaml"
CONFIG_LLAMA3_E70="configs/generated/stage2_adaptive/richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e70_candidate_audit_v2_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_QWEN17_E80B="configs/generated/stage2_adaptive/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_LLAMA3_E80B="configs/generated/stage2_adaptive/richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_QWEN17_E81="configs/generated/stage2_adaptive/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
CONFIG_LLAMA3_E81="configs/generated/stage2_adaptive/richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml"
RUN_QWEN4_E70="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e70_candidate_audit_v2_glm51_full1500_thinking_evidence_cot_full"
RUN_QWEN4_E72="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e72_e57_backbone_subtype_minarg_glm51_full1500_thinking_evidence_cot_full"
RUN_QWEN4_E73="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e73_e57_recall_first_exactness_last_glm51_full1500_thinking_evidence_cot_full"
RUN_QWEN4_E73_REPEAT1="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e73_e57_recall_first_exactness_last_glm51_full1500_thinking_evidence_cot_repeat1_full"
RUN_QWEN4_E73_REPEAT2="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e73_e57_recall_first_exactness_last_glm51_full1500_thinking_evidence_cot_repeat2_full"
RUN_QWEN4_E75="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e73_e57_recall_first_exactness_last_glm51_3k_thinking_evidence_cot_full"
RUN_QWEN4_E76="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e76_contrastive_exactness_glm51_full1500_thinking_evidence_cot_full"
RUN_QWEN4_E76_REPEAT1_2EP="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e76_contrastive_exactness_glm51_full1500_thinking_evidence_cot_repeat1_2ep_full"
RUN_QWEN4_E76_REPEAT2_2EP="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e76_contrastive_exactness_glm51_full1500_thinking_evidence_cot_repeat2_2ep_full"
RUN_QWEN4_E77="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e77_e76_direct_control_full"
RUN_QWEN4_E80A="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80a_no_type_arbitration_glm51_full1500_thinking_evidence_cot_full"
RUN_QWEN4_E80B="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot_full"
RUN_QWEN4_E80B_REPEAT1="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot_repeat1_full"
RUN_QWEN4_E80B_REPEAT2="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot_repeat2_full"
RUN_QWEN4_E81="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full"
RUN_QWEN4_E81_REPEAT1="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat1_full"
RUN_QWEN4_E81_REPEAT2="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat2_full"
RUN_QWEN4_ACE05_DIRECT="outputs/stage2_full_sft_runs_stepmatch_user/ace05_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full"
RUN_QWEN4_ACE05_E83="outputs/stage2_adaptive_runs_user/ace05_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83_ace05_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_full"
RUN_QWEN17="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_e65_e57cross_full"
RUN_QWEN17_E67="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_e67_e57cross_from_e66_ckpt2451_full"
RUN_QWEN17_E70="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e70_candidate_audit_v2_glm51_full1500_thinking_evidence_cot_full"
RUN_LLAMA3="outputs/stage2_adaptive_runs_user/richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_e65_e57cross_full"
RUN_LLAMA3_E70="outputs/stage2_adaptive_runs_user/richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e70_candidate_audit_v2_glm51_full1500_thinking_evidence_cot_full"
RUN_QWEN17_E80B="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot_full"
RUN_LLAMA3_E80B="outputs/stage2_adaptive_runs_user/richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e80b_no_argument_pruning_glm51_full1500_thinking_evidence_cot_full"
RUN_QWEN17_E81="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full"
RUN_LLAMA3_E81="outputs/stage2_adaptive_runs_user/richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full"
OUT_BASE="outputs/stage2_strategy_cot_e65/e57_cross_model_20260608"
LOG_BASE="${OUT_BASE}/logs"

docker_common() {
  docker run -d --user root --ipc host --shm-size 16g \
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
    -w /workspace/project "$@"
}

train_qwen4_e70() {
  local gpu="${1:-1}"
  local name="stage2_e70_qwen4_candidate_audit_v2_train_20260609"
  local log_path="${LOG_BASE}/qwen4_e70_candidate_audit_v2_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E70}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E70}' '${OUT_BASE}' || true
  "
}

train_qwen4_e72() {
  local gpu="${1:-1}"
  local name="stage2_e72_qwen4_e57_backbone_subtype_minarg_train_20260611"
  local log_path="${LOG_BASE}/qwen4_e72_e57_backbone_subtype_minarg_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E72}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E72}' '${OUT_BASE}' || true
  "
}

train_qwen4_e73() {
  local gpu="${1:-1}"
  local name="stage2_e73_qwen4_e57_recall_first_exactness_last_train_20260612"
  local log_path="${LOG_BASE}/qwen4_e73_e57_recall_first_exactness_last_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E73}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E73}' '${OUT_BASE}' || true
  "
}

train_qwen4_e73_repeat1() {
  local gpu="${1:-1}"
  local name="stage2_e73_repeat1_qwen4_e57_recall_first_exactness_last_train_20260613"
  local log_path="${LOG_BASE}/qwen4_e73_repeat1_e57_recall_first_exactness_last_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E73_REPEAT1}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E73_REPEAT1}' '${OUT_BASE}' || true
  "
}

train_qwen4_e73_repeat2() {
  local gpu="${1:-1}"
  local name="stage2_e73_repeat2_qwen4_e57_recall_first_exactness_last_train_20260614"
  local log_path="${LOG_BASE}/qwen4_e73_repeat2_e57_recall_first_exactness_last_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E73_REPEAT2}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E73_REPEAT2}' '${OUT_BASE}' || true
  "
}

train_qwen4_e75() {
  local gpu="${1:-1}"
  local name="stage2_e75_qwen4_e57_recall_first_exactness_last_2k_scaling_train_20260614"
  local log_path="${LOG_BASE}/qwen4_e75_e57_recall_first_exactness_last_2k_scaling_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E75}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E75}' '${OUT_BASE}' || true
  "
}

train_qwen4_e76() {
  local gpu="${1:-1}"
  local name="stage2_e76_qwen4_contrastive_exactness_train_20260614"
  local log_path="${LOG_BASE}/qwen4_e76_contrastive_exactness_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E76}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E76}' '${OUT_BASE}' || true
  "
}

train_qwen4_e76_repeat1_2ep() {
  local gpu="${1:-1}"
  local name="stage2_e76_repeat1_2ep_qwen4_contrastive_exactness_train_20260615"
  local log_path="${LOG_BASE}/qwen4_e76_repeat1_2ep_contrastive_exactness_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E76_REPEAT1_2EP}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E76_REPEAT1_2EP}' '${OUT_BASE}' || true
  "
}

train_qwen4_e76_repeat2_2ep() {
  local gpu="${1:-1}"
  local name="stage2_e76_repeat2_2ep_qwen4_contrastive_exactness_train_20260615"
  local log_path="${LOG_BASE}/qwen4_e76_repeat2_2ep_contrastive_exactness_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E76_REPEAT2_2EP}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E76_REPEAT2_2EP}' '${OUT_BASE}' || true
  "
}

train_qwen4_e77() {
  local gpu="${1:-1}"
  local name="stage2_e77_qwen4_e76_direct_control_train_20260615"
  local log_path="${LOG_BASE}/qwen4_e77_e76_direct_control_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E77}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E77}' '${OUT_BASE}' || true
  "
}

train_qwen4_e80a() {
  local gpu="${1:-1}"
  local name="stage2_e80a_qwen4_no_type_arbitration_train_20260615"
  local log_path="${LOG_BASE}/qwen4_e80a_no_type_arbitration_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E80A}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E80A}' '${OUT_BASE}' || true
  "
}

train_qwen4_e80b() {
  local gpu="${1:-1}"
  local name="stage2_e80b_qwen4_no_argument_pruning_train_20260615"
  local log_path="${LOG_BASE}/qwen4_e80b_no_argument_pruning_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E80B}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E80B}' '${OUT_BASE}' || true
  "
}

train_qwen4_e80b_repeat1() {
  local gpu="${1:-1}"
  local name="stage2_e80b_repeat1_qwen4_no_argument_pruning_train_20260616"
  local log_path="${LOG_BASE}/qwen4_e80b_repeat1_no_argument_pruning_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E80B_REPEAT1}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E80B_REPEAT1}' '${OUT_BASE}' || true
  "
}

train_qwen4_e80b_repeat2() {
  local gpu="${1:-1}"
  local name="stage2_e80b_repeat2_qwen4_no_argument_pruning_train_20260616"
  local log_path="${LOG_BASE}/qwen4_e80b_repeat2_no_argument_pruning_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E80B_REPEAT2}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E80B_REPEAT2}' '${OUT_BASE}' || true
  "
}

_train_generic() { # _train_generic <name> <config> <run> <log> <gpu>
  local name="$1"; local cfg="$2"; local run="$3"; local log="$4"; local gpu="${5:-0}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail; mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${cfg}' 2>&1 | tee '${LOG_BASE}/${log}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project); chown -R \${HOST_UGID} '${run}' '${OUT_BASE}' || true
  "
}

train_qwen4_e81_repeat3() {
  local gpu="${1:-2}"; local name="stage2_e81_repeat3_qwen4_trigger_locked_arbitration_train_20260617"
  local log_path="${LOG_BASE}/qwen4_e81_repeat3_trigger_locked_arbitration_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail; mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E81_REPEAT3}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project); chown -R \${HOST_UGID} '${RUN_QWEN4_E81_REPEAT3}' '${OUT_BASE}' || true
  "
}

train_qwen4_e81_repeat1() {
  local gpu="${1:-2}"
  local name="stage2_e81_repeat1_qwen4_trigger_locked_arbitration_train_20260616"
  local log_path="${LOG_BASE}/qwen4_e81_repeat1_trigger_locked_arbitration_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E81_REPEAT1}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E81_REPEAT1}' '${OUT_BASE}' || true
  "
}

train_qwen4_e81_repeat2() {
  local gpu="${1:-3}"
  local name="stage2_e81_repeat2_qwen4_trigger_locked_arbitration_train_20260616"
  local log_path="${LOG_BASE}/qwen4_e81_repeat2_trigger_locked_arbitration_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E81_REPEAT2}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E81_REPEAT2}' '${OUT_BASE}' || true
  "
}

train_qwen4_e83_richere() {
  local gpu="${1:-2}"; local name="stage2_e83_richere_qwen4_trigger_locked_schema_driven_train_20260617"
  local log_path="${LOG_BASE}/qwen4_e83_richere_trigger_locked_schema_driven_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail; mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E83_RICHERE}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project); chown -R \${HOST_UGID} '${RUN_QWEN4_E83_RICHERE}' '${OUT_BASE}' || true
  "
}

train_qwen4_e57_repeat1() {
  local gpu="${1:-2}"; local name="stage2_e57_repeat1_qwen4_candidate_audit_train_20260617"
  local log_path="${LOG_BASE}/qwen4_e57_repeat1_candidate_audit_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail; mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E57_REPEAT1}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project); chown -R \${HOST_UGID} '${RUN_QWEN4_E57_REPEAT1}' '${OUT_BASE}' || true
  "
}

train_qwen4_e57_repeat2() {
  local gpu="${1:-3}"; local name="stage2_e57_repeat2_qwen4_candidate_audit_train_20260617"
  local log_path="${LOG_BASE}/qwen4_e57_repeat2_candidate_audit_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail; mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E57_REPEAT2}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project); chown -R \${HOST_UGID} '${RUN_QWEN4_E57_REPEAT2}' '${OUT_BASE}' || true
  "
}

train_qwen4_direct_repeat1() {
  local gpu="${1:-5}"; local name="stage2_direct_repeat1_qwen4_train_20260617"
  local log_path="${LOG_BASE}/qwen4_direct_repeat1_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail; mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_DIRECT_REPEAT1}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project); chown -R \${HOST_UGID} '${RUN_QWEN4_DIRECT_REPEAT1}' '${OUT_BASE}' || true
  "
}

train_qwen4_direct_repeat2() {
  local gpu="${1:-6}"; local name="stage2_direct_repeat2_qwen4_train_20260617"
  local log_path="${LOG_BASE}/qwen4_direct_repeat2_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail; mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_DIRECT_REPEAT2}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project); chown -R \${HOST_UGID} '${RUN_QWEN4_DIRECT_REPEAT2}' '${OUT_BASE}' || true
  "
}

eval_richere_direct() {
  # eval_richere_direct <run_dir> <label> <ckpt> <split> <gpu>
  local run_dir="$1"; local label="$2"; local ckpt="$3"; local split="$4"; local gpu="${5:-4}"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local data="${DATA_BASE_RICHERE_DIRECT}_${split}_pos.jsonl"
  local out="${OUT_BASE}/${label}/checkpoint-${ckpt}/${split}"
  local name="stage2_${label}_ckpt${ckpt}_${split}_eval_20260617"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail; mkdir -p '${out}' '${LOG_BASE}'
    python src/stage2_quality_validation/eval_adapter_generation.py \
      --base_model '${BASE_QWEN4}' --adapter_path '${adapter}' --eval_jsonl '${data}' \
      --output_dir '${out}' --template_family qwen --max_new_tokens 256 --batch_size 8 --temperature 0.0 \
      2>&1 | tee '${LOG_BASE}/${label}_ckpt${ckpt}_${split}.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project); chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_ace05_direct() {
  # eval_ace05_direct <ckpt> <split> <gpu>  (split: dev_seen|test_seen|test_unseen)
  local ckpt="$1"
  local split="$2"
  local gpu="${3:-4}"
  local adapter="${RUN_QWEN4_ACE05_DIRECT}/checkpoint-${ckpt}"
  local data="data/stage2_formal_datasets/ace05_balanced_split1_oracle_mixed_noise_top10_shuffle_${split}_pos.jsonl"
  local out="${OUT_BASE}/qwen4_ace05_direct/checkpoint-${ckpt}/${split}"
  local name="stage2_ace05_direct_ckpt${ckpt}_${split}_eval_20260617"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${out}' '${LOG_BASE}'
    python src/stage2_quality_validation/eval_adapter_generation.py \
      --base_model '${BASE_QWEN4}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${data}' \
      --output_dir '${out}' \
      --template_family qwen \
      --max_new_tokens 256 \
      --batch_size 8 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/ace05_direct_ckpt${ckpt}_${split}.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

train_qwen4_ace05_e83() {
  local gpu="${1:-4}"
  local name="stage2_ace05_e83_qwen4_trigger_locked_schema_driven_train_20260617"
  local log_path="${LOG_BASE}/qwen4_ace05_e83_trigger_locked_schema_driven_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_ACE05_E83}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_ACE05_E83}' '${OUT_BASE}' || true
  "
}

train_qwen4_ace05_e84() {
  local gpu="${1:-4}"
  local name="stage2_ace05_e84_qwen4_no_arbitration_train_20260618"
  local log_path="${LOG_BASE}/qwen4_ace05_e84_no_arbitration_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_ACE05_E84}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_ACE05_E84}' '${OUT_BASE}' || true
  "
}

train_qwen4_e85() {
  local gpu="${1:-4}"
  local name="stage2_e85_qwen4_secondteacher_deepseek_train_20260618"
  local log_path="${LOG_BASE}/qwen4_e85_secondteacher_deepseek_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E85}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E85}' '${OUT_BASE}' || true
  "
}

train_qwen4_ace05_direct() {
  local gpu="${1:-4}"
  local name="stage2_ace05_qwen4_direct_train_20260616"
  local log_path="${LOG_BASE}/qwen4_ace05_direct_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_ACE05_DIRECT}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_ACE05_DIRECT}' '${OUT_BASE}' || true
  "
}

train_qwen4_e81() {
  local gpu="${1:-1}"
  local name="stage2_e81_qwen4_trigger_locked_arbitration_train_20260616"
  local log_path="${LOG_BASE}/qwen4_e81_trigger_locked_arbitration_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN4_E81}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN4_E81}' '${OUT_BASE}' || true
  "
}

train_qwen17_e81() {
  local gpu="${1:-2}"
  local name="stage2_e81_qwen17_trigger_locked_arbitration_train_20260616"
  local log_path="${LOG_BASE}/qwen17_e81_trigger_locked_arbitration_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN17_E81}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN17_E81}' '${OUT_BASE}' || true
  "
}

train_llama3_e81() {
  local gpu="${1:-3}"
  local name="stage2_e81_llama3_trigger_locked_arbitration_train_20260616"
  local log_path="${LOG_BASE}/llama3_e81_trigger_locked_arbitration_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_LLAMA3_E81}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_LLAMA3_E81}' '${OUT_BASE}' || true
  "
}

train_qwen17_e80b() {
  local gpu="${1:-1}"
  local name="stage2_e80b_qwen17_no_argument_pruning_train_20260616"
  local log_path="${LOG_BASE}/qwen17_e80b_no_argument_pruning_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN17_E80B}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN17_E80B}' '${OUT_BASE}' || true
  "
}

train_llama3_e80b() {
  local gpu="${1:-2}"
  local name="stage2_e80b_llama3_no_argument_pruning_train_20260616"
  local log_path="${LOG_BASE}/llama3_e80b_no_argument_pruning_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_LLAMA3_E80B}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_LLAMA3_E80B}' '${OUT_BASE}' || true
  "
}

train_qwen17() {
  local gpu="${1:-1}"
  local name="stage2_e65_qwen17_e57cross_train_20260608"
  local log_path="${LOG_BASE}/qwen17_e57cross_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN17}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN17}' '${OUT_BASE}' || true
  "
}

train_qwen17_e70() {
  local gpu="${1:-1}"
  local name="stage2_e70_qwen17_candidate_audit_v2_train_20260609"
  local log_path="${LOG_BASE}/qwen17_e70_candidate_audit_v2_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN17_E70}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN17_E70}' '${OUT_BASE}' || true
  "
}

train_llama3() {
  local gpu="${1:-2}"
  local name="stage2_e65_llama3_e57cross_train_20260608"
  local log_path="${LOG_BASE}/llama3_e57cross_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_LLAMA3}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_LLAMA3}' '${OUT_BASE}' || true
  "
}

train_llama3_e70() {
  local gpu="${1:-2}"
  local name="stage2_e70_llama3_candidate_audit_v2_train_20260609"
  local log_path="${LOG_BASE}/llama3_e70_candidate_audit_v2_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_LLAMA3_E70}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_LLAMA3_E70}' '${OUT_BASE}' || true
  "
}

train_qwen17_e67() {
  local gpu="${1:-1}"
  local name="stage2_e67_qwen17_e57cross_from_e66_ckpt2451_train_20260609"
  local log_path="${LOG_BASE}/qwen17_e67_e57cross_train.log"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${LOG_BASE}'
    FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG_QWEN17_E67}' 2>&1 | tee '${log_path}'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${RUN_QWEN17_E67}' '${OUT_BASE}' || true
  "
}

eval_pair() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local ckpt="$4"
  local gpu="$5"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e65_${label}_e57cross_ckpt${ckpt}_eval_pair_20260608"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_pair_e70() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local ckpt="$4"
  local gpu="$5"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e70_${label}_candidate_audit_v2_ckpt${ckpt}_eval_pair_20260609"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E70}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E70}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_pair_e72() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local ckpt="$4"
  local gpu="$5"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e72_${label}_e57_backbone_subtype_minarg_ckpt${ckpt}_eval_pair_20260611"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E72}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E72}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_pair_e73() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local ckpt="$4"
  local gpu="$5"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e73_${label}_e57_recall_first_exactness_last_ckpt${ckpt}_eval_pair_20260612"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E73}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E73}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_pair_e75() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local ckpt="$4"
  local gpu="$5"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e75_${label}_e57_recall_first_exactness_last_2k_ckpt${ckpt}_eval_pair_20260614"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E75}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E75}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_pair_e76() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local ckpt="$4"
  local gpu="$5"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e76_${label}_contrastive_exactness_ckpt${ckpt}_eval_pair_20260614"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E76}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E76}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_dev_e76() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local ckpt="$4"
  local gpu="$5"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e76_${label}_contrastive_exactness_ckpt${ckpt}_devpick_20260615"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/dev_seen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E76}_dev_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/dev_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_dev_seen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_pair_e80() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local data_base="$4"
  local ckpt="$5"
  local gpu="$6"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e80_${label}_ckpt${ckpt}_eval_pair_20260615"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${data_base}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${data_base}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_dev_e80() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local data_base="$4"
  local ckpt="$5"
  local gpu="$6"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e80_${label}_ckpt${ckpt}_devpick_20260615"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/dev_seen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_EVIDENCE}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${data_base}_dev_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/dev_seen' \
      --batch_size 4 \
      --max_new_tokens 1024 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_dev_seen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_pair_e77() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local ckpt="$4"
  local gpu="$5"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e77_${label}_ckpt${ckpt}_eval_pair_20260615"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_DIRECT}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E77}_test_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_seen' \
      --batch_size 8 \
      --max_new_tokens 512 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_seen.log'
    python '${EVAL_SCRIPT_DIRECT}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E77}_test_unseen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/test_unseen' \
      --batch_size 8 \
      --max_new_tokens 512 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_test_unseen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

eval_dev_e77() {
  local label="$1"
  local base_model="$2"
  local run_dir="$3"
  local ckpt="$4"
  local gpu="$5"
  local adapter="${run_dir}/checkpoint-${ckpt}"
  local name="stage2_e77_${label}_ckpt${ckpt}_devpick_20260615"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker_common --name "${name}" --gpus "device=${gpu}" "${IMAGE}" bash -lc "
    set -euo pipefail
    mkdir -p '${OUT_BASE}/${label}/checkpoint-${ckpt}/dev_seen' '${LOG_BASE}'
    python '${EVAL_SCRIPT_DIRECT}' \
      --base_model '${base_model}' \
      --adapter_path '${adapter}' \
      --eval_jsonl '${DATA_BASE_E77}_dev_seen_pos.jsonl' \
      --output_dir '${OUT_BASE}/${label}/checkpoint-${ckpt}/dev_seen' \
      --batch_size 8 \
      --max_new_tokens 512 \
      --temperature 0.0 2>&1 | tee '${LOG_BASE}/${label}_checkpoint-${ckpt}_dev_seen.log'
    HOST_UGID=\$(stat -c '%u:%g' /workspace/project)
    chown -R \${HOST_UGID} '${OUT_BASE}' || true
  "
}

case "${1:-}" in
  train-qwen4-e70)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e70 "${2:-1}"
    ;;
  train-qwen4-e72)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e72 "${2:-1}"
    ;;
  train-qwen4-e73)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e73 "${2:-1}"
    ;;
  train-qwen4-e73-repeat1)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e73_repeat1 "${2:-1}"
    ;;
  train-qwen4-e73-repeat2)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e73_repeat2 "${2:-1}"
    ;;
  train-qwen4-e75)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e75 "${2:-1}"
    ;;
  train-qwen4-e76)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e76 "${2:-1}"
    ;;
  train-qwen4-e76-repeat1-2ep)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e76_repeat1_2ep "${2:-1}"
    ;;
  train-qwen4-e76-repeat2-2ep)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e76_repeat2_2ep "${2:-1}"
    ;;
  train-qwen4-e77)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e77 "${2:-1}"
    ;;
  train-qwen4-e80a)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e80a "${2:-1}"
    ;;
  train-qwen4-e80b)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e80b "${2:-1}"
    ;;
  train-qwen4-e80b-repeat1)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e80b_repeat1 "${2:-1}"
    ;;
  train-qwen4-e80b-repeat2)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e80b_repeat2 "${2:-1}"
    ;;
  train-qwen4-e81)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e81 "${2:-1}"
    ;;
  train-qwen4-e81-repeat1)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e81_repeat1 "${2:-2}"
    ;;
  train-qwen4-e81-repeat2)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e81_repeat2 "${2:-3}"
    ;;
  train-qwen4-e81-repeat3)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e81_repeat3 "${2:-2}"
    ;;
  eval-qwen4-e81-repeat3)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e81_repeat3_trigger_locked_arbitration" "${BASE_QWEN4}" "${RUN_QWEN4_E81_REPEAT3}" "${DATA_BASE_E81}" "${2:?checkpoint}" "${3:-2}"
    ;;
  train-qwen4-e81-repeat4)
    mkdir -p "${LOG_BASE}"
    _train_generic stage2_e81_repeat4_qwen4_trigger_locked_arbitration_train_20260617 "${CONFIG_QWEN4_E81_REPEAT4}" "${RUN_QWEN4_E81_REPEAT4}" qwen4_e81_repeat4_trigger_locked_arbitration_train.log "${2:-0}"
    ;;
  eval-qwen4-e81-repeat4)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e81_repeat4_trigger_locked_arbitration" "${BASE_QWEN4}" "${RUN_QWEN4_E81_REPEAT4}" "${DATA_BASE_E81}" "${2:?checkpoint}" "${3:-0}"
    ;;
  train-qwen4-e57-repeat3)
    mkdir -p "${LOG_BASE}"
    _train_generic stage2_e57_repeat3_qwen4_candidate_audit_train_20260617 "${CONFIG_QWEN4_E57_REPEAT3}" "${RUN_QWEN4_E57_REPEAT3}" qwen4_e57_repeat3_candidate_audit_train.log "${2:-1}"
    ;;
  eval-qwen4-e57-repeat3)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e57_repeat3_candidate_audit" "${BASE_QWEN4}" "${RUN_QWEN4_E57_REPEAT3}" "${DATA_BASE_E56}" "${2:?checkpoint}" "${3:-1}"
    ;;
  train-qwen4-e57-repeat4)
    mkdir -p "${LOG_BASE}"
    _train_generic stage2_e57_repeat4_qwen4_candidate_audit_train_20260617 "${CONFIG_QWEN4_E57_REPEAT4}" "${RUN_QWEN4_E57_REPEAT4}" qwen4_e57_repeat4_candidate_audit_train.log "${2:-3}"
    ;;
  eval-qwen4-e57-repeat4)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e57_repeat4_candidate_audit" "${BASE_QWEN4}" "${RUN_QWEN4_E57_REPEAT4}" "${DATA_BASE_E56}" "${2:?checkpoint}" "${3:-3}"
    ;;
  train-qwen4-ace05-direct)
    mkdir -p "${LOG_BASE}"
    train_qwen4_ace05_direct "${2:-4}"
    ;;
  train-qwen4-ace05-e83)
    mkdir -p "${LOG_BASE}"
    train_qwen4_ace05_e83 "${2:-4}"
    ;;
  eval-qwen4-ace05-e83)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_ace05_e83_trigger_locked_schema_driven" "${BASE_QWEN4}" "${RUN_QWEN4_ACE05_E83}" "${DATA_BASE_ACE05_E83}" "${2:?checkpoint}" "${3:-4}"
    ;;
  train-qwen4-ace05-e84)
    mkdir -p "${LOG_BASE}"
    train_qwen4_ace05_e84 "${2:-4}"
    ;;
  eval-qwen4-ace05-e84)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_ace05_e84_no_arbitration" "${BASE_QWEN4}" "${RUN_QWEN4_ACE05_E84}" "${DATA_BASE_ACE05_E84}" "${2:?checkpoint}" "${3:-4}"
    ;;
  train-qwen4-e85)
    mkdir -p "${LOG_BASE}"
    train_qwen4_e85 "${2:-4}"
    ;;
  eval-qwen4-e85)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e85_secondteacher_deepseek" "${BASE_QWEN4}" "${RUN_QWEN4_E85}" "${DATA_BASE_E85}" "${2:?checkpoint}" "${3:-4}"
    ;;
  eval-qwen4-ace05-direct)
    mkdir -p "${LOG_BASE}"
    eval_ace05_direct "${2:?checkpoint}" "${3:?split}" "${4:-4}"
    ;;
  train-qwen4-e83-richere)
    mkdir -p "${LOG_BASE}"; train_qwen4_e83_richere "${2:-2}" ;;
  eval-qwen4-e83-richere)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e83_richere_trigger_locked_schema_driven" "${BASE_QWEN4}" "${RUN_QWEN4_E83_RICHERE}" "${DATA_BASE_E83_RICHERE}" "${2:?checkpoint}" "${3:-2}" ;;
  train-qwen4-e57-repeat1)
    mkdir -p "${LOG_BASE}"; train_qwen4_e57_repeat1 "${2:-2}" ;;
  train-qwen4-e57-repeat2)
    mkdir -p "${LOG_BASE}"; train_qwen4_e57_repeat2 "${2:-3}" ;;
  eval-qwen4-e57-repeat1)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e57_repeat1_candidate_audit" "${BASE_QWEN4}" "${RUN_QWEN4_E57_REPEAT1}" "${DATA_BASE_E56}" "${2:?checkpoint}" "${3:-2}" ;;
  eval-qwen4-e57-repeat2)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e57_repeat2_candidate_audit" "${BASE_QWEN4}" "${RUN_QWEN4_E57_REPEAT2}" "${DATA_BASE_E56}" "${2:?checkpoint}" "${3:-3}" ;;
  train-qwen4-direct-repeat1)
    mkdir -p "${LOG_BASE}"; train_qwen4_direct_repeat1 "${2:-5}" ;;
  train-qwen4-direct-repeat2)
    mkdir -p "${LOG_BASE}"; train_qwen4_direct_repeat2 "${2:-6}" ;;
  eval-qwen4-direct-repeat1)
    mkdir -p "${LOG_BASE}"
    eval_richere_direct "${RUN_QWEN4_DIRECT_REPEAT1}" "qwen4_direct_repeat1" "${2:?checkpoint}" "${3:?split}" "${4:-5}" ;;
  eval-qwen4-direct-repeat2)
    mkdir -p "${LOG_BASE}"
    eval_richere_direct "${RUN_QWEN4_DIRECT_REPEAT2}" "qwen4_direct_repeat2" "${2:?checkpoint}" "${3:?split}" "${4:-6}" ;;
  train-qwen17-e80b)
    mkdir -p "${LOG_BASE}"
    train_qwen17_e80b "${2:-1}"
    ;;
  train-llama3-e80b)
    mkdir -p "${LOG_BASE}"
    train_llama3_e80b "${2:-2}"
    ;;
  train-qwen17-e81)
    mkdir -p "${LOG_BASE}"
    train_qwen17_e81 "${2:-2}"
    ;;
  train-llama3-e81)
    mkdir -p "${LOG_BASE}"
    train_llama3_e81 "${2:-3}"
    ;;
  train-qwen17)
    mkdir -p "${LOG_BASE}"
    train_qwen17 "${2:-1}"
    ;;
  train-llama3)
    mkdir -p "${LOG_BASE}"
    train_llama3 "${2:-2}"
    ;;
  train-qwen17-e67)
    mkdir -p "${LOG_BASE}"
    train_qwen17_e67 "${2:-1}"
    ;;
  train-qwen17-e70)
    mkdir -p "${LOG_BASE}"
    train_qwen17_e70 "${2:-1}"
    ;;
  eval-qwen17)
    mkdir -p "${LOG_BASE}"
    eval_pair "qwen17_e57cross" "${BASE_QWEN17}" "${RUN_QWEN17}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen17-e67)
    mkdir -p "${LOG_BASE}"
    eval_pair "qwen17_e67_e57cross" "${BASE_QWEN17}" "${RUN_QWEN17_E67}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e70)
    mkdir -p "${LOG_BASE}"
    eval_pair_e70 "qwen4_e70_candidate_audit_v2" "${BASE_QWEN4}" "${RUN_QWEN4_E70}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e72)
    mkdir -p "${LOG_BASE}"
    eval_pair_e72 "qwen4_e72_e57_backbone_subtype_minarg" "${BASE_QWEN4}" "${RUN_QWEN4_E72}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e73)
    mkdir -p "${LOG_BASE}"
    eval_pair_e73 "qwen4_e73_e57_recall_first_exactness_last" "${BASE_QWEN4}" "${RUN_QWEN4_E73}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e73-repeat1)
    mkdir -p "${LOG_BASE}"
    eval_pair_e73 "qwen4_e73_repeat1_e57_recall_first_exactness_last" "${BASE_QWEN4}" "${RUN_QWEN4_E73_REPEAT1}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e73-repeat2)
    mkdir -p "${LOG_BASE}"
    eval_pair_e73 "qwen4_e73_repeat2_e57_recall_first_exactness_last" "${BASE_QWEN4}" "${RUN_QWEN4_E73_REPEAT2}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e75)
    mkdir -p "${LOG_BASE}"
    eval_pair_e75 "qwen4_e75_e57_recall_first_exactness_last_2k_scaling" "${BASE_QWEN4}" "${RUN_QWEN4_E75}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e76)
    mkdir -p "${LOG_BASE}"
    eval_pair_e76 "qwen4_e76_contrastive_exactness" "${BASE_QWEN4}" "${RUN_QWEN4_E76}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e76-repeat1-2ep)
    mkdir -p "${LOG_BASE}"
    eval_pair_e76 "qwen4_e76_repeat1_2ep_contrastive_exactness" "${BASE_QWEN4}" "${RUN_QWEN4_E76_REPEAT1_2EP}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e76-repeat2-2ep)
    mkdir -p "${LOG_BASE}"
    eval_pair_e76 "qwen4_e76_repeat2_2ep_contrastive_exactness" "${BASE_QWEN4}" "${RUN_QWEN4_E76_REPEAT2_2EP}" "${2:?checkpoint}" "${3:-1}"
    ;;
  dev-qwen4-e76)
    mkdir -p "${LOG_BASE}"
    eval_dev_e76 "qwen4_e76_contrastive_exactness_devpick" "${BASE_QWEN4}" "${RUN_QWEN4_E76}" "${2:?checkpoint}" "${3:-1}"
    ;;
  dev-qwen4-e76-repeat1-2ep)
    mkdir -p "${LOG_BASE}"
    eval_dev_e76 "qwen4_e76_repeat1_2ep_contrastive_exactness_devpick" "${BASE_QWEN4}" "${RUN_QWEN4_E76_REPEAT1_2EP}" "${2:?checkpoint}" "${3:-1}"
    ;;
  dev-qwen4-e76-repeat2-2ep)
    mkdir -p "${LOG_BASE}"
    eval_dev_e76 "qwen4_e76_repeat2_2ep_contrastive_exactness_devpick" "${BASE_QWEN4}" "${RUN_QWEN4_E76_REPEAT2_2EP}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e77)
    mkdir -p "${LOG_BASE}"
    eval_pair_e77 "qwen4_e77_e76_direct_control" "${BASE_QWEN4}" "${RUN_QWEN4_E77}" "${2:?checkpoint}" "${3:-1}"
    ;;
  dev-qwen4-e77)
    mkdir -p "${LOG_BASE}"
    eval_dev_e77 "qwen4_e77_e76_direct_control_devpick" "${BASE_QWEN4}" "${RUN_QWEN4_E77}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e80a)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e80a_no_type_arbitration" "${BASE_QWEN4}" "${RUN_QWEN4_E80A}" "${DATA_BASE_E80A}" "${2:?checkpoint}" "${3:-1}"
    ;;
  dev-qwen4-e80a)
    mkdir -p "${LOG_BASE}"
    eval_dev_e80 "qwen4_e80a_no_type_arbitration_devpick" "${BASE_QWEN4}" "${RUN_QWEN4_E80A}" "${DATA_BASE_E80A}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e80b)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e80b_no_argument_pruning" "${BASE_QWEN4}" "${RUN_QWEN4_E80B}" "${DATA_BASE_E80B}" "${2:?checkpoint}" "${3:-1}"
    ;;
  dev-qwen4-e80b)
    mkdir -p "${LOG_BASE}"
    eval_dev_e80 "qwen4_e80b_no_argument_pruning_devpick" "${BASE_QWEN4}" "${RUN_QWEN4_E80B}" "${DATA_BASE_E80B}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e80b-repeat1)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e80b_repeat1_no_argument_pruning" "${BASE_QWEN4}" "${RUN_QWEN4_E80B_REPEAT1}" "${DATA_BASE_E80B}" "${2:?checkpoint}" "${3:-1}"
    ;;
  dev-qwen4-e80b-repeat1)
    mkdir -p "${LOG_BASE}"
    eval_dev_e80 "qwen4_e80b_repeat1_no_argument_pruning_devpick" "${BASE_QWEN4}" "${RUN_QWEN4_E80B_REPEAT1}" "${DATA_BASE_E80B}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e80b-repeat2)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e80b_repeat2_no_argument_pruning" "${BASE_QWEN4}" "${RUN_QWEN4_E80B_REPEAT2}" "${DATA_BASE_E80B}" "${2:?checkpoint}" "${3:-1}"
    ;;
  dev-qwen4-e80b-repeat2)
    mkdir -p "${LOG_BASE}"
    eval_dev_e80 "qwen4_e80b_repeat2_no_argument_pruning_devpick" "${BASE_QWEN4}" "${RUN_QWEN4_E80B_REPEAT2}" "${DATA_BASE_E80B}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen17-e80b)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen17_e80b_no_argument_pruning" "${BASE_QWEN17}" "${RUN_QWEN17_E80B}" "${DATA_BASE_E80B}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen17-e81)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen17_e81_trigger_locked_arbitration" "${BASE_QWEN17}" "${RUN_QWEN17_E81}" "${DATA_BASE_E81}" "${2:?checkpoint}" "${3:-2}"
    ;;
  eval-llama3-e81)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "llama3_e81_trigger_locked_arbitration" "${BASE_LLAMA3}" "${RUN_LLAMA3_E81}" "${DATA_BASE_E81}" "${2:?checkpoint}" "${3:-3}"
    ;;
  dev-qwen4-e81)
    mkdir -p "${LOG_BASE}"
    eval_dev_e80 "qwen4_e81_trigger_locked_arbitration_devpick" "${BASE_QWEN4}" "${RUN_QWEN4_E81}" "${DATA_BASE_E81}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e81)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e81_trigger_locked_arbitration" "${BASE_QWEN4}" "${RUN_QWEN4_E81}" "${DATA_BASE_E81}" "${2:?checkpoint}" "${3:-1}"
    ;;
  eval-qwen4-e81-variant)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e81_eval_on_${2:?variant}" "${BASE_QWEN4}" "${RUN_QWEN4_E81}" "data/stage2_adaptive_datasets/richere_e81eval_on_${2}" "${3:?checkpoint}" "${4:-1}"
    ;;
  eval-qwen4-e81-repeat1)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e81_repeat1_trigger_locked_arbitration" "${BASE_QWEN4}" "${RUN_QWEN4_E81_REPEAT1}" "${DATA_BASE_E81}" "${2:?checkpoint}" "${3:-2}"
    ;;
  eval-qwen4-e81-repeat2)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "qwen4_e81_repeat2_trigger_locked_arbitration" "${BASE_QWEN4}" "${RUN_QWEN4_E81_REPEAT2}" "${DATA_BASE_E81}" "${2:?checkpoint}" "${3:-3}"
    ;;
  eval-llama3-e80b)
    mkdir -p "${LOG_BASE}"
    eval_pair_e80 "llama3_e80b_no_argument_pruning" "${BASE_LLAMA3}" "${RUN_LLAMA3_E80B}" "${DATA_BASE_E80B}" "${2:?checkpoint}" "${3:-2}"
    ;;
  eval-qwen17-e70)
    mkdir -p "${LOG_BASE}"
    eval_pair_e70 "qwen17_e70_candidate_audit_v2" "${BASE_QWEN17}" "${RUN_QWEN17_E70}" "${2:?checkpoint}" "${3:-1}"
    ;;
  train-llama3-e70)
    mkdir -p "${LOG_BASE}"
    train_llama3_e70 "${2:-2}"
    ;;
  eval-llama3-e70)
    mkdir -p "${LOG_BASE}"
    eval_pair_e70 "llama3_e70_candidate_audit_v2" "${BASE_LLAMA3}" "${RUN_LLAMA3_E70}" "${2:?checkpoint}" "${3:-2}"
    ;;
  eval-llama3)
    mkdir -p "${LOG_BASE}"
    eval_pair "llama3_e57cross" "${BASE_LLAMA3}" "${RUN_LLAMA3}" "${2:?checkpoint}" "${3:-2}"
    ;;
  *)
    echo "usage: $0 train-qwen17 [gpu] | train-qwen17-e67 [gpu] | train-llama3 [gpu] | eval-qwen17 <ckpt> [gpu] | eval-qwen17-e67 <ckpt> [gpu] | eval-llama3 <ckpt> [gpu] | train-qwen4-e70 [gpu] | train-qwen4-e72 [gpu] | train-qwen4-e73 [gpu] | train-qwen4-e73-repeat1 [gpu] | train-qwen4-e73-repeat2 [gpu] | train-qwen4-e75 [gpu] | train-qwen4-e76 [gpu] | train-qwen4-e76-repeat1-2ep [gpu] | train-qwen4-e76-repeat2-2ep [gpu] | train-qwen17-e70 [gpu] | train-llama3-e70 [gpu] | eval-qwen4-e70 <ckpt> [gpu] | eval-qwen4-e72 <ckpt> [gpu] | eval-qwen4-e73 <ckpt> [gpu] | eval-qwen4-e73-repeat1 <ckpt> [gpu] | eval-qwen4-e73-repeat2 <ckpt> [gpu] | eval-qwen4-e75 <ckpt> [gpu] | eval-qwen4-e76 <ckpt> [gpu] | eval-qwen4-e76-repeat1-2ep <ckpt> [gpu] | eval-qwen4-e76-repeat2-2ep <ckpt> [gpu] | eval-qwen17-e70 <ckpt> [gpu] | eval-llama3-e70 <ckpt> [gpu]" >&2
    exit 2
    ;;
esac
