#!/usr/bin/env bash
set -euo pipefail

python src/stage2_analysis/analyze_adaptive_checkpoint_frontier.py \
  --devpick_root outputs/stage2_adaptive_runs_user_devpick_frontier \
  --existing_free_root outputs/stage2_adaptive_runs_user_devpick \
  --protocol_selection_dir outputs/stage2_adaptive_runs_user_devpick_frontier/protocol_selections_crossmodel_20260512 \
  --formal_output_root outputs/stage2_adaptive_runs_user_formal_clean \
  --formal_manifest configs/generated/stage2_adaptive/richere_adaptive_hardconf_crossmodel_checkpoint_frontier_formal_manifest.json \
  --selected_formal_manifest configs/generated/stage2_adaptive/richere_adaptive_hardconf_crossmodel_checkpoint_frontier_formal_selected_manifest.json \
  --selected_protocols seen_stable_best hard_reason_best balanced_hardroute_best \
  --output_md reports/2026-05-12_stage2_adaptive_hardconf_crossmodel_checkpoint_frontier_analysis.md \
  --output_json reports/artifacts/2026-05-12_stage2_adaptive_hardconf_crossmodel_checkpoint_frontier_analysis.json \
  --base_model /workspace/models/LLM-Research/Qwen3-4B \
  --branch_names \
    qwen3_4b_hardconf10_heur10_type_role_hint_plan_lite \
    qwen3_4b_hardconf10_directdup

python src/stage2_analysis/analyze_adaptive_checkpoint_frontier.py \
  --devpick_root outputs/stage2_adaptive_runs_user_devpick_frontier \
  --existing_free_root outputs/stage2_adaptive_runs_user_devpick \
  --protocol_selection_dir outputs/stage2_adaptive_runs_user_devpick_frontier/protocol_selections_crossmodel_llama_20260512 \
  --formal_output_root outputs/stage2_adaptive_runs_user_formal_clean \
  --formal_manifest configs/generated/stage2_adaptive/richere_llama3_2_3b_adaptive_hardconf_checkpoint_frontier_formal_manifest.json \
  --selected_formal_manifest configs/generated/stage2_adaptive/richere_llama3_2_3b_adaptive_hardconf_checkpoint_frontier_formal_selected_manifest.json \
  --selected_protocols seen_stable_best hard_reason_best balanced_hardroute_best \
  --output_md reports/2026-05-12_stage2_adaptive_hardconf_llama3_2_3b_checkpoint_frontier_analysis.md \
  --output_json reports/artifacts/2026-05-12_stage2_adaptive_hardconf_llama3_2_3b_checkpoint_frontier_analysis.json \
  --base_model /workspace/models/LLM-Research/Llama-3.2-3B-Instruct \
  --branch_names \
    llama3_2_3b_hardconf10_heur10_type_role_hint_plan_lite \
    llama3_2_3b_hardconf10_directdup
