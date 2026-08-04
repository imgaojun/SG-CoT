#!/usr/bin/env bash
set -euo pipefail

TRAIN_CONTAINERS=(
  "richere_qwen3_adaptive_hardconf10_heur10_type_role_hint_plan_lite_train_20260510"
  "richere_qwen3_adaptive_hardconf15_heur15_type_role_hint_plan_lite_train_20260510"
  "richere_qwen3_adaptive_hardconf10_calibrated_type_role_hint_plan_lite_train_20260510"
  "richere_qwen3_adaptive_hardconf10_directdup_train_20260510"
)

wait_for_train() {
  local name="$1"
  local status
  if ! docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "missing train container: ${name}" >&2
    exit 1
  fi
  status="$(docker inspect -f '{{.State.Status}}' "${name}")"
  if [[ "${status}" == "running" ]]; then
    echo "waiting for ${name}"
    docker wait "${name}" >/dev/null
  fi
  local exit_code
  exit_code="$(docker inspect -f '{{.State.ExitCode}}' "${name}")"
  if [[ "${exit_code}" != "0" ]]; then
    echo "train failed: ${name} exit=${exit_code}" >&2
    exit "${exit_code}"
  fi
  echo "train completed: ${name}"
}

for name in "${TRAIN_CONTAINERS[@]}"; do
  wait_for_train "${name}"
done

echo "launching hardconf devpick"
bash scripts/launch_adaptive_hardconf_devpick_20260510.sh

DEVPICK_CONTAINERS=(
  "adaptive_hardconf_hardconf10_heur10_type_role_hint_plan_lite_devpick_20260510"
  "adaptive_hardconf_hardconf15_heur15_type_role_hint_plan_lite_devpick_20260510"
  "adaptive_hardconf_hardconf10_calibrated_type_role_hint_plan_lite_devpick_20260510"
  "adaptive_hardconf_hardconf10_directdup_devpick_20260510"
)

for name in "${DEVPICK_CONTAINERS[@]}"; do
  wait_for_train "${name}"
done

echo "building hardconf checkpoint frontier selections"
python3 src/stage2_analysis/analyze_adaptive_checkpoint_frontier.py \
  --branch_names \
    hardconf10_heur10_type_role_hint_plan_lite \
    hardconf15_heur15_type_role_hint_plan_lite \
    hardconf10_calibrated_type_role_hint_plan_lite \
    hardconf10_directdup \
  --selected_protocols seen_stable_best hard_reason_best balanced_hardroute_best \
  --formal_manifest configs/generated/stage2_adaptive/richere_qwen3_1_7b_adaptive_hardconf_checkpoint_frontier_formal_manifest.json \
  --selected_formal_manifest configs/generated/stage2_adaptive/richere_qwen3_1_7b_adaptive_hardconf_checkpoint_frontier_formal_selected_manifest.json \
  --output_md reports/2026-05-10_stage2_adaptive_hardconf_checkpoint_frontier_analysis.md \
  --output_json reports/artifacts/2026-05-10_stage2_adaptive_hardconf_checkpoint_frontier_analysis.json

echo "launching hardconf selected formal eval"
bash scripts/launch_adaptive_hardconf_formal_selected_20260510.sh
wait_for_train "richere_qwen3_adaptive_hardconf_formal_selected_20260510"

echo "adaptive hardconf after-train pipeline completed"
