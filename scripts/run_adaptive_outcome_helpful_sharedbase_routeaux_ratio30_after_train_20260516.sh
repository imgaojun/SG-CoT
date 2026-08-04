#!/usr/bin/env bash
set -euo pipefail

BRANCH="outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_routeauxr30_reasonos2_from_noaux"
LOG="outputs/stage2_adaptive_runs_user_logs/adaptive_outcome_helpful_sharedbase_routeaux_ratio30_after_train_20260516.log"

wait_for_container() {
  local name="$1"
  local label="${2:-container}"
  while docker ps --format '{{.Names}}' | grep -Fxq "${name}"; do
    date '+[%F %T] waiting for '"${label}"': '"${name}" | tee -a "${LOG}"
    sleep 60
  done
  if docker inspect "${name}" >/dev/null 2>&1; then
    local status
    status="$(docker inspect -f '{{.State.ExitCode}}' "${name}")"
    if [[ "${status}" != "0" ]]; then
      echo "${label} failed: ${name} exit=${status}" | tee -a "${LOG}" >&2
      exit 1
    fi
  else
    echo "${label} container not found after wait: ${name}" | tee -a "${LOG}" >&2
    exit 1
  fi
}

mkdir -p "$(dirname "${LOG}")"

wait_for_container "richere_qwen3_adaptive_outcome_helpful_sharedbase_${BRANCH}_train_20260515" "training container"

bash scripts/launch_adaptive_outcome_helpful_sharedbase_devpick_20260515.sh \
  "${BRANCH}=0,2" | tee -a "${LOG}"

wait_for_container "adaptive_outcome_helpful_sharedbase_${BRANCH}_devpick_20260515" "devpick container"

bash scripts/launch_adaptive_outcome_helpful_sharedbase_route_nll_dev_20260515.sh \
  "${BRANCH}=0" | tee -a "${LOG}"

wait_for_container "adaptive_outcome_helpful_sharedbase_route_nll_${BRANCH}_20260515" "route-nll container"

python3 src/stage2_analysis/select_adaptive_sharedbase_nll_execution_gate.py --branch "${BRANCH}" | tee -a "${LOG}"

date '+[%F %T] route-aux ratio30 dev pipeline complete' | tee -a "${LOG}"
