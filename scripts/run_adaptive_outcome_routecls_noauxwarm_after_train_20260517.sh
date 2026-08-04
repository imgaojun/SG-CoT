#!/usr/bin/env bash
set -euo pipefail

BRANCH="outcome15_l15bal30_routecls_noauxwarm_lr2e6_save50_probe"
LOG="outputs/stage2_adaptive_runs_user_logs/adaptive_outcome_routecls_noauxwarm_after_train_20260517.log"

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

wait_for_container "richere_qwen3_adaptive_routecls_noauxwarm_${BRANCH}_train_20260517" "training container"

bash scripts/launch_adaptive_outcome_route_probe_devpick_20260513.sh \
  "${BRANCH}=0" | tee -a "${LOG}"

wait_for_container "adaptive_outcome_${BRANCH}_route_devpick_20260513" "route-devpick container"

bash scripts/launch_adaptive_outcome_routecls_noauxwarm_route_nll_dev_20260517.sh \
  "${BRANCH}=0" | tee -a "${LOG}"

wait_for_container "adaptive_outcome_routecls_noauxwarm_route_nll_${BRANCH}_20260517" "route-nll container"

python3 scripts/summarize_adaptive_outcome_routecls_noauxwarm_20260517.py --branch "${BRANCH}" | tee -a "${LOG}"

date '+[%F %T] routecls no-aux warm-start dev pipeline complete' | tee -a "${LOG}"
