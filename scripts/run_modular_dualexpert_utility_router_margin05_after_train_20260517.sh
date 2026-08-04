#!/usr/bin/env bash
set -euo pipefail

BRANCH="modular_d1930_r2058_utility_margin05_routecls_noauxwarm_lr2e6_save50"
LOG="outputs/stage2_adaptive_runs_user_logs/modular_dualexpert_utility_router_margin05_after_train_20260517.log"
REPORT_MD="reports/2026-05-17_stage2_modular_dualexpert_utility_router_margin05_dev_probe.md"
REPORT_JSON="reports/artifacts/2026-05-17_stage2_modular_dualexpert_utility_router_margin05_dev_probe.json"

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

wait_for_container "modular_dualexpert_utility_router_${BRANCH}_train_20260517" "margin05 utility-router training"

bash scripts/launch_adaptive_outcome_route_probe_devpick_20260513.sh "${BRANCH}=0" | tee -a "${LOG}"
wait_for_container "adaptive_outcome_${BRANCH}_route_devpick_20260513" "margin05 utility-router route devpick"

SCORE_ROOT="outputs/stage2_modular_dualexpert/utility_router_margin05_20260517/route_likelihood" \
  bash scripts/launch_modular_dualexpert_utility_router_route_nll_dev_20260517.sh "${BRANCH}=0" | tee -a "${LOG}"
wait_for_container "modular_dualexpert_utility_router_route_nll_${BRANCH}_20260517" "margin05 utility-router route NLL"

PYTHONDONTWRITEBYTECODE=1 python3 scripts/summarize_modular_dualexpert_utility_router_dev_20260517.py \
  --branch "${BRANCH}" \
  --score_root "outputs/stage2_modular_dualexpert/utility_router_margin05_20260517/route_likelihood" \
  --output_json "${REPORT_JSON}" \
  --output_md "${REPORT_MD}" \
  --description "This dev-only report evaluates an independent utility router trained on \`reason_gain > 0.5\` labels from D1930 direct and R2058 reason experts." | tee -a "${LOG}"

date '+[%F %T] modular dual-expert margin05 utility-router dev pipeline complete' | tee -a "${LOG}"
