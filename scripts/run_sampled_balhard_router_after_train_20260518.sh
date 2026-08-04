#!/usr/bin/env bash
set -euo pipefail

BRANCH="sampled_k8_ckpt258_balhard_routecls_noauxwarm_lr2e6_save25"
LOG="outputs/stage2_adaptive_runs_user_logs/sampled_balhard_router_after_train_20260518.log"

wait_for_container() {
  local name="$1"
  local label="$2"
  if ! docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    echo "missing container for ${label}: ${name}" >&2
    exit 1
  fi
  while docker ps --format '{{.Names}}' | grep -Fxq "${name}"; do
    date "+[%F %T] waiting for ${label}: ${name}" | tee -a "${LOG}"
    sleep 60
  done
  local status
  status="$(docker inspect -f '{{.State.ExitCode}}' "${name}")"
  date "+[%F %T] ${label} exited with ${status}" | tee -a "${LOG}"
  if [[ "${status}" != "0" ]]; then
    docker logs --tail 120 "${name}" | tee -a "${LOG}" >&2
    exit 1
  fi
}

mkdir -p "$(dirname "${LOG}")"
date '+[%F %T] sampled balhard router after-train pipeline start' | tee -a "${LOG}"

wait_for_container "sampled_confident_router_${BRANCH}_train_20260518" "sampled balhard router training"

bash scripts/launch_adaptive_outcome_route_probe_devpick_20260513.sh "${BRANCH}=0" | tee -a "${LOG}"
wait_for_container "adaptive_outcome_${BRANCH}_route_devpick_20260513" "sampled balhard router route-devpick"

bash scripts/launch_sampled_confident_router_route_nll_dev_20260518.sh "${BRANCH}=0" | tee -a "${LOG}"
wait_for_container "sampled_confident_router_route_nll_${BRANCH}_20260518" "sampled balhard router route-NLL"

python3 scripts/summarize_sampled_confident_router_dev_20260518.py \
  --branch "${BRANCH}" \
  --report_stem "2026-05-18_stage2_sampled_k8_balhard_routecls_checkpoint258_dev_probe" | tee -a "${LOG}"

date '+[%F %T] sampled balhard router dev pipeline complete' | tee -a "${LOG}"
