#!/usr/bin/env bash
set -euo pipefail

BRANCH="sampled_k2_structproxy_strictv2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
LOG="outputs/stage2_adaptive_runs_user_logs/sampled_k2_structural_proxy_router_v2_after_train_20260519.log"
REPORT_STEM="2026-05-19_stage2_sampled_k2_structural_proxy_strictv2_router_dev_probe"
DESCRIPTION='This dev-only report evaluates a strict-v2 route-only classifier trained from high-precision structural proxy positives and hard failed-reason negatives.'

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
date '+[%F %T] sampled K2 strict-v2 structural-proxy router after-train pipeline start' | tee -a "${LOG}"

wait_for_container "sampled_confident_router_${BRANCH}_train_20260518" "sampled K2 strict-v2 structural-proxy router training"

bash scripts/launch_sampled_k2_structural_proxy_router_v2_devpick_20260519.sh 0 | tee -a "${LOG}"
wait_for_container "sampled_k2_structproxy_v2_router_route_devpick_20260519" "sampled K2 strict-v2 structural-proxy router route-devpick"

bash scripts/launch_sampled_k2_structural_proxy_router_v2_route_nll_20260519.sh 0 | tee -a "${LOG}"
wait_for_container "sampled_k2_structproxy_v2_router_route_nll_20260519" "sampled K2 strict-v2 structural-proxy router route-NLL"

python3 scripts/summarize_sampled_confident_router_dev_20260518.py \
  --branch "${BRANCH}" \
  --report_stem "${REPORT_STEM}" \
  --route_root "outputs/stage2_adaptive_runs_user_devpick_route/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_${BRANCH}_full_route_dev_seen_seedpairs_max16" \
  --description "${DESCRIPTION}" | tee -a "${LOG}"

date '+[%F %T] sampled K2 strict-v2 structural-proxy router dev pipeline complete' | tee -a "${LOG}"
