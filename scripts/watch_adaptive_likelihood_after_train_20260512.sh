#!/usr/bin/env bash
set -euo pipefail
TRAIN_CONTAINER="richere_qwen3_adaptive_likelihood_pairall_type_role_hint_plan_lite_scorer_train_20260512"
LOG="/tmp/adaptive_likelihood_after_train_20260512.log"
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] waiting for ${TRAIN_CONTAINER}"
  while docker ps --format '{{.Names}}' | grep -Fxq "${TRAIN_CONTAINER}"; do
    sleep 60
  done
  status=$(docker inspect -f '{{.State.ExitCode}}' "${TRAIN_CONTAINER}" 2>/dev/null || echo missing)
  echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] ${TRAIN_CONTAINER} exit=${status}"
  if [[ "${status}" != "0" ]]; then
    exit 1
  fi
  bash scripts/run_adaptive_likelihood_after_scorer_20260512.sh 4
} >> "${LOG}" 2>&1
