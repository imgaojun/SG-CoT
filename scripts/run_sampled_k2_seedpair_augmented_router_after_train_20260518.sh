#!/usr/bin/env bash
set -euo pipefail

BRANCH="sampled_k2pairaug_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
RUN_PREFIX="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
TRANSFER_DATASET_ID="sampled_k2pairaug_transfer_20260518"
OUTPUT_ROOT="outputs/stage2_adaptive_route_seedpair_transfer_augmented_20260518/${BRANCH}"
LOG="outputs/stage2_adaptive_runs_user_logs/sampled_k2_seedpair_augmented_router_after_train_20260518.log"
REPORT_STEM="2026-05-18_stage2_sampled_k2_seedpair_augmented_compact_evidence_router_dev_transfer"

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
    docker logs --tail 160 "${name}" | tee -a "${LOG}" >&2
    exit 1
  fi
}

mkdir -p "$(dirname "${LOG}")"
date '+[%F %T] sampled K2 seed-pair augmented router after-train pipeline start' | tee -a "${LOG}"

wait_for_container "sampled_confident_router_${BRANCH}_train_20260518" "sampled K2 seed-pair augmented router training"

RUN_DIR="outputs/stage2_adaptive_runs_user/${RUN_PREFIX}_${BRANCH}_full"
CKPTS="$(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tr '\n' ' ')"
if [[ -z "${CKPTS}" ]]; then
  echo "no checkpoints found under ${RUN_DIR}" | tee -a "${LOG}" >&2
  exit 1
fi

date "+[%F %T] launching seed-pair transfer for checkpoints: ${CKPTS}" | tee -a "${LOG}"
BRANCH="${BRANCH}" \
TRANSFER_ID="sampled_k2_seedpair_augmented_router_transfer_20260518" \
TRANSFER_DATASET_ID="${TRANSFER_DATASET_ID}" \
CHECKPOINTS="${CKPTS}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
LOG="outputs/stage2_adaptive_runs_user_logs/sampled_k2_seedpair_augmented_router_transfer_20260518.log" \
bash scripts/launch_sampled_k2_seedpair_transfer_router_20260518.sh 0 | tee -a "${LOG}"

wait_for_container "sampled_k2_seedpair_augmented_router_transfer_20260518_gen" "sampled K2 seed-pair augmented router transfer"

python3 scripts/summarize_sampled_k2_seedpair_transfer_router_20260518.py \
  --branch "${BRANCH}" \
  --output_root "${OUTPUT_ROOT}" \
  --checkpoints ${CKPTS} \
  --report_stem "${REPORT_STEM}" | tee -a "${LOG}"

date '+[%F %T] sampled K2 seed-pair augmented router after-train pipeline complete' | tee -a "${LOG}"
