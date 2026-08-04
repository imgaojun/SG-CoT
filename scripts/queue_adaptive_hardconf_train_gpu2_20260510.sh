#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
LAUNCH_SCRIPT="scripts/launch_adaptive_hardconf_train_20260510.sh"
BRANCHES=(
  "hardconf10_heur10_type_role_hint_plan_lite"
  "hardconf15_heur15_type_role_hint_plan_lite"
  "hardconf10_calibrated_type_role_hint_plan_lite"
  "hardconf10_directdup"
)

container_name() {
  local branch="$1"
  echo "richere_qwen3_adaptive_${branch}_train_20260510"
}

wait_for_container() {
  local name="$1"
  local status
  status="$(docker inspect -f '{{.State.Status}}' "${name}")"
  if [[ "${status}" == "running" ]]; then
    echo "waiting for running container: ${name}"
    docker wait "${name}" >/dev/null
  fi
  local exit_code
  exit_code="$(docker inspect -f '{{.State.ExitCode}}' "${name}")"
  if [[ "${exit_code}" != "0" ]]; then
    echo "container failed: ${name} exit=${exit_code}" >&2
    exit "${exit_code}"
  fi
  echo "container completed: ${name}"
}

for branch in "${BRANCHES[@]}"; do
  name="$(container_name "${branch}")"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${name}"; then
    wait_for_container "${name}"
    continue
  fi
  echo "launching ${branch} on GPU${GPU_ID}"
  bash "${LAUNCH_SCRIPT}" "${branch}=${GPU_ID}"
  wait_for_container "${name}"
done

echo "adaptive hardconf GPU${GPU_ID} training queue completed"
