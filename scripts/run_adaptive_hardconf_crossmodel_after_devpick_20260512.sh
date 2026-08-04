#!/usr/bin/env bash
set -euo pipefail

wait_for_container() {
  local name="$1"
  while true; do
    local status
    status="$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "${name}" 2>/dev/null || true)"
    if [[ -z "${status}" ]]; then
      echo "waiting for container to appear: ${name}"
      sleep 60
      continue
    fi
    local state exit_code
    state="${status%% *}"
    exit_code="${status##* }"
    echo "${name}: ${state} exit=${exit_code}"
    if [[ "${state}" == "exited" ]]; then
      if [[ "${exit_code}" != "0" ]]; then
        echo "container failed: ${name}" >&2
        exit 1
      fi
      return 0
    fi
    sleep 60
  done
}

containers=(
  "adaptive_hardconf_crossmodel_qwen3_4b_hardconf10_devpick_20260512"
  "adaptive_hardconf_crossmodel_qwen3_4b_hardconf10_directdup_devpick_20260512"
  "adaptive_hardconf_crossmodel_llama3_2_3b_hardconf10_devpick_20260512"
  "adaptive_hardconf_crossmodel_llama3_2_3b_hardconf10_directdup_devpick_20260512"
)

for name in "${containers[@]}"; do
  wait_for_container "${name}"
done

bash scripts/build_adaptive_hardconf_crossmodel_frontier_20260512.sh
bash scripts/launch_adaptive_hardconf_crossmodel_formal_selected_20260512.sh
