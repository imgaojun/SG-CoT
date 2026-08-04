#!/usr/bin/env bash
set -euo pipefail

MANIFEST="configs/generated/stage2_adaptive/richere_qwen3_1_7b_adaptive_outcome_calibrated_execution_gate_formal_manifest.json"
FORMAL_CONTAINER="richere_qwen3_adaptive_outcome_calibrated_formal_selected_20260514"

BRANCHES=(
  outcome15cal_nlltop10_type_role_hint_plan_lite_routeaux2x_reasonos2
  outcome15cal_nlltop15_type_role_hint_plan_lite_routeaux2x_reasonos2
)

wait_for_container() {
  local name="$1"
  local label="${2:-container}"
  while docker ps --format '{{.Names}}' | grep -Fxq "${name}"; do
    date '+[%F %T] waiting for '"${label}"': '"${name}"
    sleep 60
  done
  if docker inspect "${name}" >/dev/null 2>&1; then
    local status
    status="$(docker inspect -f '{{.State.ExitCode}}' "${name}")"
    if [[ "${status}" != "0" ]]; then
      echo "${label} failed: ${name} exit=${status}" >&2
      exit 1
    fi
  else
    echo "${label} container not found after wait: ${name}" >&2
    exit 1
  fi
}

manifest_run_count() {
  python3 - "${MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
print(len(payload.get("runs", [])))
PY
}

for branch in "${BRANCHES[@]}"; do
  wait_for_container "richere_qwen3_adaptive_outcome_calibrated_${branch}_train_20260514" "training container"
done

bash scripts/launch_adaptive_outcome_calibrated_devpick_20260514.sh \
  outcome15cal_nlltop10_type_role_hint_plan_lite_routeaux2x_reasonos2=0,2 \
  outcome15cal_nlltop15_type_role_hint_plan_lite_routeaux2x_reasonos2=1,3

for branch in "${BRANCHES[@]}"; do
  wait_for_container "adaptive_outcome_calibrated_${branch}_devpick_20260514" "devpick container"
  python3 src/stage2_analysis/select_adaptive_execution_gate.py --branch "${branch}"
done

python3 scripts/build_adaptive_outcome_calibrated_formal_manifest_20260514.py
runs="$(manifest_run_count)"
if [[ "${runs}" == "0" ]]; then
  python3 scripts/summarize_adaptive_outcome_calibrated_formal_20260514.py
  echo "No branch passed execution gate; skipping formal eval."
  exit 0
fi

bash scripts/launch_adaptive_outcome_calibrated_formal_selected_20260514.sh
wait_for_container "${FORMAL_CONTAINER}" "formal container"
python3 scripts/summarize_adaptive_outcome_calibrated_formal_20260514.py
