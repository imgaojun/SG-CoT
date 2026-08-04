#!/usr/bin/env bash
set -euo pipefail

REASON_CKPT="${REASON_CKPT:-checkpoint-258}"
SEEDS="${SEEDS:-17 18}"
GPU_DIRECT_SEEN="${GPU_DIRECT_SEEN:-0}"
GPU_REASON_SEEN="${GPU_REASON_SEEN:-1}"
GPU_DIRECT_UNSEEN="${GPU_DIRECT_UNSEEN:-2}"
GPU_REASON_UNSEEN="${GPU_REASON_UNSEEN:-3}"
GPU_NLL="${GPU_NLL:-0}"
RUN_ID="sampled_reason_expert_forcedreason_from_noaux_20260517_${REASON_CKPT}"
SAMPLE_ROOT="outputs/stage2_modular_dualexpert/formal_k2_counterfactual_utility_20260518/${RUN_ID}"

wait_for_container() {
  local name="$1"
  while docker ps --format '{{.Names}}' | grep -Fxq "${name}"; do
    sleep 60
  done
  if docker ps -a --format '{{.Names}} {{.Status}}' | grep -F "${name} " | grep -Fvq "Exited (0)"; then
    docker ps -a --filter "name=^/${name}$"
    exit 1
  fi
}

complete_samples() {
  local split="$1"
  local route="$2"
  local expected_count="$3"
  for seed in ${SEEDS}; do
    local dir="${SAMPLE_ROOT}/${split}/${route}/seed-${seed}"
    [[ -s "${dir}/predictions.jsonl" && -s "${dir}/summary.json" ]] || return 1
    local count
    count="$(python3 -c "import sys; print(sum(1 for _ in open(sys.argv[1], encoding='utf-8')))" "${dir}/predictions.jsonl")"
    [[ "${count}" == "${expected_count}" ]] || return 1
  done
}

python3 scripts/prepare_sampled_k2_formal_forced_datasets_20260518.py

bash scripts/launch_sampled_k2_formal_counterfactual_20260518.sh "${REASON_CKPT}" direct test_seen "${GPU_DIRECT_SEEN}" ${SEEDS}
bash scripts/launch_sampled_k2_formal_counterfactual_20260518.sh "${REASON_CKPT}" reason test_seen "${GPU_REASON_SEEN}" ${SEEDS}
bash scripts/launch_sampled_k2_formal_counterfactual_20260518.sh "${REASON_CKPT}" direct test_unseen "${GPU_DIRECT_UNSEEN}" ${SEEDS}
bash scripts/launch_sampled_k2_formal_counterfactual_20260518.sh "${REASON_CKPT}" reason test_unseen "${GPU_REASON_UNSEEN}" ${SEEDS}

wait_for_container "sampled_k2_formal_test_seen_direct_${REASON_CKPT}_20260518"
wait_for_container "sampled_k2_formal_test_seen_reason_${REASON_CKPT}_20260518"
wait_for_container "sampled_k2_formal_test_unseen_direct_${REASON_CKPT}_20260518"
wait_for_container "sampled_k2_formal_test_unseen_reason_${REASON_CKPT}_20260518"

complete_samples test_seen direct 361
complete_samples test_seen reason 361
complete_samples test_unseen direct 82
complete_samples test_unseen reason 82

python3 scripts/prepare_sampled_k2_formal_route_nll_probe_20260518.py
bash scripts/launch_sampled_k2_formal_route_nll_20260518.sh "${GPU_NLL}"
wait_for_container "sampled_k2_formal_route_nll_probe_20260518_score"
python3 scripts/summarize_sampled_k2_formal_route_nll_20260518.py
