#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
GPU="${2:-}"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
MODEL_ROOT="/mnt/disk/gaojun/models"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
IMAGE="llamafactory-lab:0.9.4-py3.12"
CONFIG="configs/generated/stage2_revision/e132_compact_v3_seed42.yaml"
OUTPUT="outputs/stage2_revision_e132/runs/e132_compact_v3_seed42"
DATA_DIR="data/stage2_revision_e132/frozen_e95_compact_v3"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
SOURCE_MODEL="outputs/stage2_full_sft_runs_stepmatch_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full"
BASELINE_MODEL="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e95_autocluster_glm51_full1500_thinking_evidence_cot_full"
BASELINE_DEV_DATA="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e95_autocluster_glm51_full1500_thinking_evidence_cot_dev_seen_pos.jsonl"
BASELINE_DEV_OUTPUT="outputs/stage2_revision_e132/eval/baseline_e95/dev_seen"
CANDIDATE_DEV_OUTPUT="outputs/stage2_revision_e132/eval/compact_v3_seed42/dev_seen"
DEV_GATE="outputs/stage2_revision_e132/gates/dev_seen.json"
UNSEEN_DATA="data/stage2_revision_e132/after_dev_gate/e132_enriched_e95_test_unseen82.jsonl"
UNSEEN_OUTPUT="outputs/stage2_revision_e132/eval/compact_v3_seed42/test_unseen"
UNSEEN_GATE="outputs/stage2_revision_e132/gates/test_unseen.json"
cd "${PROJECT_ROOT}"

release_owned_label_service() {
  local gpu="$1" state_dir="/mnt/disk/gaojun/tmp/gpu-label-service"
  local metadata="${state_dir}/gpu${gpu}.json" pid_file="${state_dir}/gpu${gpu}.pid"
  [[ -f "${metadata}" && -f "${pid_file}" ]] || return 0
  local owner pid recorded_pid cmdline
  owner="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("owner", ""))' "${metadata}")"
  pid="$(tr -d '[:space:]' < "${pid_file}")"
  recorded_pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid", ""))' "${metadata}")"
  [[ "${owner}" == "gaojun" && "${pid}" == "${recorded_pid}" && "${pid}" =~ ^[0-9]+$ ]] || return 0
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "${cmdline}" == *gpu-label-service* && "${cmdline}" == *"gpu${gpu}"* ]]; then
    docker rm -f "gpu-label-service-gaojun-gpu${gpu}" >/dev/null 2>&1 || kill "${pid}"
  fi
}

assert_gpu_idle() {
  [[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "one numeric GPU index is required" >&2; exit 2; }
  release_owned_label_service "${GPU}"
  local used util processes
  IFS=',' read -r used util < <(
    nvidia-smi -i "${GPU}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' '
  )
  processes="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "${processes}" || "${used}" -gt 1024 || "${util}" -gt 5 ]]; then
    echo "GPU ${GPU} is not idle: memory=${used} MiB utilization=${util}% processes=${processes:-none}" >&2
    exit 12
  fi
}

assert_ready() {
  local available_kib
  available_kib="$(df -Pk /mnt/disk | awk 'NR==2 {print $4}')"
  (( available_kib >= 500 * 1024 * 1024 )) || { echo "less than 500 GiB free" >&2; exit 10; }
  python3 - <<'PY'
import json
import hashlib
from pathlib import Path

root = Path("data/stage2_revision_e132/frozen_e95_compact_v3")
audit = json.loads((root / "frozen_e95_enrichment_audit.json").read_text())
tokens = json.loads((root / "token_budget_audit.json").read_text())
assert audit["passed"] and audit["render_mode"] == "compact_v3"
assert audit["train_rows"] == 1320 and audit["dev_seen_rows"] == 197
assert audit["instruction_exact"] and audit["output_exact"] and audit["gold_output_exact"]
assert audit["test_rows_read"] == 0
assert tokens["passed"] and tokens["candidate"]["over_cutoff"] <= 20
assert tokens["candidate"]["over_2048"] == 0
protocol = json.loads(Path("configs/generated/stage2_revision/e132_trigger_cue_enrichment_protocol.json").read_text())
evaluation = protocol["effectiveness_evaluation"]
assert evaluation["dev_seen"]["maximum_macro_regression_per_metric"] == 0.015
assert evaluation["test_unseen_after_dev_pass"]["minimum_trigger_f1"] == 0.285
assert evaluation["test_unseen_after_dev_pass"]["minimum_trigger_delta"] == 0.03
baseline = Path(evaluation["baseline_model"])
assert hashlib.sha256((baseline / "model.safetensors.index.json").read_bytes()).hexdigest() == evaluation["baseline_model_index_sha256"]
assert hashlib.sha256((baseline / "trainer_state.json").read_bytes()).hexdigest() == evaluation["baseline_trainer_state_sha256"]
PY
  python3 scripts/validate_training_artifact_20260712.py --model_dir "${SOURCE_MODEL}" >/dev/null
}

assert_dev_gate_pass() {
  [[ -f "${DEV_GATE}" ]] || { echo "missing frozen dev gate: ${DEV_GATE}" >&2; exit 14; }
  python3 - "${DEV_GATE}" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert gate.get("id") == "e132_dev_seen_effectiveness_gate_v1"
assert gate.get("passed") is True
assert gate.get("test_rows_read") == 0
PY
}

docker_common() {
  docker run --rm --user root --ipc host --shm-size 16g --gpus "device=${GPU}" \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -v "${LF_ROOT}/cache/torch_extensions:/workspace/.cache/torch_extensions" \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    -w /workspace/project "${IMAGE}" "$@"
}

eval_model() {
  local model="$1" data="$2" output="$3"
  [[ ! -e "${output}" ]] || { echo "refusing to reuse ${output}" >&2; exit 13; }
  docker_common python src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
    --base_model "${BASE_MODEL}" \
    --adapter_path "/workspace/project/${model}" \
    --eval_jsonl "/workspace/project/${data}" \
    --output_dir "/workspace/project/${output}" \
    --batch_size 4 --max_new_tokens 1024 --temperature 0
}

case "${ACTION}" in
  preflight)
    assert_ready
    ;;
  train)
    assert_ready
    assert_gpu_idle
    [[ ! -e "${OUTPUT}" ]] || { echo "refusing to reuse ${OUTPUT}" >&2; exit 13; }
    mkdir -p outputs/stage2_revision_e132/logs
    docker_common bash -lc \
      "FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG}' 2>&1 | tee 'outputs/stage2_revision_e132/logs/train_compact_v3_seed42.log'"
    python3 scripts/validate_training_artifact_20260712.py --model_dir "${OUTPUT}"
    ;;
  eval-baseline-dev)
    assert_ready
    assert_gpu_idle
    python3 scripts/validate_training_artifact_20260712.py --model_dir "${BASELINE_MODEL}" >/dev/null
    eval_model "${BASELINE_MODEL}" "${BASELINE_DEV_DATA}" "${BASELINE_DEV_OUTPUT}"
    ;;
  eval-dev)
    assert_ready
    assert_gpu_idle
    python3 scripts/validate_training_artifact_20260712.py --model_dir "${OUTPUT}" >/dev/null
    eval_model "${OUTPUT}" "${DATA_DIR}/e132_enriched_e95_frozen_dev_seen197.jsonl" "${CANDIDATE_DEV_OUTPUT}"
    ;;
  gate-dev)
    assert_ready
    python3 scripts/evaluate_e132_effectiveness_gate_20260713.py \
      --phase dev_seen \
      --protocol configs/generated/stage2_revision/e132_trigger_cue_enrichment_protocol.json \
      --baseline_dir "${BASELINE_DEV_OUTPUT}" \
      --candidate_dir "${CANDIDATE_DEV_OUTPUT}" \
      --output_json "${DEV_GATE}" --require_pass
    ;;
  build-unseen)
    assert_ready
    assert_dev_gate_pass
    python3 -m scripts.build_e132_unseen_after_dev_gate_20260713 \
      --protocol configs/generated/stage2_revision/e132_trigger_cue_enrichment_protocol.json \
      --dev_gate_json "${DEV_GATE}" \
      --lexicon data/stage2_revision_e132/seen_trigger_lexicon.json \
      --unseen_cards outputs/stage2_revision_e132/schema_synthesis/smoke8/accepted_cards.jsonl \
      --output_dir data/stage2_revision_e132/after_dev_gate
    ;;
  eval-unseen)
    assert_ready
    assert_dev_gate_pass
    assert_gpu_idle
    python3 scripts/validate_training_artifact_20260712.py --model_dir "${OUTPUT}" >/dev/null
    python3 - <<'PY'
import json
from pathlib import Path

audit = json.loads(Path("data/stage2_revision_e132/after_dev_gate/build_audit.json").read_text())
assert audit["id"] == "e132_unseen_after_dev_gate_build_v1" and audit["passed"]
assert audit["rows"] == 82 and audit["test_rows_read"] == 82
PY
    eval_model "${OUTPUT}" "${UNSEEN_DATA}" "${UNSEEN_OUTPUT}"
    ;;
  gate-unseen)
    assert_ready
    assert_dev_gate_pass
    python3 scripts/evaluate_e132_effectiveness_gate_20260713.py \
      --phase test_unseen \
      --protocol configs/generated/stage2_revision/e132_trigger_cue_enrichment_protocol.json \
      --baseline_dir outputs/stage2_strategy_cot_e65/e57_cross_model_20260608/qwen4_e95_autocluster/checkpoint-249/test_unseen \
      --candidate_dir "${UNSEEN_OUTPUT}" \
      --output_json "${UNSEEN_GATE}" --require_pass
    ;;
  *)
    echo "usage: $0 preflight|train <gpu>|eval-baseline-dev <gpu>|eval-dev <gpu>|gate-dev|build-unseen|eval-unseen <gpu>|gate-unseen" >&2
    exit 2
    ;;
esac
