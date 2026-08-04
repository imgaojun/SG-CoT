#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
GPU="${2:-}"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
MODEL_ROOT="/mnt/disk/gaojun/models"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
IMAGE="llamafactory-lab:0.9.4-py3.12"
PROTOCOL="configs/generated/stage2_revision/e135_sc_distillation_smoke64_protocol.json"
MANIFEST="data/stage2_revision_e135/smoke64/selected_rows.jsonl"
GENERATION_OUTPUT="outputs/stage2_revision_e135/generation/smoke64"
TARGET_OUTPUT="data/stage2_revision_e135/distillation_smoke64"
MODEL="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full"
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
  python3 scripts/validate_training_artifact_20260712.py \
    --model_dir "${MODEL}" --trainer_state "${MODEL}/trainer_state.json" \
    --min_global_step 273 --require_finite_step_log >/dev/null
  python3 - <<'PY'
import hashlib
import json
from pathlib import Path

protocol = json.loads(Path("configs/generated/stage2_revision/e135_sc_distillation_smoke64_protocol.json").read_text())
audit = json.loads(Path("data/stage2_revision_e135/smoke64/selection_audit.json").read_text())
assert protocol["test_access"] is False
assert audit["passed"] and audit["selected_rows"] == 64 and audit["test_rows_read"] == 0
assert audit["output_sha256"] == hashlib.sha256(Path("data/stage2_revision_e135/smoke64/selected_rows.jsonl").read_bytes()).hexdigest()
PY
}

docker_generate() {
  docker run --rm --user root --ipc host --shm-size 16g --gpus "device=${GPU}" \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/workspace/.cache/huggingface \
    -w /workspace/project "${IMAGE}" \
    python scripts/generate_e135_sc_paths_20260713.py \
      --protocol "${PROTOCOL}" \
      --manifest_jsonl "${MANIFEST}" \
      --output_dir "${GENERATION_OUTPUT}" "$@"
}

case "${ACTION}" in
  preflight)
    assert_ready
    ;;
  generate)
    assert_ready
    assert_gpu_idle
    [[ ! -e "${GENERATION_OUTPUT}" ]] || { echo "refusing to reuse ${GENERATION_OUTPUT}" >&2; exit 13; }
    docker_generate
    ;;
  resume)
    assert_ready
    assert_gpu_idle
    [[ -d "${GENERATION_OUTPUT}" ]] || { echo "missing generation output" >&2; exit 13; }
    docker_generate --resume
    ;;
  audit)
    assert_ready
    [[ -f "${GENERATION_OUTPUT}/summary.json" ]] || { echo "missing generation summary" >&2; exit 13; }
    python3 scripts/build_e135_sc_distillation_targets_20260713.py \
      --protocol "${PROTOCOL}" \
      --manifest_jsonl "${MANIFEST}" \
      --generations_jsonl "${GENERATION_OUTPUT}/raw_generations.jsonl" \
      --output_dir "${TARGET_OUTPUT}" --require_pass
    ;;
  *)
    echo "usage: $0 preflight|generate <gpu>|resume <gpu>|audit" >&2
    exit 2
    ;;
esac
