#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
MODEL_ROOT="/mnt/disk/gaojun/models"
LF_ROOT="/mnt/disk/gaojun/research/llamafactory-lab"
IMAGE="llamafactory-lab:0.9.4-py3.12"
PROTOCOL="configs/generated/stage2_revision/e137_mixed_oracle_retrieved_protocol.json"
TEST_MATRIX="configs/generated/stage2_revision/e137b_readonly_test_matrix.json"
CONFIG="configs/generated/stage2_revision/e137a_mixed_oracle_retrieved_seed42.yaml"
DATA_DIR="data/stage2_revision_e137"
OUTPUT_ROOT="outputs/stage2_revision_e137"
RUN_DIR="${OUTPUT_ROOT}/runs/e137a_mixed_oracle_retrieved_seed42"
BASE_MODEL="/workspace/models/LLM-Research/Qwen3-4B"
BASELINE_MODEL="outputs/stage2_adaptive_runs_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full"
BACKUP_ROOT="/mnt/manhattan/manhadun_backup/gaojun/progressive-ee/checkpoints"
cd "${PROJECT_ROOT}"

docker_common() {
  docker run --rm --user root --ipc host --shm-size 16g \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${MODEL_ROOT}:/workspace/models" \
    -v "${LF_ROOT}/cache/huggingface:/workspace/.cache/huggingface" \
    -v "${LF_ROOT}/cache/torch_extensions:/workspace/.cache/torch_extensions" \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    -w /workspace/project "$@"
}

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
  local gpu="$1" used util processes
  release_owned_label_service "${gpu}"
  IFS=',' read -r used util < <(
    nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' '
  )
  processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "${processes}" || "${used}" -gt 1024 || "${util}" -gt 5 ]]; then
    echo "GPU ${gpu} is not idle: memory=${used} MiB utilization=${util}% processes=${processes:-none}" >&2
    return 12
  fi
}

assert_two_gpus_idle() {
  local spec="$1" gpu
  local -a values
  declare -A seen=()
  IFS=',' read -r -a values <<< "${spec}"
  (( ${#values[@]} == 2 )) || { echo "E137 requires exactly two GPUs" >&2; return 12; }
  for gpu in "${values[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ && -z "${seen[${gpu}]:-}" ]] || return 12
    seen["${gpu}"]=1
    assert_gpu_idle "${gpu}"
  done
}

assert_disk_headroom() {
  local available_kib
  available_kib="$(df -Pk /mnt/disk | awk 'NR==2 {print $4}')"
  (( available_kib >= 500 * 1024 * 1024 )) || {
    echo "E137 requires at least 500 GiB free on /mnt/disk" >&2
    return 10
  }
  [[ -d "${BACKUP_ROOT}" && -w "${BACKUP_ROOT}" ]] || return 11
}

assert_config() {
  python3 - "${PROTOCOL}" "${CONFIG}" <<'PY'
import json, sys, yaml

p = json.load(open(sys.argv[1], encoding="utf-8"))
c = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
expected = {
    "model_name_or_path": "/workspace/project/" + p["training"]["warm_start"],
    "dataset_dir": "/workspace/project/data/stage2_revision_e137",
    "dataset": "e137_mixed_oracle_retrieved_train",
    "eval_dataset": "e137_oracle_dev_seen",
    "output_dir": "/workspace/project/outputs/stage2_revision_e137/runs/e137a_mixed_oracle_retrieved_seed42",
    "cutoff_len": 1536,
    "learning_rate": 2e-6,
    "num_train_epochs": 3.0,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "save_only_model": True,
    "load_best_model_at_end": False,
    "seed": 42,
    "data_seed": 42,
}
for key, value in expected.items():
    if c.get(key) != value:
        raise SystemExit(f"E137 config mismatch for {key}: {c.get(key)!r} != {value!r}")
PY
}

assert_frozen_model_hashes() {
  python3 - "${PROTOCOL}" <<'PY'
import hashlib, json, sys
from pathlib import Path

protocol = json.load(open(sys.argv[1], encoding="utf-8"))
for section, key in (("training", "warm_start_weight_sha256"), ("baseline", "weight_sha256")):
    root = Path(protocol[section]["warm_start" if section == "training" else "model"])
    actual = []
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        digest = hashlib.sha256()
        with (root / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual.append(digest.hexdigest())
    if actual != protocol[section][key]:
        raise SystemExit(f"frozen E137 {section} weights do not match protocol")
PY
}

assert_preflight() {
  python3 scripts/build_e137_mixed_oracle_retrieved_20260713.py \
    --protocol "${PROTOCOL}" --output_dir "${DATA_DIR}" --verify_existing >/dev/null
  python3 - "${DATA_DIR}/token_budget.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["id"] == "e137_mixed_oracle_retrieved_token_gate_v1"
assert value["passed"] is True
assert value["summaries"]["total"]["rows"] == 2917
PY
  assert_config
  assert_frozen_model_hashes
  python3 scripts/validate_training_artifact_20260712.py \
    --model_dir "$(python3 -c 'import json; print(json.load(open("configs/generated/stage2_revision/e137_mixed_oracle_retrieved_protocol.json"))["training"]["warm_start"])')" >/dev/null
  python3 scripts/validate_training_artifact_20260712.py --model_dir "${BASELINE_MODEL}" >/dev/null
}

eval_dev() {
  local model_kind="$1" regime="$2" gpu="$3" model data output
  assert_preflight
  assert_gpu_idle "${gpu}"
  case "${model_kind}" in
    baseline) model="${BASELINE_MODEL}" ;;
    candidate)
      model="${RUN_DIR}"
      python3 scripts/validate_training_artifact_20260712.py --model_dir "${model}" >/dev/null
      ;;
    *) return 2 ;;
  esac
  case "${regime}" in
    oracle) data="e137_oracle_dev_seen.jsonl" ;;
    predicted) data="e137_predicted_dev_seen.jsonl" ;;
    *) return 2 ;;
  esac
  output="${OUTPUT_ROOT}/eval/${model_kind}/dev_${regime}"
  [[ ! -e "${output}" ]] || { echo "refusing to reuse ${output}" >&2; return 13; }
  docker_common --gpus "device=${gpu}" "${IMAGE}" python \
    src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
    --base_model "${BASE_MODEL}" --adapter_path "/workspace/project/${model}" \
    --eval_jsonl "/workspace/project/${DATA_DIR}/${data}" \
    --output_dir "/workspace/project/${output}" \
    --batch_size 4 --max_new_tokens 1024 --temperature 0
}

assert_test_preflight() {
  docker_common "${IMAGE}" python -m scripts.audit_e137_test_matrix_20260713 \
    --matrix "${TEST_MATRIX}" \
    --output_dir "${OUTPUT_ROOT}/analysis/test_preflight" \
    --verify_existing >/dev/null
}

eval_test() {
  local regime="$1" split="$2" gpu="$3" data output
  assert_preflight
  assert_test_preflight
  assert_gpu_idle "${gpu}"
  python3 scripts/validate_training_artifact_20260712.py --model_dir "${RUN_DIR}" >/dev/null
  case "${regime}_${split}" in
    oracle_seen)
      data="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_test_seen_pos.jsonl"
      ;;
    oracle_unseen)
      data="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_test_unseen_pos.jsonl"
      ;;
    predicted_seen)
      data="data/stage2_adaptive_datasets/richere_e81eval_on_predicted_top10_test_seen_pos.jsonl"
      ;;
    predicted_unseen)
      data="data/stage2_adaptive_datasets/richere_e81eval_on_predicted_top10_test_unseen_pos.jsonl"
      ;;
    *) return 2 ;;
  esac
  output="${OUTPUT_ROOT}/eval/candidate/test_${regime}_${split}"
  [[ ! -e "${output}" ]] || { echo "refusing to reuse ${output}" >&2; return 13; }
  docker_common --gpus "device=${gpu}" "${IMAGE}" python \
    src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py \
    --base_model "${BASE_MODEL}" --adapter_path "/workspace/project/${RUN_DIR}" \
    --eval_jsonl "/workspace/project/${data}" \
    --output_dir "/workspace/project/${output}" \
    --batch_size 4 --max_new_tokens 1024 --temperature 0
}

case "${ACTION}" in
  preflight)
    assert_disk_headroom
    assert_preflight
    [[ ! -e "${RUN_DIR}" ]] || { echo "E137 run output already exists" >&2; exit 13; }
    ;;
  train)
    gpus="${2:?comma-separated two GPUs required}"
    assert_disk_headroom
    assert_preflight
    assert_two_gpus_idle "${gpus}"
    [[ ! -e "${RUN_DIR}" ]] || { echo "refusing to reuse ${RUN_DIR}" >&2; exit 13; }
    docker_common --gpus "\"device=${gpus}\"" "${IMAGE}" bash -lc \
      "mkdir -p '${OUTPUT_ROOT}/logs' && FORCE_TORCHRUN=1 llamafactory-cli train '${CONFIG}' 2>&1 | tee '${OUTPUT_ROOT}/logs/train_seed42.log'"
    python3 scripts/validate_training_artifact_20260712.py --model_dir "${RUN_DIR}"
    ;;
  eval-dev)
    eval_dev "${2:?baseline or candidate required}" "${3:?oracle or predicted required}" "${4:?GPU required}"
    ;;
  gate-dev)
    docker_common "${IMAGE}" python -m scripts.analyze_e137_dev_gate_20260713 \
      --protocol "${PROTOCOL}" \
      --baseline_oracle "${OUTPUT_ROOT}/eval/baseline/dev_oracle" \
      --baseline_predicted "${OUTPUT_ROOT}/eval/baseline/dev_predicted" \
      --candidate_oracle "${OUTPUT_ROOT}/eval/candidate/dev_oracle" \
      --candidate_predicted "${OUTPUT_ROOT}/eval/candidate/dev_predicted" \
      --output_dir "${OUTPUT_ROOT}/analysis/dev_gate"
    ;;
  preflight-test)
    assert_preflight
    [[ ! -e "${OUTPUT_ROOT}/analysis/test_preflight" ]] || {
      echo "refusing to reuse ${OUTPUT_ROOT}/analysis/test_preflight" >&2
      exit 13
    }
    for cell in oracle_seen oracle_unseen predicted_seen predicted_unseen; do
      [[ ! -e "${OUTPUT_ROOT}/eval/candidate/test_${cell}" ]] || {
        echo "candidate test output already exists for ${cell}" >&2
        exit 13
      }
    done
    docker_common "${IMAGE}" python -m scripts.audit_e137_test_matrix_20260713 \
      --matrix "${TEST_MATRIX}" \
      --output_dir "${OUTPUT_ROOT}/analysis/test_preflight"
    ;;
  eval-test)
    eval_test "${2:?oracle or predicted required}" "${3:?seen or unseen required}" "${4:?GPU required}"
    ;;
  gate-test)
    assert_test_preflight
    docker_common "${IMAGE}" python -m scripts.analyze_e137_test_gate_20260713 \
      --matrix "${TEST_MATRIX}" \
      --preflight_dir "${OUTPUT_ROOT}/analysis/test_preflight" \
      --output_dir "${OUTPUT_ROOT}/analysis/test_gate"
    ;;
  archive)
    ARCHIVE_DELETE_LOCAL="${ARCHIVE_DELETE_LOCAL:-0}" \
      bash scripts/archive_training_checkpoints_20260713.sh "${RUN_DIR}" "${BACKUP_ROOT}"
    ;;
  *)
    echo "usage: $0 preflight|train <gpu,gpu>|eval-dev <baseline|candidate> <oracle|predicted> <gpu>|gate-dev|preflight-test|eval-test <oracle|predicted> <seen|unseen> <gpu>|gate-test|archive" >&2
    exit 2
    ;;
esac
