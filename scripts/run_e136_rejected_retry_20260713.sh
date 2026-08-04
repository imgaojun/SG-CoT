#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
PROTOCOL="configs/generated/stage2_revision/e136_rejected_row_bounded_retry_protocol.json"
MANIFEST_DIR="data/stage2_revision_e136/retry116"
MANIFEST="${MANIFEST_DIR}/retry_rows.jsonl"
GEN_OUTPUT="outputs/stage2_revision_e136/generation/retry116"
GEN_DATA="data/stage2_revision_e136/generated_retry"
RETRY_TRAIN="${GEN_DATA}/richere_e136_retrieved_retry_e136_rejected_row_bounded_retry116_thinking_evidence_cot_train_pos.jsonl"
COMBINED_OUTPUT="data/stage2_revision_e136/combined"
BASE_URL="${LLM_BASE_URL}"
cd "${PROJECT_ROOT}"

assert_ready() {
  python3 - <<'PY'
import hashlib
import json
from pathlib import Path

protocol = json.loads(Path("configs/generated/stage2_revision/e136_rejected_row_bounded_retry_protocol.json").read_text())
assert protocol["id"] == "e136_rejected_row_bounded_retry_v1"
assert protocol["retry_manifest_sha256"] != "TO_BE_FROZEN_BEFORE_GENERATION"
manifest = Path(protocol["retry_manifest"])
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == protocol["retry_manifest_sha256"]
audit = json.loads((manifest.parent / "manifest_audit.json").read_text())
assert audit["passed"] and audit["retry_rows"] == 116 and audit["test_rows_read"] == 0
assert audit["retry_manifest_sha256"] == protocol["retry_manifest_sha256"]
parent_gate = json.loads(Path("outputs/stage2_revision_e134/generation/full1500/gate.json").read_text())
assert parent_gate["passed"] is False and parent_gate["test_rows_read"] == 0
PY
}

extract_api_key() {
  local key
  key="$(sed -n 's/^export OPENAI_API_KEY="${OPENAI_API_KEY:-\(.*\)}"$/\1/p' scripts/run_e76_contrastive_generation_20260614.sh | head -n 1)"
  [[ -n "${key}" ]] || { echo "missing reusable virtual key" >&2; exit 11; }
  printf '%s' "${key}"
}

assert_api_auth() {
  local key="$1" status
  status="$(curl -sS --connect-timeout 10 --max-time 30 -o /dev/null \
    -w '%{http_code}' -H "Authorization: Bearer ${key}" "${BASE_URL}/models")"
  [[ "${status}" == "200" ]] || {
    echo "API authentication check failed with HTTP ${status}" >&2
    exit 11
  }
}

case "${ACTION}" in
  preflight)
    assert_ready
    ;;
  auth-check)
    assert_ready
    api_key="$(extract_api_key)"
    assert_api_auth "${api_key}"
    unset api_key
    echo "AUTH_OK"
    ;;
  generate)
    assert_ready
    [[ ! -e "${GEN_OUTPUT}" && ! -e "${GEN_DATA}" ]] || {
      echo "refusing to reuse E136 generation/data output" >&2
      exit 13
    }
    api_key="$(extract_api_key)"
    assert_api_auth "${api_key}"
    LITELLM_API_KEY="${api_key}" python3 scripts/generate_strategy_variants_cot_e47_20260606.py \
      --run_name e136_rejected_row_bounded_retry116 \
      --limit 116 --seed 1360 --workers 24 \
      --model glm-5.1 --verifier_model glm-5.1 \
      --base_url "${BASE_URL}" \
      --prompt_profile e130_retrieved_abstention --output_protocol xml_tags \
      --gen_max_tokens 8192 --verify_max_tokens 4096 --max_attempts 3 \
      --sampled_rows_path "${MANIFEST}" --sampled_rows_mode prefix --train_only \
      --adaptive_prefix richere_e136_retrieved_retry \
      --adaptive_data_dir "${GEN_DATA}" --output_dir "${GEN_OUTPUT}"
    unset api_key
    ;;
  audit)
    assert_ready
    [[ -f "${GEN_OUTPUT}/e40_raw.jsonl" && -f "${RETRY_TRAIN}" ]] || {
      echo "missing completed E136 retry artifacts" >&2
      exit 13
    }
    python3 scripts/audit_e136_rejected_retry_20260713.py \
      --protocol "${PROTOCOL}" \
      --retry_raw_jsonl "${GEN_OUTPUT}/e40_raw.jsonl" \
      --retry_train_jsonl "${RETRY_TRAIN}" \
      --output_dir "${COMBINED_OUTPUT}" --require_pass
    ;;
  *)
    echo "usage: $0 preflight|auth-check|generate|audit" >&2
    exit 2
    ;;
esac
