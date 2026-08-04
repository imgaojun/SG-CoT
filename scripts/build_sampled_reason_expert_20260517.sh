#!/usr/bin/env bash
set -euo pipefail

ADAPT_PREFIX="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_DIR="data/stage2_adaptive_datasets"
LABEL_DIR="${DATA_DIR}/labels"
LABEL_SOURCE="outcome_l15bal30_15"
BRANCH="sampled_reason_expert_forcedreason_from_noaux_20260517"

TRAIN_LABEL="${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_train_labels.jsonl"
DEV_LABEL="${LABEL_DIR}/${ADAPT_PREFIX}_${LABEL_SOURCE}_dev_seen_labels.jsonl"

if [[ ! -s "${TRAIN_LABEL}" || ! -s "${DEV_LABEL}" ]]; then
  echo "missing labels for ${LABEL_SOURCE}" >&2
  exit 1
fi

python3 scripts/build_sampled_reason_expert_datasets_20260517.py

python3 - \
  "${DATA_DIR}/${ADAPT_PREFIX}_${BRANCH}_forced_reason_train_pos.meta.json" \
  "${DATA_DIR}/${ADAPT_PREFIX}_${BRANCH}_forced_reason_dev_seen_pos.meta.json" \
  "${DATA_DIR}/${ADAPT_PREFIX}_${BRANCH}_forced_direct_train_pos.meta.json" \
  "${DATA_DIR}/${ADAPT_PREFIX}_${BRANCH}_forced_direct_dev_seen_pos.meta.json" <<'PY'
import json
import sys
from pathlib import Path

payload = []
for raw in sys.argv[1:]:
    path = Path(raw)
    meta = json.loads(path.read_text(encoding="utf-8"))
    audit = meta["audit"]
    if audit.get("full_rows_without_final") != 0:
        raise SystemExit(json.dumps({"meta": path.as_posix(), "audit": audit}, indent=2))
    payload.append({"meta": path.as_posix(), "route_mode": meta.get("route_mode"), "audit": audit})
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

python3 scripts/prepare_sampled_reason_expert_20260517.py
