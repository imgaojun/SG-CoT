#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$ROOT/src/data_preprocessing/type_holdout/generate_type_holdout.py" \
  --input_root "$ROOT/data/processed/textee" \
  --output_root "$ROOT/data/processed/type_holdout" \
  --protocol_config "$ROOT/configs/seen_unseen_type_holdout_protocols.json"
