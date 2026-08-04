#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-data/bin/python}"

ACE_IN="$ROOT/data/raw/ace05/ACE 2005 Multilingual Training Corpus/data/English"
ERE_IN="$ROOT/data/raw/ere/DEFT_English_Light_and_Rich_ERE_Annotation/data"
ACE_OUT="$ROOT/data/processed/textee/ace05-en"
ERE_OUT="$ROOT/data/processed/textee/richere-en"

mkdir -p "$ACE_OUT" "$ERE_OUT"

for SPLIT in split1 split2 split3 split4 split5; do
  "$PYTHON_BIN" "$ROOT/src/data_preprocessing/textee/ace05/process_ace05_en.py" \
    -i "$ACE_IN" \
    -o "$ACE_OUT" \
    --lang english \
    --split_path "$ROOT/src/data_preprocessing/textee/ace05/split-en" \
    --split "$SPLIT" \
    --sent_map "$ROOT/src/data_preprocessing/textee/ace05/sent_map.json" \
    --token_map "$ROOT/src/data_preprocessing/textee/ace05/token_map.json"

  "$PYTHON_BIN" "$ROOT/src/data_preprocessing/textee/ere/process_ere_en.py" \
    -i "$ERE_IN" \
    -o "$ERE_OUT" \
    --lang english \
    --split_path "$ROOT/src/data_preprocessing/textee/ere/split-en" \
    --split "$SPLIT" \
    --sent_map "$ROOT/src/data_preprocessing/textee/ere/sent_map.json" \
    --token_map "$ROOT/src/data_preprocessing/textee/ere/token_map.json"
done
