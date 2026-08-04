#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

MODEL_KEY="${1:-}"
PRESET_KEY="${2:-}"
SLICE_KEY="${3:-}"

if [ -z "$MODEL_KEY" ] || [ -z "$PRESET_KEY" ] || [ -z "$SLICE_KEY" ]; then
  echo "Usage: sh src/stage2_benchmark/run_llamafactory_benchmark.sh <model_key> <preset_key> <slice_key>"
  exit 1
fi

python "$ROOT/src/stage2_benchmark/build_llamafactory_benchmark.py" \
  --model_key "$MODEL_KEY" \
  --preset_key "$PRESET_KEY" \
  --slice_key "$SLICE_KEY" \
  --print_docker_cmd
