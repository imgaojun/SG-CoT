#!/usr/bin/env bash
# E85 second-teacher generation retry loop (deepseek-v4-pro teacher; RichERE).
# Round 1 generates the full pass; later rounds retry only rejected rows (reuse_existing).
set -uo pipefail
cd /mnt/disk/gaojun/research/progressive-ee

GATE="${1:-1400}"
MAX_ROUNDS="${2:-30}"
SLEEP_BETWEEN="${3:-60}"
OUT_DIR="outputs/stage2_strategy_cot_e85/richere_e85_secondteacher_deepseek_full1500_20260618"
SUMMARY="${OUT_DIR}/e47_summary.json"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

acc() { python3 -c "import json;print(json.load(open('${SUMMARY}'))['accepted'])" 2>/dev/null || echo 0; }

for r in $(seq 1 "${MAX_ROUNDS}"); do
  cur=$(acc)
  echo "=== [E85 round ${r}] start $(date '+%m-%d %H:%M')  accepted=${cur}/${GATE} ==="
  if [ "${cur}" -ge "${GATE}" ]; then echo "=== E85 GATE reached (${cur}); stop. ==="; break; fi
  bash scripts/run_e85_secondteacher_generation_20260618.sh 1500 --retry_rejected 2>&1 | tail -3
  new=$(acc)
  echo "=== [E85 round ${r}] end $(date '+%m-%d %H:%M')  accepted=${cur} -> ${new} ==="
  if [ "${new}" -ge "${GATE}" ]; then echo "=== E85 GATE reached (${new}); stop. ==="; break; fi
  sleep "${SLEEP_BETWEEN}"
done
echo "=== E85 retry loop done; final accepted=$(acc) ==="
