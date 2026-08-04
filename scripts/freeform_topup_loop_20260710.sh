#!/usr/bin/env bash
cd /mnt/disk/gaojun/research/progressive-ee
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"
D=outputs/stage2_strategy_cot_e88_freeform/richere_e88_freeform_nl_glm51_full1500_20260620
POOL=outputs/strengthen_20260709/freeform_pool
LOG=outputs/strengthen_20260709/logs/freeform_loop_state.log
for pass in $(seq 1 12); do
  n=$(python3 -c "
import json,glob
ids=set()
for f in glob.glob('$POOL/*.jsonl'):
    for l in open(f): ids.add(json.loads(l).get('sample_id'))
print(len(ids))")
  echo "[$(date +%m-%d\ %H:%M)] pass $pass 开始, 池 unique=$n" >> $LOG
  [ "$n" -ge 1400 ] && { echo "[$(date +%H:%M)] 达标退出" >> $LOG; break; }
  python3 scripts/generate_strategy_variants_cot_e47_20260606.py \
    --prompt_profile freeform_nl --retry_rejected \
    --output_dir $D --limit 1500 \
    >> outputs/strengthen_20260709/logs/freeform_topup4.log 2>&1 || echo "[$(date +%H:%M)] pass $pass 出错" >> $LOG
  [ -s "$D/accepted_evidence_cot.jsonl" ] && cp $D/accepted_evidence_cot.jsonl $POOL/pass_$(date +%m%d_%H%M).jsonl
done
echo "[$(date +%m-%d\ %H:%M)] 循环结束" >> $LOG
