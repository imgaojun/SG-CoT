#!/usr/bin/env bash
# 等原生 Direct 训练完成 -> 用最终 checkpoint 评 WikiEvents-native test_seen/test_unseen -> 类型分组重打分
set -uo pipefail
cd /mnt/disk/gaojun/research/progressive-ee

RUN="outputs/stage2_full_sft_runs_wikievents/wikievents_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full"

# 1) 等训练容器结束
for i in $(seq 1 120); do
  n=$(docker ps --filter "name=wiki_direct_train" -q 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && break
  sleep 20
done
echo "训练容器已结束: $(docker ps -a --filter name=wiki_direct_train --format '{{.Status}}')"

# 2) 最终 checkpoint(最高编号)
CK=$(ls "$RUN" | grep -oE "checkpoint-[0-9]+" | sort -t- -k2 -n | tail -1)
echo "最终 checkpoint: $CK"

IMAGE="llamafactory-lab:0.9.4-py3.12"
PR="/mnt/disk/gaojun/research/progressive-ee"; MR="/mnt/disk/gaojun/models"; LF="/mnt/disk/gaojun/research/llamafactory-lab"
BASE="/workspace/models/LLM-Research/Qwen3-4B"
ADAPTER="/workspace/project/$RUN/$CK"
DATA="/workspace/project/data/stage2_formal_datasets/wikievents_split1_oracle_mixed_noise_top10_shuffle"
OUT="/workspace/project/outputs/wikievents_native_eval"
mkdir -p "$PR/outputs/wikievents_native_eval/logs"

run(){ local name=$1 gpu=$2 split=$3
docker rm -f "wiki_native_$name" 2>/dev/null
docker run -d --user root --ipc host --shm-size 16g --name "wiki_native_$name" --gpus "\"device=$gpu\"" \
 -v "$PR:/workspace/project" -v "$MR:/workspace/models" -v "$LF/cache/huggingface:/workspace/.cache/huggingface" \
 -e HF_HOME=/workspace/.cache/huggingface -w /workspace/project "$IMAGE" bash -lc "
 python src/stage2_quality_validation/eval_adapter_generation.py --base_model '$BASE' --adapter_path '$ADAPTER' \
  --eval_jsonl '${DATA}_${split}_pos.jsonl' --output_dir '$OUT/$name' --batch_size 4 --max_new_tokens 3072 2>&1 | tee '$OUT/logs/$name.log'
 chown -R \$(stat -c '%u:%g' /workspace/project) '$OUT/$name' '$OUT/logs/$name.log' || true" >/dev/null
echo "launched wiki_native_$name gpu$gpu"; }

run direct_test_seen 2 test_seen
run direct_test_unseen 3 test_unseen

# 3) 等评测完成
for i in $(seq 1 60); do
  n=$(docker ps --filter "name=wiki_native" -q 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && break
  sleep 15
done
echo "评测完成: $(docker ps -a --filter name=wiki_native --format '{{.Names}}={{.Status}}' | tr '\n' ' ')"

# 4) 类型分组重打分
python3 scripts/wikievents_native_rescore_20260709.py
