#!/usr/bin/env bash
# 加固战役 runner:每卡一个实例,按文件名顺序领取 pending 作业执行。
# 作业可选首行 #DEP=<path>:该路径存在才可领取。
# usage: strengthen_runner_20260709.sh <GPU>
set -u
GPU=$1
BASE=/mnt/disk/gaojun/research/progressive-ee/scripts/strengthen_jobs_20260709
mkdir -p "$BASE/running" "$BASE/done" "$BASE/failed"

while true; do
  claimed=""
  for job in $(ls "$BASE/pending"/*.job 2>/dev/null | sort); do
    dep=$(sed -n 's/^#DEP=//p' "$job" 2>/dev/null | head -1)
    if [ -n "$dep" ] && [ ! -e "$dep" ]; then continue; fi
    name=$(basename "$job")
    if mv "$job" "$BASE/running/gpu${GPU}_${name}" 2>/dev/null; then
      claimed="$BASE/running/gpu${GPU}_${name}"
      break
    fi
  done
  if [ -z "$claimed" ]; then
    n=$(ls "$BASE/pending"/*.job 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && break
    sleep 120   # 只剩未满足依赖的作业,等待
    continue
  fi
  export GPU
  echo "[gpu$GPU] $(date +%H:%M) 开始 $(basename "$claimed")"
  if bash "$claimed"; then
    mv "$claimed" "$BASE/done/"
  else
    mv "$claimed" "$BASE/failed/"
    echo "[gpu$GPU] $(basename "$claimed") 失败"
  fi
done
echo "[gpu$GPU] 队列空,runner 退出"
