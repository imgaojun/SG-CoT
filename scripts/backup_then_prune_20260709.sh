#!/usr/bin/env bash
# 磁盘优化:中间 checkpoint 先备份到 manhadun 再删本地。
# 每个目录:rsync -> 校验(文件数+总字节) -> rm 本地(root属主的收集起来最后用容器删)。
set -u
R=/mnt/disk/gaojun/research/progressive-ee
B=/mnt/manhattan/manhadun_backup/gaojun/archive/progressive-ee/checkpoint_prune_20260709
LIST=/mnt/disk/gaojun/research/progressive-ee/scripts/strengthen_jobs_20260709/prune_list_20260709.txt
LOG=$R/outputs/strengthen_20260709/logs/backup_prune.log
FAILED_RM=$R/outputs/strengthen_20260709/logs/prune_need_docker_rm.txt
: > "$FAILED_RM"

total=$(wc -l < "$LIST"); i=0; freed_kb=0
while IFS= read -r d; do
  i=$((i+1))
  [ -d "$d" ] || { echo "[$i/$total] 已不存在 $d" >> "$LOG"; continue; }
  rel=${d#$R/outputs/}
  tgt="$B/$rel"
  mkdir -p "$tgt"
  if ! rsync -r --size-only "$d/" "$tgt/" >> "$LOG" 2>&1; then
    echo "[$i/$total] RSYNC失败 $rel" >> "$LOG"; continue
  fi
  # 清掉上次失败残留的 rsync 临时文件(checkpoint 目录本无隐藏文件)
  find "$tgt" -name ".*" -type f -delete 2>/dev/null
  # 校验:文件数 + 总字节
  sc=$(find "$d" -type f | wc -l);  tc=$(find "$tgt" -type f | wc -l)
  sb=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
  tb=$(find "$tgt" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
  if [ "$sc" != "$tc" ] || [ "$sb" != "$tb" ]; then
    echo "[$i/$total] 校验不一致 $rel (files $sc/$tc bytes $sb/$tb),不删本地" >> "$LOG"; continue
  fi
  kb=$(du -sk "$d" | cut -f1)
  if rm -rf "$d" 2>/dev/null; then
    freed_kb=$((freed_kb+kb))
    echo "[$i/$total] OK $rel ($(  (echo "scale=1; $kb/1048576" | bc) )G) 累计释放 $((freed_kb/1048576))G" >> "$LOG"
  else
    echo "$d" >> "$FAILED_RM"
    echo "[$i/$total] 备份OK但本地删除需容器 $rel" >> "$LOG"
  fi
done < "$LIST"

# root 属主的用容器统一删
if [ -s "$FAILED_RM" ]; then
  echo "=== 容器删除 root 属主目录 ($(wc -l < $FAILED_RM) 个) ===" >> "$LOG"
  docker run --rm -v "$R/outputs:/o" -v "$FAILED_RM:/list.txt:ro" alpine sh -c '
    while IFS= read -r p; do rm -rf "/o/${p#*/outputs/}"; done < /list.txt' >> "$LOG" 2>&1 \
  || docker run --rm -v "$R/outputs:/o" -v "$FAILED_RM:/list.txt:ro" llamafactory-lab:0.9.4-py3.12 bash -c '
    while IFS= read -r p; do rm -rf "/o/${p#*/outputs/}"; done < /list.txt' >> "$LOG" 2>&1
fi

echo "=== 完成 $(date) ===" >> "$LOG"
df -h /mnt/disk | tail -1 >> "$LOG"
