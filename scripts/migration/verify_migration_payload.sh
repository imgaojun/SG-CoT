#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/migration/verify_migration_payload.sh [--root PATH]

Run this on the TARGET machine after syncing. It prints local payload sizes and
checks whether key data/output directories exist.

Options:
  --root PATH     target project path; default: current git root, else /mnt/disk/gaojun/research/progressive-ee
  -h, --help      show this help
USAGE
}

if git_root=$(git rev-parse --show-toplevel 2>/dev/null); then
  ROOT="${ROOT:-$git_root}"
else
  ROOT="${ROOT:-/mnt/disk/gaojun/research/progressive-ee}"
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      ROOT="${2:?missing value for --root}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"

echo "Project root: $ROOT"

echo
echo "==> Git status"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status -sb
  git log -1 --oneline
else
  echo "Not a Git working tree."
fi

echo
echo "==> Data directory sizes"
for path in \
  data/raw \
  data/processed \
  data/stage2_cot_datasets \
  data/stage2_cot_datasets_smoke \
  data/stage2_dualmode_datasets \
  data/stage2_formal_datasets; do
  if [ -e "$path" ]; then
    du -sh "$path"
  else
    echo "MISSING $path"
  fi
done

echo
echo "==> Output directory sizes"
if [ -e outputs ]; then
  du -h --max-depth=1 outputs 2>/dev/null | sort -hr | sed -n '1,60p'
else
  echo "MISSING outputs"
fi

echo
echo "==> Output checkpoint count"
if [ -e outputs ]; then
  find outputs -type d -name 'checkpoint-*' 2>/dev/null | wc -l | awk '{print $1 " checkpoint directories"}'
else
  echo "0 checkpoint directories"
fi

echo
echo "==> Large non-checkpoint files over 1G"
find data outputs -path '*/checkpoint-*' -prune -o -type f -size +1G -printf '%s %p\n' 2>/dev/null | sort -nr | sed -n '1,80p' || true

echo
echo "Verification summary complete."
