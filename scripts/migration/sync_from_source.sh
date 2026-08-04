#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/migration/sync_from_source.sh --source USER@HOST [--mode MODE] [options]

Run this on the TARGET machine to pull project payloads from the SOURCE machine.

Required:
  --source USER@HOST         SSH source, for example gaojun@10.0.0.8

Common options:
  --mode MODE                data | outputs-light | outputs-no-checkpoints |
                             docker | all-light | all-no-checkpoints
                             default: data
  --src-root PATH            source project path
                             default: /mnt/disk/gaojun/research/progressive-ee
  --dst-root PATH            target project path
                             default: current git root, else /mnt/disk/gaojun/research/progressive-ee
  --docker-src-root PATH     source Docker lab path
                             default: /mnt/disk/gaojun/research/llamafactory-lab
  --docker-dst-root PATH     target Docker lab path
                             default: /mnt/disk/gaojun/research/llamafactory-lab
  --ssh-port PORT            SSH port for rsync transport
  --bwlimit KBPS             rsync bandwidth limit in KB/s
  --dry-run                  print what would transfer, do not write files
  --delete                   mirror deletes from source; off by default
  -h, --help                 show this help

Recommended first run:
  scripts/migration/sync_from_source.sh --source USER@HOST --mode all-light

Modes:
  data
    Pulls non-Git data directories and formal-dataset .jsonl files.

  outputs-light
    Pulls outputs metadata/logs/eval artifacts, excluding checkpoint dirs and
    large model weight files (*.safetensors, *.bin, *.pt, *.pth).

  outputs-no-checkpoints
    Pulls outputs excluding checkpoint-* directories only. This includes root
    final model weights and can still be large.

  docker
    Pulls ../llamafactory-lab into /mnt/disk/gaojun/research/llamafactory-lab.

  all-light
    data + outputs-light.

  all-no-checkpoints
    data + outputs-no-checkpoints.
USAGE
}

SOURCE="${SOURCE:-}"
MODE="data"
SRC_ROOT="${SRC_ROOT:-/mnt/disk/gaojun/research/progressive-ee}"
DOCKER_SRC_ROOT="${DOCKER_SRC_ROOT:-/mnt/disk/gaojun/research/llamafactory-lab}"
DOCKER_DST_ROOT="${DOCKER_DST_ROOT:-/mnt/disk/gaojun/research/llamafactory-lab}"
DRY_RUN=0
DELETE=0
SSH_PORT=""
BWLIMIT=""

if git_root=$(git rev-parse --show-toplevel 2>/dev/null); then
  DST_ROOT="${DST_ROOT:-$git_root}"
else
  DST_ROOT="${DST_ROOT:-/mnt/disk/gaojun/research/progressive-ee}"
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source)
      SOURCE="${2:?missing value for --source}"
      shift 2
      ;;
    --mode)
      MODE="${2:?missing value for --mode}"
      shift 2
      ;;
    --src-root)
      SRC_ROOT="${2:?missing value for --src-root}"
      shift 2
      ;;
    --dst-root)
      DST_ROOT="${2:?missing value for --dst-root}"
      shift 2
      ;;
    --docker-src-root)
      DOCKER_SRC_ROOT="${2:?missing value for --docker-src-root}"
      shift 2
      ;;
    --docker-dst-root)
      DOCKER_DST_ROOT="${2:?missing value for --docker-dst-root}"
      shift 2
      ;;
    --ssh-port)
      SSH_PORT="${2:?missing value for --ssh-port}"
      shift 2
      ;;
    --bwlimit)
      BWLIMIT="${2:?missing value for --bwlimit}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --delete)
      DELETE=1
      shift
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

if [ -z "$SOURCE" ]; then
  echo "ERROR: --source USER@HOST is required." >&2
  usage >&2
  exit 2
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync is required on the target machine." >&2
  exit 1
fi

case "$MODE" in
  data|outputs-light|outputs-no-checkpoints|docker|all-light|all-no-checkpoints)
    ;;
  *)
    echo "ERROR: unsupported --mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

RSYNC_BASE=(-avh --info=progress2 --partial)
if [ "$DRY_RUN" -eq 1 ]; then
  RSYNC_BASE+=(--dry-run)
fi
if [ "$DELETE" -eq 1 ]; then
  RSYNC_BASE+=(--delete)
fi
if [ -n "$BWLIMIT" ]; then
  RSYNC_BASE+=(--bwlimit "$BWLIMIT")
fi
if [ -n "$SSH_PORT" ]; then
  RSYNC_BASE+=(-e "ssh -p $SSH_PORT")
fi

run_rsync() {
  local label="$1"
  local src="$2"
  local dst="$3"
  shift 3

  echo
  echo "==> $label"
  echo "    from: $src"
  echo "      to: $dst"
  mkdir -p "$dst"
  rsync "${RSYNC_BASE[@]}" "$@" "$src" "$dst"
}

sync_data() {
  mkdir -p "$DST_ROOT/data"

  run_rsync "data/raw" \
    "$SOURCE:$SRC_ROOT/data/raw/" \
    "$DST_ROOT/data/raw/"

  run_rsync "data/processed" \
    "$SOURCE:$SRC_ROOT/data/processed/" \
    "$DST_ROOT/data/processed/"

  run_rsync "data/stage2_cot_datasets" \
    "$SOURCE:$SRC_ROOT/data/stage2_cot_datasets/" \
    "$DST_ROOT/data/stage2_cot_datasets/"

  run_rsync "data/stage2_cot_datasets_smoke" \
    "$SOURCE:$SRC_ROOT/data/stage2_cot_datasets_smoke/" \
    "$DST_ROOT/data/stage2_cot_datasets_smoke/"

  run_rsync "data/stage2_dualmode_datasets" \
    "$SOURCE:$SRC_ROOT/data/stage2_dualmode_datasets/" \
    "$DST_ROOT/data/stage2_dualmode_datasets/"

  run_rsync "data/stage2_formal_datasets jsonl only" \
    "$SOURCE:$SRC_ROOT/data/stage2_formal_datasets/" \
    "$DST_ROOT/data/stage2_formal_datasets/" \
    --include='*/' --include='*.jsonl' --exclude='*'
}

sync_outputs_light() {
  run_rsync "outputs without checkpoints or model weights" \
    "$SOURCE:$SRC_ROOT/outputs/" \
    "$DST_ROOT/outputs/" \
    --exclude='checkpoint-*/' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.pt' \
    --exclude='*.pth'
}

sync_outputs_no_checkpoints() {
  run_rsync "outputs without checkpoint directories" \
    "$SOURCE:$SRC_ROOT/outputs/" \
    "$DST_ROOT/outputs/" \
    --exclude='checkpoint-*/'
}

sync_docker() {
  run_rsync "llamafactory-lab Docker environment" \
    "$SOURCE:$DOCKER_SRC_ROOT/" \
    "$DOCKER_DST_ROOT/"
}

echo "Source: $SOURCE"
echo "Mode: $MODE"
echo "Source project: $SRC_ROOT"
echo "Target project: $DST_ROOT"
echo "Dry run: $DRY_RUN"
echo "Delete mirror mode: $DELETE"

case "$MODE" in
  data)
    sync_data
    ;;
  outputs-light)
    sync_outputs_light
    ;;
  outputs-no-checkpoints)
    sync_outputs_no_checkpoints
    ;;
  docker)
    sync_docker
    ;;
  all-light)
    sync_data
    sync_outputs_light
    ;;
  all-no-checkpoints)
    sync_data
    sync_outputs_no_checkpoints
    ;;
esac

echo
echo "Sync finished. Run scripts/migration/verify_migration_payload.sh to inspect local payload sizes."
