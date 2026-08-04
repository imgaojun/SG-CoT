#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: archive_training_checkpoints_20260713.sh <run-dir> [archive-root]}"
ARCHIVE_ROOT="${2:-/mnt/manhattan/manhadun_backup/gaojun/progressive-ee/checkpoints}"
PROJECT_ROOT="/mnt/disk/gaojun/research/progressive-ee"
IMAGE="llamafactory-lab:0.9.4-py3.12"

[[ "${RUN_DIR}" = /* ]] || RUN_DIR="${PROJECT_ROOT}/${RUN_DIR}"
[[ -d "${RUN_DIR}" ]] || { echo "missing run directory: ${RUN_DIR}" >&2; exit 2; }
RUN_DIR="$(realpath -e "${RUN_DIR}")"
[[ "${RUN_DIR}" == "${PROJECT_ROOT}/outputs/"* ]] || {
  echo "refusing to archive a run outside ${PROJECT_ROOT}/outputs" >&2
  exit 2
}
run_name="$(basename "${RUN_DIR}")"
manifest_dir="${PROJECT_ROOT}/outputs/checkpoint_archive_manifests"
mkdir -p "${manifest_dir}"
exec 9>"${manifest_dir}/${run_name}.lock"
flock -n 9 || {
  echo "another archive is already running for ${run_name}" >&2
  exit 4
}
mapfile -t checkpoints < <(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "no checkpoint directories to archive under ${RUN_DIR}"
  exit 0
fi

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name="$(basename "${checkpoint}")"
  checkpoint_step="${checkpoint_name#checkpoint-}"
  [[ "${checkpoint_step}" =~ ^[0-9]+$ ]] || {
    echo "invalid checkpoint name: ${checkpoint_name}" >&2
    exit 2
  }
  python3 "${PROJECT_ROOT}/scripts/validate_training_artifact_20260712.py" \
    --model_dir "${checkpoint}" \
    --trainer_state "${checkpoint}/trainer_state.json" \
    --min_global_step "${checkpoint_step}" \
    --require_finite_step_log >/dev/null
done

destination="${ARCHIVE_ROOT}/${run_name}"
manifest_local="${manifest_dir}/${run_name}.sha256"
manifest_remote="${destination}/checkpoint_archive_manifest.sha256"
mkdir -p "${destination}"
: > "${manifest_local}"

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name="$(basename "${checkpoint}")"
  (
    cd "${RUN_DIR}"
    find "${checkpoint_name}" -type f -print0 | sort -z | xargs -0 -n 1 sha256sum
  ) >> "${manifest_local}"
  mkdir -p "${destination}/${checkpoint_name}"
  # Manhattan is a FUSE mount that rejects chmod/mtime updates; content
  # integrity is enforced by the manifest below rather than file metadata.
  rsync -r --no-owner --no-group --no-perms --no-times --omit-dir-times \
    --size-only --partial \
    "${checkpoint}/" "${destination}/${checkpoint_name}/"
done

cp "${manifest_local}" "${manifest_remote}"
(
  cd "${destination}"
  while IFS= read -r manifest_line; do
    printf '%s\n' "${manifest_line}" | sha256sum -c -
  done < "$(basename "${manifest_remote}")"
)

if [[ "${ARCHIVE_DELETE_LOCAL:-0}" != "1" ]]; then
  echo "archive verified; local checkpoints retained. Set ARCHIVE_DELETE_LOCAL=1 to remove them."
  exit 0
fi

for checkpoint in "${checkpoints[@]}"; do
  relative_checkpoint="${checkpoint#${PROJECT_ROOT}/}"
  [[ "${checkpoint}" == "${RUN_DIR}/checkpoint-"* && "${relative_checkpoint}" != "${checkpoint}" && "${relative_checkpoint}" == outputs/*/checkpoint-* ]] || {
    echo "refusing unsafe checkpoint deletion: ${checkpoint}" >&2
    exit 3
  }
  docker run --rm --user root \
    -v "${PROJECT_ROOT}:/workspace/project" \
    "${IMAGE}" rm -rf -- "/workspace/project/${relative_checkpoint}"
  [[ ! -e "${checkpoint}" ]] || {
    echo "checkpoint deletion failed: ${checkpoint}" >&2
    exit 3
  }
done
echo "archive verified and ${#checkpoints[@]} local checkpoints removed from ${RUN_DIR}"
