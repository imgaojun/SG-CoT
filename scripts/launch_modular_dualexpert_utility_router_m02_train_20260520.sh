#!/usr/bin/env bash
set -euo pipefail

BRANCH="modular_d1930_r2058_utility_m02_routecls_noauxwarm_lr2e6_save50"

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <gpu_id>" >&2
  exit 2
fi

bash scripts/launch_modular_dualexpert_utility_router_train_20260517.sh "${BRANCH}=$1"
