#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 aet_rankstable_router_m04b_routecls_noauxwarm_lr2e6_save50=<gpu>" >&2
  exit 2
fi

bash scripts/launch_modular_dualexpert_utility_router_train_20260517.sh "$@"
