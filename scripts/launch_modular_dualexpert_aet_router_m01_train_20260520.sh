#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 branch=gpu [branch=gpu ...]" >&2
  echo "branches:" >&2
  echo "  aet_safe_router_m01_routecls_noauxwarm_lr2e6_save50" >&2
  echo "  aet_event_router_m01_routecls_noauxwarm_lr2e6_save50" >&2
  exit 2
fi

bash scripts/launch_modular_dualexpert_utility_router_train_20260517.sh "$@"
