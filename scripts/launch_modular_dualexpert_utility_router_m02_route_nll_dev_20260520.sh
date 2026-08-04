#!/usr/bin/env bash
set -euo pipefail

BRANCH="modular_d1930_r2058_utility_m02_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT="outputs/stage2_modular_dualexpert/utility_router_m02_20260520/route_likelihood" \
  bash scripts/launch_modular_dualexpert_utility_router_route_nll_dev_20260517.sh "${BRANCH}=${1:-0}"
