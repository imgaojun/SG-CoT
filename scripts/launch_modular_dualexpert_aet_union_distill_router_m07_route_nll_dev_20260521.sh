#!/usr/bin/env bash
set -euo pipefail

SCORE_ROOT="outputs/stage2_modular_dualexpert/aet_union_distill_router_m07_20260521/route_likelihood" \
  bash scripts/launch_modular_dualexpert_utility_router_route_nll_dev_20260517.sh "$@"
