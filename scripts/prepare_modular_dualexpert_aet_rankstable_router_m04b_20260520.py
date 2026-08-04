#!/usr/bin/env python3
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.prepare_modular_dualexpert_aet_rankstable_router_m04a_20260520 as prep


prep.BRANCH = "aet_rankstable_router_m04b_routecls_noauxwarm_lr2e6_save50"
prep.LABEL_SOURCE = "modular_d1930_r2058_aet_rankstable_m04b"
prep.METHOD_ID = "m04b"
prep.TITLE = "Stage2 A/E/T Rank-Stable Router M04B D1930/R2058"
prep.OBJECTIVE = "Train a route-only selector with m02 stable labels plus lighter hard-negative duplication."
prep.GOAL = (
    "Test whether lighter hard-negative weighting preserves m02 Argument/Event positives while keeping "
    "some of m04a's Trigger protection."
)
prep.RULE = (
    "reason iff reason_valid_json and A/E/T-safe gain, with train/dev bucket stability; "
    "train rows are duplicated by lighter rank-stability class weights"
)
prep.WEIGHTS = {
    "stable_reason_positive": 5,
    "safe_unstable_hard_negative": 2,
    "harmful_reason_looking_hard_negative": 2,
    "ordinary_direct": 1,
}


if __name__ == "__main__":
    prep.main()
