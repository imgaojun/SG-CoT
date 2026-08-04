#!/usr/bin/env python3
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.prepare_modular_dualexpert_aet_stable_router_m02_20260520 as stable


stable.BRANCH = "aet_trigger_safe_router_m03_routecls_noauxwarm_lr2e6_save50"
stable.LABEL_SOURCE = "modular_d1930_r2058_aet_trigger_safe_m03"
stable.TITLE = "Stage2 A/E/T Trigger-Safe Router M03 D1930/R2058"
stable.OBJECTIVE = "Train a route-only selector with train/dev bucket-stable labels that require nonnegative Trigger gain."
stable.GOAL = "Train a trigger-safe stable router to preserve the A/E gains from m02 while reducing the small formal Trigger regression."
stable.RULE = (
    "reason iff reason_valid_json and argument/event/trigger gains are nonnegative, "
    "with train/dev bucket stability and trigger-safe hard negatives"
)
stable.REASON_OVERSAMPLE = 4
TRIGGER_MAX_HARM_RATE = 0.25
TRIGGER_MIN_MEAN_GAIN = 0.0


def is_trigger_safe(g, valid):
    return (
        valid
        and g["argument_gain"] >= 0.0
        and g["event_gain"] >= 0.0
        and g["trigger_gain"] >= 0.0
        and max(g["argument_gain"], g["event_gain"], g["trigger_gain"]) >= 0.005
    )


def collect_trigger_safe_bucket_stats():
    stats = defaultdict(
        lambda: {"count": 0, "safe": 0, "harm": 0, "trigger_harm": 0, "gain_sum": 0.0, "trigger_gain_sum": 0.0}
    )
    for split in ["train", "dev_seen"]:
        paths = stable.base.PREDICTIONS[split]
        direct_rows = stable.base.load_prediction_map(paths["direct"])
        reason_rows = stable.base.load_prediction_map(paths["reason"])
        for key in sorted(set(direct_rows) & set(reason_rows)):
            direct = direct_rows[key]
            reason = reason_rows[key]
            g = stable.gains(direct, reason)
            reason_gain = stable.base.score(reason) - stable.base.score(direct)
            bucket = stable.bucket_key(direct, reason)
            stats[bucket]["count"] += 1
            stats[bucket]["gain_sum"] += reason_gain
            stats[bucket]["trigger_gain_sum"] += g["trigger_gain"]
            if is_trigger_safe(g, stable.base.valid_json(reason)):
                stats[bucket]["safe"] += 1
            if g["argument_gain"] < 0 or g["event_gain"] < 0 or g["trigger_gain"] < 0:
                stats[bucket]["harm"] += 1
            if g["trigger_gain"] < 0:
                stats[bucket]["trigger_harm"] += 1
    out = {}
    for bucket, row in stats.items():
        count = row["count"]
        mean_gain = row["gain_sum"] / count if count else 0.0
        mean_trigger_gain = row["trigger_gain_sum"] / count if count else 0.0
        harm_rate = row["harm"] / count if count else 0.0
        trigger_harm_rate = row["trigger_harm"] / count if count else 0.0
        out[bucket] = {
            **row,
            "mean_gain": mean_gain,
            "mean_trigger_gain": mean_trigger_gain,
            "harm_rate": harm_rate,
            "trigger_harm_rate": trigger_harm_rate,
            "stable_reason_bucket": (
                count >= stable.BUCKET_MIN_COUNT
                and row["safe"] > 0
                and mean_gain >= stable.BUCKET_MIN_MEAN_GAIN
                and harm_rate <= stable.BUCKET_MAX_HARM_RATE
                and mean_trigger_gain >= TRIGGER_MIN_MEAN_GAIN
                and trigger_harm_rate <= TRIGGER_MAX_HARM_RATE
            ),
        }
    return out


stable.is_aet_safe = is_trigger_safe
stable.collect_bucket_stats = collect_trigger_safe_bucket_stats


if __name__ == "__main__":
    stable.main()
