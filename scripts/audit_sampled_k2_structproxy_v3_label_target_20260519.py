#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.analyze_sampled_k2_seedpair_robustness_20260518 import (  # noqa: E402
    SEED_PAIRS,
    build_feature_rows,
)
from scripts.prepare_sampled_confident_router_20260518 import (  # noqa: E402
    label_path,
    load_jsonl,
)
from scripts.summarize_sampled_confident_router_dev_20260518 import (  # noqa: E402
    pct,
    signed,
    write_json,
    write_text,
)


TZ = timezone(timedelta(hours=8))
ARG_TEXT_JACCARD_MIN = 0.40
EVENT_COUNT_DELTA_MAX = 0.0
DEFAULT_OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_structproxy_v3_label_audit_20260519"
DEFAULT_REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_structproxy_v3_label_target_audit.md"
DEFAULT_REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_structproxy_v3_label_target_audit.json"


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def structural_pass(features):
    arg_j = 1.0 - features["route_argument_disagreement"]
    return arg_j >= ARG_TEXT_JACCARD_MIN and features["reason_minus_direct_event_count_mean"] <= EVENT_COUNT_DELTA_MAX


def pair_stats(label, features_by_pair):
    key = label["wnd_id"]
    pair_rows = []
    for pair_name, _seeds in SEED_PAIRS:
        feat = features_by_pair[pair_name][key]
        arg_j = 1.0 - feat["route_argument_disagreement"]
        event_delta = feat["reason_minus_direct_event_count_mean"]
        pair_rows.append(
            {
                "pair": pair_name,
                "pass": structural_pass(feat),
                "arg_text_jaccard": arg_j,
                "event_count_delta": event_delta,
                "route_argument_disagreement": feat["route_argument_disagreement"],
            }
        )
    return {
        "key": key,
        "utility_label": label.get("utility_label"),
        "mean_gain": float(label.get("mean_gain", 0.0)),
        "p_win": float(label.get("p_win", 0.0)),
        "pass_count": sum(1 for row in pair_rows if row["pass"]),
        "fail_count": sum(1 for row in pair_rows if not row["pass"]),
        "mean_arg_text_jaccard": mean(row["arg_text_jaccard"] for row in pair_rows),
        "min_arg_text_jaccard": min(row["arg_text_jaccard"] for row in pair_rows),
        "mean_event_count_delta": mean(row["event_count_delta"] for row in pair_rows),
        "max_event_count_delta": max(row["event_count_delta"] for row in pair_rows),
        "event_expanding_pair_count": sum(1 for row in pair_rows if row["event_count_delta"] > 0.0),
        "low_arg_pair_count": sum(1 for row in pair_rows if row["arg_text_jaccard"] < ARG_TEXT_JACCARD_MIN),
        "pairs": pair_rows,
    }


def feature_maps(split, labels):
    return {
        pair_name: {row["key"]: row["features"] for row in build_feature_rows(split, seeds, labels)}
        for pair_name, seeds in SEED_PAIRS
    }


def load_split_candidates(split):
    labels = load_jsonl(label_path(split))
    confident = [row for row in labels if row.get("utility_label") in {"stable_reason", "stable_direct"}]
    features_by_pair = feature_maps(split, labels)
    candidates = [pair_stats(label, features_by_pair) for label in confident]
    return labels, candidates


def is_hard_structural_negative(candidate):
    return (
        candidate["utility_label"] == "stable_reason"
        and (
            candidate["pass_count"] <= 1
            or candidate["event_expanding_pair_count"] > 0
            or candidate["mean_arg_text_jaccard"] < ARG_TEXT_JACCARD_MIN
        )
    )


def summarize_candidates(split, labels, candidates):
    counts = Counter(row["utility_label"] for row in labels)
    confident_counts = Counter(row["utility_label"] for row in candidates)
    pass_hist = Counter(row["pass_count"] for row in candidates if row["utility_label"] == "stable_reason")
    hard_structural = [row for row in candidates if is_hard_structural_negative(row)]
    return {
        "split": split,
        "source_count": len(labels),
        "confident_count": len(candidates),
        "label_counts": dict(counts),
        "confident_label_counts": dict(confident_counts),
        "stable_reason_pass_count_histogram": {str(key): value for key, value in sorted(pass_hist.items())},
        "hard_structural_negative_unique_count": len(hard_structural),
        "hard_structural_negative_mean_gain": mean(row["mean_gain"] for row in hard_structural),
        "hard_structural_negative_mean_arg_text_jaccard": mean(row["mean_arg_text_jaccard"] for row in hard_structural),
        "hard_structural_negative_mean_event_count_delta": mean(row["mean_event_count_delta"] for row in hard_structural),
    }


def route_for_variant(candidate, pair, variant):
    if candidate["utility_label"] != "stable_reason":
        return "direct"
    pair_pass = pair["pass"]
    if variant["min_pass_count"] is not None and candidate["pass_count"] < variant["min_pass_count"]:
        return "direct"
    if variant["require_pair_pass"] and not pair_pass:
        return "direct"
    if candidate["mean_arg_text_jaccard"] < variant["mean_arg_min"]:
        return "direct"
    if candidate["max_event_count_delta"] > variant["max_event_delta_max"]:
        return "direct"
    return "reason"


def audit_variant(split, candidates, variant):
    rows = []
    unique_reason = set()
    unique_direct = set()
    hard_negative_rows = 0
    for candidate in candidates:
        hard_negative = is_hard_structural_negative(candidate)
        for pair in candidate["pairs"]:
            route = route_for_variant(candidate, pair, variant)
            repeat = variant["reason_repeat"] if route == "reason" else 1
            if route == "direct" and hard_negative:
                repeat += variant["hard_negative_extra_direct_repeat"]
                hard_negative_rows += repeat
            for _idx in range(repeat):
                rows.append((candidate, route))
            if route == "reason":
                unique_reason.add(candidate["key"])
            else:
                unique_direct.add(candidate["key"])

    route_counts = Counter(route for _candidate, route in rows)
    reason_candidates = [candidate for candidate, route in rows if route == "reason"]
    direct_candidates = [candidate for candidate, route in rows if route == "direct"]
    return {
        "split": split,
        "variant": variant["name"],
        "description": variant["description"],
        "total_rows": len(rows),
        "reason_rows": route_counts["reason"],
        "direct_rows": route_counts["direct"],
        "reason_rate": route_counts["reason"] / len(rows) if rows else 0.0,
        "unique_reason_count": len(unique_reason),
        "unique_direct_count": len(unique_direct),
        "hard_negative_direct_rows": hard_negative_rows,
        "reason_mean_gain": mean(candidate["mean_gain"] for candidate in reason_candidates),
        "direct_mean_gain": mean(candidate["mean_gain"] for candidate in direct_candidates),
        "reason_mean_pass_count": mean(candidate["pass_count"] for candidate in reason_candidates),
        "reason_mean_arg_text_jaccard": mean(candidate["mean_arg_text_jaccard"] for candidate in reason_candidates),
        "reason_mean_event_count_delta": mean(candidate["mean_event_count_delta"] for candidate in reason_candidates),
    }


def render_table(rows):
    lines = [
        "| split | variant | rows | reason | direct | reason rate | unique reason | hard-neg direct | reason gain | pass cnt | arg J | event delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['split']}` | `{row['variant']}` | {row['total_rows']} | "
            f"{row['reason_rows']} | {row['direct_rows']} | {pct(row['reason_rate'])} | "
            f"{row['unique_reason_count']} | {row['hard_negative_direct_rows']} | "
            f"{signed(row['reason_mean_gain'])} | {row['reason_mean_pass_count']:.2f} | "
            f"{row['reason_mean_arg_text_jaccard']:.4f} | {row['reason_mean_event_count_delta']:.4f} |"
        )
    return "\n".join(lines)


def render_report(payload):
    candidate_lines = []
    for row in payload["candidate_summaries"]:
        candidate_lines.extend(
            [
                f"### `{row['split']}`",
                "",
                f"- source labels: `{row['source_count']}`; confident labels: `{row['confident_count']}`.",
                f"- confident counts: `{row['confident_label_counts']}`.",
                f"- stable-reason pass-count histogram: `{row['stable_reason_pass_count_histogram']}`.",
                f"- hard structural negative unique count: `{row['hard_structural_negative_unique_count']}`.",
                f"- hard structural negative mean gain / arg-J / event-delta: "
                f"{signed(row['hard_structural_negative_mean_gain'])} / "
                f"{row['hard_structural_negative_mean_arg_text_jaccard']:.4f} / "
                f"{row['hard_structural_negative_mean_event_count_delta']:.4f}.",
                "",
            ]
        )
    lines = [
        "# Sampled K2 StructProxy v3 Label Target Audit",
        "",
        "This audits candidate v3 route-supervision targets before launching another training run.",
        "The goal is to keep the locked structural proxy as the conceptual base while teaching explicit direct labels for structurally weak reason candidates.",
        "",
        "## Candidate Pool",
        "",
        *candidate_lines,
        "## Variant Audit",
        "",
        render_table(payload["variant_rows"]),
        "",
        "## Reading",
        "",
        "- `pairwise_consensus2_repeat4` keeps the v2 positive criterion but labels every non-passing seedpair/context as direct instead of relying on a sampled hard-negative subset.",
        "- `pairwise_consensus2_hardneg3` adds direct duplicates for stable-reason candidates that are structurally weak, matching the observed v2-only failure profile.",
        "- `pairwise_consensus3_repeat4` is the conservative option; if its unique reason count collapses, it is better treated as a no-training gate rather than a supervised selector target.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['report_json']}`",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    variants = [
        {
            "name": "pairwise_consensus2_repeat4",
            "description": "reason only for stable_reason candidates with >=2 passing seedpairs and the current pair passing; repeat reason rows 4x.",
            "min_pass_count": 2,
            "require_pair_pass": True,
            "mean_arg_min": 0.0,
            "max_event_delta_max": 999.0,
            "reason_repeat": 4,
            "hard_negative_extra_direct_repeat": 0,
        },
        {
            "name": "pairwise_consensus2_hardneg3",
            "description": "same positives as consensus2, plus 3 extra direct duplicates for structurally weak stable_reason candidates.",
            "min_pass_count": 2,
            "require_pair_pass": True,
            "mean_arg_min": 0.0,
            "max_event_delta_max": 999.0,
            "reason_repeat": 4,
            "hard_negative_extra_direct_repeat": 3,
        },
        {
            "name": "pairwise_consensus3_repeat4",
            "description": "more conservative positives: stable_reason with >=3 passing seedpairs and current pair passing; repeat reason rows 4x.",
            "min_pass_count": 3,
            "require_pair_pass": True,
            "mean_arg_min": 0.0,
            "max_event_delta_max": 999.0,
            "reason_repeat": 4,
            "hard_negative_extra_direct_repeat": 0,
        },
        {
            "name": "pairwise_consensus2_globalstruct_repeat4",
            "description": "consensus2 positives with global mean arg-J >=0.40 and no positive event-count delta on any seedpair.",
            "min_pass_count": 2,
            "require_pair_pass": True,
            "mean_arg_min": ARG_TEXT_JACCARD_MIN,
            "max_event_delta_max": EVENT_COUNT_DELTA_MAX,
            "reason_repeat": 4,
            "hard_negative_extra_direct_repeat": 0,
        },
    ]
    candidate_summaries = []
    variant_rows = []
    for split in args.splits:
        labels, candidates = load_split_candidates(split)
        candidate_summaries.append(summarize_candidates(split, labels, candidates))
        for variant in variants:
            variant_rows.append(audit_variant(split, candidates, variant))

    payload = {
        "created_at": now_iso(),
        "arg_text_jaccard_min": ARG_TEXT_JACCARD_MIN,
        "event_count_delta_max": EVENT_COUNT_DELTA_MAX,
        "splits": args.splits,
        "candidate_summaries": candidate_summaries,
        "variant_rows": variant_rows,
        "report_md": args.report_md.as_posix(),
        "report_json": args.report_json.as_posix(),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.report_json, payload)
    write_json(args.output_root / "summary.json", payload)
    write_text(args.report_md, render_report(payload))
    print(json.dumps({"report_md": args.report_md.as_posix(), "report_json": args.report_json.as_posix()}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "dev_seen"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
