import json
import statistics
import sys
from collections import Counter, defaultdict
from hashlib import md5
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    DIRECT_ROOT,
    REASON_ROOT,
    POLICIES,
)
from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key, score  # noqa: E402
from src.stage2_analysis.analyze_adaptive_outcome_router_execution import row_metric, summarize_metrics  # noqa: E402
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


DEV_SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_router_m01_20260520/route_likelihood"
FORMAL_SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_router_m01_20260520/formal_route_likelihood"
DIRECT_DEV = REPO / (
    "outputs/stage2_adaptive_runs_user_devpick_frontier/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux_"
    "full_forced_direct_dev_seen_max512/checkpoint-1930/predictions.jsonl"
)
REASON_DEV = REPO / (
    "outputs/stage2_adaptive_runs_user_devpick_frontier/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux_"
    "full_forced_reason_dev_seen_max512/checkpoint-2058/predictions.jsonl"
)
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_router_m01_goldfree_guard_drift.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_router_m01_goldfree_guard_drift.md"

SPLITS = ["dev_seen", "test_seen", "test_unseen"]


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def load_score_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def score_path(policy, split):
    if split == "dev_seen":
        return DEV_SCORE_ROOT / policy["branch"] / policy["checkpoint"] / "dev_seen_scores.jsonl"
    return FORMAL_SCORE_ROOT / policy["branch"] / policy["checkpoint"] / split / "scores.jsonl"


def prediction_paths(split):
    if split == "dev_seen":
        return DIRECT_DEV, REASON_DEV
    return DIRECT_ROOT / split / "predictions.jsonl", REASON_ROOT / split / "predictions.jsonl"


def sorted_keys(score_rows, common_keys):
    items = []
    for key in set(score_rows) & set(common_keys):
        delta = score_rows[key].get("delta_direct_minus_reason_route_nll")
        if delta is None:
            delta = float("-inf")
        items.append((float(delta), key))
    items.sort(reverse=True)
    return [key for _, key in items]


def event_count(row):
    pred = row.get("predicted") or row.get("final_predicted") or {}
    events = pred.get("events") if isinstance(pred, dict) else []
    return len(events) if isinstance(events, list) else 0


def argument_count(row):
    pred = row.get("predicted") or row.get("final_predicted") or {}
    events = pred.get("events") if isinstance(pred, dict) else []
    if not isinstance(events, list):
        return 0
    total = 0
    for event in events:
        args = event.get("arguments") if isinstance(event, dict) else []
        if isinstance(args, list):
            total += len(args)
    return total


def payload_len(row):
    text = row.get("generated_payload") or row.get("generated_text") or ""
    return len(text)


def event_family(meta):
    types = meta.get("candidate_types") or []
    families = sorted({str(t).split(":", 1)[0] for t in types})
    return "+".join(families[:3]) if families else "unknown"


def row_features(key, rank, n, score_row, direct_row, reason_row):
    direct_events = event_count(direct_row)
    reason_events = event_count(reason_row)
    direct_args = argument_count(direct_row)
    reason_args = argument_count(reason_row)
    margin = float(score_row.get("delta_direct_minus_reason_route_nll") or 0.0)
    return {
        "key": key,
        "rank": rank,
        "rank_pct": rank / n if n else 0.0,
        "margin": margin,
        "nll_direct_route": score_row.get("nll_direct_route"),
        "nll_reason_route": score_row.get("nll_reason_route"),
        "direct_valid_json": bool(direct_row.get("valid_json")),
        "reason_valid_json": bool(reason_row.get("valid_json")),
        "direct_len": payload_len(direct_row),
        "reason_len": payload_len(reason_row),
        "len_ratio": payload_len(reason_row) / max(payload_len(direct_row), 1),
        "direct_event_count": direct_events,
        "reason_event_count": reason_events,
        "event_count_delta": reason_events - direct_events,
        "direct_argument_count": direct_args,
        "reason_argument_count": reason_args,
        "argument_count_delta": reason_args - direct_args,
        "candidate_type_count": len((score_row.get("meta") or {}).get("candidate_types") or []),
        "event_family": event_family(score_row.get("meta") or {}),
    }


def build_rows(policy, split):
    direct_path, reason_path = prediction_paths(split)
    score_rows = load_score_map(score_path(policy, split))
    direct_rows = load_prediction_map(direct_path)
    reason_rows = load_prediction_map(reason_path)
    keys = sorted_keys(score_rows, set(direct_rows) & set(reason_rows))
    start = round(len(keys) * policy["start_pct"])
    end = round(len(keys) * policy["end_pct"])
    selected = set(keys[start:end])
    rows = []
    for rank, key in enumerate(keys, start=1):
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        feature = row_features(key, rank, len(keys), score_rows[key], direct_row, reason_row)
        chosen = reason_row if key in selected else direct_row
        reason_gain = {
            "argument_f1": row_metric(reason_row)["argument"]["f1"] - row_metric(direct_row)["argument"]["f1"],
            "event_f1": row_metric(reason_row)["event"]["f1"] - row_metric(direct_row)["event"]["f1"],
            "trigger_f1": row_metric(reason_row)["trigger"]["f1"] - row_metric(direct_row)["trigger"]["f1"],
            "score": score(reason_row) - score(direct_row),
        }
        rows.append(
            {
                **feature,
                "split": split,
                "policy": policy["name"],
                "branch": policy["branch"],
                "checkpoint": policy["checkpoint"],
                "base_selected": key in selected,
                "direct_metric": row_metric(direct_row),
                "reason_metric": row_metric(reason_row),
                "chosen_metric": row_metric(chosen),
                "reason_gain": reason_gain,
            }
        )
    return rows


def keep_by_guard(row, guard):
    if not row["base_selected"]:
        return False
    if guard["require_reason_valid"] and not row["reason_valid_json"]:
        return False
    if row["margin"] < guard["min_margin"]:
        return False
    if row["margin"] > guard["max_margin"]:
        return False
    if row["len_ratio"] < guard["min_len_ratio"]:
        return False
    if row["len_ratio"] > guard["max_len_ratio"]:
        return False
    if abs(row["event_count_delta"]) > guard["max_abs_event_count_delta"]:
        return False
    if abs(row["argument_count_delta"]) > guard["max_abs_argument_count_delta"]:
        return False
    return True


def summarize_policy(rows, guard):
    routed_metrics = []
    direct_metrics = []
    selected_gains = []
    selected = 0
    for row in rows:
        direct_metric = row["direct_metric"]
        reason_metric = row["reason_metric"]
        if keep_by_guard(row, guard):
            selected += 1
            routed_metrics.append(reason_metric)
            selected_gains.append(row["reason_gain"]["score"])
        else:
            routed_metrics.append(direct_metric)
        direct_metrics.append(direct_metric)
    routed = summarize_metrics(routed_metrics)
    direct = summarize_metrics(direct_metrics)
    return {
        "num_examples": len(rows),
        "reason_count": selected,
        "reason_rate": selected / len(rows) if rows else 0.0,
        "selected_reason_avg_score_gain": sum(selected_gains) / len(selected_gains) if selected_gains else 0.0,
        "routed": routed,
        "direct": direct,
        "routed_minus_direct": {
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
        },
    }


def aggregate_test(seen, unseen):
    total = seen["num_examples"] + unseen["num_examples"]
    selected = seen["reason_count"] + unseen["reason_count"]
    out = {
        "num_examples": total,
        "reason_count": selected,
        "reason_rate": selected / total if total else 0.0,
    }
    denom = selected
    out["selected_reason_avg_score_gain"] = (
        (
            seen["selected_reason_avg_score_gain"] * seen["reason_count"]
            + unseen["selected_reason_avg_score_gain"] * unseen["reason_count"]
        )
        / denom
        if denom
        else 0.0
    )
    for group in ["routed", "direct"]:
        out[group] = {}
        for metric in ["argument_f1", "event_f1", "trigger_f1"]:
            out[group][metric] = (
                seen[group][metric] * seen["num_examples"] + unseen[group][metric] * unseen["num_examples"]
            ) / total
    out["routed_minus_direct"] = {
        metric: out["routed"][metric] - out["direct"][metric]
        for metric in ["argument_f1", "event_f1", "trigger_f1"]
    }
    return out


def guard_grid():
    guards = []
    for min_margin in [-2.0, -1.0, -0.5, 0.0]:
        for max_margin in [0.5, 1.0, 2.0, 99.0]:
            if max_margin <= min_margin:
                continue
            for max_event_delta in [0, 1, 2, 99]:
                for max_arg_delta in [1, 2, 4, 99]:
                    guards.append(
                        {
                            "name": (
                                f"m{min_margin:g}_{max_margin:g}_"
                                f"ed{max_event_delta}_ad{max_arg_delta}"
                            ),
                            "require_reason_valid": True,
                            "min_margin": min_margin,
                            "max_margin": max_margin,
                            "min_len_ratio": 0.25,
                            "max_len_ratio": 4.0,
                            "max_abs_event_count_delta": max_event_delta,
                            "max_abs_argument_count_delta": max_arg_delta,
                        }
                    )
    guards.append(
        {
            "name": "base_window",
            "require_reason_valid": False,
            "min_margin": -99.0,
            "max_margin": 99.0,
            "min_len_ratio": 0.0,
            "max_len_ratio": 999.0,
            "max_abs_event_count_delta": 999,
            "max_abs_argument_count_delta": 999,
        }
    )
    return guards


def window_grid():
    endpoints = [i / 40 for i in range(0, 21)]
    windows = []
    for lo in endpoints:
        for hi in endpoints:
            if hi <= lo:
                continue
            rate = hi - lo
            if 0.05 <= rate <= 0.20:
                windows.append((lo, hi))
    return windows


def summarize_window(rows, lo, hi):
    n = len(rows)
    start = round(n * lo)
    end = round(n * hi)
    keys = {row["key"] for row in rows[start:end]}
    guard = {
        "name": f"window_{int(lo * 1000):03d}_{int(hi * 1000):03d}",
        "require_reason_valid": False,
        "min_margin": -99.0,
        "max_margin": 99.0,
        "min_len_ratio": 0.0,
        "max_len_ratio": 999.0,
        "max_abs_event_count_delta": 999,
        "max_abs_argument_count_delta": 999,
    }
    routed_metrics = []
    direct_metrics = []
    selected_gains = []
    for row in rows:
        direct_metric = row["direct_metric"]
        reason_metric = row["reason_metric"]
        if row["key"] in keys:
            routed_metrics.append(reason_metric)
            selected_gains.append(row["reason_gain"]["score"])
        else:
            routed_metrics.append(direct_metric)
        direct_metrics.append(direct_metric)
    routed = summarize_metrics(routed_metrics)
    direct = summarize_metrics(direct_metrics)
    return {
        "guard": guard["name"],
        "window": {"start_pct": lo, "end_pct": hi},
        "num_examples": len(rows),
        "reason_count": len(keys),
        "reason_rate": len(keys) / len(rows) if rows else 0.0,
        "selected_reason_avg_score_gain": sum(selected_gains) / len(selected_gains) if selected_gains else 0.0,
        "routed": routed,
        "direct": direct,
        "routed_minus_direct": {
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
        },
    }


def selected_keys_for_window(rows, lo, hi):
    n = len(rows)
    start = round(n * lo)
    end = round(n * hi)
    return {row["key"] for row in rows[start:end]}


def summarize_selected_keys(rows, selected_keys):
    selected_keys = set(selected_keys)
    routed_metrics = []
    direct_metrics = []
    selected_gains = []
    selected = 0
    for row in rows:
        direct_metric = row["direct_metric"]
        reason_metric = row["reason_metric"]
        if row["key"] in selected_keys:
            selected += 1
            routed_metrics.append(reason_metric)
            selected_gains.append(row["reason_gain"]["score"])
        else:
            routed_metrics.append(direct_metric)
        direct_metrics.append(direct_metric)
    routed = summarize_metrics(routed_metrics)
    direct = summarize_metrics(direct_metrics)
    return {
        "num_examples": len(rows),
        "reason_count": selected,
        "reason_rate": selected / len(rows) if rows else 0.0,
        "selected_reason_avg_score_gain": sum(selected_gains) / len(selected_gains) if selected_gains else 0.0,
        "routed": routed,
        "direct": direct,
        "routed_minus_direct": {
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
        },
    }


def stable_fold(key):
    return int(md5(key.encode("utf-8")).hexdigest()[:8], 16) % 5


def balanced_score(summary):
    d = summary["routed_minus_direct"]
    if summary["reason_rate"] < 0.05:
        return (-99.0, -99.0, -99.0, -99.0)
    return (
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        d["event_f1"],
        d["argument_f1"],
        d["trigger_f1"],
    )


def event_score(summary):
    d = summary["routed_minus_direct"]
    if summary["reason_rate"] < 0.05:
        return (-99.0, -99.0, -99.0, -99.0)
    if d["argument_f1"] < -0.001 or d["trigger_f1"] < -0.001:
        return (-99.0, -99.0, -99.0, -99.0)
    return (d["event_f1"], min(d["argument_f1"], d["trigger_f1"]), d["argument_f1"], d["trigger_f1"])


def is_all_nonnegative(summary):
    d = summary["routed_minus_direct"]
    return d["argument_f1"] >= 0 and d["event_f1"] >= 0 and d["trigger_f1"] >= 0


def is_event_safe(summary):
    d = summary["routed_minus_direct"]
    return d["event_f1"] > 0 and d["argument_f1"] >= -0.001 and d["trigger_f1"] >= -0.001


def robust_dev_window_selection(policy_name, rows_by_split):
    dev_rows = rows_by_split["dev_seen"]
    candidates = []
    for lo, hi in window_grid():
        dev_selected = selected_keys_for_window(dev_rows, lo, hi)
        dev = summarize_selected_keys(dev_rows, dev_selected)
        if dev["reason_rate"] < 0.05 or dev["reason_rate"] > 0.20:
            continue
        if policy_name.startswith("aet_safe") and not is_all_nonnegative(dev):
            continue
        if policy_name.startswith("aet_event") and not is_event_safe(dev):
            continue

        fold_summaries = []
        for fold in range(5):
            fold_rows = [row for row in dev_rows if stable_fold(row["key"]) == fold]
            if fold_rows:
                fold_summaries.append(summarize_selected_keys(fold_rows, dev_selected))
        if not fold_summaries:
            continue
        fold_min_aet = min(
            min(
                fold["routed_minus_direct"]["argument_f1"],
                fold["routed_minus_direct"]["event_f1"],
                fold["routed_minus_direct"]["trigger_f1"],
            )
            for fold in fold_summaries
        )
        fold_event_min = min(fold["routed_minus_direct"]["event_f1"] for fold in fold_summaries)
        candidates.append(
            {
                "policy": policy_name,
                "window": {"start_pct": lo, "end_pct": hi},
                "dev_seen": dev,
                "fold_min_aet": fold_min_aet,
                "fold_event_min": fold_event_min,
                "fold_summaries": fold_summaries,
            }
        )

    if not candidates:
        return None
    if policy_name.startswith("aet_event"):
        selected = max(
            candidates,
            key=lambda row: (
                row["fold_event_min"],
                min(
                    row["dev_seen"]["routed_minus_direct"]["argument_f1"],
                    row["dev_seen"]["routed_minus_direct"]["trigger_f1"],
                ),
                row["dev_seen"]["routed_minus_direct"]["event_f1"],
            ),
        )
    else:
        selected = max(
            candidates,
            key=lambda row: (
                row["fold_min_aet"],
                min(
                    row["dev_seen"]["routed_minus_direct"]["argument_f1"],
                    row["dev_seen"]["routed_minus_direct"]["event_f1"],
                    row["dev_seen"]["routed_minus_direct"]["trigger_f1"],
                ),
                row["dev_seen"]["routed_minus_direct"]["event_f1"],
            ),
        )
    lo = selected["window"]["start_pct"]
    hi = selected["window"]["end_pct"]
    seen = summarize_window(rows_by_split["test_seen"], lo, hi)
    unseen = summarize_window(rows_by_split["test_unseen"], lo, hi)
    selected = dict(selected)
    selected["test_seen"] = seen
    selected["test_unseen"] = unseen
    selected["test"] = aggregate_test(seen, unseen)
    selected["num_candidates"] = len(candidates)
    selected["top_candidates"] = sorted(
        candidates,
        key=lambda row: (
            row["fold_event_min"] if policy_name.startswith("aet_event") else row["fold_min_aet"],
            min(
                row["dev_seen"]["routed_minus_direct"]["argument_f1"],
                row["dev_seen"]["routed_minus_direct"]["event_f1"],
                row["dev_seen"]["routed_minus_direct"]["trigger_f1"],
            ),
        ),
        reverse=True,
    )[:10]
    return selected


def formal_window_diagnostic(policy_name, rows_by_split):
    rows = []
    for lo, hi in window_grid():
        dev = summarize_window(rows_by_split["dev_seen"], lo, hi)
        seen = summarize_window(rows_by_split["test_seen"], lo, hi)
        unseen = summarize_window(rows_by_split["test_unseen"], lo, hi)
        test = aggregate_test(seen, unseen)
        rows.append(
            {
                "policy": policy_name,
                "window": {"start_pct": lo, "end_pct": hi},
                "dev_seen": dev,
                "test_seen": seen,
                "test_unseen": unseen,
                "test": test,
            }
        )
    all_nonnegative = [row for row in rows if is_all_nonnegative(row["test"])]
    event_safe = [row for row in rows if is_event_safe(row["test"])]
    return {
        "num_windows": len(rows),
        "num_formal_all_nonnegative": len(all_nonnegative),
        "num_formal_event_safe": len(event_safe),
        "best_formal_all_nonnegative": max(
            all_nonnegative, key=lambda row: balanced_score(row["test"])
        )
        if all_nonnegative
        else None,
        "best_formal_event_safe": max(event_safe, key=lambda row: event_score(row["test"]))
        if event_safe
        else None,
        "best_formal_event": max(
            rows,
            key=lambda row: (
                row["test"]["routed_minus_direct"]["event_f1"],
                row["test"]["routed_minus_direct"]["argument_f1"],
                row["test"]["routed_minus_direct"]["trigger_f1"],
            ),
        ),
        "top_formal_by_balanced": sorted(
            rows,
            key=lambda row: balanced_score(row["test"]),
            reverse=True,
        )[:10],
        "top_formal_by_event": sorted(
            rows,
            key=lambda row: (
                row["test"]["routed_minus_direct"]["event_f1"],
                row["test"]["routed_minus_direct"]["argument_f1"],
                row["test"]["routed_minus_direct"]["trigger_f1"],
            ),
            reverse=True,
        )[:10],
    }


def drift_stats(rows_by_split):
    stats = {}
    for split, rows in rows_by_split.items():
        selected = [row for row in rows if row["base_selected"]]
        harmful = [
            row
            for row in selected
            if row["reason_gain"]["argument_f1"] < 0
            or row["reason_gain"]["event_f1"] < 0
            or row["reason_gain"]["trigger_f1"] < 0
        ]
        families = Counter(row["event_family"] for row in selected)
        margins = [row["margin"] for row in selected]
        stats[split] = {
            "selected_count": len(selected),
            "harmful_any_aet_count": len(harmful),
            "harmful_any_aet_rate": len(harmful) / len(selected) if selected else 0.0,
            "margin_mean": statistics.mean(margins) if margins else 0.0,
            "margin_median": statistics.median(margins) if margins else 0.0,
            "len_ratio_mean": statistics.mean([row["len_ratio"] for row in selected]) if selected else 0.0,
            "event_count_delta_mean": statistics.mean([row["event_count_delta"] for row in selected]) if selected else 0.0,
            "argument_count_delta_mean": statistics.mean([row["argument_count_delta"] for row in selected]) if selected else 0.0,
            "top_event_families": families.most_common(8),
        }
    return stats


def fmt_delta(d):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**d)


def render_table(rows):
    lines = [
        "| policy | guard | split | reason rate | delta A/E/T | selected gain |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        s = row["summary"]
        lines.append(
            "| {policy} | {guard} | {split} | {rate:.1%} | {delta} | {gain:+.4f} |".format(
                policy=row["policy"],
                guard=row["guard"],
                split=row["split"],
                rate=s["reason_rate"],
                delta=fmt_delta(s["routed_minus_direct"]),
                gain=s["selected_reason_avg_score_gain"],
            )
        )
    return "\n".join(lines)


def render_report(payload):
    lines = [
        "# A/E/T Router M01 Gold-Free Guard Drift Diagnosis",
        "",
        "This diagnostic selects guard policies on `dev_seen` only, then applies them unchanged to formal scores.",
        "",
        "## Dev-Selected Guards",
        "",
        render_table(payload["selected_guard_rows"]),
        "",
        "## Drift Stats",
        "",
    ]
    for policy_name, stats in payload["drift_stats"].items():
        lines.append(f"### {policy_name}")
        lines.append("")
        lines.append("| split | selected | harmful any A/E/T | margin mean | len ratio mean | event delta mean | arg delta mean |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for split in SPLITS:
            row = stats[split]
            lines.append(
                "| {split} | {selected} | {harm:.1%} | {margin:.3f} | {lenr:.2f} | {ed:.2f} | {ad:.2f} |".format(
                    split=split,
                    selected=row["selected_count"],
                    harm=row["harmful_any_aet_rate"],
                    margin=row["margin_mean"],
                    lenr=row["len_ratio_mean"],
                    ed=row["event_count_delta_mean"],
                    ad=row["argument_count_delta_mean"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Formal Window Diagnostic",
            "",
            "This section uses formal labels only as a diagnostic to test whether the scored checkpoint contains any transferable rank window. It must not be used as a locked selector.",
            "",
            "| policy | formal all-nonnegative windows | formal event-safe windows | best formal event A/E/T |",
            "|---|---:|---:|---:|",
        ]
    )
    for policy_name, diag in payload["formal_window_diagnostic"].items():
        best = diag["best_formal_event"]["test"]["routed_minus_direct"]
        lines.append(
            f"| {policy_name} | {diag['num_formal_all_nonnegative']} | "
            f"{diag['num_formal_event_safe']} | {fmt_delta(best)} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Robust Dev Window",
            "",
            "This selector uses dev-only five-fold stability over windows. It is still a diagnostic rule, but unlike the formal window diagnostic it does not inspect formal labels.",
            "",
            "| policy | window | dev fold floor | dev A/E/T | formal test A/E/T | reason rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for policy_name, row in payload["robust_dev_window_selection"].items():
        if row is None:
            lines.append(f"| {policy_name} | n/a | n/a | n/a | n/a | n/a |")
            continue
        window = row["window"]
        floor = row["fold_event_min"] if policy_name.startswith("aet_event") else row["fold_min_aet"]
        lines.append(
            "| {policy} | {lo:.1%}-{hi:.1%} | {floor:+.4f} | {dev} | {test} | {rate:.1%} |".format(
                policy=policy_name,
                lo=window["start_pct"],
                hi=window["end_pct"],
                floor=floor,
                dev=fmt_delta(row["dev_seen"]["routed_minus_direct"]),
                test=fmt_delta(row["test"]["routed_minus_direct"]),
                rate=row["test"]["reason_rate"],
            )
        )
    lines.append("")
    lines.extend(
        [
            "## Reading",
            "",
        ]
    )
    for row in payload["selected_guard_rows"]:
        if row["split"] == "test":
            lines.append(
                f"- `{row['policy']}` with `{row['guard']}` formal test A/E/T: "
                f"`{fmt_delta(row['summary']['routed_minus_direct'])}` at reason rate "
                f"`{row['summary']['reason_rate']:.1%}`."
            )
    return "\n".join(lines) + "\n"


def main():
    payload = {
        "selection": "guards selected on dev_seen only; formal is locked replay",
        "policies": POLICIES,
        "guard_grid_size": len(guard_grid()),
        "drift_stats": {},
        "selected_guards": {},
        "selected_guard_rows": [],
        "all_dev_guard_results": [],
        "formal_window_diagnostic": {},
        "robust_dev_window_selection": {},
    }
    for policy in POLICIES:
        rows_by_split = {split: build_rows(policy, split) for split in SPLITS}
        payload["drift_stats"][policy["name"]] = drift_stats(rows_by_split)
        payload["formal_window_diagnostic"][policy["name"]] = formal_window_diagnostic(
            policy["name"], rows_by_split
        )
        payload["robust_dev_window_selection"][policy["name"]] = robust_dev_window_selection(
            policy["name"], rows_by_split
        )
        dev_results = []
        for guard in guard_grid():
            summary = summarize_policy(rows_by_split["dev_seen"], guard)
            dev_results.append({"policy": policy["name"], "guard": guard["name"], "guard_config": guard, "summary": summary})
        payload["all_dev_guard_results"].extend(dev_results)
        base = next(row for row in dev_results if row["guard"] == "base_window")
        if "event" in policy["name"]:
            selected = max(dev_results, key=lambda row: event_score(row["summary"]))
        else:
            selected = max(dev_results, key=lambda row: balanced_score(row["summary"]))
        if balanced_score(selected["summary"])[0] <= balanced_score(base["summary"])[0] and "event" not in policy["name"]:
            selected = base
        payload["selected_guards"][policy["name"]] = selected
        for split in SPLITS:
            summary = summarize_policy(rows_by_split[split], selected["guard_config"])
            payload["selected_guard_rows"].append(
                {
                    "policy": policy["name"],
                    "guard": selected["guard"],
                    "split": split,
                    "summary": summary,
                }
            )
        seen = payload["selected_guard_rows"][-2]["summary"]
        unseen = payload["selected_guard_rows"][-1]["summary"]
        payload["selected_guard_rows"].append(
            {
                "policy": policy["name"],
                "guard": selected["guard"],
                "split": "test",
                "summary": aggregate_test(seen, unseen),
            }
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
