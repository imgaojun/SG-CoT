#!/usr/bin/env python3
import json
import sys
from hashlib import md5
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.evaluate_modular_dualexpert_aet_m06_combo_selectors_20260521 import (  # noqa: E402
    M02_DEV_SCORE,
    M02_FORMAL_ROOT,
    M02_WINDOW,
    M05_DEV_SCORE,
    M05_FORMAL_ROOT,
    M05_WINDOW,
    selected_by_window,
)
from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    DIRECT_ROOT,
    REASON_ROOT as FULL_ROOT,
    load_prediction_map,
    load_score_rows,
    row_metric,
    score,
)
from scripts.calibrate_modular_dualexpert_utility_router_m02_rank_window_dev_20260520 import DIRECT_DEV  # noqa: E402
from scripts.calibrate_multibudget_fourclass_router_m09_dev_20260521 import (  # noqa: E402
    LIGHT_DEV,
    MID_DEV,
    FULL_DEV,
)


M09_BRANCH = "multibudget_fourclass_router_m09_routecls_noauxwarm_lr2e6_save50"
M09_DEV_SCORE_ROOT = REPO / "outputs/stage2_multibudget/route_likelihood_20260521" / M09_BRANCH
M09_FORMAL_SCORE_ROOT = REPO / "outputs/stage2_multibudget/formal_route_likelihood_20260521" / M09_BRANCH
LIGHT_ROOT = REPO / "outputs/stage2_adaptive_runs_user_formal_clean/richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_plan_lite/frontier_reason_expert_best/forced_reason"
MID_ROOT = REPO / "outputs/stage2_adaptive_runs_user_formal_clean/richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30/frontier_seen_stable_best/forced_reason"

OUT_DEV_JSON = REPO / "reports/artifacts/2026-05-21_stage2_multibudget_m10_m06_union_plus_m09_addon_dev.json"
OUT_DEV_MD = REPO / "reports/2026-05-21_stage2_multibudget_m10_m06_union_plus_m09_addon_dev.md"
OUT_FORMAL_JSON = REPO / "reports/artifacts/2026-05-21_stage2_multibudget_m10_m06_union_plus_m09_addon_formal.json"
OUT_FORMAL_MD = REPO / "reports/2026-05-21_stage2_multibudget_m10_m06_union_plus_m09_addon_formal.md"

ROUTES = ["reason_light", "reason_mid", "reason_full"]
WINDOW_ENDPOINTS = [i / 40 for i in range(0, 21)]
ADDON_MAX_RATE = 0.05
TOTAL_MAX_RATE = 0.17
FORMAL_SPLITS = ["test_seen", "test_unseen"]
POLICY_LIMIT = 2


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_fold(key):
    return int(md5(key.encode("utf-8")).hexdigest()[:8], 16) % 5


def metric_delta(direct_row, chosen_row):
    direct_m = row_metric(direct_row)
    chosen_m = row_metric(chosen_row)
    return {
        "trigger_f1": chosen_m["trigger"]["f1"] - direct_m["trigger"]["f1"],
        "argument_f1": chosen_m["argument"]["f1"] - direct_m["argument"]["f1"],
        "event_f1": chosen_m["event"]["f1"] - direct_m["event"]["f1"],
        "score": score(chosen_row) - score(direct_row),
    }


def mean_dict(rows):
    keys = ["trigger_f1", "argument_f1", "event_f1", "score"]
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def summarize_delta(routed, direct):
    return {metric: routed[metric] - direct[metric] for metric in ["trigger_f1", "argument_f1", "event_f1"]}


def choose_row(route, key, direct_rows, light_rows, mid_rows, full_rows):
    if route == "reason_light":
        return light_rows[key]
    if route == "reason_mid":
        return mid_rows[key]
    if route == "reason_full":
        return full_rows[key]
    return direct_rows[key]


def best_route_for_score_row(row):
    route = row.get("best_non_direct_route")
    if route in ROUTES:
        return route
    nll = row.get("nll_by_route") or {}
    available = [r for r in ROUTES if r in nll]
    if not available:
        return "direct"
    return min(available, key=lambda r: nll[r])


def ordered_m09_items(score_rows, keys):
    items = []
    for key in set(keys) & set(score_rows):
        row = score_rows[key]
        route = best_route_for_score_row(row)
        nll = row.get("nll_by_route") or {}
        adv = row.get("best_non_direct_advantage_vs_direct")
        if adv is None:
            adv = nll.get("direct", float("-inf")) - nll.get(route, float("inf"))
        items.append((float(adv), key, route))
    items.sort(reverse=True)
    return items


def load_m06_selected(keys, m02_score_path, m05_score_path):
    m02_scores = load_score_rows(Path(m02_score_path))
    m05_scores = load_score_rows(Path(m05_score_path))
    m02_selected, m02_window = selected_by_window(m02_scores, keys, M02_WINDOW)
    m05_selected, m05_window = selected_by_window(m05_scores, keys, M05_WINDOW)
    return {
        "m02_selected": m02_selected,
        "m05_selected": m05_selected,
        "union_selected": m02_selected | m05_selected,
        "component_windows": {
            "m02": m02_window,
            "m05": m05_window,
        },
    }


def evaluate_policy(name, split, keys, base_selected, addon_selected, direct_rows, light_rows, mid_rows, full_rows):
    routed_metrics = []
    direct_metrics = []
    selected_deltas = []
    addon_deltas = []
    route_counts = {"direct": 0, "reason_light": 0, "reason_mid": 0, "reason_full": 0, "reason_full_base": 0}
    selected_examples = []
    addon_map = {key: route for key, route in addon_selected.items()}
    selected_keys = set(base_selected) | set(addon_map)
    for key in keys:
        direct_row = direct_rows[key]
        if key in addon_map:
            route = addon_map[key]
            chosen = choose_row(route, key, direct_rows, light_rows, mid_rows, full_rows)
        elif key in base_selected:
            route = "reason_full_base"
            chosen = full_rows[key]
        else:
            route = "direct"
            chosen = direct_row
        route_counts[route] += 1
        routed_metrics.append(row_metric(chosen))
        direct_metrics.append(row_metric(direct_row))
        if key in selected_keys:
            delta = metric_delta(direct_row, chosen)
            selected_deltas.append(delta)
            if key in addon_map:
                addon_deltas.append(delta)
                selected_examples.append({"wnd_id": key, "route": route, **delta})
    direct = summarize_metrics(direct_metrics)
    routed = summarize_metrics(routed_metrics)
    reason_count = len(selected_keys)
    return {
        "policy": name,
        "split": split,
        "num_examples": len(keys),
        "base_count": len(base_selected),
        "addon_count": len(addon_map),
        "reason_count": reason_count,
        "reason_rate": reason_count / len(keys) if keys else 0.0,
        "route_counts": route_counts,
        "direct": direct,
        "routed": routed,
        "routed_minus_direct": summarize_delta(routed, direct),
        "selected_delta_mean": mean_dict(selected_deltas),
        "addon_delta_mean": mean_dict(addon_deltas),
        "addon_examples": sorted(selected_examples, key=lambda row: row["score"], reverse=True)[:20],
    }


def summarize_metrics(metric_rows):
    from src.stage2_analysis.analyze_adaptive_outcome_router_execution import summarize_metrics as _summarize

    return _summarize(metric_rows)


def add_fold_floor(row, keys, base_selected, addon_selected, direct_rows, light_rows, mid_rows, full_rows):
    folds = []
    for fold in range(5):
        fold_keys = [key for key in keys if stable_fold(key) == fold]
        fold_base = {key for key in base_selected if key in set(fold_keys)}
        fold_addon = {key: route for key, route in addon_selected.items() if key in set(fold_keys)}
        fold_row = evaluate_policy(
            f"{row['policy']}_fold{fold}",
            row["split"],
            fold_keys,
            fold_base,
            fold_addon,
            direct_rows,
            light_rows,
            mid_rows,
            full_rows,
        )
        d = fold_row["routed_minus_direct"]
        folds.append({"fold": fold, "reason_rate": fold_row["reason_rate"], "delta": d, "min_aet": min(d.values())})
    row["folds"] = folds
    row["fold_min_aet"] = min((fold["min_aet"] for fold in folds), default=-99.0)
    return row


def is_all_nonnegative(row):
    d = row["routed_minus_direct"]
    return d["argument_f1"] >= 0 and d["event_f1"] >= 0 and d["trigger_f1"] >= 0


def balanced_score(row):
    d = row["routed_minus_direct"]
    addon = row["addon_delta_mean"]
    return (
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        row["fold_min_aet"],
        d["event_f1"],
        addon["event_f1"],
        -row["reason_rate"],
    )


def event_score(row):
    d = row["routed_minus_direct"]
    addon = row["addon_delta_mean"]
    return (
        d["event_f1"],
        d["argument_f1"],
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        row["fold_min_aet"],
        addon["event_f1"],
        -row["reason_rate"],
    )


def dev_search():
    direct_rows = load_prediction_map(Path(DIRECT_DEV))
    light_rows = load_prediction_map(LIGHT_DEV)
    mid_rows = load_prediction_map(MID_DEV)
    full_rows = load_prediction_map(FULL_DEV)
    keys = sorted(set(direct_rows) & set(light_rows) & set(mid_rows) & set(full_rows))
    m06 = load_m06_selected(keys, M02_DEV_SCORE, M05_DEV_SCORE)
    base_selected = m06["union_selected"]
    rows = []
    for score_path in sorted(M09_DEV_SCORE_ROOT.glob("checkpoint-*/dev_seen_scores.jsonl")):
        ckpt = score_path.parent.name
        score_rows = load_score_rows(score_path)
        ordered = [item for item in ordered_m09_items(score_rows, keys) if item[1] not in base_selected]
        n = len(keys)
        for lo in WINDOW_ENDPOINTS:
            for hi in WINDOW_ENDPOINTS:
                if hi <= lo:
                    continue
                start = round(n * lo)
                end = round(n * hi)
                selected_items = ordered[start:end]
                if not selected_items:
                    continue
                addon = {key: route for _, key, route in selected_items if key not in base_selected}
                if not addon:
                    continue
                addon_rate = len(addon) / n
                total_rate = (len(base_selected) + len(addon)) / n
                if addon_rate > ADDON_MAX_RATE or total_rate > TOTAL_MAX_RATE:
                    continue
                row = evaluate_policy(
                    f"{ckpt}_addon_rank{int(lo * 1000):03d}_{int(hi * 1000):03d}",
                    "dev_seen",
                    keys,
                    base_selected,
                    addon,
                    direct_rows,
                    light_rows,
                    mid_rows,
                    full_rows,
                )
                row["checkpoint"] = ckpt
                row["rank_window"] = {"start_pct": lo, "end_pct": hi, "start_rank": start + 1, "end_rank": end}
                row["addon_routes"] = addon
                row = add_fold_floor(row, keys, base_selected, addon, direct_rows, light_rows, mid_rows, full_rows)
                rows.append(row)
    all_nonnegative = [row for row in rows if is_all_nonnegative(row) and row["fold_min_aet"] >= 0]
    if not all_nonnegative:
        all_nonnegative = [row for row in rows if is_all_nonnegative(row)]
    balanced = max(all_nonnegative, key=balanced_score) if all_nonnegative else None
    event = max(all_nonnegative, key=event_score) if all_nonnegative else None
    selected = []
    for row in [balanced, event]:
        if row and row["policy"] not in {x["policy"] for x in selected}:
            selected.append(row)
    return {
        "split": "dev_seen",
        "num_examples": len(keys),
        "base_policy": "m06_union_m02_m05",
        "base_reason_count": len(base_selected),
        "base_reason_rate": len(base_selected) / len(keys) if keys else 0.0,
        "m06_components": {
            "m02_count": len(m06["m02_selected"]),
            "m05_count": len(m06["m05_selected"]),
            "union_count": len(base_selected),
            "component_windows": m06["component_windows"],
        },
        "num_candidates": len(rows),
        "num_all_nonnegative": len(all_nonnegative),
        "balanced_candidate": strip_addon_routes(balanced),
        "event_candidate": strip_addon_routes(event),
        "selected_policies": [strip_addon_routes(row) for row in selected],
        "top_balanced": [strip_addon_routes(row) for row in sorted(all_nonnegative, key=balanced_score, reverse=True)[:30]],
        "top_event": [strip_addon_routes(row) for row in sorted(all_nonnegative, key=event_score, reverse=True)[:30]],
        "_runtime": {
            "keys": keys,
            "base_selected": base_selected,
            "selected_full": selected,
            "direct_rows": direct_rows,
            "light_rows": light_rows,
            "mid_rows": mid_rows,
            "full_rows": full_rows,
        },
    }


def strip_addon_routes(row):
    if row is None:
        return None
    out = {k: v for k, v in row.items() if k != "addon_routes"}
    return out


def aggregate_test(rows):
    by_policy = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row)
    out = []
    for policy, parts in by_policy.items():
        total = sum(row["num_examples"] for row in parts)
        reason_count = sum(row["reason_count"] for row in parts)
        addon_count = sum(row["addon_count"] for row in parts)
        agg = {
            "policy": policy,
            "split": "test",
            "num_examples": total,
            "base_count": sum(row["base_count"] for row in parts),
            "addon_count": addon_count,
            "reason_count": reason_count,
            "reason_rate": reason_count / total if total else 0.0,
            "route_counts": {
                route: sum(row["route_counts"].get(route, 0) for row in parts)
                for route in ["direct", "reason_light", "reason_mid", "reason_full", "reason_full_base"]
            },
        }
        for group in ["direct", "routed"]:
            agg[group] = {
                metric: sum(row[group][metric] * row["num_examples"] for row in parts) / total if total else 0.0
                for metric in ["trigger_f1", "argument_f1", "event_f1"]
            }
        agg["routed_minus_direct"] = summarize_delta(agg["routed"], agg["direct"])
        denom = reason_count
        addon_denom = addon_count
        agg["selected_delta_mean"] = {
            metric: (
                sum(row["selected_delta_mean"][metric] * row["reason_count"] for row in parts) / denom
                if denom
                else 0.0
            )
            for metric in ["trigger_f1", "argument_f1", "event_f1", "score"]
        }
        agg["addon_delta_mean"] = {
            metric: (
                sum(row["addon_delta_mean"][metric] * row["addon_count"] for row in parts) / addon_denom
                if addon_denom
                else 0.0
            )
            for metric in ["trigger_f1", "argument_f1", "event_f1", "score"]
        }
        out.append(agg)
    return out


def formal_replay(dev_payload):
    selected = dev_payload["_runtime"]["selected_full"][:POLICY_LIMIT]
    rows = []
    for split in FORMAL_SPLITS:
        direct_rows = load_prediction_map(DIRECT_ROOT / split / "predictions.jsonl")
        light_rows = load_prediction_map(LIGHT_ROOT / split / "predictions.jsonl")
        mid_rows = load_prediction_map(MID_ROOT / split / "predictions.jsonl")
        full_rows = load_prediction_map(FULL_ROOT / split / "predictions.jsonl")
        keys = sorted(set(direct_rows) & set(light_rows) & set(mid_rows) & set(full_rows))
        m06 = load_m06_selected(
            keys,
            M02_FORMAL_ROOT / "checkpoint-50" / split / "scores.jsonl",
            M05_FORMAL_ROOT / "checkpoint-100" / split / "scores.jsonl",
        )
        base_selected = m06["union_selected"]
        for policy in selected:
            score_rows = load_score_rows(M09_FORMAL_SCORE_ROOT / policy["checkpoint"] / split / "scores.jsonl")
            ordered = [item for item in ordered_m09_items(score_rows, keys) if item[1] not in base_selected]
            start = round(len(keys) * policy["rank_window"]["start_pct"])
            end = round(len(keys) * policy["rank_window"]["end_pct"])
            addon = {key: route for _, key, route in ordered[start:end] if key not in base_selected}
            row = evaluate_policy(
                policy["policy"],
                split,
                keys,
                base_selected,
                addon,
                direct_rows,
                light_rows,
                mid_rows,
                full_rows,
            )
            row["checkpoint"] = policy["checkpoint"]
            row["rank_window"] = policy["rank_window"]
            rows.append(row)
    return {"results": rows + aggregate_test(rows), "selected_dev_policies": [strip_addon_routes(row) for row in selected]}


def fmt_delta(delta):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**delta)


def render_table(rows):
    lines = [
        "| policy | split | reason rate | add-on | route counts | delta A/E/T | add-on mean A/E/T | routed A/E/T |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        counts = row["route_counts"]
        routed = row["routed"]
        addon = row["addon_delta_mean"]
        lines.append(
            "| {policy} | {split} | {rate:.1%} | {addon_count} | d/l/m/f/base={d}/{l}/{m}/{f}/{base} | {delta} | {aa:+.4f} / {ae:+.4f} / {at:+.4f} | {ra:.4f} / {re:.4f} / {rt:.4f} |".format(
                policy=row["policy"],
                split=row["split"],
                rate=row["reason_rate"],
                addon_count=row["addon_count"],
                d=counts.get("direct", 0),
                l=counts.get("reason_light", 0),
                m=counts.get("reason_mid", 0),
                f=counts.get("reason_full", 0),
                base=counts.get("reason_full_base", 0),
                delta=fmt_delta(row["routed_minus_direct"]),
                aa=addon["argument_f1"],
                ae=addon["event_f1"],
                at=addon["trigger_f1"],
                ra=routed["argument_f1"],
                re=routed["event_f1"],
                rt=routed["trigger_f1"],
            )
        )
    return "\n".join(lines)


def render_dev(payload):
    rows = [row for row in payload["selected_policies"] if row]
    lines = [
        "# M10 M06 Union Plus M09 Add-On Dev Search",
        "",
        f"- base reason rate: `{payload['base_reason_rate']:.1%}`.",
        f"- candidates swept: `{payload['num_candidates']}`; all-nonnegative retained: `{payload['num_all_nonnegative']}`.",
        "",
        "## Locked Candidates",
        "",
        render_table(rows),
        "",
        "## Top Balanced",
        "",
        render_table(payload["top_balanced"][:20]),
        "",
    ]
    return "\n".join(lines)


def render_formal(payload):
    test_rows = [row for row in payload["results"] if row["split"] == "test"]
    lines = [
        "# M10 M06 Union Plus M09 Add-On Formal Replay",
        "",
        "Formal replay uses dev-locked M10 policies. Formal labels were not used for selection.",
        "",
        "## Test Summary",
        "",
        render_table(test_rows),
        "",
        "## Split Results",
        "",
        render_table(payload["results"]),
        "",
    ]
    return "\n".join(lines)


def main():
    dev_payload = dev_search()
    runtime = dev_payload.pop("_runtime")
    write_json(OUT_DEV_JSON, dev_payload)
    write_text(OUT_DEV_MD, render_dev(dev_payload))

    dev_payload["_runtime"] = runtime
    formal_payload = formal_replay(dev_payload)
    write_json(OUT_FORMAL_JSON, formal_payload)
    write_text(OUT_FORMAL_MD, render_formal(formal_payload))
    print(json.dumps({"dev_json": OUT_DEV_JSON.as_posix(), "formal_json": OUT_FORMAL_JSON.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
