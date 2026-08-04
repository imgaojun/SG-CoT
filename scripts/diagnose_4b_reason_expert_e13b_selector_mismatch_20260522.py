import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.evaluate_4b_transfer_m06_policy_replay_20260521 import (  # noqa: E402
    M02_FORMAL_ROOT,
    M02_WINDOW,
    M05_FORMAL_ROOT,
    M05_WINDOW,
    selected_by_window,
)
from scripts.summarize_4b_reason_expert_e13b_20260521 import NEW_ROOT  # noqa: E402
from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    load_prediction_map,
    load_score_rows,
    row_metric,
    score,
    sorted_keys_by_delta,
)


OUT_JSON = REPO / "reports/artifacts/2026-05-22_stage2_4b_reason_expert_e13b_selector_mismatch_diagnosis.json"
OUT_MD = REPO / "reports/2026-05-22_stage2_4b_reason_expert_e13b_selector_mismatch_diagnosis.md"
SPLITS = ["test_seen", "test_unseen"]
METRICS = ["argument_f1", "event_f1", "trigger_f1", "score"]


def metric_delta(direct_row, reason_row):
    direct_m = row_metric(direct_row)
    reason_m = row_metric(reason_row)
    return {
        "argument_f1": reason_m["argument"]["f1"] - direct_m["argument"]["f1"],
        "event_f1": reason_m["event"]["f1"] - direct_m["event"]["f1"],
        "trigger_f1": reason_m["trigger"]["f1"] - direct_m["trigger"]["f1"],
        "score": score(reason_row) - score(direct_row),
    }


def mean_dict(rows):
    if not rows:
        return {key: 0.0 for key in METRICS}
    return {key: sum(row[key] for row in rows) / len(rows) for key in METRICS}


def pct(value):
    return f"{100 * value:.1f}%"


def signed(value):
    return f"{value:+.4f}"


def fmt(value):
    return f"{value:.4f}"


def key_meta(row):
    meta = row.get("meta", {})
    return {
        "wnd_id": meta.get("wnd_id"),
        "doc_id": meta.get("doc_id"),
        "source_part": meta.get("source_part"),
        "gold_event_types": meta.get("gold_event_types") or [],
        "candidate_types": meta.get("candidate_types") or [],
    }


def type_counter(rows):
    counter = Counter()
    for row in rows:
        for event_type in row.get("gold_event_types", []):
            counter[event_type] += 1
    return dict(counter.most_common(10))


def summarize_group(keys, direct_rows, reason_rows, ranks_by_source=None):
    cases = []
    for key in sorted(keys):
        delta = metric_delta(direct_rows[key], reason_rows[key])
        meta = key_meta(direct_rows[key])
        case = {
            **meta,
            **delta,
            "aet_all_nonnegative": (
                delta["argument_f1"] >= 0
                and delta["event_f1"] >= 0
                and delta["trigger_f1"] >= 0
            ),
        }
        if ranks_by_source:
            case["ranks"] = {
                name: ranks.get(key)
                for name, ranks in ranks_by_source.items()
            }
        cases.append(case)
    return {
        "count": len(cases),
        "mean_delta": mean_dict(cases),
        "aet_all_nonnegative_rate": (
            sum(case["aet_all_nonnegative"] for case in cases) / len(cases)
            if cases
            else 0.0
        ),
        "top_gold_event_types": type_counter(cases),
        "top_helpful": sorted(cases, key=lambda row: row["score"], reverse=True)[:10],
        "top_harmful": sorted(cases, key=lambda row: row["score"])[:10],
    }


def rank_maps(score_rows, keys):
    ranked = sorted_keys_by_delta(score_rows, keys)
    n = len(ranked)
    return {
        key: {
            "rank": rank,
            "pct": rank / n if n else 0.0,
            "delta_direct_minus_reason_route_nll": score_rows[key].get("delta_direct_minus_reason_route_nll"),
        }
        for rank, key in enumerate(ranked, start=1)
    }


def bucket_counts(keys, rank_map):
    buckets = Counter()
    for key in keys:
        info = rank_map.get(key)
        if not info:
            buckets["missing"] += 1
            continue
        start = int(info["pct"] * 10) * 10
        if start == 100:
            start = 90
        buckets[f"{start:02d}-{start + 10:02d}%"] += 1
    return dict(sorted(buckets.items()))


def load_split(split):
    direct_rows = load_prediction_map(NEW_ROOT / "forced_direct" / split / "predictions.jsonl")
    reason_rows = load_prediction_map(NEW_ROOT / "forced_reason" / split / "predictions.jsonl")
    m02_scores = load_score_rows(M02_FORMAL_ROOT / "checkpoint-50" / split / "scores.jsonl")
    m05_scores = load_score_rows(M05_FORMAL_ROOT / "checkpoint-100" / split / "scores.jsonl")
    keys = sorted(set(direct_rows) & set(reason_rows) & set(m02_scores) & set(m05_scores))
    m02_selected, m02_window = selected_by_window(m02_scores, keys, M02_WINDOW)
    m05_selected, m05_window = selected_by_window(m05_scores, keys, M05_WINDOW)
    m06_selected = m02_selected | m05_selected
    oracle_positive = {
        key for key in keys if score(reason_rows[key]) > score(direct_rows[key])
    }
    m02_rank = rank_maps(m02_scores, keys)
    m05_rank = rank_maps(m05_scores, keys)
    groups = {
        "oracle_positive": oracle_positive,
        "oracle_negative_or_tie": set(keys) - oracle_positive,
        "m06_selected": m06_selected,
        "m06_true_positive": m06_selected & oracle_positive,
        "m06_false_positive": m06_selected - oracle_positive,
        "m06_false_negative": oracle_positive - m06_selected,
        "m02_selected": m02_selected,
        "m05_selected": m05_selected,
    }
    group_summaries = {
        name: summarize_group(
            group_keys,
            direct_rows,
            reason_rows,
            {"m02": m02_rank, "m05": m05_rank},
        )
        for name, group_keys in groups.items()
    }
    return {
        "split": split,
        "num_examples": len(keys),
        "selector": {
            "m02_window": m02_window,
            "m05_window": m05_window,
            "m02_count": len(m02_selected),
            "m05_count": len(m05_selected),
            "m06_count": len(m06_selected),
            "m06_rate": len(m06_selected) / len(keys) if keys else 0.0,
        },
        "counts": {name: len(value) for name, value in groups.items()},
        "rates": {name: len(value) / len(keys) if keys else 0.0 for name, value in groups.items()},
        "m06_precision_vs_oracle": (
            len(m06_selected & oracle_positive) / len(m06_selected) if m06_selected else 0.0
        ),
        "m06_recall_vs_oracle": (
            len(m06_selected & oracle_positive) / len(oracle_positive) if oracle_positive else 0.0
        ),
        "m06_f1_vs_oracle": f1(m06_selected, oracle_positive),
        "rank_buckets": {
            "oracle_positive_by_m02_rank": bucket_counts(oracle_positive, m02_rank),
            "oracle_positive_by_m05_rank": bucket_counts(oracle_positive, m05_rank),
            "m06_false_positive_by_m02_rank": bucket_counts(m06_selected - oracle_positive, m02_rank),
            "m06_false_negative_by_m02_rank": bucket_counts(oracle_positive - m06_selected, m02_rank),
        },
        "groups": group_summaries,
    }


def f1(pred, gold):
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def aggregate(splits):
    total = sum(row["num_examples"] for row in splits)
    counts = defaultdict(int)
    weighted_mean_groups = {}
    for row in splits:
        for key, value in row["counts"].items():
            counts[key] += value
    for group_name in splits[0]["groups"]:
        denom = counts[group_name]
        weighted_mean_groups[group_name] = {
            metric: (
                sum(
                    row["groups"][group_name]["mean_delta"][metric] * row["counts"][group_name]
                    for row in splits
                )
                / denom
                if denom
                else 0.0
            )
            for metric in METRICS
        }
    tp = counts["m06_true_positive"]
    pred = counts["m06_selected"]
    gold = counts["oracle_positive"]
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    return {
        "split": "test",
        "num_examples": total,
        "counts": dict(counts),
        "rates": {key: value / total if total else 0.0 for key, value in counts.items()},
        "m06_precision_vs_oracle": precision,
        "m06_recall_vs_oracle": recall,
        "m06_f1_vs_oracle": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "group_mean_delta": weighted_mean_groups,
    }


def render_group_line(name, group, total):
    mean = group["mean_delta"]
    return (
        f"| `{name}` | {group['count']} | {pct(group['count'] / total if total else 0.0)} | "
        f"{signed(mean['argument_f1'])} / {signed(mean['event_f1'])} / {signed(mean['trigger_f1'])} | "
        f"{signed(mean['score'])} | {pct(group['aet_all_nonnegative_rate'])} |"
    )


def render_split(row):
    lines = [
        f"### {row['split']}",
        "",
        f"- examples: `{row['num_examples']}`",
        f"- M06 reason rate: `{pct(row['selector']['m06_rate'])}`",
        f"- M06 precision/recall/F1 vs E13B oracle-positive: "
        f"`{row['m06_precision_vs_oracle']:.3f} / {row['m06_recall_vs_oracle']:.3f} / {row['m06_f1_vs_oracle']:.3f}`",
        "",
        "| group | count | rate | mean delta A/E/T | mean score delta | A/E/T-safe |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in [
        "oracle_positive",
        "m06_selected",
        "m06_true_positive",
        "m06_false_positive",
        "m06_false_negative",
    ]:
        lines.append(render_group_line(name, row["groups"][name], row["num_examples"]))
    lines.extend(["", "Oracle-positive rank buckets under M02 score:", ""])
    buckets = row["rank_buckets"]["oracle_positive_by_m02_rank"]
    lines.append(", ".join(f"`{key}`={value}" for key, value in buckets.items()))
    lines.append("")
    return "\n".join(lines)


def render(payload):
    agg = payload["aggregate"]
    lines = [
        "# E13B Selector Mismatch Diagnosis",
        "",
        "This report compares transferred M06 selections against E13B direct/reason oracle-positive cases.",
        "",
        "## Aggregate",
        "",
        f"- examples: `{agg['num_examples']}`",
        f"- E13B oracle-positive rate: `{pct(agg['rates']['oracle_positive'])}`",
        f"- M06 selected rate: `{pct(agg['rates']['m06_selected'])}`",
        f"- M06 precision/recall/F1 vs E13B oracle-positive: "
        f"`{agg['m06_precision_vs_oracle']:.3f} / {agg['m06_recall_vs_oracle']:.3f} / {agg['m06_f1_vs_oracle']:.3f}`",
        "",
        "| group | count | rate | mean delta A/E/T | mean score delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in [
        "oracle_positive",
        "m06_selected",
        "m06_true_positive",
        "m06_false_positive",
        "m06_false_negative",
    ]:
        mean = agg["group_mean_delta"][name]
        count = agg["counts"][name]
        lines.append(
            f"| `{name}` | {count} | {pct(count / agg['num_examples'])} | "
            f"{signed(mean['argument_f1'])} / {signed(mean['event_f1'])} / {signed(mean['trigger_f1'])} | "
            f"{signed(mean['score'])} |"
        )
    lines.extend(["", "## By Split", ""])
    for row in payload["splits"]:
        lines.append(render_split(row))
    lines.extend(
        [
            "## Reading",
            "",
            "- The useful E13B reason cases exist, but the transferred M06 windows have poor overlap with them.",
            "- False positives are directly harmful under the E13B expert, so reusing M06 as a hard selector is not appropriate.",
            "- The next selector should be E13B-specific: either calibrate E13B route scores, or train a retention selector from E13B direct/reason outcomes rather than from 1.7B transfer decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    splits = [load_split(split) for split in SPLITS]
    payload = {
        "new_root": NEW_ROOT.as_posix(),
        "m02_formal_root": M02_FORMAL_ROOT.as_posix(),
        "m05_formal_root": M05_FORMAL_ROOT.as_posix(),
        "splits": splits,
        "aggregate": aggregate(splits),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
