import argparse
from collections import Counter, defaultdict
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import (  # noqa: E402
    build_feature_map,
    prediction_key,
    score,
)
from src.stage2_analysis.analyze_adaptive_outcome_router_execution import write_json, write_text  # noqa: E402
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


BRANCH = "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux"
CKPT = "checkpoint-2058"
SPLIT = "test_unseen"
DIRECT_JSONL = Path(
    f"outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/"
    f"richere_split1_qwen3_1_7b_adaptive_{BRANCH}/{CKPT}/forced_direct/{SPLIT}/predictions.jsonl"
)
REASON_JSONL = Path(
    f"outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/"
    f"richere_split1_qwen3_1_7b_adaptive_{BRANCH}/{CKPT}/forced_reason/{SPLIT}/predictions.jsonl"
)
SCORES_JSONL = Path(
    f"outputs/stage2_adaptive_route_likelihood_probe/outcome_helpful_sharedbase_formal_20260515/"
    f"{BRANCH}/{CKPT}/{SPLIT}/scores.jsonl"
)
EVAL_JSONL = Path("data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_test_unseen_pos.jsonl")
SCHEMA = Path("data/schema/richere-en.event_schema.json")


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def compact_events(payload):
    events = payload.get("events", []) if isinstance(payload, dict) else []
    out = []
    if not isinstance(events, list):
        return out
    for event in events:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        raw_args = event.get("arguments") if isinstance(event.get("arguments"), list) else []
        for arg in raw_args:
            if not isinstance(arg, dict):
                continue
            args.append(
                {
                    "role": arg.get("role"),
                    "text": arg.get("text"),
                    "span": [arg.get("start"), arg.get("end")],
                }
            )
        out.append(
            {
                "event_type": event.get("event_type"),
                "trigger": trigger.get("text"),
                "trigger_span": [trigger.get("start"), trigger.get("end")],
                "arguments": args,
            }
        )
    return out


def input_text(row):
    text = row.get("input") or ""
    if "Tokens:" in text:
        text = text.split("Tokens:", 1)[0]
    return " ".join(text.replace("Text:", "", 1).split())


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def std(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def feature_means(rows, keys):
    return {key: mean([(row.get("features") or {}).get(key, 0.0) for row in rows]) for key in keys}


def build_rows():
    direct = load_prediction_map(DIRECT_JSONL)
    reason = load_prediction_map(REASON_JSONL)
    scores = load_prediction_map(SCORES_JSONL)
    features = build_feature_map(EVAL_JSONL, SCHEMA)
    rows = []
    for key in sorted(set(direct) & set(reason) & set(scores)):
        drow = direct[key]
        rrow = reason[key]
        srow = scores[key]
        delta = srow.get("delta_direct_minus_reason_route_nll")
        delta = float(delta) if delta is not None else float("-inf")
        meta = drow.get("meta") or {}
        gain = score(rrow) - score(drow)
        rows.append(
            {
                "key": key,
                "delta": delta,
                "nll_direct_route": srow.get("nll_direct_route"),
                "nll_reason_route": srow.get("nll_reason_route"),
                "pred_route_argmin_nll": srow.get("pred_route_argmin_nll"),
                "reason_gain": gain,
                "helpful": gain > 1e-9,
                "harmful": gain < -1e-9,
                "direct_trigger_f1": drow.get("trigger_f1", 0.0),
                "direct_argument_f1": drow.get("argument_f1", 0.0),
                "direct_event_f1": drow.get("event_f1", 0.0),
                "reason_trigger_f1": rrow.get("trigger_f1", 0.0),
                "reason_argument_f1": rrow.get("argument_f1", 0.0),
                "reason_event_f1": rrow.get("event_f1", 0.0),
                "argument_gain": rrow.get("argument_f1", 0.0) - drow.get("argument_f1", 0.0),
                "event_gain": rrow.get("event_f1", 0.0) - drow.get("event_f1", 0.0),
                "trigger_gain": rrow.get("trigger_f1", 0.0) - drow.get("trigger_f1", 0.0),
                "meta": {
                    "doc_id": meta.get("doc_id"),
                    "wnd_id": meta.get("wnd_id"),
                    "gold_event_types": meta.get("gold_event_types") or [],
                    "candidate_types": meta.get("candidate_types") or [],
                    "candidate_count": len(meta.get("candidate_types") or []),
                    "source_part": meta.get("source_part"),
                },
                "features": features.get(key, {}),
                "text": input_text(drow)[:400],
                "gold_events": compact_events(drow.get("gold") or {}),
                "direct_events": compact_events(drow.get("predicted") or drow.get("final_predicted") or {}),
                "reason_events": compact_events(rrow.get("predicted") or rrow.get("final_predicted") or {}),
            }
        )
    ranked = sorted(rows, key=lambda row: (row["delta"], row["key"]), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
        row["rank_pct"] = idx / len(ranked) if ranked else 0.0
    by_key = {row["key"]: row for row in rows}
    return [by_key[row["key"]] for row in ranked]


def topk_summary(rows, budgets):
    helpful_count = sum(1 for row in rows if row["helpful"])
    out = {}
    for budget in budgets:
        cap = round(len(rows) * budget)
        selected = rows[:cap]
        tp = sum(1 for row in selected if row["helpful"])
        harmful = sum(1 for row in selected if row["harmful"])
        zero = cap - tp - harmful
        out[f"top{int(budget * 100):02d}"] = {
            "cap": cap,
            "tp_helpful": tp,
            "fp_non_helpful": cap - tp,
            "selected_harmful": harmful,
            "selected_zero_gain": zero,
            "precision": tp / cap if cap else 0.0,
            "recall": tp / helpful_count if helpful_count else 0.0,
            "avg_selected_delta": mean([row["delta"] for row in selected]),
            "avg_selected_reason_gain": mean([row["reason_gain"] for row in selected]),
        }
    return out


def bucket_by_event_type(rows):
    buckets = defaultdict(list)
    for row in rows:
        types = row["meta"]["gold_event_types"] or ["NO_GOLD_TYPE"]
        for event_type in types:
            buckets[event_type].append(row)
    out = []
    for event_type, items in buckets.items():
        helpful = [row for row in items if row["helpful"]]
        out.append(
            {
                "event_type": event_type,
                "count": len(items),
                "helpful_count": len(helpful),
                "helpful_rate": len(helpful) / len(items) if items else 0.0,
                "avg_delta": mean([row["delta"] for row in items]),
                "avg_reason_gain": mean([row["reason_gain"] for row in items]),
                "best_helpful_rank": min([row["rank"] for row in helpful], default=None),
                "median_helpful_rank": sorted([row["rank"] for row in helpful])[len(helpful) // 2] if helpful else None,
            }
        )
    return sorted(out, key=lambda row: (row["helpful_count"], row["count"], row["event_type"]), reverse=True)


def bucket_by_doc(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["meta"]["doc_id"] or "unknown"].append(row)
    out = []
    for doc_id, items in buckets.items():
        helpful = [row for row in items if row["helpful"]]
        out.append(
            {
                "doc_id": doc_id,
                "count": len(items),
                "helpful_count": len(helpful),
                "avg_delta": mean([row["delta"] for row in items]),
                "avg_reason_gain": mean([row["reason_gain"] for row in items]),
                "best_helpful_rank": min([row["rank"] for row in helpful], default=None),
            }
        )
    return sorted(out, key=lambda row: (row["helpful_count"], row["count"]), reverse=True)


def case_row(row):
    return {
        "rank": row["rank"],
        "key": row["key"],
        "delta": row["delta"],
        "reason_gain": row["reason_gain"],
        "argument_gain": row["argument_gain"],
        "event_gain": row["event_gain"],
        "trigger_gain": row["trigger_gain"],
        "direct_f1": {
            "trigger": row["direct_trigger_f1"],
            "argument": row["direct_argument_f1"],
            "event": row["direct_event_f1"],
        },
        "reason_f1": {
            "trigger": row["reason_trigger_f1"],
            "argument": row["reason_argument_f1"],
            "event": row["reason_event_f1"],
        },
        "meta": row["meta"],
        "features": row["features"],
        "text": row["text"],
        "gold_events": row["gold_events"],
        "direct_events": row["direct_events"],
        "reason_events": row["reason_events"],
    }


def render_markdown(payload):
    s = payload["summary"]
    lines = [
        "# Outcome-Helpful Shared-Base Unseen Ranking Diagnosis",
        "",
        "## Summary",
        "",
        f"- split: `{payload['split']}`",
        f"- examples: `{s['num_examples']}`",
        f"- helpful examples: `{s['helpful_count']}` (`{s['helpful_rate']:.1%}`)",
        f"- helpful rank min/median/max: `{s['helpful_rank_min']} / {s['helpful_rank_median']} / {s['helpful_rank_max']}`",
        f"- delta mean helpful/non-helpful: `{s['delta_mean_helpful']:.4f} / {s['delta_mean_non_helpful']:.4f}`",
        "",
        "## Top-K Hit Rates",
        "",
        "| budget | selected | helpful TP | harmful selected | zero-gain selected | precision | recall | avg selected gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in payload["topk"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.4f} |".format(
                label,
                row["cap"],
                row["tp_helpful"],
                row["selected_harmful"],
                row["selected_zero_gain"],
                row["precision"],
                row["recall"],
                row["avg_selected_reason_gain"],
            )
        )
    lines.extend(["", "## Feature Means", ""])
    lines.append("| group | hardconf | confusion | role rarity | role density | multi-event | core absence |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for group, vals in payload["feature_means"].items():
        lines.append(
            "| `{}` | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                group,
                vals.get("hardconf_score", 0.0),
                vals.get("confusion_norm", 0.0),
                vals.get("role_signature_rarity", 0.0),
                vals.get("role_density_norm", 0.0),
                vals.get("multi_event_or_multi_trigger", 0.0),
                vals.get("core_role_absence_risk", 0.0),
            )
        )
    lines.extend(["", "## Event-Type Buckets", ""])
    lines.append("| event type | count | helpful | helpful rate | avg delta | avg gain | best helpful rank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["event_type_buckets"][:12]:
        lines.append(
            "| `{}` | {} | {} | {:.1%} | {:.4f} | {:.4f} | {} |".format(
                row["event_type"],
                row["count"],
                row["helpful_count"],
                row["helpful_rate"],
                row["avg_delta"],
                row["avg_reason_gain"],
                row["best_helpful_rank"] if row["best_helpful_rank"] is not None else "-",
            )
        )
    lines.extend(["", "## Failure Reading", ""])
    lines.append(
        "- Top-ranked unseen selections are dominated by non-helpful examples: many are zero-gain cases where both direct and reason fail or tie."
    )
    lines.append(
        "- Helpful unseen examples do exist, but most have negative route-NLL deltas, so a pure route-token likelihood ranker places them too low."
    )
    lines.append(
        "- Hardness features do not cleanly separate helpful from non-helpful unseen samples; a useful reranker likely needs execution/confidence features, not only static schema hardness."
    )
    lines.extend(["", "## Case Pointers", ""])
    lines.append("- `artifact.top15_false_positives`: high-ranked non-helpful selections.")
    lines.append("- `artifact.missed_helpful`: helpful examples ranked below top20.")
    lines.append("- `artifact.high_gain_missed`: largest reason gains missed by top20.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_json",
        default="reports/artifacts/2026-05-15_stage2_adaptive_outcome_helpful_sharedbase_unseen_ranking_diagnosis.json",
    )
    parser.add_argument(
        "--output_md",
        default="reports/2026-05-15_stage2_adaptive_outcome_helpful_sharedbase_unseen_ranking_diagnosis.md",
    )
    args = parser.parse_args()

    rows = build_rows()
    helpful = [row for row in rows if row["helpful"]]
    non_helpful = [row for row in rows if not row["helpful"]]
    top15 = rows[: round(len(rows) * 0.15)]
    top20_keys = {row["key"] for row in rows[: round(len(rows) * 0.20)]}
    missed_helpful = [row for row in helpful if row["key"] not in top20_keys]
    high_gain_missed = sorted(missed_helpful, key=lambda row: row["reason_gain"], reverse=True)
    feature_keys = [
        "hardconf_score",
        "confusion_norm",
        "role_signature_rarity",
        "role_density_norm",
        "multi_event_or_multi_trigger",
        "core_role_absence_risk",
    ]
    helpful_ranks = sorted(row["rank"] for row in helpful)
    payload = {
        "branch": BRANCH,
        "checkpoint": CKPT,
        "split": SPLIT,
        "inputs": {
            "direct": DIRECT_JSONL.as_posix(),
            "reason": REASON_JSONL.as_posix(),
            "scores": SCORES_JSONL.as_posix(),
        },
        "summary": {
            "num_examples": len(rows),
            "helpful_count": len(helpful),
            "helpful_rate": len(helpful) / len(rows) if rows else 0.0,
            "harmful_count": sum(1 for row in rows if row["harmful"]),
            "zero_gain_count": sum(1 for row in rows if not row["helpful"] and not row["harmful"]),
            "helpful_rank_min": min(helpful_ranks) if helpful_ranks else None,
            "helpful_rank_median": helpful_ranks[len(helpful_ranks) // 2] if helpful_ranks else None,
            "helpful_rank_max": max(helpful_ranks) if helpful_ranks else None,
            "delta_mean_helpful": mean([row["delta"] for row in helpful]),
            "delta_mean_non_helpful": mean([row["delta"] for row in non_helpful]),
            "delta_std_helpful": std([row["delta"] for row in helpful]),
            "delta_std_non_helpful": std([row["delta"] for row in non_helpful]),
            "reason_gain_mean_helpful": mean([row["reason_gain"] for row in helpful]),
            "reason_gain_mean_non_helpful": mean([row["reason_gain"] for row in non_helpful]),
        },
        "topk": topk_summary(rows, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        "feature_means": {
            "all": feature_means(rows, feature_keys),
            "helpful": feature_means(helpful, feature_keys),
            "non_helpful": feature_means(non_helpful, feature_keys),
            "top15_selected": feature_means(top15, feature_keys),
            "missed_helpful_below_top20": feature_means(missed_helpful, feature_keys),
        },
        "event_type_buckets": bucket_by_event_type(rows),
        "doc_buckets": bucket_by_doc(rows),
        "top15_false_positives": [case_row(row) for row in top15 if not row["helpful"]][:12],
        "top20_true_positives": [case_row(row) for row in rows[: round(len(rows) * 0.20)] if row["helpful"]],
        "missed_helpful": [case_row(row) for row in sorted(missed_helpful, key=lambda row: row["rank"])],
        "high_gain_missed": [case_row(row) for row in high_gain_missed[:12]],
    }
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), render_markdown(payload))
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "helpful": len(helpful)}, indent=2))


if __name__ == "__main__":
    main()
