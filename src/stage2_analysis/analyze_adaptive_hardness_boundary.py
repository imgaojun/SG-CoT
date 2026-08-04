import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_cot.build_selective_aux_reasoning_dataset import (  # noqa: E402
    build_confrare_stats,
    hardconf_score_row,
    row_id,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl, load_schema_map  # noqa: E402


FOCUS_RUNS = [
    {
        "branch": "confrare10_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_role_hint_plan_lite",
        "selection": "direct_anchor_best",
        "checkpoint": "checkpoint-1806",
    },
    {
        "branch": "confrare5_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare5_heur5_type_role_hint_plan_lite",
        "selection": "adaptive_tradeoff_best",
        "checkpoint": "checkpoint-1161",
    },
    {
        "branch": "pairdirect",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_role_hint_plan_lite_pairdirect",
        "selection": "adaptive_tradeoff_best",
        "checkpoint": "checkpoint-1704",
    },
    {
        "branch": "pairdirect",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_role_hint_plan_lite_pairdirect",
        "selection": "direct_anchor_best",
        "checkpoint": "checkpoint-1846",
    },
    {
        "branch": "directanchor",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_role_hint_plan_lite_directanchor",
        "selection": "direct_anchor_best",
        "checkpoint": "checkpoint-2130",
    },
    {
        "branch": "directanchor",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_role_hint_plan_lite_directanchor",
        "selection": "adaptive_tradeoff_best",
        "checkpoint": "checkpoint-2272",
    },
]
HARDCONF_FOCUS_RUNS = [
    {
        "branch": "hardconf10_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_heur10_type_role_hint_plan_lite",
        "selection": "seen_stable_best",
        "checkpoint": "checkpoint-1806",
    },
    {
        "branch": "hardconf10_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_heur10_type_role_hint_plan_lite",
        "selection": "hard_reason_best",
        "checkpoint": "checkpoint-774",
    },
    {
        "branch": "hardconf10_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_heur10_type_role_hint_plan_lite",
        "selection": "balanced_hardroute_best",
        "checkpoint": "checkpoint-1806",
    },
    {
        "branch": "hardconf15_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf15_heur15_type_role_hint_plan_lite",
        "selection": "seen_stable_best",
        "checkpoint": "checkpoint-2064",
    },
    {
        "branch": "hardconf15_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf15_heur15_type_role_hint_plan_lite",
        "selection": "hard_reason_best",
        "checkpoint": "checkpoint-387",
    },
    {
        "branch": "hardconf15_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf15_heur15_type_role_hint_plan_lite",
        "selection": "balanced_hardroute_best",
        "checkpoint": "checkpoint-2064",
    },
    {
        "branch": "hardconf10_calibrated_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_calibrated_type_role_hint_plan_lite",
        "selection": "seen_stable_best",
        "checkpoint": "checkpoint-1935",
    },
    {
        "branch": "hardconf10_calibrated_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_calibrated_type_role_hint_plan_lite",
        "selection": "hard_reason_best",
        "checkpoint": "checkpoint-645",
    },
    {
        "branch": "hardconf10_calibrated_type_role_hint_plan_lite",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_calibrated_type_role_hint_plan_lite",
        "selection": "balanced_hardroute_best",
        "checkpoint": "checkpoint-1935",
    },
    {
        "branch": "hardconf10_directdup",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_directdup",
        "selection": "seen_stable_best",
        "checkpoint": "checkpoint-1278",
    },
    {
        "branch": "hardconf10_directdup",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_directdup",
        "selection": "hard_reason_best",
        "checkpoint": "checkpoint-142",
    },
    {
        "branch": "hardconf10_directdup",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_directdup",
        "selection": "balanced_hardroute_best",
        "checkpoint": "checkpoint-1278",
    },
]
SPLITS = ["test", "test_seen", "test_unseen"]
BUDGETS = [0.05, 0.10, 0.15, 0.20]
DIRECT_EVAL_JSONL = {
    "test": "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_test_pos.jsonl",
    "test_seen": "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_test_seen_pos.jsonl",
    "test_unseen": "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_test_unseen_pos.jsonl",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def prediction_key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row_id(row)


def normalize_events(events_payload):
    events = events_payload.get("events", []) if isinstance(events_payload, dict) else []
    trigger_set = set()
    argument_set = set()
    event_set = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        trigger = event.get("trigger", {})
        if not isinstance(trigger, dict):
            trigger = {}
        trig = (event_type, trigger.get("start"), trigger.get("end"))
        trigger_set.add(trig)
        args = []
        raw_arguments = event.get("arguments", [])
        if not isinstance(raw_arguments, list):
            raw_arguments = []
        for arg in raw_arguments:
            if not isinstance(arg, dict):
                continue
            argument_set.add(
                (
                    event_type,
                    trigger.get("start"),
                    trigger.get("end"),
                    arg.get("role"),
                    arg.get("start"),
                    arg.get("end"),
                )
            )
            args.append((arg.get("role"), arg.get("start"), arg.get("end")))
        sorted_args = tuple(
            sorted(
                args,
                key=lambda item: (
                    item[0] or "",
                    -1 if item[1] is None else item[1],
                    -1 if item[2] is None else item[2],
                ),
            )
        )
        event_set.add((event_type, trigger.get("start"), trigger.get("end"), sorted_args))
    return trigger_set, argument_set, event_set


def prf(pred_set, gold_set):
    if not pred_set and not gold_set:
        return {"p": 1.0, "r": 1.0, "f1": 1.0}
    if not pred_set:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    if not gold_set:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"p": p, "r": r, "f1": f1}


def merge_metric_dict(metric_list, key):
    vals = [m[key] for m in metric_list]
    return sum(vals) / len(vals) if vals else 0.0


def prediction_path(root: Path, run, mode: str, split: str):
    return root / run["formal_slug"] / f"frontier_{run['selection']}" / mode / split / "predictions.jsonl"


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def score(row):
    return float(row.get("argument_f1", 0.0) or 0.0) + float(row.get("event_f1", 0.0) or 0.0) + 0.25 * float(row.get("trigger_f1", 0.0) or 0.0)


def metric_sets(row, route: str):
    payload = row["final_predicted"] if route == "reason" else row["predicted"]
    gold = row.get("gold") or {"events": []}
    pred_trig, pred_arg, pred_event = normalize_events(payload or {"events": []})
    gold_trig, gold_arg, gold_event = normalize_events(gold)
    return {
        "trigger": prf(pred_trig, gold_trig),
        "argument": prf(pred_arg, gold_arg),
        "event": prf(pred_event, gold_event),
    }


def summarize_metric_rows(rows):
    if not rows:
        return {
            "num_examples": 0,
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "reason_rate": 0.0,
        }
    return {
        "num_examples": len(rows),
        "trigger_f1": merge_metric_dict([row["metric"]["trigger"] for row in rows], "f1"),
        "argument_f1": merge_metric_dict([row["metric"]["argument"] for row in rows], "f1"),
        "event_f1": merge_metric_dict([row["metric"]["event"] for row in rows], "f1"),
        "reason_rate": sum(1 for row in rows if row["route"] == "reason") / len(rows),
    }


def build_feature_map(eval_jsonl: Path, schema_path: Path):
    source_rows = load_jsonl(eval_jsonl)
    schema_by_type = load_schema_map(schema_path)
    stats = build_confrare_stats(source_rows)
    features = {}
    for idx, row in enumerate(source_rows):
        scored = hardconf_score_row(idx, row, schema_by_type, stats)
        features[row_id(row)] = {k: v for k, v in scored.items() if k not in {"idx"}}
    return features


def summarize_feature_means(samples):
    keys = [
        "hardconf_score",
        "confusion_norm",
        "role_signature_rarity",
        "role_density_norm",
        "multi_event_or_multi_trigger",
        "core_role_absence_risk",
    ]
    if not samples:
        return {key: 0.0 for key in keys}
    return {
        key: sum((sample.get("features") or {}).get(key, 0.0) for sample in samples) / len(samples)
        for key in keys
    }


def route_capture(samples):
    helpful = [sample for sample in samples if sample["reason_helpful"]]
    free_reason = [sample for sample in samples if sample["free_route_pred"] == "reason"]
    captured = [sample for sample in helpful if sample["free_route_pred"] == "reason"]
    return {
        "reason_helpful_count": len(helpful),
        "free_reason_count": len(free_reason),
        "captured_count": len(captured),
        "capture_recall": len(captured) / len(helpful) if helpful else 0.0,
        "capture_precision": len(captured) / len(free_reason) if free_reason else 0.0,
    }


def oracle_rows(samples, budget):
    cap = round(len(samples) * budget)
    candidates = [sample for sample in samples if sample["reason_gain"] > 1e-9]
    candidates.sort(key=lambda sample: (sample["reason_gain"], sample["key"]), reverse=True)
    reason_keys = {sample["key"] for sample in candidates[:cap]}
    routed = []
    for sample in samples:
        route = "reason" if sample["key"] in reason_keys else "direct"
        routed.append(
            {
                "route": route,
                "metric": sample[f"{route}_metric"],
            }
        )
    return summarize_metric_rows(routed)


def analyze_run_split(root: Path, run, split: str, schema_path: Path):
    paths = {
        "free_route": prediction_path(root, run, "free_route", split),
        "forced_direct": prediction_path(root, run, "forced_direct", split),
        "forced_reason": prediction_path(root, run, "forced_reason", split),
    }
    for mode, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {mode} predictions: {path}")
    free = load_prediction_map(paths["free_route"])
    direct = load_prediction_map(paths["forced_direct"])
    reason = load_prediction_map(paths["forced_reason"])
    eval_jsonl = Path(direct[next(iter(direct))]["meta"].get("source_jsonl", "")) if direct else None
    if eval_jsonl is None or not eval_jsonl.exists() or eval_jsonl.is_dir():
        eval_jsonl = Path(DIRECT_EVAL_JSONL[split])
    if eval_jsonl is None or not eval_jsonl.exists():
        # Fall back to deriving features from forced-direct prediction metadata only.
        features = {}
    else:
        features = build_feature_map(eval_jsonl, schema_path)

    samples = []
    for key in sorted(set(free) & set(direct) & set(reason)):
        free_row = free[key]
        direct_row = direct[key]
        reason_row = reason[key]
        direct_metric = {
            "trigger": {"f1": direct_row.get("trigger_f1", 0.0)},
            "argument": {"f1": direct_row.get("argument_f1", 0.0)},
            "event": {"f1": direct_row.get("event_f1", 0.0)},
        }
        reason_metric = {
            "trigger": {"f1": reason_row.get("trigger_f1", 0.0)},
            "argument": {"f1": reason_row.get("argument_f1", 0.0)},
            "event": {"f1": reason_row.get("event_f1", 0.0)},
        }
        free_metric = {
            "trigger": {"f1": free_row.get("trigger_f1", 0.0)},
            "argument": {"f1": free_row.get("argument_f1", 0.0)},
            "event": {"f1": free_row.get("event_f1", 0.0)},
        }
        direct_score = score(direct_row)
        reason_score = score(reason_row)
        samples.append(
            {
                "key": key,
                "split": split,
                "free_route_pred": free_row.get("route_pred", "unknown"),
                "reason_used": bool(free_row.get("reason_used")),
                "direct_score": direct_score,
                "reason_score": reason_score,
                "reason_gain": reason_score - direct_score,
                "reason_helpful": reason_score > direct_score + 1e-9,
                "direct_metric": direct_metric,
                "reason_metric": reason_metric,
                "free_metric": free_metric,
                "features": features.get(key, {}),
                "gold": direct_row.get("gold"),
                "free_predicted": free_row.get("final_predicted"),
                "direct_predicted": direct_row.get("final_predicted"),
                "reason_predicted": reason_row.get("final_predicted"),
            }
        )

    direct_summary = summarize_metric_rows([{"route": "direct", "metric": sample["direct_metric"]} for sample in samples])
    reason_summary = summarize_metric_rows([{"route": "reason", "metric": sample["reason_metric"]} for sample in samples])
    free_summary = summarize_metric_rows([{"route": sample["free_route_pred"], "metric": sample["free_metric"]} for sample in samples])
    oracle = {f"cap{int(budget * 100)}": oracle_rows(samples, budget) for budget in BUDGETS}
    helpful = [sample for sample in samples if sample["reason_helpful"]]
    harmful = [sample for sample in samples if not sample["reason_helpful"]]
    return {
        "split": split,
        "num_examples": len(samples),
        "direct": direct_summary,
        "forced_reason": reason_summary,
        "free_route": free_summary,
        "oracle": oracle,
        "oracle_gains_vs_direct": {
            name: {
                "trigger_f1": summary["trigger_f1"] - direct_summary["trigger_f1"],
                "argument_f1": summary["argument_f1"] - direct_summary["argument_f1"],
                "event_f1": summary["event_f1"] - direct_summary["event_f1"],
            }
            for name, summary in oracle.items()
        },
        "route_capture": route_capture(samples),
        "feature_means": {
            "reason_helpful": summarize_feature_means(helpful),
            "reason_not_helpful": summarize_feature_means(harmful),
            "free_reason": summarize_feature_means([sample for sample in samples if sample["free_route_pred"] == "reason"]),
            "free_direct": summarize_feature_means([sample for sample in samples if sample["free_route_pred"] == "direct"]),
        },
        "top_reason_helpful": sorted(
            [
                {
                    "key": sample["key"],
                    "reason_gain": sample["reason_gain"],
                    "free_route_pred": sample["free_route_pred"],
                    "features": sample["features"],
                    "gold": sample["gold"],
                    "direct_predicted": sample["direct_predicted"],
                    "reason_predicted": sample["reason_predicted"],
                }
                for sample in helpful
            ],
            key=lambda item: (item["reason_gain"], item["key"]),
            reverse=True,
        )[:10],
    }


def analyze(root: Path, schema_path: Path, focus_runs):
    runs = []
    for run in focus_runs:
        split_payload = {}
        for split in SPLITS:
            split_payload[split] = analyze_run_split(root, run, split, schema_path)
        runs.append({**run, "splits": split_payload})
    return {"runs": runs}


def md_table(payload):
    lines = [
        "# Adaptive Hardness Boundary Analysis",
        "",
        "## Oracle Hard-Route Summary",
        "",
        "| branch | selection | split | direct arg/event | free arg/event/reason | oracle15 arg/event/reason | oracle15 gain arg/event | capture recall/precision |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        for split in SPLITS:
            row = run["splits"][split]
            direct = row["direct"]
            free = row["free_route"]
            oracle15 = row["oracle"]["cap15"]
            gain = row["oracle_gains_vs_direct"]["cap15"]
            capture = row["route_capture"]
            lines.append(
                "| `{branch}` | `{sel}` | `{split}` | {darg:.4f}/{devent:.4f} | {farg:.4f}/{fevent:.4f}/{frr:.3f} | {oarg:.4f}/{oevent:.4f}/{orr:.3f} | {garg:+.4f}/{gevent:+.4f} | {rec:.3f}/{prec:.3f} |".format(
                    branch=run["branch"],
                    sel=run["selection"],
                    split=split,
                    darg=direct["argument_f1"],
                    devent=direct["event_f1"],
                    farg=free["argument_f1"],
                    fevent=free["event_f1"],
                    frr=free["reason_rate"],
                    oarg=oracle15["argument_f1"],
                    oevent=oracle15["event_f1"],
                    orr=oracle15["reason_rate"],
                    garg=gain["argument_f1"],
                    gevent=gain["event_f1"],
                    rec=capture["capture_recall"],
                    prec=capture["capture_precision"],
                )
            )
    lines.extend(["", "## Feature Contrast", ""])
    lines.append("| branch | selection | split | helpful hardconf | non-helpful hardconf | free-reason hardconf | free-direct hardconf |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for run in payload["runs"]:
        for split in SPLITS:
            features = run["splits"][split]["feature_means"]
            lines.append(
                "| `{}` | `{}` | `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    run["branch"],
                    run["selection"],
                    split,
                    features["reason_helpful"]["hardconf_score"],
                    features["reason_not_helpful"]["hardconf_score"],
                    features["free_reason"]["hardconf_score"],
                    features["free_direct"]["hardconf_score"],
                )
            )
    lines.extend(["", "## Gate Reading", ""])
    passed = []
    for run in payload["runs"]:
        for split in SPLITS:
            gain = run["splits"][split]["oracle_gains_vs_direct"]["cap15"]
            if (split in {"test", "test_seen"} and (gain["argument_f1"] >= 0.01 or gain["event_f1"] >= 0.01)) or (
                split == "test_unseen" and (gain["argument_f1"] >= 0.02 or gain["event_f1"] >= 0.02)
            ):
                passed.append((run["branch"], run["selection"], split, gain))
    if passed:
        lines.append("Wave 1 gate passes under oracle15 for:")
        for branch, selection, split, gain in passed:
            lines.append(
                f"- `{branch}/{selection}/{split}`: argument `{gain['argument_f1']:+.4f}`, event `{gain['event_f1']:+.4f}`"
            )
    else:
        lines.append("Wave 1 gate does not pass under oracle15.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_root", default="outputs/stage2_adaptive_runs_user_formal_clean")
    parser.add_argument("--schema_path", default="data/schema/richere-en.event_schema.json")
    parser.add_argument(
        "--focus_preset",
        choices=["type_role_hint", "hardconf"],
        default="type_role_hint",
    )
    parser.add_argument("--output_md", default="reports/2026-05-10_stage2_adaptive_hardness_boundary_analysis.md")
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-10_stage2_adaptive_hardness_boundary_analysis.json")
    args = parser.parse_args()

    focus_runs = HARDCONF_FOCUS_RUNS if args.focus_preset == "hardconf" else FOCUS_RUNS
    payload = analyze(Path(args.formal_root), Path(args.schema_path), focus_runs)
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), md_table(payload))
    print(json.dumps({"output_md": args.output_md, "output_json": args.output_json}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
