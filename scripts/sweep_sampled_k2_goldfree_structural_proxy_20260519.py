#!/usr/bin/env python3
import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_goldfree_harmful_cases_20260519 import build_cases, score  # noqa: E402
from scripts.summarize_sampled_confident_router_dev_20260518 import pct, signed, write_json, write_text  # noqa: E402


OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_goldfree_structural_proxy_sweep_20260519/sampledk2_structural_proxy"
REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_goldfree_structural_proxy_sweep.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_goldfree_structural_proxy_sweep.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def top_margin_rule(row):
    margins = [row["old17_18_margin"], row["new19_20_margin"], row["fresh_margin"]]
    return (
        row["fresh_margin"] >= 0.25
        and max(margins) - min(margins) <= 0.50
        and sum(1 for val in margins if val >= 0.25) >= 2
    )


def predicates():
    return {
        "base_margin_stability": top_margin_rule,
        "new_arg_text_le_0p50": lambda r: r["sample_reason_new_arg_text_count_mean"] <= 0.50,
        "new_arg_text_le_0p25": lambda r: r["sample_reason_new_arg_text_count_mean"] <= 0.25,
        "new_arg_text_eq_0": lambda r: r["sample_reason_new_arg_text_count_mean"] <= 0.0,
        "new_event_type_le_0p25": lambda r: r["sample_reason_new_event_type_count_mean"] <= 0.25,
        "new_event_type_eq_0": lambda r: r["sample_reason_new_event_type_count_mean"] <= 0.0,
        "arg_span_jaccard_ge_0p10": lambda r: r["sample_arg_span_jaccard_mean"] >= 0.10,
        "arg_span_jaccard_ge_0p25": lambda r: r["sample_arg_span_jaccard_mean"] >= 0.25,
        "arg_span_jaccard_ge_0p40": lambda r: r["sample_arg_span_jaccard_mean"] >= 0.40,
        "event_role_jaccard_ge_0p40": lambda r: r["sample_event_type_role_jaccard_mean"] >= 0.40,
        "event_role_jaccard_ge_0p50": lambda r: r["sample_event_type_role_jaccard_mean"] >= 0.50,
        "arg_role_jaccard_ge_0p50": lambda r: r["sample_arg_role_jaccard_mean"] >= 0.50,
        "arg_text_jaccard_ge_0p40": lambda r: r["sample_arg_text_jaccard_mean"] >= 0.40,
        "event_count_delta_le_0": lambda r: r["sample_event_count_delta_mean"] <= 0.0,
        "event_count_delta_le_0p25": lambda r: r["sample_event_count_delta_mean"] <= 0.25,
        "argument_count_delta_le_0p25": lambda r: r["sample_argument_count_delta_mean"] <= 0.25,
        "argument_count_delta_le_0": lambda r: r["sample_argument_count_delta_mean"] <= 0.0,
    }


def make_rule(rule_id, parts, pred_map):
    return {
        "id": rule_id,
        "parts": list(parts),
        "fn": lambda row, ps=list(parts): all(pred_map[p](row) for p in ps),
    }


def generate_rules():
    pred_map = predicates()
    rules = [
        make_rule("base_margin_stability", ["base_margin_stability"], pred_map),
        make_rule("base_new_arg_le_0p50", ["base_margin_stability", "new_arg_text_le_0p50"], pred_map),
        make_rule("base_new_event_eq_0", ["base_margin_stability", "new_event_type_eq_0"], pred_map),
        make_rule("base_arg_span_ge_0p10", ["base_margin_stability", "arg_span_jaccard_ge_0p10"], pred_map),
        make_rule("base_event_role_ge_0p40", ["base_margin_stability", "event_role_jaccard_ge_0p40"], pred_map),
        make_rule("base_arg_span_event_role", ["base_margin_stability", "arg_span_jaccard_ge_0p10", "event_role_jaccard_ge_0p40"], pred_map),
        make_rule("base_conservative_no_new", ["base_margin_stability", "new_arg_text_le_0p50", "new_event_type_eq_0"], pred_map),
        make_rule(
            "base_conservative_overlap",
            ["base_margin_stability", "new_arg_text_le_0p50", "arg_span_jaccard_ge_0p10", "event_role_jaccard_ge_0p40"],
            pred_map,
        ),
    ]
    atoms = [
        "new_arg_text_le_0p50",
        "new_arg_text_le_0p25",
        "new_arg_text_eq_0",
        "new_event_type_le_0p25",
        "new_event_type_eq_0",
        "arg_span_jaccard_ge_0p10",
        "arg_span_jaccard_ge_0p25",
        "arg_span_jaccard_ge_0p40",
        "event_role_jaccard_ge_0p40",
        "event_role_jaccard_ge_0p50",
        "arg_role_jaccard_ge_0p50",
        "arg_text_jaccard_ge_0p40",
        "event_count_delta_le_0",
        "event_count_delta_le_0p25",
        "argument_count_delta_le_0p25",
        "argument_count_delta_le_0",
    ]
    seen = {rule["id"] for rule in rules}
    for size in [1, 2, 3, 4]:
        for combo in itertools.combinations(atoms, size):
            if "new_arg_text_le_0p50" in combo and "new_arg_text_le_0p25" in combo:
                continue
            if "new_arg_text_le_0p25" in combo and "new_arg_text_eq_0" in combo:
                continue
            if "new_event_type_le_0p25" in combo and "new_event_type_eq_0" in combo:
                continue
            if "arg_span_jaccard_ge_0p10" in combo and "arg_span_jaccard_ge_0p25" in combo:
                continue
            if "arg_span_jaccard_ge_0p25" in combo and "arg_span_jaccard_ge_0p40" in combo:
                continue
            if "event_count_delta_le_0" in combo and "event_count_delta_le_0p25" in combo:
                continue
            if "argument_count_delta_le_0" in combo and "argument_count_delta_le_0p25" in combo:
                continue
            parts = ["base_margin_stability", *combo]
            name = "auto_" + "__".join(parts)
            if name in seen:
                continue
            rules.append(make_rule(name, parts, pred_map))
            seen.add(name)
    return rules


def summarize_metrics(rows):
    return {
        metric: mean(row[metric] for row in rows)
        for metric in ["argument_f1", "event_f1", "trigger_f1", "score"]
    }


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_rule(cases, split, rule, source):
    split_cases = cases if split == "test" else [row for row in cases if row["split"] == split]
    direct_rows = []
    routed_rows = []
    selected = set()
    helpful = set()
    selected_gains = []
    for row in split_cases:
        direct = row[f"{source}_direct"]
        reason = row[f"{source}_reason"]
        gain = score(reason) - score(direct)
        direct_rows.append(direct)
        if gain > 0:
            helpful.add(row["case_id"])
        if rule["fn"](row):
            selected.add(row["case_id"])
            selected_gains.append(gain)
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)
    direct_avg = summarize_metrics(direct_rows)
    routed_avg = summarize_metrics(routed_rows)
    tp = len(selected & helpful)
    fp = len(selected - helpful)
    fn = len(helpful - selected)
    return {
        "rule_id": rule["id"],
        "parts": rule["parts"],
        "split": split,
        "source": source,
        "num_examples": len(split_cases),
        "pred_reason_count": len(selected),
        "pred_reason_rate": len(selected) / len(split_cases) if split_cases else 0.0,
        "selected_reason_score_gain_mean": mean(selected_gains),
        "selected_reason_harm_rate": mean(1.0 if gain < 0 else 0.0 for gain in selected_gains),
        "route_vs_helpful": route_prf(tp, fp, fn),
        "routed_minus_direct": {
            metric: routed_avg[metric] - direct_avg[metric]
            for metric in ["argument_f1", "event_f1", "trigger_f1", "score"]
        },
    }


def consolidate(rule, rows):
    by_key = {(row["source"], row["split"]): row for row in rows}
    test = by_key[("single_gen", "test")]
    seen = by_key[("single_gen", "test_seen")]
    unseen = by_key[("single_gen", "test_unseen")]
    positive = (
        test["routed_minus_direct"]["score"] > 0
        and seen["routed_minus_direct"]["score"] > 0
        and unseen["routed_minus_direct"]["score"] >= 0
        and 0.02 <= test["pred_reason_rate"] <= 0.10
    )
    strict = (
        positive
        and test["selected_reason_harm_rate"] <= 0.15
        and seen["selected_reason_harm_rate"] <= 0.15
    )
    return {
        "rule_id": rule["id"],
        "parts": rule["parts"],
        "test_score_delta": test["routed_minus_direct"]["score"],
        "seen_score_delta": seen["routed_minus_direct"]["score"],
        "unseen_score_delta": unseen["routed_minus_direct"]["score"],
        "test_aet_delta": {
            metric: test["routed_minus_direct"][metric]
            for metric in ["argument_f1", "event_f1", "trigger_f1"]
        },
        "test_reason_rate": test["pred_reason_rate"],
        "seen_reason_rate": seen["pred_reason_rate"],
        "unseen_reason_rate": unseen["pred_reason_rate"],
        "test_harm_rate": test["selected_reason_harm_rate"],
        "seen_harm_rate": seen["selected_reason_harm_rate"],
        "unseen_harm_rate": unseen["selected_reason_harm_rate"],
        "positive_screen": positive,
        "strict_low_harm_screen": strict,
        "screen_score": min(
            test["routed_minus_direct"]["score"],
            seen["routed_minus_direct"]["score"],
            unseen["routed_minus_direct"]["score"],
        ),
    }


def compact_case(row):
    return {
        "case_id": row["case_id"],
        "wnd_id": row["wnd_id"],
        "split": row["split"],
        "single_gen_score_gain": row["single_gen_score_gain"],
        "k2_expected_score_gain": row["k2_expected_score_gain"],
        "fresh_margin": row["fresh_margin"],
        "avg_margin": row["avg_margin"],
        "margin_range": row["margin_range"],
        "sample_reason_new_arg_text_count_mean": row["sample_reason_new_arg_text_count_mean"],
        "sample_reason_new_event_type_count_mean": row["sample_reason_new_event_type_count_mean"],
        "sample_arg_span_jaccard_mean": row["sample_arg_span_jaccard_mean"],
        "sample_event_type_role_jaccard_mean": row["sample_event_type_role_jaccard_mean"],
        "gold_event_types": row.get("meta", {}).get("gold_event_types"),
    }


def render_table(rows, limit=25):
    lines = [
        "| rank | rule | positive | strict | reason test/seen/unseen | score test/seen/unseen | A/E/T test | harm test/seen/unseen |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows[:limit], 1):
        aet = row["test_aet_delta"]
        lines.append(
            f"| {idx} | `{row['rule_id']}` | `{row['positive_screen']}` | `{row['strict_low_harm_screen']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} | "
            f"{signed(aet['argument_f1'])}/{signed(aet['event_f1'])}/{signed(aet['trigger_f1'])} | "
            f"{pct(row['test_harm_rate'])}/{pct(row['seen_harm_rate'])}/{pct(row['unseen_harm_rate'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    positive = [row for row in payload["consolidated"] if row["positive_screen"]]
    strict = [row for row in payload["consolidated"] if row["strict_low_harm_screen"]]
    best = payload["consolidated"][0]
    base = next(row for row in payload["consolidated"] if row["rule_id"] == "base_margin_stability")
    lines = [
        "# Sampled K2 Gold-Free Structural Proxy Sweep",
        "",
        "This sweep adds structural overlap guards to the best margin-stability proxy.",
        "",
        f"- rules evaluated: `{payload['num_rules']}`",
        f"- cases: `{payload['num_cases']}`",
        f"- output root: `{payload['output_root']}`",
        "",
        "## Leaderboard",
        "",
        render_table(payload["consolidated"]),
        "",
        "## Strict Low-Harm Rules",
        "",
    ]
    if strict:
        lines.append(render_table(strict, limit=20))
    else:
        lines.append("No rule passed the strict low-harm screen.")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- Base margin-stability score delta test/seen/unseen: `{base['test_score_delta']:+.4f}/{base['seen_score_delta']:+.4f}/{base['unseen_score_delta']:+.4f}` with harm `{base['test_harm_rate']:.1%}/{base['seen_harm_rate']:.1%}/{base['unseen_harm_rate']:.1%}`.",
            f"- Best structural rule: `{best['rule_id']}` with score delta `{best['test_score_delta']:+.4f}/{best['seen_score_delta']:+.4f}/{best['unseen_score_delta']:+.4f}` and harm `{best['test_harm_rate']:.1%}/{best['seen_harm_rate']:.1%}/{best['unseen_harm_rate']:.1%}`.",
            f"- Positive-screen rules: `{len(positive)}`; strict low-harm rules: `{len(strict)}`.",
        ]
    )
    if strict:
        lines.append(f"- Recommended locked-validation candidate: `{strict[0]['rule_id']}` with parts `{strict[0]['parts']}`.")
    else:
        lines.append("- Structural guards improved interpretability but did not yet produce a low-harm deployable proxy.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['report_json']}`",
            f"- cases JSONL: `{payload['cases_jsonl']}`",
            f"- evaluations JSONL: `{payload['evaluations_jsonl']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    fresh_config = load_json(repo_path(args.fresh_config))
    consensus_config = load_json(repo_path(args.consensus_config))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = build_cases(fresh_config, consensus_config)
    rules = generate_rules()
    evaluations = []
    grouped = defaultdict(list)
    for rule in rules:
        for source in ["single_gen", "k2_expected"]:
            for split in ["test", "test_seen", "test_unseen"]:
                row = evaluate_rule(cases, split, rule, source)
                evaluations.append(row)
                grouped[rule["id"]].append(row)
    rule_by_id = {rule["id"]: rule for rule in rules}
    consolidated = [consolidate(rule_by_id[rid], rows) for rid, rows in grouped.items()]
    consolidated.sort(
        key=lambda row: (
            row["strict_low_harm_screen"],
            row["positive_screen"],
            row["screen_score"],
            row["test_score_delta"],
            -row["test_harm_rate"],
        ),
        reverse=True,
    )
    cases_jsonl = OUTPUT_ROOT / "cases.jsonl"
    eval_jsonl = OUTPUT_ROOT / "evaluations.jsonl"
    write_jsonl(cases_jsonl, [compact_case(row) for row in cases])
    write_jsonl(eval_jsonl, evaluations)
    payload = {
        "fresh_config": repo_path(args.fresh_config).as_posix(),
        "consensus_config": repo_path(args.consensus_config).as_posix(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "num_cases": len(cases),
        "num_rules": len(rules),
        "consolidated": consolidated,
        "cases_jsonl": cases_jsonl.as_posix(),
        "evaluations_jsonl": eval_jsonl.as_posix(),
        "report_md": REPORT_MD.as_posix(),
        "report_json": REPORT_JSON.as_posix(),
    }
    write_json(REPORT_JSON, payload)
    write_json(OUTPUT_ROOT / "summary.json", payload)
    write_text(REPORT_MD, render_report(payload))
    print(json.dumps({"report_md": REPORT_MD.as_posix(), "report_json": REPORT_JSON.as_posix()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-config", required=True)
    parser.add_argument("--consensus-config", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
