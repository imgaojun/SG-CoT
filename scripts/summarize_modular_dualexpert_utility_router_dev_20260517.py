import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from src.stage2_analysis.analyze_adaptive_outcome_router_execution import (  # noqa: E402
    analyze_router,
    analyze_score_router,
    load_prediction_map,
)


RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DEFAULT_BRANCH = "modular_d1930_r2058_utility_gainpos_routecls_noauxwarm_lr2e6_save50"
DEFAULT_SCORE_ROOT = "outputs/stage2_modular_dualexpert/utility_router_gainpos_20260517/route_likelihood"
DEFAULT_REPORT_STEM = "2026-05-17_stage2_modular_dualexpert_utility_router_gainpos_dev_probe"
BUDGETS = [None, 0.10, 0.15, 0.20, 0.30]
DIRECT_DEV = (
    "outputs/stage2_adaptive_runs_user_devpick_frontier/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux_"
    "full_forced_direct_dev_seen_max512/checkpoint-1930/predictions.jsonl"
)
REASON_DEV = (
    "outputs/stage2_adaptive_runs_user_devpick_frontier/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux_"
    "full_forced_reason_dev_seen_max512/checkpoint-2058/predictions.jsonl"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ckpt_num(tag: str):
    return int(tag.split("-", 1)[1])


def fmt(value, digits=4):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def signed(value):
    if value is None:
        return "NA"
    return f"{value:+.4f}"


def pct(value):
    if value is None:
        return "NA"
    return f"{100 * value:.1f}%"


def route_summary_rows(route_root: Path):
    rows = []
    for path in sorted(route_root.glob("checkpoint-*/summary.json"), key=lambda p: ckpt_num(p.parent.name)):
        rows.append({"checkpoint": path.parent.name, **load_json(path)})
    return rows


def nll_summary_rows(nll_root: Path):
    rows = []
    for path in sorted(nll_root.glob("checkpoint-*/dev_seen_summary.json"), key=lambda p: ckpt_num(p.parent.name)):
        rows.append({"checkpoint": path.parent.name, **load_json(path)})
    return rows


def execution_rows(route_root: Path, nll_root: Path, direct_rows, reason_rows):
    rows = []
    for path in sorted(route_root.glob("checkpoint-*/predictions.jsonl"), key=lambda p: ckpt_num(p.parent.name)):
        rows.append(analyze_router(f"{path.parent.name}_gen", path, direct_rows, reason_rows))
    for score_path in sorted(nll_root.glob("checkpoint-*/dev_seen_scores.jsonl"), key=lambda p: ckpt_num(p.parent.name)):
        for budget in BUDGETS:
            rows.append(analyze_score_router(f"{score_path.parent.name}_nll", score_path, budget, direct_rows, reason_rows))
    return rows


def best(rows, key_fn):
    return max(rows, key=key_fn) if rows else None


def render_report(payload):
    lines = [
        "# Modular Dual-Expert Utility Router Dev Probe",
        "",
        payload["description"],
        "",
        "## Route Metrics",
        "",
        "| checkpoint | gen pred reason | gen P/R/F1 | nll argmin pred reason | nll P/R/F1 | nll AUC | best-threshold F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    route_by_ckpt = {row["checkpoint"]: row for row in payload["route_generation_summaries"]}
    nll_by_ckpt = {row["checkpoint"]: row for row in payload["route_nll_summaries"]}
    for ckpt in sorted(set(route_by_ckpt) | set(nll_by_ckpt), key=ckpt_num):
        route = route_by_ckpt.get(ckpt, {})
        nll = nll_by_ckpt.get(ckpt, {})
        threshold = nll.get("best_threshold") or {}
        lines.append(
            "| {ckpt} | {grate} | {gp}/{gr}/{gf} | {nrate} | {np}/{nr}/{nf} | {auc} | {tf} |".format(
                ckpt=ckpt,
                grate=pct(route.get("pred_reason_rate")),
                gp=fmt(route.get("reason_precision"), 3),
                gr=fmt(route.get("reason_recall"), 3),
                gf=fmt(route.get("reason_f1"), 3),
                nrate=pct(nll.get("argmin_pred_reason_rate")),
                np=fmt(nll.get("argmin_reason_precision"), 3),
                nr=fmt(nll.get("argmin_reason_recall"), 3),
                nf=fmt(nll.get("argmin_reason_f1"), 3),
                auc=fmt(nll.get("delta_auc"), 4),
                tf=fmt(threshold.get("reason_f1"), 3),
            )
        )
    lines.extend(
        [
            "",
            "## Execution Simulation",
            "",
            "| router | pred reason | label P/R/F1 | helpful P/R/F1 | routed delta A/E/T | routed A/E/T |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    top_exec = sorted(
        payload["execution_results"],
        key=lambda row: (
            row["routed_delta_vs_direct"]["event_f1"],
            row["routed_delta_vs_direct"]["argument_f1"],
            row["routed_delta_vs_direct"]["trigger_f1"],
        ),
        reverse=True,
    )[:20]
    for row in top_exec:
        label = row["route_vs_label"]
        helpful = row["route_vs_positive_reason_helpful"]
        delta = row["routed_delta_vs_direct"]
        routed = row["routed"]
        lines.append(
            "| {name} | {rate} | {lp}/{lr}/{lf} | {hp}/{hr}/{hf} | {da}/{de}/{dt} | {ra}/{re}/{rt} |".format(
                name=row["name"],
                rate=pct(row["pred_reason_rate"]),
                lp=fmt(label["precision"], 3),
                lr=fmt(label["recall"], 3),
                lf=fmt(label["f1"], 3),
                hp=fmt(helpful["precision"], 3),
                hr=fmt(helpful["recall"], 3),
                hf=fmt(helpful["f1"], 3),
                da=signed(delta["argument_f1"]),
                de=signed(delta["event_f1"]),
                dt=signed(delta["trigger_f1"]),
                ra=fmt(routed["argument_f1"], 4),
                re=fmt(routed["event_f1"], 4),
                rt=fmt(routed["trigger_f1"], 4),
            )
        )

    best_gen = payload["best"]["route_generation_reason_f1"]
    best_nll = payload["best"]["route_nll_argmin_reason_f1"]
    best_exec = payload["best"]["execution_event_delta"]
    top20 = payload["top20_robustness"]
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- Best generated-route F1: `{best_gen['checkpoint']}` with F1 `{best_gen['reason_f1']:.3f}` and pred-reason rate `{best_gen['pred_reason_rate']:.1%}`.",
            f"- Best NLL argmin route F1: `{best_nll['checkpoint']}` with F1 `{best_nll['argmin_reason_f1']:.3f}`, AUC `{best_nll['delta_auc']:.4f}`, and pred-reason rate `{best_nll['argmin_pred_reason_rate']:.1%}`.",
            f"- Best simulated event gain: `{best_exec['name']}` with routed-minus-direct `{best_exec['routed_delta_vs_direct']['argument_f1']:+.4f}` argument, `{best_exec['routed_delta_vs_direct']['event_f1']:+.4f}` event, `{best_exec['routed_delta_vs_direct']['trigger_f1']:+.4f}` trigger.",
            f"- NLL top20 robustness: `{top20['nonnegative_arg_event_trigger_count']}/{top20['count']}` checkpoints have nonnegative argument/event/trigger deltas under top20.",
            "",
            "## Inputs",
            "",
            f"- route generation root: `{payload['route_generation_root']}`",
            f"- route NLL root: `{payload['route_nll_root']}`",
            f"- direct expert dev predictions: `{payload['direct_predictions']}`",
            f"- reason expert dev predictions: `{payload['reason_predictions']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--score_root", default=DEFAULT_SCORE_ROOT)
    parser.add_argument("--report_stem", default=DEFAULT_REPORT_STEM)
    parser.add_argument(
        "--description",
        default=(
            "This dev-only report evaluates an independent utility router trained on "
            "`reason_gain > 0` labels from D1930 direct and R2058 reason experts."
        ),
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    route_root = REPO / (
        "outputs/stage2_adaptive_runs_user_devpick_route/"
        f"{RUN_PREFIX}_{args.branch}_full_route_dev_seen_max16"
    )
    nll_root = REPO / args.score_root / args.branch
    direct_path = REPO / DIRECT_DEV
    reason_path = REPO / REASON_DEV
    output_json = (
        Path(args.output_json)
        if args.output_json
        else REPO / f"reports/artifacts/{args.report_stem}.json"
    )
    output_md = Path(args.output_md) if args.output_md else REPO / f"reports/{args.report_stem}.md"

    route_rows = route_summary_rows(route_root)
    nll_rows = nll_summary_rows(nll_root)
    if not route_rows:
        raise FileNotFoundError(f"no route generation summaries found under {route_root}")
    if not nll_rows:
        raise FileNotFoundError(f"no route NLL summaries found under {nll_root}")

    direct_rows = load_prediction_map(direct_path)
    reason_rows = load_prediction_map(reason_path)
    exec_rows = execution_rows(route_root, nll_root, direct_rows, reason_rows)
    top20_rows = [row for row in exec_rows if row["name"].endswith("_nll_top20")]
    top20_nonnegative = [
        row
        for row in top20_rows
        if row["routed_delta_vs_direct"]["argument_f1"] >= 0.0
        and row["routed_delta_vs_direct"]["event_f1"] >= 0.0
        and row["routed_delta_vs_direct"]["trigger_f1"] >= 0.0
    ]
    payload = {
        "branch": args.branch,
        "description": args.description,
        "route_generation_root": route_root.as_posix(),
        "route_nll_root": nll_root.as_posix(),
        "direct_predictions": direct_path.as_posix(),
        "reason_predictions": reason_path.as_posix(),
        "route_generation_summaries": route_rows,
        "route_nll_summaries": nll_rows,
        "execution_results": exec_rows,
        "top20_robustness": {
            "count": len(top20_rows),
            "nonnegative_arg_event_trigger_count": len(top20_nonnegative),
            "nonnegative_arg_event_trigger_names": [row["name"] for row in top20_nonnegative],
        },
        "best": {
            "route_generation_reason_f1": best(route_rows, lambda row: row["reason_f1"]),
            "route_nll_argmin_reason_f1": best(nll_rows, lambda row: row["argmin_reason_f1"]),
            "execution_event_delta": best(
                exec_rows,
                lambda row: (
                    row["routed_delta_vs_direct"]["event_f1"],
                    row["routed_delta_vs_direct"]["argument_f1"],
                    row["routed_delta_vs_direct"]["trigger_f1"],
                ),
            ),
        },
    }
    write_json(output_json, payload)
    write_text(output_md, render_report(payload))
    print(json.dumps({"output_json": output_json.as_posix(), "output_md": output_md.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
