import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BRANCH = "modular_d1930_r2058_utility_m02_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/utility_router_m02_20260520/route_likelihood" / BRANCH
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
REPORT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_utility_router_m02_nll_dev_probe.json"
REPORT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_utility_router_m02_nll_dev_probe.md"
BUDGETS = [None, 0.10, 0.15, 0.20, 0.30]


def ckpt_num(path: Path) -> int:
    return int(path.parent.name.split("-", 1)[1])


def main() -> None:
    summaries = sorted(SCORE_ROOT.glob("checkpoint-*/dev_seen_summary.json"), key=ckpt_num)
    if not summaries:
        raise FileNotFoundError(f"no route-NLL summaries found under {SCORE_ROOT}")

    cmd = [
        sys.executable,
        "src/stage2_analysis/analyze_adaptive_outcome_router_execution.py",
        "--forced_direct_predictions",
        DIRECT_DEV.as_posix(),
        "--forced_reason_predictions",
        REASON_DEV.as_posix(),
        "--output_json",
        REPORT_JSON.as_posix(),
        "--output_md",
        REPORT_MD.as_posix(),
    ]
    for summary_path in summaries:
        ckpt = summary_path.parent.name
        score_path = summary_path.parent / "dev_seen_scores.jsonl"
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        for budget in BUDGETS:
            if budget is None:
                spec = f"{ckpt}_nll={score_path.as_posix()}"
            else:
                spec = f"{ckpt}_nll={score_path.as_posix()}:{budget}"
            cmd.extend(["--score_router", spec])

    subprocess.run(cmd, cwd=REPO, check=True)

    payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    routers = payload["routers"]
    best_event = max(
        routers,
        key=lambda row: (
            row["routed_delta_vs_direct"]["event_f1"],
            row["routed_delta_vs_direct"]["argument_f1"],
            row["routed_delta_vs_direct"]["trigger_f1"],
        ),
    )
    best_all_nonneg = [
        row
        for row in routers
        if row["routed_delta_vs_direct"]["argument_f1"] >= 0
        and row["routed_delta_vs_direct"]["event_f1"] >= 0
        and row["routed_delta_vs_direct"]["trigger_f1"] >= 0
    ]
    nll_summaries = [json.loads(path.read_text(encoding="utf-8")) | {"checkpoint": path.parent.name} for path in summaries]
    best_argmin_f1 = max(nll_summaries, key=lambda row: row["argmin_reason_f1"])
    best_auc = max(nll_summaries, key=lambda row: row["delta_auc"] if row["delta_auc"] is not None else -1)
    digest = {
        "report_json": REPORT_JSON.as_posix(),
        "report_md": REPORT_MD.as_posix(),
        "num_nll_summaries": len(nll_summaries),
        "best_argmin_reason_f1": {
            "checkpoint": best_argmin_f1["checkpoint"],
            "argmin_reason_f1": best_argmin_f1["argmin_reason_f1"],
            "argmin_pred_reason_rate": best_argmin_f1["argmin_pred_reason_rate"],
            "delta_auc": best_argmin_f1["delta_auc"],
        },
        "best_auc": {
            "checkpoint": best_auc["checkpoint"],
            "argmin_reason_f1": best_auc["argmin_reason_f1"],
            "argmin_pred_reason_rate": best_auc["argmin_pred_reason_rate"],
            "delta_auc": best_auc["delta_auc"],
        },
        "best_execution_event_delta": {
            "name": best_event["name"],
            "pred_reason_rate": best_event["pred_reason_rate"],
            "routed_delta_vs_direct": best_event["routed_delta_vs_direct"],
            "routed": best_event["routed"],
        },
        "all_nonnegative_arg_event_trigger": [
            {
                "name": row["name"],
                "pred_reason_rate": row["pred_reason_rate"],
                "routed_delta_vs_direct": row["routed_delta_vs_direct"],
            }
            for row in best_all_nonneg
        ],
    }
    digest_path = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_utility_router_m02_nll_dev_digest.json"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(json.dumps(digest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(digest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
