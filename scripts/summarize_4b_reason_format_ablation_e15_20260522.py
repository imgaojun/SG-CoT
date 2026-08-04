import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_4b_reason_expert_e13b_20260521 import evaluate_policy, load_new_pair  # noqa: E402


VARIANTS = {
    "e13b_current": REPO / "outputs/stage2_4b_reason_expert/e13b_formal_20260521",
    "e15a_noreasonblock": REPO / "outputs/stage2_4b_reason_format_ablation/e15_formal_20260522/e15a_noreasonblock",
    "e15b_minimaltype": REPO / "outputs/stage2_4b_reason_format_ablation/e15_formal_20260522/e15b_minimaltype",
    "e15c_finalfirst": REPO / "outputs/stage2_4b_reason_format_ablation/e15_formal_20260522/e15c_finalfirst",
}
OUT_JSON = REPO / "reports/artifacts/2026-05-22_stage2_4b_reason_format_ablation_e15.json"
OUT_MD = REPO / "reports/2026-05-22_stage2_4b_reason_format_ablation_e15.md"
SPLITS = ["test_seen", "test_unseen"]


def load_pair(root, split):
    from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import load_prediction_map

    return (
        load_prediction_map(root / "forced_direct" / split / "predictions.jsonl"),
        load_prediction_map(root / "forced_reason" / split / "predictions.jsonl"),
    )


def aggregate(rows):
    out = []
    for system in sorted({row["system"] for row in rows}):
        for policy in sorted({row["policy"] for row in rows if row["system"] == system}):
            items = [row for row in rows if row["system"] == system and row["policy"] == policy]
            total = sum(row["num_examples"] for row in items)
            if not total:
                continue
            pred_reason_count = sum(row["pred_reason_count"] for row in items)
            agg = {
                "system": system,
                "split": "test",
                "policy": policy,
                "num_examples": total,
                "pred_reason_count": pred_reason_count,
                "pred_reason_rate": pred_reason_count / total,
            }
            for group in ["direct", "forced_reason_all", "routed"]:
                agg[group] = {}
                for metric in ["trigger_f1", "argument_f1", "event_f1"]:
                    agg[group][metric] = sum(row[group][metric] * row["num_examples"] for row in items) / total
            agg["routed_minus_direct"] = {
                metric: agg["routed"][metric] - agg["direct"][metric]
                for metric in ["trigger_f1", "argument_f1", "event_f1"]
            }
            agg["selected_delta_mean"] = {}
            out.append(agg)
    return out


def fmt(value):
    return f"{value:.4f}"


def signed(value):
    return f"{value:+.4f}"


def pct(value):
    return f"{100 * value:.1f}%"


def aet(row):
    m = row["routed"]
    return f"{fmt(m['argument_f1'])} / {fmt(m['event_f1'])} / {fmt(m['trigger_f1'])}"


def delta(row):
    d = row["routed_minus_direct"]
    return f"{signed(d['argument_f1'])} / {signed(d['event_f1'])} / {signed(d['trigger_f1'])}"


def render(payload):
    rows = [row for row in payload["results"] if row["split"] == "test"]
    lines = [
        "# 4B Reason Format Ablation E15",
        "",
        "This compares forced-reason output formats without routing. Deltas are against each variant's forced-direct output.",
        "",
        "| system | policy | reason rate | routed A/E/T | delta vs direct A/E/T |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["system"], item["policy"])):
        lines.append(
            f"| `{row['system']}` | `{row['policy']}` | {pct(row['pred_reason_rate'])} | {aet(row)} | {delta(row)} |"
        )
    lines.extend(["", "## Reading", ""])
    for row in sorted(rows, key=lambda item: item["system"]):
        if row["policy"] == "reason_all":
            lines.append(f"- `{row['system']}` reason-all delta A/E/T: `{delta(row)}`.")
    return "\n".join(lines) + "\n"


def main():
    rows = []
    for system, root in VARIANTS.items():
        if system == "e13b_current":
            for split in SPLITS:
                direct, reason = load_new_pair(split)
                keys = sorted(set(direct) & set(reason))
                rows.append(evaluate_policy(system, split, "direct_only", direct, reason, set()))
                rows.append(evaluate_policy(system, split, "reason_all", direct, reason, set(keys)))
            continue
        if not root.exists():
            continue
        for split in SPLITS:
            direct_path = root / "forced_direct" / split / "predictions.jsonl"
            reason_path = root / "forced_reason" / split / "predictions.jsonl"
            if not direct_path.exists() or not reason_path.exists():
                continue
            direct, reason = load_pair(root, split)
            keys = sorted(set(direct) & set(reason))
            rows.append(evaluate_policy(system, split, "direct_only", direct, reason, set()))
            rows.append(evaluate_policy(system, split, "reason_all", direct, reason, set(keys)))
    rows.extend(aggregate(rows))
    payload = {"variants": {key: value.as_posix() for key, value in VARIANTS.items()}, "results": rows}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
