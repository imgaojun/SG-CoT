import json
import sys
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_4b_reason_expert_e13b_20260521 import evaluate_policy, load_new_pair  # noqa: E402
from scripts.summarize_4b_reason_format_ablation_e15_20260522 import aggregate  # noqa: E402
from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import load_prediction_map  # noqa: E402


FORMAL_ROOT = REPO / "outputs/stage2_4b_direct_preserving_reason/e16_formal_20260523"
REPORT_JSON = REPO / "reports/artifacts/2026-05-23_stage2_4b_direct_preserving_reason_e16.json"
REPORT_MD = REPO / "reports/2026-05-23_stage2_4b_direct_preserving_reason_e16.md"
SPLITS = ["test_seen", "test_unseen"]

VARIANTS = {
    "e13b_current": REPO / "outputs/stage2_4b_reason_expert/e13b_formal_20260521",
    "e15a_noreasonblock": REPO / "outputs/stage2_4b_reason_format_ablation/e15_formal_20260522/e15a_noreasonblock",
    "e15c_finalfirst": REPO / "outputs/stage2_4b_reason_format_ablation/e15_formal_20260522/e15c_finalfirst",
    "e16a_noreasonblock_directpreserve": FORMAL_ROOT / "e16a_noreasonblock_directpreserve",
    "e16c_finalfirst_directpreserve": FORMAL_ROOT / "e16c_finalfirst_directpreserve",
}


def load_pair(root, split):
    return (
        load_prediction_map(root / "forced_direct" / split / "predictions.jsonl"),
        load_prediction_map(root / "forced_reason" / split / "predictions.jsonl"),
    )


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
        "# 4B Direct-Preserving Reason E16",
        "",
        "This compares E16 direct-preserving format runs against E13B and E15. Deltas are against each system's forced-direct output.",
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
            lines.append(f"- `{row['system']}` reason-all delta A/E/T: `{delta(row)}`; reason A/E/T `{aet(row)}`.")
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
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": REPORT_JSON.as_posix(), "md": REPORT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
