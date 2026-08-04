import json
import sys
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_4b_reason_expert_e13b_20260521 import evaluate_policy  # noqa: E402
from scripts.summarize_4b_reason_format_ablation_e15_20260522 import aggregate  # noqa: E402
from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import load_prediction_map  # noqa: E402


ROOT = REPO / "outputs/stage2_4b_reason_checkpoint_diagnosis/e17_formal_20260523"
OUT_JSON = REPO / "reports/artifacts/2026-05-23_stage2_4b_e15_checkpoint_diagnosis_e17.json"
OUT_MD = REPO / "reports/2026-05-23_stage2_4b_e15_checkpoint_diagnosis_e17.md"
VARIANTS = ["e15a_noreasonblock", "e15c_finalfirst"]
CHECKPOINTS = [386, 772, 1158, 1544]
SPLITS = ["test_seen", "test_unseen"]


def load_pair(root, split):
    return (
        load_prediction_map(root / "forced_direct" / split / "predictions.jsonl"),
        load_prediction_map(root / "forced_reason" / split / "predictions.jsonl"),
    )


def fmt(value):
    return f"{value:.4f}"


def signed(value):
    return f"{value:+.4f}"


def aet(row):
    m = row["routed"]
    return f"{fmt(m['argument_f1'])} / {fmt(m['event_f1'])} / {fmt(m['trigger_f1'])}"


def delta(row):
    d = row["routed_minus_direct"]
    return f"{signed(d['argument_f1'])} / {signed(d['event_f1'])} / {signed(d['trigger_f1'])}"


def render(payload):
    rows = [row for row in payload["results"] if row["split"] == "test"]
    lines = [
        "# E17 E15 Checkpoint Diagnosis",
        "",
        "Formal forced-direct/forced-reason sweep over E15A/E15C checkpoints. Deltas are reason-all minus direct.",
        "",
        "| system | policy | A/E/T | delta A/E/T |",
        "|---|---|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["system"], item["policy"])):
        lines.append(f"| `{row['system']}` | `{row['policy']}` | {aet(row)} | {delta(row)} |")
    lines.extend(["", "## Best By Metric", ""])
    reason_rows = [row for row in rows if row["policy"] == "reason_all"]
    for metric in ["argument_f1", "event_f1", "trigger_f1"]:
        best = max(reason_rows, key=lambda row: row["routed"][metric], default=None)
        if best:
            lines.append(f"- best reason `{metric}`: `{best['system']}` with A/E/T `{aet(best)}`, delta `{delta(best)}`.")
    best_sum = max(reason_rows, key=lambda row: row["routed"]["argument_f1"] + row["routed"]["event_f1"] + row["routed"]["trigger_f1"], default=None)
    if best_sum:
        lines.append(f"- best reason A+E+T sum: `{best_sum['system']}` with A/E/T `{aet(best_sum)}`, delta `{delta(best_sum)}`.")
    return "\n".join(lines) + "\n"


def main():
    rows = []
    for variant in VARIANTS:
        for ckpt in CHECKPOINTS:
            system = f"{variant}_checkpoint-{ckpt}"
            root = ROOT / variant / f"checkpoint-{ckpt}"
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
    payload = {"root": ROOT.as_posix(), "results": rows}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
