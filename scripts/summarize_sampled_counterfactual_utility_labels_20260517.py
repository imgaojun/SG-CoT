import argparse
import json
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def delta_line(summary):
    delta = summary["sampled_expected_routed_minus_direct"]
    return (
        f"argument `{fmt(delta['argument_f1'])}`, "
        f"event `{fmt(delta['event_f1'])}`, "
        f"trigger `{fmt(delta['trigger_f1'])}`, "
        f"score `{fmt(delta['score'])}`"
    )


def route_gate(dev_summary):
    delta = dev_summary["sampled_expected_routed_minus_direct"]
    stable_reason_rate = dev_summary["stable_reason_rate"]
    stable_reason_gain = dev_summary["stable_reason_mean_gain"]
    return {
        "dev_stable_reason_rate_gate": 0.03 <= stable_reason_rate <= 0.25,
        "dev_delta_gate": delta["argument_f1"] >= 0.0 and delta["event_f1"] >= 0.0 and delta["trigger_f1"] >= 0.0,
        "dev_stable_reason_gain_gate": stable_reason_gain is not None and stable_reason_gain >= 0.35,
    }


def baseline_summary(label_dir: Path, data_prefix: str, source: str, split: str):
    path = label_dir / f"{data_prefix}_{source}_{split}_labels.summary.json"
    if not path.exists():
        return None
    return load_json(path)


def make_report(payload):
    train = payload["splits"].get("train")
    dev = payload["splits"]["dev_seen"]
    gates = payload["gates"]
    lines = [
        "# Stage2 Sampled Counterfactual Utility K8 Label Diagnostic",
        "",
        "## Goal",
        "",
        (
            "Diagnose whether K=8 direct/reason sampling can produce route supervision that is more stable than "
            "single-output utility labels. This report is train/dev only; no formal split is touched."
        ),
        "",
        "## Label Rule",
        "",
        "- score: `argument_f1 + event_f1 + 0.25 * trigger_f1`",
        "- stable reason: reason valid rate `>= 0.875`, mean gain `>= 0.35`, pairwise win probability `>= 0.70`, trigger no-harm probability `>= 0.75`.",
        "- stable direct: reason valid rate `< 0.75`, or mean gain `<= -0.20`, or pairwise win probability `<= 0.25`, or trigger no-harm probability `< 0.50`.",
        "- otherwise: ambiguous, excluded from confident-only route supervision.",
        "",
        "## Summary",
        "",
        "| split | n | stable reason | stable direct | ambiguous | stable reason gain | routed-minus-direct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if train is not None:
        lines.append(
            f"| train | {train['num_examples']} | {train['stable_reason_count']} "
            f"({fmt(train['stable_reason_rate'])}) | {train['stable_direct_count']} "
            f"({fmt(train['stable_direct_rate'])}) | {train['ambiguous_count']} "
            f"({fmt(train['ambiguous_rate'])}) | {fmt(train['stable_reason_mean_gain'])} | {delta_line(train)} |"
        )
    lines.extend(
        [
            (
                f"| dev_seen | {dev['num_examples']} | {dev['stable_reason_count']} "
                f"({fmt(dev['stable_reason_rate'])}) | {dev['stable_direct_count']} "
                f"({fmt(dev['stable_direct_rate'])}) | {dev['ambiguous_count']} "
                f"({fmt(dev['ambiguous_rate'])}) | {fmt(dev['stable_reason_mean_gain'])} | {delta_line(dev)} |"
            ),
            "",
            "## Gate",
        ]
    )
    lines.extend(
        [
            "",
            f"- stable reason rate in `[0.03, 0.25]`: `{gates['dev_stable_reason_rate_gate']}`",
            f"- expected routed delta nonnegative on argument/event/trigger: `{gates['dev_delta_gate']}`",
            f"- stable reason mean gain `>= 0.35`: `{gates['dev_stable_reason_gain_gate']}`",
            f"- train confident-only router next: `{payload['train_confident_router_next']}`",
        ]
    )
    if train is None:
        lines.append("- train labels available: `False`; router training remains blocked until train K8 labels are built.")
    lines.extend(
        [
        "",
        "## Baseline Single-Output Labels",
        "",
        ]
    )
    baselines = payload["baseline_single_output_labels"]
    if baselines:
        lines.extend(["| source | split | reason rate | avg selected gain |", "| --- | --- | ---: | ---: |"])
        for row in baselines:
            lines.append(
                f"| {row['source']} | {row['split']} | {fmt(row['reason_rate'])} | {fmt(row['avg_selected_reason_gain'])} |"
            )
    else:
        lines.append("No baseline label summaries were found.")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Pending until sampled predictions are available.",
            "",
            "## Next",
            "",
            "- If the gate passes, build a confident-only routecls dataset from stable_reason/stable_direct labels.",
            "- If the gate fails because stable_reason is too rare, relax only one threshold at a time and regenerate this diagnostic.",
            "- If the gate fails because expected routed trigger delta is negative, keep reason labels but add a trigger-noharm calibration sweep before router training.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_summary", default=None)
    parser.add_argument("--dev_summary", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument(
        "--data_prefix",
        default="richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive",
    )
    args = parser.parse_args()

    train_summary = load_json(Path(args.train_summary)) if args.train_summary else None
    dev_summary = load_json(Path(args.dev_summary))
    gates = route_gate(dev_summary)
    train_next = train_summary is not None and all(gates.values())

    label_dir = REPO / "data/stage2_adaptive_datasets/labels"
    baselines = []
    for source in ["modular_d1930_r2058_utility_gainpos", "modular_d1930_r2058_utility_margin05"]:
        for split in ["train", "dev_seen"]:
            summary = baseline_summary(label_dir, args.data_prefix, source, split)
            if summary is None:
                continue
            baselines.append(
                {
                    "source": source,
                    "split": split,
                    "reason_rate": summary.get("reason_rate"),
                    "avg_selected_reason_gain": summary.get("avg_selected_reason_gain"),
                }
            )

    splits = {"dev_seen": dev_summary}
    if train_summary is not None:
        splits["train"] = train_summary
    payload = {
        "splits": splits,
        "gates": gates,
        "train_confident_router_next": train_next,
        "baseline_single_output_labels": baselines,
    }
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), make_report(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
