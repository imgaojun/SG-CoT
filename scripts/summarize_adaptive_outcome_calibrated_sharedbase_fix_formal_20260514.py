import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "configs/generated/stage2_adaptive/richere_qwen3_1_7b_adaptive_outcome_calibrated_sharedbase_fix_formal_manifest.json"
DEFAULT_OUTPUT_JSON = REPO / "reports/artifacts/2026-05-14_stage2_adaptive_outcome_calibrated_sharedbase_fix_formal_summary.json"
DEFAULT_OUTPUT_MD = REPO / "reports/2026-05-14_stage2_adaptive_outcome_calibrated_sharedbase_fix_formal_summary.md"


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


def metric(summary, key: str):
    if not summary:
        return None
    return summary.get(key, summary.get(f"final_{key}"))


def fmt(value, digits=4):
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def signed(value):
    if value is None:
        return "missing"
    return f"{value:+.4f}"


def branch_from_tag(tag: str):
    return tag.removesuffix("__sharedbase_fix_execution_gate")


def split_mode(name: str):
    if "/" not in name:
        return name, ""
    mode, split = name.split("/", 1)
    return mode, split


def none_sub(left, right):
    if left is None or right is None:
        return None
    return left - right


def summarize_run(run):
    selection_path = REPO / run["selection_summary"]
    selection = load_json(selection_path) if selection_path.exists() else None
    branch = branch_from_tag(run["tag"])
    rows = []
    by_split_mode = {}

    for eval_spec in run.get("evals", []):
        mode, split = split_mode(eval_spec["name"])
        summary_path = REPO / eval_spec["output_dir"] / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else None
        row = {
            "branch": branch,
            "mode": mode,
            "split": split,
            "summary_path": summary_path.as_posix(),
            "exists": summary is not None,
            "num_examples": metric(summary, "num_examples"),
            "argument_f1": metric(summary, "argument_f1"),
            "event_f1": metric(summary, "event_f1"),
            "trigger_f1": metric(summary, "trigger_f1"),
            "route_reason_rate": metric(summary, "route_reason_rate"),
            "json_valid_rate": metric(summary, "json_valid_rate"),
            "avg_latency_sec": metric(summary, "avg_latency_sec"),
        }
        rows.append(row)
        by_split_mode[(split, mode)] = row

    deltas = []
    splits = sorted({row["split"] for row in rows})
    for split in splits:
        direct = by_split_mode.get((split, "forced_direct"))
        reason = by_split_mode.get((split, "forced_reason"))
        free = by_split_mode.get((split, "free_route"))
        if direct and reason:
            deltas.append(
                {
                    "branch": branch,
                    "split": split,
                    "comparison": "forced_reason_minus_forced_direct",
                    "argument_f1_delta": none_sub(reason["argument_f1"], direct["argument_f1"]),
                    "event_f1_delta": none_sub(reason["event_f1"], direct["event_f1"]),
                    "trigger_f1_delta": none_sub(reason["trigger_f1"], direct["trigger_f1"]),
                }
            )
        if direct and free:
            deltas.append(
                {
                    "branch": branch,
                    "split": split,
                    "comparison": "free_route_minus_forced_direct",
                    "argument_f1_delta": none_sub(free["argument_f1"], direct["argument_f1"]),
                    "event_f1_delta": none_sub(free["event_f1"], direct["event_f1"]),
                    "trigger_f1_delta": none_sub(free["trigger_f1"], direct["trigger_f1"]),
                }
            )

    selected = selection.get("frontier_selected_checkpoint") if selection else None
    return {
        "branch": branch,
        "tag": run["tag"],
        "selection_summary": selection_path.as_posix(),
        "selected_checkpoint": (selection or {}).get("best", {}).get("checkpoint_tag"),
        "gate_pass": bool((selected or {}).get("gate_pass")),
        "dev_gate": selected,
        "rows": rows,
        "deltas": deltas,
    }


def table(rows, headers, cells):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cells(row)) + " |")
    return "\n".join(lines)


def render_markdown(payload):
    lines = [
        "# Adaptive Outcome-Calibrated Shared-Base Fix Formal Summary",
        "",
        "## Status",
        "",
        f"- Manifest: `{payload['manifest']}`",
        f"- Runs included: `{len(payload['runs'])}`",
        f"- Skipped branches: `{len(payload['skipped'])}`",
        f"- Artifact: `{payload['artifact_json']}`",
        "",
    ]

    if payload["skipped"]:
        lines.extend(["## Skipped", ""])
        lines.append(
            table(
                payload["skipped"],
                ["branch", "reason"],
                lambda row: [f"`{row.get('branch', '')}`", row.get("reason", "")],
            )
        )
        lines.append("")

    if not payload["runs"]:
        lines.extend(["## Reading", "", "- No branch passed the shared-base fix dev gate, so formal evaluation was not launched."])
        return "\n".join(lines) + "\n"

    lines.extend(["## Dev Gate Selection", ""])
    lines.append(
        table(
            payload["runs"],
            ["branch", "checkpoint", "gate", "dev reason rate", "dev routed delta arg/event"],
            lambda run: [
                f"`{run['branch']}`",
                f"`{run.get('selected_checkpoint') or 'missing'}`",
                "`pass`" if run.get("gate_pass") else "`fail`",
                fmt(((run.get("dev_gate") or {}).get("free_summary") or {}).get("route_reason_rate")),
                "{}/{}".format(
                    signed((((run.get("dev_gate") or {}).get("execution") or {}).get("routed_delta_vs_direct") or {}).get("argument_f1")),
                    signed((((run.get("dev_gate") or {}).get("execution") or {}).get("routed_delta_vs_direct") or {}).get("event_f1")),
                ),
            ],
        )
    )
    lines.append("")

    free_rows = [row for run in payload["runs"] for row in run["rows"] if row["mode"] == "free_route"]
    lines.extend(["## Free-Route Formal", ""])
    lines.append(
        table(
            free_rows,
            ["branch", "split", "arg", "event", "trigger", "reason rate", "json"],
            lambda row: [
                f"`{row['branch']}`",
                row["split"],
                fmt(row["argument_f1"]),
                fmt(row["event_f1"]),
                fmt(row["trigger_f1"]),
                fmt(row["route_reason_rate"]),
                fmt(row["json_valid_rate"]),
            ],
        )
    )
    lines.append("")

    delta_rows = [row for run in payload["runs"] for row in run["deltas"]]
    lines.extend(["## Formal Deltas", ""])
    lines.append(
        table(
            delta_rows,
            ["branch", "split", "comparison", "delta arg", "delta event", "delta trigger"],
            lambda row: [
                f"`{row['branch']}`",
                row["split"],
                row["comparison"],
                signed(row["argument_f1_delta"]),
                signed(row["event_f1_delta"]),
                signed(row["trigger_f1_delta"]),
            ],
        )
    )
    lines.append("")
    lines.extend(
        [
            "## Reading",
            "",
            "- Treat `free_route_minus_forced_direct` as the main formal adaptive signal.",
            "- JSON validity remains a hard precondition for interpreting extraction F1.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON.as_posix())
    parser.add_argument("--output_md", default=DEFAULT_OUTPUT_MD.as_posix())
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    manifest = load_json(manifest_path)
    runs = [summarize_run(run) for run in manifest.get("runs", [])]
    payload = {
        "manifest": manifest_path.as_posix(),
        "artifact_json": output_json.as_posix(),
        "runs": runs,
        "skipped": manifest.get("skipped", []),
    }
    write_json(output_json, payload)
    write_text(output_md, render_markdown(payload))
    print(json.dumps({"output_json": output_json.as_posix(), "output_md": output_md.as_posix(), "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
