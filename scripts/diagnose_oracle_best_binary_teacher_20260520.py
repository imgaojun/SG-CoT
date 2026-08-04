#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
INPUT_CASES = REPO / "outputs/stage2_strong_system_v0_supervised_rerank_20260520/candidate_cases.jsonl"
OUTPUT_ROOT = REPO / "outputs/stage2_oracle_best_binary_teacher_20260520"
TEACHER_JSONL = OUTPUT_ROOT / "binary_teacher_cases.jsonl"
REPORT_MD = REPO / "reports/2026-05-20_stage2_oracle_best_binary_teacher.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_oracle_best_binary_teacher.json"

MARGINS = [0.0, 0.02, 0.05]
SPLITS = ["test_seen", "test_unseen"]
ALL_SPLITS = ["test", "test_seen", "test_unseen"]
METRICS = ["json_valid_rate", "argument_f1", "event_f1", "trigger_f1", "score"]
DIRECT = "direct_modular_d1930"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def signed(value):
    return f"{value:+.4f}"


def pct(value):
    return f"{value:.1%}"


def group_cases(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["key"])].append(row)
    cases = []
    for (split, key), candidates in sorted(grouped.items()):
        by_name = {row["candidate"]: row for row in candidates}
        if DIRECT not in by_name:
            raise ValueError(f"missing direct candidate for {split}/{key}")
        non_direct = [row for row in candidates if row["candidate"] != DIRECT]
        if not non_direct:
            raise ValueError(f"missing non-direct candidates for {split}/{key}")
        direct = by_name[DIRECT]
        best_non_direct = max(
            non_direct,
            key=lambda row: (
                row["metrics"]["score"],
                row["metrics"]["event_f1"],
                row["metrics"]["argument_f1"],
                row["metrics"]["trigger_f1"],
            ),
        )
        oracle_best = max(
            candidates,
            key=lambda row: (
                row["metrics"]["score"],
                row["metrics"]["event_f1"],
                row["metrics"]["argument_f1"],
                row["metrics"]["trigger_f1"],
            ),
        )
        cases.append(
            {
                "split": split,
                "key": key,
                "direct": direct,
                "best_non_direct": best_non_direct,
                "oracle_best": oracle_best,
                "gain_best_non_direct": best_non_direct["metrics"]["score"] - direct["metrics"]["score"],
                "gain_oracle_best": oracle_best["metrics"]["score"] - direct["metrics"]["score"],
            }
        )
    return cases


def avg_metrics(cases, selector):
    selected = [selector(case)["metrics"] for case in cases]
    return {metric: mean(row[metric] for row in selected) for metric in METRICS}


def direct_metrics(cases):
    return avg_metrics(cases, lambda case: case["direct"])


def summarize_cases(cases, margin):
    direct = direct_metrics(cases)
    selected = []
    reason_cases = []
    for case in cases:
        route_label = "reason" if case["gain_best_non_direct"] >= margin else "direct"
        choice = case["best_non_direct"] if route_label == "reason" else case["direct"]
        selected.append(choice)
        if route_label == "reason":
            reason_cases.append(case)
    summary = {metric: mean(row["metrics"][metric] for row in selected) for metric in METRICS}
    delta = {metric: summary[metric] - direct[metric] for metric in METRICS}
    candidate_counts = Counter(case["best_non_direct"]["candidate"] for case in reason_cases)
    harm_count = sum(1 for case in reason_cases if case["gain_best_non_direct"] < -1e-12)
    return {
        "num_examples": len(cases),
        "margin": margin,
        "reason_count": len(reason_cases),
        "reason_rate": len(reason_cases) / len(cases) if cases else 0.0,
        "summary": summary,
        "direct": direct,
        "delta_vs_direct": delta,
        "selected_harm_count": harm_count,
        "selected_harm_rate": harm_count / len(reason_cases) if reason_cases else 0.0,
        "selected_gain_mean": mean(case["gain_best_non_direct"] for case in reason_cases) if reason_cases else 0.0,
        "best_non_direct_candidate_counts": dict(candidate_counts),
    }


def summarize_oracle(cases):
    direct = direct_metrics(cases)
    summary = avg_metrics(cases, lambda case: case["oracle_best"])
    delta = {metric: summary[metric] - direct[metric] for metric in METRICS}
    non_direct_cases = [case for case in cases if case["oracle_best"]["candidate"] != DIRECT]
    return {
        "num_examples": len(cases),
        "reason_count": len(non_direct_cases),
        "reason_rate": len(non_direct_cases) / len(cases) if cases else 0.0,
        "summary": summary,
        "direct": direct,
        "delta_vs_direct": delta,
        "selected_harm_count": 0,
        "selected_harm_rate": 0.0,
        "selected_gain_mean": mean(case["gain_oracle_best"] for case in non_direct_cases) if non_direct_cases else 0.0,
        "candidate_counts": dict(Counter(case["oracle_best"]["candidate"] for case in cases)),
    }


def aggregate(rows, margin):
    total = sum(row["num_examples"] for row in rows)
    out = {
        "split": "test",
        "margin": margin,
        "num_examples": total,
        "reason_count": sum(row["reason_count"] for row in rows),
        "selected_harm_count": sum(row["selected_harm_count"] for row in rows),
        "best_non_direct_candidate_counts": dict(
            sum((Counter(row.get("best_non_direct_candidate_counts", {})) for row in rows), Counter())
        ),
    }
    out["reason_rate"] = out["reason_count"] / total if total else 0.0
    out["selected_harm_rate"] = out["selected_harm_count"] / out["reason_count"] if out["reason_count"] else 0.0
    out["selected_gain_mean"] = (
        sum(row["selected_gain_mean"] * row["reason_count"] for row in rows) / out["reason_count"]
        if out["reason_count"]
        else 0.0
    )
    for name in ["summary", "direct"]:
        out[name] = {
            metric: sum(row[name][metric] * row["num_examples"] for row in rows) / total
            for metric in METRICS
        }
    out["delta_vs_direct"] = {metric: out["summary"][metric] - out["direct"][metric] for metric in METRICS}
    return out


def aggregate_oracle(rows):
    total = sum(row["num_examples"] for row in rows)
    out = {
        "split": "test",
        "num_examples": total,
        "reason_count": sum(row["reason_count"] for row in rows),
        "selected_harm_count": 0,
        "selected_harm_rate": 0.0,
        "candidate_counts": dict(sum((Counter(row["candidate_counts"]) for row in rows), Counter())),
    }
    out["reason_rate"] = out["reason_count"] / total if total else 0.0
    out["selected_gain_mean"] = (
        sum(row["selected_gain_mean"] * row["reason_count"] for row in rows) / out["reason_count"]
        if out["reason_count"]
        else 0.0
    )
    for name in ["summary", "direct"]:
        out[name] = {
            metric: sum(row[name][metric] * row["num_examples"] for row in rows) / total
            for metric in METRICS
        }
    out["delta_vs_direct"] = {metric: out["summary"][metric] - out["direct"][metric] for metric in METRICS}
    return out


def build_teacher_rows(cases):
    rows = []
    for case in cases:
        base = {
            "split": case["split"],
            "key": case["key"],
            "direct_score": case["direct"]["metrics"]["score"],
            "best_non_direct_score": case["best_non_direct"]["metrics"]["score"],
            "best_non_direct_candidate": case["best_non_direct"]["candidate"],
            "gain_best_non_direct": case["gain_best_non_direct"],
            "oracle_best_candidate": case["oracle_best"]["candidate"],
            "oracle_best_score": case["oracle_best"]["metrics"]["score"],
            "gain_oracle_best": case["gain_oracle_best"],
        }
        for margin in MARGINS:
            base[f"label_margin_{margin:.2f}"] = "reason" if case["gain_best_non_direct"] >= margin else "direct"
        rows.append(base)
    return rows


def evaluate():
    cases = group_cases(load_jsonl(INPUT_CASES))
    by_split = defaultdict(list)
    for case in cases:
        by_split[case["split"]].append(case)

    if len(by_split["test_seen"]) != 361 or len(by_split["test_unseen"]) != 82:
        raise ValueError(
            f"unexpected split sizes: test_seen={len(by_split['test_seen'])}, test_unseen={len(by_split['test_unseen'])}"
        )

    teacher_rows = build_teacher_rows(cases)
    write_jsonl(TEACHER_JSONL, teacher_rows)

    margin_rows = []
    for margin in MARGINS:
        split_rows = []
        for split in SPLITS:
            row = summarize_cases(by_split[split], margin)
            row["split"] = split
            split_rows.append(row)
        margin_rows.append(aggregate(split_rows, margin))
        margin_rows.extend(split_rows)

    oracle_split_rows = []
    for split in SPLITS:
        row = summarize_oracle(by_split[split])
        row["split"] = split
        oracle_split_rows.append(row)
    oracle_rows = [aggregate_oracle(oracle_split_rows)] + oracle_split_rows

    return {
        "id": "2026-05-20_stage2_oracle_best_binary_teacher",
        "input_cases": INPUT_CASES.as_posix(),
        "teacher_jsonl": TEACHER_JSONL.as_posix(),
        "report_md": REPORT_MD.as_posix(),
        "report_json": REPORT_JSON.as_posix(),
        "margins": MARGINS,
        "margin_rows": margin_rows,
        "oracle_rows": oracle_rows,
        "teacher_label_counts": {
            f"margin_{margin:.2f}": dict(Counter(row[f"label_margin_{margin:.2f}"] for row in teacher_rows))
            for margin in MARGINS
        },
    }


def metric_cell(row):
    return "{}/{}/{}/{}".format(
        fmt(row["summary"]["argument_f1"]),
        fmt(row["summary"]["event_f1"]),
        fmt(row["summary"]["trigger_f1"]),
        fmt(row["summary"]["score"]),
    )


def delta_cell(row):
    return "{}/{}/{}/{}".format(
        signed(row["delta_vs_direct"]["argument_f1"]),
        signed(row["delta_vs_direct"]["event_f1"]),
        signed(row["delta_vs_direct"]["trigger_f1"]),
        signed(row["delta_vs_direct"]["score"]),
    )


def render_margin_table(rows):
    lines = [
        "| margin | split | reason rate | A/E/T/Score | delta vs direct A/E/T/Score | harm | selected gain | best non-direct candidates |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {margin} | `{split}` | {rate} | {metrics} | {delta} | {harm} | {gain} | {counts} |".format(
                margin=fmt(row["margin"], 2),
                split=row["split"],
                rate=pct(row["reason_rate"]),
                metrics=metric_cell(row),
                delta=delta_cell(row),
                harm=pct(row["selected_harm_rate"]),
                gain=signed(row["selected_gain_mean"]),
                counts=", ".join(f"{key}:{value}" for key, value in sorted(row["best_non_direct_candidate_counts"].items())),
            )
        )
    return "\n".join(lines)


def render_oracle_table(rows):
    lines = [
        "| split | non-direct rate | A/E/T/Score | delta vs direct A/E/T/Score | candidates |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| `{split}` | {rate} | {metrics} | {delta} | {counts} |".format(
                split=row["split"],
                rate=pct(row["reason_rate"]),
                metrics=metric_cell(row),
                delta=delta_cell(row),
                counts=", ".join(f"{key}:{value}" for key, value in sorted(row["candidate_counts"].items())),
            )
        )
    return "\n".join(lines)


def render_report(payload):
    test_rows = [row for row in payload["margin_rows"] if row["split"] == "test"]
    oracle_test = next(row for row in payload["oracle_rows"] if row["split"] == "test")
    best = max(test_rows, key=lambda row: row["delta_vs_direct"]["score"])
    retained = (
        best["delta_vs_direct"]["score"] / oracle_test["delta_vs_direct"]["score"]
        if oracle_test["delta_vs_direct"]["score"]
        else 0.0
    )
    lines = [
        "# Oracle-Best Binary Teacher Diagnostic",
        "",
        "This compresses the strong-system oracle-best pool into a binary `direct` vs `reason-like` teacher. It is an upper-bound diagnostic; no router is trained here.",
        "",
        "## Binary Teacher Replay",
        "",
        render_margin_table(payload["margin_rows"]),
        "",
        "## Four-Candidate Oracle Reference",
        "",
        render_oracle_table(payload["oracle_rows"]),
        "",
        "## Reading",
        "",
        f"- Best binary teacher margin on test: `{best['margin']:.2f}` with score delta `{signed(best['delta_vs_direct']['score'])}` and reason rate `{pct(best['reason_rate'])}`.",
        f"- Four-candidate oracle test score delta: `{signed(oracle_test['delta_vs_direct']['score'])}`.",
        f"- Best binary teacher retains `{pct(retained)}` of the four-candidate oracle score delta.",
        f"- Teacher labels: `{json.dumps(payload['teacher_label_counts'], sort_keys=True)}`.",
        "",
        "## Artifacts",
        "",
        f"- teacher cases: `{payload['teacher_jsonl']}`",
        f"- JSON: `{payload['report_json']}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    payload = evaluate()
    write_json(REPORT_JSON, payload)
    write_text(REPORT_MD, render_report(payload))
    print(json.dumps({"report_md": REPORT_MD.as_posix(), "report_json": REPORT_JSON.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
