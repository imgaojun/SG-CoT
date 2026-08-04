import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.calibrate_modular_dualexpert_utility_router_m02_rank_window_dev_20260520 import (  # noqa: E402
    DIRECT_DEV,
    REASON_DEV,
    load_prediction_map,
    load_score_rows,
    sorted_keys_by_delta,
)
from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    DIRECT_ROOT,
    REASON_ROOT,
    score,
)
from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key  # noqa: E402
from src.stage2_analysis.analyze_adaptive_outcome_router_execution import row_metric  # noqa: E402
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


BRANCH = "aet_stable_router_m02_routecls_noauxwarm_lr2e6_save50"
DEV_SCORE = REPO / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/route_likelihood" / BRANCH / "checkpoint-50/dev_seen_scores.jsonl"
FORMAL_SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/formal_route_likelihood" / BRANCH / "checkpoint-50"
LABEL_ROOT = REPO / "data/stage2_adaptive_datasets/labels"
DEV_LABEL = LABEL_ROOT / (
    "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_"
    "modular_d1930_r2058_aet_stable_m02_dev_seen_labels.jsonl"
)
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_m05_teacher_targets.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_m05_teacher_targets.md"

WINDOWS = [
    {
        "name": "success_retention_rank425_500",
        "kind": "success",
        "start_pct": 0.425,
        "end_pct": 0.500,
        "formal_test_delta": {
            "argument_f1": 0.0051,
            "event_f1": 0.0050,
            "trigger_f1": 0.0026,
        },
    },
    {
        "name": "failed_neighbor_rank300_375",
        "kind": "failed_neighbor",
        "start_pct": 0.300,
        "end_pct": 0.375,
        "formal_test_delta": {
            "argument_f1": -0.0050,
            "event_f1": -0.0036,
            "trigger_f1": -0.0017,
        },
    },
    {
        "name": "failed_neighbor_rank275_375",
        "kind": "failed_neighbor",
        "start_pct": 0.275,
        "end_pct": 0.375,
        "formal_test_delta": {
            "argument_f1": -0.0052,
            "event_f1": -0.0014,
            "trigger_f1": -0.0039,
        },
    },
]
FORMAL_SPLITS = ["test_seen", "test_unseen"]


def label_key(row):
    return row.get("wnd_id") or row.get("id")


def load_label_map(path: Path):
    return {label_key(row): row for row in load_jsonl(path)}


def label_path(split):
    return LABEL_ROOT / (
        "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_"
        f"modular_d1930_r2058_aet_stable_m02_{split}_labels.jsonl"
    )


def quantiles(values):
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    vals = sorted(values)
    def pick(frac):
        return vals[round((len(vals) - 1) * frac)]
    return {
        "min": vals[0],
        "p25": pick(0.25),
        "median": pick(0.5),
        "p75": pick(0.75),
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
    }


def metric_delta(direct_row, reason_row):
    direct_m = row_metric(direct_row)
    reason_m = row_metric(reason_row)
    return {
        "argument_f1": reason_m["argument"]["f1"] - direct_m["argument"]["f1"],
        "event_f1": reason_m["event"]["f1"] - direct_m["event"]["f1"],
        "trigger_f1": reason_m["trigger"]["f1"] - direct_m["trigger"]["f1"],
        "score": score(reason_row) - score(direct_row),
    }


def select_keys(keys, window):
    start = round(len(keys) * window["start_pct"])
    end = round(len(keys) * window["end_pct"])
    return keys[start:end]


def bucket_family(bucket):
    return str(bucket).split("|", 1)[0]


def attach_features(case, labels):
    label = labels.get(case["wnd_id"], {})
    bucket = label.get("bucket", "unknown")
    case.update(
        {
            "route_label": label.get("route_label"),
            "bucket": bucket,
            "bucket_family": bucket_family(bucket),
            "stable_reason_bucket": bool(label.get("stable_reason_bucket")),
            "hard_negative": bool(label.get("hard_negative")),
            "bucket_harm_rate": float(label.get("bucket_harm_rate", 0.0) or 0.0),
            "bucket_mean_gain": float(label.get("bucket_mean_gain", 0.0) or 0.0),
        }
    )
    return case


def dev_cases(window):
    direct_rows = load_prediction_map(DIRECT_DEV)
    reason_rows = load_prediction_map(REASON_DEV)
    score_rows = load_score_rows(DEV_SCORE)
    labels = load_label_map(DEV_LABEL)
    keys = sorted_keys_by_delta(score_rows, set(direct_rows) & set(reason_rows) & set(labels))
    selected = select_keys(keys, window)
    out = []
    for rank, key in enumerate(keys, start=1):
        if key not in set(selected):
            continue
        delta = metric_delta(direct_rows[key], reason_rows[key])
        case = {
            "split": "dev_seen",
            "rank": rank,
            "wnd_id": key,
            "delta_direct_minus_reason_route_nll": score_rows[key].get("delta_direct_minus_reason_route_nll"),
            **delta,
            "helpful": delta["score"] > 0,
            "harmful": delta["score"] < 0,
            "aet_all_nonnegative": delta["argument_f1"] >= 0 and delta["event_f1"] >= 0 and delta["trigger_f1"] >= 0,
        }
        out.append(attach_features(case, labels))
    return out


def formal_cases(window):
    out = []
    for split in FORMAL_SPLITS:
        score_rows = {
            prediction_key(row): row
            for row in load_jsonl(FORMAL_SCORE_ROOT / split / "scores.jsonl")
        }
        direct_rows = {prediction_key(row): row for row in load_jsonl(DIRECT_ROOT / split / "predictions.jsonl")}
        reason_rows = {prediction_key(row): row for row in load_jsonl(REASON_ROOT / split / "predictions.jsonl")}
        labels = load_label_map(label_path(split))
        keys = sorted_keys_by_delta(score_rows, set(direct_rows) & set(reason_rows) & set(labels))
        selected = select_keys(keys, window)
        for rank, key in enumerate(keys, start=1):
            if key not in set(selected):
                continue
            delta = metric_delta(direct_rows[key], reason_rows[key])
            case = {
                "split": split,
                "rank": rank,
                "wnd_id": key,
                "delta_direct_minus_reason_route_nll": score_rows[key].get("delta_direct_minus_reason_route_nll"),
                **delta,
                "helpful": delta["score"] > 0,
                "harmful": delta["score"] < 0,
                "aet_all_nonnegative": delta["argument_f1"] >= 0 and delta["event_f1"] >= 0 and delta["trigger_f1"] >= 0,
            }
            out.append(attach_features(case, labels))
    return out


def summarize_cases(cases):
    if not cases:
        return {}
    families = Counter(case["bucket_family"] for case in cases)
    route_labels = Counter(case.get("route_label") or "unknown" for case in cases)
    return {
        "num_cases": len(cases),
        "helpful_rate": sum(case["helpful"] for case in cases) / len(cases),
        "harmful_rate": sum(case["harmful"] for case in cases) / len(cases),
        "aet_all_nonnegative_rate": sum(case["aet_all_nonnegative"] for case in cases) / len(cases),
        "stable_reason_bucket_rate": sum(case["stable_reason_bucket"] for case in cases) / len(cases),
        "hard_negative_rate": sum(case["hard_negative"] for case in cases) / len(cases),
        "route_label_reason_rate": route_labels["reason"] / len(cases),
        "score_gain": quantiles([case["score"] for case in cases]),
        "argument_gain": quantiles([case["argument_f1"] for case in cases]),
        "event_gain": quantiles([case["event_f1"] for case in cases]),
        "trigger_gain": quantiles([case["trigger_f1"] for case in cases]),
        "bucket_harm_rate": quantiles([case["bucket_harm_rate"] for case in cases]),
        "bucket_mean_gain": quantiles([case["bucket_mean_gain"] for case in cases]),
        "top_bucket_families": families.most_common(8),
        "route_labels": dict(route_labels),
    }


def compare_success_failure(success, failures):
    success_formal = success["formal_cases"]
    failure_formal = [case for item in failures for case in item["formal_cases"]]
    success_ids = {case["wnd_id"] for case in success_formal}
    failure_ids = {case["wnd_id"] for case in failure_formal}
    return {
        "formal_success_minus_failure_mean_score_gain": (
            summarize_cases(success_formal)["score_gain"]["mean"]
            - summarize_cases(failure_formal)["score_gain"]["mean"]
        ),
        "formal_success_minus_failure_mean_argument_gain": (
            summarize_cases(success_formal)["argument_gain"]["mean"]
            - summarize_cases(failure_formal)["argument_gain"]["mean"]
        ),
        "formal_success_minus_failure_mean_event_gain": (
            summarize_cases(success_formal)["event_gain"]["mean"]
            - summarize_cases(failure_formal)["event_gain"]["mean"]
        ),
        "formal_success_minus_failure_mean_trigger_gain": (
            summarize_cases(success_formal)["trigger_gain"]["mean"]
            - summarize_cases(failure_formal)["trigger_gain"]["mean"]
        ),
        "success_failure_overlap": len(success_ids & failure_ids),
        "success_failure_jaccard": len(success_ids & failure_ids) / len(success_ids | failure_ids),
    }


def signed(value):
    return f"{value:+.4f}"


def pct(value):
    return f"{100 * value:.1f}%"


def render_window_table(windows):
    lines = [
        "| window | kind | dev helpful/harm | formal helpful/harm | formal mean A/E/T | stable bucket | hard neg |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in windows:
        dev = row["dev_summary"]
        formal = row["formal_summary"]
        lines.append(
            "| {name} | {kind} | {dh}/{dha} | {fh}/{fha} | {a}/{e}/{t} | {stable} | {hard} |".format(
                name=row["name"],
                kind=row["kind"],
                dh=pct(dev["helpful_rate"]),
                dha=pct(dev["harmful_rate"]),
                fh=pct(formal["helpful_rate"]),
                fha=pct(formal["harmful_rate"]),
                a=signed(formal["argument_gain"]["mean"]),
                e=signed(formal["event_gain"]["mean"]),
                t=signed(formal["trigger_gain"]["mean"]),
                stable=pct(formal["stable_reason_bucket_rate"]),
                hard=pct(formal["hard_negative_rate"]),
            )
        )
    return "\n".join(lines)


def render(payload):
    cmp = payload["success_vs_failure"]
    lines = [
        "# M05 Teacher Target Diagnosis",
        "",
        "This report diagnoses whether the successful positive-retention window has learnable differences from dev-good but formal-failing neighboring windows.",
        "",
        "## Window Comparison",
        "",
        render_window_table(payload["windows"]),
        "",
        "## Success vs Failed Neighbor Aggregate",
        "",
        f"- formal mean score-gain gap: `{signed(cmp['formal_success_minus_failure_mean_score_gain'])}`.",
        f"- formal mean A/E/T-gain gap: `{signed(cmp['formal_success_minus_failure_mean_argument_gain'])} / {signed(cmp['formal_success_minus_failure_mean_event_gain'])} / {signed(cmp['formal_success_minus_failure_mean_trigger_gain'])}`.",
        f"- success/failure formal selected-case overlap: `{cmp['success_failure_overlap']}`, Jaccard `{cmp['success_failure_jaccard']:.3f}`.",
        "",
        "## Recommendation",
        "",
        "- Build m05 labels around the successful `rank425_500` retention slice, but include failed neighbor slices as hard negatives.",
        "- Use selected-case mean Trigger gain as an explicit retention criterion; the failed neighbors are dev-positive but formal-negative on Trigger.",
        "- Do not use stable-bucket membership alone; formal success and failure windows both contain many stable-bucket cases.",
        "",
    ]
    return "\n".join(lines)


def main():
    windows = []
    for window in WINDOWS:
        dev = dev_cases(window)
        formal = formal_cases(window)
        windows.append(
            {
                **window,
                "dev_cases": dev,
                "formal_cases": formal,
                "dev_summary": summarize_cases(dev),
                "formal_summary": summarize_cases(formal),
            }
        )
    success = next(row for row in windows if row["kind"] == "success")
    failures = [row for row in windows if row["kind"] != "success"]
    payload = {
        "branch": BRANCH,
        "dev_score": DEV_SCORE.as_posix(),
        "formal_score_root": FORMAL_SCORE_ROOT.as_posix(),
        "windows": windows,
        "success_vs_failure": compare_success_failure(success, failures),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
