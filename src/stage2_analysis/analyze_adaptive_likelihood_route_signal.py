import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import (  # noqa: E402
    DIRECT_EVAL_JSONL,
    build_feature_map,
    prediction_key,
    score,
)
from src.stage2_quality_validation.eval_adapter_generation import load_jsonl  # noqa: E402


EVAL_JSONL = {
    **DIRECT_EVAL_JSONL,
    "dev_seen": "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_dev_seen_pos.jsonl",
}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_map(path: Path, key_field=None):
    rows = load_jsonl(path)
    out = {}
    for idx, row in enumerate(rows):
        key = row.get(key_field) if key_field else prediction_key(row)
        if key is None:
            key = str(idx)
        out[key] = row
    return out


def metric(row):
    return {
        "trigger_f1": float(row.get("trigger_f1", 0.0) or 0.0),
        "argument_f1": float(row.get("argument_f1", 0.0) or 0.0),
        "event_f1": float(row.get("event_f1", 0.0) or 0.0),
        "score": score(row),
    }


def mean(vals):
    values = [v for v in vals if v is not None]
    return sum(values) / len(values) if values else 0.0


def auc(rows, score_key):
    pos = [row[score_key] for row in rows if row["reason_helpful"] and row[score_key] is not None]
    neg = [row[score_key] for row in rows if not row["reason_helpful"] and row[score_key] is not None]
    if not pos or not neg:
        return 0.0
    wins = ties = total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / total


def selected(rows, score_key, budget):
    cap = round(len(rows) * budget)
    ranked = [row for row in rows if row[score_key] is not None]
    ranked.sort(key=lambda row: (row[score_key], row["wnd_id"]), reverse=True)
    return {row["wnd_id"] for row in ranked[:cap]}


def oracle_selected(rows, budget):
    cap = round(len(rows) * budget)
    ranked = [row for row in rows if row["reason_gain"] > 1e-9]
    ranked.sort(key=lambda row: (row["reason_gain"], row["wnd_id"]), reverse=True)
    return {row["wnd_id"] for row in ranked[:cap]}


def summarize_route(rows, selected_ids):
    routed = []
    for row in rows:
        routed.append(row["reason_metric"] if row["wnd_id"] in selected_ids else row["direct_metric"])
    direct = [row["direct_metric"] for row in rows]
    helpful = {row["wnd_id"] for row in rows if row["reason_helpful"]}
    chosen_helpful = selected_ids & helpful
    return {
        "argument_f1": mean([row["argument_f1"] for row in routed]),
        "event_f1": mean([row["event_f1"] for row in routed]),
        "trigger_f1": mean([row["trigger_f1"] for row in routed]),
        "argument_gain": mean([row["argument_f1"] for row in routed]) - mean([row["argument_f1"] for row in direct]),
        "event_gain": mean([row["event_f1"] for row in routed]) - mean([row["event_f1"] for row in direct]),
        "selected_count": len(selected_ids),
        "precision": len(chosen_helpful) / len(selected_ids) if selected_ids else 0.0,
        "recall": len(chosen_helpful) / len(helpful) if helpful else 0.0,
    }


def analyze_split(scores_path: Path, direct_path: Path, reason_path: Path, split: str, schema_path: Path):
    scores = load_map(scores_path, "wnd_id")
    direct = load_map(direct_path)
    reason = load_map(reason_path)
    features = build_feature_map(Path(EVAL_JSONL[split]), schema_path)
    rows = []
    for key in sorted(set(scores) & set(direct) & set(reason)):
        d_metric = metric(direct[key])
        r_metric = metric(reason[key])
        reason_gain = r_metric["score"] - d_metric["score"]
        feat = features.get(key, {})
        rows.append(
            {
                "wnd_id": key,
                "direct_metric": d_metric,
                "reason_metric": r_metric,
                "reason_gain": reason_gain,
                "reason_helpful": reason_gain > 1e-9,
                "delta_final_nll": scores[key].get("delta_final_nll"),
                "hardconf_score": feat.get("hardconf_score"),
                "role_boundary_score": (
                    0.30 * feat.get("confusion_norm", 0.0)
                    + 0.30 * feat.get("role_signature_rarity", 0.0)
                    + 0.25 * feat.get("role_density_norm", 0.0)
                    + 0.15 * feat.get("multi_event_or_multi_trigger", 0.0)
                ),
            }
        )
    selectors = {}
    for score_key in ["delta_final_nll", "hardconf_score", "role_boundary_score"]:
        selectors[score_key] = {
            f"cap{int(b * 100)}": summarize_route(rows, selected(rows, score_key, b))
            for b in [0.10, 0.15]
        }
    oracle = {
        f"cap{int(b * 100)}": summarize_route(rows, oracle_selected(rows, b))
        for b in [0.10, 0.15]
    }
    return {
        "split": split,
        "num_examples": len(rows),
        "reason_helpful_count": sum(1 for row in rows if row["reason_helpful"]),
        "reason_helpful_rate": mean([1.0 if row["reason_helpful"] else 0.0 for row in rows]),
        "auc": {
            "delta_final_nll": auc(rows, "delta_final_nll"),
            "hardconf_score": auc(rows, "hardconf_score"),
            "role_boundary_score": auc(rows, "role_boundary_score"),
        },
        "oracle": oracle,
        "selectors": selectors,
    }


def markdown(payload):
    lines = [
        "# Adaptive Likelihood Route Signal Analysis",
        "",
        "## Summary",
        "",
        "| split | examples | helpful rate | AUC likelihood/hardconf/role_boundary | oracle15 gain arg/event | likelihood15 gain arg/event P/R | hardconf15 gain arg/event P/R |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in payload["splits"].items():
        oracle = row["oracle"]["cap15"]
        like = row["selectors"]["delta_final_nll"]["cap15"]
        hard = row["selectors"]["hardconf_score"]["cap15"]
        lines.append(
            "| `{}` | {} | {:.3f} | {:.3f}/{:.3f}/{:.3f} | {:+.4f}/{:+.4f} | {:+.4f}/{:+.4f} {:.3f}/{:.3f} | {:+.4f}/{:+.4f} {:.3f}/{:.3f} |".format(
                split,
                row["num_examples"],
                row["reason_helpful_rate"],
                row["auc"]["delta_final_nll"],
                row["auc"]["hardconf_score"],
                row["auc"]["role_boundary_score"],
                oracle["argument_gain"],
                oracle["event_gain"],
                like["argument_gain"],
                like["event_gain"],
                like["precision"],
                like["recall"],
                hard["argument_gain"],
                hard["event_gain"],
                hard["precision"],
                hard["recall"],
            )
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append("- `delta_final_nll` is useful only if it beats hardconf/role-boundary on AUC or top-k route gain.")
    lines.append("- If likelihood underperforms, the gold-plan teacher-forcing signal is not sufficient and should not be used for final labels without OOF/generated-plan checks.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_path", default="data/schema/richere-en.event_schema.json")
    parser.add_argument("--scores_prefix", required=True)
    parser.add_argument("--direct_prefix", required=True)
    parser.add_argument("--reason_prefix", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    splits = {}
    for split in ["dev_seen", "test_seen", "test_unseen", "test"]:
        score_path = Path(f"{args.scores_prefix}_{split}.jsonl")
        direct_path = Path(f"{args.direct_prefix}_{split}.jsonl")
        reason_path = Path(f"{args.reason_prefix}_{split}.jsonl")
        if not (score_path.exists() and direct_path.exists() and reason_path.exists()):
            continue
        if split not in EVAL_JSONL:
            continue
        splits[split] = analyze_split(score_path, direct_path, reason_path, split, Path(args.schema_path))
    payload = {"splits": splits}
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), markdown(payload))
    print(json.dumps({"output_md": args.output_md, "output_json": args.output_json}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
