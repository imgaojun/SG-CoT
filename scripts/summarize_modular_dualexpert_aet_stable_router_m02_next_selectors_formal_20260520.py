import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    DIRECT_ROOT,
    REASON_ROOT,
    aggregate_test,
    load_prediction_map,
    load_score_rows,
    render_table,
    row_metric,
    score,
    sorted_keys_by_delta,
    summarize_metrics,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


BRANCH = "aet_stable_router_m02_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/formal_route_likelihood"
DEV_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_next_selectors_dev.json"
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_next_selectors_formal.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_next_selectors_formal.md"
LABEL_ROOT = REPO / "data/stage2_adaptive_datasets/labels"
SPLITS = ["test_seen", "test_unseen"]


def label_path(split):
    return LABEL_ROOT / (
        "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_"
        f"modular_d1930_r2058_aet_stable_m02_{split}_labels.jsonl"
    )


def label_key(row):
    return row.get("wnd_id") or row.get("id")


def load_label_map(path: Path):
    return {label_key(row): row for row in load_jsonl(path)}


def policy_from_dev(key, row):
    return {
        "name": f"{key}_{row['name']}",
        "branch": BRANCH,
        "checkpoint": row["checkpoint"],
        "policy_family": row["policy_family"],
        "group_rules": row["group_rules"],
        "source": key,
    }


def load_policies():
    dev = json.loads(DEV_JSON.read_text(encoding="utf-8"))
    policies = []
    seen = set()
    for key in ["balanced_candidate", "source_aware_candidate", "positive_retention_candidate"]:
        row = dev[key]
        ident = (row["checkpoint"], json.dumps(row["group_rules"], sort_keys=True))
        if ident in seen:
            continue
        seen.add(ident)
        policies.append(policy_from_dev(key, row))
    return policies


def selected_keys_for_policy(policy, keys, score_rows, label_rows):
    selected = set()
    for rule in policy["group_rules"]:
        group = rule["group"]
        if group == "all":
            group_keys = list(keys)
        elif group == "stable_reason_bucket=true":
            group_keys = [
                key
                for key in keys
                if label_rows.get(key, {}).get("stable_reason_bucket") is True
            ]
        elif rule.get("action") == "direct":
            continue
        else:
            raise ValueError(f"unsupported group rule: {rule}")
        group_keys = [key for key in group_keys if key in score_rows]
        if not group_keys:
            continue
        start = round(len(group_keys) * rule["start_pct"])
        end = round(len(group_keys) * rule["end_pct"])
        selected.update(group_keys[start:end])
    return selected


def evaluate(policy, split):
    score_path = SCORE_ROOT / policy["branch"] / policy["checkpoint"] / split / "scores.jsonl"
    direct_path = DIRECT_ROOT / split / "predictions.jsonl"
    reason_path = REASON_ROOT / split / "predictions.jsonl"
    labels_path = label_path(split)
    for path in [score_path, direct_path, reason_path, labels_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    score_rows = load_score_rows(score_path)
    direct_rows = load_prediction_map(direct_path)
    reason_rows = load_prediction_map(reason_path)
    label_rows = load_label_map(labels_path)
    keys = sorted_keys_by_delta(score_rows, set(direct_rows) & set(reason_rows) & set(label_rows))
    reason_keys = selected_keys_for_policy(policy, keys, score_rows, label_rows)

    routed_metrics = []
    direct_metrics = []
    reason_metrics = []
    selected_gains = []
    selected_examples = []
    for rank, key in enumerate(keys, start=1):
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        chosen = reason_row if key in reason_keys else direct_row
        routed_metrics.append(row_metric(chosen))
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        if key in reason_keys:
            gain = score(reason_row) - score(direct_row)
            selected_gains.append(gain)
            selected_examples.append(
                {
                    "rank": rank,
                    "wnd_id": key,
                    "stable_reason_bucket": label_rows.get(key, {}).get("stable_reason_bucket"),
                    "delta_direct_minus_reason_route_nll": score_rows[key].get(
                        "delta_direct_minus_reason_route_nll"
                    ),
                    "score_gain": gain,
                }
            )

    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    routed = summarize_metrics(routed_metrics)
    return {
        "policy": policy["name"],
        "branch": policy["branch"],
        "checkpoint": policy["checkpoint"],
        "source": policy["source"],
        "policy_family": policy["policy_family"],
        "split": split,
        "num_examples": len(keys),
        "group_rules": policy["group_rules"],
        "pred_reason_count": len(reason_keys),
        "pred_reason_rate": len(reason_keys) / len(keys) if keys else 0.0,
        "selected_reason_avg_score_gain": (
            sum(selected_gains) / len(selected_gains) if selected_gains else 0.0
        ),
        "direct": direct,
        "forced_reason_all": reason,
        "routed": routed,
        "routed_minus_direct": {
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
        },
        "routed_minus_reason_all": {
            "trigger_f1": routed["trigger_f1"] - reason["trigger_f1"],
            "argument_f1": routed["argument_f1"] - reason["argument_f1"],
            "event_f1": routed["event_f1"] - reason["event_f1"],
        },
        "selected_examples": sorted(
            selected_examples,
            key=lambda row: row["delta_direct_minus_reason_route_nll"]
            if row["delta_direct_minus_reason_route_nll"] is not None
            else float("-inf"),
            reverse=True,
        )[:20],
    }


def render_report(payload):
    rows = sorted(payload["results"], key=lambda row: (row["policy"], row["split"]))
    lines = [
        "# A/E/T Stable Router M02 Next Selectors Formal Replay",
        "",
        "This report applies dev-locked next-selector policies to existing formal route-NLL scores. Formal labels are not used for policy selection.",
        "",
        "## Results",
        "",
        render_table(rows),
        "",
        "## Reading",
        "",
    ]
    for row in rows:
        if row["split"] != "test":
            continue
        d = row["routed_minus_direct"]
        lines.append(
            f"- `{row['policy']}` on `test`: reason rate `{row['pred_reason_rate']:.1%}`, "
            f"A/E/T delta `{d['argument_f1']:+.4f}/{d['event_f1']:+.4f}/{d['trigger_f1']:+.4f}`."
        )
    return "\n".join(lines) + "\n"


def main():
    policies = load_policies()
    import scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 as base  # noqa: E402

    base.POLICIES = policies
    split_rows = []
    for policy in policies:
        for split in SPLITS:
            split_rows.append(evaluate(policy, split))
    rows = split_rows + aggregate_test(split_rows)
    payload = {
        "score_root": SCORE_ROOT.as_posix(),
        "direct_root": DIRECT_ROOT.as_posix(),
        "reason_root": REASON_ROOT.as_posix(),
        "dev_json": DEV_JSON.as_posix(),
        "policies": policies,
        "results": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
