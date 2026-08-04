import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
DEFAULT_BRANCH = "sampled_k8_ckpt258_confident_routecls_noauxwarm_lr2e6_save50"
DEFAULT_SCORE_ROOT = "outputs/stage2_modular_dualexpert/sampled_confident_router_20260518/route_likelihood"
DEFAULT_REPORT_STEM = "2026-05-18_stage2_sampled_k8_confident_routecls_checkpoint258_dev_probe"
LABEL_PATH = (
    "data/stage2_adaptive_datasets/labels/"
    f"{DATA_PREFIX}_sampled_counterfactual_utility_k8_checkpoint-258_dev_seen_labels.jsonl"
)
BUDGETS = [None, 0.03, 0.05, 0.076, 0.10, 0.15, 0.20]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_route(text: str):
    lower = (text or "").lower()
    if "<route>reason</route>" in lower:
        return "reason"
    if "<route>direct</route>" in lower:
        return "direct"
    return "unknown"


def ckpt_num(tag: str):
    return int(tag.split("-", 1)[1])


def fmt(value, digits=4):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def signed(value):
    if value is None:
        return "NA"
    return f"{value:+.4f}"


def pct(value):
    if value is None:
        return "NA"
    return f"{100 * value:.1f}%"


def load_label_map(path: Path):
    return {row["wnd_id"]: row for row in load_jsonl(path)}


def prediction_key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id")


def route_generation_rows(route_root: Path):
    rows = []
    for path in sorted(route_root.glob("checkpoint-*/predictions.jsonl"), key=lambda p: ckpt_num(p.parent.name)):
        summary_path = path.parent / "summary.json"
        rows.append(
            {
                "checkpoint": path.parent.name,
                "predictions": path,
                "summary": load_json(summary_path) if summary_path.exists() else {},
            }
        )
    return rows


def nll_rows(nll_root: Path):
    rows = []
    for path in sorted(nll_root.glob("checkpoint-*/dev_seen_scores.jsonl"), key=lambda p: ckpt_num(p.parent.name)):
        summary_path = path.parent / "dev_seen_summary.json"
        rows.append(
            {
                "checkpoint": path.parent.name,
                "scores": path,
                "summary": load_json(summary_path) if summary_path.exists() else {},
            }
        )
    return rows


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def empty_metric():
    return {
        "num_examples": 0,
        "label_reason_count": 0,
        "label_reason_rate": 0.0,
        "pred_reason_count": 0,
        "pred_reason_rate": 0.0,
        "route_accuracy_vs_confident_label": 0.0,
        "route_vs_confident_label": route_prf(0, 0, 0),
        "sampled_expected_routed": {
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "score": 0.0,
        },
        "sampled_expected_direct": {
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "score": 0.0,
        },
        "sampled_expected_reason_all": {
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "score": 0.0,
        },
        "sampled_expected_routed_minus_direct": {
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "score": 0.0,
        },
    }


def summarize_routes(name: str, pred_routes: dict, label_map: dict):
    common = sorted(set(pred_routes) & set(label_map))
    if not common:
        result = empty_metric()
        result["name"] = name
        return result
    tp = fp = fn = correct = 0
    label_reason = pred_reason = 0
    sums = {
        "direct_trigger": 0.0,
        "direct_argument": 0.0,
        "direct_event": 0.0,
        "direct_score": 0.0,
        "reason_trigger": 0.0,
        "reason_argument": 0.0,
        "reason_event": 0.0,
        "reason_score": 0.0,
        "routed_trigger": 0.0,
        "routed_argument": 0.0,
        "routed_event": 0.0,
        "routed_score": 0.0,
    }
    selected_reason_gains = []
    selected_reason_labels = {"stable_reason": 0, "stable_direct": 0, "ambiguous": 0}
    for key in common:
        label = label_map[key]
        gold = label.get("route_label")
        pred = pred_routes.get(key)
        if pred not in {"direct", "reason"}:
            pred = "direct"
        direct = {
            "trigger": label.get("direct_mean_trigger_f1", 0.0),
            "argument": label.get("direct_mean_argument_f1", 0.0),
            "event": label.get("direct_mean_event_f1", 0.0),
            "score": label.get("direct_mean_score", 0.0),
        }
        reason = {
            "trigger": label.get("reason_mean_trigger_f1", 0.0),
            "argument": label.get("reason_mean_argument_f1", 0.0),
            "event": label.get("reason_mean_event_f1", 0.0),
            "score": label.get("reason_mean_score", 0.0),
        }
        routed = reason if pred == "reason" else direct
        for metric in ["trigger", "argument", "event", "score"]:
            sums[f"direct_{metric}"] += direct[metric]
            sums[f"reason_{metric}"] += reason[metric]
            sums[f"routed_{metric}"] += routed[metric]
        if gold == "reason":
            label_reason += 1
        if pred == "reason":
            pred_reason += 1
            selected_reason_gains.append(label.get("mean_gain", 0.0))
            selected_reason_labels[label.get("utility_label", "ambiguous")] = (
                selected_reason_labels.get(label.get("utility_label", "ambiguous"), 0) + 1
            )
        if pred == gold:
            correct += 1
        if pred == "reason" and gold == "reason":
            tp += 1
        elif pred == "reason" and gold != "reason":
            fp += 1
        elif pred != "reason" and gold == "reason":
            fn += 1
    n = len(common)
    result = {
        "name": name,
        "num_examples": n,
        "label_reason_count": label_reason,
        "label_reason_rate": label_reason / n,
        "pred_reason_count": pred_reason,
        "pred_reason_rate": pred_reason / n,
        "route_accuracy_vs_confident_label": correct / n,
        "route_vs_confident_label": route_prf(tp, fp, fn),
        "selected_reason_avg_sampled_gain": (
            sum(selected_reason_gains) / len(selected_reason_gains) if selected_reason_gains else 0.0
        ),
        "selected_reason_utility_labels": selected_reason_labels,
        "sampled_expected_routed": {
            "trigger_f1": sums["routed_trigger"] / n,
            "argument_f1": sums["routed_argument"] / n,
            "event_f1": sums["routed_event"] / n,
            "score": sums["routed_score"] / n,
        },
        "sampled_expected_direct": {
            "trigger_f1": sums["direct_trigger"] / n,
            "argument_f1": sums["direct_argument"] / n,
            "event_f1": sums["direct_event"] / n,
            "score": sums["direct_score"] / n,
        },
        "sampled_expected_reason_all": {
            "trigger_f1": sums["reason_trigger"] / n,
            "argument_f1": sums["reason_argument"] / n,
            "event_f1": sums["reason_event"] / n,
            "score": sums["reason_score"] / n,
        },
    }
    result["sampled_expected_routed_minus_direct"] = {
        metric: result["sampled_expected_routed"][metric] - result["sampled_expected_direct"][metric]
        for metric in ["trigger_f1", "argument_f1", "event_f1", "score"]
    }
    return result


def load_generated_pred_routes(path: Path):
    pred_routes = {}
    for row in load_jsonl(path):
        key = prediction_key(row)
        if key:
            pred_routes[key] = row.get("route_pred") or extract_route(row.get("generated") or row.get("prediction") or "")
    return pred_routes


def load_nll_pred_routes(path: Path, budget):
    rows = []
    for row in load_jsonl(path):
        key = prediction_key(row)
        if key is None:
            continue
        delta = row.get("delta_direct_minus_reason_route_nll")
        rows.append((float(delta) if delta is not None else float("-inf"), key))
    rows.sort(reverse=True)
    if budget is None:
        reason_keys = {key for delta, key in rows if delta > 0}
        label = "argmin"
    else:
        cap = round(len(rows) * budget)
        reason_keys = {key for _, key in rows[:cap]}
        label = f"top{int(budget * 1000):03d}"
    return {key: ("reason" if key in reason_keys else "direct") for _, key in rows}, label


def best(rows, key_fn):
    return max(rows, key=key_fn) if rows else None


def render_report(payload):
    lines = [
        "# Sampled K8 Confident Route Classifier Dev Probe",
        "",
        payload["description"],
        "",
        "## Summary",
        "",
        "| router | pred reason | label P/R/F1 | sampled routed delta A/E/T/Score | selected reason avg gain |",
        "|---|---:|---:|---:|---:|",
    ]
    top_exec = sorted(
        payload["execution_results"],
        key=lambda row: (
            row["sampled_expected_routed_minus_direct"]["score"],
            row["sampled_expected_routed_minus_direct"]["event_f1"],
            row["sampled_expected_routed_minus_direct"]["argument_f1"],
            row["sampled_expected_routed_minus_direct"]["trigger_f1"],
        ),
        reverse=True,
    )[:24]
    for row in top_exec:
        prf = row["route_vs_confident_label"]
        delta = row["sampled_expected_routed_minus_direct"]
        lines.append(
            "| {name} | {rate} | {p}/{r}/{f} | {da}/{de}/{dt}/{ds} | {gain} |".format(
                name=row["name"],
                rate=pct(row["pred_reason_rate"]),
                p=fmt(prf["precision"], 3),
                r=fmt(prf["recall"], 3),
                f=fmt(prf["f1"], 3),
                da=signed(delta["argument_f1"]),
                de=signed(delta["event_f1"]),
                dt=signed(delta["trigger_f1"]),
                ds=signed(delta["score"]),
                gain=fmt(row.get("selected_reason_avg_sampled_gain"), 4),
            )
        )
    route_best = payload["best"]["generated_route_f1"]
    nll_best = payload["best"]["nll_route_f1"]
    exec_best = payload["best"]["sampled_expected_score_delta"]
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- Best generated-route F1: `{route_best['name']}` with F1 `{route_best['route_vs_confident_label']['f1']:.3f}` and pred-reason rate `{route_best['pred_reason_rate']:.1%}`.",
            f"- Best NLL route F1: `{nll_best['name']}` with F1 `{nll_best['route_vs_confident_label']['f1']:.3f}` and pred-reason rate `{nll_best['pred_reason_rate']:.1%}`.",
            f"- Best sampled expected score delta: `{exec_best['name']}` with routed-minus-direct `{exec_best['sampled_expected_routed_minus_direct']['argument_f1']:+.4f}` argument, `{exec_best['sampled_expected_routed_minus_direct']['event_f1']:+.4f}` event, `{exec_best['sampled_expected_routed_minus_direct']['trigger_f1']:+.4f}` trigger, `{exec_best['sampled_expected_routed_minus_direct']['score']:+.4f}` score.",
            "",
            "## Inputs",
            "",
            f"- label path: `{payload['label_path']}`",
            f"- route generation root: `{payload['route_generation_root']}`",
            f"- route NLL root: `{payload['route_nll_root']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--score_root", default=DEFAULT_SCORE_ROOT)
    parser.add_argument("--report_stem", default=DEFAULT_REPORT_STEM)
    parser.add_argument(
        "--description",
        default=(
            "This dev-only report evaluates a route-only classifier trained on K=8 sampled "
            "counterfactual stable_reason/stable_direct labels. Metrics are computed against "
            "sampled expected utility labels, not a single generation seed."
        ),
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--route_root", default=None)
    args = parser.parse_args()

    label_path = REPO / LABEL_PATH
    route_root = (
        Path(args.route_root)
        if args.route_root
        else REPO
        / "outputs/stage2_adaptive_runs_user_devpick_route"
        / f"{RUN_PREFIX}_{args.branch}_full_route_dev_seen_max16"
    )
    nll_root = REPO / args.score_root / args.branch
    output_json = Path(args.output_json) if args.output_json else REPO / f"reports/artifacts/{args.report_stem}.json"
    output_md = Path(args.output_md) if args.output_md else REPO / f"reports/{args.report_stem}.md"
    label_map = load_label_map(label_path)

    execution_rows = []
    route_rows = []
    for item in route_generation_rows(route_root):
        pred_routes = load_generated_pred_routes(item["predictions"])
        result = summarize_routes(f"{item['checkpoint']}_gen", pred_routes, label_map)
        route_rows.append({"checkpoint": item["checkpoint"], **item["summary"], "sampled_summary": result})
        execution_rows.append(result)

    nll_summary = []
    for item in nll_rows(nll_root):
        for budget in BUDGETS:
            pred_routes, label = load_nll_pred_routes(item["scores"], budget)
            result = summarize_routes(f"{item['checkpoint']}_nll_{label}", pred_routes, label_map)
            execution_rows.append(result)
        nll_summary.append({"checkpoint": item["checkpoint"], **item["summary"]})

    if not route_rows:
        raise FileNotFoundError(f"no route generation outputs under {route_root}")
    if not nll_summary:
        raise FileNotFoundError(f"no route NLL outputs under {nll_root}")

    payload = {
        "branch": args.branch,
        "description": args.description,
        "label_path": label_path.as_posix(),
        "route_generation_root": route_root.as_posix(),
        "route_nll_root": nll_root.as_posix(),
        "route_generation_summaries": route_rows,
        "route_nll_summaries": nll_summary,
        "execution_results": execution_rows,
        "best": {
            "generated_route_f1": best(
                [row for row in execution_rows if row["name"].endswith("_gen")],
                lambda row: row["route_vs_confident_label"]["f1"],
            ),
            "nll_route_f1": best(
                [row for row in execution_rows if "_nll_" in row["name"]],
                lambda row: row["route_vs_confident_label"]["f1"],
            ),
            "sampled_expected_score_delta": best(
                execution_rows,
                lambda row: (
                    row["sampled_expected_routed_minus_direct"]["score"],
                    row["sampled_expected_routed_minus_direct"]["event_f1"],
                    row["sampled_expected_routed_minus_direct"]["argument_f1"],
                    row["sampled_expected_routed_minus_direct"]["trigger_f1"],
                ),
            ),
        },
    }
    write_json(output_json, payload)
    write_text(output_md, render_report(payload))
    print(json.dumps({"output_json": output_json.as_posix(), "output_md": output_md.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
