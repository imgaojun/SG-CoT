#!/usr/bin/env python3
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_consolidated_20260519/sampledk2_policy_report"
REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_formal_consolidated_policy_report.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_formal_consolidated_policy_report.json"

GUARD_SWEEP = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_formal_guard_sweep.json"
ROUTED_EXECUTION = REPO / "reports/artifacts/2026-05-18_stage2_sampled_k2_formal_routed_execution.json"
SEEDPAIR_19_20 = REPO / "reports/artifacts/2026-05-18_stage2_sampled_k2_formal_seedpair19_20_robustness.json"
SEEDPAIR_CONSENSUS = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_formal_seedpair_consensus.json"
UNSEEN_DIAGNOSIS = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_formal_unseen_false_positive_diagnosis.json"

SPLITS = ["test", "test_seen", "test_unseen"]
POLICIES = [
    "old_main",
    "new_main",
    "both_main",
    "avg_main",
    "old_and_reason_not_fewer_events",
    "both_and_reason_not_fewer_events",
]

POLICY_LABELS = {
    "old_main": "17/18 margin",
    "new_main": "19/20 margin",
    "both_main": "seedpair consensus",
    "avg_main": "average margin",
    "old_and_reason_not_fewer_events": "17/18 margin + evidence-sparsity guard",
    "both_and_reason_not_fewer_events": "consensus + evidence-sparsity guard",
}

POLICY_RULES = {
    "old_main": "old_margin >= 0.25",
    "new_main": "new_margin >= 0.25",
    "both_main": "old_margin >= 0.25 and new_margin >= 0.25",
    "avg_main": "(old_margin + new_margin) / 2 >= 0.25",
    "old_and_reason_not_fewer_events": "old_margin >= 0.25 and reason_minus_direct_event_count_mean >= 0",
    "both_and_reason_not_fewer_events": "old_margin >= 0.25 and new_margin >= 0.25 and reason_minus_direct_event_count_mean >= 0",
}

POLICY_STATUS = {
    "old_main": "formal preregistered route-NLL baseline",
    "new_main": "formal seedpair robustness check",
    "both_main": "formal seedpair consensus check",
    "avg_main": "formal seedpair average check",
    "old_and_reason_not_fewer_events": "post-hoc guard from formal diagnostics; needs external validation",
    "both_and_reason_not_fewer_events": "post-hoc conservative guard; lower coverage",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def signed(value, digits=4):
    return f"{value:+.{digits}f}"


def pct(value, digits=1):
    return f"{100 * value:.{digits}f}%"


def find_result(payload, source, policy, split):
    for row in payload["results"]:
        if row["source"] == source and row["policy"] == policy and row["split"] == split:
            return row
    raise KeyError((source, policy, split))


def metric_delta(row, metric):
    return row["routed_minus_direct"][metric]


def short_delta(row):
    delta = row["routed_minus_direct"]
    return (
        f"{signed(delta['argument_f1'])} / {signed(delta['event_f1'])} / "
        f"{signed(delta['trigger_f1'])} / {signed(delta['score'])}"
    )


def collect_policy_rows(guard_payload, source):
    rows = []
    for policy in POLICIES:
        split_rows = {split: find_result(guard_payload, source, policy, split) for split in SPLITS}
        rows.append(
            {
                "source": source,
                "policy": policy,
                "label": POLICY_LABELS[policy],
                "rule": POLICY_RULES[policy],
                "status": POLICY_STATUS[policy],
                "test_reason_rate": split_rows["test"]["pred_reason_rate"],
                "seen_reason_rate": split_rows["test_seen"]["pred_reason_rate"],
                "unseen_reason_rate": split_rows["test_unseen"]["pred_reason_rate"],
                "test_score_delta": metric_delta(split_rows["test"], "score"),
                "seen_score_delta": metric_delta(split_rows["test_seen"], "score"),
                "unseen_score_delta": metric_delta(split_rows["test_unseen"], "score"),
                "test_metric_delta": split_rows["test"]["routed_minus_direct"],
                "seen_metric_delta": split_rows["test_seen"]["routed_minus_direct"],
                "unseen_metric_delta": split_rows["test_unseen"]["routed_minus_direct"],
                "test_delta_compact": short_delta(split_rows["test"]),
                "seen_delta_compact": short_delta(split_rows["test_seen"]),
                "unseen_delta_compact": short_delta(split_rows["test_unseen"]),
            }
        )
    return rows


def collect_baselines(guard_payload, source):
    row = find_result(guard_payload, source, "old_main", "test")
    direct = row["direct"]
    reason = row["reason_all"]
    return {
        "source": source,
        "test_direct_score": direct["score"],
        "test_reason_all_score": reason["score"],
        "reason_all_minus_direct_score": reason["score"] - direct["score"],
        "test_direct_argument_f1": direct["argument_f1"],
        "test_reason_all_argument_f1": reason["argument_f1"],
        "test_direct_event_f1": direct["event_f1"],
        "test_reason_all_event_f1": reason["event_f1"],
        "test_direct_trigger_f1": direct["trigger_f1"],
        "test_reason_all_trigger_f1": reason["trigger_f1"],
    }


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def render_source_table(title, rows):
    body = []
    body.append(f"## {title}")
    body.append("")
    body.append(
        markdown_table(
            [
                "policy",
                "rule",
                "reason rate test/seen/unseen",
                "score delta test/seen/unseen",
                "A/E/T/score delta on test",
                "status",
            ],
            [
                [
                    row["label"],
                    f"`{row['rule']}`",
                    f"{pct(row['test_reason_rate'])} / {pct(row['seen_reason_rate'])} / {pct(row['unseen_reason_rate'])}",
                    f"{signed(row['test_score_delta'])} / {signed(row['seen_score_delta'])} / {signed(row['unseen_score_delta'])}",
                    row["test_delta_compact"],
                    row["status"],
                ]
                for row in rows
            ],
        )
    )
    return "\n".join(body)


def main():
    guard = load_json(GUARD_SWEEP)
    routed_execution = load_json(ROUTED_EXECUTION)
    seedpair_19_20 = load_json(SEEDPAIR_19_20)
    consensus = load_json(SEEDPAIR_CONSENSUS)
    unseen_diag = load_json(UNSEEN_DIAGNOSIS)

    single_gen_rows = collect_policy_rows(guard, "single_gen_execution")
    k2_rows = collect_policy_rows(guard, "k2_expected")
    baselines = [
        collect_baselines(guard, "single_gen_execution"),
        collect_baselines(guard, "k2_expected"),
    ]

    recommended = next(row for row in single_gen_rows if row["policy"] == "old_and_reason_not_fewer_events")
    preregistered = next(row for row in single_gen_rows if row["policy"] == "old_main")

    payload = {
        "output_root": OUTPUT_ROOT.as_posix(),
        "sources": {
            "guard_sweep": GUARD_SWEEP.as_posix(),
            "routed_execution": ROUTED_EXECUTION.as_posix(),
            "seedpair_19_20": SEEDPAIR_19_20.as_posix(),
            "seedpair_consensus": SEEDPAIR_CONSENSUS.as_posix(),
            "unseen_diagnosis": UNSEEN_DIAGNOSIS.as_posix(),
        },
        "baselines": baselines,
        "single_gen_execution_policies": single_gen_rows,
        "k2_expected_policies": k2_rows,
        "recommendation": {
            "current_best_policy": recommended,
            "formal_baseline_policy": preregistered,
            "interpretation": (
                "Use old_main as the clean formal baseline and treat "
                "old_and_reason_not_fewer_events as the current best post-hoc guard "
                "that must be validated on another split or dataset before paper-level claims."
            ),
        },
        "supporting_artifact_summaries": {
            "routed_execution_policy": routed_execution.get("policy"),
            "seedpair_19_20_policies": seedpair_19_20.get("policies"),
            "consensus_policies": consensus.get("policies"),
            "unseen_diagnosis_keys": sorted(unseen_diag.keys()),
        },
    }

    write_json(REPORT_JSON, payload)
    write_json(OUTPUT_ROOT / "summary.json", payload)

    baseline_rows = []
    for row in baselines:
        baseline_rows.append(
            [
                row["source"],
                signed(row["test_reason_all_score"] - row["test_direct_score"]),
                f"{row['test_direct_score']:.4f}",
                f"{row['test_reason_all_score']:.4f}",
            ]
        )

    md = []
    md.append("# Stage2 Sampled K2 Formal Consolidated Policy Report")
    md.append("")
    md.append("## Executive Takeaway")
    md.append("")
    md.append(
        "- The clean formal baseline remains `old_main` (`17/18` route-NLL margin >= `0.25`): "
        f"single-generation score delta {signed(preregistered['test_score_delta'])} on test, "
        f"{signed(preregistered['seen_score_delta'])} on seen, and {signed(preregistered['unseen_score_delta'])} on unseen."
    )
    md.append(
        "- The strongest offline policy is `old_and_reason_not_fewer_events`: "
        f"single-generation score delta {signed(recommended['test_score_delta'])} on test, "
        f"{signed(recommended['seen_score_delta'])} on seen, and {signed(recommended['unseen_score_delta'])} on unseen, "
        f"with reason rates {pct(recommended['test_reason_rate'])} / {pct(recommended['seen_reason_rate'])} / {pct(recommended['unseen_reason_rate'])}."
    )
    md.append(
        "- This guard is gold-free at inference over sampled evidence, but it is post-hoc because it was selected after the formal diagnostics. "
        "It should be the next validation target, not the final paper claim yet."
    )
    md.append("")
    md.append("## Direct-vs-Reason Baseline")
    md.append("")
    md.append(
        "Reason-all is clearly worse than Direct-all, so the route problem is not whether Reason should dominate globally; "
        "it is whether we can find a small stable subset where Reason improves extraction."
    )
    md.append("")
    md.append(markdown_table(["source", "Reason-all minus Direct score", "Direct score", "Reason-all score"], baseline_rows))
    md.append("")
    md.append(render_source_table("Single-Generation Routed Execution", single_gen_rows))
    md.append("")
    md.append(render_source_table("K2 Expected Formal Estimate", k2_rows))
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "The seedpair-robustness experiments showed that a simple high-margin Reason label is noisy, especially on `test_unseen`. "
        "The failure mode is not only low confidence; some selected unseen examples have Reason outputs that predict fewer events than Direct, "
        "which creates false-positive Reason routing. The evidence-sparsity guard removes exactly that class of cases."
    )
    md.append("")
    md.append(
        "For writing and follow-up experiments, keep two levels separate: `old_main` is the clean formal result that was available before the post-hoc guard; "
        "`old_and_reason_not_fewer_events` is the best current policy hypothesis and needs validation on a held-out split, alternate seed set, or another dataset."
    )
    md.append("")
    md.append("## Next")
    md.append("")
    md.append(
        "1. Validate the event-count guard without retuning on another split or dataset; if no extra split is available, use a locked rule and run a new seedpair/evidence batch."
    )
    md.append(
        "2. Convert the guard into a trainable route-supervision target only after the locked-policy validation passes."
    )
    md.append(
        "3. Report `old_main` and the locked guard separately in paper tables to avoid mixing preregistered and diagnostic-selected policies."
    )
    md.append("")
    md.append("## Artifacts")
    md.append("")
    for key, value in payload["sources"].items():
        md.append(f"- {key}: `{value}`")
    md.append(f"- report json: `{REPORT_JSON.as_posix()}`")
    md.append(f"- output summary: `{(OUTPUT_ROOT / 'summary.json').as_posix()}`")
    md.append("")

    write_text(REPORT_MD, "\n".join(md))

    print(f"wrote {REPORT_MD}")
    print(f"wrote {REPORT_JSON}")
    print(f"wrote {OUTPUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
