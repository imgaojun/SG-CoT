import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_paper_table.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_paper_table.md"

STRONG = REPO / "reports/artifacts/2026-05-20_stage2_strong_system_v0_supervised_rerank.json"
ADAPTIVE = {
    "m02_stable": (
        REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_formal.json",
        "early_stable_candidate_checkpoint-100_rank325_400",
    ),
    "m04a": (
        REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_rankstable_router_m04a_formal.json",
        "balanced_candidate_checkpoint-250_rank425_500",
    ),
    "m04b": (
        REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_rankstable_router_m04b_formal.json",
        "balanced_candidate_checkpoint-237_rank425_500",
    ),
    "m02_positive_retention": (
        REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_next_selectors_formal.json",
        "positive_retention_candidate_checkpoint-50_global_rank425_500",
    ),
}

POLICIES = [
    ("Direct only", "strong", "direct_only"),
    ("Reason only", "strong", "reason_only"),
    ("M02 early-stable", "adaptive", "m02_stable"),
    ("M04A balanced", "adaptive", "m04a"),
    ("M04B balanced", "adaptive", "m04b"),
    ("M02 positive-retention", "adaptive", "m02_positive_retention"),
    ("Oracle best", "strong", "oracle_best"),
]
SPLITS = ["test", "test_seen", "test_unseen"]


def signed(value):
    return f"{value:+.4f}"


def fmt(value):
    return f"{value:.4f}"


def row_from_strong(rows, policy, split):
    candidates = [row for row in rows if row["policy"] == policy and row["split"] == split]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one strong row for {policy}/{split}, got {len(candidates)}")
    row = candidates[0]
    return {
        "split": split,
        "reason_rate": row["non_direct_rate"],
        "routed": {
            "argument_f1": row["summary"]["argument_f1"],
            "event_f1": row["summary"]["event_f1"],
            "trigger_f1": row["summary"]["trigger_f1"],
        },
        "delta": {
            "argument_f1": row["delta_vs_direct"]["argument_f1"],
            "event_f1": row["delta_vs_direct"]["event_f1"],
            "trigger_f1": row["delta_vs_direct"]["trigger_f1"],
        },
        "selected_gain": row.get("selected_non_direct_gain_mean", 0.0),
    }


def row_from_adaptive(name, split):
    path, policy = ADAPTIVE[name]
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = [row for row in payload["results"] if row["policy"] == policy and row["split"] == split]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one adaptive row for {name}/{split}, got {len(candidates)}")
    row = candidates[0]
    return {
        "split": split,
        "reason_rate": row["pred_reason_rate"],
        "routed": {
            "argument_f1": row["routed"]["argument_f1"],
            "event_f1": row["routed"]["event_f1"],
            "trigger_f1": row["routed"]["trigger_f1"],
        },
        "delta": {
            "argument_f1": row["routed_minus_direct"]["argument_f1"],
            "event_f1": row["routed_minus_direct"]["event_f1"],
            "trigger_f1": row["routed_minus_direct"]["trigger_f1"],
        },
        "selected_gain": row.get("selected_reason_avg_score_gain", 0.0),
    }


def build_rows():
    strong_rows = json.loads(STRONG.read_text(encoding="utf-8"))["rows"]
    out = []
    for display, source, key in POLICIES:
        for split in SPLITS:
            if source == "strong":
                metrics = row_from_strong(strong_rows, key, split)
            else:
                metrics = row_from_adaptive(key, split)
            out.append(
                {
                    "policy": display,
                    "source": source,
                    **metrics,
                }
            )
    return out


def render_split_table(rows, split):
    lines = [
        f"### `{split}`",
        "",
        "| policy | reason rate | routed A/E/T | delta A/E/T | selected gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in [row for row in rows if row["split"] == split]:
        routed = row["routed"]
        delta = row["delta"]
        lines.append(
            f"| {row['policy']} | {row['reason_rate']:.1%} | "
            f"{fmt(routed['argument_f1'])} / {fmt(routed['event_f1'])} / {fmt(routed['trigger_f1'])} | "
            f"{signed(delta['argument_f1'])} / {signed(delta['event_f1'])} / {signed(delta['trigger_f1'])} | "
            f"{signed(row['selected_gain'])} |"
        )
    return "\n".join(lines)


def render(rows):
    main = next(row for row in rows if row["policy"] == "M02 positive-retention" and row["split"] == "test")
    d = main["delta"]
    lines = [
        "# A/E/T Paper Table",
        "",
        "This report consolidates direct/reason baselines, adaptive selectors, and oracle upper bound. Deltas are against direct-only.",
        "",
        "## Main Reading",
        "",
        f"- Current main adaptive selector: `M02 positive-retention`, reason rate `{main['reason_rate']:.1%}`.",
        f"- Formal `test` A/E/T delta: `{signed(d['argument_f1'])} / {signed(d['event_f1'])} / {signed(d['trigger_f1'])}`.",
        "- It is the strongest low-budget adaptive selector in this comparison and is all-positive on `test`, `test_seen`, and nonnegative on `test_unseen` Trigger.",
        "",
        "## Tables",
        "",
    ]
    for split in SPLITS:
        lines.append(render_split_table(rows, split))
        lines.append("")
    return "\n".join(lines)


def main():
    rows = build_rows()
    payload = {
        "sources": {
            "strong_system": STRONG.as_posix(),
            **{name: path.as_posix() for name, (path, _) in ADAPTIVE.items()},
        },
        "main_candidate": "M02 positive-retention",
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(rows), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
