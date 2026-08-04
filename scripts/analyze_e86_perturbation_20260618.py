#!/usr/bin/env python3
"""Summarize E86 perturbation faithfulness: paired deltas + Wilcoxon signed-rank.

Causal claim: on unseen, teacher-forcing the model's own full reasoning reproduces
its output (forced_full == natural); deleting the arbitration sentences (forced_noarb)
lowers the final structure, while deleting an equal number of non-arbitration
sentences (forced_placebo) does not -> the effect is specific to arbitration content.
Paired one-sided Wilcoxon over per-row F1 quantifies significance.
"""
import json
import math
from pathlib import Path

B = Path("outputs/stage2_e86_perturbation_faithfulness_20260618/e81_ck273")
OUT_MD = Path("reports/2026-06-18_e86_perturbation_faithfulness.md")
OUT_MD.parent.mkdir(parents=True, exist_ok=True)
METRICS = ["trigger_f1", "argument_f1", "event_f1"]


def load(split):
    return [json.loads(l) for l in open(B / f"{split}_perrow.jsonl", encoding="utf-8") if l.strip()]


def paired(rows, cond_a, cond_b, metric):
    """per-row (a, b) where both conditions decoded; returns lists."""
    a, b = [], []
    for r in rows:
        fa = r.get("forced", {}).get(cond_a)
        fb = r.get("forced", {}).get(cond_b)
        if fa and fb and fa.get(metric) is not None and fb.get(metric) is not None:
            a.append(fa[metric]); b.append(fb[metric])
    return a, b


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcox_one_sided_greater(a, b):
    """Manual paired Wilcoxon signed-rank, H1: a > b. Normal approx with tie + continuity
    correction. Returns (Wplus, p, n_nonzero, mean_a, mean_b)."""
    ma = sum(a) / len(a) if a else 0.0
    mb = sum(b) / len(b) if b else 0.0
    diffs = [x - y for x, y in zip(a, b) if (x - y) != 0]
    n = len(diffs)
    if n < 1:
        return None, None, 0, ma, mb
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    tie_corr = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # ranks are 1-based
        t = j - i + 1
        if t > 1:
            tie_corr += t ** 3 - t
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    mean_w = n * (n + 1) / 4.0
    var_w = (n * (n + 1) * (2 * n + 1) - tie_corr / 2.0) / 24.0
    if var_w <= 0:
        return w_plus, None, n, ma, mb
    z = (w_plus - mean_w - 0.5) / math.sqrt(var_w)  # continuity correction, one-sided greater
    p = 1.0 - _phi(z)
    return round(w_plus, 1), p, n, ma, mb


lines = ["# E86 — Perturbation-based faithfulness (causal intervention)", "",
         "Teacher-forced re-decoding of the `<final>` structure under four conditions on the trained SG-CoT model (E81 ck273), holding the input fixed. `forced_full` re-forces the model's own full `<thinking>`; `forced_noarb` deletes the contrastive type-arbitration sentences; `forced_placebo` deletes an equal number of non-arbitration sentences (length/positional control). Causal effect of arbitration = `forced_full - forced_noarb`; specificity is shown by `forced_placebo` not reproducing that drop.", ""]

summary = json.load(open(B / "summary.json"))
for split in ["test_unseen", "test_seen"]:
    rows = load(split)
    s = summary[split]
    lines += [f"## {split}  (n={s['n_rows']}, mean arb sentences removed={s['mean_n_arb']} of {s['mean_n_sent']})", ""]
    lines += ["| condition | Trigger | Argument | Event |", "| --- | --- | --- | --- |"]
    for c in ["natural", "forced_full", "forced_noarb", "forced_placebo"]:
        cc = s["conditions"][c]
        lines.append(f"| {c} | {cc['trigger_f1']} | {cc['argument_f1']} | {cc['event_f1']} |")
    lines.append("")
    # paired deltas + significance: full vs noarb, full vs placebo
    for label, cb in [("full − noarb (arbitration effect)", "forced_noarb"),
                      ("full − placebo (length control)", "forced_placebo")]:
        lines.append(f"**{label}** (paired one-sided Wilcoxon, H1: full > {cb.split('_')[1]}):  ")
        for m in METRICS:
            a, b = paired(rows, "forced_full", cb, m)
            stat, p, nnz, ma, mb = wilcox_one_sided_greater(a, b)
            delta = round(ma - mb, 4)
            ps = f"p={p:.4f}" if p is not None else "p=n/a"
            lines.append(f"- {m}: {round(ma,4)} → {round(mb,4)} (Δ={delta:+.4f}; {ps}; n≠0={nnz})  ")
        lines.append("")

lines += [
    "## Interpretation (honest)",
    "",
    "1. **Forcing is exactly faithful.** On unseen, `forced_full` reproduces `natural` to 4 decimals (0.3397/0.1962/0.1443), so re-decoding the final structure under the model's own teacher-forced reasoning is a valid, side-effect-free baseline.",
    "2. **Deleting arbitration moves the unseen mean down, and an equal-length control does not.** Removing the arbitration sentences lowers unseen Trigger/Argument/Event in the mean (Δ +0.035/+0.024/+0.012), whereas removing an equal number of non-arbitration sentences (`forced_placebo`) does *not* lower Argument/Event (they slightly rise). So the degradation is specific to the arbitration *content*, not to removing text.",
    "3. **The intervention changes few rows (the type-contested ones), so it is corroborating, not high-powered.** Because the rest of the reasoning (trigger lock, argument attachment) is held fixed, deleting only the arbitration sentences flips the final on a minority of rows — precisely the rows where the event type is genuinely contested. Hence the per-row paired Wilcoxon is directional but underpowered (few non-zero diffs).",
    "4. **Selectivity matches the mechanism.** On seen types, deleting arbitration barely changes the output and *less* than the length placebo (placebo significantly lowers seen T/A, p<0.05; deleting arbitration does not) — arbitration is load-bearing precisely on novel (unseen) types, the inference-time causal counterpart of the observational marker finding.",
    "",
    "This within-model, inference-time perturbation corroborates the two retraining-based causal controls (E77 same-data direct control; E80A prompt ablation collapse), which remain the primary causal evidence. It does not, on its own, reach row-level significance.",
    "",
    f"Artifacts: `{B}/summary.json`, per-row `{B}/test_*_perrow.jsonl`, this report.",
]
OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUT_MD)
print("\n".join(lines[:40]))
