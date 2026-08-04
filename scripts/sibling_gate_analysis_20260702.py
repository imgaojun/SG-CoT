#!/usr/bin/env python3
"""G: sibling-gate selective reasoning (post-hoc routing analysis, no training).

Rule: for each window, if ANY candidate event type belongs to a family with ZERO seen types
(sibling-less family, computed from the protocol's seen_types.json), route the window to the
DIRECT model's prediction; otherwise use SG-CoT's. Motivated by e90/e91/e92: SG-CoT's arbitration
needs a seen sibling to prune; without one it over-proposes and loses precision.

Evaluates Direct / SG-CoT / Gated with one consistent offset-based scorer (relative comparison).
"""
import argparse
import collections
import json
from pathlib import Path


def parse_events(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return []
    if isinstance(x, dict):
        x = x.get("events", [])
    out = []
    for e in (x or []):
        t = e.get("event_type") or e.get("type")
        tr = e.get("trigger") or {}
        if isinstance(tr, str):
            tr = {"text": tr}
        args = tuple(sorted(((a.get("role") or "",
                              a.get("start") if a.get("start") is not None else -1,
                              a.get("end") if a.get("end") is not None else -1)
                             for a in (e.get("arguments") or []))))
        out.append(((tr.get("start"), tr.get("end"), t), args))
    return out


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def score(rows):
    """rows: list of (gold_events, pred_events) in parse_events format."""
    t_tp = t_fp = t_fn = 0
    a_tp = a_fp = a_fn = 0
    e_tp = e_fp = e_fn = 0
    for gold, pred in rows:
        gt = [k for k, _ in gold]
        pt = [k for k, _ in pred]
        m = []
        for k in pt:
            if k in gt and k not in m:
                t_tp += 1
                m.append(k)
            else:
                t_fp += 1
        t_fn += len(gt) - sum(1 for k in gt if k in pt)
        # arguments: (trigger key, role, span)
        ga = [(k, a) for k, args in gold for a in args]
        pa = [(k, a) for k, args in pred for a in args]
        m2 = []
        for x in pa:
            if x in ga and x not in m2:
                a_tp += 1
                m2.append(x)
            else:
                a_fp += 1
        a_fn += len(ga) - sum(1 for x in ga if x in pa)
        # whole event exact
        m3 = []
        for ev in pred:
            if ev in gold and ev not in m3:
                e_tp += 1
                m3.append(ev)
            else:
                e_fp += 1
        e_fn += len(gold) - sum(1 for ev in gold if ev in pred)
    return prf(a_tp, a_fp, a_fn)[2], prf(e_tp, e_fp, e_fn)[2], prf(t_tp, t_fp, t_fn)[2]


def load_preds(path, pk):
    rows = []
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        rows.append({
            "input": r.get("input"),
            "gold": parse_events(r.get("gold")),
            "pred": parse_events(r.get(pk)),
            "meta": r.get("meta") or {},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct_preds", required=True)
    ap.add_argument("--sgcot_preds", required=True)
    ap.add_argument("--seen_types_json", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--output_json")
    args = ap.parse_args()

    with open(args.seen_types_json, encoding="utf-8") as f:
        seen = json.load(f)
    seen_fams = {t.split(":")[0] for t in seen}

    dr = load_preds(args.direct_preds, "predicted")
    sg = load_preds(args.sgcot_preds, "final_predicted")
    assert len(dr) == len(sg), f"row count mismatch {len(dr)} vs {len(sg)}"
    # sanity: same window order (inputs identical)
    input_mismatches = sum(1 for a, b in zip(dr, sg) if a["input"] != b["input"])
    if input_mismatches:
        raise ValueError(f"{input_mismatches}/{len(dr)} input mismatches; rows are not paired")
    gold_mismatches = sum(1 for a, b in zip(dr, sg) if a["gold"] != b["gold"])
    if gold_mismatches:
        raise ValueError(f"{gold_mismatches}/{len(dr)} gold mismatches; rows are not paired")

    gated, routed = [], 0
    routed_families = collections.Counter()
    missing_candidate_rows = 0
    for d, s in zip(dr, sg):
        cands = s["meta"].get("candidate_types") or d["meta"].get("candidate_types") or []
        if not cands:
            missing_candidate_rows += 1
        siblingless_families = sorted({
            c.split(":")[0] for c in cands if c.split(":")[0] not in seen_fams
        })
        siblingless = bool(siblingless_families)
        if siblingless:
            routed += 1
            routed_families.update(siblingless_families)
            gated.append((d["gold"], d["pred"]))
        else:
            gated.append((s["gold"], s["pred"]))

    n = len(dr)
    sd = score([(r["gold"], r["pred"]) for r in dr])
    ss = score([(r["gold"], r["pred"]) for r in sg])
    sgt = score(gated)
    print(f"=== sibling-gate: {args.label} (n={n}, routed-to-Direct={routed} [{100*routed/n:.0f}%]) ===")
    print(f"  {'system':10} {'Arg':>7} {'Evt':>7} {'Trig':>7}")
    print(f"  {'Direct':10} {sd[0]:7.4f} {sd[1]:7.4f} {sd[2]:7.4f}")
    print(f"  {'SG-CoT':10} {ss[0]:7.4f} {ss[1]:7.4f} {ss[2]:7.4f}")
    print(f"  {'Gated':10} {sgt[0]:7.4f} {sgt[1]:7.4f} {sgt[2]:7.4f}")
    win = all(sgt[i] >= min(sd[i], ss[i]) - 1e-9 for i in range(3))
    best = all(sgt[i] >= max(sd[i], ss[i]) - 0.02 for i in range(3))
    print(f"  gated >= min(both): {win};  gated within 0.02 of max(both) on all: {best}")
    if missing_candidate_rows:
        raise ValueError(f"candidate_types missing on {missing_candidate_rows}/{n} rows")

    result = {
        "label": args.label,
        "n_rows": n,
        "pairing": {
            "input_mismatches": input_mismatches,
            "gold_mismatches": gold_mismatches,
            "missing_candidate_rows": missing_candidate_rows,
        },
        "seen_family_count": len(seen_fams),
        "routed_to_direct": routed,
        "route_rate": routed / n if n else 0.0,
        "routed_siblingless_families": dict(sorted(routed_families.items())),
        "scores": {
            "direct": {"argument": sd[0], "event": sd[1], "trigger": sd[2]},
            "sgcot": {"argument": ss[0], "event": ss[1], "trigger": ss[2]},
            "gated": {"argument": sgt[0], "event": sgt[1], "trigger": sgt[2]},
        },
        "gated_ge_min_both": win,
        "gated_within_0.02_of_max_both": best,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
