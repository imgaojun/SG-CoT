#!/usr/bin/env python3
"""原生 Direct on WikiEvents 的类型分组一致重打分(corpus-micro exact-tuple F1)。

原生 Direct 在 WikiEvents seen 类型上训练;test_seen 里非 unseen 类型 = 真 in-domain seen,
test_unseen 里 8 个 held-out 类型 = WikiEvents-native unseen(有效的原生泛化度量)。
"""
import json
from pathlib import Path

ROOT = Path("/mnt/disk/gaojun/research/progressive-ee")
EVAL = ROOT / "outputs/wikievents_native_eval"
UNSEEN = set(json.load(open(ROOT / "data/wikievents_transfer/wikievents_unseen_types.json")))
SIBLING_PRESENT = {"Contact.Contact.Meet", "Contact.Contact.Correspondence",
                   "Conflict.Demonstrate.Unspecified", "Movement.Transportation.Evacuation"}
WHOLLY_NOVEL = {"Justice.Sentence.Unspecified", "Justice.ReleaseParole.Unspecified",
                "Cognitive.Inspection.SensoryObserve", "Disaster.DiseaseOutbreak.Unspecified"}


def tuples(events, keep):
    trig, arg, evt = set(), set(), set()
    for e in events or []:
        if not isinstance(e, dict):
            continue
        et = e.get("event_type")
        if keep is not None and et not in keep:
            continue
        t = e.get("trigger") or {}
        ts, te = t.get("start"), t.get("end")
        trig.add((et, ts, te))
        aa = []
        for a in e.get("arguments") or []:
            if not isinstance(a, dict):
                continue
            arg.add((et, ts, te, a.get("role"), a.get("start"), a.get("end")))
            aa.append((a.get("role"), a.get("start"), a.get("end")))
        sa = tuple(sorted(aa, key=lambda x: (x[0] or "", -1 if x[1] is None else x[1], -1 if x[2] is None else x[2])))
        evt.add((et, ts, te, repr(sa)))
    return trig, arg, evt


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return f, p, r, len(gold), len(pred), tp


def score(part, keep):
    f = EVAL / part / "predictions.jsonl"
    if not f.exists():
        return None
    gT, gA, gE, pT, pA, pE = (set() for _ in range(6))
    for i, line in enumerate(open(f)):
        d = json.loads(line)
        gt, ga, ge = tuples((d.get("gold") or {}).get("events"), keep)
        pt, pa, pe = tuples((d.get("predicted") or {}).get("events"), keep)
        for s in gt: gT.add((i,) + s)
        for s in ga: gA.add((i,) + s)
        for s in ge: gE.add((i,) + s)
        for s in pt: pT.add((i,) + s)
        for s in pa: pA.add((i,) + s)
        for s in pe: pE.add((i,) + s)
    return {"trigger": prf(pT, gT), "argument": prf(pA, gA), "event": prf(pE, gE)}


def fmt(m):
    f, p, r, ng, npd, tp = m
    return f"F1={f:.3f}(P{p:.3f}/R{r:.3f} tp{tp}/g{ng}/p{npd})"


if __name__ == "__main__":
    print("=" * 90)
    print("原生 Direct on WikiEvents — in-domain seen/unseen(corpus-micro exact-tuple)")
    print("=" * 90)
    seen_all = set()
    fs = EVAL / "direct_test_seen" / "predictions.jsonl"
    if fs.exists():
        for l in open(fs):
            for e in (json.loads(l).get("gold") or {}).get("events", []):
                seen_all.add(e.get("event_type"))
    SEEN = seen_all - UNSEEN
    for tag, part, keep in [
        ("SEEN (in-domain, 非unseen)", "direct_test_seen", SEEN),
        ("UNSEEN (8 held-out)", "direct_test_unseen", UNSEEN),
        ("  sibling-present", "direct_test_unseen", SIBLING_PRESENT),
        ("  wholly-novel", "direct_test_unseen", WHOLLY_NOVEL),
    ]:
        r = score(part, keep)
        if r is None:
            print(f"{tag}: 无预测"); continue
        print(f"\n### {tag}")
        print(f"    Trig {fmt(r['trigger'])}")
        print(f"    Arg  {fmt(r['argument'])}")
        print(f"    Evt  {fmt(r['event'])}")
