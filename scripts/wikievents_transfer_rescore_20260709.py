#!/usr/bin/env python3
"""WikiEvents 迁移评测的类型分组一致重打分。

两个官方评测器(offset / evidence)对 test_unseen 文件会把窗口内共现的 seen 事件
一起算进去,与 RichERE"纯 unseen 类型"口径不可比。这里从 predictions.jsonl 读
gold+predicted,按类型分组做 corpus-micro exact-tuple F1(与 normalize_events 同键),
并给出 sibling-present vs wholly-novel 的 4:4 拆分。
"""
import json
import glob
from pathlib import Path

ROOT = Path("/mnt/disk/gaojun/research/progressive-ee")
EVAL_DIR = ROOT / "outputs/wikievents_transfer_eval"
UNSEEN = set(json.load(open(ROOT / "data/wikievents_transfer/wikievents_unseen_types.json")))

# 4:4 sibling 拆分(前两级有无 seen 兄弟,见可行性侦察)
SIBLING_PRESENT = {
    "Contact.Contact.Meet", "Contact.Contact.Correspondence",
    "Conflict.Demonstrate.Unspecified", "Movement.Transportation.Evacuation",
}
WHOLLY_NOVEL = {
    "Justice.Sentence.Unspecified", "Justice.ReleaseParole.Unspecified",
    "Cognitive.Inspection.SensoryObserve", "Disaster.DiseaseOutbreak.Unspecified",
}


def tuples(events, keep_types):
    """按 normalize_events 同键构造 trig/arg/event 元组集,限定 keep_types。"""
    trig, arg, evt = set(), set(), set()
    for e in events or []:
        if not isinstance(e, dict):
            continue
        et = e.get("event_type")
        if keep_types is not None and et not in keep_types:
            continue
        t = e.get("trigger") or {}
        ts, te = t.get("start"), t.get("end")
        trig.add((et, ts, te))
        args = []
        for a in e.get("arguments") or []:
            if not isinstance(a, dict):
                continue
            arg.add((et, ts, te, a.get("role"), a.get("start"), a.get("end")))
            args.append((a.get("role"), a.get("start"), a.get("end")))
        sargs = tuple(sorted(args, key=lambda x: (x[0] or "", -1 if x[1] is None else x[1], -1 if x[2] is None else x[2])))
        evt.add((et, ts, te, sargs))
    return trig, arg, evt


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f, len(gold), len(pred), tp


def score_dir(name, keep_types):
    """corpus-micro:跨窗口 pool 后统一算 P/R/F1。"""
    f = EVAL_DIR / name / "predictions.jsonl"
    if not f.exists():
        return None
    gT, gA, gE, pT, pA, pE = set(), set(), set(), set(), set(), set()
    # 用带窗口 id 前缀防止跨窗口 offset 碰撞
    for i, line in enumerate(open(f)):
        d = json.loads(line)
        gold = d.get("gold") or {}
        pred = d.get("predicted") or {}
        gt, ga, ge = tuples(gold.get("events"), keep_types)
        ptu, pa, pe = tuples(pred.get("events"), keep_types)
        for s in gt: gT.add((i,) + s)
        for s in ga: gA.add((i,) + s)
        for s in ge: gE.add((i,) + s[:3] + (repr(s[3]),))
        for s in ptu: pT.add((i,) + s)
        for s in pa: pA.add((i,) + s)
        for s in pe: pE.add((i,) + s[:3] + (repr(s[3]),))
    return {
        "trigger": prf(pT, gT),
        "argument": prf(pA, gA),
        "event": prf(pE, gE),
    }


def fmt(r):
    if r is None:
        return "  (缺预测)"
    def one(m):
        p, rc, f, ng, npd, tp = m
        return f"F1={f:.3f}(P{p:.3f}/R{rc:.3f} tp{tp}/g{ng}/p{npd})"
    return f"Trig {one(r['trigger'])} | Arg {one(r['argument'])} | Evt {one(r['event'])}"


SEEN_TYPES = None  # 用"非 unseen"作为 seen 口径

if __name__ == "__main__":
    print("=" * 100)
    print("WikiEvents(KAIROS)第三本体零样本迁移 — 一致 exact-tuple corpus-micro 重打分")
    print("=" * 100)

    # seen:test_seen 文件,限定非 unseen 类型
    print("\n### SEEN 类型(test_seen 文件,排除 unseen 类型)")
    def seen_keep(t):  # keep_types 传函数不方便;这里用集合补集,先扫文件收集全部类型
        pass
    # 收集 test_seen 中出现的所有类型,减去 unseen
    seen_all = set()
    fseen = EVAL_DIR / "direct_test_seen" / "predictions.jsonl"
    if fseen.exists():
        for line in open(fseen):
            for e in (json.loads(line).get("gold") or {}).get("events", []):
                seen_all.add(e.get("event_type"))
    seen_keep_types = seen_all - UNSEEN
    for m in ["direct", "e83"]:
        print(f"  {m:7s}: {fmt(score_dir(f'{m}_test_seen', seen_keep_types))}")

    # unseen:test_unseen 文件,限定 unseen 类型
    print("\n### UNSEEN 类型(test_unseen 文件,仅 8 个 held-out 类型)")
    for m in ["direct", "e83"]:
        print(f"  {m:7s}: {fmt(score_dir(f'{m}_test_unseen', UNSEEN))}")

    # 4:4 sibling 拆分
    print("\n### UNSEEN 拆分:sibling-present(有 seen 兄弟)")
    for m in ["direct", "e83"]:
        print(f"  {m:7s}: {fmt(score_dir(f'{m}_test_unseen', SIBLING_PRESENT))}")
    print("\n### UNSEEN 拆分:wholly-novel(无 seen 兄弟)")
    for m in ["direct", "e83"]:
        print(f"  {m:7s}: {fmt(score_dir(f'{m}_test_unseen', WHOLLY_NOVEL))}")
