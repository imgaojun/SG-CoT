#!/usr/bin/env python3
"""Tier 0:统一 scorer 附录表。

所有系统用同一 consistent exact-tuple corpus-micro scorer
(实现复制自 scripts/teacher_zeroshot_baseline_20260702.py 的
norm_events/prf/score,即 e104/e105 的口径)重打 split1 test_seen/test_unseen。
预测缺失的系统自动跳过(06 批评测落地后重跑本脚本即可补全)。
DEGREE 直接读其 summary.json(e105 的 eval 内联了同一 scorer)。
"""
import json
from pathlib import Path

R = Path("/mnt/disk/gaojun/research/progressive-ee")
EV = R / "outputs/stage2_strategy_cot_e65/e57_cross_model_20260608"
EXT = R / "outputs/stage2_external_baseline"
NEW = R / "outputs/strengthen_20260709"

# ---- consistent scorer(teacher_zeroshot_baseline_20260702.py 原样复制)----
def norm_events(obj):
    evs = []
    if not isinstance(obj, dict):
        return evs
    events = obj.get("events") or []
    if not isinstance(events, list):
        return evs
    for e in events:
        if not isinstance(e, dict):
            continue
        t = e.get("event_type") or e.get("type")
        tr = e.get("trigger") or {}
        if isinstance(tr, list):
            tr = tr[0] if tr and isinstance(tr[0], dict) else {"text": str(tr[0]) if tr else ""}
        if isinstance(tr, str):
            tr = {"text": tr}
        if not isinstance(tr, dict):
            tr = {}
        arglist = e.get("arguments") or []
        if not isinstance(arglist, list):
            arglist = []
        args = tuple(sorted(((a.get("role") or "",
                              a.get("start") if a.get("start") is not None else -1,
                              a.get("end") if a.get("end") is not None else -1)
                             for a in arglist if isinstance(a, dict))))
        evs.append(((tr.get("start"), tr.get("end"), t), args))
    return evs

def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)

def score(rows):
    t = [0, 0, 0]; a = [0, 0, 0]; ev = [0, 0, 0]
    for gold, pred in rows:
        gt = [k for k, _ in gold]; pt = [k for k, _ in pred]; m = []
        for k in pt:
            if k in gt and k not in m: t[0] += 1; m.append(k)
            else: t[1] += 1
        t[2] += len(gt) - sum(1 for k in gt if k in pt)
        ga = [(k, x) for k, args in gold for x in args]; pa = [(k, x) for k, args in pred for x in args]; m2 = []
        for x in pa:
            if x in ga and x not in m2: a[0] += 1; m2.append(x)
            else: a[1] += 1
        a[2] += len(ga) - sum(1 for x in ga if x in pa)
        m3 = []
        for e in pred:
            if e in gold and e not in m3: ev[0] += 1; m3.append(e)
            else: ev[1] += 1
        ev[2] += len(gold) - sum(1 for e in gold if e in pred)
    return prf(*a)[2], prf(*ev)[2], prf(*t)[2]

# ---- 系统注册:name -> (seen_dir, unseen_dir, pred_field) ----
def d(p): return str(p)
SYSTEMS = {
    # 外部基线(现存)
    "GoLLIE-style":      (d(EV / "qwen4_gollie_style/checkpoint-2064/test_seen"),   d(EV / "qwen4_gollie_style/checkpoint-2064/test_unseen"),   "predicted"),
    "Teacher zero-shot": (d(EXT / "glm51_zeroshot_20260702/test_seen"),             d(EXT / "glm51_zeroshot_20260702/test_unseen"),             "parsed"),
    "DeepSeek zero-shot":(d(EXT / "deepseek_v4pro_zeroshot_20260705/test_seen"),    d(EXT / "deepseek_v4pro_zeroshot_20260705/test_unseen"),    "parsed"),
    "Few-shot ICL":      (d(EXT / "glm51_fewshot3_20260705/test_seen"),             d(EXT / "glm51_fewshot3_20260705/test_unseen"),             "parsed"),
    # 内部系统(现存)
    "G7 (direct)":       (d(EV / "g7_refresh1ep/checkpoint-129/direct_test_seen"),  d(EV / "g7_refresh1ep/checkpoint-129/direct_test_unseen"),  "predicted"),
    "G9 (cot greedy)":   (d(EV / "g9_cotcalib/checkpoint-91/cot_test_seen"),        d(EV / "g9_cotcalib/checkpoint-91/cot_test_unseen"),        "predicted"),
    "G9 (direct)":       (d(EV / "g9_cotcalib/checkpoint-91/direct_test_seen"),     d(EV / "g9_cotcalib/checkpoint-91/direct_test_unseen"),     "predicted"),
    # 06 批重建后可用
    "Direct (base,retrain)": (d(NEW / "new/direct_base/test_seen"),   d(NEW / "new/direct_base/test_unseen"),   "predicted"),
    "Direct (repeat1)":  (d(NEW / "mixed/direct_repeat1/test_seen"),  d(NEW / "mixed/direct_repeat1/test_unseen"),  "predicted"),
    "Direct (repeat2)":  (d(NEW / "mixed/direct_repeat2/test_seen"),  d(NEW / "mixed/direct_repeat2/test_unseen"),  "predicted"),
    "E81 (base)":        (d(NEW / "mixed/e81_base/test_seen"),        d(NEW / "mixed/e81_base/test_unseen"),        "predicted"),
    "E81 (r1)":          (d(NEW / "mixed/e81_r1/test_seen"),          d(NEW / "mixed/e81_r1/test_unseen"),          "predicted"),
    "E81 (r2)":          (d(NEW / "mixed/e81_r2/test_seen"),          d(NEW / "mixed/e81_r2/test_unseen"),          "predicted"),
    "E81 (r3)":          (d(NEW / "mixed/e81_r3/test_seen"),          d(NEW / "mixed/e81_r3/test_unseen"),          "predicted"),
    "E81 (r4)":          (d(NEW / "mixed/e81_r4/test_seen"),          d(NEW / "mixed/e81_r4/test_unseen"),          "predicted"),
    "SG-CoT-SD (e83)":   (d(NEW / "mixed/e83_richere/test_seen"),     d(NEW / "mixed/e83_richere/test_unseen"),     "predicted"),
    "Same-data ctrl (e77)": (d(NEW / "mixed/e77_base/test_seen"),     d(NEW / "mixed/e77_base/test_unseen"),        "predicted"),
}

def load_rows(dir_, field):
    f = Path(dir_) / "predictions.jsonl"
    if not f.exists():
        return None
    def as_obj(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {}
        return v
    rows = []
    for line in open(f):
        rec = json.loads(line)
        gold = norm_events(as_obj(rec.get("gold")))
        pred = norm_events(as_obj(rec.get(field) or rec.get("predicted") or {}))
        rows.append((gold, pred))
    return rows

def main():
    out = {}
    print("=" * 108)
    print("统一 consistent exact-tuple scorer(corpus-micro)— split1 test_seen / test_unseen,A/E/T F1")
    print("=" * 108)
    print(f"{'system':26s} | {'seen A':7s} {'seen E':7s} {'seen T':7s} | {'unseen A':8s} {'unseen E':8s} {'unseen T':8s}")
    print("-" * 108)
    for name, (sd, ud, field) in SYSTEMS.items():
        rs, ru = load_rows(sd, field), load_rows(ud, field)
        if rs is None and ru is None:
            print(f"{name:26s} | (预测缺失,待 06 批)")
            continue
        row = {}
        cells_s = cells_u = ["   -  "] * 3
        if rs is not None:
            a, e, t = score(rs)
            row["seen"] = [a, e, t]
            cells_s = [f"{a:.3f} ", f"{e:.3f} ", f"{t:.3f} "]
        if ru is not None:
            a, e, t = score(ru)
            row["unseen"] = [a, e, t]
            cells_u = [f"{a:.3f}  ", f"{e:.3f}  ", f"{t:.3f}  "]
        out[name] = row
        print(f"{name:26s} | {cells_s[0]:7s}{cells_s[1]:7s}{cells_s[2]:7s} | {cells_u[0]:8s}{cells_u[1]:8s}{cells_u[2]:8s}")
    dst = R / "reports/artifacts/2026-07-09_unified_scorer_table.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n已存 {dst}")

if __name__ == "__main__":
    main()
