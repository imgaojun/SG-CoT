#!/usr/bin/env python3
"""e83 迁移输出的鲁棒重打分。

迁移到 WikiEvents 后 e83 丢失 <final> 标签(直接 </thinking> 后跟裸 JSON),
且长推理常把 JSON 截断。这里从 generated_text 鲁棒抽取 surface JSON
(裸/带标签/截断均可,平衡括号抢救完整 event 对象),复用 evidence 评测器的
offset 恢复逻辑,再按类型分组做 corpus-micro exact-tuple F1。
"""
import json
import re
from pathlib import Path

ROOT = Path("/mnt/disk/gaojun/research/progressive-ee")
EVAL_DIR = ROOT / "outputs/wikievents_transfer_eval"
UNSEEN = set(json.load(open(ROOT / "data/wikievents_transfer/wikievents_unseen_types.json")))
SIBLING_PRESENT = {"Contact.Contact.Meet", "Contact.Contact.Correspondence",
                   "Conflict.Demonstrate.Unspecified", "Movement.Transportation.Evacuation"}
WHOLLY_NOVEL = {"Justice.Sentence.Unspecified", "Justice.ReleaseParole.Unspecified",
                "Cognitive.Inspection.SensoryObserve", "Disaster.DiseaseOutbreak.Unspecified"}

# ---- evidence 评测器恢复函数(复制,去 torch 依赖)----
def norm_token(t): return re.sub(r"[^0-9a-zA-Z]+", "", (t or "").lower())
def phrase_tokens(text):
    raw = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", text or "")
    return [norm_token(x) for x in raw if norm_token(x)]
def parse_prompt_tokens(inp):
    m = re.search(r"Tokens:\n(.*?)(?:\n\nCandidate event types:|\n\nSchema cards:|\Z)", inp, flags=re.S)
    return [t for t in m.group(1).strip().split() if t] if m else []
def find_subseqs(h, n):
    if not h or not n or len(n) > len(h): return []
    return [(i, i+len(n)) for i in range(len(h)-len(n)+1) if h[i:i+len(n)] == n]
def overlap_or_inside(a, b): return a[0] >= b[0] and a[1] <= b[1]
def choose_surface_span(tokens, surface, evidence):
    nt = [norm_token(t) for t in tokens]
    ss = find_subseqs(nt, phrase_tokens(surface)); es = find_subseqs(nt, phrase_tokens(evidence))
    if not ss: return (None, None)
    for s in ss:
        if any(overlap_or_inside(s, e) for e in es): return s
    if es:
        mid = sum(es[0]) / 2
        return min(ss, key=lambda sp: abs((sp[0]+sp[1])/2 - mid))
    return ss[0]
def recover(surface_payload, inp):
    tokens = parse_prompt_tokens(inp); out = []
    for e in surface_payload.get("events", []) if isinstance(surface_payload, dict) else []:
        if not isinstance(e, dict): continue
        tr = e.get("trigger") if isinstance(e.get("trigger"), dict) else {}
        ts = choose_surface_span(tokens, tr.get("text") or "", tr.get("evidence") or "")
        args = []
        for a in e.get("arguments", []) or []:
            if not isinstance(a, dict): continue
            asp = choose_surface_span(tokens, a.get("text") or "", a.get("evidence") or "")
            args.append({"role": a.get("role"), "start": asp[0], "end": asp[1]})
        out.append({"event_type": e.get("event_type"),
                    "trigger": {"start": ts[0], "end": ts[1]}, "arguments": args})
    return {"events": out}

# ---- 鲁棒 surface-JSON 抽取(裸/带标签/截断)----
def balanced_objects(s):
    """从 '[' 后逐个抽取平衡括号的 {..} 对象(尊重字符串/转义),截断处停止。"""
    objs = []; i = 0; n = len(s)
    while i < n:
        if s[i] == '{':
            depth = 0; instr = False; esc = False; j = i
            while j < n:
                c = s[j]
                if esc: esc = False
                elif c == '\\': esc = True
                elif c == '"': instr = not instr
                elif not instr:
                    if c == '{': depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            frag = s[i:j+1]
                            try: objs.append(json.loads(frag))
                            except Exception: pass
                            i = j + 1; break
                j += 1
            else:
                break  # 到末尾仍未闭合 -> 截断,停止
        else:
            i += 1
    return objs

def extract_surface(gen):
    # 取 <final>/最后 </thinking> 之后的区域
    region = gen
    if '<final>' in gen:
        region = gen.split('<final>', 1)[1]
        region = region.split('</final>', 1)[0]
    elif '</thinking>' in gen:
        region = gen.rsplit('</thinking>', 1)[1]
    # 先整体尝试
    m = re.search(r'\{\s*"events"\s*:', region)
    if not m:
        return {"events": []}, False
    # 找 events 数组的 '['
    br = region.find('[', m.start())
    if br == -1:
        return {"events": []}, False
    objs = balanced_objects(region[br+1:])
    # 判断是否为完整闭合(整段能直接 parse)
    whole_ok = False
    try:
        cand = region[m.start():]
        end = cand.rfind('}')
        json.loads(cand[:end+1]); whole_ok = True
    except Exception:
        whole_ok = False
    return {"events": objs}, whole_ok

# ---- 打分 ----
def tuples(events, keep):
    trig, arg, evt = set(), set(), set()
    for e in events or []:
        et = e.get("event_type")
        if keep is not None and et not in keep: continue
        t = e.get("trigger") or {}; ts, te = t.get("start"), t.get("end")
        trig.add((et, ts, te)); aa = []
        for a in e.get("arguments") or []:
            arg.add((et, ts, te, a.get("role"), a.get("start"), a.get("end")))
            aa.append((a.get("role"), a.get("start"), a.get("end")))
        sa = tuple(sorted(aa, key=lambda x: (x[0] or "", -1 if x[1] is None else x[1], -1 if x[2] is None else x[2])))
        evt.add((et, ts, te, repr(sa)))
    return trig, arg, evt

def prf(pred, gold):
    tp = len(pred & gold); p = tp/len(pred) if pred else 0.0; r = tp/len(gold) if gold else 0.0
    return (0.0 if p+r == 0 else 2*p*r/(p+r), p, r, len(gold), len(pred), tp)

def score(part, keep):
    f = EVAL_DIR / part / "predictions.jsonl"
    if not f.exists(): return None, 0, 0
    gT, gA, gE, pT, pA, pE = (set() for _ in range(6)); nvalid = 0; ntot = 0
    for i, line in enumerate(open(f)):
        d = json.loads(line); ntot += 1
        gt, ga, ge = tuples((d.get("gold") or {}).get("events"), keep)
        surf, ok = extract_surface(d.get("generated_text", ""))
        if surf["events"]: nvalid += 1
        rec = recover(surf, d.get("input", ""))
        pt, pa, pe = tuples(rec["events"], keep)
        for s in gt: gT.add((i,)+s)
        for s in ga: gA.add((i,)+s)
        for s in ge: gE.add((i,)+s)
        for s in pt: pT.add((i,)+s)
        for s in pa: pA.add((i,)+s)
        for s in pe: pE.add((i,)+s)
    return {"trigger": prf(pT, gT), "argument": prf(pA, gA), "event": prf(pE, gE)}, nvalid, ntot

def line(m):
    f, p, r, ng, npd, tp = m
    return f"F1={f:.3f}(P{p:.3f}/R{r:.3f} tp{tp}/g{ng}/p{npd})"

if __name__ == "__main__":
    print("="*100); print("e83 迁移鲁棒重打分(修复 <final> 缺失 + 截断抢救)"); print("="*100)
    # seen 类型集合
    seen_all = set()
    fs = EVAL_DIR / "direct_test_seen" / "predictions.jsonl"
    if fs.exists():
        for l in open(fs):
            for e in (json.loads(l).get("gold") or {}).get("events", []): seen_all.add(e.get("event_type"))
    SEEN = seen_all - UNSEEN
    for tag, part, keep in [
        ("SEEN   (test_seen, 非unseen类型)", "e83_test_seen", SEEN),
        ("UNSEEN (test_unseen, 8 held-out)", "e83_test_unseen", UNSEEN),
        ("  └ sibling-present", "e83_test_unseen", SIBLING_PRESENT),
        ("  └ wholly-novel   ", "e83_test_unseen", WHOLLY_NOVEL),
    ]:
        r, nv, nt = score(part, keep)
        if r is None: print(f"{tag}: 无预测"); continue
        print(f"\n### {tag}  [surface抽取成功 {nv}/{nt}]")
        print(f"    Trig {line(r['trigger'])}")
        print(f"    Arg  {line(r['argument'])}")
        print(f"    Evt  {line(r['event'])}")
