#!/usr/bin/env python3
"""Eval for the DEGREE-style baseline: generate template lines -> parse -> recover token offsets -> score.
Uses the same offset scorer as the other external baselines for identical evaluation."""
import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _prf(tp,fp,fn):
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); return p,r,2*p*r/max(p+r,1e-9)
def score(pairs):
    t=[0,0,0];a=[0,0,0];e=[0,0,0]
    for gold,pred in pairs:
        gt=[k for k,_ in gold];pt=[k for k,_ in pred];m=[]
        for k in pt:
            if k in gt and k not in m: t[0]+=1;m.append(k)
            else: t[1]+=1
        t[2]+=len(gt)-sum(1 for k in gt if k in pt)
        ga=[(k,x) for k,args in gold for x in args];pa=[(k,x) for k,args in pred for x in args];m2=[]
        for x in pa:
            if x in ga and x not in m2: a[0]+=1;m2.append(x)
            else: a[1]+=1
        a[2]+=len(ga)-sum(1 for x in ga if x in pa)
        m3=[]
        for ev in pred:
            if ev in gold and ev not in m3: e[0]+=1;m3.append(ev)
            else: e[1]+=1
        e[2]+=len(gold)-sum(1 for ev in gold if ev in pred)
    return _prf(*a)[2],_prf(*e)[2],_prf(*t)[2]


def build_prompt(tok, instruction, input_text):
    msgs = [{"role": "user", "content": f"{instruction}\n{input_text}"}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{instruction}\n{input_text}"


def tokens_of(input_text):
    m = re.search(r"Tokens:\n(.+?)\n\n", input_text, re.S)
    if not m:
        m = re.search(r"Tokens:\n(.+)", input_text)
    return m.group(1).split() if m else []


def find_span(tokens, text):
    """first token subsequence whose joined form equals text (case-insensitive, ws-normalized)."""
    if not text:
        return None
    tgt = " ".join(text.split()).lower()
    tl = [t.lower() for t in tokens]
    words = tgt.split()
    L = len(words)
    for i in range(0, len(tl) - L + 1):
        if " ".join(tl[i:i + L]) == tgt:
            return (i, i + L)
    # fallback: substring on the joined-token string via cumulative char map is overkill; try single-token contains
    for i, t in enumerate(tl):
        if tgt in t or t in tgt and len(t) > 2:
            return (i, i + 1)
    return None


def parse_events(text, tokens):
    events = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.upper().startswith("NONE"):
            continue
        parts = [p.strip() for p in line.split("||")]
        if not parts or ":" not in parts[0] and "trigger" not in line:
            continue
        etype = parts[0].strip()
        trig_text = None
        args = []
        for p in parts[1:]:
            if p.lower().startswith("trigger:"):
                trig_text = p.split(":", 1)[1].strip()
            elif ":" in p:
                role, val = p.split(":", 1)
                args.append((role.strip(), val.strip()))
        if not etype or trig_text is None:
            continue
        tspan = find_span(tokens, trig_text)
        if tspan is None:
            continue
        ev = {"event_type": etype, "trigger": {"text": trig_text, "start": tspan[0], "end": tspan[1]},
              "arguments": []}
        for role, val in args:
            asp = find_span(tokens, val)
            if asp:
                ev["arguments"].append({"role": role, "text": val, "start": asp[0], "end": asp[1]})
        events.append(ev)
    return {"events": events}


def norm(obj):
    out = []
    for e in obj.get("events", []) or []:
        tr = e.get("trigger") or {}
        ek = (tr.get("start"), tr.get("end"), e.get("event_type"))
        aks = frozenset((a.get("role") or "", a.get("start"), a.get("end")) for a in (e.get("arguments") or []))
        out.append((ek, aks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(args.eval_jsonl)]
    tok = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.adapter_path, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    preds = []
    for s in range(0, len(rows), args.batch_size):
        batch = rows[s:s + args.batch_size]
        prompts = [build_prompt(tok, r["instruction"], r["input"]) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        for j, r in enumerate(batch):
            txt = tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pe = parse_events(txt, tokens_of(r["input"]))
            preds.append((norm(json.loads(r["gold_output"])), norm(pe), txt))
        if (s // args.batch_size) % 10 == 0:
            print(f"  {s+len(batch)}/{len(rows)}", flush=True)

    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as f:
        for g, p, txt in preds:
            f.write(json.dumps({"gen": txt[:400]}, ensure_ascii=False) + "\n")
    A, E, T = score([(g, p) for g, p, _ in preds])
    json.dump({"n": len(rows), "argument_f1": A, "event_f1": E, "trigger_f1": T},
              open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)
    print(f"DEGREE {os.path.basename(args.eval_jsonl)}: {A:.4f}/{E:.4f}/{T:.4f}")


if __name__ == "__main__":
    main()
