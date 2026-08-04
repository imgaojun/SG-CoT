#!/usr/bin/env python3
"""Test-time self-consistency for SG-CoT: sample N reasoning paths, vote on the final structure.

An event (trigger span + type) is kept if it appears in >= vote_k of N samples; its arguments
(role + span) are kept if they appear in >= vote_k of the samples that contain the event.
vote_k is selected on dev_seen (pre-committed), then applied to test splits unchanged.
Compares against the greedy single decode scored identically (consistent offset scorer).
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from src.stage2_quality_validation.eval_adaptive_route_generation_evidence import (  # noqa: E402
    recover_offsets_from_evidence,
)


def build_prompt(tokenizer, instruction, input_text):
    messages = [{"role": "user", "content": f"{instruction}\n{input_text}"}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{instruction}\n{input_text}"


def parse_final(text):
    m = re.search(r"<final>\s*(\{.*)", text, re.S)
    if not m:
        return None
    s = m.group(1).split("</final>")[0]
    try:
        return json.loads(s)
    except Exception:
        for cut in range(len(s), max(len(s) - 3000, 0), -1):
            if s[:cut].rstrip().endswith("}"):
                try:
                    return json.loads(s[:cut])
                except Exception:
                    continue
    return None


def norm_events(obj):
    """-> list of (ev_key, frozenset(arg_keys)); ev_key=(start,end,type), arg_key=(role,start,end)"""
    out = []
    if not isinstance(obj, dict):
        return out
    for e in obj.get("events") or []:
        if not isinstance(e, dict):
            continue
        t = e.get("event_type") or e.get("type")
        tr = e.get("trigger") or {}
        if isinstance(tr, str):
            tr = {"text": tr}
        if not isinstance(tr, dict):
            continue
        ek = (tr.get("start"), tr.get("end"), t)
        aks = frozenset((a.get("role") or "", a.get("start"), a.get("end"))
                        for a in (e.get("arguments") or []) if isinstance(a, dict))
        out.append((ek, aks))
    return out


def vote(sample_events, vote_k):
    """sample_events: list over samples of norm_events lists -> voted [(ek, argset)]"""
    ev_count = collections.Counter()
    arg_count = collections.defaultdict(collections.Counter)
    for evs in sample_events:
        seen_ek = set()
        for ek, aks in evs:
            if ek in seen_ek:
                continue
            seen_ek.add(ek)
            ev_count[ek] += 1
            for ak in aks:
                arg_count[ek][ak] += 1
    out = []
    for ek, c in ev_count.items():
        if c >= vote_k:
            args = frozenset(ak for ak, ac in arg_count[ek].items() if ac >= vote_k)
            out.append((ek, args))
    return out


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def score(pairs):
    t = [0, 0, 0]; a = [0, 0, 0]; e = [0, 0, 0]
    for gold, pred in pairs:
        gt = [ek for ek, _ in gold]; pt = [ek for ek, _ in pred]; m = []
        for k in pt:
            if k in gt and k not in m: t[0] += 1; m.append(k)
            else: t[1] += 1
        t[2] += len(gt) - sum(1 for k in gt if k in pt)
        ga = [(ek, ak) for ek, aks in gold for ak in aks]
        pa = [(ek, ak) for ek, aks in pred for ak in aks]
        m2 = []
        for x in pa:
            if x in ga and x not in m2: a[0] += 1; m2.append(x)
            else: a[1] += 1
        a[2] += len(ga) - sum(1 for x in ga if x in pa)
        m3 = []
        for ev in pred:
            if ev in gold and ev not in m3: e[0] += 1; m3.append(ev)
            else: e[1] += 1
        e[2] += len(gold) - sum(1 for ev in gold if ev in pred)
    return prf(*a)[2], prf(*e)[2], prf(*t)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--batch_rows", type=int, default=1, help="rows per generate call (each row expands to n_samples)")
    ap.add_argument("--vote_k", type=int, default=None, help="if set, report only this k; else sweep 1..N")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.eval_jsonl)]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    dump = open(Path(args.output_dir) / "samples.jsonl", "w")
    all_gold, all_greedy, all_samples = [], [], []
    for s in range(0, len(rows), args.batch_rows):
        batch = rows[s:s + args.batch_rows]
        prompts = [build_prompt(tok, r["instruction"], r["input"]) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            greedy = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    pad_token_id=tok.pad_token_id)
            sampled = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p,
                                     num_return_sequences=args.n_samples,
                                     pad_token_id=tok.pad_token_id)
        L = enc["input_ids"].shape[1]
        for j, r in enumerate(batch):
            gold = norm_events(json.loads(r.get("gold_output") or r["output"]))

            def to_offsets(text):
                surface = parse_final(text) or {}
                try:
                    recovered, _diag = recover_offsets_from_evidence(surface, r["input"])
                except Exception:
                    recovered = {}
                return surface, norm_events(recovered)

            gtext = tok.decode(greedy[j][L:], skip_special_tokens=True)
            gsurf, gev = to_offsets(gtext)
            sev, ssurf = [], []
            for k in range(args.n_samples):
                stext = tok.decode(sampled[j * args.n_samples + k][L:], skip_special_tokens=True)
                surf, evs = to_offsets(stext)
                sev.append(evs); ssurf.append(surf)
            all_gold.append(gold); all_greedy.append(gev); all_samples.append(sev)
            dump.write(json.dumps({"idx": s + j, "greedy_surface": gsurf, "sample_surfaces": ssurf,
                                   "greedy": [[list(ek), sorted((list(ak) for ak in aks), key=repr)] for ek, aks in gev],
                                   "samples": [[[list(ek), sorted((list(ak) for ak in aks), key=repr)] for ek, aks in evs] for evs in sev]},
                                  ensure_ascii=False) + "\n")
        if (s // args.batch_rows) % 10 == 0:
            print(f"  {s + len(batch)}/{len(rows)}", flush=True)
    dump.close()

    res = {"label": args.label, "n": len(rows), "n_samples": args.n_samples,
           "temperature": args.temperature}
    g = score(list(zip(all_gold, all_greedy)))
    res["greedy"] = {"A": g[0], "E": g[1], "T": g[2]}
    print(f"[{args.label}] greedy       A/E/T = {g[0]:.4f}/{g[1]:.4f}/{g[2]:.4f}")
    ks = [args.vote_k] if args.vote_k else list(range(1, args.n_samples + 1))
    res["vote"] = {}
    for k in ks:
        voted = [vote(sev, k) for sev in all_samples]
        v = score(list(zip(all_gold, voted)))
        res["vote"][k] = {"A": v[0], "E": v[1], "T": v[2]}
        print(f"[{args.label}] vote k={k}    A/E/T = {v[0]:.4f}/{v[1]:.4f}/{v[2]:.4f}")
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
