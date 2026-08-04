#!/usr/bin/env python3
"""E1: teacher zero-shot external baseline.

Prompts a large general-purpose LLM (the CoT teacher, glm-5.1) with the SAME candidate-conditioned
instruction+input as the fine-tuned models and parses its strict-JSON extraction. External anchor:
"can a much larger general LLM simply do this task zero-shot?" — no training, API only.
"""
import argparse
import collections
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


def call_once(client, model, instruction, text_input, timeout, max_tokens):
    sys_prompt = (
        "You are an information extraction engine. Follow the task instruction exactly and return "
        "ONLY one strict JSON object, no markdown, no code fences, no commentary."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": instruction + "\n\n" + text_input},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_json_obj(s):
    if not s:
        return None
    s = re.sub(r"^```(json)?|```$", "", s.strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="glm-5.1")
    ap.add_argument("--base_url", default="${LLM_BASE_URL}")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=os.environ["OPENAI_API_KEY"])
    rows = [json.loads(l) for l in open(args.eval_jsonl)]
    preds_path = os.path.join(args.output_dir, "predictions.jsonl")

    # resume: load existing predictions
    done = {}
    if os.path.exists(preds_path):
        for ln in open(preds_path):
            r = json.loads(ln)
            done[r["idx"]] = r

    def work(i):
        if i in done and done[i].get("parsed") is not None:
            return done[i]
        row = rows[i]
        content, parsed = "", None
        for attempt in range(args.max_retries):
            try:
                content = call_once(client, args.model, row["instruction"], row["input"],
                                    args.timeout, args.max_tokens)
                parsed = parse_json_obj(content)
                if parsed is not None:
                    break
            except Exception:
                pass
            time.sleep(2 + attempt * 3)
        return {"idx": i, "raw": content, "parsed": parsed,
                "gold": row["output"], "meta": row.get("meta", {})}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, range(len(rows))):
            results.append(r)
    results.sort(key=lambda r: r["idx"])
    with open(preds_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if r["parsed"] is not None)
    scored = [(norm_events(json.loads(r["gold"])), norm_events(r["parsed"] or {})) for r in results]
    A, E, T = score(scored)
    summary = {"label": args.label, "model": args.model, "n": len(rows), "parsed_ok": ok,
               "argument_f1": A, "event_f1": E, "trigger_f1": T}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.label}] n={len(rows)} parsed={ok}  A/E/T = {A:.4f}/{E:.4f}/{T:.4f}")


if __name__ == "__main__":
    main()
