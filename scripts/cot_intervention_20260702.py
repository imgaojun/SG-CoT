#!/usr/bin/env python3
"""C: inference-time causal intervention on the CoT's type-arbitration step.

For each unseen-test window where SG-CoT predicted >=1 event, we force-decode the final output
under a controlled synthetic <thinking> whose arbitration step selects either:
  (i)  the model's ORIGINAL predicted type (consistency condition), or
  (ii) a WRONG near-neighbor type from the auto-derived cluster map (counterfactual condition).
Everything else in the thinking (trigger lock, structure) is identical between conditions.
If the emitted <final> follows the injected type in (ii) at a high rate, the verbalized arbitration
is causally upstream of the output (not post-hoc rationalization).
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer, instruction, input_text):
    messages = [{"role": "user", "content": f"{instruction}\n{input_text}"}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{instruction}\n{input_text}"


def synth_thinking(trigger, chosen, rejected):
    return (
        "Step 1: Candidate frame recall. Scanning the text in order, the mention "
        f"\"{trigger}\" evokes a locally supported event frame. "
        f"Step 2: Trigger anchor lock. I lock \"{trigger}\" as the exact minimal event-evoking "
        "lexical anchor; this locked anchor is final and will not move or be dropped. "
        f"Step 3: Contrastive type arbitration over the locked frame. Comparing {chosen} against "
        f"{rejected}, the local wording around \"{trigger}\" matches the schema cues of {chosen} "
        f"rather than {rejected}, so the type is set to {chosen}; the locked trigger is unchanged. "
        "Step 4: Local argument attachment. I attach the locally supported arguments for this frame "
        "from the surrounding wording. "
        f"Step 5: Final check. The locked trigger \"{trigger}\" is preserved with its exact minimal "
        f"span and the frame has type {chosen}; no supported frame is missing."
    )


def first_event(pred):
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except Exception:
            return None
    evs = (pred or {}).get("events") or []
    if not evs:
        return None
    e = evs[0]
    tr = e.get("trigger") or {}
    if isinstance(tr, str):
        tr = {"text": tr}
    return {"type": e.get("event_type"), "trigger": tr.get("text")}


def parse_final(text):
    m = re.search(r"(\{.*)", text, re.S)
    if not m:
        return None
    s = m.group(1)
    s = s.split("</final>")[0]
    try:
        return json.loads(s)
    except Exception:
        # try trimming to last closing brace
        for cut in range(len(s), max(len(s) - 2000, 0), -1):
            if s[:cut].rstrip().endswith("}"):
                try:
                    return json.loads(s[:cut])
                except Exception:
                    continue
    return None


def type_for_trigger(obj, trigger):
    evs = (obj or {}).get("events") or []
    for e in evs:
        tr = e.get("trigger") or {}
        if isinstance(tr, str):
            tr = {"text": tr}
        if (tr.get("text") or "").strip() == (trigger or "").strip():
            return e.get("event_type")
    return evs[0].get("event_type") if evs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--preds_jsonl", required=True, help="existing SG-CoT predictions.jsonl (source of rows + original predictions)")
    ap.add_argument("--cluster_map", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    clusters = json.loads(Path(args.cluster_map).read_text())["clusters"]

    rows = [json.loads(l) for l in open(args.preds_jsonl)]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    cases = []
    for i, r in enumerate(rows):
        fe = first_event(r.get("final_predicted") or r.get("predicted"))
        if not fe or not fe["type"] or not fe["trigger"]:
            continue
        cands = (r.get("meta") or {}).get("candidate_types") or []
        nbrs = [n for n in clusters.get(fe["type"], []) if n in cands and n != fe["type"]]
        if not nbrs:
            fam = fe["type"].split(":")[0]
            nbrs = [c for c in cands if c != fe["type"] and c.split(":")[0] == fam]
        if not nbrs:
            nbrs = [c for c in cands if c != fe["type"]]
        if not nbrs:
            continue
        cf = nbrs[0]
        base_prompt = build_prompt(tok, r["instruction"], r["input"])
        for cond, chosen, rejected in [("consistency", fe["type"], cf), ("counterfactual", cf, fe["type"])]:
            prefix = base_prompt + "<thinking>" + synth_thinking(fe["trigger"], chosen, rejected) + "</thinking>\n<final>"
            cases.append({"idx": i, "cond": cond, "injected": chosen, "orig": fe["type"],
                          "cf": cf, "trigger": fe["trigger"], "prompt": prefix})

    print(f"{len(cases)} forced-decode cases from {len(rows)} rows")
    outs = []
    for s in range(0, len(cases), args.batch_size):
        batch = cases[s:s + args.batch_size]
        enc = tok([c["prompt"] for c in batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, c in enumerate(batch):
            text = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            obj = parse_final(text)
            ftype = type_for_trigger(obj, c["trigger"]) if obj else None
            outs.append({**{k: c[k] for k in ("idx", "cond", "injected", "orig", "cf", "trigger")},
                         "final_type": ftype, "parsed": obj is not None, "gen_head": text[:300]})
        if (s // args.batch_size) % 10 == 0:
            print(f"  {s+len(batch)}/{len(cases)}")

    with open(Path(args.output_dir) / "intervention_results.jsonl", "w") as f:
        for o in outs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    def rate(cond, pred):
        sel = [o for o in outs if o["cond"] == cond and o["parsed"]]
        return (sum(1 for o in sel if pred(o)) / max(len(sel), 1), len(sel))

    fc, nc = rate("consistency", lambda o: o["final_type"] == o["injected"])
    ff, nf = rate("counterfactual", lambda o: o["final_type"] == o["injected"])
    fr, _ = rate("counterfactual", lambda o: o["final_type"] == o["orig"])
    summary = {"n_cases": len(cases), "consistency_follow": fc, "n_consistency_parsed": nc,
               "counterfactual_follow": ff, "n_counterfactual_parsed": nf,
               "counterfactual_resist_keep_orig": fr}
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
