#!/usr/bin/env python3
"""E86 - perturbation-based faithfulness test (causal intervention, GPU-only).

Tests whether the contrastive type-arbitration step CAUSALLY drives the trained
SG-CoT model's final structure, beyond the observational marker analysis (E-faith).

Design (Lanham et al. 2023 "corrupt-the-CoT" style, teacher-forced):
  For each test row we take the model's OWN natural reasoning trace (from the
  existing E81 predictions) and re-decode the <final> structure under four
  conditions, holding the input fixed:
    - natural        : the model's free decoding (read from predictions; the paper number)
    - forced_full    : teacher-force the model's full <thinking>, regen <final>
                       (controls for the forcing procedure: should match natural)
    - forced_noarb   : teacher-force <thinking> with the ARBITRATION sentences
                       deleted, regen <final>  (treatment)
    - forced_placebo : teacher-force <thinking> with an EQUAL NUMBER of
                       NON-arbitration sentences deleted (length/positional control)
  Causal effect of arbitration = forced_full - forced_noarb (identical except the
  arbitration content). If forced_placebo >> forced_noarb, the effect is specific
  to arbitration content, not to removing text.

Arbitration sentences are identified by the same contrastive-language detector used
in the marker analysis (>=1 contrastive phrase). Offsets are recovered from copied
evidence and scored with the project's Trigger/Argument/Event F1, identical to the
evidence eval.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_quality_validation.eval_adapter_generation import (  # noqa: E402
    build_prompt,
    merge_metric_dict,
    normalize_events,
    prf,
)
from src.stage2_quality_validation.eval_adaptive_route_generation import (  # noqa: E402
    extract_final_json,
)
from src.stage2_quality_validation.eval_adaptive_route_generation_evidence import (  # noqa: E402
    recover_offsets_from_evidence,
)

# Same contrastive-arbitration detector as analyze_cot_faithfulness_20260617.py.
CONTRAST = re.compile(
    r"\b(not\s|rather than|instead of|reject|competing|versus|\bvs\b|too broad|"
    r"finer|generic|over the|over Contact|distinguish|but it is|but the target|"
    r"not a |would be too|prefer\w* .* over)",
    re.I,
)

THINK_RE = re.compile(r"<thinking>(.*?)</thinking>(.*?)<final>", re.S | re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]


def excise(thinking: str):
    """Return (noarb_text, placebo_text, n_arb, n_total).

    noarb: drop every sentence containing contrastive arbitration language.
    placebo: drop an equal number of NON-arbitration sentences, evenly spaced
             (deterministic length/positional control).
    """
    sents = split_sentences(thinking)
    n = len(sents)
    arb_idx = {i for i, s in enumerate(sents) if CONTRAST.search(s)}
    nonarb_idx = [i for i in range(n) if i not in arb_idx]
    k = len(arb_idx)
    # evenly spaced k indices from the non-arbitration pool
    placebo_drop = set()
    if k and nonarb_idx:
        m = len(nonarb_idx)
        if k >= m:
            placebo_drop = set(nonarb_idx)
        else:
            placebo_drop = {nonarb_idx[round(j * (m - 1) / (k - 1)) if k > 1 else 0] for j in range(k)}
    noarb = " ".join(s for i, s in enumerate(sents) if i not in arb_idx)
    placebo = " ".join(s for i, s in enumerate(sents) if i not in placebo_drop)
    return noarb, placebo, k, n


def score(payload_text: str, input_text: str, gold_json: dict):
    surface = extract_final_json(payload_text)
    recovered, _ = recover_offsets_from_evidence(surface or {"events": []}, input_text)
    p_tr, p_ar, p_ev = normalize_events(recovered or {"events": []})
    g_tr, g_ar, g_ev = normalize_events(gold_json)
    return (
        prf(p_tr, g_tr)["f1"],
        prf(p_ar, g_ar)["f1"],
        prf(p_ev, g_ev)["f1"],
        surface is not None,
    )


def batched(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i : i + bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--model_path", required=True, help="full SG-CoT checkpoint (E81 ck273)")
    ap.add_argument("--pred_seen", required=True, help="existing E81 test_seen predictions.jsonl")
    ap.add_argument("--pred_unseen", required=True, help="existing E81 test_unseen predictions.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=640)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    CONDS = ["forced_full", "forced_noarb", "forced_placebo"]
    splits = {"test_seen": args.pred_seen, "test_unseen": args.pred_unseen}
    all_summary = {}

    for split_name, pred_path in splits.items():
        rows = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]
        jobs = []  # (row_i, cond, prompt_text)
        per_row = []  # bookkeeping
        for ri, r in enumerate(rows):
            gen = r.get("generated_text", "") or ""
            m = THINK_RE.search(gen)
            instr, inp = r["instruction"], r["input"]
            gold = r["gold"] if isinstance(r.get("gold"), dict) else json.loads(r["gold_output"])
            rec = {
                "row": ri,
                "natural_AET": [r.get("trigger_f1"), r.get("argument_f1"), r.get("event_f1")],
                "has_struct": bool(m),
                "n_arb": 0,
                "n_sent": 0,
                "input": inp,
                "gold": gold,
            }
            if m:
                think, mid = m.group(1), m.group(2)
                noarb, placebo, k, n = excise(think)
                rec["n_arb"], rec["n_sent"] = k, n
                base_prompt = build_prompt(tok, instr, inp)
                variants = {"forced_full": think, "forced_noarb": noarb, "forced_placebo": placebo}
                for cond in CONDS:
                    prefix = f"<thinking>{variants[cond]}</thinking>{mid}<final>"
                    jobs.append((ri, cond, base_prompt + prefix))
            per_row.append(rec)

        # batched teacher-forced decoding of the <final> structure
        results = {}  # (ri, cond) -> (t,a,e,valid)
        t0 = time.time()
        for batch in batched(jobs, args.batch_size):
            prompts = [b[2] for b in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            width = enc["input_ids"].shape[1]
            with torch.no_grad():
                outs = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            for bi, (ri, cond, _) in enumerate(batch):
                cont = tok.decode(outs[bi][width:], skip_special_tokens=True)
                payload = "<final>" + cont  # prefix ended at "<final>"; reconstruct one final block
                rrow = per_row[ri]
                results[(ri, cond)] = score(payload, rrow["input"], rrow["gold"])
        dt = time.time() - t0

        # attach + aggregate
        agg = {c: {"t": [], "a": [], "e": [], "valid": []} for c in ["natural"] + CONDS}
        agg_manip = {c: {"t": [], "a": [], "e": []} for c in ["natural"] + CONDS}  # rows with n_arb>=1
        for rec in per_row:
            ri = rec["row"]
            nat = rec["natural_AET"]
            if all(v is not None for v in nat):
                agg["natural"]["t"].append(nat[0]); agg["natural"]["a"].append(nat[1]); agg["natural"]["e"].append(nat[2])
                agg["natural"]["valid"].append(1.0)
                if rec["n_arb"] >= 1:
                    agg_manip["natural"]["t"].append(nat[0]); agg_manip["natural"]["a"].append(nat[1]); agg_manip["natural"]["e"].append(nat[2])
            rec["forced"] = {}
            for cond in CONDS:
                if (ri, cond) in results:
                    t, a, e, v = results[(ri, cond)]
                    rec["forced"][cond] = {"trigger_f1": t, "argument_f1": a, "event_f1": e, "valid": v}
                    agg[cond]["t"].append(t); agg[cond]["a"].append(a); agg[cond]["e"].append(e)
                    agg[cond]["valid"].append(1.0 if v else 0.0)
                    if rec["n_arb"] >= 1:
                        agg_manip[cond]["t"].append(t); agg_manip[cond]["a"].append(a); agg_manip[cond]["e"].append(e)

        def mean(xs):
            return round(sum(xs) / len(xs), 4) if xs else None

        summ = {"n_rows": len(rows), "n_with_struct": sum(1 for r in per_row if r["has_struct"]),
                "n_manipulated(n_arb>=1)": sum(1 for r in per_row if r["n_arb"] >= 1),
                "mean_n_arb": round(sum(r["n_arb"] for r in per_row) / len(per_row), 2),
                "mean_n_sent": round(sum(r["n_sent"] for r in per_row) / len(per_row), 2),
                "decode_sec": round(dt, 1), "conditions": {}, "conditions_manipulated_only": {}}
        for c in ["natural"] + CONDS:
            summ["conditions"][c] = {"trigger_f1": mean(agg[c]["t"]), "argument_f1": mean(agg[c]["a"]),
                                     "event_f1": mean(agg[c]["e"]),
                                     "valid_rate": mean(agg[c]["valid"]) if agg[c]["valid"] else None,
                                     "n": len(agg[c]["a"])}
            summ["conditions_manipulated_only"][c] = {"trigger_f1": mean(agg_manip[c]["t"]),
                                                      "argument_f1": mean(agg_manip[c]["a"]),
                                                      "event_f1": mean(agg_manip[c]["e"]),
                                                      "n": len(agg_manip[c]["a"])}
        all_summary[split_name] = summ
        with open(out / f"{split_name}_perrow.jsonl", "w", encoding="utf-8") as f:
            for rec in per_row:
                rec.pop("gold", None); rec.pop("input", None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{split_name}]", json.dumps(summ, ensure_ascii=False, indent=2))

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
