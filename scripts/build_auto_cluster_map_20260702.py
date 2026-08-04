#!/usr/bin/env python3
"""Derive per-type near-neighbor (confusable) clusters automatically from the event schema.

Score(T, T') = 0.5 * jaccard(trigger-cue token sets)
             + 0.3 * jaccard(core-role sets)
             + 0.2 * same-coarse-family indicator
Top-k highest-scoring neighbors per type. Purely schema-derived (no gold data, no hand-written
lists), so the same procedure applies to any ontology (RichERE, ACE05, ...). This makes the
hand-coded arbitration clusters of E81 a special case of an automatic construction (e95).
"""
import argparse
import json
from pathlib import Path


def cue_tokens(cues):
    toks = set()
    for cue in cues or []:
        for piece in cue.lower().replace("-", " ").replace("/", " ").split():
            if piece:
                toks.add(piece)
    return toks


def jaccard(a, b):
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def family(event_type):
    return event_type.split(":", 1)[0] if ":" in event_type else event_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--w_cue", type=float, default=0.5)
    ap.add_argument("--w_role", type=float, default=0.3)
    ap.add_argument("--w_family", type=float, default=0.2)
    args = ap.parse_args()

    schema = json.loads(Path(args.schema_path).read_text())
    types = [e["event_type"] for e in schema]
    cues = {e["event_type"]: cue_tokens(e.get("trigger_cues")) for e in schema}
    roles = {e["event_type"]: set(e.get("core_roles") or []) for e in schema}

    cluster_map = {}
    for t in types:
        scored = []
        for u in types:
            if u == t:
                continue
            s = (
                args.w_cue * jaccard(cues[t], cues[u])
                + args.w_role * jaccard(roles[t], roles[u])
                + args.w_family * (1.0 if family(t) == family(u) else 0.0)
            )
            scored.append((s, u))
        scored.sort(key=lambda x: (-x[0], x[1]))
        cluster_map[t] = [u for s, u in scored[: args.top_k] if s > 0]

    out = {
        "schema_path": args.schema_path,
        "top_k": args.top_k,
        "weights": {"cue": args.w_cue, "role": args.w_role, "family": args.w_family},
        "clusters": cluster_map,
    }
    Path(args.output_path).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_path} ({len(cluster_map)} types)")
    # quick sanity print for the families we studied
    for t in ["Contact:Broadcast", "Contact:Correspondence", "Contact:Meet", "Contact:Contact",
              "Justice:Arrest-Jail", "Life:Injure"]:
        if t in cluster_map:
            print(f"  {t:28s} -> {cluster_map[t]}")


if __name__ == "__main__":
    main()
