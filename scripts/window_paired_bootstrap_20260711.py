#!/usr/bin/env python3
"""窗口级 paired bootstrap(深度审稿 #7):
SG-CoT(5 seeds)× Direct(3 seeds)= 15 个配对,split1 test_unseen(82 窗),
corpus-micro exact-tuple scorer,窗口重采样 10k,报 ΔA/E/T 的 95% CI。
配对按行索引(两侧同源同序的 82 窗),先验证 gold 一致。
"""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "u", ROOT / "scripts/unified_scorer_table_20260709.py")
U = importlib.util.module_from_spec(spec); spec.loader.exec_module(U)

R = ROOT / "outputs/strengthen_20260709"

def load(dir_):
    rows = []
    for line in open(Path(dir_) / "predictions.jsonl"):
        rec = json.loads(line)
        def as_obj(v):
            if isinstance(v, str):
                try: return json.loads(v)
                except Exception: return {}
            return v
        rows.append((U.norm_events(as_obj(rec.get("gold"))),
                     U.norm_events(as_obj(rec.get("predicted")))))
    return rows

SG = {f"sg_{t}": R / f"mixed/e81_{t}/test_unseen" for t in ("base", "r1", "r2", "r3", "r4")}
DI = {"d_base": R / "new/direct_base/test_unseen",
      "d_r1": R / "mixed/direct_repeat1/test_unseen",
      "d_r2": R / "mixed/direct_repeat2/test_unseen"}

def gold_key(rows):
    return [tuple(sorted(k for k, _ in g)) for g, _ in rows]

def deltas(sg_rows, di_rows, idxs):
    a1, e1, t1 = U.score([sg_rows[i] for i in idxs])
    a2, e2, t2 = U.score([di_rows[i] for i in idxs])
    return a1 - a2, e1 - e2, t1 - t2

def main():
    rng = random.Random(20260711)
    NBOOT = 10000
    sg_loaded = {k: load(v) for k, v in SG.items()}
    di_loaded = {k: load(v) for k, v in DI.items()}
    ref = gold_key(next(iter(sg_loaded.values())))
    for k, rows in {**sg_loaded, **di_loaded}.items():
        assert gold_key(rows) == ref, f"gold 不对齐: {k}"
    n = len(ref)
    print(f"gold 对齐通过,n={n} 窗;{NBOOT} 次重采样,15 配对")
    print(f"{'pair':16s} {'ΔA [CI]':28s} {'ΔE [CI]':28s} {'ΔT [CI]':28s}")
    excl = [0, 0, 0]
    positive = [0, 0, 0]
    pair_results = []
    for sk, sr in sg_loaded.items():
        for dk, dr in di_loaded.items():
            samples = [[], [], []]
            for _ in range(NBOOT):
                idxs = [rng.randrange(n) for _ in range(n)]
                d = deltas(sr, dr, idxs)
                for j in range(3):
                    samples[j].append(d[j])
            cells = []
            metric_results = {}
            for j in range(3):
                s = sorted(samples[j])
                lo, hi = s[int(0.025 * NBOOT)], s[int(0.975 * NBOOT)]
                point = deltas(sr, dr, list(range(n)))[j]
                if lo > 0: excl[j] += 1
                if point > 0: positive[j] += 1
                cells.append(f"{point:+.3f} [{lo:+.3f},{hi:+.3f}]")
                metric_results[("argument", "event", "trigger")[j]] = {
                    "point": point,
                    "lower_95": lo,
                    "upper_95": hi,
                }
            print(f"{sk}x{dk:8s} {cells[0]:28s} {cells[1]:28s} {cells[2]:28s}")
            pair_results.append({
                "sg_seed": sk,
                "direct_seed": dk,
                "metrics": metric_results,
            })
    print(f"\nCI 全体>0 的配对数(A/E/T): {excl[0]}/15  {excl[1]}/15  {excl[2]}/15")
    out = {
        "n_windows": n,
        "n_boot": NBOOT,
        "bootstrap_seed": 20260711,
        "scorer": "corpus_micro_exact_tuple",
        "pair_count": len(pair_results),
        "point_gt0_pairs": {"A": positive[0], "E": positive[1], "T": positive[2]},
        "ci_gt0_pairs": {"A": excl[0], "E": excl[1], "T": excl[2]},
        "pairs": pair_results,
    }
    (ROOT / "reports/artifacts/2026-07-11_window_paired_bootstrap.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

if __name__ == "__main__":
    main()
