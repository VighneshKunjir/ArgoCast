#!/usr/bin/env python
"""Paired comparison of Bi-LSTM runs (e.g. different --lambda-stab) on the SAME splits.

  python backend/scripts/compare_runs.py --base experiments --others experiments/lam50 experiments/lam100
Reads <dir>/splits.csv (written by experiment_uq_and_splits.py). The base run must
contain persistence rows; 'others' may have been run with --skip-gb.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ART = Path(__file__).resolve().parent.parent / "app" / "ml" / "artifacts"
COLS = ["rmse", "therm", "deep", "sal_rmse", "inv"]

def t_ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2: return m, np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    p = float(stats.ttest_1samp(x, 0).pvalue)
    return m, m - h, m + h, p

def paired(a, b, label):
    a, b = a.set_index("split"), b.set_index("split")
    idx = a.index.intersection(b.index)
    out = {}
    for c in COLS:
        d = (a.loc[idx, c] - b.loc[idx, c]).to_numpy()
        m, lo, hi, p = t_ci(d)
        out[c] = dict(mean_diff=m, ci_lo=lo, ci_hi=hi, p=p, first_lower_in=int((d < 0).sum()), of=len(d))
    print(f"\n{label}  (negative = first minus second is lower/better)")
    for c, r in out.items():
        print(f"  {c:9s} diff {r['mean_diff']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p']:.3f}  first lower in {r['first_lower_in']}/{r['of']}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ART / "experiments")
    ap.add_argument("--others", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    B = pd.read_csv(a.base / "splits.csv")
    base_l = B[B.model == "lstm"]; pers = B[B.model == "persistence"]
    res = {}
    for d in a.others:
        O = pd.read_csv(d / "splits.csv"); ol = O[O.model == "lstm"]
        s = ol.set_index("split")[COLS].agg(["mean", "std"]).round(3)
        print(f"\n== {d.name}: LSTM mean/SD over {len(ol)} splits\n{s}")
        res[d.name] = {"vs_base_lstm": paired(ol, base_l, f"{d.name} minus base LSTM"),
                       "vs_persistence": paired(ol, pers, f"{d.name} minus persistence")}
    (a.base / "compare_runs.json").write_text(json.dumps(res, indent=2, default=float))

if __name__ == "__main__":
    main()
