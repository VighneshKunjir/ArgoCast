#!/usr/bin/env python
"""Stage 3 - build forecasting windows and the train / validation / test splits.

Window definition (same task as FloatChat): 3 consecutive profiles of a float
(t-3, t-2, t-1) -> the next profile t. A window is only valid when every step
between consecutive profiles is between --min-gap and --max-gap days (default
3-20 days). This guarantees each window really is a ~10-day-ahead forecast and
not a jump across a data gap of months.

Two complementary splits are produced. Both are leakage-safe and the scalers
are later fitted on the training part only.

Split A - "chronological" (primary, operational)
  Windows are ordered by the date of their target profile.
    train : target before T1
    val   : T1 <= target < T2
    test  : target >= T2
  T1 and T2 are placed on 1 January boundaries closest to the 70 % and 85 %
  points of the window timeline (or set with --t1/--t2). A purge rule drops
  any window whose *input* profiles start before the boundary of its own
  split, so no input of a validation/test window was ever seen as a training
  target. This mimics real use: train on the past, forecast the future.

Split B - "unseen floats" (generalisation)
  Floats (not windows) are assigned 70 / 15 / 15 to train / val / test,
  stratified by each float's median year so every period is represented in
  every split. Test floats are never seen during training. Seed 42.

Outputs (data/processed/):
  windows.npz          X (n,3,32) physical units, yT, yS (n,16), metadata
  split_chronological.npz, split_unseen_floats.npz   index arrays
  split_summary.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import PROCESSED_DIR, SAL_COLS, TEMP_COLS, season_of, write_json

IN_CSV = PROCESSED_DIR / "profiles_arabian_2000_2026.csv.gz"


def build_windows(df: pd.DataFrame, min_gap: float, max_gap: float):
    X, yT, yS, meta = [], [], [], []
    n_gap_reject = 0
    for wmo, g in df.groupby("wmo", sort=True):
        g = g.sort_values("date")
        if len(g) < 4:
            continue
        t = g[TEMP_COLS].to_numpy(np.float32)
        s = g[SAL_COLS].to_numpy(np.float32)
        d = pd.to_datetime(g["date"]).to_numpy()
        gaps = np.diff(d).astype("timedelta64[s]").astype(float) / 86400.0
        lat, lon, cyc = g["latitude"].to_numpy(), g["longitude"].to_numpy(), g["cycle"].to_numpy()
        for i in range(len(g) - 3):
            gw = gaps[i:i + 3]
            if (gw < min_gap).any() or (gw > max_gap).any():
                n_gap_reject += 1
                continue
            X.append(np.concatenate([t[i:i + 3], s[i:i + 3]], axis=1))
            yT.append(t[i + 3]); yS.append(s[i + 3])
            meta.append((int(wmo), int(cyc[i + 3]), d[i], d[i + 3], float(lat[i + 3]), float(lon[i + 3]),
                         float(gw.mean())))
    m = pd.DataFrame(meta, columns=["wmo", "cycle", "input_start", "target_date", "lat", "lon", "mean_gap_days"])
    return np.stack(X), np.stack(yT), np.stack(yS), m, n_gap_reject


def auto_boundaries(dates: pd.Series, q1=0.70, q2=0.85):
    years = sorted(dates.dt.year.unique())
    starts = [pd.Timestamp(f"{y}-01-01") for y in years[1:]]
    frac = np.array([(dates < s).mean() for s in starts])
    t1 = starts[int(np.argmin(np.abs(frac - q1)))]
    later = [s for s in starts if s > t1]
    frac2 = np.array([(dates < s).mean() for s in later])
    t2 = later[int(np.argmin(np.abs(frac2 - q2)))] if later else t1
    return t1, t2


def chronological_split(m: pd.DataFrame, t1, t2):
    td, st = m["target_date"], m["input_start"]
    train = td < t1
    val = (td >= t1) & (td < t2) & (st >= t1)
    test = (td >= t2) & (st >= t2)
    purged = int(((td >= t1) & ~val & ~test).sum())
    return np.where(train)[0], np.where(val)[0], np.where(test)[0], purged


def unseen_float_split(m: pd.DataFrame, seed: int, frac=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    per = m.groupby("wmo").agg(year=("target_date", lambda x: x.dt.year.median()), n=("wmo", "size"))
    per["stratum"] = pd.qcut(per["year"].rank(method="first"), q=max(1, min(5, len(per) // 6)), labels=False)
    tr, va, te = [], [], []
    for _, g in per.groupby("stratum"):
        w = g.index.to_numpy().copy()
        rng.shuffle(w)
        n = len(w)
        n_te = max(1, int(round(frac[2] * n))) if n >= 3 else 0
        n_va = max(1, int(round(frac[1] * n))) if n >= 3 else 0
        te += list(w[:n_te]); va += list(w[n_te:n_te + n_va]); tr += list(w[n_te + n_va:])
    idx = lambda ws: np.where(m["wmo"].isin(ws))[0]
    return idx(tr), idx(va), idx(te), sorted(map(int, tr)), sorted(map(int, va)), sorted(map(int, te))


def describe(name, m, tr, va, te, extra):
    def part(ix):
        sub = m.iloc[ix]
        if len(sub) == 0:
            return {"windows": 0, "floats": 0}
        return {"windows": int(len(sub)), "floats": int(sub["wmo"].nunique()),
                "date_min": str(sub["target_date"].min().date()), "date_max": str(sub["target_date"].max().date()),
                "share_of_windows": round(len(sub) / len(m), 4)}
    out = {"name": name, "train": part(tr), "val": part(va), "test": part(te)}
    out.update(extra)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-gap", type=float, default=3.0)
    ap.add_argument("--max-gap", type=float, default=20.0)
    ap.add_argument("--t1", default="auto", help="chronological val start, e.g. 2020-01-01")
    ap.add_argument("--t2", default="auto", help="chronological test start, e.g. 2022-01-01")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(IN_CSV)
    print(f"Profiles: {len(df):,} from {df['wmo'].nunique()} floats")
    X, yT, yS, m, n_rej = build_windows(df, args.min_gap, args.max_gap)
    m["month"] = m["target_date"].dt.month
    m["season"] = m["month"].map(season_of)
    print(f"Windows: {len(X):,} (rejected for time gaps outside {args.min_gap}-{args.max_gap} d: {n_rej:,})")

    np.savez_compressed(
        PROCESSED_DIR / "windows.npz", X=X, yT=yT, yS=yS,
        wmo=m["wmo"].to_numpy(), cycle=m["cycle"].to_numpy(),
        target_date=m["target_date"].astype("datetime64[s]").to_numpy().astype("int64"),
        input_start=m["input_start"].astype("datetime64[s]").to_numpy().astype("int64"),
        lat=m["lat"].to_numpy(), lon=m["lon"].to_numpy(), month=m["month"].to_numpy(),
        season=m["season"].to_numpy().astype(str), gap=m["mean_gap_days"].to_numpy())

    if args.t1 == "auto" or args.t2 == "auto":
        t1, t2 = auto_boundaries(m["target_date"])
    else:
        t1, t2 = pd.Timestamp(args.t1), pd.Timestamp(args.t2)
    tr, va, te, purged = chronological_split(m, t1, t2)
    np.savez(PROCESSED_DIR / "split_chronological.npz", train=tr, val=va, test=te)
    sa = describe("chronological", m, tr, va, te,
                  {"t1_val_start": str(t1.date()), "t2_test_start": str(t2.date()),
                   "purged_boundary_windows": purged,
                   "floats_in_both_train_and_test": int(len(set(m.iloc[tr]["wmo"]) & set(m.iloc[te]["wmo"])))})

    tr2, va2, te2, wtr, wva, wte = unseen_float_split(m, args.seed)
    np.savez(PROCESSED_DIR / "split_unseen_floats.npz", train=tr2, val=va2, test=te2)
    sb = describe("unseen_floats", m, tr2, va2, te2,
                  {"seed": args.seed, "train_floats": wtr, "val_floats": wva, "test_floats": wte})

    summary = {"n_profiles": int(len(df)), "n_windows": int(len(X)), "gap_rejected_windows": int(n_rej),
               "gap_rule_days": [args.min_gap, args.max_gap],
               "windows_per_year": m["target_date"].dt.year.value_counts().sort_index().to_dict(),
               "chronological": sa, "unseen_floats": sb}
    write_json(PROCESSED_DIR / "split_summary.json", summary)
    for s in (sa, sb):
        print(f"[{s['name']}] " + " | ".join(f"{k}: {s[k]['windows']:,} win / {s[k]['floats']} floats"
                                             for k in ("train", "val", "test")))
    print(f"Chronological boundaries: val from {t1.date()}, test from {t2.date()} (purged {purged})")


if __name__ == "__main__":
    main()
