"""Post-hoc checks on saved predictions (no retraining).

1. Thresholded density-inversion rates (TEOS-10): share of profiles with any
   adjacent-level drop in sigma0 larger than 0, 0.01, 0.03 kg m-3, and the
   median magnitude of the largest drop.
2. Float-level two-fold cross-conformal rescaling of MC-dropout intervals:
   test floats split into two halves; scale factor alpha (global and per depth)
   fitted on one half, coverage evaluated on the other, then swapped. Repeated
   for 20 random float partitions. Gaussian CRPS before and after.

Usage: python s07_postcheck.py <experiment_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import gsw
import numpy as np
from scipy.stats import norm

EXP = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
P = np.array([5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000], float)
W = np.load(EXP / "data/processed/windows.npz", allow_pickle=True)
MODELS = ["Persistence", "Ridge", "Gradient_Boosting", "Bi-LSTM_MSE_only", "FloatChat_Bi-LSTM"]
THR = [1e-6, 0.01, 0.03]


def sigma0(t, s, lat, lon):
    p = np.broadcast_to(P, t.shape)
    la = np.broadcast_to(lat[:, None], t.shape)
    lo = np.broadcast_to(lon[:, None], t.shape)
    sa = gsw.SA_from_SP(s, p, lo, la)
    return gsw.sigma0(sa, gsw.CT_from_t(sa, t, p))


def inv_stats(t, s, lat, lon):
    drop = -np.diff(sigma0(t, s, lat, lon), axis=1)  # positive = lighter below
    worst = drop.max(axis=1)
    out = {f"pct_gt_{thr:g}": float((worst > thr).mean() * 100) for thr in THR}
    inv = worst[worst > 1e-6]
    out["median_worst_drop_if_inverted"] = float(np.median(inv)) if inv.size else 0.0
    out["mean_inverted_pairs_per_profile"] = float((drop > 1e-6).sum(1).mean())
    return out


def crps_gauss(y, mu, sd):
    sd = np.maximum(sd, 1e-9)
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def cov(y, mu, sd, q):
    zc = norm.ppf(0.5 + q / 2)
    return float((np.abs(y - mu) <= zc * sd).mean() * 100)


results = {}
for split in ["chronological", "unseen_floats"]:
    idx = np.load(EXP / f"data/processed/split_{split}.npz")["test"]
    yT, yS = W["yT"][idx].astype(float), W["yS"][idx].astype(float)
    lat, lon, wmo = W["lat"][idx], W["lon"][idx], W["wmo"][idx]
    r = {"n_windows": int(len(idx)), "n_floats": int(len(np.unique(wmo)))}
    inv = {"Observed": inv_stats(yT, yS, lat, lon)}
    for m in MODELS:
        d = np.load(EXP / f"outputs/predictions/pred_{split}_{m}.npz")
        inv[m] = inv_stats(d["T"].astype(float), d["S"].astype(float), lat, lon)
    r["inversions"] = inv

    mc = np.load(EXP / f"outputs/predictions/mc_{split}.npz")
    rng = np.random.default_rng(0)
    floats = np.unique(wmo)
    calib = {}
    for var, y in (("T", yT), ("S", yS)):
        mu, sd = mc[f"mean{var}"].astype(float), mc[f"std{var}"].astype(float)
        raw = {f"cov{q}": cov(y, mu, sd, q / 100) for q in (50, 80, 90, 95)}
        raw["crps"] = float(crps_gauss(y, mu, sd).mean())
        raw["mean_width95"] = float((2 * 1.96 * sd).mean())
        acc = {k: [] for k in ("g50", "g80", "g90", "g95", "d90", "d95", "crps_g", "crps_d",
                               "w95_g", "w95_d", "alpha_g")}
        for rep in range(20):
            perm = rng.permutation(floats)
            halves = [perm[: len(perm) // 2], perm[len(perm) // 2:]]
            for a, b in ((0, 1), (1, 0)):
                fit = np.isin(wmo, halves[a])
                ev = np.isin(wmo, halves[b])
                ratio = np.abs(y[fit] - mu[fit]) / np.maximum(sd[fit], 1e-9)
                ag = np.quantile(ratio, 0.95) / 1.96
                ad = np.quantile(ratio, 0.95, axis=0) / 1.96
                for q in (50, 80, 90, 95):
                    acc[f"g{q}"].append(cov(y[ev], mu[ev], ag * sd[ev], q / 100))
                for q in (90, 95):
                    acc[f"d{q}"].append(cov(y[ev], mu[ev], ad * sd[ev], q / 100))
                acc["crps_g"].append(float(crps_gauss(y[ev], mu[ev], ag * sd[ev]).mean()))
                acc["crps_d"].append(float(crps_gauss(y[ev], mu[ev], ad * sd[ev]).mean()))
                acc["w95_g"].append(float((2 * 1.96 * ag * sd[ev]).mean()))
                acc["w95_d"].append(float((2 * 1.96 * ad * sd[ev]).mean()))
                acc["alpha_g"].append(float(ag))
        calib[var] = {"raw": raw,
                      "rescaled_mean": {k: float(np.mean(v)) for k, v in acc.items()},
                      "rescaled_sd": {k: float(np.std(v)) for k, v in acc.items()}}
        # per-depth coverage: raw and global-rescaled (alpha fitted on all test floats, descriptive only)
        ratio = np.abs(y - mu) / np.maximum(sd, 1e-9)
        calib[var]["raw_cov95_by_depth"] = ((ratio <= 1.96).mean(0) * 100).round(1).tolist()
    r["calibration"] = calib
    results[split] = r

(OUT / "postcheck_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=1)[:6000])
