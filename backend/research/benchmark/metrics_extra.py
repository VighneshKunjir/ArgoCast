"""Extended evaluation metrics for the ArgoCast 10-model benchmark.

Called by run_benchmark.py for every model and every seed, on the same test windows
as the main table. Two separate result groups:

  Metrics 1-7 (METRICS_1_7.md)
    1 Error size          RMSE, MAE, bias, max |error|, 95th / 99th percentile |error|
    2 Agreement           R2, anomaly correlation coefficient (ACC) against the
                          location-by-month training climatology
    3 Skill               skill vs persistence, skill vs climatology (location-by-month)
    4 Where errors occur  RMSE / bias / ACC by depth, thermocline and deep RMSE, RMSE by monsoon season
    5 Derived quantities  mixed layer depth, 20 degC isotherm depth (D20), depth of maximum
                          temperature gradient, ocean heat content 0-700 dbar
    6 Physical            density-inversion rate (any, > 0.01, > 0.03 kg m-3), inversion strength
    7 Uncertainty         point-forecast CRPS (= MAE) for every model; MC dropout for every
                          model with dropout: raw and validation-calibrated coverage, interval
                          width, Gaussian CRPS, spread-skill ratio

  Classification view (CLASSIFICATION_METRICS.md)
    Direction of change (warmer/colder, saltier/fresher than the last observed profile):
    accuracy, precision, recall, F1, macro-F1.
    Stability detection (does the profile contain an inversion?): accuracy, precision,
    recall, F1 and the confusion counts, at three thresholds.

The module has no dependency on run_benchmark.py; model lists, colours and labels are
passed in by the caller.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm

DEPTHS = np.array([5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000], float)
N_LEV = 16
THERM_IDX = [2, 3, 4, 5]                 # 50-150 dbar
DEEP_IDX = [10, 11, 12, 13, 14, 15]      # 500-1000 dbar
SEASONS = {  # Arabian Sea monsoon calendar (same as research/pipeline/common.py)
    "Winter monsoon (DJF)": [12, 1, 2],
    "Spring inter-monsoon (MAM)": [3, 4, 5],
    "Summer monsoon (JJAS)": [6, 7, 8, 9],
    "Fall inter-monsoon (ON)": [10, 11],
}
MLD_DSIG = 0.03                  # kg m-3, de Boyer Montegut et al. (2004) density criterion
OHC_IDX = np.arange(13)          # levels 5 ... 700 dbar
RHO0, CP0 = 1025.0, 3991.86795711963
INV_THR = {"any": None, "0.01": 0.01, "0.03": 0.03}   # 'any' = same rule as the main table
DEADBAND = {"T": 0.05, "S": 0.01}  # true changes smaller than this are ignored in the direction metric
THERM_SEARCH = (1, 10)           # gradient intervals between 20 and 500 dbar
RELIAB_NOM = np.arange(10, 100, 10) / 100
RELIAB_Z = [float(norm.ppf(0.5 + c / 2)) for c in RELIAB_NOM]
_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz

DEFINITIONS = {
    "climatology (reference)": "mean training-target profile for the same calendar month in the same lat/lon grid cell; "
                               "the cell size (1, 2 or 3 degrees) is chosen on validation temperature RMSE; cells with "
                               "fewer than the minimum number of training profiles fall back to the regional monthly mean",
    "bias": "mean of (forecast - truth); positive = too warm / too salty",
    "acc": "centred anomaly correlation, anomalies taken against the location-by-month climatology; computed per depth "
           "and averaged over the 16 depths; undefined for the climatology that is itself the reference",
    "skill": "1 - RMSE(model) / RMSE(reference forecast)",
    "mld": f"depth where sigma0 first exceeds its 5 dbar value by {MLD_DSIG} kg m-3 (linear interpolation between "
           "levels; 1000 dbar if never reached). Limited by the 16-level grid",
    "d20": "depth of the 20 degC isotherm (linear interpolation); windows where it is undefined in truth or forecast "
           "are excluded",
    "max_gradient_depth": "mid-depth of the steepest temperature drop between 20 and 500 dbar",
    "ohc": "rho0 * cp0 * integral of TEOS-10 conservative temperature over 0-700 dbar, in GJ m-2; the 0-5 dbar layer "
           "takes the 5 dbar value",
    "inversion (any)": "sigma0 decreases with depth between any adjacent levels (gradient < -1e-6 kg m-3 per dbar); "
                       "identical to the inversion rate in the main table",
    "inversion (> x)": "largest sigma0 drop between adjacent levels exceeds x kg m-3",
    "crps_point": "CRPS of a single-valued forecast, which equals its MAE",
    "mc_dropout": "dropout kept active at test time; mean and standard deviation over the passes; Gaussian intervals",
    "calibration": "one scale factor k per variable, fitted on the validation set so that 95% validation coverage is "
                   "95%, then applied unchanged to the test set",
    "spread_skill": "RMS predicted standard deviation / RMSE of the MC mean; 1 is ideal, below 1 = over-confident",
    "direction": "sign of (forecast - last observed profile) vs sign of (truth - last observed profile), per depth; "
                 f"true changes smaller than {DEADBAND['T']} degC / {DEADBAND['S']} PSU are ignored",
    "stability detection": "positive class = profile contains an inversion above the threshold",
}


# ----------------------------------------------------------------- basics
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def nanfloat(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def r2(p, t):
    ss = np.sum((t - t.mean()) ** 2)
    return float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else float("nan")


def acc(pa, ta):
    pa = pa - pa.mean()
    ta = ta - ta.mean()
    den = np.sqrt((pa ** 2).sum() * (ta ** 2).sum())
    return float((pa * ta).sum() / den) if den > 0 else float("nan")


def prf(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum()); tn = int((~pred & ~true).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and p + r > 0 else float("nan")
    return {"precision": nanfloat(p), "recall": nanfloat(r), "f1": nanfloat(f),
            "accuracy": float((tp + tn) / max(tp + fp + fn + tn, 1)), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def gauss_crps(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def season_of(month):
    return np.array([next((s for s, ms in SEASONS.items() if mo in ms), "unknown") for mo in month])


# ------------------------------------------------------------ climatology
class LocationMonthClimatology:
    """Mean training profile per (grid cell, calendar month), with regional-monthly fallback."""

    def __init__(self, lat, lon, month, yT, yS, cell=2.0, min_n=10):
        self.cell, self.min_n = float(cell), int(min_n)
        yT, yS = np.asarray(yT, float), np.asarray(yS, float)
        gT, gS = yT.mean(0), yS.mean(0)
        self.regT = {m: yT[month == m].mean(0) if (month == m).any() else gT for m in range(1, 13)}
        self.regS = {m: yS[month == m].mean(0) if (month == m).any() else gS for m in range(1, 13)}
        keys = self._keys(lat, lon, month)
        self.cells = {}
        for k in set(keys):
            sel = np.array([kk == k for kk in keys])
            if sel.sum() >= self.min_n:
                self.cells[k] = (yT[sel].mean(0), yS[sel].mean(0))

    def _keys(self, lat, lon, month):
        return list(zip(np.floor(np.asarray(lat) / self.cell).astype(int),
                        np.floor(np.asarray(lon) / self.cell).astype(int), np.asarray(month).astype(int)))

    def predict(self, lat, lon, month):
        keys = self._keys(lat, lon, month)
        T = np.empty((len(keys), N_LEV)); S = np.empty_like(T); hit = np.zeros(len(keys), bool)
        for i, k in enumerate(keys):
            if k in self.cells:
                T[i], S[i] = self.cells[k]; hit[i] = True
            else:
                T[i], S[i] = self.regT[k[2]], self.regS[k[2]]
        return T, S, hit


def fit_climatology(train, val, cells=(1.0, 2.0, 3.0), min_n=10):
    """Pick the grid-cell size on validation temperature RMSE (training data only for the means)."""
    mtr, mva = train["meta"], val["meta"]
    scores, models = {}, {}
    for c in cells:
        clim = LocationMonthClimatology(mtr["lat"].to_numpy(float), mtr["lon"].to_numpy(float),
                                        mtr["month"].to_numpy(), train["yT"], train["yS"], c, min_n)
        T, _, _ = clim.predict(mva["lat"].to_numpy(float), mva["lon"].to_numpy(float), mva["month"].to_numpy())
        scores[c], models[c] = rmse(T, val["yT"]), clim
    best = min(scores, key=scores.get)
    return models[best], {"cell_deg": best, "min_profiles_per_cell": min_n,
                          "val_temp_rmse_by_cell": {f"{k:g}": v for k, v in scores.items()},
                          "n_cells_used": len(models[best].cells)}


# ------------------------------------------------------- derived quantities
def teos(T, S, lat, lon):
    import gsw
    p = np.broadcast_to(DEPTHS, T.shape)
    la = np.broadcast_to(lat[:, None], T.shape)
    lo = np.broadcast_to(lon[:, None], T.shape)
    SA = gsw.SA_from_SP(S, p, lo, la)
    CT = gsw.CT_from_t(SA, T, p)
    return gsw.sigma0(SA, CT), CT


def crossing_depth(v, target, increasing):
    past = v >= target[:, None] if increasing else v <= target[:, None]
    ok = past.any(1) & ~past[:, 0]
    k = np.where(ok, past.argmax(1), 1)
    i = np.arange(len(v))
    v0, v1 = v[i, k - 1], v[i, k]
    den = np.where(np.abs(v1 - v0) < 1e-12, 1e-12, v1 - v0)
    frac = np.clip((target - v0) / den, 0, 1)
    z = DEPTHS[k - 1] + frac * (DEPTHS[k] - DEPTHS[k - 1])
    return np.where(ok, z, np.nan)


def derived(T, S, lat, lon):
    sig, CT = teos(T, S, lat, lon)
    mld = crossing_depth(sig, sig[:, 0] + MLD_DSIG, increasing=True)
    mld_capped = np.isnan(mld) & (sig.max(1) < sig[:, 0] + MLD_DSIG)
    mld = np.where(mld_capped, DEPTHS[-1], mld)
    d20 = crossing_depth(T, np.full(len(T), 20.0), increasing=False)
    a, b = THERM_SEARCH
    grad = -(T[:, a + 1:b + 1] - T[:, a:b]) / np.diff(DEPTHS)[a:b]
    zmax = ((DEPTHS[a:b] + DEPTHS[a + 1:b + 1]) / 2)[grad.argmax(1)]
    ohc = RHO0 * CP0 * (CT[:, 0] * DEPTHS[0] + _trapz(CT[:, OHC_IDX], DEPTHS[OHC_IDX], axis=1)) / 1e9
    drop = np.clip(-np.diff(sig, axis=1), 0, None)
    unstable_any = ((np.diff(sig, axis=1) / np.diff(DEPTHS)) < -1e-6).any(1)
    return {"mld": mld, "mld_capped": mld_capped, "d20": d20, "zmax_grad": zmax, "ohc": ohc,
            "max_drop": drop.max(1), "sum_drop": drop.sum(1), "unstable_any": unstable_any}


def unstable(d, key):
    return d["unstable_any"] if INV_THR[key] is None else d["max_drop"] > INV_THR[key]


def err_stats(p, t):
    e = (p - t).ravel()
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {"rmse": None, "mae": None, "bias": None, "n": 0}
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))), "bias": float(e.mean()), "n": int(len(e))}


# --------------------------------------------------------------- context
def make_context(test, clim):
    """Everything about the test set that does not depend on the model."""
    m = test["meta"]
    lat, lon, month = m["lat"].to_numpy(float), m["lon"].to_numpy(float), m["month"].to_numpy()
    tT, tS = test["yT"].astype(float), test["yS"].astype(float)
    cT, cS, hit = clim.predict(lat, lon, month)
    lastT, lastS = test["X"][:, 2, :N_LEV].astype(float), test["X"][:, 2, N_LEV:].astype(float)
    truth = derived(tT, tS, lat, lon)
    ctx = {"tT": tT, "tS": tS, "lat": lat, "lon": lon, "month": month, "season": season_of(month),
           "cT": cT, "cS": cS, "lastT": lastT, "lastS": lastS, "truth": truth,
           "rmse_pers": {"T": rmse(lastT, tT), "S": rmse(lastS, tS)},
           "rmse_clim": {"T": rmse(cT, tT), "S": rmse(cS, tS)},
           "clim_cell_share": float(hit.mean())}
    ctx["observed"] = {
        "mld_mean_m": nanfloat(np.nanmean(truth["mld"])), "mld_std_m": nanfloat(np.nanstd(truth["mld"])),
        "mld_capped_pct": float(truth["mld_capped"].mean() * 100),
        "d20_mean_m": nanfloat(np.nanmean(truth["d20"])), "d20_std_m": nanfloat(np.nanstd(truth["d20"])),
        "d20_valid_pct": float(np.isfinite(truth["d20"]).mean() * 100),
        "max_gradient_depth_mean_m": float(truth["zmax_grad"].mean()),
        "ohc_mean_GJ_m2": float(truth["ohc"].mean()), "ohc_std_GJ_m2": float(truth["ohc"].std()),
        "inversion_rate_pct": {k: float(unstable(truth, k).mean() * 100) for k in INV_THR},
        "mean_max_drop_inverted": nanfloat(truth["max_drop"][truth["unstable_any"]].mean())
        if truth["unstable_any"].any() else 0.0,
        "season_counts": {s: int((ctx["season"] == s).sum()) for s in SEASONS},
    }
    return ctx


# ------------------------------------------------------------ metrics 1-6
def metrics_1_6(pT, pS, ctx, is_reference_clim=False):
    pT, pS = np.asarray(pT, float), np.asarray(pS, float)
    out = {}
    for v, p, t, c in (("T", pT, ctx["tT"], ctx["cT"]), ("S", pS, ctx["tS"], ctx["cS"])):
        e = p - t
        ae = np.abs(e).ravel()
        g = {"rmse": rmse(p, t), "mae": float(ae.mean()), "bias": float(e.mean()),
             "max_abs_error": float(ae.max()), "p95_abs_error": float(np.percentile(ae, 95)),
             "p99_abs_error": float(np.percentile(ae, 99)), "r2": r2(p, t)}
        pa, ta = p - c, t - c
        g["acc"] = None if is_reference_clim else nanfloat(np.nanmean([acc(a, b) for a, b in zip(pa.T, ta.T)]))
        g["skill_vs_persistence"] = 1 - g["rmse"] / ctx["rmse_pers"][v]
        g["skill_vs_climatology"] = 1 - g["rmse"] / ctx["rmse_clim"][v]
        g["thermocline_rmse"] = rmse(p[:, THERM_IDX], t[:, THERM_IDX])
        g["deep_rmse"] = rmse(p[:, DEEP_IDX], t[:, DEEP_IDX])
        g["per_depth_rmse"] = np.sqrt((e ** 2).mean(0)).tolist()
        g["per_depth_bias"] = e.mean(0).tolist()
        g["per_depth_acc"] = [None if is_reference_clim else nanfloat(acc(a, b)) for a, b in zip(pa.T, ta.T)]
        g["season_rmse"] = {s: rmse(p[ctx["season"] == s], t[ctx["season"] == s])
                            for s in SEASONS if (ctx["season"] == s).any()}
        out[v] = g
    dd = derived(pT, pS, ctx["lat"], ctx["lon"])
    tr = ctx["truth"]
    out["derived"] = {"mld_m": err_stats(dd["mld"], tr["mld"]), "d20_m": err_stats(dd["d20"], tr["d20"]),
                      "max_gradient_depth_m": err_stats(dd["zmax_grad"], tr["zmax_grad"]),
                      "ohc_0_700_GJ_m2": err_stats(dd["ohc"], tr["ohc"])}
    inv = dd["unstable_any"]
    out["physical"] = {"rate_pct": {k: float(unstable(dd, k).mean() * 100) for k in INV_THR},
                       "mean_max_drop_inverted": nanfloat(dd["max_drop"][inv].mean()) if inv.any() else 0.0,
                       "p95_max_drop": float(np.percentile(dd["max_drop"], 95)),
                       "mean_total_drop": float(dd["sum_drop"].mean())}
    out["crps_point"] = {"T": out["T"]["mae"], "S": out["S"]["mae"]}
    out["_derived_arrays"] = dd   # reused by the classification view, removed before saving
    return out


# ---------------------------------------------------------- classification
def classification(pT, pS, ctx, dd):
    pT, pS = np.asarray(pT, float), np.asarray(pS, float)
    out = {"direction": {}, "stability": {}}
    for v, p, t, last in (("T", pT, ctx["tT"], ctx["lastT"]), ("S", pS, ctx["tS"], ctx["lastS"])):
        dt, dp = t - last, p - last
        mask = np.abs(dt) > DEADBAND[v]
        if np.allclose(dp, 0) or not mask.any():
            out["direction"][v] = None
            continue
        up_t, up_p = dt[mask] > 0, dp[mask] > 0
        inc, dec = prf(up_p, up_t), prf(~up_p, ~up_t)
        f_vals = [x for x in (inc["f1"], dec["f1"]) if x is not None]
        out["direction"][v] = {"accuracy": inc["accuracy"], "precision_increase": inc["precision"],
                               "recall_increase": inc["recall"], "f1_increase": inc["f1"],
                               "precision_decrease": dec["precision"], "recall_decrease": dec["recall"],
                               "f1_decrease": dec["f1"], "macro_f1": nanfloat(np.mean(f_vals)) if f_vals else None,
                               "n": int(mask.sum()), "deadband": DEADBAND[v]}
    for k in INV_THR:
        out["stability"][k] = prf(unstable(dd, k), unstable(ctx["truth"], k))
    return out


# ------------------------------------------------------------- MC dropout
def has_dropout(model):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout) and mod.p > 0:
            return True
        if isinstance(mod, (torch.nn.LSTM, torch.nn.GRU)) and mod.dropout > 0:
            return True
    return False


@torch.no_grad()
def _mc(model, X, passes, bs=8192):
    model.train()  # dropout active; none of these models uses batch norm
    s = np.zeros((len(X), 2 * N_LEV)); s2 = np.zeros_like(s)
    for _ in range(passes):
        out = []
        for i in range(0, len(X), bs):
            a, b = model(X[i:i + bs])
            out.append(torch.cat([a, b], 1).double().cpu().numpy())
        o = np.concatenate(out)
        s += o; s2 += o ** 2
    model.eval()
    mu = s / passes
    return mu, np.sqrt(np.maximum(s2 / passes - mu ** 2, 0))


def mc_dropout(model, Xva, Xte, val, test, scaler, passes, seed):
    """Group 7b for one trained model. X tensors are standardised and on the model's device."""
    if not has_dropout(model):
        return None
    torch.manual_seed(seed)
    st = {}
    for k, X, part in (("val", Xva, val), ("test", Xte, test)):
        mu_z, sd_z = _mc(model, X, passes)
        muT, muS = scaler.inv(mu_z)
        st[k] = {"T": (muT, np.maximum(sd_z[:, :N_LEV] * scaler.t.scale_, 1e-9), part["yT"].astype(float)),
                 "S": (muS, np.maximum(sd_z[:, N_LEV:] * scaler.s.scale_, 1e-9), part["yS"].astype(float))}
    res = {"passes": passes}
    for v in ("T", "S"):
        mu_v, sd_v, y_v = st["val"][v]
        k95 = float(np.quantile(np.abs(y_v - mu_v) / sd_v, 0.95) / 1.96)
        mu, sd, y = st["test"][v]
        z = np.abs(y - mu) / sd
        r = {"mc_mean_rmse": rmse(mu, y), "mean_sd": float(sd.mean()),
             "spread_skill_ratio": float(np.sqrt((sd ** 2).mean()) / rmse(mu, y)), "calibration_scale_k": k95}
        for nom, zc in ((50, 0.6745), (90, 1.6449), (95, 1.96)):
            r[f"coverage{nom}_raw"] = float((z <= zc).mean())
            r[f"coverage{nom}_cal"] = float((z <= zc * k95).mean())
        r["width95_raw"] = float((2 * 1.96 * sd).mean())
        r["width95_cal"] = float((2 * 1.96 * sd * k95).mean())
        r["crps_raw"] = float(gauss_crps(mu, sd, y).mean())
        r["crps_cal"] = float(gauss_crps(mu, sd * k95, y).mean())
        r["reliability_raw"] = [float((z <= q).mean()) for q in RELIAB_Z]
        r["reliability_cal"] = [float((z <= q * k95).mean()) for q in RELIAB_Z]
        res[v] = r
    return res


# ------------------------------------------------------ flat rows (per seed)
def flat_1_7(r, uq):
    f = {}
    for v in ("T", "S"):
        for k in ("rmse", "mae", "bias", "max_abs_error", "p95_abs_error", "p99_abs_error", "r2", "acc",
                  "skill_vs_persistence", "skill_vs_climatology", "thermocline_rmse", "deep_rmse"):
            f[f"{v}_{k}"] = r[v][k]
        for s, val in r[v]["season_rmse"].items():
            f[f"{v}_rmse_{s.split(' (')[1].rstrip(')')}"] = val
    for k, s in r["derived"].items():
        f[f"{k}_rmse"], f[f"{k}_bias"] = s["rmse"], s["bias"]
    for k, val in r["physical"]["rate_pct"].items():
        f[f"inversion_rate_{k}_pct"] = val
    f["mean_max_drop_inverted"] = r["physical"]["mean_max_drop_inverted"]
    f["crps_point_T"], f["crps_point_S"] = r["crps_point"]["T"], r["crps_point"]["S"]
    for v in ("T", "S"):
        u = uq.get(v) if uq else None
        for k in ("coverage95_raw", "coverage95_cal", "width95_cal", "crps_raw", "crps_cal", "spread_skill_ratio"):
            f[f"{v}_mc_{k}"] = u[k] if u else None
    return f


def flat_cls(c):
    f = {}
    for v in ("T", "S"):
        d = c["direction"][v]
        for k in ("accuracy", "precision_increase", "recall_increase", "f1_increase", "macro_f1"):
            f[f"{v}_direction_{k}"] = d[k] if d else None
    for k, s in c["stability"].items():
        for m in ("accuracy", "precision", "recall", "f1"):
            f[f"stability_{k}_{m}"] = s[m]
    return f


def seed_summary(rows):
    """mean and std over seeds for every numeric key."""
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return out


# ---------------------------------------------------------------- reports
def _f(v, d=4, pct=False, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if pct:
        return f"{v * 100:+.1f}%" if sign else f"{v * 100:.1f}%"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def _ms(m, k, d=4, scale=1.0):
    s = m.get(k)
    if not s:
        return "-"
    return f"{s['mean'] * scale:.{d}f} ± {s['std'] * scale:.{d}f}"


def report_1_7(p, order, short):
    R, ob, U = p["models"], p["observed"], p["uq"]
    ms = [m for m in order if m in R]
    ci = p["climatology_reference"]
    L = [f"# Metrics 1-7: {p['tag']}", "",
         f"Test windows {p['n_test']:,}. Neural models: values from the first seed ({p['first_seed']}), the same "
         f"predictions as the main table; mean ± std over seeds {p['seeds']} is in the seed-stability section and in "
         "`metrics_1_7_seeds.csv`. Lower is better unless stated.", "",
         f"Climatology reference (for ACC, skill vs climatology and the 'Climatology (local)' row): "
         f"{ci['cell_deg']:g}° grid cells by calendar month, at least {ci['min_profiles_per_cell']} training profiles per "
         f"cell, chosen on validation (temperature RMSE by cell size: "
         + ", ".join(f"{k}° {v:.4f}" for k, v in ci["val_temp_rmse_by_cell"].items())
         + f"). {p['clim_cell_share'] * 100:.1f}% of test windows use a grid-cell mean; the rest use the regional monthly mean.", ""]

    for v, unit, name in (("T", "°C", "temperature"), ("S", "PSU", "salinity")):
        L += [f"## 1. Error size: {name} ({unit})", "",
              "| Model | RMSE | MAE | Bias | Max abs error | 95th pct abs error | 99th pct abs error |",
              "|---|---|---|---|---|---|---|"]
        for m in ms:
            g = R[m][v]
            L.append(f"| {m} | {_f(g['rmse'])} | {_f(g['mae'])} | {_f(g['bias'], sign=True)} | {_f(g['max_abs_error'], 3)} | "
                     f"{_f(g['p95_abs_error'])} | {_f(g['p99_abs_error'])} |")
        L.append("")

    L += ["## 2-3. Agreement and skill (higher is better)", "",
          "| Model | R² T | R² S | ACC T | ACC S | Skill vs persistence T | Skill vs persistence S | Skill vs climatology T | Skill vs climatology S |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in ms:
        T, S = R[m]["T"], R[m]["S"]
        L.append(f"| {m} | {_f(T['r2'])} | {_f(S['r2'])} | {_f(T['acc'], 3)} | {_f(S['acc'], 3)} | "
                 f"{_f(T['skill_vs_persistence'], pct=True, sign=True)} | {_f(S['skill_vs_persistence'], pct=True, sign=True)} | "
                 f"{_f(T['skill_vs_climatology'], pct=True, sign=True)} | {_f(S['skill_vs_climatology'], pct=True, sign=True)} |")
    L += ["", "ACC is not defined for the local climatology, because it is the reference (its anomaly is zero).", ""]

    seasons = [s for s in SEASONS if s in R[ms[0]]["T"]["season_rmse"]]
    L += ["## 4. Where errors occur", "", "### 4a. Layer RMSE", "",
          "| Model | Thermocline T (50-150 dbar, °C) | Deep T (500-1000 dbar, °C) | Thermocline S (PSU) | Deep S (PSU) |",
          "|---|---|---|---|---|"]
    for m in ms:
        T, S = R[m]["T"], R[m]["S"]
        L.append(f"| {m} | {_f(T['thermocline_rmse'])} | {_f(T['deep_rmse'])} | {_f(S['thermocline_rmse'])} | {_f(S['deep_rmse'])} |")
    L.append("")
    for v, unit in (("T", "°C"), ("S", "PSU")):
        L += [f"### 4b. RMSE by monsoon season ({'temperature' if v == 'T' else 'salinity'}, {unit})", "",
              "| Model | " + " | ".join(f"{s} (n={ob['season_counts'][s]})" for s in seasons) + " |",
              "|---|" + "---|" * len(seasons)]
        for m in ms:
            L.append(f"| {m} | " + " | ".join(_f(R[m][v]["season_rmse"][s]) for s in seasons) + " |")
        L.append("")
    for key, title, d in (("per_depth_rmse", "4c. Temperature RMSE by depth (°C)", 3),
                          ("per_depth_bias", "4d. Temperature bias by depth (°C, forecast - truth)", 3),
                          ("per_depth_acc", "4e. Temperature ACC by depth (higher is better)", 3)):
        L += [f"### {title}", "", "| dbar | " + " | ".join(short.get(m, m) for m in ms) + " |", "|---|" + "---|" * len(ms)]
        for k, z in enumerate(DEPTHS.astype(int)):
            L.append(f"| {z} | " + " | ".join(_f(R[m]["T"][key][k], d, sign=key.endswith("bias")) for m in ms) + " |")
        L.append("")
    L += ["Salinity per-depth RMSE, bias and ACC are in `metrics_1_7.json`.", ""]

    L += ["## 5. Derived oceanographic quantities", "",
          f"Observed test profiles: mixed layer depth {_f(ob['mld_mean_m'], 1)} ± {_f(ob['mld_std_m'], 1)} m "
          f"({ob['mld_capped_pct']:.1f}% deeper than 1000 dbar); D20 {_f(ob['d20_mean_m'], 1)} ± {_f(ob['d20_std_m'], 1)} m "
          f"(defined in {ob['d20_valid_pct']:.1f}% of profiles); depth of maximum gradient {ob['max_gradient_depth_mean_m']:.1f} m; "
          f"OHC 0-700 {ob['ohc_mean_GJ_m2']:.2f} ± {ob['ohc_std_GJ_m2']:.2f} GJ m⁻².", "",
          "| Model | MLD RMSE (m) | MLD bias (m) | D20 RMSE (m) | D20 bias (m) | Max-gradient depth RMSE (m) | OHC RMSE (GJ m⁻²) | OHC bias (GJ m⁻²) |",
          "|---|---|---|---|---|---|---|---|"]
    for m in ms:
        d = R[m]["derived"]
        L.append(f"| {m} | {_f(d['mld_m']['rmse'], 2)} | {_f(d['mld_m']['bias'], 2, sign=True)} | {_f(d['d20_m']['rmse'], 2)} | "
                 f"{_f(d['d20_m']['bias'], 2, sign=True)} | {_f(d['max_gradient_depth_m']['rmse'], 2)} | "
                 f"{_f(d['ohc_0_700_GJ_m2']['rmse'], 3)} | {_f(d['ohc_0_700_GJ_m2']['bias'], 3, sign=True)} |")
    L += ["", "The 16-level grid (5, 20, 50, 75, 100 ... dbar) limits the vertical resolution of MLD and of the "
          "maximum-gradient depth; D20 and OHC are interpolated and less affected.", ""]

    L += ["## 6. Physical consistency", "",
          "| Model | Inversions, any (%) | > 0.01 kg m⁻³ (%) | > 0.03 kg m⁻³ (%) | Mean largest drop, inverted profiles (kg m⁻³) | 95th pct largest drop (kg m⁻³) |",
          "|---|---|---|---|---|---|"]
    for m in ms:
        ph = R[m]["physical"]
        L.append(f"| {m} | " + " | ".join(f"{ph['rate_pct'][k]:.2f}" for k in INV_THR) +
                 f" | {_f(ph['mean_max_drop_inverted'])} | {_f(ph['p95_max_drop'])} |")
    L.append("| Observed (reference) | " + " | ".join(f"{ob['inversion_rate_pct'][k]:.2f}" for k in INV_THR) +
             f" | {_f(ob['mean_max_drop_inverted'])} | - |")
    L.append("")

    L += ["## 7. Uncertainty", "", "### 7a. Point-forecast CRPS (all models)", "",
          "For a single-valued forecast the CRPS equals the MAE, so every model is on the same scale.", "",
          "| Model | CRPS T (°C) | CRPS S (PSU) |", "|---|---|---|"]
    for m in ms:
        L.append(f"| {m} | {_f(R[m]['crps_point']['T'])} | {_f(R[m]['crps_point']['S'])} |")
    L.append("")
    if U:
        for v, unit in (("T", "°C"), ("S", "PSU")):
            L += [f"### 7b. MC dropout, {'temperature' if v == 'T' else 'salinity'} ({unit}), {p['mc_passes']} passes", "",
                  "Nominal coverage 50 / 90 / 95%. 'cal' = after the scale factor k fitted on validation data.", "",
                  "| Model | MC-mean RMSE | Coverage raw 50/90/95 | Coverage cal 50/90/95 | k | 95% width raw | 95% width cal | CRPS raw | CRPS cal | Spread-skill |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for m in [m for m in order if m in U]:
                u = U[m][v]
                L.append(f"| {m} | {_f(u['mc_mean_rmse'])} | "
                         f"{u['coverage50_raw']:.2f} / {u['coverage90_raw']:.2f} / {u['coverage95_raw']:.2f} | "
                         f"{u['coverage50_cal']:.2f} / {u['coverage90_cal']:.2f} / {u['coverage95_cal']:.2f} | "
                         f"{u['calibration_scale_k']:.2f} | {_f(u['width95_raw'])} | {_f(u['width95_cal'])} | "
                         f"{_f(u['crps_raw'])} | {_f(u['crps_cal'])} | {u['spread_skill_ratio']:.2f} |")
            L.append("")
        no_mc = [m for m in ms if m not in U]
        L += ["No MC dropout for: " + ", ".join(no_mc) + " (no dropout layers or not a neural model; point forecasts only).", ""]
    else:
        L += ["MC dropout was not run.", ""]

    S = p["seed_stats"]
    if S:
        L += [f"## Seed stability (neural models, mean ± std over seeds {p['seeds']})", "",
              "| Model | Bias T (°C) | ACC T | Skill vs clim T | MLD RMSE (m) | D20 RMSE (m) | OHC RMSE (GJ m⁻²) | Inversions > 0.03 (%) | CRPS T, MC cal | Coverage 95 T, cal |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for m in [m for m in order if m in S]:
            s = S[m]
            L.append(f"| {m} | {_ms(s, 'T_bias')} | {_ms(s, 'T_acc', 3)} | {_ms(s, 'T_skill_vs_climatology', 1, 100)}% | "
                     f"{_ms(s, 'mld_m_rmse', 2)} | {_ms(s, 'd20_m_rmse', 2)} | {_ms(s, 'ohc_0_700_GJ_m2_rmse', 3)} | "
                     f"{_ms(s, 'inversion_rate_0.03_pct', 2)} | {_ms(s, 'T_mc_crps_cal')} | {_ms(s, 'T_mc_coverage95_cal', 3)} |")
        L += ["", "Every metric for every seed is in `metrics_1_7_seeds.csv`.", ""]

    L += ["## Definitions", ""] + [f"- **{k}**: {v}" for k, v in DEFINITIONS.items() if k not in ("direction", "stability detection")]
    L += ["", "## Files", "", "- `metrics_1_7.json`: every number above, including salinity per-depth values and reliability curves",
          "- `metrics_1_7.csv`: one row per model (first seed); `metrics_1_7_seeds.csv`: one row per model and seed",
          "- `figures/fig_m17_*.png|pdf`", ""]
    return "\n".join(L)


def report_cls(p, order):
    C = p["models"]
    ms = [m for m in order if m in C]
    L = [f"# Classification metrics: {p['tag']}", "",
         f"Test windows {p['n_test']:,}; first-seed predictions ({p['first_seed']}). These turn parts of the forecast into "
         "yes/no questions so that accuracy, precision, recall and F1 can be reported. They complement, and do not "
         "replace, the error metrics in METRICS_1_7.md. Higher is better.", "",
         "## A. Direction of change versus the last observed profile", "",
         f"At each depth: does the model predict correctly whether the next profile is warmer or colder (saltier or fresher) "
         f"than the last one? True changes smaller than {DEADBAND['T']} °C / {DEADBAND['S']} PSU are ignored. "
         "Persistence predicts no change, so the metric does not apply to it.", ""]
    for v, name, up, down in (("T", "Temperature", "warmer", "colder"), ("S", "Salinity", "saltier", "fresher")):
        L += [f"### {name}", "",
              f"| Model | Accuracy | Precision ({up}) | Recall ({up}) | F1 ({up}) | Precision ({down}) | Recall ({down}) | F1 ({down}) | Macro-F1 | n |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for m in ms:
            d = C[m]["direction"][v]
            if not d:
                L.append(f"| {m} | - | - | - | - | - | - | - | - | - |"); continue
            L.append(f"| {m} | {_f(d['accuracy'], 3)} | {_f(d['precision_increase'], 3)} | {_f(d['recall_increase'], 3)} | "
                     f"{_f(d['f1_increase'], 3)} | {_f(d['precision_decrease'], 3)} | {_f(d['recall_decrease'], 3)} | "
                     f"{_f(d['f1_decrease'], 3)} | {_f(d['macro_f1'], 3)} | {d['n']:,} |")
        L.append("")
    L += ["## B. Stability detection", "",
          "Does the forecast correctly say whether the real profile contains a density inversion? Positive class = unstable "
          "profile. A model that almost never forecasts instability (ArgoCast PI-BiLSTM, by design) gets high precision or "
          "undefined precision and low recall: most observed inversions are sensor noise or short-lived features that a "
          "10-day forecast is not meant to reproduce. Read this table together with the inversion rates in METRICS_1_7.md.", ""]
    for k, label in (("any", "any inversion"), ("0.01", "inversion larger than 0.01 kg m⁻³"),
                     ("0.03", "inversion larger than 0.03 kg m⁻³")):
        L += [f"### Threshold: {label}", "",
              "| Model | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|---|"]
        for m in ms:
            s = C[m]["stability"][k]
            L.append(f"| {m} | {_f(s['accuracy'], 3)} | {_f(s['precision'], 3)} | {_f(s['recall'], 3)} | {_f(s['f1'], 3)} | "
                     f"{s['tp']:,} | {s['fp']:,} | {s['fn']:,} | {s['tn']:,} |")
        L.append("")
    S = p["seed_stats"]
    if S:
        L += [f"## Seed stability (neural models, mean ± std over seeds {p['seeds']})", "",
              "| Model | T direction accuracy | T macro-F1 | S direction accuracy | S macro-F1 | Stability F1 (> 0.03) |",
              "|---|---|---|---|---|---|"]
        for m in [m for m in order if m in S]:
            s = S[m]
            L.append(f"| {m} | {_ms(s, 'T_direction_accuracy', 3)} | {_ms(s, 'T_direction_macro_f1', 3)} | "
                     f"{_ms(s, 'S_direction_accuracy', 3)} | {_ms(s, 'S_direction_macro_f1', 3)} | {_ms(s, 'stability_0.03_f1', 3)} |")
        L.append("")
    L += ["## Definitions", "", f"- **direction**: {DEFINITIONS['direction']}",
          f"- **stability detection**: {DEFINITIONS['stability detection']}; 'any' uses the same rule as the main-table "
          "inversion rate", "- **precision** = TP / (TP + FP); **recall** = TP / (TP + FN); **F1** = harmonic mean of the two; "
          "**macro-F1** = mean of the F1 of both classes", "",
          "## Files", "", "- `classification_metrics.json`, `classification_metrics.csv` (first seed), "
          "`classification_metrics_seeds.csv` (every seed)", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- figures
def figures(fig_dir, tag, p, order, short, colors, reference, note_fn=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})

    def save(fig, name):
        if note_fn:
            note_fn(fig)
        fig.savefig(fig_dir / f"{name}.png"); fig.savefig(fig_dir / f"{name}.pdf"); plt.close(fig)

    R = p["models"]
    ms = [m for m in order if m in R and not m.startswith("Climatology")]
    title = tag.replace("_", " ")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), sharey=True)
    for m in ms:
        st = dict(color=colors[m], lw=2.2 if m == reference else 1.0, ls="--" if m == "Persistence" else "-")
        axes[0].plot(R[m]["T"]["per_depth_bias"], DEPTHS, label=short.get(m, m), **st)
        axes[1].plot(R[m]["S"]["per_depth_bias"], DEPTHS, **st)
        axes[2].plot(np.array([np.nan if x is None else x for x in R[m]["T"]["per_depth_acc"]], float), DEPTHS, **st)
    for ax in axes[:2]:
        ax.axvline(0, color="black", lw=0.8)
    axes[0].set_ylim(1020, 0); axes[0].set_ylabel("Pressure (dbar)")
    axes[0].set_xlabel("Temperature bias (°C)"); axes[1].set_xlabel("Salinity bias (PSU)")
    axes[2].set_xlabel("Temperature anomaly correlation")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"Bias and anomaly correlation by depth - {title}", fontsize=9)
    fig.tight_layout(); save(fig, "fig_m17_bias_acc_depth")

    seasons = list(R[ms[0]]["T"]["season_rmse"].keys())
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        ax.bar(np.arange(len(seasons)) + i * w - 0.4 + w / 2, [R[m]["T"]["season_rmse"][s] for s in seasons],
               w, color=colors[m], edgecolor="black", lw=0.2, label=short.get(m, m))
    ax.set_xticks(range(len(seasons))); ax.set_xticklabels(seasons, fontsize=7)
    ax.set_ylabel("Temperature RMSE (°C)"); ax.legend(fontsize=6, ncol=4)
    ax.set_ylim(min(R[m]["T"]["season_rmse"][s] for m in ms for s in seasons) * 0.85, None)
    ax.set_title(f"Temperature RMSE by monsoon season - {title}", fontsize=9)
    save(fig, "fig_m17_seasonal_rmse")

    keys = [("mld_m", "Mixed layer depth RMSE (m)"), ("d20_m", "D20 RMSE (m)"), ("ohc_0_700_GJ_m2", "OHC 0-700 RMSE (GJ m$^{-2}$)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    x = np.arange(len(ms))
    for ax, (k, lab) in zip(axes, keys):
        v = np.array([R[m]["derived"][k]["rmse"] if R[m]["derived"][k]["rmse"] is not None else np.nan for m in ms])
        ax.bar(x, v, color=[colors[m] for m in ms], edgecolor="black", lw=0.3)
        ax.set_xticks(x); ax.set_xticklabels([short.get(m, m) for m in ms], rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(lab); ax.set_ylim(np.nanmin(v) * 0.85, None)
    fig.suptitle(f"Errors in derived oceanographic quantities - {title}", fontsize=9)
    fig.tight_layout(); save(fig, "fig_m17_derived")

    if p["uq"]:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
        for ax, v, lab in ((axes[0], "T", "Temperature"), (axes[1], "S", "Salinity")):
            ax.plot([0, 100], [0, 100], "k--", lw=0.8)
            for m in [m for m in order if m in p["uq"]]:
                u = p["uq"][m][v]
                ax.plot(RELIAB_NOM * 100, np.array(u["reliability_raw"]) * 100, color=colors[m], lw=0.8, ls=":")
                ax.plot(RELIAB_NOM * 100, np.array(u["reliability_cal"]) * 100, color=colors[m],
                        lw=2.0 if m == reference else 1.0, label=short.get(m, m))
            ax.set_xlabel("Nominal interval (%)"); ax.set_title(lab, fontsize=8)
        axes[0].set_ylabel("Observed coverage (%)"); axes[0].legend(fontsize=6)
        fig.suptitle(f"MC-dropout reliability (dotted raw, solid calibrated on validation) - {title}", fontsize=9)
        fig.tight_layout(); save(fig, "fig_m17_uq_reliability")
