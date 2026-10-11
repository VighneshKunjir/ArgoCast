#!/usr/bin/env python
"""
Reviewer-oriented experiments for FloatChat v6 (place in backend/scripts/).

  --mode coverage : calibration of the MC-dropout intervals of the SHIPPED model
                    on its own test floats (reads artifacts/metadata.json for the split).
  --mode splits   : repeated float-level splits (different seeds); retrains the
                    Bi-LSTM per split, evaluates persistence + gradient boosting
                    on identical windows, reports mean/SD/95% CI and paired differences.
  --mode all      : both.

Outputs (backend/app/ml/artifacts/experiments/): coverage.json, splits.csv,
splits_summary.json, tables.tex (LaTeX tables to paste into the paper).

Examples (from the repo root):
  python backend/scripts/experiment_uq_and_splits.py --mode coverage
  python backend/scripts/experiment_uq_and_splits.py --mode splits --n-splits 10 --epochs 60
  python backend/scripts/experiment_uq_and_splits.py --mode splits --n-splits 3 --skip-gb   # quick check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

DEPTHS = [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
TCOLS = [f"temp_{d}" for d in DEPTHS]
SCOLS = [f"sal_{d}" for d in DEPTHS]
ARCTIC = {6903058, 6903059, 6903060, 6903062, 6903063}  # atypical-regime floats (code label)
THERM = [2, 3, 4, 5]
DEEP = [10, 11, 12, 13, 14, 15]
ART = BACKEND / "app" / "ml" / "artifacts"
OUT = ART / "experiments"
LEVELS = [0.50, 0.80, 0.90, 0.95, 0.99]


# ----------------------------------------------------------------- device
def pick_device(name="auto"):
    """auto -> cuda if available, else cpu. Explicit: cuda, cuda:1, cpu."""
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False. "
                         "Your PyTorch build is probably CPU-only; install a CUDA build (see pytorch.org).")
    dev = torch.device(name)
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""))
    return dev


# ----------------------------------------------------------------- data
def load_df(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv).dropna(subset=TCOLS + SCOLS).reset_index(drop=True)
    ok = ((df[TCOLS].ge(-2) & df[TCOLS].le(35)).all(axis=1)
          & (df[SCOLS].ge(30) & df[SCOLS].le(42)).all(axis=1))
    return df[ok].reset_index(drop=True)


def windows(df: pd.DataFrame):
    """Physical-unit windows: X (n,3,32), yT (n,16), yS (n,16), wmo (n,)."""
    X, yT, yS, w = [], [], [], []
    for wmo, g in df.sort_values(["wmo", "cycle"]).groupby("wmo"):
        g = g.dropna(subset=TCOLS + SCOLS + ["cycle"]).reset_index(drop=True)
        if len(g) < 4:
            continue
        t = g[TCOLS].to_numpy(np.float32)
        s = g[SCOLS].to_numpy(np.float32)
        for i in range(len(g) - 3):
            X.append(np.concatenate([t[i:i + 3], s[i:i + 3]], axis=1))
            yT.append(t[i + 3]); yS.append(s[i + 3]); w.append(int(wmo))
    return np.stack(X), np.stack(yT), np.stack(yS), np.array(w)


def scale_X(X, ts, ss):
    Z = X.copy()
    n = len(X)
    Z[:, :, :16] = ts.transform(X[:, :, :16].reshape(-1, 16)).reshape(n, 3, 16)
    Z[:, :, 16:] = ss.transform(X[:, :, 16:].reshape(-1, 16)).reshape(n, 3, 16)
    return Z.astype(np.float32)


# ------------------------------------------------------------- pure stats
def rmse(a, b, idx=None):
    if idx is not None:
        a, b = a[:, idx], b[:, idx]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def inversion_pct(t, s):
    sig = 28.14 - 0.0735 * t - 0.00469 * t ** 2 + 0.802 * (s - 35.0)
    d = np.diff(sig, axis=1) / np.diff(np.array(DEPTHS, float))
    return float((d < -1e-6).any(axis=1).mean() * 100)


def metrics(yT, yS, pT, pS):
    return {"rmse": rmse(pT, yT), "mae": float(np.mean(np.abs(pT - yT))),
            "therm": rmse(pT, yT, THERM), "deep": rmse(pT, yT, DEEP),
            "sal_rmse": rmse(pS, yS), "inv": inversion_pct(pT, pS)}


def coverage_table(y, mu, sd, levels=LEVELS):
    """Empirical coverage of Gaussian intervals mu +/- z*sd for each nominal level."""
    sd = np.maximum(sd, 1e-9)
    z = np.abs(y - mu) / sd
    return {f"{l:.2f}": float(np.mean(z <= stats.norm.ppf((1 + l) / 2))) for l in levels}


def gaussian_crps(y, mu, sd):
    sd = np.maximum(sd, 1e-9)
    z = (y - mu) / sd
    return float(np.mean(sd * (z * (2 * stats.norm.cdf(z) - 1)
                               + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))))


def interval_score(y, mu, sd, level=0.95):
    a = 1 - level
    h = stats.norm.ppf(1 - a / 2) * np.maximum(sd, 1e-9)
    lo, hi = mu - h, mu + h
    return float(np.mean((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)))


def cluster_bootstrap(values_by_cluster, stat, n_boot=2000, seed=0):
    """values_by_cluster: list of 1-D arrays (one per float). Resample floats, not windows."""
    rng = np.random.default_rng(seed)
    k = len(values_by_cluster)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        out.append(stat(np.concatenate([values_by_cluster[i] for i in pick])))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def t_ci(x, conf=0.95):
    x = np.asarray(x, float)
    m, n = x.mean(), len(x)
    if n < 2:
        return float(m), float("nan"), float("nan"), float("nan")
    sd = x.std(ddof=1)
    h = stats.t.ppf((1 + conf) / 2, n - 1) * sd / np.sqrt(n)
    return float(m), float(sd), float(m - h), float(m + h)


# --------------------------------------------------------------- model utils
def mc_predict(model, Z, ts, ss, n_passes=50, bs=512, seed=42):
    import torch
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    model.train()  # dropout ON
    T, S = [], []
    with torch.no_grad():
        for _ in range(n_passes):
            pt, ps = [], []
            for i in range(0, len(Z), bs):
                a, b = model(torch.from_numpy(Z[i:i + bs]).to(dev))
                pt.append(a.cpu().numpy()); ps.append(b.cpu().numpy())
            T.append(ts.inverse_transform(np.concatenate(pt)))
            S.append(ss.inverse_transform(np.concatenate(ps)))
    model.eval()
    T, S = np.stack(T), np.stack(S)
    return T.mean(0), T.std(0, ddof=1), S.mean(0), S.std(0, ddof=1)


# ------------------------------------------------------------ coverage mode
def run_coverage(df, args):
    import joblib
    import torch
    from app.ml.model import PhysicsInformedBiLSTM

    meta = json.loads((ART / "metadata.json").read_text())
    pre = joblib.load(ART / "preprocessor.joblib")
    ts, ss = pre["temp_scaler"], pre["sal_scaler"]
    model = PhysicsInformedBiLSTM()
    model.load_state_dict(torch.load(ART / "model_weights.pt", map_location="cpu"))
    model.to(args.device_obj)

    res = {}
    sets = {}
    for name, key in (("val", "val_floats"), ("test", "test_floats")):
        X, yT, yS, w = windows(df[df["wmo"].isin(meta[key])])
        mT, sT, mS, sS = mc_predict(model, scale_X(X, ts, ss), ts, ss, args.passes)
        sets[name] = dict(yT=yT, yS=yS, mT=mT, sT=sT, mS=mS, sS=sS, w=w)
        print(f"{name}: {len(X)} windows, {len(set(w))} floats")

    # variance scaling factor fitted on VALIDATION floats only (no test leakage):
    # s = empirical 95th percentile of |err|/sd divided by the Gaussian 1.96
    scale = {}
    for var, (y, m, s) in {"temp": ("yT", "mT", "sT"), "sal": ("yS", "mS", "sS")}.items():
        v = sets["val"]
        z = np.abs(v[y] - v[m]) / np.maximum(v[s], 1e-9)
        scale[var] = float(np.quantile(z, 0.95) / stats.norm.ppf(0.975))
    res["inflation_factor_from_val"] = scale

    t = sets["test"]
    for var, (y, m, s) in {"temp": ("yT", "mT", "sT"), "sal": ("yS", "mS", "sS")}.items():
        sd_raw, sd_adj = t[s], t[s] * scale[var]
        block = {
            "raw_coverage": coverage_table(t[y], t[m], sd_raw),
            "adjusted_coverage": coverage_table(t[y], t[m], sd_adj),
            "mean_width95_raw": float(np.mean(2 * 1.96 * sd_raw)),
            "mean_width95_adj": float(np.mean(2 * 1.96 * sd_adj)),
            "crps_raw": gaussian_crps(t[y], t[m], sd_raw),
            "crps_adj": gaussian_crps(t[y], t[m], sd_adj),
            "interval_score95_raw": interval_score(t[y], t[m], sd_raw),
            "interval_score95_adj": interval_score(t[y], t[m], sd_adj),
            "spread_skill_ratio": float(np.mean(sd_raw) / rmse(t[m], t[y])),
        }
        # per-depth 95% coverage (raw)
        zq = stats.norm.ppf(0.975)
        block["per_depth_cov95_raw"] = {
            str(d): float(np.mean(np.abs(t[y][:, i] - t[m][:, i]) <= zq * np.maximum(sd_raw[:, i], 1e-9)))
            for i, d in enumerate(DEPTHS)}
        # cluster bootstrap over test floats for the 95% coverage
        ind = (np.abs(t[y] - t[m]) <= zq * np.maximum(sd_raw, 1e-9)).astype(float)
        by = [ind[t["w"] == f].ravel() for f in sorted(set(t["w"]))]
        block["cov95_raw_ci_floatbootstrap"] = cluster_bootstrap(by, np.mean)
        res[var] = block
        print(f"[{var}] 95% coverage raw={block['raw_coverage']['0.95']:.3f} "
              f"adj={block['adjusted_coverage']['0.95']:.3f} "
              f"CI(raw)={block['cov95_raw_ci_floatbootstrap']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage.json").write_text(json.dumps(res, indent=2))
    return res


# ----------------------------------------------------------- reference mode
LAT_COLS = ("latitude", "lat", "LATITUDE")
LON_COLS = ("longitude", "lon", "LONGITUDE")


def windows_with_pos(df):
    """Same windows as windows(), plus target-profile lat/lon when the CSV has them."""
    latc = next((c for c in LAT_COLS if c in df.columns), None)
    lonc = next((c for c in LON_COLS if c in df.columns), None)
    X, yT, yS, w, la, lo = [], [], [], [], [], []
    for wmo, g in df.sort_values(["wmo", "cycle"]).groupby("wmo"):
        g = g.dropna(subset=TCOLS + SCOLS + ["cycle"]).reset_index(drop=True)
        if len(g) < 4:
            continue
        t = g[TCOLS].to_numpy(np.float32)
        s_ = g[SCOLS].to_numpy(np.float32)
        for i in range(len(g) - 3):
            X.append(np.concatenate([t[i:i + 3], s_[i:i + 3]], axis=1))
            yT.append(t[i + 3]); yS.append(s_[i + 3]); w.append(int(wmo))
            la.append(float(g[latc].iloc[i + 3]) if latc else np.nan)
            lo.append(float(g[lonc].iloc[i + 3]) if lonc else np.nan)
    return np.stack(X), np.stack(yT), np.stack(yS), np.array(w), np.array(la), np.array(lo)


def teos10_inversion_pct(t, s, lat, lon):
    """Percentage of profiles with any negative sigma0 gradient using the full TEOS-10 (gsw)."""
    try:
        import gsw
    except Exception:
        return None, "gsw not installed"
    p = np.array(DEPTHS, float)
    note = "per-profile lat/lon"
    if np.isnan(lat).all() or np.isnan(lon).all():
        lat = np.full(len(t), 17.0); lon = np.full(len(t), 65.0)
        note = "lat/lon columns not found; used 17N 65E for all profiles"
    bad = 0
    for k in range(len(t)):
        sa = gsw.SA_from_SP(s[k], p, lon[k], lat[k])
        ct = gsw.CT_from_t(sa, t[k], p)
        sig = gsw.sigma0(sa, ct)
        if (np.diff(sig) / np.diff(p) < -1e-6).any():
            bad += 1
    return 100.0 * bad / len(t), note


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def run_reference(df, args):
    """Evaluate the SHIPPED model and both baselines on the shipped (metadata.json) split."""
    import joblib
    import torch
    from app.ml.model import PhysicsInformedBiLSTM
    from app.ml.baselines import (persistence_predict, predict_gradient_boosting,
                                  train_gradient_boosting)

    meta = json.loads((ART / "metadata.json").read_text())
    pre = joblib.load(ART / "preprocessor.joblib")
    ts, ss = pre["temp_scaler"], pre["sal_scaler"]
    test_f = list(meta["test_floats"]); val_f = list(meta.get("val_floats", []))
    train_f = meta.get("train_floats") or sorted(set(df["wmo"]) - set(test_f) - set(val_f))

    Xtr, yTtr, yStr, _ = windows(df[df["wmo"].isin(train_f)])
    Xte, yTte, ySte, wte, la, lo = windows_with_pos(df[df["wmo"].isin(test_f)])
    print(f"reference split: {len(train_f)} train floats ({len(Xtr)} windows), "
          f"{len(test_f)} test floats ({len(Xte)} windows)")

    dev = args.device_obj
    model = PhysicsInformedBiLSTM()
    model.load_state_dict(torch.load(ART / "model_weights.pt", map_location="cpu"))
    model.to(dev).eval()
    Z = scale_X(Xte, ts, ss)
    pt, ps = [], []
    with torch.no_grad():
        for i in range(0, len(Z), 512):
            a, b = model(torch.from_numpy(Z[i:i + 512]).to(dev))
            pt.append(a.cpu().numpy()); ps.append(b.cpu().numpy())
    lT = ts.inverse_transform(np.concatenate(pt)); lS = ss.inverse_transform(np.concatenate(ps))

    preds = {"lstm": (lT, lS), "persistence": persistence_predict(Xte)}
    if not args.skip_gb:
        gb = train_gradient_boosting(Xtr, yTtr, yStr, seed=42)
        preds["gboost"] = predict_gradient_boosting(gb, Xte)

    res = {"n_test_windows": int(len(Xte)), "test_floats": [int(f) for f in test_f],
           "n_train_floats": len(train_f), "models": {}}
    for name, (a, b) in preds.items():
        m = metrics(yTte, ySte, a, b)
        m.update({"sal_mae": float(np.mean(np.abs(b - ySte))),
                  "temp_r2": r2(yTte, a), "sal_r2": r2(ySte, b)})
        inv_t, note = teos10_inversion_pct(a, b, la, lo)
        m["inv_teos10"] = inv_t
        res["models"][name] = m
        res["teos10_note"] = note
        print(f"{name:12s} RMSE {m['rmse']:.3f}  MAE {m['mae']:.3f}  therm {m['therm']:.3f}  "
              f"deep {m['deep']:.3f}  salRMSE {m['sal_rmse']:.4f}  inv(poly) {m['inv']:.1f}%  "
              f"inv(TEOS-10) {inv_t if inv_t is None else round(inv_t, 1)}%")
    obs_inv, _ = teos10_inversion_pct(yTte, ySte, la, lo)
    res["observed_targets_inv_teos10"] = obs_inv
    res["observed_targets_inv_poly"] = inversion_pct(yTte, ySte)
    print(f"observed test targets: inversions poly {res['observed_targets_inv_poly']:.1f}%  "
          f"TEOS-10 {obs_inv if obs_inv is None else round(obs_inv, 1)}%")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reference.json").write_text(json.dumps(res, indent=2))
    print("wrote", OUT / "reference.json")
    return res


# -------------------------------------------------------------- splits mode
def make_split(df, seed, n_test=5, n_val=6):
    from sklearn.model_selection import train_test_split
    wmos = sorted(df["wmo"].unique().tolist())
    arctic = sorted(set(wmos) & ARCTIC)
    arabian = sorted(set(wmos) - ARCTIC)
    rest, test = train_test_split(arabian, test_size=n_test, random_state=seed)
    train_ar, val_ar = train_test_split(rest, test_size=n_val - (1 if len(arctic) > 1 else 0),
                                        random_state=seed)
    val_ac = arctic[-1:] if len(arctic) > 1 else []
    train_ac = arctic[:-1] if len(arctic) > 1 else arctic
    return sorted(train_ar + train_ac), sorted(val_ar + val_ac), sorted(test)


def train_one(Xtr, yTtr, yStr, Xva, yTva, ySva, ts, ss, args, seed):
    import random
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from app.ml.loss import PhysicsConstrainedLoss
    from app.ml.model import PhysicsInformedBiLSTM

    dev = args.device_obj
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = PhysicsInformedBiLSTM().to(dev)
    crit = PhysicsConstrainedLoss(temp_mean=ts.mean_, temp_std=ts.scale_,
                                  sal_mean=ss.mean_, sal_std=ss.scale_,
                                  lambda_stability=args.lambda_stab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(TensorDataset(*(torch.from_numpy(a.astype(np.float32)) for a in (Xtr, yTtr, yStr))),
                    batch_size=args.batch_size, shuffle=True, generator=g)
    Xv = torch.from_numpy(Xva.astype(np.float32)).to(dev)
    best, best_state, bad = 1e9, None, 0
    for ep in range(args.epochs):
        model.train()
        for X, yT, yS in tl:
            X, yT, yS = X.to(dev), yT.to(dev), yS.to(dev)
            opt.zero_grad()
            pt, ps = model(X)
            loss, _ = crit(pt, yT, ps, yS)
            loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            pv, _ = model(Xv)
        v = rmse(ts.inverse_transform(pv.cpu().numpy()), ts.inverse_transform(yTva))
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, dev


def run_splits(df, args):
    import torch
    from app.ml.baselines import (persistence_predict, predict_gradient_boosting,
                                  train_gradient_boosting)
    from sklearn.preprocessing import StandardScaler

    rows, test_unit_rows = [], []
    for k in range(args.n_splits):
        seed = args.seed0 + k
        tr, va, te = make_split(df, seed)
        trd, vad, ted = (df[df["wmo"].isin(x)] for x in (tr, va, te))
        ts = StandardScaler().fit(trd[TCOLS].to_numpy(np.float64))
        ss = StandardScaler().fit(trd[SCOLS].to_numpy(np.float64))
        Xtr, yTtr, yStr, _ = windows(trd)
        Xva, yTva, ySva, _ = windows(vad)
        Xte, yTte, ySte, wte = windows(ted)
        sT = lambda a: ts.transform(a)
        sS = lambda a: ss.transform(a)
        print(f"split {k} (seed {seed}): train {len(Xtr)} val {len(Xva)} test {len(Xte)} windows; "
              f"test floats {te}")

        model, dev = train_one(scale_X(Xtr, ts, ss), sT(yTtr), sS(yStr),
                               scale_X(Xva, ts, ss), sT(yTva), sS(ySva), ts, ss, args, seed)
        with torch.no_grad():
            pt, ps = model(torch.from_numpy(scale_X(Xte, ts, ss)).to(dev))
        lT, lS = ts.inverse_transform(pt.cpu().numpy()), ss.inverse_transform(ps.cpu().numpy())
        pT, pS = persistence_predict(Xte)
        preds = {"lstm": (lT, lS), "persistence": (pT, pS)}
        if not args.skip_gb:
            gb = train_gradient_boosting(Xtr, yTtr, yStr, seed=seed)
            preds["gboost"] = predict_gradient_boosting(gb, Xte)
        for name, (a, b) in preds.items():
            m = metrics(yTte, ySte, a, b)
            rows.append({"split": k, "seed": seed, "model": name, "n_test_windows": len(Xte),
                         "test_floats": " ".join(map(str, te)), **m})
            # per-float squared errors for the pooled cluster bootstrap
            for f in sorted(set(wte)):
                ix = wte == f
                test_unit_rows.append({"split": k, "model": name, "wmo": f,
                                       "n": int(ix.sum()),
                                       "sse": float(((a[ix] - yTte[ix]) ** 2).sum())})
        print("   ", {r["model"]: round(r["rmse"], 3) for r in rows if r["split"] == k})

    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "splits.csv", index=False)
    pd.DataFrame(test_unit_rows).to_csv(OUT / "splits_float_units.csv", index=False)

    summ = {"n_splits": args.n_splits, "per_model": {}, "paired": {}}
    for mod, g in R.groupby("model"):
        summ["per_model"][mod] = {c: dict(zip(("mean", "sd", "ci_lo", "ci_hi"), t_ci(g[c])))
                                  for c in ("rmse", "mae", "therm", "deep", "sal_rmse", "inv")}
    for other in [m for m in ("persistence", "gboost") if m in set(R["model"])]:
        a = R[R.model == "lstm"].set_index("split")
        b = R[R.model == other].set_index("split")
        summ["paired"][f"lstm_minus_{other}"] = {}
        for c in ("rmse", "therm", "deep", "sal_rmse", "inv"):
            d = (a[c] - b[c]).to_numpy()
            m, sd, lo, hi = t_ci(d)
            p = float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 else float("nan")
            summ["paired"][f"lstm_minus_{other}"][c] = {
                "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "p_value": p,
                "lstm_better_in": int((d < 0).sum()), "of": len(d)}
    (OUT / "splits_summary.json").write_text(json.dumps(summ, indent=2))
    write_tables(summ, args)
    return summ


def write_tables(summ, args):
    names = {"persistence": "Persistence", "gboost": "Gradient boosting", "lstm": "FloatChat Bi-LSTM"}
    L = ["% auto-generated by experiment_uq_and_splits.py",
         "\\begin{table}[t]\\centering\\small",
         f"\\caption{{Mean $\\pm$ SD over {summ['n_splits']} random float-level splits (95\\,\\% $t$-interval of the mean in brackets).}}\\label{{tab:splits}}",
         "\\begin{tabular}{lccc}\\toprule", "Model & RMSE ($^\\circ$C) & Thermocline RMSE & Inversions (\\%)\\\\\\midrule"]
    for m, d in summ["per_model"].items():
        f = lambda c: f"{d[c]['mean']:.3f}$\\pm${d[c]['sd']:.3f} [{d[c]['ci_lo']:.3f},{d[c]['ci_hi']:.3f}]"
        L.append(f"{names.get(m, m)} & {f('rmse')} & {f('therm')} & {d['inv']['mean']:.1f}$\\pm${d['inv']['sd']:.1f}\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}", "",
          "\\begin{table}[t]\\centering\\small",
          "\\caption{Paired differences in RMSE (Bi-LSTM minus comparator; negative favours the Bi-LSTM).}\\label{tab:paired}",
          "\\begin{tabular}{lcccc}\\toprule", "Comparator & Mean diff. & 95\\,\\% CI & $p$ & Bi-LSTM better\\\\\\midrule"]
    for k, d in summ["paired"].items():
        r = d["rmse"]
        L.append(f"{names.get(k.split('_minus_')[1])} & {r['mean_diff']:+.3f} & [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                 f"& {r['p_value']:.3f} & {r['lstm_better_in']}/{r['of']}\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    cov = OUT / "coverage.json"
    if cov.exists():
        c = json.loads(cov.read_text())
        L += ["", "\\begin{table}[t]\\centering\\small",
              "\\caption{Empirical coverage of Monte Carlo dropout intervals on the test floats. ``Adjusted'' multiplies the standard deviation by a factor fitted on validation floats only.}\\label{tab:coverage}",
              "\\begin{tabular}{llccccc}\\toprule", "Variable & Intervals & 50\\% & 80\\% & 90\\% & 95\\% & 99\\%\\\\\\midrule"]
        for v in ("temp", "sal"):
            for kind in ("raw", "adjusted"):
                cc = c[v][f"{kind}_coverage"]
                L.append(f"{v} & {kind} & " + " & ".join(f"{100*cc[k]:.1f}" for k in ("0.50", "0.80", "0.90", "0.95", "0.99")) + "\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{table}"]
    (OUT / "tables.tex").write_text("\n".join(L))
    print(f"LaTeX tables -> {OUT/'tables.tex'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["reference", "coverage", "splits", "all"], default="all")
    p.add_argument("--csv", type=Path, default=BACKEND / "data" / "processed" / "argo_floats_canonical.csv")
    p.add_argument("--passes", type=int, default=50)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--seed0", type=int, default=100, help="first split seed (42 is the shipped split)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-stab", type=float, default=10.0)
    p.add_argument("--skip-gb", action="store_true")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    p.add_argument("--tag", default="", help="write outputs to experiments/<tag>/ (e.g. lam50)")
    args = p.parse_args()
    global OUT
    if args.tag:
        OUT = OUT / args.tag
    args.device_obj = pick_device(args.device)
    df = load_df(args.csv)
    if args.mode == "reference":
        run_reference(df, args)
    if args.mode in ("coverage", "all"):
        run_coverage(df, args)
    if args.mode in ("splits", "all"):
        run_splits(df, args)


if __name__ == "__main__":
    main()
