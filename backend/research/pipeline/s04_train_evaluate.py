#!/usr/bin/env python
"""Stage 4 - train every model on both splits and evaluate on the held-out test set.

Models (all predict the full 16-level T and S profile of the next cycle)
  Persistence          next profile = last observed profile (t-1)
  Climatology          monthly mean profile of the training targets
  Ridge                linear regression on the 96 inputs (alpha = 1)
  Gradient Boosting    HistGradientBoosting, one regressor per output (32)
  Bi-LSTM (MSE only)   ArgoCast architecture, loss without physics terms (ablation)
  ArgoCast Bi-LSTM    ArgoCast architecture + full physics-constrained loss

Neural models are trained with several seeds (default 42, 43, 44); tables report
mean +/- std over seeds. Confidence intervals, per-depth and per-season results
use the first seed only (fixed in advance, not the best one) so the numbers are
not cherry-picked. 95 % CIs come from a float-level block bootstrap (1000
resamples): floats, not windows, are resampled because windows of one float are
correlated.

Monte-Carlo dropout (50 passes) is run for the ArgoCast model to check whether
its 90 % / 95 % intervals contain the truth as often as they claim.

Outputs: outputs/tables/metrics_*.json|csv, outputs/predictions/*.npz,
         outputs/models/*.pt, outputs/figures/*.png|pdf
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from common import (DEEP_IDX, FIG_DIR, MODEL_DIR, N_LEV, PALETTE, PRED_DIR, PROCESSED_DIR,
                    SEASONS, STANDARD_DEPTHS, TAB_DIR, THERM_IDX, environment_info,
                    inversion_rate_poly, inversion_rate_teos10, mae, r2, rmse, savefig,
                    set_seed, setup_matplotlib, write_json)
from app.ml.loss import PhysicsConstrainedLoss
from app.ml.model import PhysicsInformedBiLSTM

SPLITS = ["chronological", "unseen_floats"]
MODEL_ORDER = ["Persistence", "Climatology", "Ridge", "Gradient Boosting",
               "Bi-LSTM (MSE only)", "ArgoCast Bi-LSTM"]
DL_MODELS = {"Bi-LSTM (MSE only)": False, "ArgoCast Bi-LSTM": True}


# ------------------------------------------------------------------ data
def load_split(name):
    w = np.load(PROCESSED_DIR / "windows.npz", allow_pickle=False)
    s = np.load(PROCESSED_DIR / f"split_{name}.npz")
    meta = pd.DataFrame({
        "wmo": w["wmo"], "cycle": w["cycle"], "lat": w["lat"], "lon": w["lon"],
        "month": w["month"], "season": w["season"],
        "target_date": pd.to_datetime(w["target_date"], unit="s")})
    parts = {}
    for k in ("train", "val", "test"):
        ix = s[k]
        parts[k] = {"X": w["X"][ix], "yT": w["yT"][ix], "yS": w["yS"][ix],
                    "meta": meta.iloc[ix].reset_index(drop=True)}
    return parts


class Scaler:
    """Per-depth standardisation fitted on training targets only."""

    def __init__(self, yT, yS):
        self.t = StandardScaler().fit(yT.astype(np.float64))
        self.s = StandardScaler().fit(yS.astype(np.float64))

    def X(self, X):
        out = np.empty_like(X, dtype=np.float32)
        for k in range(X.shape[1]):
            out[:, k, :N_LEV] = self.t.transform(X[:, k, :N_LEV])
            out[:, k, N_LEV:] = self.s.transform(X[:, k, N_LEV:])
        return out

    def yT(self, y):
        return self.t.transform(y).astype(np.float32)

    def yS(self, y):
        return self.s.transform(y).astype(np.float32)

    def invT(self, y):
        return self.t.inverse_transform(y)

    def invS(self, y):
        return self.s.inverse_transform(y)


# ---------------------------------------------------------------- metrics
def metric_row(pT, pS, tT, tS, lat, lon):
    return {
        "temp_rmse": rmse(pT, tT), "temp_mae": mae(pT, tT), "temp_r2": r2(pT, tT),
        "therm_rmse": rmse(pT[:, THERM_IDX], tT[:, THERM_IDX]),
        "deep_rmse": rmse(pT[:, DEEP_IDX], tT[:, DEEP_IDX]),
        "sal_rmse": rmse(pS, tS), "sal_mae": mae(pS, tS), "sal_r2": r2(pS, tS),
        "violation_poly_pct": inversion_rate_poly(pT, pS),
        "violation_teos10_pct": inversion_rate_teos10(pT, pS, lat, lon),
    }


def float_bootstrap(err2_T, err2_S, groups, ref_T, B=1000, seed=0):
    """Block bootstrap over floats. err2_* are per-window mean squared errors."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    n_g = len(uniq)
    cnt = np.bincount(inv, minlength=n_g).astype(float)
    sT = np.bincount(inv, weights=err2_T, minlength=n_g)
    sS = np.bincount(inv, weights=err2_S, minlength=n_g)
    sR = np.bincount(inv, weights=ref_T, minlength=n_g)
    W = rng.multinomial(n_g, np.full(n_g, 1 / n_g), size=B).astype(float)
    n = W @ cnt
    rT, rS, rR = np.sqrt(W @ sT / n), np.sqrt(W @ sS / n), np.sqrt(W @ sR / n)
    skill = 1 - rT / rR
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"temp_rmse_ci": q(rT), "sal_rmse_ci": q(rS), "temp_skill_vs_persistence_ci": q(skill),
            "p_not_better_than_persistence": float((skill <= 0).mean())}


# --------------------------------------------------------------- training
def train_dl(parts, sc, physics: bool, seed: int, args, device):
    set_seed(seed)
    Xtr, Xva = sc.X(parts["train"]["X"]), sc.X(parts["val"]["X"])
    tr = TensorDataset(torch.tensor(Xtr), torch.tensor(sc.yT(parts["train"]["yT"])),
                       torch.tensor(sc.yS(parts["train"]["yS"])))
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, generator=g)
    model = PhysicsInformedBiLSTM().to(device)
    lam = dict(lambda_stability=10.0, lambda_therm=1.5) if physics else dict(lambda_stability=0.0, lambda_therm=0.0)
    crit = PhysicsConstrainedLoss(lambda_sal=2.5, temp_mean=sc.t.mean_, temp_std=sc.t.scale_,
                                  sal_mean=sc.s.mean_, sal_std=sc.s.scale_, **lam).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    Xva_t = torch.tensor(Xva, device=device)
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for X, yT, yS in loader:
            X, yT, yS = X.to(device), yT.to(device), yS.to(device)
            opt.zero_grad()
            pT, pS = model(X)
            loss, _ = crit(pT, yT, pS, yS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(X); n += len(X)
        sched.step()
        pT, pS = predict(model, Xva_t)
        vT = rmse(sc.invT(pT), parts["val"]["yT"])
        vS = rmse(sc.invS(pS), parts["val"]["yS"])
        hist.append({"epoch": ep, "train_loss": tot / n, "val_temp_rmse": vT, "val_sal_rmse": vS})
        if vT < best - 1e-6:
            best, bad = vT, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(f"      epoch {ep:3d} loss={tot/n:.4f} val T={vT:.4f} S={vS:.4f}")
        if bad >= args.patience:
            print(f"      early stop at epoch {ep}")
            break
    model.load_state_dict(best_state)
    return model, hist, time.time() - t0


@torch.no_grad()
def predict(model, X, bs=4096):
    model.eval()
    outT, outS = [], []
    for i in range(0, len(X), bs):
        a, b = model(X[i:i + bs])
        outT.append(a.cpu().numpy()); outS.append(b.cpu().numpy())
    return np.concatenate(outT), np.concatenate(outS)


@torch.no_grad()
def mc_dropout(model, X, passes, bs=4096):
    model.train()  # keep dropout active
    T, S = [], []
    for _ in range(passes):
        a, b = [], []
        for i in range(0, len(X), bs):
            p, q = model(X[i:i + bs]); a.append(p.cpu().numpy()); b.append(q.cpu().numpy())
        T.append(np.concatenate(a)); S.append(np.concatenate(b))
    model.eval()
    return np.stack(T), np.stack(S)


# ------------------------------------------------------------------ main
def run_split(name, args, device):
    print(f"\n=== Split: {name} ===")
    parts = load_split(name)
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(parts[k]['X']):,} windows, {parts[k]['meta']['wmo'].nunique()} floats")
    if min(len(parts[k]["X"]) for k in parts) == 0:
        raise SystemExit(f"Split {name} has an empty part; adjust s03_split.py options.")
    sc = Scaler(parts["train"]["yT"], parts["train"]["yS"])
    te = parts["test"]
    tT, tS = te["yT"], te["yS"]
    lat, lon = te["meta"]["lat"].to_numpy(), te["meta"]["lon"].to_numpy()
    preds, results, timings, histories = {}, {}, {}, {}

    # Persistence
    preds["Persistence"] = (te["X"][:, 2, :N_LEV].astype(float), te["X"][:, 2, N_LEV:].astype(float))
    # Climatology (monthly, from training targets)
    m_tr = parts["train"]["meta"]["month"].to_numpy()
    clT = {m: parts["train"]["yT"][m_tr == m].mean(0) for m in range(1, 13) if (m_tr == m).any()}
    clS = {m: parts["train"]["yS"][m_tr == m].mean(0) for m in range(1, 13) if (m_tr == m).any()}
    gT, gS = parts["train"]["yT"].mean(0), parts["train"]["yS"].mean(0)
    mt = te["meta"]["month"].to_numpy()
    preds["Climatology"] = (np.stack([clT.get(m, gT) for m in mt]), np.stack([clS.get(m, gS) for m in mt]))

    Xtr_f = sc.X(parts["train"]["X"]).reshape(len(parts["train"]["X"]), -1)
    Xte_s = sc.X(te["X"])
    Xte_f = Xte_s.reshape(len(Xte_s), -1)
    Ytr = np.concatenate([sc.yT(parts["train"]["yT"]), sc.yS(parts["train"]["yS"])], axis=1)

    t0 = time.time()
    rg = Ridge(alpha=1.0).fit(Xtr_f, Ytr)
    p = rg.predict(Xte_f)
    preds["Ridge"] = (sc.invT(p[:, :N_LEV]), sc.invS(p[:, N_LEV:]))
    timings["Ridge"] = time.time() - t0

    if not args.skip_gb:
        print("  Gradient Boosting (32 regressors) ...")
        t0 = time.time()
        # n_jobs=1: HistGradientBoosting already uses every core through OpenMP. Extra worker
        # processes each reload torch's CUDA DLLs on Windows and exhaust the paging file.
        gb = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=args.gb_iter, random_state=42),
                                  n_jobs=1).fit(Xtr_f, Ytr)
        p = gb.predict(Xte_f)
        preds["Gradient Boosting"] = (sc.invT(p[:, :N_LEV]), sc.invS(p[:, N_LEV:]))
        timings["Gradient Boosting"] = time.time() - t0

    Xte_t = torch.tensor(Xte_s, device=device)
    seed_rows = {}
    for mname, physics in DL_MODELS.items():
        seed_rows[mname] = []
        for j, seed in enumerate(args.seeds):
            print(f"  {mname} seed {seed} ...")
            model, hist, el = train_dl(parts, sc, physics, seed, args, device)
            pT, pS = predict(model, Xte_t)
            pT, pS = sc.invT(pT), sc.invS(pS)
            row = metric_row(pT, pS, tT, tS, lat, lon)
            row.update(seed=seed, epochs=len(hist), train_seconds=el)
            seed_rows[mname].append(row)
            print(f"    test T RMSE {row['temp_rmse']:.4f} | S RMSE {row['sal_rmse']:.4f} | "
                  f"viol {row['violation_teos10_pct']:.2f}% | {el:.0f}s")
            if j == 0:
                preds[mname] = (pT, pS)
                histories[mname] = hist
                timings[mname] = el
                tag = "argocast" if physics else "bilstm_mse"
                torch.save(model.state_dict(), MODEL_DIR / f"{tag}_{name}_seed{seed}.pt")
                if physics:
                    mT, mS = mc_dropout(model, Xte_t, args.mc_passes)
                    mT = np.stack([sc.invT(x) for x in mT]); mS = np.stack([sc.invS(x) for x in mS])
                    np.savez_compressed(PRED_DIR / f"mc_{name}.npz",
                                        meanT=mT.mean(0), stdT=mT.std(0), meanS=mS.mean(0), stdS=mS.std(0),
                                        q05T=np.percentile(mT, 5, 0), q95T=np.percentile(mT, 95, 0),
                                        q025T=np.percentile(mT, 2.5, 0), q975T=np.percentile(mT, 97.5, 0),
                                        q05S=np.percentile(mS, 5, 0), q95S=np.percentile(mS, 95, 0),
                                        q025S=np.percentile(mS, 2.5, 0), q975S=np.percentile(mS, 97.5, 0))
    joblib.dump({"temp_scaler": sc.t, "sal_scaler": sc.s}, MODEL_DIR / f"scalers_{name}.joblib")

    # ---- metrics for every model (first-seed predictions)
    pers_err = ((preds["Persistence"][0] - tT) ** 2).mean(1)
    groups = te["meta"]["wmo"].to_numpy()
    for mname in MODEL_ORDER:
        if mname not in preds:
            continue
        pT, pS = preds[mname]
        row = metric_row(pT, pS, tT, tS, lat, lon)
        e2T, e2S = ((pT - tT) ** 2).mean(1), ((pS - tS) ** 2).mean(1)
        row.update(float_bootstrap(e2T, e2S, groups, pers_err, B=args.bootstrap))
        row["temp_skill_vs_persistence"] = 1 - row["temp_rmse"] / rmse(preds["Persistence"][0], tT)
        row["sal_skill_vs_persistence"] = 1 - row["sal_rmse"] / rmse(preds["Persistence"][1], tS)
        row["per_depth_temp_rmse"] = np.sqrt(((pT - tT) ** 2).mean(0))
        row["per_depth_sal_rmse"] = np.sqrt(((pS - tS) ** 2).mean(0))
        seas = {}
        for sname in SEASONS:
            mk = (te["meta"]["season"] == sname).to_numpy()
            if mk.sum():
                seas[sname] = {"n": int(mk.sum()), "temp_rmse": rmse(pT[mk], tT[mk]), "sal_rmse": rmse(pS[mk], tS[mk])}
        row["by_season"] = seas
        yrs = te["meta"]["target_date"].dt.year.to_numpy()
        row["by_year"] = {int(y): {"n": int((yrs == y).sum()), "temp_rmse": rmse(pT[yrs == y], tT[yrs == y])}
                          for y in np.unique(yrs)}
        if mname in seed_rows:
            df = pd.DataFrame(seed_rows[mname])
            row["seeds"] = seed_rows[mname]
            for k in ("temp_rmse", "sal_rmse", "therm_rmse", "deep_rmse", "violation_teos10_pct", "violation_poly_pct"):
                row[f"{k}_seed_mean"] = float(df[k].mean())
                row[f"{k}_seed_std"] = float(df[k].std(ddof=0))
        row["fit_seconds"] = timings.get(mname)
        results[mname] = row
        np.savez_compressed(PRED_DIR / f"pred_{name}_{mname.replace(' ', '_').replace('(', '').replace(')', '')}.npz",
                            T=pT, S=pS)
    results["Observed (reference)"] = {
        "violation_poly_pct": inversion_rate_poly(tT, tS),
        "violation_teos10_pct": inversion_rate_teos10(tT, tS, lat, lon)}

    # ---- uncertainty calibration
    mc = np.load(PRED_DIR / f"mc_{name}.npz")
    uq = {}
    for v, truth in (("T", tT), ("S", tS)):
        mu, sd = mc[f"mean{v}"], np.maximum(mc[f"std{v}"], 1e-9)
        z = np.abs(truth - mu) / sd
        uq[v] = {
            "gauss_90_coverage": float((z <= 1.645).mean()), "gauss_95_coverage": float((z <= 1.96).mean()),
            "quantile_90_coverage": float(((truth >= mc[f"q05{v}"]) & (truth <= mc[f"q95{v}"])).mean()),
            "quantile_95_coverage": float(((truth >= mc[f"q025{v}"]) & (truth <= mc[f"q975{v}"])).mean()),
            "per_depth_gauss_95": ((z <= 1.96).mean(0)),
            "per_depth_mean_std": sd.mean(0),
            "nominal_vs_empirical": [[float(p), float((z <= _zq(p)).mean())] for p in np.linspace(0.1, 0.99, 12)],
            "mean_interval_width_95": float((2 * 1.96 * sd).mean()),
        }
    write_json(TAB_DIR / f"metrics_{name}.json", {
        "split": name, "n_test_windows": int(len(tT)), "n_test_floats": int(te["meta"]["wmo"].nunique()),
        "n_train_windows": int(len(parts["train"]["X"])), "n_val_windows": int(len(parts["val"]["X"])),
        "models": results, "uq_mc_dropout": uq, "histories": histories,
        "settings": {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
                     "lr": args.lr, "seeds": args.seeds, "mc_passes": args.mc_passes,
                     "bootstrap": args.bootstrap, "gb_iter": args.gb_iter}})
    flat = []
    for mname, r in results.items():
        flat.append({"model": mname, **{k: v for k, v in r.items() if np.isscalar(v) or v is None}})
    pd.DataFrame(flat).to_csv(TAB_DIR / f"metrics_{name}.csv", index=False)
    figures(name, parts, preds, results, uq, histories, mc)
    return results


def _zq(p):
    from scipy.stats import norm
    return norm.ppf(0.5 + p / 2)


# ---------------------------------------------------------------- figures
def figures(name, parts, preds, results, uq, histories, mc):
    plt = setup_matplotlib()
    te = parts["test"]
    tT, tS = te["yT"], te["yS"]
    models = [m for m in MODEL_ORDER if m in results]
    depth = STANDARD_DEPTHS

    # Bar chart with CIs
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, key, lab in zip(axes, ["temp_rmse", "sal_rmse", "violation_teos10_pct"],
                            ["Temperature RMSE (°C)", "Salinity RMSE (PSU)", "Density inversions (% of profiles)"]):
        vals = [results[m][key] for m in models]
        cols = [PALETTE[m] for m in models]
        ax.bar(range(len(models)), vals, color=cols)
        if key != "violation_teos10_pct":
            ci = np.array([results[m][key + "_ci"] for m in models])
            ax.errorbar(range(len(models)), vals, yerr=[np.clip(np.array(vals) - ci[:, 0], 0, None), np.clip(ci[:, 1] - np.array(vals), 0, None)],
                        fmt="none", ecolor="black", capsize=3, lw=0.8)
        else:
            obs = results["Observed (reference)"]["violation_teos10_pct"]
            ax.axhline(obs, color="black", ls="--", lw=0.8, label=f"observed profiles {obs:.1f}%")
            ax.legend(fontsize=7)
        ax.set_xticks(range(len(models)))
        short = {"Gradient Boosting": "Grad. Boosting", "Bi-LSTM (MSE only)": "Bi-LSTM (MSE)",
                 "ArgoCast Bi-LSTM": "ArgoCast"}
        ax.set_xticklabels([short.get(m, m) for m in models], fontsize=7, rotation=35, ha="right")
        ax.set_ylabel(lab)
    fig.suptitle(f"Test-set accuracy and physical consistency - {name.replace('_', ' ')} split "
                 f"({len(tT):,} windows, 95% float-bootstrap CI)", fontsize=9)
    fig.tight_layout()
    savefig(fig, f"fig_model_comparison_{name}")

    # Per-depth RMSE
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
    for m in models:
        st = dict(color=PALETTE[m], lw=2.0 if "ArgoCast" in m else 1.1, ls="-" if m != "Persistence" else "--")
        axes[0].plot(results[m]["per_depth_temp_rmse"], depth, label=m, **st)
        axes[1].plot(results[m]["per_depth_sal_rmse"], depth, **st)
        if m != "Persistence":
            sk = 1 - results[m]["per_depth_temp_rmse"] / results["Persistence"]["per_depth_temp_rmse"]
            axes[2].plot(sk * 100, depth, **st)
    axes[2].axvline(0, color="black", lw=0.8)
    axes[0].set_ylim(1020, 0)
    axes[0].set_ylabel("Pressure (dbar)")
    axes[0].set_xlabel("Temperature RMSE (°C)"); axes[1].set_xlabel("Salinity RMSE (PSU)")
    axes[2].set_xlabel("Temperature skill vs persistence (%)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Error by depth - {name.replace('_', ' ')} split", fontsize=9)
    fig.tight_layout()
    savefig(fig, f"fig_depth_rmse_{name}")

    # Seasonal RMSE
    fig, ax = plt.subplots(figsize=(8, 3.2))
    seas = [s for s in SEASONS if s in results["Persistence"]["by_season"]]
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        v = [results[m]["by_season"].get(s, {}).get("temp_rmse", np.nan) for s in seas]
        ax.bar(np.arange(len(seas)) + i * w - 0.4 + w / 2, v, width=w, color=PALETTE[m], label=m)
    ax.set_xticks(range(len(seas)))
    ax.set_xticklabels([f"{s}\n(n={results['Persistence']['by_season'][s]['n']:,})" for s in seas], fontsize=7)
    ax.set_ylabel("Temperature RMSE (°C)")
    ax.legend(fontsize=6, ncol=3)
    ax.set_title(f"Error by monsoon season - {name.replace('_', ' ')} split")
    savefig(fig, f"fig_seasonal_rmse_{name}")

    # Training curves
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for m, h in histories.items():
            h = pd.DataFrame(h)
            axes[0].plot(h["epoch"], h["val_temp_rmse"], color=PALETTE[m], label=m)
            axes[1].plot(h["epoch"], h["val_sal_rmse"], color=PALETTE[m])
        axes[0].set_xlabel("Epoch"); axes[1].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation T RMSE (°C)"); axes[1].set_ylabel("Validation S RMSE (PSU)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        savefig(fig, f"fig_training_curves_{name}")

    # Uncertainty calibration
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for v, c in (("T", "#0072B2"), ("S", "#D55E00")):
        a = np.array(uq[v]["nominal_vs_empirical"])
        axes[0].plot(a[:, 0] * 100, a[:, 1] * 100, "o-", color=c, ms=3, label="Temperature" if v == "T" else "Salinity")
        axes[1].plot(np.array(uq[v]["per_depth_gauss_95"]) * 100, depth, color=c)
    axes[0].plot([0, 100], [0, 100], "k--", lw=0.8, label="perfect calibration")
    axes[0].set_xlabel("Nominal interval (%)"); axes[0].set_ylabel("Observed coverage (%)")
    axes[0].legend(fontsize=7)
    axes[1].axvline(95, color="black", ls="--", lw=0.8)
    axes[1].set_ylim(1020, 0); axes[1].set_xlabel("Coverage of 95% interval (%)"); axes[1].set_ylabel("Pressure (dbar)")
    fig.suptitle(f"MC-dropout uncertainty calibration - {name.replace('_', ' ')} split", fontsize=9)
    fig.tight_layout()
    savefig(fig, f"fig_uq_calibration_{name}")

    # Scatter at 100 dbar
    cand = [m for m in ["Gradient Boosting", "ArgoCast Bi-LSTM"] if m in preds]
    fig, axes = plt.subplots(1, len(cand), figsize=(3.6 * len(cand), 3.4), squeeze=False)
    for ax, m in zip(axes[0], cand):
        ax.scatter(tT[:, 4], preds[m][0][:, 4], s=2, alpha=0.3, color=PALETTE[m])
        lo, hi = np.percentile(tT[:, 4], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("Observed T at 100 dbar (°C)"); ax.set_ylabel("Forecast (°C)")
        ax.set_title(f"{m}: RMSE {rmse(preds[m][0][:, 4], tT[:, 4]):.3f}")
    fig.tight_layout()
    savefig(fig, f"fig_scatter_100dbar_{name}")

    # Example forecast with MC band (window with median ArgoCast error)
    if "ArgoCast Bi-LSTM" in preds:
        e = ((preds["ArgoCast Bi-LSTM"][0] - tT) ** 2).mean(1)
        i = int(np.argsort(e)[len(e) // 2])
        md = te["meta"].iloc[i]
        fig, axes = plt.subplots(1, 2, figsize=(7, 4.2), sharey=True)
        for ax, v, k, unit in ((axes[0], "T", 0, "°C"), (axes[1], "S", 1, "PSU")):
            truth = tT[i] if v == "T" else tS[i]
            ax.fill_betweenx(depth, mc[f"q025{v}"][i], mc[f"q975{v}"][i], color="#0072B2", alpha=0.2, label="95% MC interval")
            ax.plot(truth, depth, "k-o", ms=3, label="Observed")
            ax.plot(preds["ArgoCast Bi-LSTM"][k][i], depth, color="#0072B2", lw=2, label="ArgoCast")
            ax.plot(preds["Persistence"][k][i], depth, color="#999999", ls="--", label="Persistence")
            ax.set_xlabel(f"{'Temperature' if v=='T' else 'Salinity'} ({unit})")
        axes[0].set_ylim(1020, 0); axes[0].set_ylabel("Pressure (dbar)"); axes[0].legend(fontsize=7)
        fig.suptitle(f"Example forecast - float {md['wmo']} cycle {md['cycle']}, "
                     f"{md['target_date'].date()} ({md['lat']:.2f}N, {md['lon']:.2f}E)", fontsize=9)
        fig.tight_layout()
        savefig(fig, f"fig_example_forecast_{name}")


def data_figures():
    plt = setup_matplotlib()
    w = np.load(PROCESSED_DIR / "windows.npz", allow_pickle=False)
    years = pd.to_datetime(w["target_date"], unit="s").year
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), gridspec_kw={"height_ratios": [1.3, 1]})
    cols = {"train": "#0072B2", "val": "#E69F00", "test": "#D55E00"}
    for j, name in enumerate(SPLITS):
        s = np.load(PROCESSED_DIR / f"split_{name}.npz")
        ax = axes[0, j]
        for k in ("train", "val", "test"):
            ix = s[k]
            ax.scatter(w["lon"][ix], w["lat"][ix], s=1, alpha=0.4, color=cols[k], label=f"{k} ({len(ix):,})")
        ax.set_xlim(49.5, 75.5); ax.set_ylim(9.5, 25.5)
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"{name.replace('_', ' ').title()} split - target positions")
        ax.legend(markerscale=8, fontsize=7, loc="lower left")
        ax = axes[1, j]
        yr = np.arange(years.min(), years.max() + 1)
        bottom = np.zeros(len(yr))
        for k in ("train", "val", "test"):
            c = np.array([(years[s[k]] == y).sum() for y in yr])
            ax.bar(yr, c, bottom=bottom, color=cols[k], label=k)
            bottom += c
        ax.set_xlabel("Year of target profile"); ax.set_ylabel("Forecast windows")
        ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig_data_and_splits")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mc-passes", type=int, default=50)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--gb-iter", type=int, default=200)
    ap.add_argument("--skip-gb", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    t0 = time.time()
    data_figures()
    for name in args.splits:
        run_split(name, args, device)
    write_json(TAB_DIR / "run_info.json", {"environment": environment_info(), "total_seconds": time.time() - t0,
                                           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)})
    print(f"\nAll done in {(time.time()-t0)/60:.1f} min. Tables: {TAB_DIR}  Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
